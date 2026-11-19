"""
k-NN experiments for both tasks, using the shared setup in common.py.

    python knn_experiments.py --sweep     full grid, fast (writes results/knn/)
    python knn_experiments.py --log       logs the important settings through
                                          common.evaluate() -> results/results.csv,
                                          confusion matrices, per-class metrics, preds
    python knn_experiments.py --all       both of the above
    python knn_experiments.py --submit    backup Kaggle file submission_knn.csv
                                          (k-NN only, refit on all training rows)

Grid, per task:
    features : raw / z-score normalized (common.get_split normalize=False/True)
    metric   : euclidean, cosine, manhattan
    k        : 1, 3, 5, 10, 20, 50, 100, 200
    voting   : majority / 1-over-distance weighted

Shortcut used by --sweep: neighbours are searched once per (features, metric)
at k = 200; every smaller k and both voting schemes are scored from that one
search. This gives exactly the same predictions as separate KNN(k=...) fits,
because the k nearest neighbours are just the first k of the 50 nearest.

Outputs in results/knn/:
    knn_sweep_<task>.csv        every setting + validation accuracy
    knn_k_curve_<task>.png      validation accuracy vs k
    knn_train_vs_val_<task>.png training vs validation accuracy vs k
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR, evaluate, get_split, set_seed
from knn import KNN

TASKS = ["image", "sentiment"]
K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200]
METRICS = ["euclidean", "cosine", "manhattan"]
OUT_DIR = os.path.join(RESULTS_DIR, "knn")
N_TRAIN_EVAL = 4000  # same training-accuracy sample size as common.evaluate()


def _vote_name(weighted):
    return "weighted" if weighted else "majority"


# ------------------------------------------------------------------ sweep ---
def sweep_task(task):
    rows = []
    k_max = max(K_VALUES)
    rng = np.random.default_rng(42)
    for normalize in (False, True):
        X_tr, X_val, y_tr, y_val = get_split(task, normalize=normalize)
        tr_idx = rng.choice(len(X_tr), size=min(N_TRAIN_EVAL, len(X_tr)), replace=False)
        for metric in METRICS:
            model = KNN(k=k_max, metric=metric).fit(X_tr, y_tr)
            t0 = time.perf_counter()
            d_val, i_val = model.kneighbors(X_val, k_max)
            search_s = time.perf_counter() - t0
            # training accuracy on a 4,000-row sample. Each of these rows is
            # also in the stored training set, so it is its own nearest
            # neighbour (distance 0): that is why k=1 scores 100% on training.
            d_tr, i_tr = model.kneighbors(X_tr[tr_idx], k_max)
            for k in K_VALUES:
                for weighted in (False, True):
                    rows.append({
                        "task": task, "normalize": normalize, "metric": metric,
                        "k": k, "voting": _vote_name(weighted),
                        "train_acc": float(np.mean(model.vote(d_tr, i_tr, k, weighted) == y_tr[tr_idx])),
                        "val_acc": float(np.mean(model.vote(d_val, i_val, k, weighted) == y_val)),
                        "val_search_s": round(search_s, 2),
                    })
            best = max(r["val_acc"] for r in rows
                       if r["metric"] == metric and r["normalize"] == normalize)
            print(f"[{task}] {'norm' if normalize else 'raw '} {metric:9s} "
                  f"search {search_s:6.1f}s  best val {best:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"knn_sweep_{task}.csv"), index=False)
    plot_k_curve(df, task)
    plot_train_vs_val(df, task)
    b = best_setting(df)
    print(f"[{task}] BEST: {'norm' if b['normalize'] else 'raw'} {b['metric']} "
          f"k={b['k']} {b['voting']} -> val {b['val_acc']:.4f}")
    return df


def best_setting(df, fast_only=True):
    """Highest validation accuracy; ties go to the smaller k.
    fast_only: choose among euclidean/cosine only. Manhattan has no matrix-
    multiply shortcut, so it is ~50x slower, and on these data it is either
    clearly worse (image) or within validation noise of the others (sentiment)."""
    if fast_only:
        df = df[df.metric != "manhattan"]
    return df.sort_values(["val_acc", "k"], ascending=[False, True]).iloc[0].to_dict()


# ------------------------------------------------------------------ plots ---
def _k_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)
    ax.minorticks_off()
    ax.set_xlabel("k (number of neighbours)")
    ax.grid(alpha=0.3)


def plot_k_curve(df, task):
    colors = {"euclidean": "#1f77b4", "cosine": "#d62728", "manhattan": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, normalize in zip(axes, (False, True)):
        sub = df[df.normalize == normalize]
        for (metric, voting), g in sub.groupby(["metric", "voting"]):
            g = g.sort_values("k")
            ax.plot(g.k, g.val_acc, marker="o", color=colors[metric],
                    linestyle="-" if voting == "weighted" else "--",
                    label=f"{metric}, {voting}")
        _k_axis(ax)
        ax.set_title(f"{task}: {'z-score normalized' if normalize else 'raw'} features")
    axes[0].set_ylabel("validation accuracy")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"knn_k_curve_{task}.png"), dpi=150)
    plt.close(fig)


def plot_train_vs_val(df, task):
    """Training vs validation accuracy vs k for the best features/metric,
    majority voting (weighted voting always scores ~100% on training rows,
    because each row is its own neighbour at distance 0)."""
    b = best_setting(df)
    g = df[(df.normalize == b["normalize"]) & (df.metric == b["metric"])
           & (df.voting == "majority")].sort_values("k")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(g.k, g.train_acc, marker="o", label="training (4,000-row sample)")
    ax.plot(g.k, g.val_acc, marker="o", label="validation")
    _k_axis(ax)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{task}: {b['metric']}, "
                 f"{'normalized' if b['normalize'] else 'raw'}, majority vote", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"knn_train_vs_val_{task}.png"), dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- log ---
def log_task(task, df):
    """Send the important settings through common.evaluate() so they appear in
    the shared results.csv with confusion matrices and per-class metrics:
      - the best setting for each (features, metric) except manhattan
        (manhattan is ~50x slower and never clearly wins; its numbers are in the sweep csv)
      - every k for the overall best features/metric, with MAJORITY voting
        (the k curve). Majority voting is used here because with weighted
        voting training accuracy is always ~1.0 (each training row votes for
        itself at distance 0), which hides the over/underfitting trend.
    """
    configs = set()
    for (normalize, metric), g in df[df.metric != "manhattan"].groupby(["normalize", "metric"]):
        b = best_setting(g)
        configs.add((bool(normalize), metric, int(b["k"]), b["voting"] == "weighted"))
    b = best_setting(df)
    for k in K_VALUES:
        configs.add((bool(b["normalize"]), b["metric"], k, False))

    for normalize, metric, k, weighted in sorted(configs):
        params = {"k": k, "metric": metric, "voting": _vote_name(weighted)}
        evaluate(KNN(k=k, metric=metric, weighted=weighted), task, name="knn",
                 params=params, normalize=normalize, verbose=False)
        print(f"[{task}] logged {json.dumps(params)} normalize={normalize}", flush=True)


# Settings chosen on the validation split (see reports/knn_summary.md).
FINAL = {
    "image": {"k": 1, "metric": "cosine", "weighted": False, "normalize": True},
    "sentiment": {"k": 100, "metric": "cosine", "weighted": True, "normalize": True},
}


def submit():
    """Backup Kaggle file from k-NN alone: refit on all training rows, predict test."""
    from common import BASE_DIR, fit_full_and_predict, make_submission
    out = {}
    for task, s in FINAL.items():
        model = KNN(k=s["k"], metric=s["metric"], weighted=s["weighted"])
        out[task] = fit_full_and_predict(model, task, normalize=s["normalize"])
    make_submission(*out["image"], *out["sentiment"],
                    path=os.path.join(BASE_DIR, "submission_knn.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--submit", action="store_true",
                    help="write submission_knn.csv with the chosen k-NN settings")
    ap.add_argument("--task", choices=TASKS, help="run one task only")
    args = ap.parse_args()
    set_seed()
    if args.submit:
        submit()
        return
    if not (args.sweep or args.log or args.all):
        ap.print_help()
        return
    for task in ([args.task] if args.task else TASKS):
        path = os.path.join(OUT_DIR, f"knn_sweep_{task}.csv")
        if args.sweep or args.all:
            df = sweep_task(task)
        else:
            df = pd.read_csv(path)
        if args.log or args.all:
            log_task(task, df)


if __name__ == "__main__":
    main()
