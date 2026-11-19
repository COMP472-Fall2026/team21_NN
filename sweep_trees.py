"""Hyperparameter sweep for the decision tree and random forest, on both tasks.

Each setting is changed one at a time from a base configuration
(gini, min_samples_split=2, min_samples_leaf=1; forest: sqrt features).

Two shortcuts keep the number of fits small. Both give exactly the same
predictions as separate fits (checked on both tasks):
  - One unlimited-depth tree is cut off at depths 5, 10, 20 and None.
    Trees are grown top-down, so the top of a deep tree is the shallow tree.
  - One 100-tree forest gives the 10-, 50- and 100-tree forests.
    Trees are drawn in order from one seeded generator.
Rows that come from a cut-off or a subset have an empty fit_s; the real
fit time of the shared model is in results/sweep/fit_times.csv.

Usage:
    python sweep_trees.py --all            # run every job in parallel, then merge
    python sweep_trees.py --all --workers 4
    python sweep_trees.py --job image_dt_base
    python sweep_trees.py --list
    python sweep_trees.py --summary        # write results/sweep/summary.md
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import common
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

SWEEP_DIR = os.path.join(common.RESULTS_DIR, "sweep")
DEPTHS = [5, 10, 20, None]

DT_BASE = {"criterion": "gini", "min_samples_split": 2, "min_samples_leaf": 1}
DT_VARIANTS = {
    "dt_base": {},
    "dt_entropy": {"criterion": "entropy"},
    "dt_split20": {"min_samples_split": 20},
    "dt_leaf10": {"min_samples_leaf": 10},
}
# (constructor kwargs, tree counts to evaluate)
RF_VARIANTS = {
    "rf_trees": ({"n_estimators": 100, "max_depth": None, "max_features": "sqrt"}, [10, 50, 100]),
    "rf_depth10": ({"n_estimators": 50, "max_depth": 10, "max_features": "sqrt"}, [50]),
    "rf_depth20": ({"n_estimators": 50, "max_depth": 20, "max_features": "sqrt"}, [50]),
    "rf_log2": ({"n_estimators": 50, "max_depth": None, "max_features": "log2"}, [50]),
}

JOBS = [f"{task}_{name}" for task in common.TASKS for name in [*DT_VARIANTS, *RF_VARIANTS]]


class _SharedFit:
    """Fits the wrapped model on the first fit() call only, and predicts with fixed kwargs."""

    def __init__(self, model, predict_kwargs, state):
        self.model, self.predict_kwargs, self.state = model, predict_kwargs, state

    def fit(self, X, y):
        if not self.state.get("fitted"):
            t0 = time.perf_counter()
            self.model.fit(X, y)
            self.state["fit_s"] = time.perf_counter() - t0
            self.state["fitted"] = True
        return self

    def predict(self, X):
        return self.model.predict(X, **self.predict_kwargs)


def run_job(job):
    task, name = job.split("_", 1)
    os.makedirs(SWEEP_DIR, exist_ok=True)
    job_csv = os.path.join(SWEEP_DIR, f"{job}.csv")
    if os.path.exists(job_csv):
        os.remove(job_csv)
    common.RESULTS_CSV = job_csv  # one file per job, so parallel jobs never write the same file
    common.set_seed()
    state = {}

    if name in DT_VARIANTS:
        kwargs = {**DT_BASE, **DT_VARIANTS[name]}
        model = DecisionTreeClassifier(max_depth=None, **kwargs)
        # None first, so the real fit happens (and is timed) in that call.
        points = [None] + [d for d in DEPTHS if d is not None]
        for d in points:
            params = {**kwargs, "max_depth": d}
            common.evaluate(_SharedFit(model, {"max_depth": d}, state), task,
                            name="decision_tree", params=params)
        full = {**kwargs, "max_depth": None}
        model_name = "decision_tree"
    else:
        kwargs, counts = RF_VARIANTS[name]
        model = RandomForestClassifier(criterion="gini", **kwargs)
        for k in sorted(counts, reverse=True):
            params = {"criterion": "gini", "max_depth": kwargs["max_depth"],
                      "max_features": kwargs["max_features"], "n_estimators": k}
            common.evaluate(_SharedFit(model, {"n_trees": k}, state), task,
                            name="random_forest", params=params)
        full = {"criterion": "gini", **kwargs}
        model_name = "random_forest"

    # The first row is the real fit. Later rows reuse it, so their fit time is not meaningful.
    rows = pd.read_csv(job_csv)
    rows["fit_s"] = rows["fit_s"].astype(float)
    rows.loc[rows.index[1:], "fit_s"] = np.nan
    rows.to_csv(job_csv, index=False)
    pd.DataFrame([{"job": job, "task": task, "model": model_name,
                   "params": json.dumps(full, sort_keys=True),
                   "fit_s": round(state["fit_s"], 2)}]).to_csv(
        os.path.join(SWEEP_DIR, f"{job}.fit_time.csv"), index=False)


def run_all(workers):
    log_dir = os.path.join(SWEEP_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # One thread per process, so 16 processes don't oversubscribe the cores.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    pending, running, failed = list(JOBS), {}, []
    t0 = time.perf_counter()
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            log = open(os.path.join(log_dir, f"{job}.log"), "w")
            running[job] = (subprocess.Popen([sys.executable, os.path.abspath(__file__), "--job", job],
                                             stdout=log, stderr=subprocess.STDOUT, env=env,
                                             cwd=common.BASE_DIR), log)
        for job, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[job]
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode != 0:
                    failed.append(job)
                print(f"[{time.perf_counter() - t0:6.0f}s] {job}: {status}", flush=True)
        time.sleep(1)
    if failed:
        sys.exit(f"Failed jobs: {failed}. See {log_dir}.")
    merge()


def merge():
    """Append every job's rows to the shared results/results.csv and collect fit times."""
    rows = pd.concat([pd.read_csv(os.path.join(SWEEP_DIR, f"{j}.csv")) for j in JOBS], ignore_index=True)
    shared_csv = os.path.join(common.RESULTS_DIR, "results.csv")
    rows.to_csv(shared_csv, mode="a", header=not os.path.exists(shared_csv), index=False)
    times = pd.concat([pd.read_csv(os.path.join(SWEEP_DIR, f"{j}.fit_time.csv")) for j in JOBS],
                      ignore_index=True)
    times.to_csv(os.path.join(SWEEP_DIR, "fit_times.csv"), index=False)
    print(f"Appended {len(rows)} rows to {shared_csv}")


def _md(df):
    """DataFrame as a markdown table, without the tabulate dependency."""
    cells = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


def _files(task, model, params):
    tag = common._tag(params, True)
    stem = f"{task}_{model}_{tag}"
    return (os.path.join(common.RESULTS_DIR, "preds", f"{stem}.csv"),
            os.path.join(common.RESULTS_DIR, "cm", f"{stem}.csv"),
            os.path.join(common.RESULTS_DIR, "class_metrics", f"{stem}.csv"))


def summary():
    rows = pd.concat([pd.read_csv(os.path.join(SWEEP_DIR, f"{j}.csv")) for j in JOBS], ignore_index=True)
    rows["gap"] = (rows["train_acc"] - rows["val_acc"]).round(4)
    out = ["# Tree sweep results", "",
           "Validation split: shared 80/20 stratified split (seed 42), ids in `splits/`.",
           "train_acc is on a fixed 4,000-row sample of the training split. gap = train_acc - val_acc.",
           "Empty fit_s: the row reuses another row's fit (depth cut-off or tree subset).", ""]
    for task in common.TASKS:
        t = rows[rows.task == task]
        for model in ("decision_tree", "random_forest"):
            m = t[t.model == model][["params", "train_acc", "val_acc", "gap", "macro_f1", "fit_s", "predict_s"]]
            out += [f"## {task}: {model}", "", _md(m), ""]
        out += [f"## {task}: error analysis (best tree and best forest by val_acc)", ""]
        preds = {}
        for model in ("decision_tree", "random_forest"):
            best = t[t.model == model].sort_values("val_acc", ascending=False).iloc[0]
            params = json.loads(best["params"])
            pred_f, cm_f, cls_f = _files(task, model, params)
            preds[model] = pd.read_csv(pred_f)
            out.append(f"**{model}** {best['params']}: val_acc {best['val_acc']}")
            if task == "sentiment":
                out += ["", _md(pd.read_csv(cls_f).round(3)), ""]
            else:
                cm = pd.read_csv(cm_f, index_col=0).values
                pairs = common.top_confusions(cm, k=5)
                recall = pd.read_csv(cls_f).sort_values("recall")
                out += ["", "Top confused (true -> predicted, count): " +
                        ", ".join(f"{a}->{b} ({c})" for a, b, c in pairs),
                        "Lowest-recall classes: " + ", ".join(
                            f"{int(r.label)} ({r.recall:.2f})" for r in recall.head(5).itertuples()),
                        "Highest-recall classes: " + ", ".join(
                            f"{int(r.label)} ({r.recall:.2f})" for r in recall.tail(5)[::-1].itertuples()), ""]
        a, b = preds["decision_tree"], preds["random_forest"]
        both = (~a.correct.values) & (~b.correct.values)
        out += [f"Validation rows both models get wrong: {both.sum()} of {len(a)}. "
                f"Tree wrong, forest right: {((~a.correct.values) & b.correct.values).sum()}. "
                f"Forest wrong, tree right: {(a.correct.values & ~b.correct.values).sum()}.", ""]
    times = pd.read_csv(os.path.join(SWEEP_DIR, "fit_times.csv"))
    out += ["## Fit time of each shared model", "", _md(times), ""]
    path = os.path.join(SWEEP_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every job in parallel, then merge")
    ap.add_argument("--workers", type=int, default=min(len(JOBS), os.cpu_count() or 1))
    ap.add_argument("--job", choices=JOBS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(JOBS))
    elif args.job:
        run_job(args.job)
    elif args.all:
        run_all(args.workers)
    elif args.summary:
        summary()
    else:
        ap.print_help()
