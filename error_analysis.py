"""
Error analysis across all four classifiers.

Reads the validation predictions saved by common.evaluate() (results/**/preds/*.csv), keeps the
best run of each model per task, and compares where the models fail.

Usage:
    python error_analysis.py

Needs the prediction files first, e.g. from:
    python knn_experiments.py --all, python sweep_trees.py --all, python neural_net.py
"""

import glob
import os

import numpy as np
import pandas as pd

from common import BASE_DIR, RESULTS_DIR, SENTIMENT_NAMES

MODELS = ["decision_tree", "random_forest", "knn", "neural_net"]
OUT_DIR = os.path.join(RESULTS_DIR, "error_analysis")
FIG_DIR = os.path.join(BASE_DIR, "reports", "error_analysis_figures")


def best_runs(task):
    """For each model, the prediction file with the highest validation accuracy."""
    best = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "**", "preds", f"{task}_*.csv"), recursive=True):
        name = os.path.basename(path)[len(task) + 1:]
        model = next((m for m in MODELS if name.startswith(m + "_")), None)
        if model is None:
            continue
        df = pd.read_csv(path)
        acc = df["correct"].mean()
        if model not in best or acc > best[model][0]:
            best[model] = (acc, name[len(model) + 1:-4], df)
    missing = [m for m in MODELS if m not in best]
    if missing:
        raise SystemExit(f"No {task} predictions found for {missing}. Run their experiments first.")
    ids = best[MODELS[0]][2]["id"].values
    for m in MODELS:
        assert (best[m][2]["id"].values == ids).all(), f"{m} uses a different validation split"
    return best


def confusion(y, p, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm


def per_class(y, p, n):
    cm = confusion(y, p, n)
    recall = cm.diagonal() / cm.sum(1)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    return recall, precision, cm


def top_confusions(cm, k):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(flat, off.shape))]


def same_mistakes(task, best):
    """Overlap of the models' errors, row by row."""
    wrong = pd.DataFrame({m: ~best[m][2]["correct"].values for m in MODELS})
    n_wrong = wrong.sum(1)
    summary = {
        "rows": len(wrong),
        "all four right": int((n_wrong == 0).sum()),
        "all four wrong": int((n_wrong == 4).sum()),
        "at least one right (oracle accuracy)": round(float((n_wrong < 4).mean()), 3),
    }
    # Jaccard overlap of error sets: of the rows either model gets wrong, share both get wrong
    jac = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for a in MODELS:
        for b in MODELS:
            both = (wrong[a] & wrong[b]).sum()
            either = (wrong[a] | wrong[b]).sum()
            jac.loc[a, b] = round(both / either, 3)
    # does the neural net fix the simpler models' mistakes?
    nn_fix = {}
    for m in MODELS[:-1]:
        rows = wrong[m]
        nn_fix[m] = round(float((~wrong["neural_net"][rows]).mean()), 3)
    # same mistake = both wrong AND same wrong label
    preds = {m: best[m][2]["pred"].values for m in MODELS}
    same_label = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for a in MODELS:
        for b in MODELS:
            both = (wrong[a] & wrong[b]).values
            same_label.loc[a, b] = round(float((preds[a][both] == preds[b][both]).mean()), 3)
    return summary, jac, nn_fix, same_label


def plot_sentiment_cms(best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m in zip(axes, MODELS):
        df = best[m][2]
        cm = confusion(df["true"].values, df["pred"].values, 3)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=800)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > 400 else "black")
        ax.set_xticks(range(3), SENTIMENT_NAMES)
        ax.set_yticks(range(3), SENTIMENT_NAMES)
        ax.set_xlabel("predicted")
        ax.set_title(f"{m} (acc {best[m][0]:.3f})")
    axes[0].set_ylabel("true")
    fig.suptitle("Sentiment confusion matrices, best run of each model")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sentiment_confusion_all_models.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_image_recall(recalls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = recalls.mean(1).sort_values().index
    fig, ax = plt.subplots(figsize=(16, 3.8))
    im = ax.imshow(recalls.loc[order].T.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_xticks(range(len(order)), order, fontsize=7)
    ax.set_xlabel("image class (sorted from hardest to easiest by mean recall)")
    fig.colorbar(im, ax=ax, label="recall")
    ax.set_title("Per-class recall on the image validation set, best run of each model")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "image_recall_by_class.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def show(title, obj):
    print(f"\n{title}")
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for task, n in [("sentiment", 3), ("image", 50)]:
        best = best_runs(task)
        print(f"\n===== {task} =====")
        for m in MODELS:
            print(f"  {m:14s} acc {best[m][0]:.4f}  ({best[m][1]})")

        recalls, precisions, cms = {}, {}, {}
        for m in MODELS:
            df = best[m][2]
            recalls[m], precisions[m], cms[m] = per_class(df["true"].values, df["pred"].values, n)
        recalls = pd.DataFrame(recalls).round(3)
        precisions = pd.DataFrame(precisions).round(3)
        if task == "sentiment":
            recalls.index = precisions.index = SENTIMENT_NAMES
        recalls.to_csv(os.path.join(OUT_DIR, f"{task}_recall_by_class.csv"))
        precisions.to_csv(os.path.join(OUT_DIR, f"{task}_precision_by_class.csv"))

        summary, jac, nn_fix, same_label = same_mistakes(task, best)
        jac.to_csv(os.path.join(OUT_DIR, f"{task}_error_overlap_jaccard.csv"))
        same_label.to_csv(os.path.join(OUT_DIR, f"{task}_same_wrong_label.csv"))

        if task == "sentiment":
            show("Recall by class:", recalls)
            show("Precision by class:", precisions)
            for m in MODELS:
                cm = cms[m]
                print(f"  {m:14s} neutral -> negative {cm[1, 0]:4d}, neutral -> positive {cm[1, 2]:4d}, "
                      f"negative <-> positive {cm[0, 2] + cm[2, 0]:4d}, predicted counts {cm.sum(0)}")
            plot_sentiment_cms(best)
        else:
            mean_recall = recalls.mean(1).sort_values()
            show("Hardest classes (mean recall over the four models):",
                 recalls.loc[mean_recall.index[:10]].assign(mean=mean_recall[:10].round(3)))
            show("Easiest classes:",
                 recalls.loc[mean_recall.index[-5:]].assign(mean=mean_recall[-5:].round(3)))
            hard = mean_recall.index[:10]
            hard_acc = {}
            for m in MODELS:
                df = best[m][2]
                rows = df["true"].isin(hard)
                hard_acc[m] = round(float(df.loc[rows, "correct"].mean()), 3)
            show("Accuracy on the 10 hardest classes:", pd.Series(hard_acc))

            # confused pairs (either direction) and how many models share them
            pair_rows = []
            for m in MODELS:
                sym = cms[m] + cms[m].T
                np.fill_diagonal(sym, 0)
                a, b = np.triu_indices(n, 1)
                for idx in np.argsort(-sym[a, b])[:15]:
                    pair_rows.append((m, int(a[idx]), int(b[idx]), int(sym[a[idx], b[idx]])))
            pairs = pd.DataFrame(pair_rows, columns=["model", "class_a", "class_b", "count"])
            shared = (pairs.groupby(["class_a", "class_b"])
                      .agg(models=("model", "nunique"), total=("count", "sum"))
                      .sort_values(["models", "total"], ascending=False))
            shared.to_csv(os.path.join(OUT_DIR, "image_confused_pairs_shared.csv"))
            pairs.to_csv(os.path.join(OUT_DIR, "image_confused_pairs_by_model.csv"), index=False)
            show("Most-confused class pairs (in the top 15 of how many models):", shared.head(12))
            for m in MODELS:
                print(f"  {m:14s} top confusions (true, pred, count): {top_confusions(cms[m], 5)}")
            plot_image_recall(recalls)

        show("Do the models make the same mistakes?", pd.Series(summary))
        show("Error overlap (Jaccard: rows both get wrong / rows either gets wrong):", jac)
        show("When both are wrong, share of rows where they predict the same wrong label:", same_label)
        show("Of the rows each simpler model gets wrong, share the neural net gets right:",
             pd.Series(nn_fix))
