"""
COMP472 shared testing setup.

Everyone uses this so all models are trained/validated on the SAME data:
  - same stratified 80/20 train/validation split (seed 42), saved to splits/
  - same normalization (z-score, fit on the train split only)
  - same evaluation + one shared results/results.csv table
  - one function to build the Kaggle submission

Your model just needs two methods:
    model.fit(X, y)      # X: (n, d) numpy array, y: (n,) int array
    model.predict(X)     # returns (n,) int array

Quick start:
    from common import evaluate, fit_full_and_predict, make_submission
    evaluate(MyKNN(k=5), "image", name="knn", params={"k": 5})

Check the setup works:  python common.py --check
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- config ---
SEED = 42
VAL_SIZE = 0.20
# All folders live next to this file, so it works no matter where you run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("COMP472_DATA", os.path.join(BASE_DIR, "data"))  # Kaggle CSVs
SPLIT_DIR = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

TASKS = {
    "image": {"train": "image_train.csv", "test": "image_test.csv", "n_classes": 50},
    "sentiment": {"train": "sentiment_train.csv", "test": "sentiment_test.csv", "n_classes": 3},
}
SENTIMENT_NAMES = ["negative", "neutral", "positive"]


def set_seed(seed=SEED):
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ data ---
_cache = {}


def _check_task(task):
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {task!r}")


def load_task(task):
    """Return (train_df, test_df) for 'image' or 'sentiment'. Cached after first load."""
    _check_task(task)
    if task not in _cache:
        missing = [f for f in (TASKS[task]["train"], TASKS[task]["test"])
                   if not os.path.exists(os.path.join(DATA_DIR, f))]
        if missing:
            raise FileNotFoundError(
                f"Can't find {missing} in {DATA_DIR}\n"
                f"Put the Kaggle CSVs in a 'data' folder next to common.py.")
        tr = pd.read_csv(os.path.join(DATA_DIR, TASKS[task]["train"]))
        te = pd.read_csv(os.path.join(DATA_DIR, TASKS[task]["test"]))
        _cache[task] = (tr, te)
    return _cache[task]


def _feature_cols(df):
    return [c for c in df.columns if c.startswith("feature_")]


class Standardizer:
    """z-score normalization. Fit on TRAIN only, then apply to val/test."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def _val_ids(task, train_df):
    """Validation ids are saved to splits/ the first time, so the split never
    changes between people or library versions. Commit the splits/ folder."""
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = os.path.join(SPLIT_DIR, f"{task}_val_ids.csv")
    if os.path.exists(path):
        return set(pd.read_csv(path)["id"])
    _, val_ids = train_test_split(
        train_df["id"].values,
        test_size=VAL_SIZE,
        stratify=train_df["label"].values,
        random_state=SEED,
    )
    pd.DataFrame({"id": np.sort(val_ids)}).to_csv(path, index=False)
    return set(val_ids)


def get_split(task, normalize=True):
    """The shared 80/20 split.

    Returns X_train, X_val, y_train, y_val (float32 / int64 numpy arrays).
    With normalize=True, features are z-scored using train-split statistics.
    """
    train_df, _ = load_task(task)
    val_ids = _val_ids(task, train_df)
    is_val = train_df["id"].isin(val_ids).values
    cols = _feature_cols(train_df)
    X = train_df[cols].values.astype(np.float32)
    y = train_df["label"].values.astype(np.int64)
    X_tr, X_val, y_tr, y_val = X[~is_val], X[is_val], y[~is_val], y[is_val]
    if normalize:
        sc = Standardizer().fit(X_tr)
        X_tr, X_val = sc.transform(X_tr), sc.transform(X_val)
    return X_tr, X_val, y_tr, y_val


def get_full(task, normalize=True):
    """ALL training data + test features (for the final Kaggle model).

    Returns X_train_full, y_train_full, X_test, test_ids.
    """
    train_df, test_df = load_task(task)
    cols = _feature_cols(train_df)
    X = train_df[cols].values.astype(np.float32)
    y = train_df["label"].values.astype(np.int64)
    X_te = test_df[cols].values.astype(np.float32)
    if normalize:
        sc = Standardizer().fit(X)
        X, X_te = sc.transform(X), sc.transform(X_te)
    return X, y, X_te, test_df["id"].values


# ------------------------------------------------------------ evaluation ---
def evaluate(model, task, name, params=None, normalize=True, save=True, verbose=True):
    """Fit on the train split, score on the validation split, log the result.

    name   : short model name, e.g. "knn", "decision_tree", "random_forest", "mlp"
    params : dict of hyperparameters, e.g. {"k": 5, "distance": "cosine"}

    Saves (when save=True):
      results/results.csv                      one row per run (shared table)
      results/preds/<task>_<name>_<tag>.npy    validation predictions (for error analysis)
      results/cm/<task>_<name>_<tag>.png       confusion matrix plot
    Returns a dict with the metrics.
    """
    _check_task(task)
    params = params or {}
    X_tr, X_val, y_tr, y_val = get_split(task, normalize=normalize)

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    val_pred = np.asarray(model.predict(X_val)).astype(np.int64)
    predict_s = time.perf_counter() - t0

    # Training accuracy shows over/underfitting. On a 4,000-row sample to keep slow models (k-NN) fast.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr), size=min(4000, len(X_tr)), replace=False)
    train_acc = accuracy_score(y_tr[idx], np.asarray(model.predict(X_tr[idx])))

    val_acc = accuracy_score(y_val, val_pred)
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    n_classes = TASKS[task]["n_classes"]
    cm = confusion_matrix(y_val, val_pred, labels=list(range(n_classes)))

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model": name,
        "params": json.dumps(params, sort_keys=True),
        "normalize": normalize,
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "fit_s": round(fit_s, 2),
        "predict_s": round(predict_s, 2),
    }

    if save:
        tag = _tag(params, normalize)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False
        )
        os.makedirs(os.path.join(RESULTS_DIR, "preds"), exist_ok=True)
        np.save(os.path.join(RESULTS_DIR, "preds", f"{task}_{name}_{tag}.npy"), val_pred)
        _plot_cm(cm, task, f"{name} ({tag})",
                 os.path.join(RESULTS_DIR, "cm", f"{task}_{name}_{tag}.png"))

    if verbose:
        print(f"[{task}] {name} {params} -> train {train_acc:.4f} | val {val_acc:.4f} "
              f"| macro-F1 {macro_f1:.4f} | fit {fit_s:.1f}s, predict {predict_s:.1f}s")
        if task == "sentiment":
            print(classification_report(y_val, val_pred, target_names=SENTIMENT_NAMES, digits=3, zero_division=0))

    return {**row, "confusion_matrix": cm, "val_pred": val_pred}


def _tag(params, normalize):
    """Short filename-safe tag from the hyperparameters, e.g. k5_distancecosine_norm."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    parts.append("norm" if normalize else "raw")
    tag = "_".join(parts)
    return "".join(ch for ch in tag if ch.isalnum() or ch in "_-.")[:80]


def _plot_cm(cm, task, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(4.5, 4) if n <= 3 else (10, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{task}: {title}", fontsize=9)
    if n <= 3:
        ax.set_xticks(range(n), SENTIMENT_NAMES)
        ax.set_yticks(range(n), SENTIMENT_NAMES)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def top_confusions(cm, k=10):
    """Most frequently confused (true, predicted, count) pairs, off-diagonal."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(flat, off.shape)) if off[i, j] > 0]


# ------------------------------------------------------------ submission ---
def fit_full_and_predict(model, task, normalize=True):
    """Retrain on ALL training data and predict the test set.
    Returns (test_ids, predictions)."""
    X, y, X_te, ids = get_full(task, normalize=normalize)
    model.fit(X, y)
    return ids, np.asarray(model.predict(X_te)).astype(np.int64)


def make_submission(image_ids, image_preds, sentiment_ids, sentiment_preds,
                    path=os.path.join(BASE_DIR, "submission.csv")):
    """Build submission.csv in sample_submission.csv's order, with the official checks."""
    preds = pd.concat([
        pd.DataFrame({"id": image_ids, "label": image_preds}),
        pd.DataFrame({"id": sentiment_ids, "label": sentiment_preds}),
    ], ignore_index=True)
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sub = sample[["id"]].merge(preds, on="id", how="left", validate="one_to_one")

    assert len(sub) == 10000, f"expected 10000 rows, got {len(sub)}"
    assert sub["label"].notna().all(), "missing predictions"
    sub["label"] = sub["label"].astype(int)
    img = sub["id"].str.startswith("img_")
    txt = sub["id"].str.startswith("text_")
    assert sub.loc[img, "label"].between(0, 49).all(), "image labels out of range"
    assert sub.loc[txt, "label"].between(0, 2).all(), "sentiment labels out of range"

    sub.to_csv(path, index=False)
    print(f"Wrote {path} ({len(sub)} rows)")
    return sub


# ------------------------------------------------------------- self-test ---
class MajorityBaseline:
    """Always predicts the most common class. Use it to check the pipeline works."""

    def fit(self, X, y):
        self.label_ = np.bincount(y).argmax()
        return self

    def predict(self, X):
        return np.full(len(X), self.label_, dtype=np.int64)


class NearestCentroid:
    """Tiny from-scratch baseline: predict the class whose mean vector is closest."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ self.centroids_.T + (self.centroids_ ** 2).sum(1)[None]
        return self.classes_[d.argmin(axis=1)]


def _check():
    set_seed()
    for task in TASKS:
        X_tr, X_val, y_tr, y_val = get_split(task)
        print(f"[{task}] train {X_tr.shape}, val {X_val.shape}, "
              f"classes in val: {len(np.unique(y_val))}")
        evaluate(MajorityBaseline(), task, "majority", save=False, verbose=True)
        evaluate(NearestCentroid(), task, "nearest_centroid", save=False, verbose=True)
    print("\nSetup OK. Split saved in splits/ (commit it so everyone shares it).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="sanity-check data + split")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        ap.print_help()