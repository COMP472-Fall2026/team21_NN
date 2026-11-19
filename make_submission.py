"""
Build the final Kaggle submission.

    python make_submission.py                                  # default models below
    python make_submission.py --image knn --sentiment random_forest
    python make_submission.py --list                           # show the options

For each task, the chosen model is retrained on ALL training rows (not just
the 80% train split) and then predicts the test set. Predictions are matched to
sample_submission.csv by id and checked (common.make_submission), then written
to submission.csv.

How to choose: pick the model with the best VALIDATION accuracy for each task
(see results/results.csv and reports/), never the Kaggle public leaderboard.
Each run is recorded in results/submissions_log.csv so we can show which
models went into each file we uploaded.
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from common import BASE_DIR, RESULTS_DIR, fit_full_and_predict, make_submission, set_seed

# Best setting found for each model on the validation split.
# val_acc is copied from the reports; it is for the log only.
SETTINGS = {
    "knn": {
        "image": {"params": {"k": 1, "metric": "cosine", "weighted": False},
                  "normalize": True, "val_acc": 0.534},
        "sentiment": {"params": {"k": 100, "metric": "cosine", "weighted": True},
                      "normalize": True, "val_acc": 0.584},
    },
    "random_forest": {
        "image": {"params": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt",
                             "criterion": "gini"}, "normalize": True, "val_acc": 0.284},
        "sentiment": {"params": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt",
                                 "criterion": "gini"}, "normalize": True, "val_acc": 0.587},
    },
    "decision_tree": {
        "image": {"params": {"max_depth": None, "min_samples_leaf": 10, "criterion": "gini"},
                  "normalize": True, "val_acc": 0.119},
        "sentiment": {"params": {"max_depth": 5, "criterion": "gini"},
                      "normalize": True, "val_acc": 0.483},
    },
    # Neural network: add its best settings here once neural_net.py is done, e.g.
    # "neural_net": {
    #     "image":     {"params": {...}, "normalize": True, "val_acc": ...},
    #     "sentiment": {"params": {...}, "normalize": True, "val_acc": ...},
    # },
}

# Models chosen for the submission. Update after comparing validation accuracy.
DEFAULT = {"image": "knn", "sentiment": "knn"}


def build(model_name, task):
    """Create an untrained model with its chosen settings for this task."""
    params = SETTINGS[model_name][task]["params"]
    if model_name == "knn":
        from knn import KNN
        return KNN(**params)
    if model_name == "random_forest":
        from random_forest import RandomForestClassifier
        return RandomForestClassifier(**params)
    if model_name == "decision_tree":
        from decision_tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)
    if model_name == "neural_net":
        # neural_net.py must provide a class with fit(X, y) and predict(X),
        # like the other models. Change the class name here if it differs.
        from neural_net import NeuralNet
        return NeuralNet(task=task, **params)
    raise ValueError(f"unknown model {model_name!r}")


def log_run(choice, path):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "submissions_log.csv")
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "file": os.path.basename(path)}
    for task, model_name in choice.items():
        s = SETTINGS[model_name][task]
        row[f"{task}_model"] = model_name
        row[f"{task}_params"] = str(s["params"])
        row[f"{task}_val_acc"] = s["val_acc"]
    row["expected_score"] = round((row["image_val_acc"] + row["sentiment_val_acc"]) / 2, 4)
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path),
                               index=False)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", choices=sorted(SETTINGS), default=DEFAULT["image"])
    ap.add_argument("--sentiment", choices=sorted(SETTINGS), default=DEFAULT["sentiment"])
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "submission.csv"))
    ap.add_argument("--list", action="store_true", help="show models and their settings")
    args = ap.parse_args()

    if args.list:
        for name, tasks in SETTINGS.items():
            for task, s in tasks.items():
                print(f"{name:14s} {task:9s} val {s['val_acc']:.3f}  {s['params']}")
        return

    set_seed()
    choice = {"image": args.image, "sentiment": args.sentiment}
    preds = {}
    for task, model_name in choice.items():
        print(f"[{task}] training {model_name} on all training rows...", flush=True)
        model = build(model_name, task)
        preds[task] = fit_full_and_predict(model, task,
                                           normalize=SETTINGS[model_name][task]["normalize"])

    make_submission(*preds["image"], *preds["sentiment"], path=args.out)
    row = log_run(choice, args.out)
    print(f"Expected score from validation: {row['expected_score']:.3f} "
          f"(image {row['image_val_acc']:.3f}, sentiment {row['sentiment_val_acc']:.3f})")


if __name__ == "__main__":
    main()
