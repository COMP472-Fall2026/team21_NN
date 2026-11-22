# COMP472 Project: image and sentiment classification

k-NN, decision tree, random forest and neural network classifiers on the image and sentiment
feature vectors from the COMP472 Kaggle competition, plus a k-Means clustering experiment.

| File | What it does |
|---|---|
| `common.py` | shared data loading, 80/20 split, evaluation and submission helpers |
| `knn.py`, `knn_experiments.py` | k-NN and its hyperparameter sweep |
| `decision_tree.py`, `random_forest.py`, `sweep_trees.py` | decision tree, random forest and their sweep |
| `neural_net.py` | feed-forward neural network (PyTorch) and its sweep |
| `clustering.py` | k-Means clustering experiment |
| `error_analysis.py` | compares the mistakes of all four models |
| `make_submission.py` | builds the final Kaggle `submission.csv` |
| `reports/` | results summaries and figures for each part |

All models use `common.py`, so they get the same data, the same 80/20 split and the same metrics.

## Setup (once)

1. Put the Kaggle files in a `data/` folder next to `common.py`:
   `image_train.csv`, `image_test.csv`, `sentiment_train.csv`, `sentiment_test.csv`, `sample_submission.csv`
2. `pip install numpy pandas scikit-learn matplotlib` (plus `torch` for the neural net)
3. Run `python common.py --check`

The check creates `splits/image_val_ids.csv` and `splits/sentiment_val_ids.csv`. **Share that `splits/` folder** so we all use the exact same validation rows.

## What the split is

- Stratified 80/20 split of the training data, seed 42
  - image: 16,000 train / 4,000 val
  - sentiment: 9,600 train / 2,400 val
- Features are z-score normalized using train-split stats only. Pass `normalize=False` for raw features (useful for the k-NN "does normalization help" experiment).

## Plugging in your model

Your class just needs `fit(X, y)` and `predict(X)` (numpy arrays in, int labels out):

```python
from common import evaluate, set_seed

class MyKNN:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        ...
        return self
    def predict(self, X):
        ...

set_seed()
for k in [1, 3, 5, 10, 20]:
    evaluate(MyKNN(k=k), "image", name="knn", params={"k": k})
    evaluate(MyKNN(k=k), "sentiment", name="knn", params={"k": k})
```

Each `evaluate` call prints train/val accuracy, macro-F1 and timing, and saves:

| File | What it's for |
|---|---|
| `results/results.csv` | one row per run, the shared comparison table |
| `results/cm/<task>_<model>_<params>.png` | confusion matrix plot |
| `results/cm/<task>_<model>_<params>.csv` | confusion matrix counts (rows true, columns predicted) |
| `results/class_metrics/<task>_<model>_<params>.csv` | per-class precision, recall, F1 and support |
| `results/preds/<task>_<model>_<params>.npy` | validation predictions, used for error analysis |
| `results/preds/<task>_<model>_<params>.csv` | `id, true, pred, correct` per validation row, to find examples several models get wrong |

**Always pass `params`** with the hyperparameters you used, so the rows in the table can be told apart.

Train accuracy vs val accuracy is how we show overfitting (e.g. a deep tree with train ≈ 1.0 but much lower val).

## Making a Kaggle submission

```bash
python make_submission.py --list                                  # models and their chosen settings
python make_submission.py                                         # default models (see DEFAULT)
python make_submission.py --image knn --sentiment random_forest   # pick per task
```

Retrains the chosen model for each task on all training rows, writes `submission.csv`, and records
the choice in `results/submissions_log.csv`. Or by hand:

```python
from common import fit_full_and_predict, make_submission

img_ids, img_pred = fit_full_and_predict(BestImageModel(), "image")        # retrains on all 20k
txt_ids, txt_pred = fit_full_and_predict(BestSentimentModel(), "sentiment") # retrains on all 12k
make_submission(img_ids, img_pred, txt_ids, txt_pred)  # writes submission.csv, runs the official checks
```

Pick the model using **validation accuracy**, not the Kaggle leaderboard.

## Other helpers

- `get_val_ids(task)` gives the validation ids in the same order as `X_val`
- `get_split(task)` gives `X_train, X_val, y_train, y_val` directly (e.g. for the NN's early stopping or clustering)
- `top_confusions(cm, k=10)` lists the most-confused (true, predicted, count) class pairs
- `set_seed()` seeds python, numpy and torch

## Decision tree and random forest

`decision_tree.py` and `random_forest.py` are from-scratch implementations (NumPy only).
`python decision_tree.py` or `python random_forest.py` evaluates one configuration on both tasks.

The hyperparameter sweep changes one setting at a time from a base configuration
(gini, `min_samples_split=2`, `min_samples_leaf=1`; forest: `sqrt` features per split):

| Model | Setting | Values |
|---|---|---|
| Decision tree | max depth | 5, 10, 20, None |
| Decision tree | criterion | gini, entropy |
| Decision tree | min samples to split | 2, 20 |
| Decision tree | min samples per leaf | 1, 10 |
| Random forest | number of trees | 10, 50, 100 |
| Random forest | max depth | 10, 20, None |
| Random forest | features per split | sqrt(d), log2(d) |

```bash
python sweep_trees.py --all        # 16 jobs in parallel, one process each (--workers N to limit)
python sweep_trees.py --summary    # writes results/sweep/summary.md
```

Each decision tree job fits one unlimited-depth tree and evaluates it cut off at
each depth. Each forest's tree-count results come from the first 10, 50 and 100
trees of one 100-tree forest. Both give the same predictions as separate fits;
this was checked on both tasks.

Results, analysis and chosen settings: [reports/trees_summary.md](reports/trees_summary.md).

## k-Nearest Neighbours

`knn.py` is a from-scratch k-NN (NumPy only): euclidean / cosine / manhattan distance,
majority or 1/distance-weighted voting, ties go to the closest neighbour's class.

```bash
python knn_experiments.py --all      # full sweep + logs key settings to results/results.csv (~8 min)
python knn_experiments.py --submit   # backup Kaggle file submission_knn.csv (k-NN only, ~10 s)
```

The sweep covers raw / normalized features × 3 distances × k = 1, 3, 5, 10, 20, 50, 100, 200
× majority / weighted voting (96 settings per task). Neighbours are searched once at k = 200 and
every smaller k is scored from that search, which gives the same predictions as separate fits.

Results, analysis and chosen settings: [reports/knn_summary.md](reports/knn_summary.md).
Charts: `reports/knn_figures/`.

## Neural network

`neural_net.py` is a feed-forward network in PyTorch: Linear + ReLU + Dropout for each hidden
layer, then a Linear output layer (50 outputs for images, 3 for sentiment). Trained with Adam and
cross-entropy loss, batch size 128. Early stopping uses a 10% hold-out taken from the training rows,
so the shared validation split is never used for training decisions.

```bash
python neural_net.py            # best settings on both tasks (~20 s on CPU)
python neural_net.py --sweep    # grid over learning rate, hidden sizes, dropout, weight decay (~15 min per task)
python neural_net.py --curves   # train vs hold-out loss plots -> reports/nn_figures/
```

| Task | Hidden layers | Dropout | Learning rate | Weight decay | Val accuracy |
|---|---|---|---|---|---|
| Image | 1024, 256 | 0.5 | 3e-4 | 1e-3 | 0.366 |
| Sentiment | 512 | 0.5 | 3e-4 | 1e-3 | 0.664 |

## Clustering

`clustering.py` is a from-scratch k-Means (NumPy only) with k-means++ initialisation and 5 restarts.
It runs k = 3 on sentiment and k = 50 on images, sweeps k, and scores how well the clusters match
the true labels (purity, ARI, NMI).

```bash
python clustering.py            # full experiment (~10 min)
python clustering.py --quick    # fewer k values and restarts (~30 s)
```

Results and analysis: [reports/clustering_summary.md](reports/clustering_summary.md).
Figures: `reports/clustering_figures/`.

## Error analysis

`error_analysis.py` compares the best run of every model on the shared validation set: per-class
recall and precision, the most-confused classes, whether the models make the same mistakes, and
whether the neural network fixes the simpler models' errors. It reads the prediction files saved by
`common.evaluate()`, so run the model experiments first.

```bash
python error_analysis.py
```

Results and analysis: [reports/error_analysis_summary.md](reports/error_analysis_summary.md).
Figures: `reports/error_analysis_figures/`.

## Final Kaggle submission

k-NN (k = 1, cosine) for images and the neural network for sentiment, chosen on validation accuracy
(expected score 0.599):

```bash
python make_submission.py --image knn --sentiment neural_net
```

### Reproducibility

- Python 3.12.3, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 (split and metrics only), matplotlib 3.11.2
- PyTorch 2.8 for the neural network (runs on CPU, no GPU needed)
- Seed 42 for the split (`splits/`), as `random_state` for every tree and forest, and for the
  neural network and k-Means
- Timings are from one Windows 11 laptop with 16 logical cores, with up to 16 jobs running at once
