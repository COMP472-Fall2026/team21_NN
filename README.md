# COMP472 shared testing setup

Everyone uses `common.py` so all models get the same data, the same 80/20 split and the same metrics.

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
| `results/preds/<task>_<model>_<params>.npy` | validation predictions, used for error analysis |

**Always pass `params`** with the hyperparameters you used, so the rows in the table can be told apart.

Train accuracy vs val accuracy is how we show overfitting (e.g. a deep tree with train ≈ 1.0 but much lower val).

## Making a Kaggle submission

```python
from common import fit_full_and_predict, make_submission

img_ids, img_pred = fit_full_and_predict(BestImageModel(), "image")        # retrains on all 20k
txt_ids, txt_pred = fit_full_and_predict(BestSentimentModel(), "sentiment") # retrains on all 12k
make_submission(img_ids, img_pred, txt_ids, txt_pred)  # writes submission.csv, runs the official checks
```

Pick the model using **validation accuracy**, not the Kaggle leaderboard.

## Other helpers

- `get_split(task)` gives `X_train, X_val, y_train, y_val` directly (e.g. for the NN's early stopping or clustering)
- `top_confusions(cm, k=10)` lists the most-confused (true, predicted, count) class pairs
- `set_seed()` seeds python, numpy and torch