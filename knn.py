"""
k-Nearest Neighbours classifier, implemented from scratch with NumPy.

COMP 472 project: no sklearn here. The full algorithm (distance computation,
neighbour search and voting) is written by hand.

How it works
------------
k-NN is a "lazy" learner: fit() only stores the training data. At prediction
time, for each test point we:
  1. compute its distance to every training point,
  2. take the k closest training points (its "neighbours"),
  3. let those neighbours vote on the label (plain majority, or weighted by
     1/distance so closer neighbours count more).

Design choices we can tune
--------------------------
- k               : number of neighbours (small k = low bias / high variance,
                    large k = smoother decision boundary / higher bias)
- metric          : 'euclidean', 'cosine' or 'manhattan'
- weighted        : False = majority vote, True = 1/distance weighted vote
- normalization   : handled outside this class (common.py fits a scaler on the
                    training split only), because k-NN distances are very
                    sensitive to feature scale.

Speed trick for hyperparameter sweeps
-------------------------------------
Finding neighbours is the expensive part; voting is cheap. So kneighbors()
finds the k_max closest points ONCE, and vote() can then produce predictions
for every k <= k_max and both voting schemes without recomputing distances.
"""

import numpy as np

METRICS = ("euclidean", "cosine", "manhattan")


class KNN:
    def __init__(self, k=5, metric="euclidean", weighted=False, batch_size=256):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.batch_size = batch_size

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        """Lazy learner: just store the training data."""
        self.X_train = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        # map labels to 0..C-1 so we can use np.bincount-style voting
        self.classes_, self.y_train = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes_)

        if self.metric == "euclidean":
            # cache ||x||^2 of training points for the fast distance formula
            self._train_sq = np.sum(self.X_train ** 2, axis=1)
        elif self.metric == "cosine":
            # cache unit-length training vectors
            self._train_unit = self.X_train / (
                np.linalg.norm(self.X_train, axis=1, keepdims=True) + 1e-12
            )
        return self

    # ------------------------------------------------------------ distances
    def _distances(self, Xb):
        """Distances between a batch of query points Xb and all training points.
        Returns an array of shape (len(Xb), n_train)."""
        if self.metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (one matrix multiply, fast)
            q_sq = np.sum(Xb ** 2, axis=1, keepdims=True)
            d2 = q_sq + self._train_sq[None, :] - 2.0 * (Xb @ self.X_train.T)
            np.maximum(d2, 0.0, out=d2)  # remove tiny negatives from rounding
            return np.sqrt(d2)

        if self.metric == "cosine":
            # cosine distance = 1 - cos(angle) between the two vectors
            q_unit = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)
            d = 1.0 - q_unit @ self._train_unit.T
            # float rounding can make a point's distance to itself slightly
            # negative (e.g. -1e-7); clamp so 1/distance weights stay positive
            np.maximum(d, 0.0, out=d)
            return d

        # manhattan: sum |a - b|; no matrix trick exists, so loop over the
        # (small) query batch to keep memory bounded
        out = np.empty((len(Xb), len(self.X_train)), dtype=np.float32)
        for i, q in enumerate(Xb):
            out[i] = np.abs(self.X_train - q).sum(axis=1)
        return out

    # ------------------------------------------------------- neighbour search
    def kneighbors(self, X, k=None):
        """Return (distances, indices) of the k nearest training points for
        each row of X, sorted from closest to farthest. Shapes: (n, k)."""
        k = self.k if k is None else k
        k = min(k, len(self.X_train))
        X = np.asarray(X, dtype=np.float32)
        bs = self.batch_size if self.metric != "manhattan" else 32

        all_d = np.empty((len(X), k), dtype=np.float32)
        all_i = np.empty((len(X), k), dtype=np.int64)
        for start in range(0, len(X), bs):
            D = self._distances(X[start:start + bs])
            # argpartition finds the k smallest in O(n) without a full sort...
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            d = np.take_along_axis(D, idx, axis=1)
            # ...then we only sort those k
            order = np.argsort(d, axis=1, kind="stable")
            all_i[start:start + bs] = np.take_along_axis(idx, order, axis=1)
            all_d[start:start + bs] = np.take_along_axis(d, order, axis=1)
        return all_d, all_i

    # ---------------------------------------------------------------- voting
    def vote(self, dist, idx, k=None, weighted=None):
        """Turn neighbour lists into class predictions.
        dist, idx: outputs of kneighbors() (may contain more than k columns)."""
        k = self.k if k is None else k
        weighted = self.weighted if weighted is None else weighted
        d = np.maximum(dist[:, :k].astype(np.float64), 0.0)  # safety: no negative distances
        labels = self.y_train[idx[:, :k]]            # (n, k) neighbour labels

        if weighted:
            w = 1.0 / (d + 1e-8)                     # closer => bigger vote
        else:
            w = np.ones_like(d)
        # tie-break: a tiny bonus that decreases with rank, so if two classes
        # tie the class of the closer neighbour wins (never changes a clear win)
        w = w + 1e-6 * (k - np.arange(k))[None, :]

        scores = np.zeros((len(d), self.n_classes))
        rows = np.repeat(np.arange(len(d)), k)
        np.add.at(scores, (rows, labels.ravel()), w.ravel())
        return self.classes_[np.argmax(scores, axis=1)]

    # --------------------------------------------------------------- predict
    def predict(self, X):
        dist, idx = self.kneighbors(X, self.k)
        return self.vote(dist, idx, self.k, self.weighted)

    def get_params(self):
        return {"k": self.k, "metric": self.metric, "weighted": self.weighted}
