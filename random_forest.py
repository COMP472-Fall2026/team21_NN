
"""From-scratch random forest built on decision_tree.DecisionTreeClassifier."""

import numpy as np
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self, n_estimators=50, max_depth=20, min_samples_split=2,
        min_samples_leaf=1, criterion="gini", max_features="sqrt",
        bootstrap=True, random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit decision trees on seeded bootstrap samples."""
        X = np.asarray(X)
        y = np.asarray(y)

        if (isinstance(self.n_estimators, (bool, np.bool_))
                or not isinstance(self.n_estimators, (int, np.integer))
                or self.n_estimators < 1):
            raise ValueError("n_estimators must be a positive integer.")

        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean.")

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")

        n_samples, n_features = X.shape

        if n_samples == 0 or n_features == 0 or len(y) != n_samples:
            raise ValueError("Invalid training data dimensions.")

        if self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            max_features = max(1, int(np.log2(n_features)))
        elif self.max_features is None:
            max_features = None
        elif (isinstance(self.max_features, (int, np.integer))
              and not isinstance(self.max_features, (bool, np.bool_))
              and 1 <= self.max_features <= n_features):
            max_features = int(self.max_features)
        else:
            raise ValueError("Invalid max_features value.")

        rng = np.random.default_rng(self.random_state)

        self.trees = []
        self.n_features_in_ = n_features

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.integers(0, n_samples, size=n_samples)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y

            tree_seed = int(rng.integers(0, 2**32 - 1))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_features,
                random_state=tree_seed
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray, n_trees=None) -> np.ndarray:
        """Return the majority vote of the trees for each row.

        n_trees uses only the first n_trees trees; this matches training with n_estimators=n_trees.
        """

        if not self.trees:
            raise ValueError("Random Forest has not been fitted.")

        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Invalid prediction input dimensions.")

        trees = self.trees if n_trees is None else self.trees[:n_trees]
        votes = np.stack([tree.predict(X) for tree in trees]).astype(np.int64)

        n_classes = int(votes.max()) + 1
        counts = np.zeros((X.shape[0], n_classes), dtype=np.int64)
        rows = np.broadcast_to(np.arange(X.shape[0]), votes.shape)
        np.add.at(counts, (rows, votes), 1)

        # Ties go to the lowest label.
        return np.argmax(counts, axis=1).astype(int)


if __name__ == "__main__":
    from common import evaluate, set_seed

    set_seed()
    params = {"n_estimators": 50, "max_depth": 20, "max_features": "sqrt", "criterion": "gini"}
    for task in ("image", "sentiment"):
        evaluate(RandomForestClassifier(**params), task, name="random_forest", params=params)
