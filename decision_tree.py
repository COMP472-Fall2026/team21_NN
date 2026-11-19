"""From-scratch decision tree classifier (CART, gini or entropy)."""

import numpy as np


class DecisionTreeClassifier:
    def __init__(
        self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
        criterion="gini", max_features=None, random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree classifier to the training data."""
        X, y = self._validate_inputs(X, y)
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build_tree(X, y)
        return self
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate the Gini impurity of a node."""
        if(len(y) == 0):
            return 0.0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate the entropy of a node."""
        if(len(y) == 0):
            return 0.0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build the decision tree."""
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))
        majority = self._most_common_label(y)

        if ((self.max_depth is not None and depth >= self.max_depth) or
            num_samples < self.min_samples_split or
            num_classes == 1):
            return Node(value=majority, majority=majority)

        best_feature, best_threshold = self.best_split(X, y, self.criterion, self.max_features)

        if best_feature is None:
            return Node(value=majority, majority=majority)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child, majority=majority)


    def best_split(self, X: np.ndarray, y: np.ndarray, criterion: str, max_features: int = None):
        """Find the best feature and threshold using cumulative class counts."""
        n_samples, n_features = X.shape

        counts = np.bincount(y)
        one_hot = np.eye(len(counts), dtype=np.uint8)[y]

        if criterion not in ("gini", "entropy"):
            raise ValueError("Invalid criterion.")

        impurity_func = self._gini if criterion == "gini" else self._entropy
        best_impurity = impurity_func(y)
        best_feature, best_threshold = None, None

        if max_features is None:
            features = range(n_features)
        else:
            features = self._rng.choice(n_features, max_features, replace=False)

        if n_samples * len(counts) <= self._BATCH_LIMIT:
            return self._best_split_batched(X, y, counts, one_hot, np.asarray(features),
                                            criterion, best_impurity)

        for feature in features:
            order = np.argsort(X[:, feature])
            values = X[order, feature]

            # Find valid split positions
            splits = np.flatnonzero(values[1:] > values[:-1])
            splits = splits[
                (splits + 1 >= self.min_samples_leaf) &
                (n_samples - splits - 1 >= self.min_samples_leaf)
            ]

            if splits.size == 0:
                continue

            # Calculate class counts on each side of every split
            left = np.cumsum(one_hot[order], axis=0)[splits]
            right = counts - left

            nl = splits + 1
            nr = n_samples - nl

            # Calculate weighted impurity for all candidate splits
            if criterion == "gini":
                scores = 1 - (
                    (left * left).sum(axis=1) / nl +
                    (right * right).sum(axis=1) / nr
                ) / n_samples
            else:
                scores = (
                    nl * np.log2(nl) + nr * np.log2(nr)
                    - (left * np.log2(np.maximum(left, 1))).sum(axis=1)
                    - (right * np.log2(np.maximum(right, 1))).sum(axis=1)
                ) / n_samples

            j = np.argmin(scores)

            if scores[j] < best_impurity - 1e-12:
                best_impurity = scores[j]
                best_feature = int(feature)

                i = splits[j]
                best_threshold = self._threshold(values[i], values[i + 1])

        return best_feature, best_threshold

    # Measured crossover: batching is faster when n_samples * n_classes is below ~2,500.
    _BATCH_LIMIT = 2_500

    def _best_split_batched(self, X, y, counts, one_hot, features, criterion, best_impurity):
        """Same result as the per-feature loop in best_split, computed for all features at once."""
        n_samples = X.shape[0]
        Xf = X[:, features]
        order = np.argsort(Xf, axis=0)
        values = np.take_along_axis(Xf, order, axis=0)

        pos = np.arange(n_samples - 1)[:, None]
        valid = (values[1:] > values[:-1]) & \
                (pos + 1 >= self.min_samples_leaf) & \
                (n_samples - pos - 1 >= self.min_samples_leaf)

        # left[i, f, c]: count of class c among the first i + 1 rows sorted by feature f.
        left = np.cumsum(one_hot[order], axis=0)[:-1]
        right = counts - left
        nl = pos + 1
        nr = n_samples - nl

        if criterion == "gini":
            scores = 1 - (
                (left * left).sum(axis=2) / nl +
                (right * right).sum(axis=2) / nr
            ) / n_samples
        else:
            scores = (
                nl * np.log2(nl) + nr * np.log2(nr)
                - (left * np.log2(np.maximum(left, 1))).sum(axis=2)
                - (right * np.log2(np.maximum(right, 1))).sum(axis=2)
            ) / n_samples
        scores[~valid] = np.inf

        best_pos = np.argmin(scores, axis=0)
        best_scores = scores[best_pos, np.arange(len(features))]
        has_split = valid.any(axis=0)

        # Walk features in order with the same tolerance as the loop, so ties resolve identically.
        best_feature, best_threshold = None, None
        for k in range(len(features)):
            if has_split[k] and best_scores[k] < best_impurity - 1e-12:
                best_impurity = best_scores[k]
                best_feature = int(features[k])
                i = best_pos[k]
                best_threshold = self._threshold(values[i, k], values[i + 1, k])
        return best_feature, best_threshold

    @staticmethod
    def _threshold(lo, hi):
        """Midpoint of lo and hi, or lo if the midpoint rounds onto hi in float32."""
        mid = (float(lo) + float(hi)) / 2.0
        if np.asarray(mid, dtype=np.asarray(hi).dtype) >= hi:
            return float(lo)
        return mid


    def _most_common_label(self, y: np.ndarray) -> int:
        """Return the most common class label in y."""
        if len(y) == 0:
            raise ValueError("Cannot determine the most common label of an empty array.")
        counts = np.bincount(y)
        return int(np.argmax(counts))

    def predict(self, X: np.ndarray, max_depth=None) -> np.ndarray:
        """Traverse the learned tree and return predictions.

        max_depth cuts the tree off at that depth; this matches training with that max_depth.
        """
        predictions = [self._traverse_tree(x, self.root, max_depth) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x: np.ndarray, node, max_depth=None) -> int:
        """Traverse the tree to classify one sample."""
        depth = 0
        while not node.is_leaf_node():
            if max_depth is not None and depth >= max_depth:
                return node.majority
            depth += 1
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate training data and model hyperparameters."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numerical features.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer labels.")
        if np.any(y < 0):
            raise ValueError("Class labels must be nonnegative.")
        if self.max_depth is not None and (
                isinstance(self.max_depth, (bool, np.bool_))
                or not isinstance(self.max_depth, (int, np.integer))
                or self.max_depth <= 0):
            raise ValueError("max_depth must be a positive integer or None (unlimited).")
        if not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if not isinstance(self.criterion, str) or    self.criterion not in ["gini", "entropy"]:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if self.max_features is not None:
            if (not isinstance(self.max_features, (int, np.integer)) or self.max_features < 1 or self.max_features > X.shape[1]):
                raise ValueError("max_features must be an integer between 1 and the number of features.")
        return X, y 
    
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None,
                 majority=None):
        self.feature_index = feature_index  
        self.threshold = threshold            
        self.left = left                      
        self.right = right                    
        self.value = value                    # Class label for leaf nodes
        self.majority = majority              # Most common training label at this node
    def is_leaf_node(self):
        return self.value is not None


if __name__ == "__main__":
    from common import evaluate, set_seed

    set_seed()
    params = {"max_depth": 70, "min_samples_split": 5, "min_samples_leaf": 3, "criterion": "gini"}
    for task in ("image", "sentiment"):
        evaluate(DecisionTreeClassifier(**params), task, name="decision_tree", params=params)
