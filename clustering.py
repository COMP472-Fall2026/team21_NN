"""
k-Means clustering experiment (NumPy only).

Checks how well the feature vectors separate the classes without using labels.

Usage:
    python clustering.py            run all experiments, save tables and plots
    python clustering.py --quick    fewer k values and restarts (about 1 min)
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import BASE_DIR, RESULTS_DIR, SEED, SENTIMENT_NAMES, Standardizer, load_task, _feature_cols

FIG_DIR = os.path.join(BASE_DIR, "reports", "clustering_figures")
OUT_DIR = os.path.join(RESULTS_DIR, "clustering")


class KMeans:
    """k-Means with k-means++ initialisation and several random restarts."""

    def __init__(self, k, n_init=5, max_iter=300, tol=1e-6, seed=SEED):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    @staticmethod
    def _sq_dists(X, C):
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c, done with one matrix multiply
        d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
        return np.maximum(d, 0)

    def _init_plus_plus(self, X, rng):
        # first centre at random, each next one picked with probability ~ distance^2
        # to the closest centre so far, so the starting centres are spread out
        centres = [X[rng.integers(len(X))]]
        closest = self._sq_dists(X, centres[0][None])[:, 0]
        for _ in range(1, self.k):
            nxt = X[rng.choice(len(X), p=closest / closest.sum())]
            centres.append(nxt)
            closest = np.minimum(closest, self._sq_dists(X, nxt[None])[:, 0])
        return np.array(centres)

    def _run_once(self, X, rng):
        C = self._init_plus_plus(X, rng)
        prev = None
        for it in range(self.max_iter):
            # assign each point to its closest centre
            d = self._sq_dists(X, C)
            labels = d.argmin(1)
            inertia = d[np.arange(len(X)), labels].sum()
            # stop once the total distance barely improves
            if prev is not None and prev - inertia <= self.tol * prev:
                break
            prev = inertia
            # move each centre to the mean of its points
            for j in range(self.k):
                members = X[labels == j]
                if len(members):
                    C[j] = members.mean(0)
                else:
                    # empty cluster: restart it on the point furthest from its centre
                    C[j] = X[d.min(1).argmax()]
        return labels, C, inertia, it + 1

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            res = self._run_once(X, rng)
            if best is None or res[2] < best[2]:
                best = res
        self.labels_, self.centers_, self.inertia_, self.n_iter_ = best
        return self


def purity(y, clusters):
    """Fraction of points whose cluster's most common true label matches their own."""
    total = 0
    for c in np.unique(clusters):
        total += np.bincount(y[clusters == c]).max()
    return total / len(y)


def scores(y, clusters):
    return {"purity": purity(y, clusters),
            "ari": adjusted_rand_score(y, clusters),
            "nmi": normalized_mutual_info_score(y, clusters)}


def load(task, prep):
    """All labelled training rows. prep: 'zscore' or 'l2' (unit length rows = cosine geometry)."""
    train_df, _ = load_task(task)
    X = train_df[_feature_cols(train_df)].values.astype(np.float64)
    y = train_df["label"].values.astype(np.int64)
    if prep == "zscore":
        X = Standardizer().fit(X).transform(X)
    elif prep == "l2":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, y


def contingency(y, clusters, n_classes):
    """Rows = clusters, columns = true classes, values = counts."""
    table = np.zeros((clusters.max() + 1, n_classes), dtype=int)
    np.add.at(table, (clusters, y), 1)
    return table


def sweep_k(task, ks, preps, n_init):
    rows = []
    for prep in preps:
        X, y = load(task, prep)
        for k in ks:
            km = KMeans(k, n_init=n_init).fit(X)
            r = {"task": task, "prep": prep, "k": k, "inertia": km.inertia_, **scores(y, km.labels_)}
            rows.append(r)
            print(f"  [{task}] {prep:6s} k={k:3d}  purity {r['purity']:.3f}  ARI {r['ari']:.3f}  "
                  f"NMI {r['nmi']:.3f}", flush=True)
    return pd.DataFrame(rows)


def sentiment_detail(prep, n_init):
    """k = 3: does each cluster line up with one of negative / neutral / positive?"""
    X, y = load("sentiment", prep)
    km = KMeans(3, n_init=n_init).fit(X)
    table = pd.DataFrame(contingency(y, km.labels_, 3), columns=SENTIMENT_NAMES)
    table.index.name = "cluster"
    table["size"] = table.sum(1)
    table["majority"] = table[SENTIMENT_NAMES].idxmax(1)
    table["majority_share"] = (table[SENTIMENT_NAMES].max(1) / table["size"]).round(3)
    return table, scores(y, km.labels_)


def image_detail(prep, n_init):
    """k = 50: which classes share clusters, and which classes get split up."""
    X, y = load("image", prep)
    km = KMeans(50, n_init=n_init).fit(X)
    table = contingency(y, km.labels_, 50)

    # how spread out each class is: share of the class that lands in its biggest cluster
    per_class = pd.DataFrame({
        "class": np.arange(50),
        "top_cluster_share": (table.max(0) / table.sum(0)).round(3),
        "n_clusters_with_10pct": (table / table.sum(0) >= 0.10).sum(0),
    }).sort_values("top_cluster_share")

    # classes that get grouped together: two classes are similar if they are spread
    # over the clusters in the same way (cosine similarity of their cluster histograms)
    dist = table / table.sum(0)
    unit = dist / np.linalg.norm(dist, axis=0)
    sim = unit.T @ unit
    a, b = np.triu_indices(50, k=1)
    pair_df = pd.DataFrame({"class_a": a, "class_b": b, "similarity": sim[a, b].round(3)})
    pair_df = pair_df.sort_values("similarity", ascending=False)
    return per_class, pair_df, scores(y, km.labels_)


def plot_sweep(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    for task, g in df.groupby("task"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for prep, gp in g.groupby("prep"):
            gp = gp.sort_values("k")
            # inertia is on a different scale for each prep, so show it relative to the smallest k
            rel = gp["inertia"] / gp["inertia"].iloc[0]
            for ax, vals, title in zip(axes, [gp["purity"], gp["ari"], gp["nmi"], rel],
                                       ["purity", "ARI", "NMI", "inertia (relative to smallest k)"]):
                ax.plot(gp["k"], vals, marker="o", label=prep)
                ax.set_title(title)
                ax.set_xlabel("k")
        n_true = 50 if task == "image" else 3
        for ax in axes[:3]:
            ax.axvline(n_true, color="gray", ls="--", lw=1)
        axes[0].legend()
        fig.suptitle(f"k-Means on {task} features (dashed line = number of true classes)")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"kmeans_vs_k_{task}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Saved {path}")


def plot_sentiment_table(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = table[SENTIMENT_NAMES].values
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.imshow(counts, cmap="Blues")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, counts[i, j], ha="center", va="center",
                    color="white" if counts[i, j] > counts.max() / 2 else "black")
    ax.set_xticks(range(3), SENTIMENT_NAMES)
    ax.set_yticks(range(counts.shape[0]), [f"cluster {i}" for i in range(counts.shape[0])])
    ax.set_xlabel("true label")
    ax.set_title("Sentiment, k-Means k = 3")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "kmeans_sentiment_k3.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_init = 2 if args.quick else 5
    ks = {"sentiment": [2, 3, 4, 6, 10] if args.quick else [2, 3, 4, 5, 6, 8, 10, 15, 20],
          "image": [25, 50, 100] if args.quick else [10, 25, 40, 50, 75, 100, 150]}
    preps = ["zscore", "l2"]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Sweeping k...")
    sweep = pd.concat([sweep_k(t, ks[t], preps, n_init) for t in ["sentiment", "image"]])
    sweep.round(4).to_csv(os.path.join(OUT_DIR, "kmeans_vs_k.csv"), index=False)
    plot_sweep(sweep)

    for prep in preps:
        table, s = sentiment_detail(prep, n_init)
        table.to_csv(os.path.join(OUT_DIR, f"sentiment_k3_{prep}.csv"))
        print(f"\nSentiment k=3 ({prep}): purity {s['purity']:.3f}  ARI {s['ari']:.3f}  NMI {s['nmi']:.3f}")
        print(table.to_string())
        if prep == "zscore":
            plot_sentiment_table(table)

        per_class, pairs, s = image_detail(prep, n_init)
        per_class.to_csv(os.path.join(OUT_DIR, f"image_k50_per_class_{prep}.csv"), index=False)
        pairs.to_csv(os.path.join(OUT_DIR, f"image_k50_pairs_{prep}.csv"), index=False)
        print(f"\nImage k=50 ({prep}): purity {s['purity']:.3f}  ARI {s['ari']:.3f}  NMI {s['nmi']:.3f}")
        print("Most split-up classes:\n" + per_class.head(5).to_string(index=False))
        print("Most cleanly clustered classes:\n" + per_class.tail(5).to_string(index=False))
        print("Classes most often grouped together:\n" + pairs.head(8).to_string(index=False))
