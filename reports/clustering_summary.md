# k-Means clustering: results summary

Source data: `results/clustering/` (produced by `python clustering.py`, about 10 min).
Figures: `reports/clustering_figures/`.

## 1. Setup

**k-Means** (`clustering.py`, NumPy only).
- k-means++ initialisation: the first centre is a random point, each next centre is picked with
  probability proportional to its squared distance from the nearest centre so far.
- Lloyd iterations: assign every point to its nearest centre, move each centre to the mean of its
  points, repeat until the total within-cluster squared distance (inertia) improves by less than 1e-6.
- 5 random restarts, keep the one with the lowest inertia. Seed 42.
- Distances for a whole batch use one matrix multiply: `||x - c||² = ||x||² + ||c||² - 2 x·c`.
- **Checked against sklearn's KMeans** (testing only): inertia within 0.2% on both tasks and all
  preprocessing options, and the label-agreement scores match to 2-3 decimals.

Data: all labelled training rows (20,000 image, 12,000 sentiment). Labels are only used afterwards
to score the clusters.

Preprocessing compared:
- `zscore`: each feature standardised (same as the classifiers use)
- `l2`: each row scaled to length 1, so Euclidean k-Means behaves like clustering by cosine similarity

Scores (all 0 = no relation to the labels, 1 = clusters are exactly the classes):
- **Purity**: label each cluster with its most common class, fraction of points that match.
  Goes up with k on its own, so only compare at the same k.
- **ARI** (adjusted Rand index): pair-counting agreement, corrected so random clusters score 0.
- **NMI** (normalised mutual information): how much knowing the cluster tells you about the class.

## 2. Sentiment: do negative / neutral / positive separate? (k = 3)

| Preprocessing | Purity | ARI | NMI |
|---|---|---|---|
| zscore | 0.423 | 0.020 | 0.019 |
| l2 | 0.387 | 0.014 | 0.014 |
| random guess | 0.333 | 0 | 0 |

Cluster vs true label (zscore):

| Cluster | Negative | Neutral | Positive | Size | Majority |
|---|---|---|---|---|---|
| 0 | 386 | 754 | 461 | 1,601 | neutral (47%) |
| 1 | 1,346 | 1,635 | 2,053 | 5,034 | positive (41%) |
| 2 | 2,268 | 1,611 | 1,486 | 5,365 | negative (42%) |

**No, they do not separate naturally.** Each cluster leans slightly towards one sentiment
(there is one "mostly negative", one "mostly positive" and one small "mostly neutral" cluster),
but every cluster still contains a large share of all three classes. ARI is 0.02, very close to random.

Why: the BERT embeddings were not trained for sentiment. The biggest directions of variation in them
are things like topic, length and writing style, and k-Means follows those. Sentiment is a smaller
signal that a supervised model can pick out (the neural net reaches 66% validation accuracy) but
that does not dominate the distances.

## 3. Image: do similar objects cluster together? (k = 50)

| Preprocessing | Purity | ARI | NMI |
|---|---|---|---|
| zscore | 0.115 | 0.029 | 0.133 |
| l2 | 0.109 | 0.025 | 0.117 |
| random guess | 0.020 | 0 | 0 |

Clusters are about 6 times purer than chance, but far from the 50 classes.

**How spread out each class is** (zscore): the share of a class that lands in its single biggest cluster.
- Most cleanly clustered: class 46 (43%), 48 (39%), 2 (38%), 13 (37%), 45 (30%)
- Most split up: class 35 (7%), 38 (8%), 36 (8%), 0 (8%), 15 (8%), spread over roughly 15 clusters each

**Classes most often grouped together** (classes whose points are spread over the clusters in the
most similar way, cosine similarity of their cluster histograms):

| Class pair | Similarity |
|---|---|
| 12 and 25 | 0.94 |
| 27 and 29 | 0.92 |
| 30 and 31 | 0.91 |
| 37 and 39 | 0.90 |
| 15 and 16 | 0.88 |
| 24 and 42 | 0.88 |
| 3 and 4 | 0.88 |

Full lists: `results/clustering/image_k50_per_class_*.csv` and `image_k50_pairs_*.csv`.

**Link to the classifiers' mistakes.** For each model we took the 25 class pairs it confuses most
on the validation set and checked how many are also among the 25 pairs k-Means groups together most.
By chance about 0.5 of 25 would overlap.

| Model | Overlap with k-Means' top 25 pairs | Mean cluster similarity of its 25 most-confused pairs (all pairs: 0.50) |
|---|---|---|
| Neural net | 10 / 25 | 0.78 |
| Random forest | 7 / 25 | 0.68 |
| k-NN (k = 1, cosine) | 6 / 25 | 0.62 |

So the classes that k-Means cannot tell apart without labels are largely the same classes the
classifiers confuse with labels: they really do sit close together in feature space.

## 4. Effect of k

Figures: `kmeans_vs_k_sentiment.png`, `kmeans_vs_k_image.png`.

| Sentiment k | 2 | 3 | 4 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| Purity | 0.391 | 0.423 | 0.426 | 0.421 | 0.441 | 0.456 | 0.471 | 0.475 |
| ARI | 0.014 | 0.020 | 0.022 | 0.022 | 0.024 | 0.027 | 0.019 | 0.016 |

| Image k | 10 | 25 | 40 | 50 | 75 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| Purity | 0.054 | 0.078 | 0.107 | 0.115 | 0.117 | 0.133 | 0.152 |
| ARI | 0.011 | 0.019 | 0.025 | 0.029 | 0.028 | 0.030 | 0.030 |
| NMI | 0.054 | 0.098 | 0.123 | 0.133 | 0.141 | 0.155 | 0.176 |

(zscore features.)
- Purity and NMI keep rising with k, partly because smaller clusters are automatically purer.
- ARI, which corrects for that, levels off around the true number of classes for images (k = 50 to 100)
  and peaks around k = 10 for sentiment, then drops. Using more clusters than classes does not
  uncover a hidden class structure.
- Inertia falls smoothly with no clear "elbow", another sign that the data has no strong natural
  grouping at any k.
- zscore beats l2 slightly on ARI and NMI for both tasks.

## 5. Takeaways

- Neither feature set separates into its labels without supervision. The class information is
  there (the classifiers find it) but it is not the dominant structure in the features.
- Images cluster better than sentiment relative to chance. Some object classes form fairly tight
  groups, while sentiment barely affects the overall geometry of the BERT embeddings.
- This matches the classifier results: k-NN with k = 1 works well on images (close neighbours are
  usually the same class) even though bigger regions mix many classes, which is exactly what
  k-Means is measuring.
- Class pairs that k-Means merges are the pairs the classifiers confuse, which is useful input for
  the error analysis.
