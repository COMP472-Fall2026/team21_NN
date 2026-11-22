# k-Means Clustering

All numbers below come from `results/clustering/`, which is produced by `python clustering.py`
(about 10 minutes). The figures are in `reports/clustering_figures/`.

## 1. Setup

We wrote k-Means from scratch in NumPy (`clustering.py`). It starts with k-means++ initialisation:
the first centre is a random point, and each next centre is picked with probability proportional to
its squared distance from the closest centre so far, which spreads the starting centres out. From
there it runs the usual Lloyd iterations, assigning every point to its nearest centre and then moving
each centre to the mean of its points, until the total within-cluster squared distance (the inertia)
improves by less than 1e-6. We do 5 random restarts with seed 42 and keep the one with the lowest
inertia. Like our k-NN, the distances for a whole batch come from one matrix multiply using
`||x - c||² = ||x||² + ||c||² - 2 x·c`.

To verify the implementation, we compared it against sklearn's KMeans (for testing only). The
inertia came out within 0.2% on both tasks and every preprocessing option, and the agreement scores
matched to 2 or 3 decimals. This check also caught a bug in our first version, where the stopping
condition triggered after a single iteration.

We clustered all of the labelled training rows (20,000 images and 12,000 sentiment examples). The
labels are never used by k-Means; we only use them afterwards to score how well the clusters line up
with the real classes. We tried two kinds of preprocessing: `zscore`, where each feature is
standardised the same way the classifiers use, and `l2`, where each row is scaled to length 1 so that
Euclidean k-Means behaves like clustering by cosine similarity.

We scored the clusters three ways, where 0 means no relation to the labels and 1 means the clusters
are exactly the classes. Purity labels each cluster with its most common class and counts how many
points match; it goes up with k on its own, so it only makes sense to compare it at the same k. ARI
(adjusted Rand index) counts how many pairs of points agree, and it is corrected so random clusters
score 0. NMI (normalised mutual information) measures how much knowing a point's cluster tells you
about its class.

## 2. Sentiment: do negative, neutral and positive separate? (k = 3)

| Preprocessing | Purity | ARI | NMI |
|---|---|---|---|
| zscore | 0.423 | 0.020 | 0.019 |
| l2 | 0.387 | 0.014 | 0.014 |
| random guess | 0.333 | 0 | 0 |

Here is how the three clusters line up with the true labels (zscore):

| Cluster | Negative | Neutral | Positive | Size | Majority |
|---|---|---|---|---|---|
| 0 | 386 | 754 | 461 | 1,601 | neutral (47%) |
| 1 | 1,346 | 1,635 | 2,053 | 5,034 | positive (41%) |
| 2 | 2,268 | 1,611 | 1,486 | 5,365 | negative (42%) |

The sentiment classes do not separate on their own. Each cluster leans slightly towards one
sentiment, so we get one mostly negative cluster, one mostly positive and a small mostly neutral one,
but every cluster still holds a big share of all three classes. An ARI of 0.02 is very close to
random.

This is likely due to how the BERT embeddings were produced. They were not trained for
sentiment, so the biggest directions of variation in them are probably things like topic, length and
writing style, and those are what k-Means follows. Sentiment is a smaller signal. A supervised model
can still pick it out (our neural network reaches 66% validation accuracy), but it does not dominate
the distances.

## 3. Images: do similar objects cluster together? (k = 50)

| Preprocessing | Purity | ARI | NMI |
|---|---|---|---|
| zscore | 0.115 | 0.029 | 0.133 |
| l2 | 0.109 | 0.025 | 0.117 |
| random guess | 0.020 | 0 | 0 |

The image clusters are about 6 times purer than chance, but they are still far from matching the 50
classes.

Some classes hold together much better than others. If we look at the share of each class that lands
in its single biggest cluster, the most cleanly clustered classes are 46 (43%), 48 (39%), 2 (38%),
13 (37%) and 45 (30%). The most scattered are 35 (7%), 38 (8%), 36 (8%), 0 (8%) and 15 (8%), which
are each spread over roughly 15 clusters.

To find which classes get grouped together, we compared how each pair of classes is spread across
the clusters (the cosine similarity of their cluster histograms). These pairs end up together the
most:

| Class pair | Similarity |
|---|---|
| 12 and 25 | 0.94 |
| 27 and 29 | 0.92 |
| 30 and 31 | 0.91 |
| 37 and 39 | 0.90 |
| 15 and 16 | 0.88 |
| 24 and 42 | 0.88 |
| 3 and 4 | 0.88 |

The full lists are in `results/clustering/image_k50_per_class_*.csv` and `image_k50_pairs_*.csv`.

We also examined whether these are the same classes our classifiers mix up. For each model we
took the 25 class pairs it confuses most on the validation set and checked how many of them are also
in k-Means' top 25 pairs. By chance, about 0.5 of the 25 would overlap.

| Model | Overlap with k-Means' top 25 pairs | Mean cluster similarity of its 25 most-confused pairs (all pairs: 0.50) |
|---|---|---|
| Neural net | 10 / 25 | 0.78 |
| Random forest | 7 / 25 | 0.68 |
| k-NN (k = 1, cosine) | 6 / 25 | 0.62 |

The overlap is far above chance for all three models. The classes k-Means cannot tell apart without
labels are largely the same ones the classifiers confuse even with labels, which tells us those
classes really do sit close together in the feature space.

## 4. Effect of k

The plots are `kmeans_vs_k_sentiment.png` and `kmeans_vs_k_image.png`. The tables below use the
zscore features.

| Sentiment k | 2 | 3 | 4 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|
| Purity | 0.391 | 0.423 | 0.426 | 0.421 | 0.441 | 0.456 | 0.471 | 0.475 |
| ARI | 0.014 | 0.020 | 0.022 | 0.022 | 0.024 | 0.027 | 0.019 | 0.016 |

| Image k | 10 | 25 | 40 | 50 | 75 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| Purity | 0.054 | 0.078 | 0.107 | 0.115 | 0.117 | 0.133 | 0.152 |
| ARI | 0.011 | 0.019 | 0.025 | 0.029 | 0.028 | 0.030 | 0.030 |
| NMI | 0.054 | 0.098 | 0.123 | 0.133 | 0.141 | 0.155 | 0.176 |

Purity and NMI keep going up as k grows, partly because smaller clusters are
automatically purer. ARI corrects for this and shows a different pattern: for images it levels off
around the real number of classes (k = 50 to 100), and for sentiment it peaks around k = 10 and then
drops. Using more clusters than there are classes does not reveal additional class structure. The
inertia also falls smoothly with no clear "elbow", which is another sign that the data does not have a
strong natural grouping at any k. Between the two preprocessing options, zscore was slightly better
than l2 on ARI and NMI for both tasks.

## 5. Conclusions

Neither feature set separates into its labels without supervision. The class information is clearly
there, since the classifiers find it, but it is not the main structure in the features. Images cluster
noticeably better than sentiment relative to chance; some object classes form fairly tight groups,
while sentiment barely affects the overall shape of the BERT embeddings.

This also fits with our classifier results. k-NN with k = 1 works well on images because the closest
neighbour is usually the same class, even though larger regions mix many classes together, and that
larger-scale mixing is exactly what k-Means picks up on. Finally, the class pairs that k-Means merges
are the same pairs our classifiers confuse, which feeds directly into the error analysis.
