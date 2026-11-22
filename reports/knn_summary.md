# k-Nearest Neighbours

All numbers below come from `results/knn/`, which is produced by `python knn_experiments.py --all`.
Unless we say otherwise, values are validation accuracy with training accuracy in brackets.

## 1. How our k-NN works

Our k-NN is written from scratch in NumPy (`knn.py`). Fitting does nothing except store the training
rows, since k-NN is a "lazy" learner with no real training step. All the work happens at prediction
time: for each query row we compute its distance to every training row, take the k closest ones, and
let those neighbours vote on the label. With majority voting every neighbour gets one vote; with
weighted voting each vote counts 1/distance, so closer neighbours matter more.

We support three distances. Euclidean distance, `||a - b||`, is computed for a whole batch at once
with a single matrix multiply using `||a - b||² = ||a||² + ||b||² - 2 a·b`. Cosine distance,
`1 - cos(a, b)`, only cares about the direction of the vectors and not their length; it is also one
matrix multiply, done on unit-length vectors. Manhattan distance, `Σ |a_i - b_i|`, has no such
shortcut, which makes it about 50 times slower than the other two.

To find the k smallest distances we use `np.argpartition`, which is O(n), and then only sort those k.
Ties are frequent: on the image validation set at k = 5, 38% of votes
are tied because there are 50 classes. We send ties to the class of the closest neighbour, while
sklearn sends them to the lowest label. On synthetic data of the same shape, our approach scored 2 to
5 points higher. We also checked our implementation against sklearn (for testing only). With weighted
voting the predictions are identical, and with majority voting they only differ where the vote is
tied.

Finding neighbours is the slow part and voting is cheap, so the sweep finds the 200 nearest
neighbours once for each combination of features and metric, then scores every smaller k and both
voting schemes from that single search. This gives exactly the same predictions as fitting each
setting separately, since the k nearest neighbours are the first k of the 200 nearest.

## 2. How we validated

We used the shared stratified 80/20 split from `common.py` (seed 42, ids saved in `splits/`), which
gives 16,000 training and 4,000 validation rows for images and 9,600 / 2,400 for sentiment. For each
task we tried raw and z-score normalized features, the three distances, k = 1, 3, 5, 10, 20, 50, 100
and 200, and both voting schemes. That is 96 settings per task, all scored on validation. Training
accuracy is measured on a 4,000-row sample of the training split, the same way `common.evaluate`
does it. We never used the Kaggle leaderboard to make a choice.

Validation accuracy is itself subject to noise. One standard error is
about ±0.008 on images (4,000 rows) and ±0.010 on sentiment (2,400 rows), so differences
smaller than about 0.02 are not treated as significant.

## 3. What the features look like

The features were examined before tuning, since their properties explain most of the results.

| | Image | Sentiment |
|---|---|---|
| Length of each row vector (L2 norm) | exactly 1.000 for every row | 7.6 to 13.6 |
| Spread of each feature (std) | 0.007 to 0.100 | 0.15 to 0.63 |
| Nearest neighbour has the same label | 53.4% (chance 2%) | 49.7% (chance 33%) |
| Share of the 10 nearest with the same label | 23.6% | 47.1% |

(The last two rows use cosine distance on normalized features.)

The image vectors all have length exactly 1. For unit vectors `||a - b||² = 2 - 2 cos(a, b)`, so
Euclidean and cosine distance rank neighbours in exactly the same order. The image classes also seem
to form small, tight groups: the single nearest neighbour has the right label 53% of the time, but
most of the 10 nearest belong to other classes. Sentiment is the opposite. Any single neighbour is
barely better than chance, but almost half of the 10 nearest share the label, so the neighbourhoods
are mixed but lean the right way.

## 4. Results

### Image (50 classes), majority vote

| k | raw euclidean | raw cosine | norm euclidean | norm cosine | raw manhattan |
|---|---|---|---|---|---|
| 1 | 0.531 (1.000) | 0.531 (1.000) | 0.434 (1.000) | **0.534** (1.000) | 0.393 (1.000) |
| 3 | 0.521 (0.953) | 0.521 (0.953) | 0.416 (0.906) | 0.526 (0.959) | 0.348 (0.825) |
| 5 | 0.488 (0.853) | 0.488 (0.853) | 0.370 (0.740) | 0.501 (0.864) | 0.285 (0.595) |
| 10 | 0.430 (0.698) | 0.430 (0.698) | 0.304 (0.512) | 0.450 (0.702) | 0.210 (0.356) |
| 20 | 0.385 (0.565) | 0.386 (0.565) | 0.253 (0.380) | 0.412 (0.574) | 0.165 (0.238) |
| 50 | 0.326 (0.430) | 0.326 (0.430) | 0.198 (0.264) | 0.346 (0.454) | 0.118 (0.153) |
| 100 | 0.286 (0.354) | 0.285 (0.354) | 0.171 (0.210) | 0.303 (0.370) | 0.107 (0.132) |
| 200 | 0.252 (0.288) | 0.253 (0.289) | 0.160 (0.184) | 0.263 (0.309) | 0.110 (0.123) |

Weighted voting was never worse than majority voting. It is identical at k = 1 and up to 0.017
better at larger k (raw cosine at k = 20 goes from 0.386 to 0.402). Normalized Manhattan was the
worst setting overall, at 0.310 for k = 1.

### Sentiment (3 classes), validation accuracy

| k | raw euclidean | raw cosine | norm euclidean | norm cosine | norm manhattan |
|---|---|---|---|---|---|
| 1 | 0.501 | 0.512 | 0.503 | 0.497 | 0.510 |
| 3 | 0.528 | 0.531 | 0.528 | 0.527 | 0.537 |
| 5 | 0.543 | 0.548 | 0.545 | 0.548 | 0.543 |
| 10 | 0.550 | 0.562 | 0.552 | 0.558 | 0.560 |
| 20 | 0.575 | 0.563 | 0.568 | 0.566 | 0.568 |
| 50 | 0.576 | 0.576 | 0.583 | 0.572 | **0.588** |
| 100 | 0.579 | 0.581 | 0.579 | 0.584 | 0.581 |
| 200 | 0.571 | 0.575 | 0.575 | 0.579 | 0.567 |

(Majority vote. Weighted voting changes these by -0.004 to +0.009, with no consistent direction.)

Training vs validation for normalized cosine with majority vote:

| k | 1 | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| Training | 1.000 | 0.824 | 0.752 | 0.690 | 0.654 | 0.619 | 0.610 | 0.606 |
| Validation | 0.497 | 0.527 | 0.548 | 0.558 | 0.566 | 0.572 | 0.584 | 0.579 |

The plots are in `reports/knn_figures/`: `knn_k_curve_<task>.png` shows validation accuracy against k
for every setting, and `knn_train_vs_val_<task>.png` shows training against validation.

## 5. What the results show

### The effect of k is opposite on the two tasks

On images, k = 1 is the best setting and every larger k is worse, dropping from 0.534 all the way to
0.263 at k = 200. This lines up with what we saw in the data: the closest row is usually the right
class, but the next few are mostly other classes, and they outvote it. Training accuracy falls at the
same rate as validation, so a large k is underfitting here. The vote ends up averaging over an area
much bigger than any one class.

Sentiment behaves like the textbook bias-variance curve. It is worst at k = 1 (0.497), climbs to
about 0.58 at k = 50 to 100, and then drops again. At k = 1 the training accuracy is 1.000 against
0.497 on validation, which is overfitting; the model memorises noisy neighbours. Each extra neighbour
is a noisy vote that leans the right way, so averaging more of them helps, up until about k = 100
where the neighbourhood starts taking in too much of the other classes and the model underfits. At
k = 100 the gap between training and validation is small (0.610 vs 0.584), so it generalizes well.

k = 1 always reaches 100% training accuracy because every training row is stored, so each one is its
own nearest neighbour at distance 0. This does not make k = 1 a good choice: it is the best setting
on images but the worst on sentiment.

### Does normalization help?

On images it depends on the distance. With cosine it barely changes anything (0.531 to 0.534, within
noise), but with Euclidean it hurts a lot (0.531 to 0.434). The reason is that the raw image vectors
all have length 1, while after z-scoring their lengths range from 14 to 31. Euclidean distance then
starts depending on how long each vector is, which carries very little class information, whereas
cosine ignores length completely. We tested this directly: z-scoring and then rescaling every row
back to length 1 gives 0.534 with Euclidean, the same as cosine. Euclidean accuracy is also lowest on
the shortest z-scored rows (0.373 for the shortest quarter against 0.492 for the longest).

On sentiment, normalization has no meaningful effect. The best result moves by less than 0.01 for
every metric, which is inside the noise. The BERT features already have similar scales (std 0.15 to
0.63), so normalization has little to correct. In general, normalization is not neutral for k-NN; it
changes which points count as "close", which can either help or hurt.

### Distance function

On images, cosine and Euclidean are the same on raw features, since they rank neighbours identically
for unit vectors. Manhattan is much worse (0.393 raw, 0.310 normalized). A likely explanation is that
Manhattan counts every dimension's difference equally, while Euclidean squares the differences, so a
few large and informative differences outweigh many small noisy ones.

On sentiment, all three distances are within 0.01 of each other at their best k, so cosine was
selected for its speed. Manhattan's 0.588 is the single highest number, but it is
within noise and more than 60 times slower.

### Weighted vs majority voting

Weighting has no effect at k = 1, as expected with a single vote. On images it helps slightly
at large k, because the closest neighbours are the ones most likely to have the right label, but the
gain is too small to matter at the best k. On sentiment there is no consistent effect (±0.009). One
side effect is that with weighted voting the training accuracy is always about 1.000 at every k,
since each training row votes for itself at distance 0 with a huge weight. That is why our
overfitting plots use majority voting instead.

## 6. Error analysis (best settings)

### Sentiment per class (norm cosine, k = 100, weighted)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.584 | **0.692** | 0.634 |
| Neutral | 0.541 | **0.460** | 0.497 |
| Positive | 0.622 | 0.600 | 0.611 |

Confusion matrix (rows are the true class, columns the prediction):

| | neg | neu | pos |
|---|---|---|---|
| **neg** | 554 | 126 | 120 |
| **neu** | 260 | 368 | 172 |
| **pos** | 134 | 186 | 480 |

Neutral is the hardest class, like it is for the tree and the forest, with a recall of only
0.460. Neutral examples get pulled to both sides, but most often towards negative (260 of 800). The
model also leans towards predicting negative in general: 948 of the 2,400 predictions are negative
even though the validation set is balanced at 800 per class. The lean is even stronger on the test
set, with 2,645 of 5,000 predictions negative in `submission_knn.csv`. At k = 1 every class gets
worse and the recalls even out (0.54 / 0.44 / 0.51), since a single noisy neighbour hurts all three.

### Image: most-confused classes (norm cosine, k = 1)

The most common mistakes (true → predicted, count) are 4→2 (9), 39→38 (9), 3→2 (9), 6→7 (9),
10→12 (8), 38→37 (7), 37→38 (7), 27→25 (7) and 13→12 (7). The classes with the lowest recall are 29
(0.31), 9 (0.35), 22 (0.36), 30 (0.36) and 42 (0.38), and the highest are 46 (0.89), 48 (0.76),
2 (0.71), 12 (0.71) and 5 (0.71).

These are the same hard spots the tree and forest run into (see `trees_summary.md`). The group
2 / 3 / 4, the pair 38 / 39 and 6 → 7 get confused by all three models, 9 and 29 are among the
hardest classes for all of them, and 46 is the easiest for all of them. That tells us these classes
overlap in the feature space itself, not for one particular model. Most of the
top confusions are between labels with nearby numbers (2/3/4, 37/38/39, 6/7, 10/12/13), which hints
that the dataset numbers similar object types next to each other. This cannot be confirmed,
since the dataset does not provide class names.

## 7. Comparison with the other models

| | Image val | Sentiment val |
|---|---|---|
| Nearest-centroid baseline (`common.py --check`) | 0.223 | 0.578 |
| Best decision tree | 0.119 | 0.483 |
| Best random forest | 0.284 | 0.587 |
| **Best k-NN** | **0.534** | 0.584 |

On images, k-NN is by far the best of the classical models, almost doubling the forest (0.534 vs
0.284). Distances use all 500 features at once, which suits embeddings where the class information
is spread over many dimensions, while each tree split only ever looks at one feature.

On sentiment, k-NN, the forest and even the nearest-centroid baseline all land around 0.58, and the
differences are within noise. That looks like a ceiling for these features with simple methods. The
classes overlap heavily (each neighbour is only slightly better than chance), so there is little
local structure to exploit.

The baseline comparison also says something about the image data. Nearest centroid scores 0.223 but
1-NN scores 0.534, so the image classes are not one round blob each. They are spread out or made of
several small groups, which a single mean vector cannot represent but a nearest neighbour can.

## 8. Time and memory

| | Image | Sentiment |
|---|---|---|
| Fit (store the data) | 0.01 s | 0.01 s |
| Predict 4,000 / 2,400 validation rows, cosine or Euclidean | about 0.9 s | about 0.4 s |
| Predict, Manhattan | about 50 s | about 30 s |
| Model size (stored training rows, float32) | 32 MB (16,000 × 500) | 29 MB (9,600 × 768) |

These were measured on a 2-core cloud machine running one job at a time. The trade-off is the
opposite of the forest's: training is free, but every prediction scans the whole training set, so
prediction cost grows with the size of the training data and the model is as big as the data itself.

## 9. Chosen settings

For images we use cosine distance with k = 1 on normalized features (raw features give the same
result within noise). For sentiment we use cosine distance with k = 100, weighted voting and
normalized features. Together that gives an overall validation score of (0.534 + 0.584) / 2 =
**0.559**. Running `python knn_experiments.py --submit` refits on all the training rows and writes
`submission_knn.csv`, which is a valid backup Kaggle file. In the end our final submission used k-NN
for images and the neural network for sentiment.

## 10. Limitations and what we would try next

We found and fixed one bug along the way. Floating-point rounding could make a point's cosine
distance to itself slightly negative (around -1e-7), and with weighted voting that gave a huge
negative vote to the correct label. Sentiment training accuracy showed 0.666 at every k, even k = 1,
instead of 1.000. We now clamp distances at 0 and re-ran everything. The validation numbers did not
change, since validation rows are never in the training set.

With more time, we would try dimensionality reduction such as PCA, which could strip out noisy directions
and speed up the distance computation. Our k values were also a fixed grid, so the true best k for
sentiment is somewhere between 50 and 200 and we did not pin it down. Finally, k-NN treats every
feature as equally important. A neural network can learn which directions matter, and that
is the main advantage it should have over k-NN on this data.
