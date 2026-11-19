# k-Nearest Neighbours: results summary

Source data: `results/knn/` (produced by `python knn_experiments.py --all`).
Values are validation accuracy, with training accuracy in brackets, unless stated.

## 1. How the model works

**k-NN** (`knn.py`, NumPy only).
- `fit` only stores the training rows. k-NN is a "lazy" learner: there is no training step.
- `predict`, for each query row:
  1. compute its distance to every training row,
  2. take the k closest rows (its neighbours),
  3. let the neighbours vote. **Majority**: one vote each. **Weighted**: each vote counts 1/distance,
     so closer neighbours count more.
- Distances:
  - **Euclidean** `||a - b||`. Computed for a whole batch with one matrix multiply,
    using `||a - b||² = ||a||² + ||b||² - 2 a·b`.
  - **Cosine** `1 - cos(angle between a and b)`. Only the direction of the vectors matters, not
    their length. Also one matrix multiply, on unit-length vectors.
  - **Manhattan** `Σ |a_i - b_i|`. It has no matrix-multiply shortcut, so it is about 50× slower.
- Finding the k smallest distances uses `np.argpartition` (O(n)), then sorts only those k.
- **Ties** (common with 50 classes: on the image validation set at k = 5, 38% of votes are tied)
  go to the class of the closest neighbour. sklearn's version gives ties to the lowest label
  instead. On synthetic data of the same shape, ours scored 2 to 5 points higher.
- **Checked against sklearn** (used for testing only): with weighted voting the predictions are
  identical. With majority voting they differ only where the vote is tied.

**Sweep shortcut.** Finding neighbours is the slow part and voting is cheap. So neighbours are
found once per (features, metric) at k = 200, and every smaller k and both voting schemes are
scored from that one search. This gives exactly the same predictions as separate fits, because the
k nearest neighbours are the first k of the 200 nearest.

## 2. Validation method

- The shared stratified 80/20 split from `common.py` (seed 42, ids in `splits/`).
  Image: 16,000 train / 4,000 validation. Sentiment: 9,600 train / 2,400 validation.
- Grid, per task: features raw / z-score normalized × metric euclidean / cosine / manhattan ×
  k = 1, 3, 5, 10, 20, 50, 100, 200 × voting majority / weighted. That is 96 settings per task, all
  scored on validation.
- Training accuracy uses a 4,000-row sample of the training split, as in `common.evaluate`.
- The Kaggle leaderboard was not used for any choice.
- **Validation noise:** one standard error is about ±0.008 on image (4,000 rows) and ±0.010 on
  sentiment (2,400 rows). Differences under about 0.02 should not be treated as real.

## 3. What the features look like

Checked before tuning, because it explains most of the results:

| | Image | Sentiment |
|---|---|---|
| Length of each row vector (L2 norm) | exactly 1.000 for every row | 7.6 to 13.6 |
| Spread of each feature (std) | 0.007 to 0.100 | 0.15 to 0.63 |
| Nearest neighbour has the same label | 53.4% (chance 2%) | 49.7% (chance 33%) |
| Share of the 10 nearest with the same label | 23.6% | 47.1% |

(The last two rows use cosine distance on normalized features.)

- **The image vectors are already unit length.** For unit vectors `||a - b||² = 2 - 2 cos(a, b)`,
  so Euclidean and cosine distance rank neighbours in exactly the same order.
- **Image classes are small, tight groups.** The single nearest neighbour is right 53% of the time,
  but most of the 10 nearest belong to other classes.
- **Sentiment neighbourhoods are mixed but lean the right way.** Any one neighbour is barely better
  than chance, but nearly half of the 10 nearest share the label.

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

Weighted voting: never worse, identical at k = 1, and up to +0.017 better at larger k
(raw cosine, k = 20: 0.386 → 0.402).
Normalized manhattan is the worst setting: 0.310 at k = 1.

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

(Majority vote. Weighted voting changes these by -0.004 to +0.009, in no consistent direction.)

Training vs validation, norm cosine, majority vote:

| k | 1 | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| Training | 1.000 | 0.824 | 0.752 | 0.690 | 0.654 | 0.619 | 0.610 | 0.606 |
| Validation | 0.497 | 0.527 | 0.548 | 0.558 | 0.566 | 0.572 | 0.584 | 0.579 |

Plots (copies in `reports/knn_figures/`): `knn_k_curve_<task>.png` (validation accuracy vs k,
all settings) and `knn_train_vs_val_<task>.png` (training vs validation vs k).

## 5. What the results show

**Effect of k: opposite on the two tasks.**
- *Image: best at k = 1, worse with every larger k* (0.534 → 0.263 at k = 200). The classes form
  small, tight groups: the closest row is usually the right class, but the next ones are mostly
  other classes, which outvote it. Larger k mostly adds wrong votes. Training accuracy falls at the
  same rate, so large k is **underfitting**: the vote averages over an area much bigger than one
  class.
- *Sentiment: worst at k = 1 (0.497), rising to about 0.58 at k = 50 to 100, then dropping.* This
  is the textbook bias–variance curve. At k = 1, training accuracy is 1.000 against 0.497 on
  validation: the model memorises noisy neighbours (**overfitting**). Each extra neighbour is a
  noisy vote that leans the right way, so averaging more of them helps. Past about 100 the
  neighbourhood takes in too much of the other classes and accuracy falls again (**underfitting**).
  At k = 100 the gap is small (0.610 vs 0.584), so the model generalises well.
- *Why k = 1 scores 100% on training:* every training row is stored, so it is its own nearest
  neighbour at distance 0. This does not mean k = 1 is good. On image it happens to be the best
  setting anyway; on sentiment it is the worst.

**Does feature normalization help?**
- *Image: it depends on the distance.* With cosine, it changes almost nothing (0.531 → 0.534,
  within noise). With Euclidean, it hurts a lot (0.531 → 0.434).
  **Why:** the raw image vectors all have length 1. After z-scoring, their lengths range from 14
  to 31. Euclidean distance now depends on how long each vector is, which carries little class
  information. Cosine ignores length, so it is unaffected. We tested this: z-scoring and then
  rescaling every row back to length 1 gives Euclidean 0.534, the same as cosine.
  Euclidean accuracy is also lowest on the shortest z-scored rows (0.373 for the shortest quarter
  vs 0.492 for the longest).
- *Sentiment: no meaningful effect.* The best result changes by less than 0.01 for every metric,
  inside validation noise. The BERT features already have similar scales (std 0.15 to 0.63).
- Normalization is not free for k-NN. It changes which points count as "close", and that can
  help or hurt.

**Distance function.**
- Image: cosine ≈ Euclidean on raw features (identical ranking for unit vectors). Manhattan is much
  worse (0.393 raw, 0.310 normalized). A likely reason: Manhattan counts every dimension's
  difference equally, while Euclidean squares the differences, so a few large, informative
  differences weigh more than many small noisy ones.
- Sentiment: all three are within 0.01 of each other at their best k, so we pick cosine for speed.
  Manhattan's 0.588 is the single highest number, but it is within noise and over 60× slower.

**Weighted vs majority voting.**
- No effect at k = 1, as expected (one vote).
- Image: helps slightly at large k, because the closest neighbours are the ones most likely to be
  the right class. Too small to matter at the best k.
- Sentiment: no consistent effect (±0.009).
- With weighted voting, training accuracy is always about 1.000 at every k, because each training
  row votes for itself at distance 0 with a huge weight. That is why the over/underfitting plots
  use majority voting.

## 6. Error analysis (best settings)

### Sentiment: per class (norm cosine, k = 100, weighted)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.584 | **0.692** | 0.634 |
| Neutral | 0.541 | **0.460** | 0.497 |
| Positive | 0.622 | 0.600 | 0.611 |

Confusion matrix (rows true, columns predicted):

| | neg | neu | pos |
|---|---|---|---|
| **neg** | 554 | 126 | 120 |
| **neu** | 260 | 368 | 172 |
| **pos** | 134 | 186 | 480 |

- **Neutral is the hardest class**, as for the tree and forest (recall 0.460). Neutral
  examples are pulled to both sides, and most often to negative (260 of 800).
- The model leans toward predicting negative: 948 of 2,400 predictions are negative, though
  validation is balanced (800 each). On the test set the lean is stronger (2,645 of 5,000 in
  `submission_knn.csv`).
- At k = 1 every class is worse and more balanced (recall 0.54 / 0.44 / 0.51). The noisy single
  neighbour hurts all three.

### Image: most-confused classes (norm cosine, k = 1)

- True → predicted (count): 4→2 (9), 39→38 (9), 3→2 (9), 6→7 (9), 10→12 (8), 38→37 (7),
  37→38 (7), 27→25 (7), 13→12 (7).
- Lowest recall: 29 (0.31), 9 (0.35), 22 (0.36), 30 (0.36), 42 (0.38).
  Highest recall: 46 (0.89), 48 (0.76), 2 (0.71), 12 (0.71), 5 (0.71).
- **Same hard spots as the tree and forest** (`trees_summary.md`): the group 2 / 3 / 4,
  38 ↔ 39 and 6 → 7 are confused by all three models. 9 and 29 are among the hardest and 46 is the
  easiest for all three. These classes overlap in the feature space itself, not just for one model.
- Most top confusions are between labels with nearby numbers (2/3/4, 37/38/39, 6/7, 10/12/13).
  That suggests the dataset numbers similar object types next to each other, though the dataset
  gives no class names to confirm it.
- Per-row predictions are in `results/preds/`, so `error_analysis.py` can check which rows k-NN,
  the forest and the neural net all get wrong.

## 7. Comparison with the other models so far

| | Image val | Sentiment val |
|---|---|---|
| Nearest-centroid baseline (`common.py --check`) | 0.223 | 0.578 |
| Best decision tree | 0.119 | 0.483 |
| Best random forest | 0.284 | 0.587 |
| **Best k-NN** | **0.534** | 0.584 |

- **Image: k-NN is by far the best so far.** It almost doubles the forest (0.534 vs 0.284).
  Distances use all 500 features at once, which suits embeddings where class information is spread
  across many dimensions. Each tree split looks at one feature only.
- **Sentiment: k-NN, the forest and the nearest-centroid baseline are all about 0.58.** The
  differences are within noise. This looks like a ceiling for these features with simple methods.
  The classes overlap heavily (each neighbour is only slightly better than chance), so there is
  little local structure to find.
- Nearest centroid scores 0.223 on image but 1-NN scores 0.534. So image classes are **not** one
  round cluster each: they are spread out or made of several small groups, which a single mean
  vector cannot represent but a nearest neighbour can.

## 8. Time and memory

| | Image | Sentiment |
|---|---|---|
| Fit (store the data) | 0.01 s | 0.01 s |
| Predict 4,000 / 2,400 validation rows, cosine or Euclidean | about 0.9 s | about 0.4 s |
| Predict, Manhattan | about 50 s | about 30 s |
| Model size (stored training rows, float32) | 32 MB (16,000 × 500) | 29 MB (9,600 × 768) |

Measured on a 2-core cloud machine, one job at a time. The trade-off is the opposite of the
forest's: training is free, but every prediction scans the whole training set, so prediction cost
grows with the training set size, and the model is as big as the training data.

## 9. Chosen settings

- **Image:** cosine distance, k = 1, normalized features (raw features give the same result within
  noise).
- **Sentiment:** cosine distance, k = 100, weighted voting, normalized features.
- Overall validation score: (0.534 + 0.584) / 2 = **0.559**.
- `python knn_experiments.py --submit` refits on all training rows and writes
  `submission_knn.csv`, a valid backup Kaggle file. Whether k-NN is part of the final submission
  depends on the neural network's validation results.

## 10. Limitations and what to try next

- **Bug found and fixed:** floating-point rounding could make a point's cosine distance to itself
  slightly negative (about -1e-7). Weighted voting then gave a huge negative vote to the correct
  label, so sentiment training accuracy showed 0.666 at every k, even k = 1, instead of 1.000.
  Distances are now clamped at 0. Every result was re-run after the fix; validation numbers did not
  change, because validation rows are not in the training set.
- No dimensionality reduction (e.g. PCA) was tried. It could remove noisy directions and speed up
  the distance computation.
- The k values form a grid. The true sentiment optimum is somewhere between 50 and 200.
- No learned distance metric: every feature counts equally. A neural network can learn which
  directions matter, which is the main advantage it should have here.
