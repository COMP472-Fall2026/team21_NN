# Decision tree and random forest: results summary

Source data: `results/sweep/` (produced by `python sweep_trees.py --all`).
Values are validation accuracy, with training accuracy in brackets, unless stated.

## 1. How the models work

**Decision tree** (`decision_tree.py`, NumPy only).
- Binary CART tree. Each internal node tests one feature against a threshold: `x[f] <= t` goes left.
- Split search: for each feature, sort the node's rows by that feature, and use running class counts
  to score every possible threshold in one pass. The threshold is the midpoint between two
  neighbouring distinct values.
- Split score: weighted impurity of the two children, gini or entropy. The split with the lowest
  weighted impurity is kept if it is lower than the node's own impurity.
- Stops (makes a leaf) when: the node is pure, `max_depth` is reached, the node has fewer than
  `min_samples_split` rows, or no split leaves at least `min_samples_leaf` rows on each side.
- A leaf predicts the most common training label in it. Ties go to the lowest label.

**Random forest** (`random_forest.py`, built on the tree above).
- Each tree trains on a bootstrap sample: n rows drawn with replacement from the n training rows.
- At every node, a tree only considers a random subset of features: sqrt(d) or log2(d).
  Image: d = 500, so 22 (sqrt) or 8 (log2). Sentiment: d = 768, so 27 (sqrt) or 9 (log2).
- Prediction is the majority vote of all trees. Ties go to the lowest label.
- Why it helps: one deep tree has high variance (small changes in data change it a lot). Bootstrap
  samples and random feature subsets make the trees different from each other, and averaging
  different trees cancels much of that variance. The bias of each tree stays about the same.

**Feature normalization does not matter for trees.** A split only depends on the order of values
within one feature, and z-scoring does not change that order.

## 2. Validation method

- Shared stratified 80/20 split of the training data, seed 42. Ids are in `splits/`.
  Image: 16,000 train / 4,000 validation. Sentiment: 9,600 train / 2,400 validation.
  All team models use the same split.
- All settings are chosen by validation accuracy. The Kaggle leaderboard was not used for tuning.
- Training accuracy is measured on a fixed 4,000-row sample of the training split.
- Hyperparameters are changed one at a time from a base setting:
  gini, `min_samples_split=2`, `min_samples_leaf=1`, unlimited depth; forest: sqrt features.
- Shortcut: each tree setting is fitted once at unlimited depth and evaluated cut off at depths
  5, 10 and 20. The 10- and 50-tree forests are the first trees of the 100-tree forest.
  Both give exactly the same predictions as separate fits; this was checked on both tasks.

## 3. Decision tree results

### Image (50 classes)

| Max depth | gini (base) | entropy | min split 20 | min leaf 10 |
|---|---|---|---|---|
| 5 | 0.055 (0.058) | 0.072 (0.079) | 0.055 (0.058) | 0.055 (0.058) |
| 10 | 0.069 (0.097) | 0.086 (0.164) | 0.069 (0.093) | 0.070 (0.090) |
| 20 | 0.099 (0.205) | 0.101 (0.680) | 0.103 (0.177) | 0.102 (0.165) |
| None | 0.112 (1.000) | 0.100 (1.000) | 0.110 (0.552) | **0.119** (0.381) |

### Sentiment (3 classes)

| Max depth | gini (base) | entropy | min split 20 | min leaf 10 |
|---|---|---|---|---|
| 5 | **0.483** (0.535) | 0.463 (0.516) | **0.483** (0.535) | 0.482 (0.536) |
| 10 | 0.471 (0.812) | 0.458 (0.799) | 0.474 (0.777) | 0.478 (0.751) |
| 20 | 0.448 (0.997) | 0.432 (1.000) | 0.456 (0.889) | 0.481 (0.804) |
| None | 0.445 (1.000) | 0.432 (1.000) | 0.452 (0.891) | 0.481 (0.804) |

### What the tree results show

- **Depth on sentiment: overfitting.** From depth 5 to unlimited, training accuracy rises from
  0.535 to 1.000 while validation falls from 0.483 to 0.445. Deeper splits fit noise in the
  training rows. Best: depth 5.
- **Depth on image: underfitting when shallow.** A depth-5 tree has at most 32 leaves, so it
  cannot predict all 50 classes. Validation rises with depth, from 0.055 to 0.112. The full tree
  still overfits heavily: training 1.000 vs validation 0.112.
- **Does more depth raise training accuracy but lower validation accuracy?** Yes on sentiment.
  On image, validation keeps rising up to unlimited depth.
- **min_samples_leaf 10** is the best regulariser here. It stops leaves from memorising single
  rows. At unlimited depth, image training accuracy falls from 1.000 to 0.381 and validation rises
  from 0.112 to 0.119. Sentiment validation rises from 0.445 to 0.481.
- **min_samples_split 20** has a smaller effect in the same direction. At depth 5 it changes
  almost nothing, because few nodes that shallow have fewer than 20 rows.
- **gini vs entropy.** Entropy is better on image at shallow depths (depth 5: 0.072 vs 0.055) but
  worse at unlimited depth (0.100 vs 0.112). Gini is better on sentiment at every depth.
  The differences are small compared with the effect of depth and leaf size.

## 4. Random forest results

| Setting | Image | Sentiment |
|---|---|---|
| 10 trees, depth None, sqrt | 0.144 (0.997) | 0.479 (0.994) |
| 50 trees, depth None, sqrt | 0.244 (1.000) | 0.578 (1.000) |
| **100 trees, depth None, sqrt** | **0.284** (1.000) | **0.587** (1.000) |
| 50 trees, depth 10, sqrt | 0.107 (0.165) | 0.572 (0.948) |
| 50 trees, depth 20, sqrt | 0.155 (0.495) | 0.574 (1.000) |
| 50 trees, depth None, log2 | 0.233 (1.000) | 0.558 (1.000) |

### What the forest results show

- **More trees help and have not levelled off at 100.** Image: 0.144, 0.244, 0.284 for 10, 50, 100
  trees. Sentiment: 0.479, 0.578, 0.587. More trees reduce variance. They do not cause overfitting.
- **Depth.** Forests do best with unlimited depth. Each tree overfits, and averaging removes most
  of that. Depth 10 underfits image badly (0.107 validation, 0.165 training).
  On sentiment, depth matters much less (0.572 to 0.578).
- **Features per split.** sqrt beats log2 on both tasks (image 0.244 vs 0.233, sentiment 0.578 vs
  0.558, both at 50 trees). With only 8 or 9 candidate features, each split is weaker.
- **Training accuracy is 1.000 for the best forest, yet it validates far better than any tree.**
  The gap between training and validation is not the whole story. What matters is that the forest's
  validation accuracy is higher.

## 5. Tree vs forest

| | Best tree | Best forest |
|---|---|---|
| Image setting | gini, unlimited depth, min leaf 10 | 100 trees, unlimited depth, sqrt |
| Image validation accuracy / macro-F1 | 0.119 / 0.119 | 0.284 / 0.270 |
| Sentiment setting | gini, depth 5 | 100 trees, unlimited depth, sqrt |
| Sentiment validation accuracy / macro-F1 | 0.483 / 0.478 | 0.587 / 0.585 |

- The forest more than doubles image accuracy and adds 0.104 on sentiment.
- **Does the forest reduce overfitting?** Yes, in the sense that matters: its unlimited-depth trees
  each fit the training data perfectly, but the vote generalises much better than one tree.
- Context: the nearest-centroid baseline in `common.py --check` scores 0.223 (image) and 0.578
  (sentiment). The best forest beats it on image and roughly matches it on sentiment. A single tree
  is far below it on both.
- **Why trees struggle on image features:** 50 classes and only 320 training rows per class. Each
  split looks at one feature, while class information in these embeddings is likely spread across
  many features at once. Splits fragment the data quickly, so leaves end up with few rows.

## 6. Error analysis

### Sentiment: per class

| Class | Tree precision / recall | Forest precision / recall |
|---|---|---|
| Negative | 0.514 / 0.562 | 0.610 / 0.695 |
| Neutral | 0.421 / **0.342** | 0.509 / **0.475** |
| Positive | 0.497 / 0.542 | 0.638 / 0.591 |

Neutral is the hardest class for both models. Negative is the easiest.
A likely reason: neutral sits between the other two, so its examples share features with both.

### Image: most-confused classes

Labels are numbers only; the dataset gives no class names.

- **Tree** (true → predicted, count): 4→2 (11), 3→4 (10), 2→4 (10), 3→2 (8), 38→21 (8).
  Lowest recall: 9 (0.01), 10 (0.03), 30 (0.03), 17 (0.03), 29 (0.03).
  Highest recall: 46 (0.46), 2 (0.40), 5 (0.39), 48 (0.33), 45 (0.29).
- **Forest:** 4→2 (17), 3→2 (17), 38→39 (11), 25→12 (10), 6→7 (10).
  Lowest recall: 29 (0.06), 9 (0.07), 40 (0.09), 21 (0.11), 27 (0.11).
  Highest recall: 46 (0.75), 2 (0.68), 11 (0.59), 48 (0.59), 5 (0.54).
- Classes 2, 3 and 4 are confused with each other by both models. Classes 9 and 29 are among the
  hardest for both. Class 46 is the easiest for both.

### Do the models make the same mistakes?

| | Image (4,000 rows) | Sentiment (2,400 rows) |
|---|---|---|
| Both wrong | 2,726 | 703 |
| Tree wrong, forest right | 798 | 539 |
| Forest wrong, tree right | 140 | 288 |

On both tasks the forest fixes more of the tree's mistakes than it adds. Per-row predictions with ids are in
`results/preds/` for comparison with the other team models.

## 7. Training and prediction time

All 16 jobs ran at the same time on one 16-thread laptop, so times are inflated. Compare them
with each other only.

| Model | Image fit | Sentiment fit | Image predict (4,000 rows) |
|---|---|---|---|
| One tree, unlimited depth | 322 to 382 s | 49 to 69 s | 0.04 s |
| Forest, 50 trees, depth 10 | 293 s | 91 s | 0.63 s |
| Forest, 50 trees, depth None | 376 s (log2) | 97 s (log2) | 2.14 s (log2) |
| Forest, 100 trees, depth None | 874 s | 217 s | 3.27 s |

## 8. Chosen settings

- Decision tree: image gini, unlimited depth, min leaf 10. Sentiment gini, depth 5.
- Random forest, both tasks: 100 trees, unlimited depth, sqrt features.
- Use the forest over the tree on both tasks. Whether it is the team's Kaggle model depends on the
  k-NN and neural network validation results.

## 9. Limitations and what to try next

- Settings were changed one at a time, so combinations (for example min leaf 10 inside a forest)
  were not tested.
- Forest accuracy was still rising at 100 trees. More trees would likely help, at a linear cost
  in time.
- Fitting is slow: about 15 minutes for a 100-tree image forest. The split search sorts every
  candidate feature at every node.
- Bug found during the sweep: the original threshold, a float64 midpoint, could round onto the
  upper float32 value, sending both sides of the split to one child. It is fixed in
  `DecisionTreeClassifier._threshold`. All results above are from after the fix.
