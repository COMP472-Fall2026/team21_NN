# Decision Tree and Random Forest

All numbers below come from `results/sweep/`, which is produced by `python sweep_trees.py --all`.
Unless we say otherwise, values are validation accuracy with training accuracy in brackets.

## 1. How our models work

### Decision tree

Our decision tree (`decision_tree.py`) is a binary CART tree written in NumPy. Each internal node
tests one feature against a threshold, and rows with `x[f] <= t` go left. To find the best split, we
sort the node's rows by each feature and keep running class counts, which lets us score every
possible threshold in a single pass. The threshold itself is the midpoint between two neighbouring
distinct values. Each split is scored by the weighted impurity (gini or entropy) of the two children,
and the best one is kept as long as it is lower than the node's own impurity.

A node becomes a leaf when it is pure, when it hits `max_depth`, when it has fewer than
`min_samples_split` rows, or when no split leaves at least `min_samples_leaf` rows on each side. A
leaf predicts the most common training label inside it, with ties going to the lowest label.

### Random forest

Our random forest (`random_forest.py`) is built on top of that tree. Each tree trains on a bootstrap
sample, meaning n rows drawn with replacement from the n training rows. At every node a tree only
looks at a random subset of the features, either sqrt(d) or log2(d). For images d = 500, so that is
22 (sqrt) or 8 (log2) features per split; for sentiment d = 768, so 27 or 9. The forest predicts by
majority vote across all its trees, again with ties going to the lowest label.

The reason this helps is that a single deep tree has high variance; small changes in the data can
change it completely. Bootstrap samples and random feature subsets make the trees different from each
other, and averaging many different trees cancels out a lot of that variance while each tree's bias
stays about the same.

Feature normalization does not matter for either model. A split only depends on the order of values
within one feature, and z-scoring does not change that order.

## 2. How we validated

We used the shared stratified 80/20 split (seed 42, ids in `splits/`), giving 16,000 training and
4,000 validation rows for images and 9,600 / 2,400 for sentiment. Every model on the team uses this
same split. All settings were picked by validation accuracy and we never tuned anything on the Kaggle
leaderboard. Training accuracy is measured on a fixed 4,000-row sample of the training split.

We changed one hyperparameter at a time from a base setting of gini, `min_samples_split=2`,
`min_samples_leaf=1` and unlimited depth, with sqrt features for the forest. To save time, each tree
setting is fitted once at unlimited depth and then evaluated cut off at depths 5, 10 and 20, and the
10- and 50-tree forests are the first trees of the 100-tree forest. We checked on both tasks that
this gives exactly the same predictions as fitting each one separately.

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

On sentiment, depth leads straight to overfitting. Going from depth 5 to unlimited depth, training
accuracy rises from 0.535 to 1.000 while validation accuracy drops from 0.483 to 0.445; the deeper
splits are fitting noise in the training rows. Depth 5 was the best setting.

The image task behaves differently. A depth-5 tree has at most 32 leaves, so it cannot even predict all 50
classes, and it clearly underfits. Validation accuracy keeps rising with depth, from 0.055 up to
0.112. The full tree still overfits heavily though, with 1.000 training accuracy against 0.112 on
validation. Regarding whether more depth raises training accuracy but lowers validation accuracy:
it does on sentiment, but on images validation keeps improving all the way to
unlimited depth.

Setting `min_samples_leaf` to 10 was the best regulariser we tried, since it stops leaves from
memorising single rows. At unlimited depth it drops image training accuracy from 1.000 to 0.381 while
validation goes up from 0.112 to 0.119, and sentiment validation goes from 0.445 to 0.481.
`min_samples_split` of 20 has a smaller effect in the same direction, and at depth 5 it changes
almost nothing because very few nodes that shallow have fewer than 20 rows.

The comparison between gini and entropy is mixed. Entropy is better on images at shallow depths (0.072 vs 0.055 at depth 5)
but worse at unlimited depth (0.100 vs 0.112), and gini is better on sentiment at every depth. These
differences are small next to the effect of depth and leaf size.

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

More trees helped on both tasks and had not levelled off by 100. Images went from 0.144 to 0.244 to
0.284 for 10, 50 and 100 trees, and sentiment went from 0.479 to 0.578 to 0.587. Adding trees reduces
variance and does not cause overfitting.

The forest did best with unlimited depth. Each individual tree overfits, but averaging removes most
of that. Capping depth at 10 underfits images badly (0.107 validation, 0.165 training), while on
sentiment depth matters much less (0.572 to 0.578).

For features per split, sqrt beat log2 on both tasks (0.244 vs 0.233 on images and 0.578 vs 0.558 on
sentiment, both at 50 trees). With only 8 or 9 candidate features to choose from, each split is
weaker.

Notably, the best forest has 1.000 training accuracy, yet it validates far better than any single
tree. The gap between training and validation is therefore not the deciding factor on its own; what
matters is that the forest's validation accuracy is higher.

## 5. Tree vs forest

| | Best tree | Best forest |
|---|---|---|
| Image setting | gini, unlimited depth, min leaf 10 | 100 trees, unlimited depth, sqrt |
| Image validation accuracy / macro-F1 | 0.119 / 0.119 | 0.284 / 0.270 |
| Sentiment setting | gini, depth 5 | 100 trees, unlimited depth, sqrt |
| Sentiment validation accuracy / macro-F1 | 0.483 / 0.478 | 0.587 / 0.585 |

The forest more than doubles image accuracy and adds 0.104 on sentiment. It does reduce overfitting
in the sense that matters: its unlimited-depth trees each fit the training data perfectly, but their
combined vote generalizes much better than any one tree. For context, the nearest-centroid baseline
from `common.py --check` scores 0.223 on images and 0.578 on sentiment. The best forest beats it on
images and roughly matches it on sentiment, while a single tree falls well below it on both.

Trees struggle on the image features for a few reasons. There are 50 classes and only 320 training
rows per class, and each split only looks at one feature, while the class information in these
embeddings is probably spread across many features at once. The splits break the data up quickly,
so leaves end up with very few rows.

## 6. Error analysis

### Sentiment per class

| Class | Tree precision / recall | Forest precision / recall |
|---|---|---|
| Negative | 0.514 / 0.562 | 0.610 / 0.695 |
| Neutral | 0.421 / **0.342** | 0.509 / **0.475** |
| Positive | 0.497 / 0.542 | 0.638 / 0.591 |

Neutral is the hardest class for both models and negative is the easiest. A likely explanation is that neutral
sits between the other two, so its examples share features with both sides.

### Image: most-confused classes

The labels are only numbers since the dataset does not give class names.

For the tree, the most common mistakes (true → predicted, count) are 4→2 (11), 3→4 (10), 2→4 (10),
3→2 (8) and 38→21 (8). The lowest recall is on classes 9 (0.01), 10 (0.03), 30 (0.03), 17 (0.03) and
29 (0.03), and the highest is on 46 (0.46), 2 (0.40), 5 (0.39), 48 (0.33) and 45 (0.29).

For the forest, the most common mistakes are 4→2 (17), 3→2 (17), 38→39 (11), 25→12 (10) and 6→7 (10).
The lowest recall is on 29 (0.06), 9 (0.07), 40 (0.09), 21 (0.11) and 27 (0.11), and the highest is
on 46 (0.75), 2 (0.68), 11 (0.59), 48 (0.59) and 5 (0.54).

Both models mix up classes 2, 3 and 4 with each other, both find 9 and 29 among the hardest, and both
find 46 the easiest.

### Do the two models make the same mistakes?

| | Image (4,000 rows) | Sentiment (2,400 rows) |
|---|---|---|
| Both wrong | 2,726 | 703 |
| Tree wrong, forest right | 798 | 539 |
| Forest wrong, tree right | 140 | 288 |

On both tasks the forest fixes a lot more of the tree's mistakes than it introduces. The per-row
predictions (with ids) are saved in `results/preds/` so they can be compared with the other models.

## 7. Training and prediction time

All 16 jobs ran at the same time on one 16-thread laptop, so these times are inflated and are only
useful for comparing against each other.

| Model | Image fit | Sentiment fit | Image predict (4,000 rows) |
|---|---|---|---|
| One tree, unlimited depth | 322 to 382 s | 49 to 69 s | 0.04 s |
| Forest, 50 trees, depth 10 | 293 s | 91 s | 0.63 s |
| Forest, 50 trees, depth None | 376 s (log2) | 97 s (log2) | 2.14 s (log2) |
| Forest, 100 trees, depth None | 874 s | 217 s | 3.27 s |

## 8. Chosen settings

For the decision tree we picked gini with unlimited depth and a minimum leaf size of 10 for images,
and gini with depth 5 for sentiment. For the random forest we used 100 trees, unlimited depth and
sqrt features on both tasks. The forest beats the tree on both tasks, so it is the one we would use out of
the two. For the final Kaggle submission, k-NN and the neural network did better on validation.

## 9. Limitations and what we would try next

Since we changed settings one at a time, we never tested combinations, like a minimum leaf size of 10
inside a forest. Forest accuracy was also still climbing at 100 trees, so more trees would probably
help, at a cost in time that grows linearly. Fitting is already slow, at about 15 minutes for a
100-tree image forest, because the split search sorts every candidate feature at every node.

A bug was also found during the sweep. The original threshold was a float64 midpoint, and it could round
onto the upper float32 value, which sent both sides of a split to the same child. It is fixed in
`DecisionTreeClassifier._threshold`, and every result above comes from after the fix.
