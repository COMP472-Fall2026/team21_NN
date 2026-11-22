# Error Analysis

All numbers below come from `results/error_analysis/`, which is produced by `python error_analysis.py`.
The figures are in `reports/error_analysis_figures/`.

## 1. Method

The script reads the validation predictions saved by `common.evaluate()` for every run, and keeps the
best run of each model on each task. All four models were evaluated on the same validation rows
(the shared split in `splits/`), which the script checks before comparing them, so the predictions
can be compared row by row.

| Model | Best setting | Image val | Sentiment val |
|---|---|---|---|
| Decision tree | gini; image: unlimited depth, min leaf 10; sentiment: depth 5 | 0.119 | 0.483 |
| Random forest | 100 trees, unlimited depth, sqrt features | 0.284 | 0.587 |
| k-NN | cosine; image: k = 1; sentiment: k = 100, weighted | 0.534 | 0.584 |
| Neural network | image: 1024-256, sentiment: 512; dropout 0.5, lr 3e-4, wd 1e-3 | 0.366 | 0.664 |

## 2. Sentiment

### Is neutral the hardest class?

Yes, for every model. Neutral has the lowest recall and the lowest precision across all four
classifiers.

| Recall | Decision tree | Random forest | k-NN | Neural network |
|---|---|---|---|---|
| Negative | 0.562 | 0.695 | 0.692 | **0.752** |
| Neutral | 0.342 | 0.475 | 0.460 | **0.550** |
| Positive | 0.542 | 0.591 | 0.600 | **0.689** |

| Precision | Decision tree | Random forest | k-NN | Neural network |
|---|---|---|---|---|
| Negative | 0.514 | 0.610 | 0.584 | **0.686** |
| Neutral | 0.421 | 0.509 | 0.541 | **0.595** |
| Positive | 0.497 | 0.638 | 0.622 | **0.704** |

Neutral sits between the other two classes, so its examples share features with both sides and are
pulled in both directions. Every model sends more neutral examples to negative than to positive
(neural network: 204 to negative and 156 to positive; k-NN: 260 and 172). k-NN also shows an overall
bias towards negative, predicting it 948 times out of 2,400 even though the validation set is
balanced at 800 per class.

The neural network is the best model on every class, and its main gain over the others is on the
two clear-cut classes. It confuses negative and positive with each other only 148 times, against
204 for the forest, 254 for k-NN and 339 for the tree. The side-by-side confusion matrices are in
`sentiment_confusion_all_models.png`.

### Do the models make the same mistakes?

| | Count |
|---|---|
| Validation rows | 2,400 |
| All four models correct | 706 |
| All four models wrong | 419 |
| At least one model correct (oracle accuracy) | 0.825 |

Error overlap between pairs of models (Jaccard: rows both get wrong divided by rows either gets wrong):

| | Decision tree | Random forest | k-NN | Neural network |
|---|---|---|---|---|
| Decision tree | 1.000 | 0.459 | 0.465 | 0.374 |
| Random forest | 0.459 | 1.000 | 0.596 | 0.533 |
| k-NN | 0.465 | 0.596 | 1.000 | 0.517 |
| Neural network | 0.374 | 0.533 | 0.517 | 1.000 |

The models share a core of 419 rows (17%) that none of them can classify, but outside of that their
errors only partly overlap. The random forest and k-NN are the most similar pair (0.596), while the
neural network's errors overlap least with the decision tree's (0.374). When two of the stronger
models are wrong on the same row, they usually agree on the wrong label (87 to 88% of the time),
although with only two wrong options this is less informative than it looks.

Since at least one model is correct on 82.5% of rows, against 66.4% for the best single model, the
models carry complementary information that a combined model could potentially use.

### Does the neural network correct the simpler models' mistakes?

Partly. Of the rows each simpler model gets wrong, the neural network gets the following share right:

| Decision tree | Random forest | k-NN |
|---|---|---|
| 0.551 | 0.369 | 0.384 |

It fixes more than half of the decision tree's errors, but only about a third of the forest's and
k-NN's. Most of the rows those two models miss are also difficult for the neural network.

## 3. Image

### Which classes are hardest and easiest?

Classes are ranked by their recall averaged over the four models.

| Class | Decision tree | Random forest | k-NN | Neural network | Mean |
|---|---|---|---|---|---|
| 29 | 0.025 | 0.062 | 0.312 | 0.125 | 0.131 |
| 9 | 0.012 | 0.075 | 0.350 | 0.100 | 0.134 |
| 22 | 0.050 | 0.150 | 0.362 | 0.138 | 0.175 |
| 41 | 0.062 | 0.112 | 0.475 | 0.125 | 0.194 |
| 21 | 0.075 | 0.112 | 0.475 | 0.150 | 0.203 |
| ... | | | | | |
| 12 | 0.238 | 0.475 | 0.712 | 0.538 | 0.491 |
| 5 | 0.388 | 0.538 | 0.712 | 0.588 | 0.556 |
| 48 | 0.325 | 0.588 | 0.762 | 0.650 | 0.581 |
| 2 | 0.400 | 0.675 | 0.712 | 0.550 | 0.584 |
| 46 | 0.462 | 0.750 | 0.888 | 0.788 | 0.722 |

The full table is in `results/error_analysis/image_recall_by_class.csv`, and
`image_recall_by_class.png` shows all 50 classes for every model.

The difficulty ranking is very consistent across models. Classes 29 and 9 are the two hardest and
class 46 is the easiest for all four. This means the difficulty comes from the data itself rather
than from any particular algorithm: these classes overlap with others in the feature space.

### Which model does best on the difficult classes?

Accuracy on the 10 hardest classes (29, 9, 22, 41, 21, 10, 30, 23, 1, 42):

| Decision tree | Random forest | k-NN | Neural network |
|---|---|---|---|
| 0.046 | 0.125 | **0.404** | 0.191 |

k-NN is by far the best on the difficult classes, and its lead is larger here than overall (0.404
against 0.191 for the neural network, compared with 0.534 against 0.366 on all classes). A 1-NN
classifier only needs a single close training example of the right class, so it can still recognise
rows from classes that are spread out or overlap with others. The neural network and the trees have
to learn one decision region per class, which fails when a class is not compact.

### Which classes are confused most often?

Class pairs ranked by how many models have them among their 15 most-confused pairs (counting both
directions):

| Classes | Models | Total errors |
|---|---|---|
| 2 and 4 | 4 | 71 |
| 2 and 3 | 4 | 59 |
| 6 and 7 | 4 | 50 |
| 37 and 39 | 4 | 44 |
| 1 and 3 | 3 | 41 |
| 38 and 39 | 3 | 41 |
| 12 and 25 | 3 | 31 |
| 25 and 27 | 3 | 30 |

Four pairs are among the top confusions of every model, and the group 2 / 3 / 4 accounts for the
largest share. Most of these pairs have nearby label numbers (2/3/4, 6/7, 37/38/39), which suggests
the dataset numbers similar object types next to each other, although this cannot be confirmed
without class names. Several of the same pairs (12 and 25, 37 and 39, 3 and 4) are also the classes
that k-Means groups together without labels (see `clustering_summary.md`), which confirms that they
sit close together in the feature space.

### Do the models make the same mistakes?

| | Count |
|---|---|
| Validation rows | 4,000 |
| All four models correct | 252 |
| All four models wrong | 1,291 |
| At least one model correct (oracle accuracy) | 0.677 |

| Jaccard | Decision tree | Random forest | k-NN | Neural network |
|---|---|---|---|---|
| Decision tree | 1.000 | 0.744 | 0.479 | 0.651 |
| Random forest | 0.744 | 1.000 | 0.520 | 0.671 |
| k-NN | 0.479 | 0.520 | 1.000 | 0.501 |
| Neural network | 0.651 | 0.671 | 0.501 | 1.000 |

On images the models fail on many of the same rows: about a third of the validation set (1,291 rows)
is misclassified by all four. The tree, forest and neural network have heavily overlapping errors
(0.65 to 0.74), which fits with all three learning class regions from the training data. k-NN's
errors overlap the least with every other model (about 0.5), so it fails in a different way.

However, when two models are wrong on the same row they rarely predict the same wrong class (4 to 16%
of the time). They agree that a row is hard but not on what it looks like, which is consistent with
these rows lying in regions where many classes overlap.

### Does the neural network correct the simpler models' mistakes?

| Decision tree | Random forest | k-NN |
|---|---|---|
| 0.322 | 0.243 | 0.212 |

On images the neural network fixes only about a quarter of the forest's errors and a fifth of
k-NN's. Its overall accuracy is lower than k-NN's, and the rows k-NN misses are mostly rows the
neural network misses as well. The neural network also has some confusions of its own that are less
common for the other models, such as 1 → 3 (15 errors) and 11 → 28 (10 errors).

## 4. Conclusions

On sentiment, neutral is clearly the hardest class for every model, and it is mostly confused with
negative. The neural network is the strongest model on all three classes, and its advantage comes
mainly from separating negative and positive more cleanly.

On images, the hardest and easiest classes are the same for all four models, and the most confused
pairs (2 / 3 / 4, 6 / 7, 37 / 38 / 39) are shared by every model and also merged by k-Means. This
indicates that these errors come from overlap in the features rather than from a weakness of a
single algorithm. k-NN handles these difficult classes much better than the other models, which
explains most of its overall lead on the image task.

On both tasks the oracle accuracy (at least one model correct) is well above the best single model:
0.825 against 0.664 on sentiment and 0.677 against 0.534 on images. k-NN's errors overlap least with
the other models on images, so combining k-NN with the neural network is the most promising
direction for future work.
