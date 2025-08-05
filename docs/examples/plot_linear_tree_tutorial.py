"""
Handling Data with Many Labels Using Linear Methods
====================================================

For datasets with a very large number of labels, the training time of the standard ``train_1vsrest`` method can be prohibitively long. LibMultiLabel offers tree-based methods like ``train_tree`` and ``train_ensemble_tree`` to vastly improve training time in such scenarios.


We will use the `EUR-Lex dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#EUR-Lex>`_, which contains 3,956 labels. The data is assumed to be downloaded under the directory ``data/eur-lex``.
"""

import math
import libmultilabel.linear as linear
import time

# Load and preprocess the dataset
datasets = linear.load_dataset("txt", "data/eurlex/train.txt", "data/eurlex/test.txt")
preprocessor = linear.Preprocessor()
datasets = preprocessor.fit_transform(datasets)


######################################################################
# Standard Training and Prediction
# --------------------------------
#
# Users can use the following command to easily apply the ``train_tree`` method.
#
# .. code-block:: bash
#
#     $ python3 main.py --training_file data/eur-lex/train.txt \\
#                       --test_file data/eur-lex/test.txt \\
#                       --linear \\
#                       --linear_technique tree
#
# Besides CLI usage, users can also use API to apply ``train_tree`` method.
# Below is an example.

training_start = time.time()
# the standard one-vs-rest method for multi-label problems
ovr_model = linear.train_1vsrest(datasets["train"]["y"], datasets["train"]["x"])
training_end = time.time()
print("Training time of one-versus-rest: {:10.2f}".format(training_end - training_start))

training_start = time.time()
# the train_tree method for fast training on data with many labels
tree_model = linear.train_tree(datasets["train"]["y"], datasets["train"]["x"])
training_end = time.time()
print("Training time of tree-based: {:10.2f}".format(training_end - training_start))

######################################################################
# On a machine with an AMD-7950X CPU,
# the ``train_1vsrest`` function took `578.30` seconds,
# while the ``train_tree`` function only took `144.37` seconds.
#
# .. note::
#
#   The ``train_tree`` function in this tutorial is based on the work of :cite:t:`SK20a`.
#
# ``train_tree`` achieves this speedup by approximating ``train_1vsrest``. To check whether the approximation
# performs well, we'll compute some metrics on the test set.

ovr_preds = linear.predict_values(ovr_model, datasets["test"]["x"])
tree_preds = linear.predict_values(tree_model, datasets["test"]["x"])

target = datasets["test"]["y"].toarray()

ovr_score = linear.compute_metrics(ovr_preds, target, ["P@1", "P@3", "P@5"])
print("Score of 1vsrest:", ovr_score)

tree_score = linear.compute_metrics(tree_preds, target, ["P@1", "P@3", "P@5"])
print("Score of tree:", tree_score)

######################################################################
#  :math:`P@K`, a ranking-based criterion, is a metric often used for data with a large amount of labels.
#
# .. code-block::
#
#   Score of 1vsrest: {'P@1': 0.833117723156533, 'P@3': 0.6988357050452781, 'P@5': 0.585666235446313}
#   Score of tree: {'P@1': 0.8217335058214748, 'P@3': 0.692539887882708, 'P@5': 0.578835705045278}
#
# For this data set, ``train_tree`` gives a slightly lower :math:`P@K`, but has a significantly faster training time.
# Typcially, the speedup of ``train_tree`` over ``train_1vsrest`` increases with the amount of labels.
#
# For even larger data sets, we may not be able to store the entire ``preds`` and ``target`` in memory at once.
# In this case, the metrics can be computed in batches.


def metrics_in_batches(model):
    batch_size = 256
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])

    for i in range(num_batches):
        preds = linear.predict_values(model, datasets["test"]["x"][i * batch_size : (i + 1) * batch_size])
        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        metrics.update(preds, target)

    return metrics.compute()


print("Score of 1vsrest:", metrics_in_batches(ovr_model))
print("Score of tree:", metrics_in_batches(tree_model))


######################################################################
# Ensemble of Tree Models
# -----------------------
#
# While the ``train_tree`` method offers a significant speedup, its accuracy can sometimes be slightly lower than the standard one-vs-rest approach.
# The ``train_ensemble_tree`` method can help bridge this gap by training multiple tree models and averaging their predictions.
#
# Users can use the following command to easily apply the ``train_ensemble_tree`` method.
# The number of trees in the ensemble can be controlled with the ``--tree_ensemble_models`` argument.
#
# .. code-block:: bash
#
#     $ python3 main.py --training_file data/eur-lex/train.txt \\
#                       --test_file data/eur-lex/test.txt \\
#                       --linear \\
#                       --linear_technique tree \\
#                       --tree_ensemble_models 3
#
# This command trains an ensemble of 3 tree models. If ``--tree_ensemble_models`` is not specified, it defaults to 1 (a single tree).
#
# Besides CLI usage, users can also use the API to apply the ``train_ensemble_tree`` method.
# Below is an example.

# We have already trained a single tree model as a baseline.
# Now, let's train an ensemble of 3 tree models.
training_start = time.time()
ensemble_model = linear.train_ensemble_tree(
    datasets["train"]["y"], datasets["train"]["x"], n_trees=3
)
training_end = time.time()
print("Training time of ensemble tree: {:10.2f}".format(training_end - training_start))

######################################################################
# On a machine with an AMD-7950X CPU,
# the ``train_ensemble_tree`` function with 3 trees took `421.15` seconds,
# while the single tree took `144.37` seconds.
# As expected, training an ensemble takes longer, roughly proportional to the number of trees.
#
# Now, let's see if this additional training time translates to better performance.
# We'll compute the same P@K metrics on the test set for both the single tree and the ensemble model.

# `tree_preds` and `target` are already computed in the previous section.
ensemble_preds = linear.predict_values(ensemble_model, datasets["test"]["x"])

# `tree_score` is already computed.
print("Score of single tree:", tree_score)

ensemble_score = linear.compute_metrics(ensemble_preds, target, ["P@1", "P@3", "P@5"])
print("Score of ensemble tree:", ensemble_score)

######################################################################
# While training an ensemble takes longer, it often leads to better predictive performance.
# The following table shows a comparison between a single tree and ensembles
# of 3, 10, and 15 trees on several benchmark datasets.
#
# .. table:: Benchmark Results for Single and Ensemble Tree Models (P@K in %)
#
#    +---------------+-----------------+-------+-------+-------+
#    | Dataset       | Model           | P@1   | P@3   | P@5   |
#    +===============+=================+=======+=======+=======+
#    | EURLex-4k     | Single Tree     | 82.35 | 68.98 | 57.62 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-3      | 82.38 | 69.28 | 58.01 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-10     | 82.74 | 69.66 | 58.39 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-15     | 82.61 | 69.56 | 58.29 |
#    +---------------+-----------------+-------+-------+-------+
#    | EURLex-57k    | Single Tree     | 90.77 | 80.81 | 67.82 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-3      | 91.02 | 81.06 | 68.26 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-10     | 91.23 | 81.22 | 68.34 |
#    |               +-----------------+-------+-------+-------+
#    |               | Ensemble-15     | 91.25 | 81.31 | 68.34 |
#    +---------------+-----------------+-------+-------+-------+

