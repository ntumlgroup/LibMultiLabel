"""
Hyperparameter Search for Tree-Based Linear Method
=============================================================
.. warning::

    If you are using the one-vs-rest linear methods,
    please check `Hyperparameter Search for One-vs-rest Linear Methods  <../auto_examples/plot_linear_gridsearch_tutorial.html>`_.

To apply tree-based linear methods,
we first convert raw text into numerical TF-IDF features.
During training, the method builds a label tree and trains linear classifiers.
At inference, the model traverses the tree and selects
only a few candidate labels at each level to speed up prediction.

To improve model performance, we need to search the hyperparameter space.
Therefore, in this guide, we help users tune the hyperparameters of the tree-based linear method.

.. seealso::

    `Implementation Document <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/libmultilabel_implementation.pdf>`_:
        For more details about the implementation of tree-based linear methods and hyperparameter search.

Here we show an example of tuning a tree-based linear text classifier with the `rcv1 dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)>`_.
Starting with loading the data:
"""

import logging

from libmultilabel import linear

logging.basicConfig(level=logging.INFO)

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
L = len(datasets["train"]["y"])

######################################################################
# Next, we set up the search space.

import numpy as np

dmax = 10
K_factors = [-2, 5]
search_space_dict = {
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "stop_words": ["english"],
    "dmax": [dmax],
    "K": [max(2, int(np.round(np.power(L, 1 / dmax) * np.power(2.0, alpha) + 0.5))) for alpha in K_factors],
    "s": [1],
    "c": [0.5, 1, 2],
    "B": [1],
    "beam_width": [10],
    "prob_A": [3]
}

######################################################################
# Following the suggestions in the `implementation document <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/libmultilabel_implementation.pdf>`_,
# we define 18 configurations to build a simple yet strong baseline.
#
# The search space covers several key parts of the search process:
#
# - Text feature extraction: (``ngram_range``, ``stop_words``)
#
#       - We use the vectorizer ``TfidfVectorizer`` from ``sklearn`` to generate features from raw text.
#
# - Label tree structure: (``dmax``, ``K``)
#
#      - The depth and node degree of the label tree. Note that ``K`` is the number of clusters and is calculated using the formula from the `implementation document <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/libmultilabel_implementation.pdf>`_.
#
# - Linear classifier: (``s``, ``c``, ``B``)
#
#       - We combined them into a LIBLINEAR option string for training linear classifiers. (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README)
#
# - Prediction: (``beam_width``, ``prob_A``)
#
#       - The number of candidates considered and the parameter for the probability estimation function at each level during prediction.
#
# .. tip::
#
#     Available hyperparameters (and their defaults) are defined in the class variables of :py:class:`~libmultilabel.linear.TreeGridParameter`.
#
# In :py:class:`~libmultilabel.linear.TreeGridSearch`, we perform cross-validation for evaluation.
# Specifically, we split the training data into ``n_folds``,
# sequentially using each fold as the validation set while training on the remaining folds.
# Finally, we aggregate the validation outputs from each fold and compute the ``monitor_metrics``.
# Initialization requires the dataset, the number of cross-validation folds, and the evaluation metrics.

n_folds = 3
monitor_metrics = ["P@1", "P@3", "P@5"]
search = linear.TreeGridSearch(datasets, n_folds, monitor_metrics)
cv_scores = search(search_space_dict)

######################################################################
# ``cv_scores`` is a dictionary where keys are :py:class:`~libmultilabel.linear.TreeGridParameter` instances and values are the ``monitor_metrics`` results.
#
# Here we sort the results in descending order by the first metric in ``monitor_metrics``.
# You can retrieve the best parameters after the grid search with the following code:

sorted_cv_scores = sorted(cv_scores.items(), key=lambda x: x[1][monitor_metrics[0]], reverse=True)
print(sorted_cv_scores)

best_params, best_cv_scores = list(sorted_cv_scores)[0]
print(best_params, best_cv_scores)

######################################################################
# The best parameters are::
#
#   {'ngram_range': (1, 3), 'stop_words': 'english', 'dmax': 10, 'K': 88, 's': 1, 'c': 1, 'B': 1, 'beam_width': 10, 'prob_A': 3}
#
# with best cross-validation scores::
#
#   {'P@1': 0.9669, 'P@3': 0.8137, 'P@5': 0.5640}
#
# We can then retrain using the best parameters,
# and use :py:meth:`~libmultilabel.linear.linear_test` and :py:meth:`~libmultilabel.linear.get_metrics` to compute test performance.

from dataclasses import asdict

preprocessor = linear.Preprocessor(tfidf_params=asdict(best_params.tfidf))
transformed_dataset = preprocessor.fit_transform(datasets)

model = linear.train_tree(
    transformed_dataset["train"]["y"],
    transformed_dataset["train"]["x"],
    best_params.linear_options,
    **asdict(best_params.tree),
)

metrics, _, _, _ = linear.linear_test(
    y = transformed_dataset["test"]["y"],
    x = transformed_dataset["test"]["x"],
    model = model,
    metrics = linear.get_metrics(monitor_metrics, num_classes=-1),
    predict_kwargs = asdict(best_params.predict),
)

print(metrics.compute())

######################################################################
# The result of the best parameters will look similar to::
#
#   {'P@1': 0.9554, 'P@3': 0.7968, 'P@5': 0.5576}
