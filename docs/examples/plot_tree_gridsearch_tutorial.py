"""
Hyperparameter Search for Tree-Based Linear Method
=============================================================
This guide helps users to tune the hyperparameters across four fields:
TF-IDF, tree, linear, and predict.

Here we show an example of tuning a tree-based linear text classifier with the `rcv1 dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)>`_.
Starting with loading the data and defining the search space:
"""

from dataclasses import asdict

import numpy as np
import grid

from libmultilabel import linear

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
L = len(datasets["train"]["y"])

######################################################################
# Next, we set up the search space.

n_folds = 3
dmax = 10
K_factors = [-2, 5]
monitor_metrics = ["P@1", "P@3", "P@5"]
search_space_dict = {
    "c": [0.5, 1, 2],
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "stop_words": ["english"],
    "dmax": [dmax],
    "K": [max(2, int(np.round(np.power(L, 1 / dmax) * np.power(2.0, alpha) + 0.5))) for alpha in K_factors],
    "beam_width": [4, 10, 128],
}

######################################################################
# Following the suggestions in this `paper <https://drive.google.com/file/d/1kxqNJwg4E_EKjVG-umoG876XKxz3mfm9/view>`__,
# we define 18 configurations. With three ``beam_width`` values each, this yields 54 total configurations,
# to build a simple yet strong baseline.
# Since the hyperparameter ``K`` must be at least 2, we also handle the edge case where the equation
# produces a value smaller than 2.
#
# We use ``P@1``, ``P@3``, and ``P@5`` for evaluation metrics.
# 
# The vectorizer ``TfidfVectorizer`` from ``sklearn`` is used in the TF-IDF stage to generate features from raw text.
# In the linear stage, the hyperparameters are combined into a LIBLINEAR option string
# (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README).
#
# Available hyperparameters (and their defaults) are defined in the class variables of ``grid.GridParameter``
# (``_tfidf_fields``, ``_tree_fields``, ``_linear_fields``, and ``_predict_fields``).
#
# To search for the best setting, we employ ``grid.GridSearch``.
# ``grid.GridSearch`` is a functor that runs a grid search while avoiding redundant computation.
# Initialize it with the datasets, the number of folds for cross-validation, and the evaluation metrics,
# then call it with the search space to obtain the results.

search = grid.GridSearch(datasets, n_folds, monitor_metrics)
cv_scores = search(search_space_dict)

######################################################################
# The returned scores is a ``dict`` whose keys are ``grid.GridParameter`` instances from the search space,
# and whose values are the scores for ``monitor_metrics``.
#
# Here we sort the results in descending order by the first metric in ``monitor_metrics``.
# You can retrieve the best parameters after the grid search with the following code:

sorted_cv_scores = sorted(cv_scores.items(), key=lambda x: x[1][monitor_metrics[0]], reverse=True)
print(sorted_cv_scores)

best_params, best_cv_scores = list(sorted_cv_scores)[0]
print(best_params, best_cv_scores)

######################################################################
# The best parameter is::
#
#   {'c': 0.5, 'ngram_range': (1, 2), 'stop_words': 'english', 'dmax': 10, 'K': 88, 'beam_width': 4}
#
# We can then retrain using the best parameter,
# and use ``grid.GridSearch.compute_scores`` and ``linear.get_metrics`` to compute test performance.

preprocessor = linear.Preprocessor(tfidf_params=asdict(best_params.tfidf))
transformed_dataset = preprocessor.fit_transform(datasets)

model = linear.train_tree(
    transformed_dataset["train"]["y"],
    transformed_dataset["train"]["x"],
    best_params.linear_options,
    **asdict(best_params.tree),
)

metrics = grid.GridSearch.compute_scores(
    transformed_dataset["test"]["y"],
    transformed_dataset["test"]["x"],
    model,
    best_params,
    {best_params: linear.get_metrics(monitor_metrics, num_classes=-1)}
    )
print(metrics[best_params].compute())

######################################################################
# The result of the best parameters will look similar to::
#
#   {'P@1': 0.8100209275981901, 'P@3': 0.7310622302718446, 'P@5': 0.5290965293466371}
