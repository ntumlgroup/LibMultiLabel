"""
Hyperparameter Search for Tree-Based Linear Method
=============================================================
This guide helps users to tune the hyperparameters for tree-based linear method.
If you are using the one-vs-rest linear methods,
please check `Hyperparameter Search for One-vs-rest Linear Methods  <../auto_examples/plot_linear_gridsearch_tutorial.html>`_.

The hyperparameters for the tree-based linear method cover several aspects.
First, during feature generation, we tune the TF-IDF parameters to vectorize the raw text data in different ways.
Next, when constructing the label tree, we can adjust its depth and width.
Once the data and label tree are prepared, we can configure the LIBLINEAR options
to select the solver, tune the cost, and enable/disable the bias.
Finally, in the tree-based linear method, prediction is not as intuitive as in the one-vs-rest linear methods.
We need to consider the beam width used to traverse the label tree,
as well as the function that estimates the probability for each edge.

Here we show an example of tuning a tree-based linear text classifier with the `rcv1 dataset <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#rcv1v2%20(topics;%20full%20sets)>`_.
Starting with loading the data and defining the search space:
"""

import logging

from libmultilabel import linear

logging.basicConfig(level=logging.INFO)

datasets = linear.load_dataset("txt", "data/rcv1/train.txt", "data/rcv1/test.txt")
L = len(datasets["train"]["y"])

######################################################################
# we enable logging to provide more information during the search process.
#
# Next, we set up the search space.

import numpy as np

dmax = 10
K_factors = [-2, 5]
search_space_dict = {
    "s": [1],
    "c": [0.5, 1, 2],
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "stop_words": ["english"],
    "dmax": [dmax],
    "K": [max(2, int(np.round(np.power(L, 1 / dmax) * np.power(2.0, alpha) + 0.5))) for alpha in K_factors],
    "beam_width": [10],
}

######################################################################
# Following the suggestions in this `paper <https://drive.google.com/file/d/1kxqNJwg4E_EKjVG-umoG876XKxz3mfm9/view>`__,
# we define 18 configurations to build a simple yet strong baseline.
# Since the hyperparameter ``K`` must be at least 2, we also handle the edge case where the equation
# produces a value smaller than 2.
#
# We use ``P@1``, ``P@3``, and ``P@5`` for evaluation metrics.
#
# The vectorizer ``TfidfVectorizer`` from ``sklearn`` is used in the TF-IDF stage to generate features from raw text.
# In the linear stage, the hyperparameters are combined into a LIBLINEAR option string
# (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README).
#
# Available hyperparameters (and their defaults) are defined in the class variables of ``linear.GridParameter``.
#
# In this example, ``s`` and ``c`` are for solver and cost in the LIBLINEAR,
# which influence the linear classifiers on each edge in the tree model.
# Next, ``ngram_range`` and ``stop_words`` are parameters of ``TfidfVectorizer``.
# ``ngram_range`` control the different n-grams to be extracted,
# and ``stop_words`` define the very common words to remove from text.
# We observe that enabling stop words for the data language(s) is an empirically effective approach.
# For ``dmax`` and ``K``, they are related to the construction of the label tree.
# We use the formula in the paper to calculate ``K`` (maximum degree of nodes in the tree),
# which considers ``dmax`` (maximum depth of the tree) and ``L`` (number of instances)
# along with a scaling factor ``alpha``.
# Finally, we have ``beam_width`` (number of candidates considered during beam search) for prediction.
#
# To search for the best setting, we employ ``linear.GridSearch``.
# First, we should set ``n_folds`` to choose the number of folds in cross-validation
# and ``monitor_metrics`` to select the desired metrics for the grid search process.
# We implement the whole search process in ``linear.GridSearch``. You should
# initialize it with the datasets, the number of folds for cross-validation,
# and the metrics you want to monitor during the grid search process.
# You can start the search with the search space, and after the search ends,
# it returns the scores for ``monitor_metrics`` for each set of ``params`` in the search space.

n_folds = 3
monitor_metrics = ["P@1", "P@3", "P@5"]
search = linear.GridSearch(datasets, n_folds, monitor_metrics)
cv_scores = search(search_space_dict)

######################################################################
# The returned scores are a ``dict`` whose keys are ``linear.GridParameter`` instances from the search space,
# and whose values are the scores for ``monitor_metrics``.
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
#   {'s': 1, 'c': 0.5, 'ngram_range': (1, 2), 'stop_words': 'english', 'dmax': 10, 'K': 88, 'beam_width': 10}
#
# We can then retrain using the best parameters,
# and use ``linear.GridSearch.compute_scores`` and ``linear.get_metrics`` to compute test performance.

from dataclasses import asdict

preprocessor = linear.Preprocessor(tfidf_params=asdict(best_params.tfidf))
transformed_dataset = preprocessor.fit_transform(datasets)

model = linear.train_tree(
    transformed_dataset["train"]["y"],
    transformed_dataset["train"]["x"],
    best_params.linear_options,
    **asdict(best_params.tree),
)

metrics = linear.GridSearch.compute_scores(
    transformed_dataset["test"]["y"],
    transformed_dataset["test"]["x"],
    model,
    best_params,
    {best_params: linear.get_metrics(monitor_metrics, num_classes=-1)},
)
print(metrics[best_params].compute())

######################################################################
# The result of the best parameters will look similar to::
#
#   {'P@1': 0.8100209275981901, 'P@3': 0.7310622302718446, 'P@5': 0.5290965293466371}
