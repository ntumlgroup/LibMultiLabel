"""
Hyperparameter Search for Tree-Based Linear Method
=============================================================
.. warning::

    If you are using the one-vs-rest linear methods,
    please check `Hyperparameter Search for One-vs-rest Linear Methods  <../auto_examples/plot_linear_gridsearch_tutorial.html>`_.

To apply tree-based linear methods,
we first convert raw text into numerical BoW features.
During training, the method builds a label tree and trains classifiers.
At inference, the model traverses the tree to make prediction.
Each stage involves multiple hyperparameters that can be tuned to improve model performance.

In this guide, we help users tune the hyperparameters of the tree-based linear method.

.. seealso::

    `Implementation Document <https://www.csie.ntu.edu.tw/~cjlin/papers/libmultilabel/libmultilabel_implementation.pdf>`_:
        For more details about the implementation of tree-based linear methods.

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
}

######################################################################
# Following the suggestions in this `paper <https://drive.google.com/file/d/1kxqNJwg4E_EKjVG-umoG876XKxz3mfm9/view>`__,
# we define 18 configurations to build a simple yet strong baseline.
#
# The search space covers several key parts of the pipeline:
#
# - Text feature extraction: (``ngram_range``, ``stop_words``)
#
#       - We use the vectorizer ``TfidfVectorizer`` from ``sklearn`` to generate features from raw text.
#
# - Label tree structure: (``dmax``, ``K``)
#
#      - Note that ``K`` is the number of clusters and is calculated using the formula from the paper.
#
# - Linear classifier: (``s``, ``c``, ``B``)
#
#       - We combined them into a LIBLINEAR option string. (see *train Usage* in `liblinear <https://github.com/cjlin1/liblinear>`__ README)
#
# - Prediction: (``beam_width``)
#
# .. tip::
#
#     Available hyperparameters (and their defaults) are defined in the class variables of :py:class:`~libmultilabel.linear.GridParameter`.
#
# We implement the entire search process in linear.GridSearch.
# Initialize it with the dataset, the number of cross-validation folds, 
# and the evaluation metrics to monitor.

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
