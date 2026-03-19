from dataclasses import make_dataclass, field, fields, asdict
from typing import Callable

import os
import sys
import itertools
import logging

import libmultilabel.linear as linear
from libmultilabel.linear.tree import _build_tree

import sklearn.preprocessing
import numpy as np
from scipy import sparse
import math


# suppress inevitable outputs from sparsekmeans and sklearn preprocessors
class __silent__:
    def __init__(self):
        self.stderr = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        os.dup2(self.devnull, 2)
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, type, value, traceback):
        os.dup2(self.stderr, 2)
        os.close(self.devnull)
        os.close(self.stderr)
        sys.stdout.close()
        sys.stdout = self.stdout


class GridParameter:

    _tfidf_fields = [
        ("ngram_range", tuple[int, int], field(default=(1, 1))),
        ("max_features", int, field(default=None)),
        ("min_df", float | int, field(default=1)),
        ("stop_words", str | list, field(default=None)),
        ("strip_accents", str | Callable, field(default=None)),
        ("tokenizer", Callable, field(default=None)),
    ]
    _tree_fields = [
        ("dmax", int, field(default=10)),
        ("K", int, field(default=8)),
    ]
    _linear_fields = [
        ("s", int, field(default=1)),
        ("c", float, field(default=1)),
        ("B", int, field(default=-1)),
    ]
    _predict_fields = [
        ("beam_width", int, field(default=10)),
        ("prob_A", int, field(default=3)),
    ]

    # set frozen=True to make instances hashable.
    # set order=True to enable comparison operations.
    param_types = {
        "tfidf": make_dataclass("TfidfParams", _tfidf_fields, frozen=True, order=True),
        "tree": make_dataclass("TreeParams", _tree_fields, frozen=True, order=True),
        "linear": make_dataclass("LinearParams", _linear_fields, frozen=True, order=True),
        "predict": make_dataclass("PredictParams", _predict_fields, frozen=True, order=True),
    }
    _param_field_names = {
        param_type: {f.name for f in fields(class_name)} for param_type, class_name in param_types.items()
    }

    def __init__(self, params: dict | None = None, fold: int = -1):
        self.params = params or {}

        params_set = set(self.params)
        for param_type, class_name in self.param_types.items():
            field_names = self._param_field_names[param_type]
            filtered_keys = params_set & field_names
            params_set -= field_names

            filtered_params = {k: self.params[k] for k in filtered_keys}
            setattr(self, param_type, class_name(**filtered_params))

    @property
    def linear_options(self):
        options = ""
        for field_name in self._param_field_names["linear"]:
            options += f" -{field_name} {getattr(self.linear, field_name)}"
        return options.strip()

    def __repr__(self):  # provide a readable string representation of the object
        return str(self.params)

    def __eq__(self, other):  # compare instance attributes to define equality.
        return all(getattr(self, t) == getattr(other, t) for t in self.param_types)

    def __lt__(self, other):  # define ordering for sorting.
        # "<" for tuple is automatically lexicographic ordering
        my_values = tuple(getattr(self, t) for t in self.param_types)
        other_values = tuple(getattr(other, t) for t in self.param_types)
        return my_values < other_values

    def __hash__(self):  # make instances hashable for use as dict keys
        return hash(tuple(getattr(self, t) for t in self.param_types))


class GridSearch:
    def __init__(
        self,
        datasets: dict[str, np.matrix],
        n_folds: int = 3,
        monitor_metrics: list[str] = ["P@1", "P@3", "P@5"],
    ):
        self.datasets = datasets
        self.n_folds = n_folds
        self.monitor_metrics = monitor_metrics

        self._cached_params = GridParameter()
        for param_type in self._cached_params.param_types:
            setattr(self._cached_params, param_type, None)
        self._cached_transformed_dataset = None
        self._cached_tree_root = None
        self._cached_fold_data = None
        self._cached_model = None
        self.no_cache = True

        self.num_instances = len(self.datasets["train"]["y"])

    def get_fold_dataset(self, train_idx, valid_idx):
        def take(data, idx):
            if isinstance(data, list):
                return [data[i] for i in idx]
            else:
                return data[idx]

        return {
            "data_format": self.datasets["data_format"],
            "train": {
                "y": take(self.datasets["train"]["y"], train_idx),
                "x": take(self.datasets["train"]["x"], train_idx),
            },
            "test": {
                "y": take(self.datasets["train"]["y"], valid_idx),
                "x": take(self.datasets["train"]["x"], valid_idx),
            },
        }

    def get_transformed_dataset(
        self, dataset: dict[str, dict[str, list[str]]], params: GridParameter
    ) -> dict[str, dict[str, sparse.csr_matrix]]:
        """
        Get and cache the dataset for the given TF-IDF params.
        If we have processed the coming params, return the cached dataset directly without computation.

        Args:
            dataset (dict[str, dict[str, list[str]]]): The training and/or test data, with keys 'train' and 'test' respectively.
                The data has keys 'x' for input features and 'y' for labels.
            params (GridParameter): The params to build the dataset.

        Returns:
            dict[str, dict[str, sparse.csr_matrix]]: The transformed dataset.
        """
        tfidf_params = params.tfidf
        self.no_cache = tfidf_params != self._cached_params.tfidf
        if self.no_cache:
            logging.info(f"TFIDF  - Preprocessing: {tfidf_params}")
            if self.datasets["data_format"] not in {"txt", "dataframe"}:
                logging.info(
                    "Please make sure the data format is 'txt' or 'dataframe'. Otherwise, the TF-IDF parameters have no effect on the dataset."
                )
            with __silent__():
                preprocessor = linear.Preprocessor(tfidf_params=asdict(tfidf_params))
                self._cached_params.tfidf = tfidf_params
                self._cached_transformed_dataset = preprocessor.fit_transform(dataset)
        else:
            logging.info(f"TFIDF  - Using cached data: {tfidf_params}")

        return self._cached_transformed_dataset

    def get_tree(self, y, x, params):
        tree_params = params.tree
        self.no_cache |= tree_params != self._cached_params.tree
        if self.no_cache:
            logging.info(f"Tree   - Preprocessing: {tree_params}")
            with __silent__():
                label_representation = (y.T * x).tocsr()
                label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
                self._cached_params.tree = tree_params
                self._cached_tree_root = _build_tree(
                    label_representation, np.arange(y.shape[1]), 0, **asdict(tree_params)
                )
                self._cached_tree_root.is_root = True
        else:
            logging.info(f"Tree   - Using cached data: {tree_params}")

        return self._cached_tree_root

    def get_model(self, y: sparse.csr_matrix, x: sparse.csr_matrix, params: GridParameter) -> linear.TreeModel:
        """
        Get and cache the model for the given params.
        If we have processed the coming params, return the cached model directly without computation.

        Args:
            y (sparse.csr_matrix): The labels of the training data.
            x (sparse.csr_matrix): The features of the training data.
            params (GridParameter): The params to build the model.

        Returns:
            linear.TreeModel: The model for the given params.
        """
        root = self.get_tree(y, x, params)

        linear_params = params.linear

        if self.no_cache or (linear_params != self._cached_params.linear):
            logging.info(f"Model  - Training: {linear_params}")
            with __silent__():
                self._cached_params.linear = linear_params
                self._cached_model = linear.train_tree(
                    y,
                    x,
                    root=root,
                    options=params.linear_options,
                )
        else:
            logging.info(f"Model  - Using cached data: {linear_params}")

        return self._cached_model

    @staticmethod
    def compute_scores(
        y: sparse.csr_matrix,
        x: sparse.csr_matrix,
        model: linear.TreeModel,
        params: GridParameter,
        param_metrics: dict[str, linear.MetricCollection],
    ) -> dict[str, linear.MetricCollection]:
        """
        Update the metric values in param_metrics with y, x, and model.

        Args:
            y (sparse.csr_matrix): The labels of the test data.
            x (sparse.csr_matrix): The features of the test data.
            model (linear.TreeModel): The trained model.
            params (GridParameter): The params used to compute the scores.
            param_metrics (dict[str, linear.MetricCollection]): The metric values for each GridParameter.

        Returns:
            dict[str, linear.MetricCollection]: The updated metric values.
        """
        logging.info(f"Metric - Scoring: {params.predict}\n")

        batch_size = 256
        num_instances = x.shape[0]
        num_batches = math.ceil(num_instances / batch_size)

        for i in range(num_batches):
            preds = model.predict_values(x[i * batch_size : (i + 1) * batch_size], **asdict(params.predict))
            target = y[i * batch_size : (i + 1) * batch_size].toarray()
            param_metrics[params].update(preds, target)

        return param_metrics

    def __call__(self, search_space_dict: dict[str, list]) -> dict[GridParameter, dict[str, float]]:
        """
        Run the grid search on the search space.

        Args:
            search_space_dict (dict[str, list]): The search space for the grid search.

        Returns:
            dict[GridParameter, dict[str, float]]: The cross-validation scores for each GridParameter in the search space.
        """
        param_names = search_space_dict.keys()

        # To avoid redundant computation (e.g., building the same tree multiple times across different params),
        # we group identical settings in fields and process them continuously.
        # This is implemented by sorting the params in the order of the four fields:
        # TF-IDF, tree, linear, and predict. Finally, cache and reuse the most recent result of each field.
        self.search_space = sorted(
            [
                GridParameter(dict(zip(param_names, param_values)))
                for param_values in itertools.product(*search_space_dict.values())
            ],
            reverse=True,
        )

        # When the number of labels is large, evaluation often focuses on top-ranked
        # metrics (e.g., Precision@K), which do not depend on num_classes.
        # We therefore use -1 as a placeholder.
        self.param_metrics = {
            params: linear.get_metrics(self.monitor_metrics, num_classes=-1) for params in self.search_space
        }

        permutation = np.random.permutation(self.num_instances)
        index_per_fold = []
        for fold in range(self.n_folds):
            index = permutation[
                int(fold * self.num_instances / self.n_folds) : int((fold + 1) * self.num_instances / self.n_folds)
            ]
            index_per_fold.append(index)

        for fold in range(self.n_folds):
            train_idx = np.concatenate(index_per_fold[:fold] + index_per_fold[fold + 1 :])
            valid_idx = index_per_fold[fold]
            fold_dataset = self.get_fold_dataset(train_idx, valid_idx)

            self._cached_params.tfidf = None
            for params in self.search_space:
                logging.info(f"Status - Running fold {fold}, params: {params}")

                transformed_dataset = self.get_transformed_dataset(fold_dataset, params)
                model = self.get_model(transformed_dataset["train"]["y"], transformed_dataset["train"]["x"], params)

                self.param_metrics = self.compute_scores(
                    transformed_dataset["test"]["y"],
                    transformed_dataset["test"]["x"],
                    model,
                    params,
                    self.param_metrics,
                )

        return {params: metrics.compute() for params, metrics in self.param_metrics.items()}
