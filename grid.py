from dataclasses import make_dataclass, field, fields, asdict
from typing import Callable

import os
import sys
import itertools
import argparse
import logging

import libmultilabel.linear as linear
from libmultilabel.linear.tree import _build_tree
from libmultilabel.common_utils import timer

import sklearn.preprocessing
import numpy as np
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

    def __repr__(self):
        return str(self.params)

    def __eq__(self, other):
        return all(getattr(self, t) == getattr(other, t) for t in self.param_types)

    def __lt__(self, other):
        # "<" for tuple is automatically lexicographic ordering
        my_values = tuple(getattr(self, t) for t in self.param_types)
        other_values = tuple(getattr(other, t) for t in self.param_types)
        return my_values < other_values

    def __hash__(self):
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
        self.param_metrics = dict()

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

    def get_transformed_dataset(self, dataset, params):
        """
        Get the dataset for the given tf-idf params.

        Args:
            params (GridParameter): The params to build the dataset.

        Returns:
            dict[str, np.matrix]: The keys should be "y" and "x".
        """
        tfidf_params = params.tfidf
        self.no_cache = tfidf_params != self._cached_params.tfidf
        if self.no_cache:
            logging.info(f"TFIDF  - Preprocessing: {tfidf_params}")
            if self.datasets["data_format"] not in {"txt", "dataframe"}:
                logging.info("The TF-IDF parameters are only meaningful for the “txt” and “dataframe” data formats.")
            with __silent__():
                preprocessor = linear.Preprocessor(tfidf_params=asdict(tfidf_params))
                self._cached_params.tfidf = tfidf_params
                self._cached_transformed_dataset = preprocessor.fit_transform(dataset)
        else:
            logging.info(f"TFIDF  - Using cached data: {tfidf_params}")

        return self._cached_transformed_dataset

    def get_tree_root(self, y, x, params):
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

    def get_model(self, y, x, params):
        """
        Get the model for the given params.

        Args:
            y (np.matrix): The labels of the training data.
            x (np.matrix): The features of the training data.
            params (GridParameter): The params to build the model.

        Returns:
            linear.FlatModel | linear.TreeModel: The model for the given params.
        """
        root = self.get_tree_root(y, x, params)

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

    def compute_scores(self, y, x, model, params):
        logging.info(f"Metric - Scoring: {params.predict}\n")

        batch_size = 256
        num_instances = x.shape[0]
        num_batches = math.ceil(num_instances / batch_size)

        if params not in self.param_metrics.keys():
            self.param_metrics[params] = linear.get_metrics(self.monitor_metrics, num_classes=y.shape[1])

        for i in range(num_batches):
            preds = model.predict_values(x[i * batch_size : (i + 1) * batch_size], **asdict(params.predict))
            target = y[i * batch_size : (i + 1) * batch_size].toarray()
            self.param_metrics[params].update(preds, target)

    def __call__(self, search_space_dict: dict[str, list]) -> dict[GridParameter, dict[str, float]]:
        self.param_metrics.clear()

        param_names = search_space_dict.keys()
        self.search_space = sorted(
            [
                GridParameter(dict(zip(param_names, param_values)))
                for param_values in itertools.product(*search_space_dict.values())
            ],
            reverse=True,
        )

        permutation = np.random.permutation(self.num_instances)
        index_per_fold = [
            permutation[
                int(fold * self.num_instances / self.n_folds) : int((fold + 1) * self.num_instances / self.n_folds)
            ]
            for fold in range(self.n_folds)
        ]

        for fold in range(self.n_folds):
            train_idx = np.concatenate(index_per_fold[:fold] + index_per_fold[fold + 1 :])
            valid_idx = index_per_fold[fold]
            fold_dataset = self.get_fold_dataset(train_idx, valid_idx)

            self._cached_params.tfidf = None
            for params in self.search_space:
                logging.info(f"Status - Running fold {fold}, params: {params}")

                transformed_dataset = self.get_transformed_dataset(fold_dataset, params)
                model = self.get_model(transformed_dataset["train"]["y"], transformed_dataset["train"]["x"], params)
                self.compute_scores(transformed_dataset["test"]["y"], transformed_dataset["test"]["x"], model, params)

        return {params: metrics.compute() for params, metrics in self.param_metrics.items()}


@timer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--training_file", help="Path to training data.")
    parser.add_argument("--test_file", help="Path to test data.")
    parser.add_argument(
        "--data_format", type=str, default="txt", help="'svm' for SVM format or 'txt' for LibMultiLabel format."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = linear.load_dataset(
        args.data_format,
        args.training_file,
        args.test_file,
    )

    retrain = True
    n_folds = 3
    monitor_metrics = ["P@1", "P@3", "P@5"]
    search_space_dict = {
        "max_features": [10000, 20000, 100000],
        "K": [10, 50, 100],
        "min_df": [1, 2],
        "prob_A": [2, 3, 4],
        "c": [0.1, 0.2, 1, 10],
    }

    search = GridSearch(dataset, n_folds, monitor_metrics)
    cv_scores = search(search_space_dict)
    sorted_cv_scores = sorted(cv_scores.items(), key=lambda x: x[1][monitor_metrics[0]], reverse=True)
    print(sorted_cv_scores)

    if retrain:
        best_params, best_cv_scores = list(sorted_cv_scores)[0]
        print(best_params, best_cv_scores)

        preprocessor = linear.Preprocessor(tfidf_params=asdict(best_params.tfidf))
        transformed_dataset = preprocessor.fit_transform(dataset)
        model = linear.train_tree(
            transformed_dataset["train"]["y"],
            transformed_dataset["train"]["x"],
            best_params.linear_options,
            **asdict(best_params.tree),
        )


if __name__ == "__main__":
    main()
