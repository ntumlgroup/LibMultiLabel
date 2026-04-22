from __future__ import annotations

import os
import sys
import math
import itertools
import logging
import pathlib
import pickle
import re
from math import ceil
from tqdm import tqdm
from typing import Any, Callable
from dataclasses import make_dataclass, field, fields, asdict

import numpy as np
import scipy.sparse as sparse
import sklearn.base
import sklearn.model_selection
import sklearn.pipeline
import sklearn.utils
import sklearn.preprocessing

import libmultilabel.linear as linear

from .preprocessor import Preprocessor
from .tree import _build_tree

__all__ = ["save_pipeline", "load_pipeline", "MultiLabelEstimator", "GridSearchCV", "TreeGridParameter", "TreeGridSearch"]


LINEAR_TECHNIQUES = {
    "1vsrest": linear.train_1vsrest,
    "thresholding": linear.train_thresholding,
    "cost_sensitive": linear.train_cost_sensitive,
    "cost_sensitive_micro": linear.train_cost_sensitive_micro,
    "binary_and_multiclass": linear.train_binary_and_multiclass,
    "tree": linear.train_tree,
}


def save_pipeline(checkpoint_dir: str, preprocessor: Preprocessor, model):
    """Save preprocessor and model to checkpoint_dir/linear_pipline.pickle.

    Args:
        checkpoint_dir (str): The directory to save to.
        preprocessor (Preprocessor): A Preprocessor.
        model: A model returned from one of the training functions.
    """
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "linear_pipeline.pickle")

    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "model": model,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_pipeline(checkpoint_path: str) -> tuple[Preprocessor, Any]:
    """Load preprocessor and model from checkpoint_path.

    Args:
        checkpoint_path (str): The path to a previously saved pipeline.

    Returns:
        tuple[Preprocessor, Any]: A tuple of the preprocessor and model.
    """
    with open(checkpoint_path, "rb") as f:
        pipeline = pickle.load(f)
    return (pipeline["preprocessor"], pipeline["model"])


class MultiLabelEstimator(sklearn.base.BaseEstimator):
    """Customized sklearn estimator for the multi-label classifier.

    Args:
        options (str, optional): The option string passed to liblinear. Defaults to ''.
        linear_technique (str, optional): Multi-label technique defined in `utils.LINEAR_TECHNIQUES`.
            Defaults to '1vsrest'.
        scoring_metric (str, optional): The scoring metric. Defaults to 'P@1'.
    """

    def __init__(self, options: str = "", linear_technique: str = "1vsrest", scoring_metric: str = "P@1", multiclass: bool = False):
        super().__init__()
        self.options = options
        self.linear_technique = linear_technique
        self.scoring_metric = scoring_metric
        self._is_fitted = False
        self.multiclass = multiclass

    def fit(self, X: sparse.csr_matrix, y: sparse.csr_matrix):
        X, y = sklearn.utils.validation.check_X_y(X, y, accept_sparse=True, multi_output=True)
        self._is_fitted = True
        self.model = LINEAR_TECHNIQUES[self.linear_technique](y, X, options=self.options)
        return self

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        sklearn.utils.validation.check_is_fitted(self, attributes=["_is_fitted"])
        preds = linear.predict_values(self.model, X)
        return preds

    def score(self, X: sparse.csr_matrix, y: sparse.csr_matrix) -> float:
        metrics = linear.get_metrics(
            monitor_metrics=[self.scoring_metric],
            num_classes=y.shape[1],
            multiclass=self.multiclass
        )
        preds = self.predict(X)
        metrics.update(preds, y.toarray())
        metric_dict = metrics.compute()
        return metric_dict[self.scoring_metric]


class GridSearchCV(sklearn.model_selection.GridSearchCV):
    """A customized `sklearn.model_selection.GridSearchCV`` class for Liblinear.
    The usage is similar to sklearn's, except that the parameter ``scoring`` is unavailable. Instead, specify ``scoring_metric`` in ``MultiLabelEstimator`` in the Pipeline.

    Args:
        estimator (estimator object): An estimator for grid search.
        param_grid (dict): Search space for a grid search containing a dictionary of
            parameters and their corresponding list of candidate values.
        n_jobs (int, optional): Number of CPU cores run in parallel. Defaults to None.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(self, estimator, param_grid: dict, n_jobs=None, **kwargs):
        if n_jobs is not None and n_jobs > 1:
            param_grid = self._set_singlecore_options(estimator, param_grid)
        if "scoring" in kwargs.keys():
            raise ValueError(
                "Please specify the validation metric with `MultiLabelEstimator.scoring_metric` in the Pipeline instead of using the parameter `scoring`."
            )

        super().__init__(estimator=estimator, n_jobs=n_jobs, param_grid=param_grid, **kwargs)

    def _set_singlecore_options(self, estimator, param_grid: dict):
        """Set liblinear options to `-m 1`. The grid search option `n_jobs`
        runs multiple processes in parallel. Using multithreaded liblinear
        in conjunction with grid search oversubscribes the CPU and deteriorates
        the performance significantly.
        """
        params = estimator.get_params()
        for name, transform in params.items():
            if isinstance(transform, MultiLabelEstimator):
                regex = r"-m \d+"
                key = f"{name}__options"
                param_grid[key] = [f"{re.sub(regex, '', v)} -m 1" for v in param_grid[key]]
        return param_grid


def linear_test(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    model: linear.FlatModel | linear.TreeModel | linear.EnsembleTreeModel,
    eval_batch_size: int = 256,
    monitor_metrics: list[str] | None = None,
    metrics: linear.MetricCollection | None = None,
    predict_kwargs: dict | None = None,
    beam_width: int | None = None,
    prob_A: float | None = None,
    label_mapping: np.ndarray | None = None,
    save_k_predictions: int | None = None,
    save_positive_predictions: bool | None = False,
) -> tuple[linear.MetricCollection, dict, list | np.ndarray, list | np.ndarray]:
    """
    Evaluate a linear model on test data with batched prediction and compute metrics.

    Args:
        y (scipy.sparse.csr_matrix): The labels of the test data with dimensions number of instances * number of classes.
        x (scipy.sparse.csr_matrix): The features of the test data with dimensions number of instances * number of features.
        model (linear.FlatModel | linear.TreeModel | linear.EnsembleTreeModel): The trained model.
        eval_batch_size (int): Batch size used during evaluation.
        monitor_metrics (list[str], optional): The evaluation metrics to monitor.
        metrics (linear.MetricCollection, optional): The metric values.
        predict_kwargs (dict, optional): Extra parameters passed to model.predict_values.
        beam_width (int, optional): Number of candidates considered during beam search.
        prob_A (float, optional):
            The hyperparameter used in the probability estimation function for
            binary classification: sigmoid(prob_A * decision_value_matrix).
        label_mapping (np.ndarray, optional): A np.ndarray of class labels that maps each index (from 0 to ``num_class-1``) to its label.
        save_k_predictions (int, optional): Determine how many classes per instance should be predicted.
        save_positive_predictions (bool, optional): If True, return all labels and scores with positive decision value.

    Returns:
        tuple:
            metrics (linear.MetricCollection): The updated metric values.
            metric_dict (dict[str, float]): The computed metric results.
            labels (list or np.ndarray): Labels and scores of top k predictions from decision values if save_k_predictions is set.
            scores (list or np.ndarray): Labels and scores with positive decision value if save_positive_predictions is True.
    """
    if monitor_metrics is None:
        monitor_metrics = ["P@1", "P@3", "P@5"]
    if metrics is None:
        metrics = linear.get_metrics(monitor_metrics, y.shape[1], multiclass=model.multiclass)
    num_instance = x.shape[0]
    k = save_k_predictions
    if k is not None and k > 0:
        labels = np.zeros((num_instance, k), dtype=object)
        scores = np.zeros((num_instance, k), dtype="d")
    else:
        labels = []
        scores = []

    if predict_kwargs is None and isinstance(model, (linear.TreeModel, linear.EnsembleTreeModel)):
        predict_kwargs = {}
        if beam_width is not None:
            predict_kwargs["beam_width"] = beam_width
        if prob_A is not None:
            predict_kwargs["prob_A"] = prob_A

    for i in tqdm(range(ceil(num_instance / eval_batch_size))):
        slice = np.s_[i * eval_batch_size : (i + 1) * eval_batch_size]
        preds = model.predict_values(x[slice], **predict_kwargs)
        target = y[slice].toarray()
        metrics.update(preds, target)
        if k is not None and label_mapping is not None and k > 0:
            labels[slice], scores[slice] = linear.get_topk_labels(preds, label_mapping, save_k_predictions)
        elif save_positive_predictions and label_mapping is not None:
            res = linear.get_positive_labels(preds, label_mapping)
            labels.append(res[0])
            scores.append(res[1])
    metric_dict = metrics.compute()
    return metrics, metric_dict, labels, scores


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


class TreeGridParameter:
    """A tree-based linear method hyperparameter class for TreeGridSearch.
    Transform the parameter dict into dataclass instances.
    Parameters not in the dict will be set to default values.

    Args:
        params (dict, optional): The keys are the parameter names, and the valus are the parameter values.
    """

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

    def __init__(self, params: dict | None = None):
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


class TreeGridSearch:
    """Grid search the search space and find the best parameters for the tree-based linear method,
    according to the monitored metrics.

    Args:
        datasets (dict[str, dict[str, list[str]]]): The training and/or test data, with keys 'train' and 'test' respectively.
                The data has keys 'x' for input features and 'y' for labels.
        n_folds (int, optional): The number of cross-validation folds.
        monitor_metrics (list[str], optional): The evaluation metrics to monitor.
    """

    def __init__(
        self,
        datasets: dict[str, dict[str, list[str]]],
        n_folds: int = 3,
        monitor_metrics: list[str] = ["P@1", "P@3", "P@5"],
    ):
        self.datasets = datasets
        self.n_folds = n_folds
        self.monitor_metrics = monitor_metrics

        self._cached_params = TreeGridParameter()
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
        self, dataset: dict[str, dict[str, list[str]]], params: TreeGridParameter
    ) -> dict[str, dict[str, sparse.csr_matrix]]:
        """
        Get and cache the dataset for the given TF-IDF params.
        If we have processed the coming params, return the cached dataset directly without computation.

        Args:
            dataset (dict[str, dict[str, list[str]]]): The training and/or test data, with keys 'train' and 'test' respectively.
                The data has keys 'x' for input features and 'y' for labels.
            params (TreeGridParameter): The params to build the dataset.

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

    def get_model(self, y: sparse.csr_matrix, x: sparse.csr_matrix, params: TreeGridParameter) -> linear.TreeModel:
        """
        Get and cache the model for the given params.
        If we have processed the coming params, return the cached model directly without computation.

        Args:
            y (sparse.csr_matrix): The labels of the training data.
            x (sparse.csr_matrix): The features of the training data.
            params (TreeGridParameter): The params to build the model.

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

    def __call__(self, search_space_dict: dict[str, list]) -> dict[TreeGridParameter, dict[str, float]]:
        """
        Run the grid search on the search space.

        Args:
            search_space_dict (dict[str, list]): The search space for the grid search.

        Returns:
            dict[TreeGridParameter, dict[str, float]]: The cross-validation scores for each TreeGridParameter in the search space.
        """
        param_names = search_space_dict.keys()

        # To avoid redundant computation (e.g., building the same tree multiple times across different params),
        # we group identical settings in fields and process them continuously.
        # This is implemented by sorting the params in the order of the four fields:
        # TF-IDF, tree, linear, and predict. Finally, cache and reuse the most recent result of each field.
        self.search_space = sorted(
            [
                TreeGridParameter(dict(zip(param_names, param_values)))
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

                logging.info(f"Metric - Scoring: {params.predict}\n")
                self.param_metrics[params], _, _, _ = linear_test(
                    y = transformed_dataset["test"]["y"],
                    x = transformed_dataset["test"]["x"],
                    model = model,
                    metrics = self.param_metrics[params],
                    predict_kwargs = asdict(params.predict),
                )

        return {params: metrics.compute() for params, metrics in self.param_metrics.items()}
