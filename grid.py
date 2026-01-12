from dataclasses import make_dataclass, field, fields, asdict
from typing import Callable

import os
import sys
import logging

import libmultilabel.linear as linear
from libmultilabel.linear.tree import _build_tree

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
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, type, value, traceback):
        os.dup2(self.stderr, 2)
        os.close(self.devnull)
        os.close(self.stderr)
        sys.stdout.close()
        sys.stdout = self.stdout


class GridParameter:

    _tfidf_fields = [
        ('ngram_range', tuple[int, int], field(default=(1, 1))),
        ('max_features', int, field(default=None)),
        ('min_df', float | int, field(default=1)),
        ('stop_words', str | list, field(default=None)),
        ('strip_accents', str | Callable, field(default=None)),
        ('tokenizer', Callable, field(default=None)),
        ]
    _tree_fields = [
        ('dmax', int, field(default=10)),
        ('K', int, field(default=8)),
        ]
    _linear_fields = [
        ('s', int, field(default=1)),
        ('c', float, field(default=1)),
        ('B', int, field(default=-1)),
        ('alpha', float, field(default=1)),
        ]
    _predict_fields = [
        ('beam_width', int, field(default=10)),
        ('A', int, field(default=1)),
        ]

    _param_types = {
        'tfidf': make_dataclass('TfidfParams', _tfidf_fields, frozen=True, order=True),
        'tree': make_dataclass('TreeParams', _tree_fields, frozen=True, order=True),
        'linear': make_dataclass('LinearParams', _linear_fields, frozen=True, order=True),
        'predict': make_dataclass('PredictParams', _predict_fields, frozen=True, order=True),
    }

    def __init__(self, params: dict, fold: int = -1):
        self.params = params
        for param_type, class_name in self._param_types.items():
            field_names = {f.name for f in fields(class_name)}
            _params = {k: v for k, v in self.params.items() if k in field_names}
            setattr(self, param_type, class_name(**_params))
        self.param_types = dict(self._param_types, fold=-1)
        self.fold = fold

    @property
    def linear_options(self):
        options = ''
        for f in fields(self.linear):
            options += f" -{f.name} {getattr(self.linear, f.name)}"
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
        metrics: list[str] = ["P@1", "P@3", "P@5"],
    ):
        self.datasets = datasets
        self.n_folds = n_folds
        self.metrics = metrics

        self._cached_params = GridParameter()
        for param_type in self._cached_params.param_types:
            self._cached_params[param_type] = None
        self._cached_tfidf_data = None
        self._cached_tree_root = None
        self._cached_fold_data = None
        self._cached_model = None

        self.num_instances = len(self.datasets["train"]["y"])

    def init_tfidf_cache(self, datasets, params):
        self._cached_tfidf_params = params.tfidf
        self._cached_tfidf_data = datasets

    def sort_search_space(self):
        self.search_space.sort()

    def build_fold_idx(self):
        permutation = np.random.permutation(self.num_instances)
        index_per_fold = [
            permutation[int(fold * self.num_instances / self.n_folds):int((fold+1) * self.num_instances / self.n_folds)]
            for fold in range(self.n_folds)
        ]

        self.fold_idx = {
            fold: {
                'train': np.concatenate(index_per_fold[:fold] + index_per_fold[fold+1:]),
                'valid': index_per_fold[fold]
                } for fold in range(self.n_folds)
            }

    def get_dataset(self, params):
        """
        Get the dataset for the given params.

        Args:
            params (GridParameter): The params to build the dataset.

        Returns:
            dict[str, np.matrix]: The keys should be 'y' and 'x'.
        """
        tfidf_params = params.tfidf
        if tfidf_params != self._cached_params.tfidf:
            logging.info(f'Preprocessing tfidf: {tfidf_params}..')
            with __silent__():
                preprocessor = linear.Preprocessor(tfidf_params=asdict(tfidf_params))
                self._cached_params.tfidf = tfidf_params
                self._cached_tfidf_data = preprocessor.fit_transform(self.datasets)['train']

        return self._cached_tfidf_data

    def get_fold_data(self, dataset, params):
        fold = params.fold
        if params.tfidf != self._cached_params.tfidf or fold != self._cached_params.fold:
            logging.info(f'Preprocessing fold: {fold} for tfidf: {params.tfidf}..')
            self._cached_params.fold = fold
            self._cached_fold_data = (
                dataset["y"][self.fold_idx[fold]['train']], dataset["x"][self.fold_idx[fold]['train']],
                dataset["y"][self.fold_idx[fold]['valid']], dataset["x"][self.fold_idx[fold]['valid']]
                )

        return self._cached_fold_data

    def get_tree_root(self, y, x, params):
        tree_params = params.tree
        if params.tfidf != self._cached_params.tfidf or tree_params != self._cached_params.tree or \
            params.fold != self._cached_params.fold:
            logging.info(f'Preprocessing tree: {tree_params} on fold {params.fold} for tfidf: {params.tfidf}..')
            with __silent__():
                label_representation = (y.T * x).tocsr()
                label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
                self._cached_params.tree = tree_params
                self._cached_tree_root = _build_tree(label_representation, np.arange(y.shape[1]), 0, **asdict(tree_params))
                self._cached_tree_root.is_root = True

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
        logging.info(f'\nRunning fold {params.fold}\nparams: {params}')

        linear_params = params.linear
        if params.tfidf != self._cached_params.tfidf or params.tree != self._cached_params.tree or \
            linear_params != self._cached_params.linear or params.fold != self._cached_params.fold:
            logging.info(f'Preprocessing linear: {linear_params}, tree: {params.tree} on fold {params.fold} for tfidf: {params.tfidf}..')
            root = self.get_tree_root(y, x, params)
            self._cached_params.linear = linear_params
            self._cached_model = linear.train_tree(y, x, root=root, options=params.linear_options)

        return self._cached_model

    def get_cv_score(self, y, x, model, params):
        logging.info(f'Scoring params: {params.predict}')

        batch_size = 256
        num_instances = x.shape[0]
        num_batches = math.ceil(num_instances / batch_size)

        metrics = linear.get_metrics(self.metrics, num_classes=y.shape[1])

        for i in range(num_batches):
            preds = model.predict_values(
                x[i * batch_size : (i + 1) * batch_size],
                **asdict(params.predict))
            target = y[i * batch_size : (i + 1) * batch_size].toarray()
            metrics.update(preds, target)

        scores = metrics.compute()
        logging.info(f'cv_score: {scores}\n')

        return scores

    def output(self):
        return sorted(self.results.items(), key=lambda x: x[1][self.metrics[0]], reverse=True)

    def __call__(self, search_space):
        self.search_space = [GridParameter(params, fold) for params in search_space for fold in range(self.n_folds)]
        self.sort_search_space()
        self.build_fold_idx()

        self.results = {
            GridParameter(params): {metric: 0 for metric in self.metrics} for params in search_space
            }
        for params in self.search_space:
            dataset = self.get_dataset(params)
            y_train_fold, x_train_fold, y_valid_fold, x_valid_fold = \
                self.get_fold_data(dataset, params)

            model = self.get_model(y_train_fold, x_train_fold, params)
            cv_score = self.get_cv_score(y_valid_fold, x_valid_fold, model, params)

            params.fold = -1
            for metric in self.metrics:
                self.results[params][metric] += cv_score[metric] / self.n_folds

        return self.output()
