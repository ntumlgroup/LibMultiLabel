import os
import sys
from abc import abstractmethod
from dataclasses import make_dataclass, field, fields, asdict
from typing import Callable

import libmultilabel.linear as linear
from libmultilabel.linear.tree import _build_tree, silent_print

import sklearn.preprocessing
import numpy as np
import math


class silent_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
        ]
    _predict_fields = [
        ('beam_width', int, field(default=10)),
        ('A', int, field(default=1)),
        ]

    param_types = {
        'tfidf': make_dataclass('_TfidfParams', _tfidf_fields, frozen=True, order=True),
        'tree': make_dataclass('_TreeParams', _tree_fields, frozen=True, order=True),
        'linear': make_dataclass('_LinearParams', _linear_fields, frozen=True, order=True),
        'predict': make_dataclass('_PredictParams', _predict_fields, frozen=True, order=True),
    }

    def __init__(self, params: dict):
        self.params = params
        for param_type, class_name in self.param_types.items():
            field_names = {f.name for f in fields(class_name)}
            _params = {k: v for k, v in self.params.items() if k in field_names}
            setattr(self, param_type, class_name(**_params))

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
    def __init__(self, data_source: tuple[str, str], n_folds: int, search_space: list[dict], config=None):
        self.data_source = data_source
        self.search_space = [GridParameter(params) for params in search_space]
        self.config = config
        self.n_folds = n_folds
        self.metrics = ["P@1", "P@3", "P@5"]

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

    def get_fold_data(self, dataset, i, params):
        return (
            dataset["y"][self.fold_idx[i]['train']], dataset["x"][self.fold_idx[i]['train']],
            dataset["y"][self.fold_idx[i]['valid']], dataset["x"][self.fold_idx[i]['valid']]
            )

    def get_cv_score(self, y, x, model, params):
        batch_size = 256
        num_instances = x.shape[0]
        num_batches = math.ceil(num_instances / batch_size)

        metrics = linear.get_metrics(self.metrics, num_classes=y.shape[1])

        for i in range(num_batches):
            preds = linear.predict_values(model, x[i * batch_size : (i + 1) * batch_size])
            target = y[i * batch_size : (i + 1) * batch_size].toarray()
            metrics.update(preds, target)

        return metrics.compute()

    def output(self):  # return sorted params list with scores by default
        return sorted(self.results.items(), key=lambda x: x[1][self.metrics[0]], reverse=True)

    def __call__(self):
        self.sort_search_space()
        self.build_fold_idx()

        self.results = {
            params: {metric: 0 for metric in self.metrics}
            for params in self.search_space
            }
        # for fold, params in zip(self.fold_space, self.search_space):
        for params in self.search_space:  # params should be an instance of a config class
            avg_score = {metric: 0 for metric in self.metrics}
            dataset = self.get_dataset(params)
            # should be 000111222... or 012012012... (for same tfidf params but different params)
            # don't know whether 012012012 waste space (view or new data)?
            for fold in range(self.n_folds):
                # secretly caching the tree root for each fold..
                y_train_fold, x_train_fold, y_valid_fold, x_valid_fold = \
                    self.get_fold_data(dataset, fold, params)

                print(f'\nRunning fold {fold}\nparams: {params}')
                self.model = self.get_model(y_train_fold, x_train_fold, fold, params)
                cv_score = self.get_cv_score(y_valid_fold, x_valid_fold, self.model, params)
                print(f'cv_score: {cv_score}\n')

                for metric in self.metrics:
                    self.results[params][metric] += cv_score[metric] / self.n_folds

        return self.output()

    @abstractmethod
    def get_dataset(self, params) -> dict[str, np.matrix]:
        """
        Get the dataset for the given params.

        Args:
            params (GridParameter): The params to build the dataset.

        Returns:
            dict[str, np.matrix]: The keys should be 'y' and 'x'.
        """
        pass

    @abstractmethod
    def get_model(self, y, x, fold, params) -> linear.FlatModel | linear.TreeModel:
        """
        Get the model for the given params.

        Args:
            y (np.matrix): The labels of the training data.
            x (np.matrix): The features of the training data.
            params (GridParameter): The params to build the model.

        Returns:
            linear.FlatModel | linear.TreeModel: The model for the given params.
        """
        pass


class HyperparameterSearch(GridSearch):
    def __init__(self, data_source, n_folds, search_space, config=None):
        super().__init__(data_source, n_folds, search_space, config)
        self._cached_tfidf_params = None
        self._cached_tfidf_data = None
        self._cached_tree_params = None
        self._cached_tree_roots = {fold: None for fold in range(self.n_folds)}
        # pass directly in the product code (linear_trainer.py)
        self.dataset = linear.load_dataset("svm", self.data_source[0], self.data_source[1])
        self.num_instances = len(self.dataset["train"]["y"])

    def get_dataset(self, params):
        tfidf_params = params.tfidf
        if tfidf_params != self._cached_tfidf_params:
            print(f'Preprocessing tfidf: {tfidf_params}..')
            self._cached_tfidf_params = tfidf_params
            with silent_print():
                preprocessor = linear.Preprocessor(tfidf_params=asdict(tfidf_params))
                self._cached_tfidf_data = preprocessor.fit_transform(self.dataset)['train']

        return self._cached_tfidf_data

    def get_tree_root(self, y, x, params):
        with silent_print():
            label_representation = (y.T * x).tocsr()
            label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
            root = _build_tree(label_representation, np.arange(y.shape[1]), 0, **params)
            root.is_root = True

        return root

    def get_model(self, y, x, fold, params):
        tree_params = params.tree
        if tree_params != self._cached_tree_params:
            self._cached_tree_params = tree_params
            self._cached_tree_roots = {fold: None for fold in range(self.n_folds)}

        if self._cached_tree_roots[fold] is None:
            print(f'Preprocessing tree: {tree_params} on fold {fold}..')
            self._cached_tree_roots[fold] = self.get_tree_root(y, x, asdict(tree_params))

        model = linear.train_tree(y, x, root=self._cached_tree_roots[fold], options=params.linear_options)

        return model


class ProbEstimatiteSearch(GridSearch):
    def __init__(self, data_source, n_folds, search_space, config=None):
        super().__init__(data_source, n_folds, search_space, config)

    def build_data(self):
        data = {'unique': {}}
        unique_data = None  # from libmultilabel preprocessing
        for i in range(self.n_folds):
            train_idx, valid_idx = None, None
            y_train_fold, x_train_fold = unique_data[train_idx]
            y_valid_fold, x_valid_fold = unique_data[valid_idx]
            data['unique'][i] = unique_data

        return data

    def get_fold_data(self, data, i, params):
        return data['unique'][i]

    def get_model(self, y_train_fold, x_train_fold, params):
        model = None  # train normally with fold data
        return model

    def get_cv_score(self, y_valid_fold, x_valid_fold, model, params):
        score = None  # calculate the metric with the model and the hyperparameter A
        return score
