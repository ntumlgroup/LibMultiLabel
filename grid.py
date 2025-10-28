from abc import abstractmethod

import libmultilabel.linear as linear
import numpy as np
import math

class Parameter:
    def __init__(self, **params):
        self.params = params
    
    def tfidf(self):  # pad default value for compatibility
        return self.params['tfidf']

    def tree(self):
        return self.params['tree']

    def params(self):
        return self.params['params']

    def inference(self):
        return self.params['inference']


param = Parameter(tfidf={'min_df': 1, 'max_features': 10000}, tree={'K': 2, 'dmax': 100})

class GridSearch:
    def __init__(self, data_source, n_folds, search_space, config=None):
        self.data_source = data_source
        self.search_space = search_space
        self.config = config
        self.n_folds = n_folds
        self.metrics = ["P@1", "P@3", "P@5"]

    def __call__(self):
        self.build_data()
        self.build_fold_idx()

        results = {
            (str(tfidf_param), str(param)): {metric: 0 for metric in self.metrics}
            for tfidf_param in self.search_space['tfidf'] for param in self.search_space['params']
            }
        # for fold, params in zip(self.fold_space, self.search_space):
        for tfidf_param in self.search_space['tfidf']:  # param should be an instance of a config class
            avg_score = {metric: 0 for metric in self.metrics}
            for i in range(self.n_folds):
                y_train_fold, x_train_fold, y_valid_fold, x_valid_fold = \
                    self.get_fold_data(i, tfidf_param)
                for tree
                for param in self.search_space['params']:
                    print(f'\nRunning fold {i}\ntfidf: {tfidf_param}\nparams: {param}')
                    model = self.get_model(y_train_fold, x_train_fold, param)
                    cv_score = self.get_cv_score(y_valid_fold, x_valid_fold, model, param)
                    print(f'cv_score: {cv_score}\n')
                    for metric in self.metrics:
                        results[(str(tfidf_param), str(param))][metric] += cv_score[metric] / self.n_folds

        # TODO: Return a function
        return sorted(results.items(), key=lambda x: x[1][self.metrics[0]], reverse=True)

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

    @abstractmethod
    def build_data(self):
        pass

    @abstractmethod
    def get_fold_data(self, i, param):
        pass

    @abstractmethod
    def get_model(self, y_train_fold, x_train_fold, param):
        pass

    @abstractmethod
    def get_cv_score(self, y_valid_fold, x_valid_fold, model, param):
        pass


class HyperparameterSearch(GridSearch):
    def __init__(self, data_source, n_folds, search_space, config=None):
        super().__init__(data_source, n_folds, search_space, config)

    def preprocess_tfidf(self, dataset, param):
        preprocessor = linear.Preprocessor(tfidf_params=param)
        return preprocessor.fit_transform(dataset)

    def build_data(self):
        self.data = {}

        dataset = linear.load_dataset("svm", self.data_source[0], self.data_source[1])
        self.num_instances = len(dataset["train"]["y"])
        tfidf_params = self.search_space['tfidf']
        for param in tfidf_params:
            print(f'Preprocessing tfidf: {param}..')
            tfidf_data = self.preprocess_tfidf(dataset, param)
            self.data[str(param)] = {'dataset': tfidf_data}
        # use yield? (however, hard to reuse)

    def get_fold_data(self, i, param):
        dataset = self.data[str(param)]['dataset']["train"]
        return (
            dataset["y"][self.fold_idx[i]['train']], dataset["x"][self.fold_idx[i]['train']],
            dataset["y"][self.fold_idx[i]['valid']], dataset["x"][self.fold_idx[i]['valid']]
            )

    def get_model(self, y_train_fold, x_train_fold, param):
        model = linear.train_tree(y_train_fold, x_train_fold, **param)  # train with param and fold data
        return model

    def metrics_in_batches(self, y, x, model, *args, **kwargs):
        batch_size = 256
        num_instances = x.shape[0]
        num_batches = math.ceil(num_instances / batch_size)

        metrics = linear.get_metrics(self.metrics, num_classes=y.shape[1])

        for i in range(num_batches):
            preds = linear.predict_values(model, x[i * batch_size : (i + 1) * batch_size])
            target = y[i * batch_size : (i + 1) * batch_size].toarray()
            metrics.update(preds, target)

        return metrics.compute()

    def get_cv_score(self, y_valid_fold, x_valid_fold, model, param):
        # calculate the metric with the model
        score = self.metrics_in_batches(
            y_valid_fold,
            x_valid_fold,
            model,
            **param
            )
        return score


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

    def get_fold_data(self, data, i, param):
        return data['unique'][i]

    def get_model(self, y_train_fold, x_train_fold, param):
        model = None  # train normally with fold data
        return model

    def get_cv_score(self, y_valid_fold, x_valid_fold, model, param):
        score = None  # calculate the metric with the model and the hyperparameter A
        return score
