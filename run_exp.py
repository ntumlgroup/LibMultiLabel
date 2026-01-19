import libmultilabel.linear as linear
import grid

import numpy as np
from dataclasses import asdict

import time
import json
from tqdm import tqdm
import itertools


def prune_model(*args, **kwargs):
    pass


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)
    np.random.seed(20250820)

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--dataset", type=str, default="EUR-Lex", help="Dataset name (e.g., AmazonCat-13K, EUR-Lex)")
    args = parser.parse_args()

    dataset_ = args.dataset

    datasets = linear.load_dataset("svm", f"data/{dataset_}/train.svm")  # , f"data/{dataset}/test.svm"
    # data_source = [f'data/{dataset_}/train.svm', f'data/{dataset_}/test.svm']
    # search_space = {
    #     'tfidf': {
    #         'min_df': [1, 2],
    #         'max_features': [10000, 320000],
    #     },
    #     'params': {
    #         'C': [1, 2],
    #         'K': [2, 100],
    #     },
    # }
    n_folds = 3
    retrain = False
    linear_technique = 'tree'
    search_space_dict = {
        'max_features': [10000, 20000],
        'K': [10, 100],
        'min_df': [1, 2],
        'A': [2, 3],
        'c': [0.1, 0.2],
    }
    param_names = search_space_dict.keys()
    search_space = [
        dict(zip(param_names, param_values))
        for param_values in itertools.product(*search_space_dict.values())
    ]
    # search_space = [dict()]  # all default values

    # search_space = [
    #     {'max_features': i, 'K': j, 'min_df': k, 'c': l}
    #     for i in [10000, 20000] for j in [10, 100] for k in [1, 2] for l in [0.1, 0.2]
    # ]

    for i in search_space:
        print(i)

    search = grid.GridSearch(datasets, n_folds)
    best_params = search(search_space)
    print(best_params)
    breakpoint()

    # if best_params.tfidf == search._cached_tfidf_params:
    #     datasets = search._cached_tfidf_data
    # else:
    #     preprocessor = linear.Preprocessor(tfidf_params=asdict(best_params.tfidf))
    #     datasets = preprocessor.fit_transform(datasets)
    #     search.init_tfidf_cache(datasets, best_params)

    # best_alpha = search(['alpha'])[0]
    # best_A = search(['A'])[0]
    # # TODO (the fields are frozen)
    # best_params.linear.alpha = best_alpha
    # best_params.linear.A = best_A

    if retrain:
        model = linear.LINEAR_TECHNIQUES[linear_technique](
                    datasets["train"]["y"],
                    datasets["train"]["x"],
                    **asdict(best_params.linear),
                )
