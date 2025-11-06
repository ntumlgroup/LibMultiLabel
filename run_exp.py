import libmultilabel.linear as linear
import grid as grid
import numpy as np

import time
import json
from tqdm import tqdm


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
    search_space = [
        {'max_features': i, 'K': j, 'min_df': k, 'c': l}
        for i in [10000, 20000] for j in [10, 100] for k in [1, 2] for l in [0.1, 0.2]
    ]

    for i in search_space:
        print(i)

    n_folds = 3
    grid_search = grid.HyperparameterSearch(datasets, n_folds, search_space)
    results = grid_search()

    for i in results:
        print(i)
