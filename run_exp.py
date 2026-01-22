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
    parser.add_argument("--data_format", type=str, default="txt", help="Data format.")
    args = parser.parse_args()

    dataset = linear.load_dataset(args.data_format, f"data/{args.dataset}/train.{args.data_format}")  # , f"data/{dataset}/test.{args.data_format}"

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

    # for i in search_space:
    #     print(i)

    search = grid.GridSearch(dataset, n_folds)
    scores = search(search_space_dict)
    print(scores)
    breakpoint()

    if retrain:
        # TODO
        best_params = None
        model = linear.LINEAR_TECHNIQUES[linear_technique](
                    dataset["train"]["y"],
                    dataset["train"]["x"],
                    **asdict(best_params.linear),
                )
