import libmultilabel.linear as linear
import grid as grid
import numpy as np

import time
import json
from tqdm import tqdm


def run_ovr(dataset, options, *args, **kwargs):
    training_start = time.time()
    ovr_model = linear.train_1vsrest(
        dataset["train"]["y"],
        dataset["train"]["x"],
        options=options
        )
    training_time = time.time() - training_start
    return ovr_model, training_time

def run_tree(dataset, options, K, dmax, *args, **kwargs):
    training_start = time.time()
    tree_model = linear.train_tree(
        dataset["train"]["y"],
        dataset["train"]["x"],
        options=options,
        K=K,
        dmax=dmax
        )
    training_time = time.time() - training_start
    return tree_model, training_time


if __name__ == "__main__":
    import argparse
    np.random.seed(20250820)

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--dataset", type=str, default="EUR-Lex", help="Dataset name (e.g., AmazonCat-13K, EUR-Lex)")
    args = parser.parse_args()

    dataset_ = args.dataset

    # dataset = linear.load_dataset("svm", f"data/{dataset_}/train.svm")  # , f"data/{dataset}/test.svm"
    data_source = [f'data/{dataset_}/train.svm', f'data/{dataset_}/test.svm']
    search_space = {
        'tfidf': {
            'min_df': [1, 2],
            'max_features': [10000, 320000],
        },
        'params': {
            'C': [1, 2],
            'K': [2, 100],
        },
    }
    search_space = [
        {'max_features': i, 'K': j} for i in [10000] for j in [2, 100]
    ]
    print(search_space)
    n_folds = 3
    grid_search = grid.HyperparameterSearch(data_source, n_folds, search_space)
    results = grid_search()
    print(results)
    # if num_classes != -1:
    #     dataset["train"]["y"] = [[yij % num_classes for yij in yi] for yi in dataset["train"]["y"]]

    # preprocessor = linear.Preprocessor()
    # dataset = preprocessor.fit_transform(dataset)

    # results = {
    #     exp_name: {
    #         t: 0 for t in exp_threads
    #     }
    #     for exp_name in exp_names
    # }

    # for exp_name in exp_names:
    #     for exp_thread in tqdm(exp_threads, leave=True, colour="blue", desc=exp_name):
    #         if exp_name == 'Strategy B':
    #             do_parallel = True
    #             options = "-m 1"
    #             num_threads = exp_thread
    #         else:
    #             do_parallel = False
    #             options = f"-m {exp_thread}"
    #             num_threads = -1

    #         _, training_time = run_ovr(dataset, options, num_threads, do_parallel, use_dedicated_x)
    #         results[exp_name][exp_thread] = training_time
