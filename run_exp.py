import libmultilabel.linear as linear
import grid

import numpy as np
from dataclasses import asdict


def prune_model(*args, **kwargs):
    pass


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)
    np.random.seed(20260123)

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--dataset", type=str, default="EUR-Lex", help="Dataset name (e.g., AmazonCat-13K, EUR-Lex)")
    parser.add_argument("--data_format", type=str, default="txt", help="Data format.")
    args = parser.parse_args()

    dataset = linear.load_dataset(args.data_format, f"data/{args.dataset}/train.{args.data_format}")  # , f"data/{dataset}/test.{args.data_format}"

    retrain = True
    n_folds = 3
    monitor_metrics = ["P@1", "P@3", "P@5"]
    search_space_dict = {
        'max_features': [10000, 20000],
        'K': [10, 100],
        'min_df': [1, 2],
        'prob_A': [2, 3],
        'c': [0.1, 0.2],
        'pruning_alpha': [0.9, 0.7],
    }

    search = grid.GridSearch(dataset, n_folds, monitor_metrics)
    cv_scores = search(search_space_dict)
    sorted_cv_scores = sorted(cv_scores.items(), key=lambda x: x[1][monitor_metrics[0]], reverse=True)
    print(sorted_cv_scores)

    if retrain:
        # TODO: test set
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
