import logging
from math import ceil

import numpy as np
from tqdm import tqdm

import libmultilabel.linear as linear
from libmultilabel.common_utils import dump_log, is_multiclass_dataset
from libmultilabel.linear.tree import train_ensemble_tree
from libmultilabel.linear.utils import LINEAR_TECHNIQUES, linear_test


def linear_train(datasets, config):
    # detect task type
    multiclass = is_multiclass_dataset(datasets["train"], "y")

    # train
    if config.linear_technique == "tree":
        if multiclass:
            raise ValueError("Tree model should only be used with multilabel datasets.")

        if config.tree_ensemble_models > 1:
            model = train_ensemble_tree(
                datasets["train"]["y"],
                datasets["train"]["x"],
                options=config.liblinear_options,
                K=config.tree_degree,
                dmax=config.tree_max_depth,
                n_trees=config.tree_ensemble_models,
                seed=config.seed,
            )
        else:
            model = LINEAR_TECHNIQUES[config.linear_technique](
                datasets["train"]["y"],
                datasets["train"]["x"],
                options=config.liblinear_options,
                K=config.tree_degree,
                dmax=config.tree_max_depth,
            )
    else:
        model = LINEAR_TECHNIQUES[config.linear_technique](
            datasets["train"]["y"],
            datasets["train"]["x"],
            multiclass=multiclass,
            options=config.liblinear_options,
        )
    return model


def linear_run(config):
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.eval:
        preprocessor, model = linear.load_pipeline(config.checkpoint_path)
        datasets = linear.load_dataset(config.data_format, config.training_file, config.test_file)
        datasets = preprocessor.transform(datasets)
    else:
        preprocessor = linear.Preprocessor(config.include_test_labels, config.remove_no_label_data)
        datasets = linear.load_dataset(
            config.data_format,
            config.training_file,
            config.test_file,
            config.label_file,
        )
        datasets = preprocessor.fit_transform(datasets)
        model = linear_train(datasets, config)
        linear.save_pipeline(config.checkpoint_dir, preprocessor, model)

    if config.test_file is not None:
        assert not (
            config.save_positive_predictions and config.save_k_predictions > 0
        ), """
            If save_k_predictions is larger than 0, only top k labels are saved.
            Save all labels with decision value larger than 0 by using save_positive_predictions and save_k_predictions=0."""
        metrics, metric_dict, labels, scores = linear_test(
            y = datasets["test"]["y"],
            x = datasets["test"]["x"],
            model = model,
            eval_batch_size = config.eval_batch_size,
            monitor_metrics = config.monitor_metrics,
            beam_width = config.beam_width,
            prob_A = config.prob_A,
            label_mapping = preprocessor.label_mapping,
            save_k_predictions = config.save_k_predictions,
            save_positive_predictions = config.save_positive_predictions,
        )
        dump_log(config=config, metrics=metric_dict, split="test", log_path=config.log_path)
        print(linear.tabulate_metrics(metric_dict, "test"))
        if config.save_k_predictions > 0:
            with open(config.predict_out_path, "w") as fp:
                for label, score in zip(labels, scores):
                    out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                    fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
        elif config.save_positive_predictions:
            with open(config.predict_out_path, "w") as fp:
                for batch_labels, batch_scores in zip(labels, scores):
                    for label, score in zip(batch_labels, batch_scores):
                        out_str = " ".join([f"{i}:{s:.4}" for i, s in zip(label, score)])
                        fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {config.predict_out_path}")
