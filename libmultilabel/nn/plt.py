from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from scipy.special import expit
from lightning import Trainer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning.pytorch.callbacks import ModelCheckpoint

from .cluster import CLUSTER_NAME, FILE_EXTENSION as CLUSTER_FILE_EXTENSION, build_label_tree
from .data_utils import UNK
from .datasets_AttentionXML import PlainDataset, PLTDataset
from .model_AttentionXML import PLTModel
from ..common_utils import dump_log
from ..linear.preprocessor import Preprocessor
from ..nn import networks
from ..nn.model import Model

__all__ = ["PLTTrainer"]

from .nn_utils import init_trainer, init_model

logger = logging.getLogger(__name__)


class PLTTrainer:
    CHECKPOINT_NAME = "model_"

    def __init__(
        self,
        config,
        classes: Optional[list] = None,
        embed_vecs: Optional[Tensor] = None,
        word_dict: Optional[dict] = None,
    ):
        # The number of levels is set to 2. In other words, there will be 2 models
        self.multiclass = config.multiclass
        if self.multiclass:
            raise ValueError(
                "The label space of multi-class datasets is usually not large, so PLT training is unnecessary."
                "Please consider other methods."
                "If you have a multi-class set with numerous labels, please let us know"
            )

        # cluster
        self.cluster_size = config.cluster_size
        # predict the top k clusters for deciding relevant/irrelevant labels of each instance in level 1 model training
        self.beam_width = config.beam_width
        self.save_k_predictions = config.save_k_predictions

        # dataset meta info
        self.embed_vecs = embed_vecs
        self.word_dict = word_dict
        self.classes = classes
        self.max_seq_length = config.max_seq_length
        self.num_classes = len(classes)

        # multilabel binarizer fitted to the datasets
        self.binarizer = None

        # cluster meta info
        self.cluster_size = config.cluster_size

        # network parameters
        self.network_config = config.network_config
        self.init_weight = "xavier_uniform"  # AttentionXML-specific setting
        self.loss_function = config.loss_function

        # optimizer parameters
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        # learning rate scheduler
        self.lr_scheduler = config.lr_scheduler
        self.scheduler_config = config.scheduler_config

        # Trainer parameters
        self.use_cpu = config.cpu
        self.accelerator = "cpu" if self.use_cpu else "gpu"
        self.devices = 1
        self.num_nodes = 1
        self.epochs = config.epochs
        self.limit_train_batches = config.limit_train_batches
        self.limit_val_batches = config.limit_val_batches
        self.limit_test_batches = config.limit_test_batches

        # callbacks
        self.silent = config.silent
        # EarlyStopping
        self.early_stopping_metric = config.early_stopping_metric
        self.patience = config.patience
        # ModelCheckpoint
        self.val_metric = config.val_metric
        self.checkpoint_dir = Path(config.checkpoint_dir)

        self.metrics = config.monitor_metrics
        self.metric_threshold = config.metric_threshold
        self.monitor_metrics = config.monitor_metrics

        # dataloader parameters
        # whether shuffle the training dataset or not during the training process
        self.shuffle = config.shuffle
        pin_memory = True if self.accelerator == "gpu" else False
        # training DataLoader
        self.dataloader = partial(
            DataLoader,
            batch_size=config.batch_size,
            num_workers=config.data_workers,
            pin_memory=pin_memory,
        )
        # evaluation DataLoader
        self.eval_dataloader = partial(
            self.dataloader,
            batch_size=config.eval_batch_size,
        )

        # save path
        self.log_path = config.log_path
        self.predict_out_path = config.predict_out_path
        self.config = config

    def label2cluster(self, cluster_mapping, *labels) -> Generator[csr_matrix, ...]:
        """Map labels to their corresponding clusters in CSR sparse format.
        Notice that this function deals with SPARSE matrix.
        Assume there are 6 labels clustered as [(0, 1), (2, 3), (4, 5)]. Here (0, 1) is cluster with index 0 and so on.
        Given the ground-truth labels, [0, 1, 4], the resulting clusters are [0, 2].

        Args:
            cluster_mapping (np.ndarray): mapping from clusters to labels generated by build_label_tree.
            *labels (csr_matrix): labels in CSR sparse format.

        Returns:
            Generator[csr_matrix]: resulting clusters converted from labels in CSR sparse format
        """
        mapping = np.empty(self.num_classes, dtype=np.uint32)
        for idx, clusters in enumerate(cluster_mapping):
            mapping[clusters] = idx

        def _label2cluster(label: csr_matrix) -> csr_matrix:
            row = []
            col = []
            data = []
            for i in range(label.shape[0]):
                # n include all mapped ancestor clusters
                n = np.unique(mapping[label.indices[label.indptr[i] : label.indptr[i + 1]]])
                row += [i] * len(n)
                col += n.tolist()
                data += [1] * len(n)
            return csr_matrix((data, (row, col)), shape=(label.shape[0], len(cluster_mapping)))

        return (_label2cluster(label) for label in labels)

    @staticmethod
    def cluster2label(cluster_mapping, clusters, cluster_scores=None):
        """Expand clusters to their corresponding labels and, if available, assign scores to each label.
        Labels inside the same cluster have the same scores. This function is applied to predictions from model 0.
        Notice that the behaviors of this function are different from label2cluster.
        Also notice that this function deals with DENSE matrix.

        Args:
            cluster_mapping (np.ndarray): mapping from clusters to labels generated by build_label_tree.
            clusters (np.ndarray): predicted clusters from model 0.
            cluster_scores (Optional: np.ndarray): predicted scores of each cluster from model 0.

        Returns:
            Generator[np.ndarray]: resulting labels expanded from clusters
        """

        labels_selected = []

        if cluster_scores is not None:
            # label_scores are corresponding scores for selected labels and
            # shape: (len(x), cluster_size * top_k)
            label_scores = []
            for score, cluster in zip(cluster_scores, clusters):
                label_scores += [np.repeat(score, [len(labels) for labels in cluster_mapping[cluster]])]
                labels_selected += [np.concatenate(cluster_mapping[cluster])]
            return labels_selected, label_scores
        else:
            labels_selected = [np.concatenate(cluster_mapping[cluster]) for cluster in clusters]
            return labels_selected

    def fit(self, datasets):
        """fit model to the training dataset

        Args:
            datasets: dict containing training, validation, and/or test datasets
        """
        if self.get_best_model_path(level=1).exists():
            return

        # AttentionXML-specific data preprocessing
        train_val_dataset = datasets["train"] + datasets["val"]
        train_val_dataset = {
            "x": [" ".join(i["text"]) for i in train_val_dataset],
            "y": [i["label"] for i in train_val_dataset],
        }

        # Preprocessor does tf-idf vectorization and multilabel binarization
        # For details, see libmultilabel.linear.preprocessor.Preprocessor
        preprocessor = Preprocessor()
        datasets_temp = {"data_format": "txt", "train": train_val_dataset, "classes": self.classes}
        # Preprocessor requires the input dictionary to has a key named "train" and will return a new dictionary with
        # the same key.
        train_val_dataset_tf = preprocessor.fit_transform(datasets_temp)["train"]
        # save binarizer for testing
        self.binarizer = preprocessor.binarizer

        train_x = self.reformat_text(datasets["train"])
        val_x = self.reformat_text(datasets["val"])

        train_y = train_val_dataset_tf["y"][: len(datasets["train"])]
        val_y = train_val_dataset_tf["y"][len(datasets["train"]) :]

        # clusters are saved to the disk so that users doesn't need to provide the original training data when they want
        # to do predicting solely
        build_label_tree(
            sparse_x=train_val_dataset_tf["x"],
            sparse_y=train_val_dataset_tf["y"],
            cluster_size=self.cluster_size,
            output_dir=self.checkpoint_dir,
        )

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)

        # each y has been mapped to the cluster indices of its parent
        train_y_clustered, val_y_clustered = self.label2cluster(clusters, train_y, val_y)

        trainer = init_trainer(
            self.checkpoint_dir,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping_metric=self.early_stopping_metric,
            val_metric=self.val_metric,
            silent=self.silent,
            use_cpu=self.use_cpu,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            save_checkpoints=True,
        )
        trainer.checkpoint_callback.filename = f"{self.CHECKPOINT_NAME}0"

        train_dataloader = self.dataloader(PlainDataset(train_x, train_y_clustered), shuffle=self.shuffle)
        val_dataloader = self.dataloader(PlainDataset(val_x, val_y_clustered))

        best_model_path = self.get_best_model_path(level=0)
        if not best_model_path.exists():
            model_0 = init_model(
                model_name="AttentionXML_0",
                network_config=self.network_config,
                classes=clusters,
                word_dict=self.word_dict,
                embed_vecs=self.embed_vecs,
                init_weight=self.init_weight,
                log_path=self.log_path,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                lr_scheduler=self.lr_scheduler,
                scheduler_config=self.scheduler_config,
                val_metric=self.val_metric,
                metric_threshold=self.metric_threshold,
                monitor_metrics=self.monitor_metrics,
                multiclass=self.multiclass,
                loss_function=self.loss_function,
                silent=self.silent,
                save_k_predictions=self.beam_width,
            )

            logger.info(f"Training level 0. Number of clusters: {len(clusters)}")
            trainer.fit(model_0, train_dataloader, val_dataloader)
            logger.info(f"Finish training level 0")

        logger.info(f"Best model loaded from {best_model_path}")
        model_0 = Model.load_from_checkpoint(best_model_path)

        logger.info(
            f"Predicting clusters by level-0 model. We then select {self.beam_width} clusters and "
            f"extract labels from them for level 1 training."
        )
        # load training and validation data and predict corresponding level 0 clusters
        train_dataloader = self.dataloader(PlainDataset(train_x))
        val_dataloader = self.dataloader(PlainDataset(val_x))

        train_pred = trainer.predict(model_0, train_dataloader)
        val_pred = trainer.predict(model_0, val_dataloader)

        train_clusters_pred = np.vstack([i["top_k_pred"] for i in train_pred])
        val_scores_pred = expit(np.vstack([i["top_k_pred_scores"] for i in val_pred]))
        val_clusters_pred = np.vstack([i["top_k_pred"] for i in val_pred])

        train_clusters_selected = np.empty((len(train_x), self.beam_width), dtype=np.uint)
        for i, ys in enumerate(tqdm(train_clusters_pred, leave=False, desc="Sampling clusters")):
            # relevant clusters are positive
            pos = set(train_y_clustered.indices[train_y_clustered.indptr[i] : train_y_clustered.indptr[i + 1]])
            # Select relevant clusters first. Then from top-predicted clusters, sequentially include them until
            # cluster number reaches beam width
            if len(pos) <= self.beam_width:
                selected = pos
                for y in ys:
                    y = y.item()
                    if len(selected) == self.beam_width:
                        break
                    selected.add(y)
            # Regard positive (true) label as samples iff they appear in the predicted labels
            # if the number of positive labels is more than top_k. If samples are not of length top_k
            # add unseen predicted labels until reaching top_k.
            else:
                selected = set()
                for y in ys:
                    y = y.item()
                    if y in pos:
                        selected.add(y)
                    if len(selected) == self.beam_width:
                        break
                if len(selected) < self.beam_width:
                    selected = (list(selected) + list(pos - selected))[: self.beam_width]
            train_clusters_selected[i] = np.asarray(list(selected))

        train_labels_selected = PLTTrainer.cluster2label(clusters, train_clusters_selected)
        val_labels_pred, val_scores_pred = PLTTrainer.cluster2label(clusters, val_clusters_pred, val_scores_pred)
        num_labels_selected = self.beam_width * max(len(c) for c in clusters)

        trainer = init_trainer(
            self.checkpoint_dir,
            epochs=self.epochs,
            patience=self.patience,
            early_stopping_metric=self.val_metric,
            val_metric=self.val_metric,
            silent=self.silent,
            use_cpu=self.use_cpu,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            save_checkpoints=True,
        )
        trainer.checkpoint_callback.filename = f"{self.CHECKPOINT_NAME}1"

        # train & val dataloaders for training
        train_dataloader = self.dataloader(
            PLTDataset(
                train_x,
                train_y,
                num_classes=self.num_classes,
                num_labels_selected=num_labels_selected,
                labels_selected=train_labels_selected,
            ),
            shuffle=self.shuffle,
        )
        val_dataloader = self.dataloader(
            PLTDataset(
                val_x,
                val_y,
                num_classes=self.num_classes,
                num_labels_selected=num_labels_selected,
                labels_selected=val_labels_pred,
                label_scores=val_scores_pred,
            ),
        )

        try:
            network = getattr(networks, "AttentionXML_1")(
                embed_vecs=self.embed_vecs, num_classes=len(self.classes), **dict(self.network_config)
            )
        except Exception:
            raise AttributeError("Failed to initialize AttentionXML")

        model_1 = PLTModel(
            classes=self.classes,
            word_dict=self.word_dict,
            embed_vecs=self.embed_vecs,
            network=network,
            log_path=self.log_path,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            lr_scheduler=self.lr_scheduler,
            scheduler_config=self.scheduler_config,
            val_metric=self.val_metric,
            metric_threshold=self.metric_threshold,
            monitor_metrics=self.monitor_metrics,
            multiclass=self.multiclass,
            loss_function=self.loss_function,
            silent=self.silent,
            save_k_predictions=self.save_k_predictions,
        )
        logger.info(f"Initialize model with weights from level 0")
        # For weights not initialized by the level-0 model, use xavier uniform initialization
        torch.nn.init.xavier_uniform_(model_1.network.attention.attention.weight)
        # As the attention layer of model 1 is different from model 0, each layer needs to be initialized separately
        model_1.network.embedding.load_state_dict(model_0.network.embedding.state_dict())
        model_1.network.encoder.load_state_dict(model_0.network.encoder.state_dict())
        model_1.network.output.load_state_dict(model_0.network.output.state_dict())

        del model_0

        logger.info(
            f"Training level 1. Number of labels: {self.num_classes}."
            f"Number of labels selected: {train_dataloader.dataset.num_labels_selected}"
        )
        trainer.fit(model_1, train_dataloader, val_dataloader)
        logger.info(f"Best model loaded from {best_model_path}")
        logger.info(f"Finish training level 1")

    def test(self, dataset):
        # retrieve word_dict from model_1
        # prediction starts from level 0
        model_0 = Model.load_from_checkpoint(
            self.get_best_model_path(level=0),
            save_k_predictions=self.beam_width,
        )
        model_1 = PLTModel.load_from_checkpoint(
            self.get_best_model_path(level=1),
            save_k_predictions=self.save_k_predictions,
            metrics=self.metrics,
        )
        self.word_dict = model_1.word_dict
        classes = model_1.classes

        test_x = self.reformat_text(dataset)

        if self.binarizer is None:
            binarizer = MultiLabelBinarizer(classes=classes, sparse_output=True)
            binarizer.fit(None)
            test_y = binarizer.transform((i["label"] for i in dataset))
        else:
            test_y = self.binarizer.transform((i["label"] for i in dataset))
        logger.info("Testing process started")
        trainer = Trainer(
            devices=1,
            accelerator=self.accelerator,
            logger=False,
        )

        test_dataloader = self.eval_dataloader(PlainDataset(test_x))

        logger.info(f"Predicting level 0. Number of clusters: {self.beam_width}")
        test_pred = trainer.predict(model_0, test_dataloader)
        test_scores_pred = expit(np.vstack([i["top_k_pred_scores"] for i in test_pred]))
        test_clusters_pred = np.vstack([i["top_k_pred"] for i in test_pred])

        clusters = np.load(self.get_cluster_path(), allow_pickle=True)
        test_labels_pred, test_scores_pred = PLTTrainer.cluster2label(clusters, test_clusters_pred, test_scores_pred)
        num_labels_selected = self.beam_width * max(len(c) for c in clusters)

        test_dataloader = self.eval_dataloader(
            PLTDataset(
                test_x,
                test_y,
                num_classes=self.num_classes,
                num_labels_selected=num_labels_selected,
                labels_selected=test_labels_pred,
                label_scores=test_scores_pred,
            ),
        )

        logger.info(f"Testing level 1")
        trainer.test(model_1, test_dataloader)
        logger.info("Testing process finished")

        if self.save_k_predictions > 0:
            batch_predictions = trainer.predict(model_1, test_dataloader)
            pred_labels = np.vstack([batch["top_k_pred"] for batch in batch_predictions])
            pred_scores = np.vstack([batch["top_k_pred_scores"] for batch in batch_predictions])
            with open(self.predict_out_path, "w") as fp:
                for pred_label, pred_score in zip(pred_labels, pred_scores):
                    out_str = " ".join(
                        [f"{model_1.classes[label]}:{score:.4}" for label, score in zip(pred_label, pred_score)]
                    )
                    fp.write(out_str + "\n")
            logging.info(f"Saved predictions to: {self.predict_out_path}")

        dump_log(self.log_path, config=self.config)

    def reformat_text(self, dataset):
        # Convert words to numbers according to their indices in word_dict. Then pad each instance to a certain length.
        encoded_text = list(
            map(
                lambda text: torch.tensor([self.word_dict[word] for word in text], dtype=torch.int64)
                if text
                else torch.tensor([self.word_dict[UNK]], dtype=torch.int64),
                [instance["text"][: self.max_seq_length] for instance in dataset],
            )
        )
        # pad the first entry to be of length 500 if necessary
        encoded_text[0] = torch.cat(
            (
                encoded_text[0],
                torch.tensor(0, dtype=torch.int64).repeat(self.max_seq_length - encoded_text[0].shape[0]),
            )
        )
        encoded_text = pad_sequence(encoded_text, batch_first=True)
        return encoded_text

    def get_best_model_path(self, level: int) -> Path:
        return self.checkpoint_dir / f"{self.CHECKPOINT_NAME}{level}{ModelCheckpoint.FILE_EXTENSION}"

    def get_cluster_path(self) -> Path:
        return self.checkpoint_dir / f"{CLUSTER_NAME}{CLUSTER_FILE_EXTENSION}"
