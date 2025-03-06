from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing
from tqdm import tqdm
import psutil
from dataclasses import dataclass

from . import linear

__all__ = ["train_tree", "TreeModel"]


class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children
        self.is_root = False

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None]):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    """A model returned from train_tree."""

    def __init__(
        self,
        root: Node,
        flat_model: linear.FlatModel,
        weight_map: np.ndarray,
        subtrees: list[SubTree],
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.weight_map = weight_map
        self.multiclass = False
        self.subtrees = subtrees

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
    ) -> np.ndarray:
        """Calculates the probability estimates associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        # number of instances * number of labels + total number of metalabels
        all_preds = self._prune_tree_predictions(x, beam_width)
        return np.vstack([self._beam_search(all_preds[i], beam_width) for i in range(all_preds.shape[0])])

    def _prune_tree_predictions(self, x: sparse.csr_matrix, beam_width: int) -> np.ndarray:
        """Calculates the decision values associated with x.

        If the beam width is smaller than the number of nodes at a some level, many nodes become unreachable, resulting in unnecessary computations.
        In LibMultiLabel's default setting, the beam width is smaller than the root's degree in the tree.
        To mitigate unnecessary computations, pruning is applied to predictions starting from the root.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int): Number of top candidate branches considered for prediction.

        Returns:
            np.ndarray: A matrix with dimension number of instances * (number of labels + total number of metalabels).
        """
        # Initialize space for all predictions with negative infinity
        num_instances, num_labels = x.shape[0], self.weight_map[-1]
        all_preds = np.full((num_instances, num_labels), np.NINF)

        # Calculate root decision value and scores
        root_preds = linear.predict_values(self.flat_model, x)
        children_scores = 0.0 - np.maximum(0, 1 - root_preds) ** 2

        slice = np.s_[:num_instances, self.weight_map[self.root.index] : self.weight_map[self.root.index + 1]]
        all_preds[slice] = root_preds

        if not self.root.isLeaf():
            # Find the top k subtree for each instance
            top_k_indices = np.argsort(-children_scores, axis=1, kind="stable")[:, :beam_width]

            # Building a mapping from subtree to instances
            subtree_to_instances = {
                self.subtrees[subtree_idx]: np.where(top_k_indices == subtree_idx)[0]
                for subtree_idx in np.unique(top_k_indices)
            }

            # Calculate predictions for each subtree with its corresponding instances
            for subtree, instances in subtree_to_instances.items():
                reduced_instances = x[np.s_[instances], :]
                # Locate the position of the subtree root in the weight mapping of all nodes.
                subtree_weights_start = self.weight_map[subtree.root.index]
                subtree_weights_end = subtree_weights_start + subtree.flat_model.weights.shape[1]

                slice = np.s_[instances, subtree_weights_start:subtree_weights_end]
                all_preds[slice] = linear.predict_values(subtree.flat_model, reduced_instances)

        return all_preds

    def _beam_search(self, instance_preds: np.ndarray, beam_width: int) -> np.ndarray:
        """Predict with beam search using cached probability estimates for a single instance.

        Args:
            instance_preds (np.ndarray): A vector of cached probability estimates of each node, has dimension number of labels + total number of metalabels.
            beam_width (int): Number of candidates considered.

        Returns:
            np.ndarray: A vector with dimension number of classes.
        """
        cur_level = [(self.root, 0.0)]  # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score - np.square(np.maximum(0, 1 - pred))
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        scores = np.zeros(num_labels)
        for node, score in cur_level:
            slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score - np.square(np.maximum(0, 1 - pred)))
        return scores

@dataclass(frozen=True)
class SubTree:
    """Represents a subtree with its root node and the linear flattened model which builts from the subtree's root."""
    root: Node
    flat_model: linear.FlatModel


def train_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> TreeModel:
    """Trains a linear model for multi-label data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree(label_representation, np.arange(y.shape[1]), 0, K, dmax)
    root.is_root = True

    num_nodes = 0
    # Both type(x) and type(y) are sparse.csr_matrix
    # However, type((x != 0).T) becomes sparse.csc_matrix
    # So type((x != 0).T * y) results in sparse.csc_matrix
    features_used_perlabel = (x != 0).T * y

    def count(node):
        nonlocal num_nodes
        num_nodes += 1
        node.num_features_used = np.count_nonzero(features_used_perlabel[:, node.label_map].sum(axis=1))

    root.dfs(count)

    model_size = get_estimated_model_size(root)
    print(f"The estimated tree model size is: {model_size / (1024**3):.3f} GB")

    # Calculate the total memory (excluding swap) on the local machine
    total_memory = psutil.virtual_memory().total
    print(f"Your system memory is: {total_memory / (1024**3):.3f} GB")

    if total_memory <= model_size:
        raise MemoryError(f"Not enough memory to train the model.")

    pbar = tqdm(total=num_nodes, disable=not verbose)

    index = 0

    def visit(node):
        nonlocal index
        node.index = index
        index += 1
        if node.is_root:
            _train_node(y, x, options, node)
        else:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
            _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    return _tree_model(root)


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Builds the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: root of the (sub)tree built from label_representation.
    """
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    metalabels = (
        sklearn.cluster.KMeans(
            K,
            random_state=np.random.randint(2**31 - 1),
            n_init=1,
            max_iter=300,
            tol=0.0001,
            algorithm="elkan",
        )
        .fit(label_representation)
        .labels_
    )

    children = []
    for i in range(K):
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]
        child = _build_tree(child_representation, child_map, d + 1, K, dmax)
        children.append(child)

    return Node(label_map=label_map, children=children)


def get_estimated_model_size(root):
    total_num_weights = 0

    def collect_stat(node: Node):
        nonlocal total_num_weights

        if node.isLeaf():
            total_num_weights += len(node.label_map) * node.num_features_used
        else:
            total_num_weights += len(node.children) * node.num_features_used

    root.dfs(collect_stat)

    # 16 is because when storing sparse matrices, indices (int64) require 8 bytes and floats require 8 bytes
    # Our study showed that among the used features of every binary classification problem, on average no more than 2/3 of weights obtained by the dual coordinate descent method are non-zeros.
    return total_num_weights * 16 * 2 / 3


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    node.model.weights = sparse.csc_matrix(node.model.weights)


def _flatten_model(root: Node) -> linear.FlatModel:
    """Flattens tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
        flat_model = _flatten_model(root)
    Args:
        root (Node): Root of the tree.

    Returns:
        linear.FlatModel: The flattened model.
    """
    weights = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        weights.append(node.model.__dict__.pop("weights"))

    root.dfs(visit)

    model = linear.FlatModel(
        name="flattened-tree",
        weights=sparse.hstack(weights, "csc"),
        bias=bias,
        thresholds=0,
        multiclass=False,
    )

    return model


def _tree_model(root: Node) -> TreeModel:
    """Constructs a tree model by aggregating the weights of all nodes in the tree.
    To speed up inference in Python, we avoid using a single flattened weight matrix,
    which would involve many unnecessary computations.
    Instead, we build a hierarchical tree model by aggregating the weights of each root's child
    into different flattened weight matrices, representing subtrees as `TreeModel` instances.
    Additionally, the root itself is also a `TreeModel`, containing subtree `TreeModel` instances.

    Consecutive values of the weight map denotes the start and end indices of the
    weights of each node. Conceptually, given root and node:
        slice = np.s_[weight_map[node.index]:
                      weight_map[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.

    Returns:
        Tree Model: A tree model containing the root's flattened model,
                   weight index mappings of all nodes, and subtrees.
    """
    # Build weights mapping which contains the start and end indices of the weights of each node.
    weight_map = [0]
    subtrees = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        # weights.shape[1] is the number of labels/metalabels of each node
        weight_map.append(node.model.weights.shape[1])

    root.dfs(visit)

    weight_map = np.cumsum(weight_map)

    # Build root's subtrees
    for child in root.children:
        child_flat_model = _flatten_model(child)
        subtrees.append(SubTree(child, child_flat_model))
    # Build root's flatten model with root model weights
    model = linear.FlatModel(
        name="root-flattened-tree",
        weights=root.model.__dict__.pop("weights"),
        bias=root.model.bias,
        thresholds=0,
        multiclass=False,
    )

    return TreeModel(root, model, weight_map, subtrees)
