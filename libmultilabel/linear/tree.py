from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sparse
from sparsekmeans import LloydKmeans, ElkanKmeans
import sklearn.preprocessing
from tqdm import tqdm
import psutil

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
        node_ptr: np.ndarray,
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.node_ptr = node_ptr
        self.multiclass = False
        self._model_separated = False # Indicates whether the model has been separated for pruning tree.

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
    ) -> np.ndarray:
        """Calculate the probability estimates associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        if beam_width >= len(self.root.children):
            # Beam_width is sufficiently large; pruning not applied.
            # Calculates decision values for all nodes.
            all_preds = linear.predict_values(self.flat_model, x) # number of instances * (number of labels + total number of metalabels)
        else:
            # Beam_width is small; pruning applied to reduce computation.
            if not self._model_separated:
                self._separate_model_for_pruning_tree()
                self._model_separated = True
            all_preds = self._prune_tree_and_predict_values(x, beam_width) # number of instances * (number of labels + total number of metalabels)
        return np.vstack([self._beam_search(all_preds[i], beam_width) for i in range(all_preds.shape[0])])

    def _separate_model_for_pruning_tree(self):
        """
        This function separates the weights for the root node and its children into (K+1) FlatModel
        for efficient beam search traversal in Python.
        """
        tree_flat_model_params = {
            'bias': self.root.model.bias,
            'thresholds': 0,
            'multiclass': False
        }
        slice = np.s_[:, self.node_ptr[self.root.index] : self.node_ptr[self.root.index + 1]]
        self.root_model = linear.FlatModel(
            name="root-flattened-tree",
            weights=self.flat_model.weights[slice].tocsr(),
            **tree_flat_model_params
        )

        self.subtree_models = []
        for i in range(len(self.root.children)):
            subtree_weights_start = self.node_ptr[self.root.children[i].index]
            subtree_weights_end = self.node_ptr[self.root.children[i+1].index] if i+1 < len(self.root.children) else self.node_ptr[-1]
            slice = np.s_[:, subtree_weights_start:subtree_weights_end]
            subtree_flatmodel = linear.FlatModel(
                name="subtree-flattened-tree",
                weights=self.flat_model.weights[slice].tocsr(),
                **tree_flat_model_params
            )
            self.subtree_models.append(subtree_flatmodel)
        
    def _prune_tree_and_predict_values(self, x: sparse.csr_matrix, beam_width: int) -> np.ndarray:
        """Calculates the selective decision values associated with instances x by evaluating only the most relevant subtrees.

        Only subtrees corresponding to the top beam_width candidates from the root are evaluated,
        skipping the rest to avoid unnecessary computation.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int): Number of top candidate branches considered for prediction.

        Returns:
            np.ndarray: A matrix with dimension number of instances * (number of labels + total number of metalabels).
        """
        # Initialize space for all predictions with negative infinity
        num_instances, num_labels = x.shape[0], self.node_ptr[-1]
        all_preds = np.full((num_instances, num_labels), -np.inf)

        # Calculate root decision values and scores
        root_preds = linear.predict_values(self.root_model, x)
        children_scores = 0.0 - np.square(np.maximum(0, 1 - root_preds))

        slice = np.s_[:, self.node_ptr[self.root.index] : self.node_ptr[self.root.index + 1]]
        all_preds[slice] = root_preds

        # Select indices of the top beam_width subtrees for each instance
        top_beam_width_indices = np.argsort(-children_scores, axis=1, kind="stable")[:, :beam_width]

        # Build a mask where mask[i, j] is True if the j-th subtree is among the top beam_width subtrees for the i-th instance
        mask = np.zeros_like(children_scores, dtype=np.bool_)
        np.put_along_axis(mask, top_beam_width_indices, True, axis=1)
        
        # Calculate predictions for each subtree with its corresponding instances
        for subtree_idx in range(len(self.root.children)):
            subtree_model = self.subtree_models[subtree_idx]
            instances_mask = mask[:, subtree_idx]
            reduced_instances = x[np.s_[instances_mask], :]

            # Locate the position of the subtree root in the weight mapping of all nodes
            subtree_weights_start = self.node_ptr[self.root.children[subtree_idx].index]
            subtree_weights_end = subtree_weights_start + subtree_model.weights.shape[1]

            slice = np.s_[instances_mask, subtree_weights_start:subtree_weights_end]
            all_preds[slice] = linear.predict_values(subtree_model, reduced_instances)

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
                slice = np.s_[self.node_ptr[node.index] : self.node_ptr[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score - np.square(np.maximum(0, 1 - pred))
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        scores = np.zeros(num_labels)
        for node, score in cur_level:
            slice = np.s_[self.node_ptr[node.index] : self.node_ptr[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score - np.square(np.maximum(0, 1 - pred)))
        return scores


def train_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> TreeModel:
    """Train a linear model for multi-label data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        TreeModel: A model which can be used in predict_values.
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

    def visit(node):
        if node.is_root:
            _train_node(y, x, options, node)
        else:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
            _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, node_ptr = _flatten_model(root)
    return TreeModel(root, flat_model, node_ptr)


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Build the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: Root of the (sub)tree built from label_representation.
    """
    children = []
    if d < dmax and label_representation.shape[0] > K:
        if label_representation.shape[0] > 10000:
            kmeans_algo = ElkanKmeans
        else:
            kmeans_algo = LloydKmeans

        kmeans = kmeans_algo(
            n_clusters=K, max_iter=300, tol=0.0001, random_state=np.random.randint(2**31 - 1), verbose=True
        )
        metalabels = kmeans.fit(label_representation)

        unique_labels = np.unique(metalabels)
        if len(unique_labels) == K:
            create_child_node = lambda i: _build_tree(
                label_representation[metalabels == i], label_map[metalabels == i], d + 1, K, dmax
            )
        else:
            create_child_node = lambda i: Node(label_map=label_map[metalabels == i], children=[])

        for i in range(K):
            child = create_child_node(i)
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
    """If node is internal, compute the metalabels representing each child and train
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


def _flatten_model(root: Node) -> tuple[linear.FlatModel, np.ndarray]:
    """Flatten tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
    Consecutive values of the returned array denote the start and end indices of each node in the tree.
    To extract a node's classifiers:
        slice = np.s_[node_ptr[node.index]:
                      node_ptr[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    index = 0
    weights = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        nonlocal index
        node.index = index
        index += 1
        weights.append(node.model.__dict__.pop("weights"))

    root.dfs(visit)

    model = linear.FlatModel(
        name="flattened-tree",
        weights=sparse.hstack(weights, "csc"),
        bias=bias,
        thresholds=0,
        multiclass=False,
    )

    # w.shape[1] is the number of labels/metalabels of each node
    node_ptr = np.cumsum([0] + list(map(lambda w: w.shape[1], weights)))

    return model, node_ptr
