"""Regression tests for bounded-memory assembly of tree weights.

Run with: python -m unittest discover -s tests/linear -p 'test_*.py'
"""

import copy
import pickle
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import sparse

from libmultilabel.linear import linear, tree


def make_tree(bias=1, dtype=np.float64):
    branch = tree.Node(np.array([2, 0, 3]), [tree.Node(np.array([2, 0]), []), tree.Node(np.array([3]), [])])
    root = tree.Node(np.arange(4), [branch, tree.Node(np.array([1]), [])])
    root.is_root = True
    nodes = []
    root.dfs(nodes.append)
    rng = np.random.default_rng(42)
    for i, node in enumerate(nodes):
        columns = len(node.label_map) if node.isLeaf() else len(node.children)
        weights = rng.normal(size=(5, columns)).astype(dtype)
        weights[weights < 0] = 0
        if i == 2:
            weights[:] = 0
        node.model = linear.FlatModel("node", sparse.csc_matrix(weights), bias, 0, False)
    return root, nodes


def reference_flatten(root):
    """The original in-memory hstack path, retained only as a test oracle."""
    nodes = []
    root.dfs(nodes.append)
    weights = []
    for i, node in enumerate(nodes):
        node.index = i
        weights.append(node.model.__dict__.pop("weights"))
    flat = linear.FlatModel("flattened-tree", sparse.hstack(weights, format="csc"), root.model.bias, 0, False)
    return flat, np.cumsum([0] + [w.shape[1] for w in weights])


class TreeMemoryTests(unittest.TestCase):
    def assert_sparse_equal(self, actual, expected):
        self.assertEqual(actual.format, "csc")
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        assert_array_equal(actual.data, expected.data)
        assert_array_equal(actual.indices, expected.indices)
        assert_array_equal(actual.indptr, expected.indptr)

    def test_matches_hstack_and_predictions(self):
        for dtype in (np.float32, np.float64):
            for bias in (-1, 1):
                with self.subTest(dtype=dtype, bias=bias):
                    root, nodes = make_tree(bias, dtype)
                    expected_root = copy.deepcopy(root)
                    expected = tree.TreeModel(expected_root, *reference_flatten(expected_root))
                    # Force multiple chunks even for these small matrices.
                    with patch.object(tree, "_WEIGHT_CHUNK_SIZE", 2):
                        actual = tree.TreeModel(root, *tree._flatten_model(root))
                    self.assert_sparse_equal(actual.flat_model.weights, expected.flat_model.weights)
                    assert_array_equal(actual.node_ptr, expected.node_ptr)
                    self.assertEqual([node.index for node in nodes], list(range(len(nodes))))
                    self.assertTrue(all(not hasattr(node.model, "weights") for node in nodes))
                    x = sparse.csr_matrix(np.random.default_rng(7).normal(size=(6, 4 if bias > 0 else 5)))
                    for beam_width in (1, 2, 4):
                        assert_array_equal(actual.predict_values(x, beam_width), expected.predict_values(x, beam_width))

    def test_mixed_dtypes_and_noncanonical_entries(self):
        root, nodes = make_tree()
        nodes[1].model.weights = nodes[1].model.weights.astype(np.float32)
        # Duplicates, unsorted indices, and an explicit zero must be preserved.
        root.model.weights = sparse.csc_matrix(([2.0, 0.0, 3.0], [3, 1, 3], [0, 3, 3]), shape=(5, 2))
        expected, _ = reference_flatten(copy.deepcopy(root))
        actual, _ = tree._flatten_model(root)
        self.assert_sparse_equal(actual.weights, expected.weights)

    def test_empty_and_single_node_models(self):
        for shape in ((5, 0), (5, 3), (0, 3)):
            with self.subTest(shape=shape):
                root = tree.Node(np.arange(shape[1]), [])
                root.model = linear.FlatModel("node", sparse.csc_matrix(shape), -1, 0, False)
                flat, node_ptr = tree._flatten_model(root)
                self.assertEqual(flat.weights.shape, shape)
                self.assertEqual(flat.weights.nnz, 0)
                assert_array_equal(node_ptr, [0, shape[1]])

    def test_large_row_indices_use_int64(self):
        rows = np.iinfo(np.int32).max + 2
        root = tree.Node(np.arange(2), [])
        weights = sparse.csc_matrix(([1.5, 2.5], [0, rows - 1], [0, 1, 2]), shape=(rows, 2))
        root.model = linear.FlatModel("node", weights, -1, 0, False)
        flat, _ = tree._flatten_model(root)
        self.assert_sparse_equal(flat.weights, weights)
        self.assertEqual(flat.weights.indices.dtype, np.int64)
        self.assertEqual(flat.weights.indptr.dtype, np.int64)

    def test_training_releases_each_node_and_reads_bounded_chunks(self):
        root, nodes = make_tree()
        for node in nodes:
            del node.model
        refs = []
        visited = []
        original_load = np.load

        def train_node(node):
            self.assertTrue(all(ref() is None for ref in refs))
            visited.append(node)
            columns = len(node.label_map) if node.isLeaf() else len(node.children)
            weights = sparse.csc_matrix(np.ones((5, columns)))
            refs.extend(weakref.ref(array) for array in (weights.data, weights.indices, weights.indptr))
            node.model = linear.FlatModel("node", weights, 1, 0, False)

        def load_chunk(*args, **kwargs):
            self.assertEqual(visited, nodes)
            self.assertTrue(all(ref() is None for ref in refs))
            chunk = original_load(*args, **kwargs)
            self.assertLessEqual(chunk.size, 2)
            return chunk

        with patch.object(tree, "_WEIGHT_CHUNK_SIZE", 2), patch.object(tree.np, "load", side_effect=load_chunk):
            flat, _ = tree._flatten_model(root, train_node)
        assert_array_equal(flat.weights.toarray(), np.ones((5, 8)))

    def test_final_csc_reuses_allocated_buffers(self):
        root, _ = make_tree()
        original_csc = sparse.csc_matrix
        checked = []

        def create_csc(arg, *args, **kwargs):
            result = original_csc(arg, *args, **kwargs)
            if isinstance(arg, tuple) and len(arg) == 3:
                for original, final in zip(arg, (result.data, result.indices, result.indptr)):
                    self.assertTrue(np.shares_memory(original, final))
                checked.append(True)
            return result

        with patch.object(tree.sparse, "csc_matrix", side_effect=create_csc):
            tree._flatten_model(root)
        self.assertEqual(checked, [True])

    def test_temporary_file_closes_on_success_and_failure(self):
        for failure in (None, "train", "write", "read"):
            with self.subTest(failure=failure):
                root, _ = make_tree()
                staging_file = tempfile.TemporaryFile()

                def train_node(node):
                    if failure == "train" and node is not root:
                        raise RuntimeError("training interrupted")

                with patch.object(tree.tempfile, "TemporaryFile", return_value=staging_file):
                    if failure in ("write", "read"):
                        operation = "save" if failure == "write" else "load"
                        with patch.object(tree.np, operation, side_effect=OSError("disk failure")):
                            with self.assertRaisesRegex(OSError, "disk failure"):
                                tree._flatten_model(root, train_node)
                    elif failure == "train":
                        with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                            tree._flatten_model(root, train_node)
                    else:
                        model = tree.TreeModel(root, *tree._flatten_model(root, train_node))
                self.assertTrue(staging_file.closed)
                if failure is None:
                    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
                    self.assert_sparse_equal(restored.flat_model.weights, model.flat_model.weights)
                    x = sparse.csr_matrix(np.ones((2, 4)))
                    assert_array_equal(restored.predict_values(x), model.predict_values(x))

    def test_real_training_matches_original_assembly(self):
        rng = np.random.default_rng(6)
        x = sparse.csr_matrix(rng.normal(size=(32, 4)))
        y = sparse.csr_matrix(rng.integers(0, 2, size=(32, 4)))
        root, nodes = make_tree()
        for node in nodes:
            del node.model

        def old_assembly(root, train_node):
            root.dfs(train_node)
            return reference_flatten(root)

        # A primal solver avoids differences from LIBLINEAR's random dual updates.
        with patch.object(tree, "_flatten_model", side_effect=old_assembly):
            expected = tree.train_tree(y, x, options="-s 2 -B 1 -m 1 -q", root=copy.deepcopy(root), verbose=False)
        actual = tree.train_tree(y, x, options="-s 2 -B 1 -m 1 -q", root=root, verbose=False)
        assert_allclose(
            actual.flat_model.weights.toarray(), expected.flat_model.weights.toarray(), rtol=1e-12, atol=1e-12
        )
        assert_array_equal(actual.node_ptr, expected.node_ptr)
        for beam_width in (1, 3):
            assert_allclose(actual.predict_values(x, beam_width), expected.predict_values(x, beam_width), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
