"""Compare the original and staged tree assembly in separate processes.

Run from the repository root:
    PYTHONPATH=. python tests/linear/benchmark_tree_memory.py

The default synthetic model occupies about 387 MiB and needs comparable temporary
disk space. This measures weight construction/assembly, not full dataset training.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time

import numpy as np
import psutil
from scipy import sparse

from libmultilabel.linear import linear, tree


def benchmark(args):
    children = [tree.Node(np.arange(8 * i, 8 * (i + 1)), []) for i in range(args.nodes)]
    root = tree.Node(np.arange(8 * args.nodes), children)
    node_number = 0

    def train_node(node):
        nonlocal node_number
        columns = len(node.label_map) if node.isLeaf() else len(node.children)
        per_column = args.nnz_per_node // columns
        nnz = columns * per_column
        weights = sparse.csc_matrix(
            (
                np.full(nnz, node_number + 0.5),
                np.tile(np.arange(per_column, dtype=np.int32), columns),
                np.arange(columns + 1, dtype=np.int32) * per_column,
            ),
            shape=(args.nnz_per_node, columns),
        )
        node.model = linear.FlatModel("node", weights, -1, 0, False)
        node_number += 1

    baseline_rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    if args.mode == "original":
        root.dfs(train_node)
        blocks = []
        root.dfs(lambda node: blocks.append(node.model.__dict__.pop("weights")))
        node_ptr = np.cumsum([0] + [block.shape[1] for block in blocks])
        weights = sparse.hstack(blocks, format="csc")
        del blocks
    else:
        model, node_ptr = tree._flatten_model(root, train_node)
        weights = model.weights
    elapsed = time.perf_counter() - start

    # Hash buffers directly, without allocating another byte string or dense matrix.
    digest = hashlib.sha256()
    for array in (weights.data, weights.indices, weights.indptr, node_ptr):
        digest.update(memoryview(array))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return {
        "mode": args.mode,
        "nodes": args.nodes + 1,
        "nnz": weights.nnz,
        "model_mib": sum(a.nbytes for a in (weights.data, weights.indices, weights.indptr)) / 2**20,
        "baseline_rss_mib": baseline_rss / 2**20,
        "peak_rss_mib": peak / 2**20,
        "assembly_seconds": elapsed,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=128, help="Number of leaf nodes.")
    parser.add_argument("--nnz-per-node", type=int, default=262144)
    parser.add_argument("--mode", choices=("original", "staged"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.nodes <= 0 or args.nnz_per_node < max(args.nodes, 8):
        parser.error("Use positive node counts and at least max(nodes, 8) nonzeros per node.")
    if args.mode:
        print(json.dumps(benchmark(args)))
        return

    results = []
    for mode in ("original", "staged"):
        output = subprocess.check_output(
            [
                sys.executable,
                __file__,
                "--mode",
                mode,
                "--nodes",
                str(args.nodes),
                "--nnz-per-node",
                str(args.nnz_per_node),
            ],
            text=True,
        )
        results.append(json.loads(output))
    if results[0]["sha256"] != results[1]["sha256"]:
        raise AssertionError("The original and staged weight buffers differ.")
    print(json.dumps({"results": results, "identical_buffers": True}, indent=2))


if __name__ == "__main__":
    main()
