from __future__ import annotations

import os
import re
import logging
import threading
import queue
from tqdm import tqdm

import numpy as np
import scipy.sparse as sparse
from liblinear.liblinearutil import train, parameter, problem, solver_names

from ctypes import c_double


class ParallelTrainer(threading.Thread):
    """A trainer for parallel 1vsrest training."""

    y: sparse.csc_matrix
    x: sparse.csr_matrix
    prob: problem
    param: parameter
    weights: np.ndarray
    pbar: tqdm
    queue: queue.SimpleQueue

    def __init__(self):
        threading.Thread.__init__(self)

    @classmethod
    def init_trainer(
        cls,
        y: sparse.csc_matrix,
        x: sparse.csr_matrix,
        options: str,
        verbose: bool,
    ):
        """Initialize the parallel trainer by setting y, x, parameter and threading related
        variables as class variable of ParallelTrainer.

        Args:
            y (sparse.csc_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
            x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
            options (str): The option string passed to liblinear.
            verbose (bool): Output extra progress information.
        """
        cls.y = y.tocsc()
        cls.x = x
        num_instances, num_classes = cls.y.shape
        num_features = cls.x.shape[1]
        cls.prob = problem(np.ones((num_instances,)), cls.x)
        cls.param = parameter(re.sub(r"-m\s+\d+", "", options))
        if cls.param.solver_type in [solver_names.L2R_L1LOSS_SVC_DUAL, solver_names.L2R_L2LOSS_SVC_DUAL]:
            cls.param.w_recalc = True  # only works for solving L1/L2-SVM dual
        cls.weights = np.zeros((num_features, num_classes), order="F")
        cls.pbar = tqdm(total=num_classes, disable=not verbose)
        cls.queue = queue.SimpleQueue()

        if verbose:
            logging.info(f"Training one-vs-rest model on {num_classes} labels")
        for i in range(num_classes):
            cls.queue.put(i)

    def _do_parallel_train(self, y: np.ndarray) -> np.matrix:
        """Wrap around liblinear.liblinearutil.train.

        Args:
            y (np.ndarray): A +1/-1 array with dimensions number of instances * 1.

        Returns:
            np.matrix: The weights.
        """
        if y.shape[0] == 0:
            return np.matrix(np.zeros((self.prob.n, 1)))

        prob = self.prob.copy()
        prob.y = (c_double * prob.l)(*y)
        model = train(prob, self.param)

        w = np.ctypeslib.as_array(model.w, (self.prob.n, 1))
        w = np.asmatrix(w)
        # When all labels are -1, we must flip the sign of the weights
        # because LIBLINEAR treats the first label as positive, which
        # is -1 in this case. But for our usage we need them to be negative.
        # For labels with both +1 and -1, LIBLINEAR guarantees that +1
        # is always the first label.
        if model.get_labels()[0] == -1:
            return -w
        else:
            # The memory is freed on model deletion so we make a copy.
            return w.copy()

    def run(self):
        while True:
            try:
                label_idx = self.queue.get_nowait()
            except queue.Empty:
                break

            yi = self.y[:, label_idx].toarray().reshape(-1)
            self.weights[:, label_idx] = self._do_parallel_train(2 * yi - 1).ravel()

            self.pbar.update()


def train_parallel_1vsrest(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str,
    verbose: bool,
) -> np.matrix:
    """Parallel training on labels when using one-vs-rest strategy.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        verbose (bool): Output extra progress information.

    Returns:
        np.matrix: The weights.
    """
    ParallelTrainer.init_trainer(y, x, options, verbose)
    num_threads = int(os.cpu_count() / 2)
    trainers = [ParallelTrainer() for _ in range(num_threads)]

    for trainer in trainers:
        trainer.start()
    for trainer in trainers:
        trainer.join()

    ParallelTrainer.pbar.close()
    return ParallelTrainer.weights
