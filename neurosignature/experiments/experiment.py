"""Experiment: multi-trial simulation and descriptor collection."""

import math
import numpy as np
from typing import Optional, Union

from neurosignature.inputs.input_generator import InputGenerator
from neurosignature.simulation.simulator import Simulator, BatchSimulator
from neurosignature.summaries.descriptors import (
    VectorDescriptor,
    BatchVectorDescriptor,
)


class Experiment:
    """Run many input trials through a simulator and collect descriptor vectors.

    Supports two execution paths, selected by the types of ``simulator``
    and ``descriptor`` passed at construction:

    **Serial path** (``Simulator`` + ``VectorDescriptor``):
        Loops over trials one at a time.  ``trials_per_batch`` is ignored.

    **Batch path** (``BatchSimulator`` + ``BatchVectorDescriptor``):
        Loops over chunks of ``trials_per_batch`` trials, passing each
        chunk to the batch simulator and batch descriptor in a single
        call.  This allows parallel simulation inside the user's callable.

    Args:
        input_gen: Input event stream generator.
        simulator: ``Simulator`` (serial) or ``BatchSimulator`` (batch).
        descriptor: ``VectorDescriptor`` (serial) or
            ``BatchVectorDescriptor`` (batch).
        trials_per_batch: Chunk size for the batch path.  Required when
            using ``BatchSimulator``; ignored otherwise.
    """

    def __init__(
        self,
        input_gen: InputGenerator,
        simulator: Union[Simulator, BatchSimulator],
        descriptor: Union[VectorDescriptor, BatchVectorDescriptor],
        trials_per_batch: Optional[int] = None,
    ):
        self.input_gen = input_gen
        self.simulator = simulator
        self.descriptor = descriptor
        self.trials_per_batch = trials_per_batch

        self._batch_mode = isinstance(simulator, BatchSimulator)

        if self._batch_mode and trials_per_batch is None:
            raise ValueError("trials_per_batch is required when using BatchSimulator.")
        if self._batch_mode and not isinstance(descriptor, BatchVectorDescriptor):
            raise TypeError("BatchSimulator requires a BatchVectorDescriptor.")
        if not self._batch_mode and not isinstance(descriptor, VectorDescriptor):
            raise TypeError("Simulator requires a VectorDescriptor.")

    def run(self, n_trials: int, duration_ms: float) -> np.ndarray:
        """Run the experiment and return all descriptor vectors.

        Args:
            n_trials: Total number of trials to run.
            duration_ms: Duration of each trial in milliseconds.

        Returns:
            Descriptor matrix, shape (n_trials, descriptor_dim).
        """
        if self._batch_mode:
            return self._run_batch(n_trials, duration_ms)
        return self._run_serial(n_trials, duration_ms)

    def _run_serial(self, n_trials: int, duration_ms: float) -> np.ndarray:
        results = []
        for _ in range(n_trials):
            ts_group = self.input_gen.generate(duration_ms)
            traces = self.simulator.run(ts_group)
            desc = self.descriptor.compute(traces)
            results.append(desc)
        return np.array(results)

    def _run_batch(self, n_trials: int, duration_ms: float) -> np.ndarray:
        n_chunks = math.ceil(n_trials / self.trials_per_batch)
        results = []
        remaining = n_trials
        for _ in range(n_chunks):
            chunk_size = min(self.trials_per_batch, remaining)
            ts_groups = self.input_gen.generate_batch(chunk_size, duration_ms)
            tsdtensor = self.simulator.run(ts_groups)
            chunk_descs = self.descriptor.compute(tsdtensor)
            results.append(chunk_descs)
            remaining -= chunk_size
        return np.vstack(results)

    def normalize(self, matrix: np.ndarray) -> tuple:
        """Z-score each descriptor column independently.

        Degenerate columns (std = 0) are returned as all-zeros rather
        than NaN.

        Args:
            matrix: Descriptor matrix, shape (n_trials, descriptor_dim).

        Returns:
            Tuple of ``(normalized_matrix, mean_vec, std_vec)`` where
            each vector has shape ``(descriptor_dim,)``.
        """
        mu = matrix.mean(axis=0)
        sigma = matrix.std(axis=0)
        safe_sigma = np.where(sigma == 0, 1.0, sigma)
        normed = (matrix - mu) / safe_sigma
        normed[:, sigma == 0] = 0.0
        return normed, mu, sigma
