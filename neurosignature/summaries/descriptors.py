"""VectorDescriptor and BatchVectorDescriptor for assembling descriptor vectors."""

import numpy as np
import pynapple as nap
from typing import List
from .scalar_descriptors import ScalarDescriptor, BatchScalarDescriptor


class VectorDescriptor:
    """Assemble a descriptor vector from an ordered list of ScalarDescriptors.

    Each element of the descriptor vector is the scalar produced by one
    ``ScalarDescriptor`` applied to the output ``TsdFrame`` of a single trial.
    A reference channel is extracted from ``traces`` and passed as ``v_ref``
    to every descriptor's ``compute`` method.

    Args:
        descriptors: Ordered list of ``ScalarDescriptor`` instances.
        reference_channel: Index of the output channel used as reference
            trace ``v_ref``.  Defaults to 0.
    """

    def __init__(
        self,
        descriptors: List[ScalarDescriptor],
        reference_channel: int = 0,
    ):
        self.descriptors = descriptors
        self.reference_channel = reference_channel

    def compute(self, traces: nap.TsdFrame) -> np.ndarray:
        """Compute descriptor vector for one trial.

        Args:
            traces: Output voltage traces, shape (n_timesteps, n_outputs).

        Returns:
            Descriptor vector, shape (len(descriptors),).
        """
        arr = np.asarray(traces)
        v_ref = arr[:, self.reference_channel]
        return np.array([d.compute(traces, v_ref) for d in self.descriptors])

    def labels(self) -> list:
        """Return the label of each constituent descriptor in order.

        Returns:
            List of label strings, length ``len(self.descriptors)``.
        """
        return [d.label for d in self.descriptors]

    def __len__(self) -> int:
        return len(self.descriptors)


class BatchVectorDescriptor:
    """Assemble descriptor vectors for a batch of trials simultaneously.

    Each element of the descriptor vector is a ``BatchScalarDescriptor``
    whose ``compute_batch`` method is called with the full ``TsdTensor``
    of trial outputs and the reference-channel index.

    Args:
        descriptors: Ordered list of ``BatchScalarDescriptor`` instances.
        reference_channel: Index of the output channel used as reference
            trace.  Defaults to 0.
    """

    def __init__(
        self,
        descriptors: List[BatchScalarDescriptor],
        reference_channel: int = 0,
    ):
        self.descriptors = descriptors
        self.reference_channel = reference_channel

    def compute(self, tsdtensor: nap.TsdTensor) -> np.ndarray:
        """Compute descriptor vectors for all trials in a batch.

        Args:
            tsdtensor: Output traces for all trials,
                shape (n_timesteps, n_outputs, n_trials).

        Returns:
            Descriptor matrix, shape (n_trials, len(descriptors)).
        """
        columns = [
            d.compute_batch(tsdtensor, self.reference_channel) for d in self.descriptors
        ]
        return np.column_stack(columns)

    def __len__(self) -> int:
        return len(self.descriptors)
