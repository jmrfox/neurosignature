"""Simulator wrappers for user-defined system callables."""

import pynapple as nap
from typing import Callable, List


class Simulator:
    """Thin wrapper around a user-defined simulation callable.

    The callable must accept a single ``nap.TsGroup`` of input event
    streams and return a ``nap.TsdFrame`` of output voltage traces.

    Args:
        fn: Callable with signature ``fn(ts_group: TsGroup) -> TsdFrame``
    """

    def __init__(self, fn: Callable[[nap.TsGroup], nap.TsdFrame]):
        self.fn = fn

    def run(self, ts_group: nap.TsGroup) -> nap.TsdFrame:
        """Run simulation for one trial.

        Args:
            ts_group: Input event streams (one ``Ts`` per channel).

        Returns:
            ``TsdFrame`` of output voltage traces.
        """
        return self.fn(ts_group)


class BatchSimulator:
    """Thin wrapper around a batch-capable simulation callable.

    The callable accepts a list of ``nap.TsGroup`` objects (one per
    trial) and returns a ``nap.TsdTensor`` of output traces with shape
    (n_timesteps, n_outputs, n_trials), enabling parallel computation
    inside the user's function.

    Args:
        fn: Callable with signature
            ``fn(ts_groups: list[TsGroup]) -> TsdTensor``
    """

    def __init__(
        self,
        fn: Callable[[List[nap.TsGroup]], nap.TsdTensor],
    ):
        self.fn = fn

    def run(self, ts_groups: List[nap.TsGroup]) -> nap.TsdTensor:
        """Run simulation for a batch of trials.

        Args:
            ts_groups: List of input event streams, one per trial.

        Returns:
            ``TsdTensor`` of output traces for all trials.
        """
        return self.fn(ts_groups)
