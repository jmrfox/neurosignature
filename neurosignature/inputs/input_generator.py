"""Poisson input event stream generation."""

import numpy as np
import pynapple as nap
from typing import List, Optional
from .poisson_generator import PoissonGenerator


class InputGenerator:
    """Generate Poisson input event streams for simulation.

    Wraps PoissonGenerator to produce pynapple TsGroup objects
    (one spike-timestamp stream per input channel) ready for
    direct consumption by a Simulator.

    Args:
        n_channels: Number of input channels
        rate_hz: Poisson firing rate in Hz (default: 100.0)
        dt_ms: Time step in ms, used for event generation (default: 1.0)
        routing_probs: Channel routing probabilities. If None, uniform.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_channels: int,
        rate_hz: float = 100.0,
        dt_ms: float = 1.0,
        routing_probs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.n_channels = n_channels
        self.dt_ms = dt_ms

        self.poisson = PoissonGenerator(
            n_channels=n_channels,
            lambda_max=rate_hz,
            routing_probs=routing_probs,
            seed=seed,
        )

    def generate(self, duration_ms: float) -> nap.TsGroup:
        """Generate one trial of input event streams.

        Args:
            duration_ms: Trial duration in milliseconds

        Returns:
            ``TsGroup`` with one ``Ts`` per input channel containing
            spike timestamps.
        """
        return self.poisson.generate_events(duration_ms, self.dt_ms)

    def generate_batch(self, n_trials: int, duration_ms: float) -> List[nap.TsGroup]:
        """Generate a batch of independent input trials.

        Args:
            n_trials: Number of trials to generate
            duration_ms: Duration of each trial in milliseconds

        Returns:
            List of ``TsGroup`` objects, one per trial.
        """
        return [self.generate(duration_ms) for _ in range(n_trials)]
