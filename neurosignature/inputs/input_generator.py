"""Unified input generation combining Poisson spikes with synaptic kernels."""

import numpy as np
import pynapple as nap
from typing import Optional, Tuple
from .poisson_generator import PoissonGenerator
from .synaptic_kernel import SynapticKernel


class InputGenerator:
    """Generate smooth synaptic input currents from Poisson spike trains.

    Combines Poisson event generation with synaptic kernel convolution
    for a one-step input generation pipeline.

    Args:
        n_channels: Number of input channels
        rate_hz: Poisson firing rate in Hz (default: 100.0)
        tau_ms: Synaptic decay time constant in ms (default: 10.0)
        dt_ms: Simulation time step in ms (default: 1.0)
        routing_probs: Channel routing probabilities. If None, uniform.
        seed: Random seed for reproducibility

    Example:
        >>> gen = InputGenerator(n_channels=5, rate_hz=50.0, tau_ms=5.0)
        >>> currents = gen.generate(duration_ms=1000.0)
        >>> currents.shape
        (1000, 5)
    """

    def __init__(
        self,
        n_channels: int,
        rate_hz: float = 100.0,
        tau_ms: float = 10.0,
        dt_ms: float = 1.0,
        routing_probs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.n_channels = n_channels
        self.dt_ms = dt_ms

        # Create internal generators
        self.poisson = PoissonGenerator(
            n_channels=n_channels,
            lambda_max=rate_hz,
            routing_probs=routing_probs,
            seed=seed,
        )
        self.kernel = SynapticKernel(tau_ms=tau_ms, dt_ms=dt_ms)

    def generate(self, duration_ms: float) -> nap.TsdFrame:
        """Generate smooth input currents.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            ``TsdFrame`` of shape (n_timesteps, n_channels) with smooth
            input currents in nA.
        """
        # Generate events
        events = self.poisson.generate_events(duration_ms, self.dt_ms)

        # Convolve with kernel
        currents = self.kernel.generate_input_currents(events, duration_ms)

        return currents

    def generate_with_spikes(
        self, duration_ms: float
    ) -> Tuple[nap.TsdFrame, nap.TsdFrame]:
        """Generate both spike trains and smoothed currents.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            (spike_train, currents) tuple:
            - spike_train: ``TsdFrame`` binary (n_timesteps, n_channels)
            - currents: ``TsdFrame`` smooth currents (n_timesteps, n_channels)
        """
        ts_group = self.poisson.generate_events(duration_ms, self.dt_ms)
        spike_train = self.poisson.generate_spike_train(duration_ms, self.dt_ms)
        currents = self.kernel.generate_input_currents(ts_group, duration_ms)

        return spike_train, currents
