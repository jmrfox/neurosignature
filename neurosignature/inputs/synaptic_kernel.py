"""Synaptic kernel for converting events to smooth currents."""

import numpy as np
import pynapple as nap
from typing import List, Union
from neurosignature.math import alpha_kernel as _alpha_kernel


class SynapticKernel:
    """Alpha-function synaptic kernel.

    Converts discrete event timestamps into smooth input currents using:
    α(t) = H(t) · (t/τ) · exp(-t/τ)

    where H(t) is the Heaviside step function and τ is the synaptic decay.

    Args:
        tau_ms: Synaptic decay constant in milliseconds (default: 10.0)
        dt_ms: Simulation time step in milliseconds (default: 1.0)
    """

    def __init__(
        self,
        tau_ms: float = 10.0,
        dt_ms: float = 1.0,
    ):
        self.tau_ms = tau_ms
        self.dt_ms = dt_ms

    def kernel(self, t_ms: np.ndarray) -> np.ndarray:
        """Compute alpha kernel values at given times.

        Args:
            t_ms: Time array in milliseconds (should be >= 0)

        Returns:
            Kernel values at each time point
        """
        return _alpha_kernel(t_ms, self.tau_ms)

    def convolve_events(
        self,
        event_times: Union[np.ndarray, nap.Ts],
        duration_ms: float,
    ) -> np.ndarray:
        """Convolve events with kernel to produce smooth current.

        Args:
            event_times: Event times in milliseconds.  May be a plain
                numpy array or a pynapple ``Ts`` object (whose index is
                stored in seconds internally and converted here).
            duration_ms: Total duration in milliseconds

        Returns:
            Current trace array, shape (n_timesteps,)
        """
        if isinstance(event_times, nap.Ts):
            times_ms = event_times.index * 1000.0
        else:
            times_ms = np.asarray(event_times, dtype=float)

        n_steps = int(duration_ms / self.dt_ms)
        current = np.zeros(n_steps)

        # Time grid
        t_grid = np.arange(n_steps) * self.dt_ms

        for event_time in times_ms:
            # Compute kernel contribution for this event
            dt = t_grid - event_time
            contribution = self.kernel(dt)
            current += contribution

        return current

    def generate_input_currents(
        self,
        channel_events: Union[nap.TsGroup, List[np.ndarray]],
        duration_ms: float,
    ) -> nap.TsdFrame:
        """Generate input currents for all channels.

        Args:
            channel_events: Either a pynapple ``TsGroup`` (one unit per
                channel, as returned by ``PoissonGenerator.generate_events``)
                or a list of numpy arrays of event times in milliseconds.
            duration_ms: Total duration in milliseconds

        Returns:
            ``TsdFrame`` of shape (n_timesteps, n_channels) with smooth
            input currents in nA. Time index is in ms at construction
            (internally stored in seconds by pynapple).
        """
        if isinstance(channel_events, nap.TsGroup):
            keys = sorted(channel_events.keys())
            n_channels = len(keys)
        else:
            keys = list(range(len(channel_events)))
            n_channels = len(channel_events)

        n_steps = int(duration_ms / self.dt_ms)
        currents = np.zeros((n_steps, n_channels))

        for col_idx, key in enumerate(keys):
            if isinstance(channel_events, nap.TsGroup):
                events = channel_events[key]
            else:
                events = channel_events[key]
            if len(events) > 0:
                currents[:, col_idx] = self.convolve_events(events, duration_ms)

        t_ms = np.arange(n_steps, dtype=float) * self.dt_ms
        return nap.TsdFrame(t=t_ms, d=currents, time_units="ms")

    def get_kernel_duration(self, threshold: float = 0.01) -> float:
        """Get effective kernel duration until it decays below threshold.

        Args:
            threshold: Amplitude threshold relative to peak

        Returns:
            Duration in milliseconds
        """
        # Peak of alpha function is at t = tau_ms
        peak_value = 1.0 * np.exp(-1.0)  # (tau/tau) * exp(-1) = exp(-1)

        # Find when kernel drops below threshold
        t_test = np.linspace(0, 10 * self.tau_ms, 10000)
        kernel_vals = self.kernel(t_test)
        below_threshold = kernel_vals < (peak_value * threshold)

        if np.any(below_threshold):
            cutoff_idx = np.where(below_threshold)[0][0]
            return t_test[cutoff_idx]
        return 10 * self.tau_ms

    def get_peak_amplitude(self) -> float:
        """Get peak amplitude of the kernel (at t = tau_ms)."""
        return 1.0 * np.exp(-1.0)
