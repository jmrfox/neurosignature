"""Synaptic kernel for converting events to smooth currents."""

import numpy as np
from typing import List, Optional
from neurosignature.math import alpha_kernel as _alpha_kernel


class SynapticKernel:
    """Alpha-function synaptic kernel.

    Converts discrete event timestamps into smooth input currents using:
    α(t) = H(t) · (t/τ_s) · exp(-t/τ_s)

    where H(t) is the Heaviside step function and τ_s is the synaptic decay.

    Args:
        tau_s: Synaptic decay constant in milliseconds (default: 10.0)
        dt_ms: Simulation time step in milliseconds (default: 1.0)
    """

    def __init__(
        self,
        tau_s: float = 10.0,
        dt_ms: float = 1.0,
    ):
        self.tau_s = tau_s
        self.dt_ms = dt_ms

    def kernel(self, t_ms: np.ndarray) -> np.ndarray:
        """Compute alpha kernel values at given times.

        Args:
            t_ms: Time array in milliseconds (should be >= 0)

        Returns:
            Kernel values at each time point
        """
        return _alpha_kernel(t_ms, self.tau_s)

    def convolve_events(
        self,
        event_times: np.ndarray,
        duration_ms: float,
    ) -> np.ndarray:
        """Convolve events with kernel to produce smooth current.

        Args:
            event_times: Array of event times in milliseconds
            duration_ms: Total duration in milliseconds

        Returns:
            Current trace array, shape (n_timesteps,)
        """
        n_steps = int(duration_ms / self.dt_ms)
        current = np.zeros(n_steps)

        # Time grid
        t_grid = np.arange(n_steps) * self.dt_ms

        for event_time in event_times:
            # Compute kernel contribution for this event
            dt = t_grid - event_time
            contribution = self.kernel(dt)
            current += contribution

        return current

    def generate_input_currents(
        self,
        channel_events: List[np.ndarray],
        duration_ms: float,
    ) -> np.ndarray:
        """Generate input currents for all channels.

        Args:
            channel_events: List of event time arrays per channel
            duration_ms: Total duration in milliseconds

        Returns:
            Input current matrix, shape (n_timesteps, n_channels)
        """
        n_channels = len(channel_events)
        n_steps = int(duration_ms / self.dt_ms)
        currents = np.zeros((n_steps, n_channels))

        for i, events in enumerate(channel_events):
            if len(events) > 0:
                currents[:, i] = self.convolve_events(events, duration_ms)

        return currents

    def get_kernel_duration(self, threshold: float = 0.01) -> float:
        """Get effective kernel duration until it decays below threshold.

        Args:
            threshold: Amplitude threshold relative to peak

        Returns:
            Duration in milliseconds
        """
        # Peak of alpha function is at t = tau_s
        peak_value = 1.0 * np.exp(-1.0)  # (tau/tau) * exp(-1) = exp(-1)

        # Find when kernel drops below threshold
        t_test = np.linspace(0, 10 * self.tau_s, 10000)
        kernel_vals = self.kernel(t_test)
        below_threshold = kernel_vals < (peak_value * threshold)

        if np.any(below_threshold):
            cutoff_idx = np.where(below_threshold)[0][0]
            return t_test[cutoff_idx]
        return 10 * self.tau_s

    def get_peak_amplitude(self) -> float:
        """Get peak amplitude of the kernel (at t = tau_s)."""
        return 1.0 * np.exp(-1.0)
