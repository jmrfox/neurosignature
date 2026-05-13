"""Poisson process input generation with routing."""

import numpy as np
import pynapple as nap
from typing import Optional, List


class PoissonGenerator:
    """Master Poisson process with channel routing.

    Generates a global Poisson event stream N(t) ~ Poisson(lambda_max)
    and routes each event to an input channel according to probability
    distribution P ∈ R^(N_S) where sum(P_i) = 1.

    Args:
        n_channels: Number of input channels (N_S)
        lambda_max: Master Poisson rate in Hz (events per second)
        routing_probs: Channel routing probabilities, shape (n_channels,).
            If None, uses uniform distribution.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_channels: int,
        lambda_max: float = 100.0,
        routing_probs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.n_channels = n_channels
        self.lambda_max = lambda_max
        self.rng = np.random.default_rng(seed)

        # Set routing probabilities
        if routing_probs is None:
            self.routing_probs = np.ones(n_channels) / n_channels
        else:
            self.routing_probs = np.asarray(routing_probs)
            assert len(self.routing_probs) == n_channels
            assert np.abs(np.sum(self.routing_probs) - 1.0) < 1e-10

    def generate_events(
        self,
        duration_ms: float,
        dt_ms: float = 1.0,
    ) -> nap.TsGroup:
        """Generate Poisson events for all channels.

        Args:
            duration_ms: Total simulation duration in milliseconds
            dt_ms: Time step in milliseconds

        Returns:
            TsGroup with one Ts per channel. Timestamps are in ms
            (time_units="ms" at construction; internally stored in seconds
            by pynapple).
        """
        # Total simulation time in seconds for rate calculation
        duration_s = duration_ms / 1000.0

        # Generate master Poisson events
        # Expected number of events
        n_expected = int(self.lambda_max * duration_s * 1.5)  # Buffer

        # Inter-event times are exponential
        inter_event_times = self.rng.exponential(1.0 / self.lambda_max, size=n_expected)
        event_times_s = np.cumsum(inter_event_times)

        # Keep only events within duration
        event_times_s = event_times_s[event_times_s < duration_s]
        event_times_ms = event_times_s * 1000.0

        # Route events to channels
        channel_events: List[List[float]] = [[] for _ in range(self.n_channels)]
        for t in event_times_ms:
            channel = self.rng.choice(self.n_channels, p=self.routing_probs)
            channel_events[channel].append(t)

        # Build TsGroup: one Ts per channel, times in ms
        epoch = nap.IntervalSet(start=0.0, end=duration_ms, time_units="ms")
        ts_dict = {
            ch: nap.Ts(t=np.array(events), time_units="ms")
            for ch, events in enumerate(channel_events)
        }
        return nap.TsGroup(ts_dict, time_support=epoch)

    def generate_spike_train(
        self,
        duration_ms: float,
        dt_ms: float = 1.0,
    ) -> nap.TsdFrame:
        """Generate binary spike train matrix.

        Args:
            duration_ms: Duration in milliseconds
            dt_ms: Time step in milliseconds

        Returns:
            TsdFrame of shape (n_timesteps, n_channels) with binary spike
            indicators. Time index is in ms (time_units="ms" at construction;
            internally stored in seconds by pynapple).
        """
        n_steps = int(duration_ms / dt_ms)
        spike_train = np.zeros((n_steps, self.n_channels), dtype=np.int32)

        ts_group = self.generate_events(duration_ms, dt_ms)

        for channel_idx in range(self.n_channels):
            ts = ts_group[channel_idx]
            event_times_ms = ts.index * 1000.0  # pynapple stores in seconds
            for t in event_times_ms:
                step = int(t / dt_ms)
                if step < n_steps:
                    spike_train[step, channel_idx] = 1

        t_ms = np.arange(n_steps, dtype=float) * dt_ms
        return nap.TsdFrame(
            t=t_ms,
            d=spike_train,
            time_units="ms",
        )

    def get_event_count_stats(self, events: List[np.ndarray]) -> dict:
        """Compute event count statistics.

        Args:
            events: List of event arrays per channel

        Returns:
            Dictionary with count statistics
        """
        counts = [len(e) for e in events]
        return {
            "total_events": sum(counts),
            "mean_per_channel": np.mean(counts),
            "std_per_channel": np.std(counts),
            "min_per_channel": min(counts),
            "max_per_channel": max(counts),
        }
