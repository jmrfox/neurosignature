"""Tests for input generation module."""

import numpy as np
from neurosignature.inputs import PoissonGenerator, SynapticKernel


def test_poisson_generator_init():
    """Test Poisson generator initialization."""
    gen = PoissonGenerator(n_channels=25, lambda_max=100.0, seed=42)
    assert gen.n_channels == 25
    assert gen.lambda_max == 100.0
    assert len(gen.routing_probs) == 25
    np.testing.assert_almost_equal(np.sum(gen.routing_probs), 1.0)


def test_poisson_generator_custom_routing():
    """Test Poisson generator with custom routing."""
    probs = np.array([0.5, 0.3, 0.2])
    gen = PoissonGenerator(n_channels=3, routing_probs=probs, seed=42)
    np.testing.assert_array_almost_equal(gen.routing_probs, probs)


def test_event_generation():
    """Test event generation produces reasonable counts."""
    gen = PoissonGenerator(n_channels=10, lambda_max=100.0, seed=42)
    events = gen.generate_events(duration_ms=1000.0, dt_ms=1.0)

    assert len(events) == 10

    stats = gen.get_event_count_stats(events)
    # Expected ~100 events per second over 1 second = ~100 events total
    # Distributed across 10 channels = ~10 per channel
    assert 50 < stats["total_events"] < 150  # Allow variance


def test_spike_train_shape():
    """Test spike train has correct shape."""
    gen = PoissonGenerator(n_channels=5, lambda_max=50.0, seed=42)
    spike_train = gen.generate_spike_train(duration_ms=100.0, dt_ms=1.0)
    assert spike_train.shape == (100, 5)
    assert np.all((spike_train == 0) | (spike_train == 1))


def test_synaptic_kernel_init():
    """Test synaptic kernel initialization."""
    kernel = SynapticKernel(tau_s=10.0, dt_ms=1.0)
    assert kernel.tau_s == 10.0
    assert kernel.dt_ms == 1.0


def test_kernel_shape():
    """Test alpha kernel has correct shape."""
    kernel = SynapticKernel(tau_s=10.0)
    t = np.linspace(0, 100, 200)
    alpha = kernel.kernel(t)

    # Peak should be at t = tau_s
    peak_idx = np.argmax(alpha)
    assert abs(t[peak_idx] - 10.0) < 0.6  # Within half time step

    # Should decay to near zero (at t=100 with tau=10, alpha ~ 0.0004)
    assert alpha[-1] < 0.01


def test_kernel_causality():
    """Test kernel is causal (zero for t < 0)."""
    kernel = SynapticKernel(tau_s=10.0)
    t = np.linspace(-20, 50, 100)
    alpha = kernel.kernel(t)
    assert np.all(alpha[t < 0] == 0)


def test_convolve_events():
    """Test event convolution produces smooth current."""
    kernel = SynapticKernel(tau_s=10.0, dt_ms=1.0)
    events = np.array([10.0, 50.0, 80.0])  # Events at 10, 50, 80 ms
    current = kernel.convolve_events(events, duration_ms=100.0)

    assert len(current) == 100
    assert np.all(current >= 0)
    assert np.all(np.isfinite(current))

    # Should have peaks near event times
    peaks = np.where(current > np.percentile(current, 95))[0]
    assert len(peaks) > 0


def test_generate_input_currents():
    """Test multi-channel input current generation."""
    kernel = SynapticKernel(tau_s=10.0, dt_ms=1.0)

    # Create events for 3 channels
    events = [
        np.array([10.0, 30.0]),
        np.array([20.0]),
        np.array([]),  # No events
    ]

    currents = kernel.generate_input_currents(events, duration_ms=50.0)
    assert currents.shape == (50, 3)
    assert np.all(currents[:, 2] == 0)  # Channel 3 has no events
    assert np.all(currents >= 0)


def test_kernel_peak_amplitude():
    """Test kernel peak amplitude is correct."""
    kernel = SynapticKernel(tau_s=10.0)
    peak = kernel.get_peak_amplitude()
    expected = np.exp(-1.0)
    assert abs(peak - expected) < 1e-10


def test_end_to_end_pipeline():
    """Test full pipeline: events -> currents."""
    gen = PoissonGenerator(n_channels=5, lambda_max=100.0, seed=42)
    kernel = SynapticKernel(tau_s=10.0, dt_ms=1.0)

    # Generate events
    events = gen.generate_events(duration_ms=100.0, dt_ms=1.0)

    # Convert to currents
    currents = kernel.generate_input_currents(events, duration_ms=100.0)

    assert currents.shape == (100, 5)
    assert np.all(np.isfinite(currents))
    assert np.all(currents >= 0)
