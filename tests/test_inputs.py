"""Tests for input generation module."""

import numpy as np
import pynapple as nap
from neurosignature.inputs import PoissonGenerator, InputGenerator


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

    assert isinstance(events, nap.TsGroup)
    assert len(events) == 10

    counts = [len(events[k]) for k in events.keys()]
    total = sum(counts)
    # Expected ~100 events per second over 1 second = ~100 events total
    assert 50 < total < 150  # Allow variance


def test_spike_train_shape():
    """Test spike train has correct shape."""
    gen = PoissonGenerator(n_channels=5, lambda_max=50.0, seed=42)
    spike_train = gen.generate_spike_train(duration_ms=100.0, dt_ms=1.0)
    assert isinstance(spike_train, nap.TsdFrame)
    assert spike_train.shape == (100, 5)
    arr = np.asarray(spike_train)
    assert np.all((arr == 0) | (arr == 1))


def test_input_generator_generate_returns_tsgroup():
    """InputGenerator.generate returns a TsGroup."""
    ig = InputGenerator(n_channels=3, rate_hz=50.0, dt_ms=1.0, seed=7)
    result = ig.generate(duration_ms=200.0)
    assert isinstance(result, nap.TsGroup)
    assert len(result) == 3


def test_input_generator_generate_batch():
    """InputGenerator.generate_batch returns a list of TsGroups."""
    ig = InputGenerator(n_channels=3, rate_hz=50.0, dt_ms=1.0, seed=7)
    batch = ig.generate_batch(n_trials=5, duration_ms=100.0)
    assert len(batch) == 5
    for ts_group in batch:
        assert isinstance(ts_group, nap.TsGroup)
        assert len(ts_group) == 3


def test_input_generator_trials_are_independent():
    """Each trial from generate_batch should have different spike times."""
    ig = InputGenerator(n_channels=2, rate_hz=100.0, dt_ms=1.0, seed=0)
    batch = ig.generate_batch(n_trials=3, duration_ms=500.0)
    # At 100 Hz over 500 ms we expect ~50 events; comparing total counts
    counts = [sum(len(batch[i][k]) for k in batch[i].keys()) for i in range(3)]
    # Trials should not all be identical
    assert not (counts[0] == counts[1] == counts[2])
