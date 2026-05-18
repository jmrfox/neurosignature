"""Tests verifying pynapple return types across the new pipeline."""

import numpy as np
import pynapple as nap
from neurosignature.inputs import PoissonGenerator, InputGenerator
from neurosignature.simulation.simulator import Simulator
from neurosignature.systems import ContinuousTimeRNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctrnn(n_hidden=8, n_inputs=4, n_outputs=4, seed=0) -> ContinuousTimeRNN:
    rng = np.random.default_rng(seed)
    W_int = rng.standard_normal((n_hidden, n_hidden)) * 0.1
    W_prop = rng.standard_normal((n_hidden, n_hidden)) * 0.1
    W_prop = (W_prop + W_prop.T) / 2
    W_u = np.abs(rng.standard_normal((n_hidden, n_inputs))) * 0.1
    W_o = rng.standard_normal((n_outputs, n_hidden)) * 0.1
    return ContinuousTimeRNN(
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        W_int=W_int,
        W_prop=W_prop,
        W_u=W_u,
        W_o=W_o,
        g=0.1,
        tau_syn_ms=5.0,
        dt_ms=1.0,
    )


# ---------------------------------------------------------------------------
# PoissonGenerator
# ---------------------------------------------------------------------------


def test_generate_events_returns_tsgroup():
    """generate_events returns a TsGroup."""
    gen = PoissonGenerator(n_channels=5, lambda_max=80.0, seed=7)
    result = gen.generate_events(duration_ms=200.0)
    assert isinstance(result, nap.TsGroup)
    assert len(result) == 5


def test_generate_spike_train_returns_tsdframe():
    """generate_spike_train returns a TsdFrame."""
    gen = PoissonGenerator(n_channels=5, lambda_max=80.0, seed=7)
    result = gen.generate_spike_train(duration_ms=100.0, dt_ms=1.0)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape == (100, 5)


def test_tsgroup_time_support_in_ms():
    """TsGroup time_support end should match duration in seconds."""
    gen = PoissonGenerator(n_channels=3, lambda_max=50.0, seed=1)
    duration_ms = 500.0
    result = gen.generate_events(duration_ms=duration_ms)
    expected_end_s = duration_ms / 1000.0
    end_val = np.asarray(result.time_support["end"])[0]
    assert abs(end_val - expected_end_s) < 1e-6


# ---------------------------------------------------------------------------
# InputGenerator
# ---------------------------------------------------------------------------


def test_input_generator_generate_returns_tsgroup():
    """InputGenerator.generate returns TsGroup."""
    ig = InputGenerator(n_channels=4, rate_hz=80.0, dt_ms=1.0, seed=3)
    result = ig.generate(duration_ms=100.0)
    assert isinstance(result, nap.TsGroup)
    assert len(result) == 4


def test_input_generator_generate_batch_returns_list():
    """InputGenerator.generate_batch returns list of TsGroups."""
    ig = InputGenerator(n_channels=4, rate_hz=80.0, dt_ms=1.0, seed=4)
    batch = ig.generate_batch(n_trials=5, duration_ms=100.0)
    assert isinstance(batch, list)
    assert len(batch) == 5
    for ts_group in batch:
        assert isinstance(ts_group, nap.TsGroup)


# ---------------------------------------------------------------------------
# Simulator (thin wrapper)
# ---------------------------------------------------------------------------


def test_simulator_run_returns_tsdframe():
    """Simulator.run delegates to the callable and returns TsdFrame."""
    ctrnn = _make_ctrnn()
    sim = Simulator(ctrnn)
    gen = PoissonGenerator(n_channels=4, lambda_max=80.0, seed=5)
    ts_group = gen.generate_events(duration_ms=100.0)
    result = sim.run(ts_group)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape[1] == 4


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_pynapple():
    """Full pipeline: InputGenerator -> CTRNN -> Simulator -> TsdFrame."""
    ig = InputGenerator(n_channels=4, rate_hz=100.0, dt_ms=1.0, seed=99)
    ts_group = ig.generate(duration_ms=200.0)
    assert isinstance(ts_group, nap.TsGroup)

    ctrnn = _make_ctrnn(seed=99)
    sim = Simulator(ctrnn)
    outputs = sim.run(ts_group)
    assert isinstance(outputs, nap.TsdFrame)
    assert outputs.shape == (200, 4)
    assert np.all(np.isfinite(np.asarray(outputs)))
