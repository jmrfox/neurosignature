"""Tests verifying pynapple return types across the pipeline."""

import numpy as np
import pynapple as nap
from neurosignature.inputs import PoissonGenerator, SynapticKernel
from neurosignature.inputs.input_generator import InputGenerator
from neurosignature.simulation.simulator import Simulator
from neurosignature.systems import SystemGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_system(seed=0):
    gen = SystemGenerator(n_hidden=16, n_inputs=4, n_outputs=4)
    return gen.generate_passive_system(seed=seed)


def _make_inputs(n_steps=200, n_channels=4):
    return np.random.randn(n_steps, n_channels).astype(float)


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
    # pynapple stores seconds internally
    expected_end_s = duration_ms / 1000.0
    end_val = np.asarray(result.time_support["end"])[0]
    assert abs(end_val - expected_end_s) < 1e-6


# ---------------------------------------------------------------------------
# SynapticKernel
# ---------------------------------------------------------------------------


def test_generate_input_currents_from_tsgroup():
    """generate_input_currents accepts TsGroup and returns TsdFrame."""
    gen = PoissonGenerator(n_channels=4, lambda_max=60.0, seed=2)
    kernel = SynapticKernel(tau_ms=10.0, dt_ms=1.0)
    ts_group = gen.generate_events(duration_ms=100.0)
    result = kernel.generate_input_currents(ts_group, duration_ms=100.0)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape == (100, 4)
    assert np.all(np.asarray(result) >= 0)


def test_generate_input_currents_from_list():
    """generate_input_currents accepts list-of-arrays and returns TsdFrame."""
    kernel = SynapticKernel(tau_ms=10.0, dt_ms=1.0)
    events = [np.array([10.0, 40.0]), np.array([25.0]), np.array([])]
    result = kernel.generate_input_currents(events, duration_ms=60.0)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape == (60, 3)


# ---------------------------------------------------------------------------
# InputGenerator
# ---------------------------------------------------------------------------


def test_input_generator_generate_returns_tsdframe():
    """InputGenerator.generate returns TsdFrame."""
    ig = InputGenerator(n_channels=4, rate_hz=80.0, tau_ms=10.0, dt_ms=1.0, seed=3)
    result = ig.generate(duration_ms=100.0)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape == (100, 4)


def test_input_generator_with_spikes_returns_tsdframes():
    """InputGenerator.generate_with_spikes returns (TsdFrame, TsdFrame)."""
    ig = InputGenerator(n_channels=4, rate_hz=80.0, tau_ms=10.0, dt_ms=1.0, seed=4)
    spikes, currents = ig.generate_with_spikes(duration_ms=100.0)
    assert isinstance(spikes, nap.TsdFrame)
    assert isinstance(currents, nap.TsdFrame)
    assert spikes.shape == (100, 4)
    assert currents.shape == (100, 4)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_simulator_run_returns_tsdframe():
    """Simulator.run returns TsdFrame outputs."""
    system = _make_system()
    sim = Simulator(system, dt_ms=1.0)
    inputs = _make_inputs(n_steps=100, n_channels=4)
    outputs, states = sim.run(inputs)
    assert isinstance(outputs, nap.TsdFrame)
    assert states is None
    assert outputs.shape == (100, system.n_outputs)


def test_simulator_run_record_states():
    """Simulator.run with record_states=True returns TsdFrame states."""
    system = _make_system()
    sim = Simulator(system, dt_ms=1.0)
    inputs = _make_inputs(n_steps=50, n_channels=4)
    outputs, states = sim.run(inputs, record_states=True)
    assert isinstance(outputs, nap.TsdFrame)
    assert isinstance(states, nap.TsdFrame)
    assert states.shape == (50, system.n_hidden)


def test_simulator_run_time_index_in_ms():
    """Simulator outputs have correct ms time index (stored as seconds)."""
    system = _make_system()
    dt_ms = 2.0
    sim = Simulator(system, dt_ms=dt_ms)
    n_steps = 50
    inputs = _make_inputs(n_steps=n_steps, n_channels=4)
    outputs, _ = sim.run(inputs)
    # pynapple stores seconds internally; last index ~ (n_steps-1)*dt_ms/1000
    expected_last_s = (n_steps - 1) * dt_ms / 1000.0
    assert abs(outputs.index[-1] - expected_last_s) < 1e-9


def test_simulator_accepts_tsdframe_input():
    """Simulator.run accepts TsdFrame as inputs."""
    system = _make_system()
    sim = Simulator(system, dt_ms=1.0)
    t_ms = np.arange(80, dtype=float)
    inputs_tsd = nap.TsdFrame(
        t=t_ms,
        d=np.random.randn(80, 4),
        time_units="ms",
    )
    outputs, _ = sim.run(inputs_tsd)
    assert isinstance(outputs, nap.TsdFrame)
    assert outputs.shape == (80, system.n_outputs)


def test_simulator_run_batch_returns_list():
    """Simulator.run_batch returns list of TsdFrame."""
    system = _make_system()
    sim = Simulator(system, dt_ms=1.0)
    batch = np.random.randn(3, 60, 4)
    results = sim.run_batch(batch)
    assert isinstance(results, list)
    assert len(results) == 3
    for tsd in results:
        assert isinstance(tsd, nap.TsdFrame)
        assert tsd.shape == (60, system.n_outputs)


def test_simulator_run_multiple_systems_returns_list():
    """run_with_multiple_systems returns list of TsdFrame."""
    systems = [_make_system(seed=i) for i in range(3)]
    sim = Simulator(systems[0], dt_ms=1.0)
    inputs = _make_inputs(n_steps=40, n_channels=4)
    results = sim.run_with_multiple_systems(systems, inputs)
    assert isinstance(results, list)
    assert len(results) == 3
    for tsd in results:
        assert isinstance(tsd, nap.TsdFrame)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_pynapple():
    """Full pipeline: InputGenerator -> Simulator -> TsdFrame outputs."""
    ig = InputGenerator(n_channels=4, rate_hz=100.0, tau_ms=10.0, dt_ms=1.0, seed=99)
    currents = ig.generate(duration_ms=200.0)
    assert isinstance(currents, nap.TsdFrame)

    system = _make_system(seed=99)
    sim = Simulator(system, dt_ms=1.0)
    outputs, _ = sim.run(currents)
    assert isinstance(outputs, nap.TsdFrame)
    assert outputs.shape[0] == 200
    assert np.all(np.isfinite(np.asarray(outputs)))
