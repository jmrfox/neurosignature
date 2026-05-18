"""Tests for dynamical systems module."""

import numpy as np
import pynapple as nap
from neurosignature.systems import ContinuousTimeRNN


def _make_ctrnn(
    n_hidden=10, n_inputs=3, n_outputs=2, seed=42, tau_syn_ms=5.0, dt_ms=1.0
) -> ContinuousTimeRNN:
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
        tau_syn_ms=tau_syn_ms,
        dt_ms=dt_ms,
    )


def test_integration_step():
    """Test that a single integration step works."""
    system = _make_ctrnn()
    h = system.reset_state()
    u = np.zeros(3)
    h_new = system.step(h, u, dt=1.0)
    assert h_new.shape == (10,)
    assert np.all(np.isfinite(h_new))


def test_bounded_dynamics():
    """Test that system dynamics remain bounded."""
    system = _make_ctrnn(n_hidden=20, n_inputs=3, n_outputs=2)
    h = system.reset_state()

    # Run simulation with random inputs
    for _ in range(1000):
        u = np.random.randn(3) * 0.1
        h = system.step(h, u, dt=1.0)

    # Check outputs remain bounded
    v = system.compute_output(h)
    assert np.all(np.isfinite(v))
    # With resting potential at -65mV, outputs should be in reasonable range
    assert np.all(np.abs(v + 65.0) < 50)  # perturbations within 50mV


def test_resting_potential():
    """Test that system returns to resting potential with zero input."""
    system = _make_ctrnn(n_hidden=20, n_inputs=3, n_outputs=2)
    system.V_rest = -70.0
    system.polarity = "bipolar"

    h = np.random.randn(20) * 0.5
    for _ in range(500):
        h = system.step(h, np.zeros(3), dt=1.0)

    v = system.compute_output(h)
    np.testing.assert_allclose(v, -70.0, atol=1.0)


def test_direct_matrix_construction():
    """Test creating CTRNN directly with explicit matrices."""
    n_hidden, n_inputs, n_outputs = 10, 5, 3

    W_int = np.random.randn(n_hidden, n_hidden) * 0.1
    W_prop = np.random.randn(n_hidden, n_hidden) * 0.1
    W_prop = (W_prop + W_prop.T) / 2
    W_u = np.random.randn(n_hidden, n_inputs) * 0.1
    W_o = np.random.randn(n_outputs, n_hidden) * 0.1

    system = ContinuousTimeRNN(
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        W_int=W_int,
        W_prop=W_prop,
        W_u=W_u,
        W_o=W_o,
        g=0.1,
        V_rest=-65.0,
        polarity="bipolar",
    )

    assert np.array_equal(system.W_int, W_int)
    assert np.array_equal(system.W_prop, W_prop)
    assert np.array_equal(system.W_u, W_u)
    assert np.array_equal(system.W_o, W_o)


def test_ctrnn_call_returns_tsdframe():
    """ContinuousTimeRNN.__call__ returns TsdFrame given a TsGroup."""
    from neurosignature.inputs import PoissonGenerator

    system = _make_ctrnn(n_hidden=8, n_inputs=3, n_outputs=2, dt_ms=1.0)
    gen = PoissonGenerator(n_channels=3, lambda_max=50.0, seed=1)
    ts_group = gen.generate_events(duration_ms=100.0, dt_ms=1.0)
    result = system(ts_group)
    assert isinstance(result, nap.TsdFrame)
    assert result.shape == (100, 2)
    assert np.all(np.isfinite(np.asarray(result)))


def test_ctrnn_syn_params_stored():
    """tau_syn_ms and dt_ms are stored on the system."""
    system = _make_ctrnn(tau_syn_ms=8.0, dt_ms=0.5)
    assert system.tau_syn_ms == 8.0
    assert system.dt_ms == 0.5


def test_matrix_shape_validation():
    """Test that CTRNN validates matrix shapes."""
    n_hidden, n_inputs, n_outputs = 10, 5, 3

    W_int = np.random.randn(n_hidden, n_hidden)
    W_prop = np.random.randn(n_hidden, n_hidden)
    W_u = np.random.randn(n_hidden, n_inputs)
    W_o = np.random.randn(n_outputs, n_hidden)

    # Wrong W_int shape
    try:
        ContinuousTimeRNN(
            n_hidden=n_hidden,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            W_int=np.random.randn(n_hidden, n_hidden + 1),
            W_prop=W_prop,
            W_u=W_u,
            W_o=W_o,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Wrong W_prop shape
    try:
        ContinuousTimeRNN(
            n_hidden=n_hidden,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            W_int=W_int,
            W_prop=np.random.randn(n_hidden, n_hidden + 1),
            W_u=W_u,
            W_o=W_o,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
