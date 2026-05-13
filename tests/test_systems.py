"""Tests for dynamical systems module."""

import numpy as np
from neurosignature.systems import ContinuousTimeRNN, SystemGenerator


def test_system_generator_basic():
    """Test system generator factory creates valid systems."""
    generator = SystemGenerator(n_hidden=32, n_inputs=10, n_outputs=8)
    system = generator.generate_passive_system(seed=42)
    assert system.n_hidden == 32
    assert system.n_inputs == 10
    assert system.n_outputs == 8
    assert system.W_int.shape == (32, 32)
    assert system.W_prop.shape == (32, 32)
    assert system.W_u.shape == (32, 10)
    assert system.W_o.shape == (8, 32)


def test_spectral_radius_control():
    """Test that spectral radius is properly controlled for W_int and W_prop."""
    generator = SystemGenerator(n_hidden=50, n_inputs=10, n_outputs=10)

    # Test r_int (integration matrix)
    for target_radius in [0.5, 0.7, 0.8]:
        system = generator.generate_passive_system(
            r_int=target_radius,
            r_prop=0.1,
            sparsity=0.2,
            seed=42,
        )
        eigenvalues = np.linalg.eigvals(system.W_int)
        actual_radius = np.max(np.abs(eigenvalues))
        assert (
            actual_radius <= target_radius * 1.05
        ), f"W_int radius {actual_radius} exceeds target {target_radius}"

    # Test r_prop (propagation matrix)
    for target_radius in [0.1, 0.2, 0.3]:
        system = generator.generate_passive_system(
            r_int=0.5,
            r_prop=target_radius,
            sparsity=0.2,
            seed=42,
        )
        eigenvalues = np.linalg.eigvals(system.W_prop)
        actual_radius = np.max(np.abs(eigenvalues))
        assert (
            actual_radius <= target_radius * 1.05
        ), f"W_prop radius {actual_radius} exceeds target {target_radius}"


def test_sparsity():
    """Test that recurrent matrices have correct sparsity level."""
    sparsity = 0.2
    generator = SystemGenerator(n_hidden=100, n_inputs=10, n_outputs=10)
    system = generator.generate_passive_system(
        r_int=0.5,
        r_prop=0.1,
        sparsity=sparsity,
        seed=42,
    )
    # Check W_int sparsity
    nonzero_int = np.count_nonzero(system.W_int)
    total = system.W_int.size
    actual_sparsity = nonzero_int / total
    assert abs(actual_sparsity - sparsity) < 0.05

    # Check W_prop sparsity - note: symmetrization increases density
    # so we just check it's within a reasonable range (sparsity to 2x sparsity)
    nonzero_prop = np.count_nonzero(system.W_prop)
    actual_sparsity_prop = nonzero_prop / total
    assert sparsity * 0.5 < actual_sparsity_prop < sparsity * 2.5


def test_integration_step():
    """Test that a single integration step works."""
    generator = SystemGenerator(n_hidden=10, n_inputs=5, n_outputs=3)
    system = generator.generate_passive_system(seed=42)
    h = system.reset_state()
    u = np.zeros(5)
    h_new = system.step(h, u, dt=1.0)
    assert h_new.shape == (10,)
    assert np.all(np.isfinite(h_new))


def test_bounded_dynamics():
    """Test that system dynamics remain bounded."""
    generator = SystemGenerator(n_hidden=20, n_inputs=10, n_outputs=5)
    system = generator.generate_passive_system(
        r_int=0.5,
        r_prop=0.1,
        g=0.1,
        seed=42,
    )
    h = system.reset_state()

    # Run simulation with random inputs
    for _ in range(1000):
        u = np.random.randn(10) * 0.1
        h = system.step(h, u, dt=1.0)

    # Check outputs remain bounded
    v = system.compute_output(h)
    assert np.all(np.isfinite(v))
    # With resting potential at -65mV, outputs should be in reasonable range
    assert np.all(np.abs(v + 65.0) < 50)  # perturbations within 50mV


def test_output_computation():
    """Test output computation with polarity modes."""
    generator = SystemGenerator(n_hidden=10, n_inputs=5, n_outputs=3)

    # Test bipolar mode (linear)
    system_bipolar = generator.generate_passive_system(
        polarity="bipolar",
        V_rest=-65.0,
        seed=42,
    )
    h = np.random.randn(10)
    v = system_bipolar.compute_output(h)
    assert v.shape == (3,)
    # For bipolar, should be V_rest + W_o @ h
    expected = -65.0 + system_bipolar.W_o @ h
    np.testing.assert_array_almost_equal(v, expected)

    # Test excitatory mode (softplus - always positive)
    system_exc = generator.generate_passive_system(
        polarity="excitatory",
        V_rest=-65.0,
        seed=42,
    )
    v_exc = system_exc.compute_output(h)
    assert v_exc.shape == (3,)
    # Should be >= V_rest (depolarization only)
    assert np.all(v_exc >= -65.0 - 1e-10)  # small tolerance for numerical error

    # Test inhibitory mode (negative softplus)
    system_inh = generator.generate_passive_system(
        polarity="inhibitory",
        V_rest=-65.0,
        seed=42,
    )
    v_inh = system_inh.compute_output(h)
    assert v_inh.shape == (3,)
    # Should be <= V_rest (hyperpolarization only)
    assert np.all(v_inh <= -65.0 + 1e-10)


def test_system_generator():
    """Test backward-compatible system generator."""
    generator = SystemGenerator(n_hidden=32, n_inputs=10, n_outputs=8)
    system = generator.generate_random_system(seed=42)
    assert system.n_hidden == 32
    assert system.n_inputs == 10
    assert system.n_outputs == 8


def test_spectral_radius_sweep():
    """Test spectral radius sweep generation."""
    generator = SystemGenerator()
    radii = [0.5, 0.7, 0.9]
    systems = generator.generate_spectral_radius_sweep(radii)
    assert len(systems) == 3


def test_resting_potential():
    """Test that system returns to resting potential with zero input."""
    generator = SystemGenerator(n_hidden=20, n_inputs=5, n_outputs=3)
    system = generator.generate_passive_system(
        g=0.1,
        V_rest=-70.0,
        polarity="bipolar",
        seed=42,
    )

    # Initialize with non-zero state
    h = np.random.randn(20) * 0.5

    # Run with zero input for many steps (should decay to rest)
    for _ in range(500):
        h = system.step(h, np.zeros(5), dt=1.0)

    # Output should be close to resting potential
    v = system.compute_output(h)
    np.testing.assert_allclose(v, -70.0, atol=1.0)


def test_polarity_excitatory_w_u_positive():
    """Test that excitatory mode has positive W_u weights."""
    generator = SystemGenerator(n_hidden=20, n_inputs=5, n_outputs=3)
    system = generator.generate_passive_system(
        polarity="excitatory",
        seed=42,
    )
    # W_u should be non-negative
    assert np.all(system.W_u >= 0)


def test_propagation_matrix_symmetric():
    """Test that W_prop is symmetric."""
    generator = SystemGenerator(n_hidden=30, n_inputs=5, n_outputs=3)
    system = generator.generate_passive_system(
        r_prop=0.2,
        seed=42,
    )
    # W_prop should be symmetric
    np.testing.assert_array_almost_equal(system.W_prop, system.W_prop.T)


def test_ensemble_generation():
    """Test ensemble generation with new parameters."""
    generator = SystemGenerator()
    systems = generator.generate_ensemble(n_systems=10)
    assert len(systems) == 10
    # Check systems are different (different W_int)
    assert not np.array_equal(systems[0].W_int, systems[1].W_int)


def test_direct_matrix_construction():
    """Test creating CTRNN directly with explicit matrices."""
    n_hidden, n_inputs, n_outputs = 10, 5, 3

    # Create explicit matrices
    W_int = np.random.randn(n_hidden, n_hidden) * 0.1
    W_prop = np.random.randn(n_hidden, n_hidden) * 0.1
    W_prop = (W_prop + W_prop.T) / 2  # Make symmetric
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
