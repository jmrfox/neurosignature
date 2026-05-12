"""Tests for dynamical systems module."""

import numpy as np
import pytest
from neurosignature.systems import ContinuousTimeRNN, SystemGenerator


def test_continuous_time_rnn_init():
    """Test system initialization."""
    system = ContinuousTimeRNN(
        n_hidden=64,
        n_inputs=25,
        n_outputs=32,
        spectral_radius=0.8,
        sparsity=0.1,
        seed=42,
    )
    assert system.n_hidden == 64
    assert system.n_inputs == 25
    assert system.n_outputs == 32
    assert system.W_h.shape == (64, 64)
    assert system.W_u.shape == (64, 25)
    assert system.W_o.shape == (32, 64)


def test_spectral_radius_control():
    """Test that spectral radius is properly controlled."""
    for target_radius in [0.5, 0.8, 0.9]:
        system = ContinuousTimeRNN(
            n_hidden=50,
            n_inputs=10,
            n_outputs=10,
            spectral_radius=target_radius,
            sparsity=0.2,
            seed=42,
        )
        # Check actual spectral radius
        eigenvalues = np.linalg.eigvals(system.W_h)
        actual_radius = np.max(np.abs(eigenvalues))
        # Allow small tolerance for numerical errors
        assert actual_radius <= target_radius * 1.05, (
            f"Spectral radius {actual_radius} exceeds target {target_radius}"
        )


def test_sparsity():
    """Test that weights have correct sparsity level."""
    sparsity = 0.2
    system = ContinuousTimeRNN(
        n_hidden=100,
        n_inputs=10,
        n_outputs=10,
        spectral_radius=0.8,
        sparsity=sparsity,
        seed=42,
    )
    # Count non-zero elements
    nonzero = np.count_nonzero(system.W_h)
    total = system.W_h.size
    actual_sparsity = nonzero / total
    # Should be close to target (allow 5% tolerance due to randomness)
    assert abs(actual_sparsity - sparsity) < 0.05


def test_integration_step():
    """Test that a single integration step works."""
    system = ContinuousTimeRNN(
        n_hidden=10,
        n_inputs=5,
        n_outputs=3,
        seed=42,
    )
    h = system.reset_state()
    u = np.zeros(5)
    h_new = system.step(h, u, dt=1.0)
    assert h_new.shape == (10,)
    assert np.all(np.isfinite(h_new))


def test_bounded_dynamics():
    """Test that system dynamics remain bounded."""
    system = ContinuousTimeRNN(
        n_hidden=20,
        n_inputs=10,
        n_outputs=5,
        spectral_radius=0.8,
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
    assert np.all(np.abs(v) < 100)  # Should be well-bounded


def test_output_computation():
    """Test output computation."""
    system = ContinuousTimeRNN(
        n_hidden=10,
        n_inputs=5,
        n_outputs=3,
        seed=42,
    )
    h = np.random.randn(10)
    v = system.compute_output(h)
    assert v.shape == (3,)
    expected = system.W_o @ h + system.b_o
    np.testing.assert_array_almost_equal(v, expected)


def test_system_generator():
    """Test system generator factory."""
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


def test_ensemble_generation():
    """Test ensemble generation."""
    generator = SystemGenerator()
    systems = generator.generate_ensemble(n_systems=10)
    assert len(systems) == 10
    # Check systems are different (different seeds)
    assert not np.array_equal(systems[0].W_h, systems[1].W_h)
