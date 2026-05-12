"""Tests for summary statistics and descriptor computation."""

import numpy as np
from neurosignature.summaries import (
    compute_channel_statistics,
    compute_global_statistics,
    compute_spectral_statistics,
    DescriptorAssembler,
)


def test_channel_statistics():
    """Test per-channel statistics computation."""
    # Generate simple test traces
    t = np.linspace(0, 100, 1000)
    traces = np.column_stack(
        [
            np.sin(t),  # Channel 1: sine
            np.cos(t),  # Channel 2: cosine
            np.random.randn(1000) * 0.1,  # Channel 3: noise
        ]
    )

    stats = compute_channel_statistics(traces)

    assert "mean" in stats
    assert "variance" in stats
    assert "rms" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats

    assert len(stats["mean"]) == 3
    assert len(stats["variance"]) == 3

    # Sine wave should have zero mean
    assert abs(stats["mean"][0]) < 0.1
    # Noise should have near-zero skewness
    assert abs(stats["skewness"][2]) < 1.0


def test_global_statistics():
    """Test global statistics computation."""
    traces = np.random.randn(1000, 5)

    stats = compute_global_statistics(traces)

    assert "correlation_matrix" in stats
    assert "covariance_eigenvalues" in stats
    assert "autocorrelation_mean" in stats

    # Correlation matrix should be 5x5
    assert stats["correlation_matrix"].shape == (5, 5)

    # Eigenvalues should be 5 (one per channel)
    assert len(stats["covariance_eigenvalues"]) == 5

    # All eigenvalues should be non-negative for covariance
    assert np.all(stats["covariance_eigenvalues"] >= 0)


def test_spectral_statistics():
    """Test spectral statistics computation."""
    # Generate signal with known frequency
    t = np.linspace(0, 1, 1000)  # 1 second at 1 kHz
    dt_ms = 1.0

    # 10 Hz sine wave
    signal = np.sin(2 * np.pi * 10 * t)
    traces = signal.reshape(-1, 1)

    stats = compute_spectral_statistics(traces, dt_ms=dt_ms)

    assert "dominant_frequencies" in stats
    assert "spectral_centroid" in stats
    assert "spectral_entropy" in stats
    assert "band_powers" in stats

    # Dominant frequency should be near 10 Hz
    dom_freq = stats["dominant_frequencies"][0, 0]
    assert abs(dom_freq - 10.0) < 2.0  # Allow some FFT bin spread


def test_descriptor_assembler():
    """Test full descriptor assembly."""
    traces = np.random.randn(1000, 5)

    assembler = DescriptorAssembler()
    descriptor = assembler.compute_descriptor(traces)

    assert isinstance(descriptor, np.ndarray)
    assert len(descriptor) > 0
    assert np.all(np.isfinite(descriptor))


def test_descriptor_consistency():
    """Test that descriptors are consistent for similar traces."""
    # Generate two similar traces
    base = np.random.randn(1000, 4)
    traces1 = base + np.random.randn(1000, 4) * 0.01
    traces2 = base + np.random.randn(1000, 4) * 0.01

    assembler = DescriptorAssembler()
    desc1 = assembler.compute_descriptor(traces1)
    desc2 = assembler.compute_descriptor(traces2)

    # Similar traces should have similar descriptors
    distance = np.linalg.norm(desc1 - desc2)
    assert distance < np.linalg.norm(desc1) * 0.5


def test_descriptor_distinguishes_systems():
    """Test that descriptors distinguish different signal types."""
    # Slow oscillation (multiple channels for valid correlation)
    t = np.linspace(0, 100, 1000)
    slow = np.column_stack(
        [
            np.sin(0.1 * t),
            np.sin(0.1 * t + 0.5),
            np.sin(0.1 * t + 1.0),
        ]
    )

    # Fast oscillation
    fast = np.column_stack(
        [
            np.sin(10 * t),
            np.sin(10 * t + 0.5),
            np.sin(10 * t + 1.0),
        ]
    )

    assembler = DescriptorAssembler()
    desc_slow = assembler.compute_descriptor(slow)
    desc_fast = assembler.compute_descriptor(fast)

    # Different signals should have different descriptors
    distance = np.linalg.norm(desc_slow - desc_fast)
    assert distance > 0.1


def test_descriptor_info():
    """Test descriptor info computation."""
    assembler = DescriptorAssembler()
    info = assembler.get_descriptor_info(n_channels=8)

    assert "basic_statistics_dim" in info
    assert "global_statistics_dim" in info
    assert "total_dim" in info

    # Total should be sum of components
    assert info["total_dim"] > info["basic_statistics_dim"]
