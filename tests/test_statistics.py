"""Tests for summary statistics and descriptor computation."""

import numpy as np
import pynapple as nap
from neurosignature.summaries import (
    compute_channel_statistics,
    compute_global_statistics,
    compute_spectral_statistics,
    ReferenceMean,
    ReferenceStd,
    ResidualEnergy,
    ResidualParticipationRatio,
    CrossCorrelationMean,
    TransmissionEfficiency,
    VectorDescriptor,
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


def _make_tsdframe(arr: np.ndarray) -> nap.TsdFrame:
    t_ms = np.arange(arr.shape[0], dtype=float)
    return nap.TsdFrame(t=t_ms, d=arr, time_units="ms")


def test_vector_descriptor_length():
    """VectorDescriptor returns a vector with one entry per descriptor."""
    descriptors = [ReferenceMean(), ReferenceStd(), ResidualEnergy(dt_ms=1.0)]
    vd = VectorDescriptor(descriptors, reference_channel=0)
    traces = _make_tsdframe(np.random.randn(500, 3))
    result = vd.compute(traces)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_vector_descriptor_reference_mean_value():
    """ReferenceMean via VectorDescriptor matches np.mean of channel 0."""
    arr = np.random.randn(500, 3)
    traces = _make_tsdframe(arr)
    vd = VectorDescriptor([ReferenceMean()], reference_channel=0)
    result = vd.compute(traces)
    np.testing.assert_almost_equal(result[0], np.mean(arr[:, 0]))


def test_cross_correlation_mean():
    """CrossCorrelationMean returns a scalar in [-1, 1]."""
    arr = np.random.randn(500, 4)
    traces = _make_tsdframe(arr)
    v_ref = arr[:, 0]
    val = CrossCorrelationMean().compute(traces, v_ref)
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0


def test_transmission_efficiency_positive():
    """TransmissionEfficiency (log transform) returns a finite scalar."""
    arr = np.random.randn(300, 4)
    traces = _make_tsdframe(arr)
    v_ref = arr[:, 0]
    val = TransmissionEfficiency().compute(traces, v_ref)
    assert np.isfinite(val)


def test_vector_descriptor_distinguishes_signals():
    """VectorDescriptor distinguishes slow vs fast oscillations."""
    t = np.linspace(0, 100, 1000)
    slow = _make_tsdframe(np.column_stack([np.sin(0.1 * t), np.cos(0.1 * t)]))
    fast = _make_tsdframe(np.column_stack([np.sin(10 * t), np.cos(10 * t)]))
    vd = VectorDescriptor(
        [
            ReferenceMean(),
            ReferenceStd(),
            ResidualEnergy(dt_ms=0.1),
            ResidualParticipationRatio(),
        ],
        reference_channel=0,
    )
    desc_slow = vd.compute(slow)
    desc_fast = vd.compute(fast)
    assert np.linalg.norm(desc_slow - desc_fast) > 0.01
