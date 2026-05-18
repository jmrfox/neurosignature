"""Tests for the redesigned ScalarDescriptor subclasses and VectorDescriptor."""

import numpy as np
import pynapple as nap
import pytest
from neurosignature.summaries import (
    ReferenceMean,
    ReferenceStd,
    ReferenceSpectralCentroid,
    ResidualMean,
    ResidualStd,
    ResidualEnergy,
    ResidualSpectralCentroidMean,
    ResidualSpectralCentroidStd,
    ResidualParticipationRatio,
    ResidualMaxEigenvalue,
    ResidualEigenvalueEntropy,
    CrossCorrelationMean,
    TransmissionEfficiency,
    VectorDescriptor,
)
from neurosignature.summaries.scalar_descriptors import _apply_transform


def _tsdframe(arr: np.ndarray) -> nap.TsdFrame:
    t = np.arange(arr.shape[0], dtype=float)
    return nap.TsdFrame(t=t, d=arr, time_units="ms")


def _traces_and_ref(n_t=200, n_ch=4, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_t, n_ch))
    traces = _tsdframe(arr)
    v_ref = arr[:, 0]
    return traces, v_ref, arr


# ---------------------------------------------------------------------------
# Reference-trace descriptors
# ---------------------------------------------------------------------------


def test_reference_mean():
    arr = np.ones((100, 3)) * 5.0
    v_ref = arr[:, 0]
    val = ReferenceMean().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(5.0)


def test_reference_std_constant():
    arr = np.ones((100, 3)) * 3.0
    v_ref = arr[:, 0]
    val = ReferenceStd().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(np.log(1e-8))


def test_reference_std_nonconstant():
    arr = np.random.randn(200, 3)
    v_ref = arr[:, 0]
    val = ReferenceStd().compute(_tsdframe(arr), v_ref)
    assert np.isfinite(val)


def test_reference_spectral_centroid_positive():
    t = np.linspace(0, 1, 1000)
    arr = np.column_stack([np.sin(2 * np.pi * 20 * t)] * 3)
    v_ref = arr[:, 0]
    val = ReferenceSpectralCentroid(dt_ms=1.0).compute(_tsdframe(arr), v_ref)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Residual descriptors
# ---------------------------------------------------------------------------


def test_residual_mean_zero_when_identical():
    arr = np.random.randn(100, 1)
    arr = np.hstack([arr, arr, arr])
    v_ref = arr[:, 0]
    val = ResidualMean().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(0.0)


def test_residual_std_zero_when_identical():
    arr = np.random.randn(100, 1)
    arr = np.hstack([arr, arr, arr])
    v_ref = arr[:, 0]
    val = ResidualStd().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(np.log(1e-8))


def test_residual_std_positive():
    traces, v_ref, arr = _traces_and_ref()
    val = ResidualStd().compute(traces, v_ref)
    assert np.isfinite(val)


def test_residual_energy_positive():
    traces, v_ref, _ = _traces_and_ref()
    val = ResidualEnergy(dt_ms=1.0).compute(traces, v_ref)
    assert np.isfinite(val)


def test_residual_energy_zero_when_identical():
    arr = np.random.randn(100, 1)
    arr = np.hstack([arr, arr, arr])
    v_ref = arr[:, 0]
    val = ResidualEnergy(dt_ms=1.0).compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(np.log(1e-8))


def test_residual_energy_scales_with_dt():
    traces, v_ref, _ = _traces_and_ref()
    e1 = ResidualEnergy(dt_ms=1.0, transform="identity").compute(traces, v_ref)
    e2 = ResidualEnergy(dt_ms=2.0, transform="identity").compute(traces, v_ref)
    assert e2 == pytest.approx(e1 * 2.0)


def test_residual_spectral_centroid_mean_positive():
    traces, v_ref, _ = _traces_and_ref(n_t=500)
    val = ResidualSpectralCentroidMean(dt_ms=1.0).compute(traces, v_ref)
    assert np.isfinite(val)


def test_residual_spectral_centroid_std_nonneg():
    traces, v_ref, _ = _traces_and_ref(n_t=500)
    val = ResidualSpectralCentroidStd(dt_ms=1.0).compute(traces, v_ref)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Covariance of R descriptors
# ---------------------------------------------------------------------------


def test_participation_ratio_positive():
    traces, v_ref, _ = _traces_and_ref()
    val = ResidualParticipationRatio().compute(traces, v_ref)
    assert np.isfinite(val)


def test_participation_ratio_leq_n_channels():
    traces, v_ref, arr = _traces_and_ref(n_ch=5)
    n_residual = arr.shape[1] - 1
    raw = ResidualParticipationRatio(transform="identity").compute(traces, v_ref)
    assert raw <= n_residual + 1e-6


def test_max_eigenvalue_positive():
    traces, v_ref, _ = _traces_and_ref()
    val = ResidualMaxEigenvalue().compute(traces, v_ref)
    assert np.isfinite(val)


def test_eigenvalue_entropy_nonneg():
    traces, v_ref, _ = _traces_and_ref()
    val = ResidualEigenvalueEntropy().compute(traces, v_ref)
    assert val >= 0.0  # identity transform; entropy is non-negative


def test_eigenvalue_entropy_zero_rank1():
    x = np.random.randn(200, 1)
    arr = np.hstack([np.zeros((200, 1)), x, x, x])
    v_ref = arr[:, 0]
    val = ResidualEigenvalueEntropy().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(0.0, abs=1e-6)  # identity transform


# ---------------------------------------------------------------------------
# Cross-channel descriptors
# ---------------------------------------------------------------------------


def test_cross_correlation_mean_range():
    traces, v_ref, _ = _traces_and_ref()
    val = CrossCorrelationMean().compute(traces, v_ref)
    assert -1.0 <= val <= 1.0


def test_cross_correlation_mean_perfectly_correlated():
    x = np.arange(200, dtype=float)
    arr = np.column_stack([x, x, x])
    v_ref = arr[:, 0]
    val = CrossCorrelationMean().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(1.0)


def test_transmission_efficiency_positive():
    traces, v_ref, _ = _traces_and_ref()
    val = TransmissionEfficiency().compute(traces, v_ref)
    assert np.isfinite(val)


def test_transmission_efficiency_high_when_ref_dominates():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 500)
    v_ref = np.sin(2 * np.pi * 5 * t) * 10.0
    noise = rng.standard_normal((500, 3)) * 0.01
    arr = np.column_stack([v_ref, v_ref[:, None] + noise])
    v_ref_arr = arr[:, 0]
    raw = TransmissionEfficiency(transform="identity").compute(
        _tsdframe(arr), v_ref_arr
    )
    assert raw > 1.0


# ---------------------------------------------------------------------------
# VectorDescriptor with reference_channel
# ---------------------------------------------------------------------------


def test_vector_descriptor_length_and_finite():
    descriptors = [
        ReferenceMean(),
        ReferenceStd(),
        ResidualEnergy(dt_ms=1.0),
        ResidualParticipationRatio(),
        TransmissionEfficiency(),
    ]
    vd = VectorDescriptor(descriptors, reference_channel=0)
    traces, _, _ = _traces_and_ref()
    result = vd.compute(traces)
    assert result.shape == (5,)
    assert np.all(np.isfinite(result))


def test_vector_descriptor_reference_channel_respected():
    arr = np.random.randn(100, 3)
    traces = _tsdframe(arr)
    vd0 = VectorDescriptor([ReferenceMean()], reference_channel=0)
    vd1 = VectorDescriptor([ReferenceMean()], reference_channel=1)
    assert vd0.compute(traces)[0] == pytest.approx(np.mean(arr[:, 0]))
    assert vd1.compute(traces)[0] == pytest.approx(np.mean(arr[:, 1]))


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


def test_apply_transform_identity():
    assert _apply_transform(3.7, "identity") == pytest.approx(3.7)


def test_apply_transform_log():
    val = _apply_transform(1.0, "log")
    assert val == pytest.approx(np.log(1.0 + 1e-8))


def test_apply_transform_log_near_zero():
    val = _apply_transform(0.0, "log")
    assert val == pytest.approx(np.log(1e-8))


def test_apply_transform_tanh():
    val = _apply_transform(0.0, "tanh")
    assert val == pytest.approx(0.0)
    val2 = _apply_transform(1.0, "tanh")
    assert val2 == pytest.approx(np.tanh(1.0))


def test_apply_transform_sigmoid():
    val = _apply_transform(0.0, "sigmoid")
    assert val == pytest.approx(0.5)
    val2 = _apply_transform(2.0, "sigmoid")
    assert val2 == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_apply_transform_invalid():
    with pytest.raises(ValueError, match="Unknown transform"):
        _apply_transform(1.0, "sqrt")


def test_descriptor_invalid_transform_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        ReferenceMean(transform="sqrt")


def test_default_transforms():
    assert ReferenceMean().transform == "identity"
    assert ReferenceStd().transform == "log"
    assert ReferenceSpectralCentroid(dt_ms=1.0).transform == "log"
    assert ResidualMean().transform == "identity"
    assert ResidualStd().transform == "log"
    assert ResidualEnergy(dt_ms=1.0).transform == "log"
    assert ResidualSpectralCentroidMean(dt_ms=1.0).transform == "log"
    assert ResidualSpectralCentroidStd(dt_ms=1.0).transform == "log"
    assert ResidualParticipationRatio().transform == "log"
    assert ResidualMaxEigenvalue().transform == "log"
    assert ResidualEigenvalueEntropy().transform == "identity"
    assert CrossCorrelationMean().transform == "identity"
    assert TransmissionEfficiency().transform == "log"


def test_transform_override():
    arr = np.ones((100, 3)) * 2.0
    v_ref = arr[:, 0]
    raw_std = float(np.std(arr[:, 0]))
    log_val = ReferenceStd().compute(_tsdframe(arr), v_ref)
    id_val = ReferenceStd(transform="identity").compute(_tsdframe(arr), v_ref)
    assert log_val == pytest.approx(np.log(raw_std + 1e-8))
    assert id_val == pytest.approx(raw_std)


def test_log_transform_applied_to_reference_std():
    arr = np.random.randn(200, 3)
    v_ref = arr[:, 0]
    raw = float(np.std(v_ref))
    val = ReferenceStd().compute(_tsdframe(arr), v_ref)
    assert val == pytest.approx(np.log(raw + 1e-8))
