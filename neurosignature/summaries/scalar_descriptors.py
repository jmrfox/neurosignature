"""ScalarDescriptor and BatchScalarDescriptor ABCs and descriptor implementations.

Every ``ScalarDescriptor`` returns a **single float** that summarises the full
output ``TsdFrame`` V of one trial.  Descriptors receive both V and the
reference trace ``v_ref`` (a 1-D array, shape (n_timesteps,)).  Descriptors
that do not need the reference may simply ignore the argument.

Residual convention
-------------------
``R_i = V_i - v_ref`` for each output channel i (reference channel excluded
from residual averages).  All residual descriptors aggregate across channels
to produce one scalar.
"""

import numpy as np
import pynapple as nap
from abc import ABC, abstractmethod

_EPSILON = 1e-8
_VALID_TRANSFORMS = ("identity", "log", "tanh", "sigmoid")


def _apply_transform(value: float, transform: str) -> float:
    """Apply a named output transform to a scalar descriptor value.

    Args:
        value: Raw scalar output from a descriptor.
        transform: One of ``"identity"``, ``"log"``,
            ``"tanh"``, or ``"sigmoid"``.

    Returns:
        Transformed scalar.

    Raises:
        ValueError: If ``transform`` is not one of the allowed values.
    """
    if transform == "identity":
        return value
    if transform == "log":
        return float(np.log(value + _EPSILON))
    if transform == "tanh":
        return float(np.tanh(value))
    if transform == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-value)))
    raise ValueError(
        f"Unknown transform {transform!r}. " f"Must be one of {_VALID_TRANSFORMS}."
    )


def _spectral_centroid(signal: np.ndarray, dt_ms: float) -> float:
    """Power-weighted mean frequency of a 1-D signal."""
    dt_s = dt_ms / 1000.0
    freqs = np.fft.rfftfreq(len(signal), dt_s)
    power = np.abs(np.fft.rfft(signal)) ** 2
    power_no_dc = power[1:]
    freqs_no_dc = freqs[1:]
    total = np.sum(power_no_dc)
    if total == 0:
        return 0.0
    return float(np.sum(freqs_no_dc * power_no_dc) / total)


def _residual_matrix(arr: np.ndarray, ref_ch: int) -> np.ndarray:
    """Return R = V - v_ref broadcast, excluding the reference column.

    Args:
        arr: shape (n_timesteps, n_outputs)
        ref_ch: index of the reference channel

    Returns:
        R, shape (n_timesteps, n_outputs - 1)
    """
    v_ref = arr[:, ref_ch : ref_ch + 1]
    other_cols = [i for i in range(arr.shape[1]) if i != ref_ch]
    return arr[:, other_cols] - v_ref


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class ScalarDescriptor(ABC):
    """Base class for scalar summary statistics computed from one trial.

    Subclasses implement ``compute``, which maps a single output
    ``TsdFrame`` V and a reference trace ``v_ref`` to a single float.

    Args:
        transform: Output transform applied to the raw scalar before
            returning.  Must be one of ``"identity"`` (default),
            ``"log"`` (``log(z + ε)``), ``"tanh"``, or
            ``"sigmoid"`` (``1 / (1 + exp(-z))``).
    """

    def __init__(self, transform: str = "identity", label: str = ""):
        if transform not in _VALID_TRANSFORMS:
            raise ValueError(
                f"Unknown transform {transform!r}. "
                f"Must be one of {_VALID_TRANSFORMS}."
            )
        self.transform = transform
        self.label = label or type(self).__name__

    def _transform(self, value: float) -> float:
        """Apply ``self.transform`` to *value* and return the result."""
        return _apply_transform(value, self.transform)

    @abstractmethod
    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        """Compute scalar from output traces of one trial.

        Args:
            traces: Output voltage traces, shape (n_timesteps, n_outputs).
            v_ref: Reference trace, shape (n_timesteps,).

        Returns:
            Transformed scalar summary statistic.
        """


class BatchScalarDescriptor(ABC):
    """Base class for scalar descriptors with a vectorized batch path.

    Subclasses implement ``compute_batch``, which maps a
    ``TsdTensor`` of shape (n_timesteps, n_outputs, n_trials) and a
    reference-channel index to a 1-D array of length n_trials.
    """

    @abstractmethod
    def compute_batch(self, tsdtensor: nap.TsdTensor, ref_ch: int) -> np.ndarray:
        """Compute scalar for every trial in a batch.

        Args:
            tsdtensor: Output traces for all trials,
                shape (n_timesteps, n_outputs, n_trials).
            ref_ch: Index of the reference channel.

        Returns:
            Array of scalars, shape (n_trials,).
        """


# ---------------------------------------------------------------------------
# Reference-trace descriptors  (operate on v_ref)
# ---------------------------------------------------------------------------


class ReferenceMean(ScalarDescriptor):
    """Mean of the reference trace over time."""

    def __init__(self, transform: str = "identity", label: str = "ref_mean"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        return self._transform(float(np.mean(v_ref)))


class ReferenceStd(ScalarDescriptor):
    """Standard deviation of the reference trace over time."""

    def __init__(self, transform: str = "log", label: str = "ref_std"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        return self._transform(float(np.std(v_ref)))


class ReferenceSpectralCentroid(ScalarDescriptor):
    """Spectral centroid of the reference trace.

    Args:
        dt_ms: Sampling interval in milliseconds.
        transform: Output transform (default ``"log"``).
    """

    def __init__(
        self, dt_ms: float = 1.0, transform: str = "log", label: str = "ref_SC"
    ):
        super().__init__(transform=transform, label=label)
        self.dt_ms = dt_ms

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        return self._transform(_spectral_centroid(v_ref, self.dt_ms))


# ---------------------------------------------------------------------------
# Residual descriptors  (operate on R = V - v_ref, excluding ref channel)
# ---------------------------------------------------------------------------


class ResidualMean(ScalarDescriptor):
    """Mean across channels of the time-mean of each residual R_i."""

    def __init__(self, transform: str = "identity", label: str = "resid_mean"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        if arr.shape[1] < 2:
            return self._transform(0.0)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(0.0)
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        return self._transform(float(np.mean(np.mean(R_other, axis=0))))


class ResidualStd(ScalarDescriptor):
    """Mean across channels of the std of each residual R_i."""

    def __init__(self, transform: str = "log", label: str = "resid_std"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(0.0)
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        return self._transform(float(np.mean(np.std(R_other, axis=0))))


class ResidualEnergy(ScalarDescriptor):
    """Total residual energy: sum_i integral dt (R_i(t))^2.

    Args:
        dt_ms: Sampling interval in milliseconds (scales the integral).
        transform: Output transform (default ``"log"``).
    """

    def __init__(
        self, dt_ms: float = 1.0, transform: str = "log", label: str = "resid_energy"
    ):
        super().__init__(transform=transform, label=label)
        self.dt_ms = dt_ms

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(0.0)
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        return self._transform(float(np.sum(R_other**2) * self.dt_ms))


class ResidualSpectralCentroidMean(ScalarDescriptor):
    """Mean spectral centroid across residual channels R_i.

    Args:
        dt_ms: Sampling interval in milliseconds.
        transform: Output transform (default ``"log"``).
    """

    def __init__(
        self, dt_ms: float = 1.0, transform: str = "log", label: str = "resid_SC_mean"
    ):
        super().__init__(transform=transform, label=label)
        self.dt_ms = dt_ms

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(0.0)
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        centroids = [
            _spectral_centroid(R_other[:, j], self.dt_ms)
            for j in range(R_other.shape[1])
        ]
        return self._transform(float(np.mean(centroids)))


class ResidualSpectralCentroidStd(ScalarDescriptor):
    """Std of spectral centroids across residual channels R_i.

    Args:
        dt_ms: Sampling interval in milliseconds.
        transform: Output transform (default ``"log"``).
    """

    def __init__(
        self, dt_ms: float = 1.0, transform: str = "log", label: str = "resid_SC_std"
    ):
        super().__init__(transform=transform, label=label)
        self.dt_ms = dt_ms

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(0.0)
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        centroids = [
            _spectral_centroid(R_other[:, j], self.dt_ms)
            for j in range(R_other.shape[1])
        ]
        return self._transform(float(np.std(centroids)))


# ---------------------------------------------------------------------------
# Covariance of R descriptors  (C_R = R^T R / T)
# ---------------------------------------------------------------------------


def _residual_cov_eigenvalues(arr: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """Eigenvalues of C_R = R^T R / T, descending, non-negative."""
    other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
    if not other:
        return np.array([0.0])
    R = arr[:, other] - v_ref[:, np.newaxis]
    T = R.shape[0]
    C_R = (R.T @ R) / T
    eigvals = np.linalg.eigvalsh(C_R)
    eigvals = np.clip(eigvals, 0, None)
    return np.sort(eigvals)[::-1]


class ResidualParticipationRatio(ScalarDescriptor):
    """Participation ratio of C_R eigenspectrum: (sum λ)^2 / sum λ^2.

    Measures the effective number of dimensions in the residual activity.
    """

    def __init__(self, transform: str = "log", label: str = "part_ratio"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        eigvals = _residual_cov_eigenvalues(np.asarray(traces), v_ref)
        s1 = float(np.sum(eigvals))
        s2 = float(np.sum(eigvals**2))
        if s2 == 0:
            return self._transform(0.0)
        return self._transform(float(s1**2 / s2))


class ResidualMaxEigenvalue(ScalarDescriptor):
    """Largest eigenvalue of C_R = R^T R / T."""

    def __init__(self, transform: str = "log", label: str = "max_eigval"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        eigvals = _residual_cov_eigenvalues(np.asarray(traces), v_ref)
        return self._transform(float(eigvals[0]))


class ResidualEigenvalueEntropy(ScalarDescriptor):
    """Shannon entropy of the normalised C_R eigenspectrum.

    A uniform eigenspectrum (maximum entropy) means all residual dimensions
    contribute equally.
    """

    def __init__(self, transform: str = "identity", label: str = "eigval_entropy"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        eigvals = _residual_cov_eigenvalues(np.asarray(traces), v_ref)
        total = np.sum(eigvals)
        if total == 0:
            return self._transform(0.0)
        p = eigvals / total
        p = p[p > 0]
        return self._transform(float(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Cross-channel descriptors
# ---------------------------------------------------------------------------


class CrossCorrelationMean(ScalarDescriptor):
    """Mean of off-diagonal elements of the output correlation matrix of V.

    Captures average linear co-variation across all output channels.
    """

    def __init__(self, transform: str = "identity", label: str = "xcorr_mean"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        corr = np.corrcoef(arr.T)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        return self._transform(float(np.mean(corr[mask])))


class TransmissionEfficiency(ScalarDescriptor):
    """Var(v_ref) / sum_i Var(R_i).

    Close to 1 → output dominates; small → residual (latent) activity
    dominates.
    """

    def __init__(self, transform: str = "log", label: str = "trans_eff"):
        super().__init__(transform=transform, label=label)

    def compute(self, traces: nap.TsdFrame, v_ref: np.ndarray) -> float:
        arr = np.asarray(traces)
        other = [i for i in range(arr.shape[1]) if not np.array_equal(arr[:, i], v_ref)]
        if not other:
            return self._transform(float("inf"))
        R_other = arr[:, other] - v_ref[:, np.newaxis]
        var_ref = float(np.var(v_ref))
        sum_var_R = float(np.sum(np.var(R_other, axis=0)))
        if sum_var_R == 0:
            return self._transform(float("inf"))
        return self._transform(var_ref / sum_var_R)
