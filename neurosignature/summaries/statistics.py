"""Basic and global summary statistics for output traces."""

import numpy as np
import pynapple as nap
from scipy import stats
from typing import Dict, Union


def compute_channel_statistics(
    traces: Union[np.ndarray, nap.TsdFrame],
) -> Dict[str, np.ndarray]:
    """Compute per-channel summary statistics.

    Args:
        traces: Output traces, shape (n_timesteps, n_channels)

    Returns:
        Dictionary with per-channel statistics:
        - mean, variance, std, min, max
        - rms (root-mean-square)
        - skewness, kurtosis
    """
    traces = np.asarray(traces)
    return {
        "mean": np.mean(traces, axis=0),
        "variance": np.var(traces, axis=0),
        "std": np.std(traces, axis=0),
        "min": np.min(traces, axis=0),
        "max": np.max(traces, axis=0),
        "rms": np.sqrt(np.mean(traces**2, axis=0)),
        "skewness": stats.skew(traces, axis=0),
        "kurtosis": stats.kurtosis(traces, axis=0),
    }


def compute_global_statistics(
    traces: Union[np.ndarray, nap.TsdFrame],
    max_lag: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute global (cross-channel) summary statistics.

    Args:
        traces: Output traces, shape (n_timesteps, n_channels)
        max_lag: Maximum lag for autocorrelation computation

    Returns:
        Dictionary with global statistics:
        - correlation_matrix: Pairwise correlations between channels
        - covariance_eigenvalues: Eigenvalues of covariance matrix
        - autocorrelation_decay: Autocorrelation decay per channel
        - approximate_entropy: Approximate entropy estimate
    """
    traces = np.asarray(traces)
    n_steps, n_channels = traces.shape

    # Pairwise correlation matrix
    # Standardize each channel
    std = np.std(traces, axis=0) + 1e-10
    standardized = (traces - np.mean(traces, axis=0)) / std
    correlation_matrix = np.corrcoef(standardized.T)

    # Covariance eigenvalues
    cov_matrix = np.cov(traces.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Autocorrelation decay (average across channels)
    autocorr_values = []
    for ch in range(n_channels):
        trace = traces[:, ch]
        mean_trace = trace - np.mean(trace)

        autocorr = np.correlate(mean_trace, mean_trace, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if len(autocorr) > 1 and autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Take first max_lag values
        autocorr_lags = autocorr[: min(max_lag, len(autocorr))]
        autocorr_values.append(autocorr_lags)

    autocorr_mean = np.mean(autocorr_values, axis=0)

    # Approximate entropy (simplified)
    # Using coefficient of variation as proxy for complexity
    mean_abs_diff = np.mean(np.abs(np.diff(traces, axis=0)), axis=0)
    approximate_entropy = mean_abs_diff / (np.std(traces, axis=0) + 1e-10)

    return {
        "correlation_matrix": correlation_matrix,
        "covariance_eigenvalues": eigenvalues,
        "autocorrelation_mean": autocorr_mean,
        "approximate_entropy": approximate_entropy,
    }


def concatenate_statistics(
    channel_stats: Dict[str, np.ndarray],
    global_stats: Dict[str, np.ndarray],
    top_k_eigenvalues: int = 10,
    n_autocorr_lags: int = 10,
) -> np.ndarray:
    """Concatenate all statistics into a single descriptor vector.

    Args:
        channel_stats: Output from compute_channel_statistics
        global_stats: Output from compute_global_statistics
        top_k_eigenvalues: Number of top covariance eigenvalues to include
        n_autocorr_lags: Number of autocorrelation lags to include

    Returns:
        Concatenated descriptor vector
    """
    descriptors = []

    # Add channel statistics (each is (n_channels,) array)
    for key in ["mean", "variance", "std", "rms", "skewness", "kurtosis"]:
        descriptors.append(channel_stats[key])

    # Add top eigenvalues
    eigenvalues = global_stats["covariance_eigenvalues"]
    descriptors.append(eigenvalues[:top_k_eigenvalues])

    # Add mean autocorrelation at selected lags
    autocorr = global_stats["autocorrelation_mean"]
    indices = np.linspace(0, len(autocorr) - 1, n_autocorr_lags, dtype=int)
    descriptors.append(autocorr[indices])

    # Add approximate entropy statistics
    descriptors.append(global_stats["approximate_entropy"])

    # Add correlation matrix statistics (mean, std of off-diagonal)
    corr = global_stats["correlation_matrix"]
    off_diag_mask = ~np.eye(corr.shape[0], dtype=bool)
    off_diag_corr = corr[off_diag_mask]
    descriptors.append(np.array([np.mean(off_diag_corr), np.std(off_diag_corr)]))

    return np.concatenate([d.flatten() for d in descriptors])


def compute_descriptor(
    traces: Union[np.ndarray, nap.TsdFrame],
    top_k_eigenvalues: int = 10,
    n_autocorr_lags: int = 10,
) -> np.ndarray:
    """Compute complete descriptor vector from traces in one step.

    Combines channel and global statistics into a single flat vector.
    This is a convenience wrapper around compute_channel_statistics,
    compute_global_statistics, and concatenate_statistics.

    Args:
        traces: Output traces, shape (n_timesteps, n_channels)
        top_k_eigenvalues: Number of top covariance eigenvalues (default: 10)
        n_autocorr_lags: Number of autocorrelation lags (default: 10)

    Returns:
        Complete descriptor vector

    Example:
        >>> traces = np.random.randn(1000, 8)  # 1000 steps, 8 channels
        >>> desc = compute_descriptor(traces)
        >>> desc.shape
        (74,)  # depends on n_channels and parameters
    """
    channel_stats = compute_channel_statistics(traces)
    global_stats = compute_global_statistics(traces, max_lag=n_autocorr_lags)

    return concatenate_statistics(
        channel_stats,
        global_stats,
        top_k_eigenvalues=top_k_eigenvalues,
        n_autocorr_lags=n_autocorr_lags,
    )
