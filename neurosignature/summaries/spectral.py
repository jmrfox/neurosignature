"""Spectral statistics using FFT-based analysis."""

import numpy as np
from typing import Dict, List, Optional
from neurosignature.math import safe_normalize


def compute_spectral_statistics(
    traces: np.ndarray,
    dt_ms: float = 1.0,
    n_dominant_freqs: int = 5,
    freq_bands: Optional[List[tuple]] = None,
) -> Dict[str, np.ndarray]:
    """Compute FFT-based spectral descriptors.

    Args:
        traces: Output traces, shape (n_timesteps, n_channels)
        dt_ms: Sampling interval in milliseconds
        n_dominant_freqs: Number of dominant frequencies to extract
        freq_bands: List of (low, high) frequency band tuples in Hz.
            If None, uses [(0, 10), (10, 50), (50, 100)] Hz

    Returns:
        Dictionary with spectral statistics:
        - dominant_frequencies: Top N frequencies per channel
        - spectral_centroid: Centroid of power spectrum per channel
        - spectral_entropy: Entropy of normalized spectrum per channel
        - band_powers: Power in each frequency band per channel
        - band_ratios: Ratios between low/high frequency power
    """
    n_steps, n_channels = traces.shape
    dt_s = dt_ms / 1000.0
    sample_rate = 1.0 / dt_s

    # Frequency axis
    freqs = np.fft.rfftfreq(n_steps, dt_s)

    # Default frequency bands in Hz
    if freq_bands is None:
        freq_bands = [(0, 10), (10, 50), (50, 100)]

    # Storage
    dominant_frequencies = []
    spectral_centroids = []
    spectral_entropies = []
    band_powers_all = []

    for ch in range(n_channels):
        # Compute power spectrum
        fft_vals = np.fft.rfft(traces[:, ch])
        power = np.abs(fft_vals) ** 2

        # Skip DC component for some statistics
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        # Dominant frequencies (indices of top power values)
        if len(power_no_dc) > 0:
            top_indices = np.argsort(power_no_dc)[-n_dominant_freqs:][::-1]
            dominant = freqs_no_dc[top_indices]
        else:
            dominant = np.zeros(n_dominant_freqs)
        dominant_frequencies.append(dominant)

        # Spectral centroid (weighted mean frequency)
        if np.sum(power_no_dc) > 0:
            centroid = np.sum(freqs_no_dc * power_no_dc) / np.sum(power_no_dc)
        else:
            centroid = 0.0
        spectral_centroids.append(centroid)

        # Spectral entropy
        if np.sum(power_no_dc) > 0:
            power_norm = safe_normalize(power_no_dc)
            entropy = -np.sum(power_norm * np.log2(power_norm))
        else:
            entropy = 0.0
        spectral_entropies.append(entropy)

        # Band powers
        band_powers = []
        for low_hz, high_hz in freq_bands:
            # Find frequency indices in band
            mask = (freqs >= low_hz) & (freqs < high_hz)
            if np.any(mask):
                power_in_band = np.sum(power[mask])
            else:
                power_in_band = 0.0
            band_powers.append(power_in_band)
        band_powers_all.append(band_powers)

    # Convert to arrays
    dominant_frequencies = np.array(dominant_frequencies)  # (n_channels, n_dominant)
    spectral_centroids = np.array(spectral_centroids)  # (n_channels,)
    spectral_entropies = np.array(spectral_entropies)  # (n_channels,)
    band_powers = np.array(band_powers_all)  # (n_channels, n_bands)

    # Compute band ratios (low/high frequency ratios)
    if band_powers.shape[1] >= 2:
        # Ratio of lowest band to highest band
        band_ratios = band_powers[:, 0] / (band_powers[:, -1] + 1e-12)
    else:
        band_ratios = np.ones(n_channels)

    return {
        "dominant_frequencies": dominant_frequencies,
        "spectral_centroid": spectral_centroids,
        "spectral_entropy": spectral_entropies,
        "band_powers": band_powers,
        "band_ratios": band_ratios,
    }


def concatenate_spectral_descriptors(
    spectral_stats: Dict[str, np.ndarray],
    include_dominant: bool = True,
    include_bands: bool = True,
) -> np.ndarray:
    """Concatenate spectral statistics into descriptor vector.

    Args:
        spectral_stats: Output from compute_spectral_statistics
        include_dominant: Whether to include dominant frequencies
        include_bands: Whether to include band powers

    Returns:
        Concatenated spectral descriptor vector
    """
    descriptors = []

    # Spectral centroids
    descriptors.append(spectral_stats["spectral_centroid"])

    # Spectral entropies
    descriptors.append(spectral_stats["spectral_entropy"])

    # Band ratios (low/high frequency)
    descriptors.append(spectral_stats["band_ratios"])

    if include_dominant:
        # Flatten dominant frequencies
        descriptors.append(spectral_stats["dominant_frequencies"].flatten())

    if include_bands:
        # Flatten band powers
        descriptors.append(spectral_stats["band_powers"].flatten())

    return np.concatenate([d.flatten() for d in descriptors])
