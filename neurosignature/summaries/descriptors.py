"""Descriptor assembly for system characterization."""

import numpy as np
from typing import Optional, Dict
from .statistics import compute_channel_statistics, compute_global_statistics, concatenate_statistics
from .spectral import compute_spectral_statistics, concatenate_spectral_descriptors


class DescriptorAssembler:
    """Assemble complete system descriptors from output traces.

    Combines basic statistics, global statistics, and spectral statistics
    into a single fixed-dimensional descriptor vector.

    Args:
        top_k_eigenvalues: Number of top covariance eigenvalues to include
        n_autocorr_lags: Number of autocorrelation lags to include
        n_dominant_freqs: Number of dominant frequencies to include
        include_spectral: Whether to include spectral statistics
    """

    def __init__(
        self,
        top_k_eigenvalues: int = 10,
        n_autocorr_lags: int = 10,
        n_dominant_freqs: int = 5,
        include_spectral: bool = True,
    ):
        self.top_k_eigenvalues = top_k_eigenvalues
        self.n_autocorr_lags = n_autocorr_lags
        self.n_dominant_freqs = n_dominant_freqs
        self.include_spectral = include_spectral

    def compute_descriptor(
        self,
        traces: np.ndarray,
        dt_ms: float = 1.0,
    ) -> np.ndarray:
        """Compute full descriptor vector from output traces.

        Args:
            traces: Output traces, shape (n_timesteps, n_channels)
            dt_ms: Sampling interval in milliseconds

        Returns:
            Descriptor vector S(F_theta) ∈ R^K
        """
        descriptors = []

        # Basic channel statistics
        channel_stats = compute_channel_statistics(traces)
        global_stats = compute_global_statistics(traces)

        basic_descriptor = concatenate_statistics(
            channel_stats,
            global_stats,
            top_k_eigenvalues=self.top_k_eigenvalues,
            n_autocorr_lags=self.n_autocorr_lags,
        )
        descriptors.append(basic_descriptor)

        # Spectral statistics
        if self.include_spectral:
            spectral_stats = compute_spectral_statistics(
                traces,
                dt_ms=dt_ms,
                n_dominant_freqs=self.n_dominant_freqs,
            )
            spectral_descriptor = concatenate_spectral_descriptors(
                spectral_stats,
                include_dominant=True,
                include_bands=True,
            )
            descriptors.append(spectral_descriptor)

        return np.concatenate(descriptors)

    def compute_descriptors_batch(
        self,
        traces_batch: np.ndarray,
        dt_ms: float = 1.0,
    ) -> np.ndarray:
        """Compute descriptors for batch of trace arrays.

        Args:
            traces_batch: Array of trace arrays, shape (n_systems, n_timesteps, n_channels)
            dt_ms: Sampling interval in milliseconds

        Returns:
            Descriptor matrix, shape (n_systems, descriptor_dim)
        """
        n_systems = traces_batch.shape[0]

        # Compute first descriptor to get dimension
        first_desc = self.compute_descriptor(traces_batch[0], dt_ms)
        desc_dim = len(first_desc)

        descriptors = np.zeros((n_systems, desc_dim))
        descriptors[0] = first_desc

        for i in range(1, n_systems):
            descriptors[i] = self.compute_descriptor(traces_batch[i], dt_ms)

        return descriptors

    def get_descriptor_info(self, n_channels: int) -> Dict[str, int]:
        """Get information about descriptor composition.

        Args:
            n_channels: Number of output channels

        Returns:
            Dictionary with descriptor dimension breakdown
        """
        # Basic stats: mean, var, std, rms, skew, kurt per channel = 6 * n_channels
        basic_per_channel = 6

        # Global: top_k_eigenvalues + n_autocorr_lags + approx_entropy (per channel) +
        #          correlation stats (2)
        global_components = (
            self.top_k_eigenvalues
            + self.n_autocorr_lags
            + n_channels  # approximate entropy per channel
            + 2  # correlation mean/std
        )

        basic_dim = basic_per_channel * n_channels + global_components

        info = {
            "basic_statistics_dim": basic_per_channel * n_channels,
            "global_statistics_dim": global_components,
            "basic_total": basic_dim,
        }

        if self.include_spectral:
            # Spectral: centroids (n_channels) + entropies (n_channels) +
            #           band_ratios (n_channels) + dominant_freqs (n_channels * n_dominant) +
            #           band_powers (n_channels * n_bands, assume 3 bands)
            n_bands = 3
            spectral_dim = (
                n_channels  # centroids
                + n_channels  # entropies
                + n_channels  # band_ratios
                + n_channels * self.n_dominant_freqs
                + n_channels * n_bands
            )
            info["spectral_dim"] = spectral_dim
            info["total_dim"] = basic_dim + spectral_dim
        else:
            info["total_dim"] = basic_dim

        return info
