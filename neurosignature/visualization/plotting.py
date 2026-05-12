"""Plotting utilities for traces and distance matrices."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_traces(
    traces: np.ndarray,
    dt_ms: float = 1.0,
    title: str = "Output Traces",
    n_channels_to_plot: Optional[int] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot output traces as a heatmap and sample traces.

    Args:
        traces: Output traces, shape (n_timesteps, n_channels)
        dt_ms: Time step in milliseconds
        title: Plot title
        n_channels_to_plot: Number of individual traces to overlay (default: min(5, n_channels))
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_steps, n_channels = traces.shape
    time_ms = np.arange(n_steps) * dt_ms

    if n_channels_to_plot is None:
        n_channels_to_plot = min(5, n_channels)

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # Heatmap
    ax = axes[0]
    im = ax.imshow(
        traces.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.percentile(np.abs(traces), 99),
        vmax=np.percentile(np.abs(traces), 99),
        extent=[0, time_ms[-1], 0, n_channels],
    )
    ax.set_ylabel("Channel")
    ax.set_title(f"{title} (Heatmap)")
    plt.colorbar(im, ax=ax, label="Amplitude")

    # Sample traces
    ax = axes[1]
    channel_indices = np.linspace(0, n_channels - 1, n_channels_to_plot, dtype=int)
    for i in channel_indices:
        offset = i * 3  # Offset for visibility
        ax.plot(time_ms, traces[:, i] + offset, label=f"Ch {i}", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channel (offset)")
    ax.set_title(f"Sample Traces (n={n_channels_to_plot})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, time_ms[-1]])

    plt.tight_layout()
    return fig


def plot_distance_matrix(
    distance_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Pairwise Distance Matrix",
    figsize: tuple = (8, 7),
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot pairwise distance matrix as a heatmap.

    Args:
        distance_matrix: Square distance matrix
        labels: Optional labels for systems
        title: Plot title
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    n = distance_matrix.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(distance_matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("System Index")
    ax.set_ylabel("System Index")

    if labels is not None and len(labels) == n:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Distance")

    # Add values as text for small matrices
    if n <= 10:
        for i in range(n):
            for j in range(n):
                text = ax.text(
                    j, i, f"{distance_matrix[i, j]:.2f}",
                    ha="center", va="center", color="white", fontsize=8
                )

    plt.tight_layout()
    return fig


def plot_descriptor_trajectory(
    parameter_values: np.ndarray,
    descriptors: np.ndarray,
    parameter_name: str = "Parameter",
    n_components_to_plot: int = 3,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot how descriptors change along a parameter sweep.

    Args:
        parameter_values: Parameter values for each descriptor
        descriptors: Descriptor matrix, shape (n_params, descriptor_dim)
        parameter_name: Name of parameter being swept
        n_components_to_plot: Number of descriptor components to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_params, desc_dim = descriptors.shape

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot individual descriptor components
    ax = axes[0]
    for i in range(min(n_components_to_plot, desc_dim)):
        ax.plot(parameter_values, descriptors[:, i], marker="o", label=f"Comp {i}")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Descriptor Value")
    ax.set_title(f"Descriptor Components vs {parameter_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot descriptor norm change
    ax = axes[1]
    descriptor_norms = np.linalg.norm(descriptors, axis=1)
    ax.plot(parameter_values, descriptor_norms, marker="o", color="black", linewidth=2)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("||Descriptor||")
    ax.set_title(f"Descriptor Norm vs {parameter_name}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_system_comparison_summary(
    group_names: List[str],
    within_distances: List[float],
    between_distances: List[float],
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot summary of within-group vs between-group distances.

    Args:
        group_names: Names of the groups
        within_distances: List of mean within-group distances
        between_distances: Mean between-group distance
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(group_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, within_distances, width, label="Within-group", alpha=0.8)
    bars2 = ax.bar(x + width/2, [between_distances] * len(group_names), width,
                   label="Between-groups", alpha=0.8)

    ax.set_ylabel("Mean Distance")
    ax.set_title("Within-group vs Between-group Distances")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
