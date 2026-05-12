"""Visualization utilities for outputs and embeddings."""

from .plotting import plot_traces, plot_distance_matrix
from .embeddings import compute_pca, compute_tsne, compute_umap, plot_embedding

__all__ = [
    "plot_traces",
    "plot_distance_matrix",
    "compute_pca",
    "compute_tsne",
    "compute_umap",
    "plot_embedding",
]
