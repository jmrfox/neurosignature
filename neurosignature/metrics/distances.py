"""Distance metrics for comparing system descriptors."""

import numpy as np
from scipy.spatial.distance import cosine, cdist
from typing import Optional


def euclidean_distance(
    desc1: np.ndarray,
    desc2: np.ndarray,
) -> float:
    """Compute Euclidean distance between descriptors.

    D = ||S₁ - S₂||₂

    Args:
        desc1: First descriptor vector
        desc2: Second descriptor vector

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(desc1 - desc2))


def cosine_distance(
    desc1: np.ndarray,
    desc2: np.ndarray,
) -> float:
    """Compute cosine distance between descriptors.

    D = 1 - cosine_similarity(S₁, S₂)

    Args:
        desc1: First descriptor vector
        desc2: Second descriptor vector

    Returns:
        Cosine distance (0 = identical direction, 1 = orthogonal)
    """
    return float(cosine(desc1, desc2))


def mahalanobis_distance(
    desc1: np.ndarray,
    desc2: np.ndarray,
    cov_inv: Optional[np.ndarray] = None,
) -> float:
    """Compute Mahalanobis distance between descriptors.

    D = sqrt((S₁ - S₂)ᵀ · Σ⁻¹ · (S₁ - S₂))

    Args:
        desc1: First descriptor vector
        desc2: Second descriptor vector
        cov_inv: Inverse covariance matrix. If None, computes from
            the two descriptors (not recommended for actual use).

    Returns:
        Mahalanobis distance
    """
    diff = desc1 - desc2

    if cov_inv is None:
        # Fallback: use identity (equivalent to Euclidean)
        return euclidean_distance(desc1, desc2)

    dist = np.sqrt(diff @ cov_inv @ diff)
    return float(dist)


def compute_pairwise_distance_matrix(
    descriptors: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute pairwise distance matrix for a set of descriptors.

    Args:
        descriptors: Descriptor matrix, shape (n_systems, descriptor_dim)
        metric: Distance metric ("euclidean", "cosine", "correlation")

    Returns:
        Distance matrix, shape (n_systems, n_systems)
    """
    if metric == "euclidean":
        # Vectorized computation
        return cdist(descriptors, descriptors, metric="euclidean")
    elif metric == "cosine":
        return cdist(descriptors, descriptors, metric="cosine")
    elif metric == "correlation":
        return cdist(descriptors, descriptors, metric="correlation")
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_distance_statistics(
    distance_matrix: np.ndarray,
) -> dict:
    """Compute statistics from a pairwise distance matrix.

    Args:
        distance_matrix: Square distance matrix

    Returns:
        Dictionary with distance statistics
    """
    # Get upper triangle (excluding diagonal)
    n = distance_matrix.shape[0]
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    distances = distance_matrix[mask]

    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
    }


def find_nearest_neighbors(
    distance_matrix: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """Find k-nearest neighbors for each descriptor.

    Args:
        distance_matrix: Square distance matrix
        n_neighbors: Number of neighbors to find

    Returns:
        Indices of nearest neighbors, shape (n_systems, n_neighbors)
    """
    n = distance_matrix.shape[0]
    neighbors = np.zeros((n, n_neighbors), dtype=int)

    for i in range(n):
        # Get distances to all other systems
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self

        # Find indices of n smallest distances
        neighbor_indices = np.argsort(dists)[:n_neighbors]
        neighbors[i] = neighbor_indices

    return neighbors
