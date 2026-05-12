"""Embedding computations for descriptor space visualization."""

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_pca(
    descriptors: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, PCA]:
    """Compute PCA embedding of descriptors.

    Args:
        descriptors: Descriptor matrix, shape (n_systems, descriptor_dim)
        n_components: Number of PCA components

    Returns:
        Tuple of (embedded coordinates, fitted PCA object)
    """
    pca = PCA(n_components=n_components)
    embedded = pca.fit_transform(descriptors)
    return embedded, pca


def compute_tsne(
    descriptors: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE embedding of descriptors.

    Args:
        descriptors: Descriptor matrix, shape (n_systems, descriptor_dim)
        n_components: Number of dimensions (2 or 3)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed

    Returns:
        Embedded coordinates, shape (n_systems, n_components)
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, descriptors.shape[0] - 1),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(descriptors)
    return embedded


def compute_umap(
    descriptors: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """Compute UMAP embedding of descriptors.

    Args:
        descriptors: Descriptor matrix, shape (n_systems, descriptor_dim)
        n_components: Number of dimensions
        n_neighbors: Number of neighbors for local manifold approximation
        min_dist: Minimum distance between embedded points
        random_state: Random seed

    Returns:
        Embedded coordinates or None if UMAP not installed
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, descriptors.shape[0] - 1),
            min_dist=min_dist,
            random_state=random_state,
        )
        embedded = reducer.fit_transform(descriptors)
        return embedded
    except ImportError:
        print("Warning: umap-learn not installed. Install with: uv add umap-learn")
        return None


def plot_embedding(
    embedded: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Descriptor Space Embedding",
    figsize: tuple = (8, 7),
    cmap: str = "viridis",
    alpha: float = 0.7,
) -> "matplotlib.figure.Figure":
    """Plot 2D embedding of descriptors.

    Args:
        embedded: Embedded coordinates, shape (n_systems, 2)
        labels: Optional labels for coloring points
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        alpha: Point transparency

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(
            embedded[:, 0], embedded[:, 1],
            c=labels, cmap=cmap, alpha=alpha, s=50, edgecolors="white", linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(
            embedded[:, 0], embedded[:, 1],
            alpha=alpha, s=50, edgecolors="white", linewidth=0.5
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_embedding_comparison(
    descriptors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    methods: list = ["pca", "tsne"],
    figsize: tuple = (14, 6),
) -> "matplotlib.figure.Figure":
    """Compare multiple embedding methods side by side.

    Args:
        descriptors: Descriptor matrix
        labels: Optional labels for coloring
        methods: List of methods to compare ("pca", "tsne", "umap")
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        if method == "pca":
            embedded, pca = compute_pca(descriptors)
            title = f"PCA ({pca.explained_variance_ratio_[0]:.2f}, " \
                    f"{pca.explained_variance_ratio_[1]:.2f})"
        elif method == "tsne":
            embedded = compute_tsne(descriptors)
            title = "t-SNE"
        elif method == "umap":
            embedded = compute_umap(descriptors)
            if embedded is None:
                ax.text(0.5, 0.5, "UMAP not installed",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("UMAP")
                continue
            title = "UMAP"
        else:
            ax.text(0.5, 0.5, f"Unknown method: {method}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        if labels is not None:
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1],
                c=labels, cmap="viridis", alpha=0.7, s=50,
                edgecolors="white", linewidth=0.5
            )
        else:
            ax.scatter(
                embedded[:, 0], embedded[:, 1],
                alpha=0.7, s=50, edgecolors="white", linewidth=0.5
            )

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
