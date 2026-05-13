"""Common mathematical utility functions."""

import numpy as np
from typing import Optional


def relu(x):
    """Rectified Linear Unit: max(0, x)."""
    return np.maximum(0, x)


def softplus(x):
    """Softplus: log(1 + exp(x)). Smooth approximation to ReLU."""
    return np.log(1 + np.exp(x))


def generate_random_matrix(
    m: int,
    n: int,
    dist: str = "normal",
    matrix_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a random matrix with specified distribution and structure.

    Args:
        m: Number of rows
        n: Number of columns
        dist: Entry sampling distribution - "normal", "uniform", "binary",
            "laplace", "exponential", "bernoulli"
        matrix_type: Structure type - None (dense/general), "diagonal",
            "tridiagonal", "upper_triangular", "lower_triangular",
            "orthogonal", "symmetric", "toeplitz"
        seed: Random seed for reproducibility

    Returns:
        Random matrix of shape (m, n)

    Example:
        >>> W = generate_random_matrix(
        ...     10, 10, dist="normal", matrix_type="symmetric")
        >>> W.shape
        (10, 10)
        >>> np.allclose(W, W.T)
        True
    """
    rng = np.random.default_rng(seed)

    # Generate base matrix from distribution
    if dist == "normal":
        base = rng.standard_normal((m, n))
    elif dist == "uniform":
        base = rng.uniform(-1, 1, (m, n))
    elif dist == "binary":
        base = rng.integers(0, 2, (m, n)).astype(float)
    elif dist == "laplace":
        base = rng.laplace(0, 1, (m, n))
    elif dist == "exponential":
        base = rng.exponential(1, (m, n))
    elif dist == "bernoulli":
        base = (rng.random((m, n)) < 0.5).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Apply structure mask/transform
    if matrix_type is None or matrix_type == "dense":
        result = base
    elif matrix_type == "diagonal":
        # Square matrices only
        min_dim = min(m, n)
        result = np.zeros((m, n))
        result[:min_dim, :min_dim] = np.diag(np.diag(base[:min_dim, :min_dim]))
    elif matrix_type == "tridiagonal":
        # Square matrices only
        min_dim = min(m, n)
        result = np.zeros((m, n))
        for i in range(min_dim):
            for j in range(max(0, i - 1), min(min_dim, i + 2)):
                result[i, j] = base[i, j]
    elif matrix_type == "upper_triangular":
        result = np.triu(base)
    elif matrix_type == "lower_triangular":
        result = np.tril(base)
    elif matrix_type == "symmetric":
        # Square matrices only
        min_dim = min(m, n)
        sym_part = (base[:min_dim, :min_dim] + base[:min_dim, :min_dim].T) / 2
        result = np.zeros((m, n))
        result[:min_dim, :min_dim] = sym_part
    elif matrix_type == "skew_symmetric":
        # Square matrices only, W = -W.T
        min_dim = min(m, n)
        skew = (base[:min_dim, :min_dim] - base[:min_dim, :min_dim].T) / 2
        result = np.zeros((m, n))
        result[:min_dim, :min_dim] = skew
    elif matrix_type == "orthogonal":
        # Square matrices only - QR decomposition of random matrix
        min_dim = min(m, n)
        q, _ = np.linalg.qr(base[:min_dim, :min_dim])
        result = np.zeros((m, n))
        result[:min_dim, :min_dim] = q
    elif matrix_type == "toeplitz":
        # First row and column determine the matrix
        c = base[:m, 0]  # First column
        r = base[0, :n]  # First row
        from scipy.linalg import toeplitz

        result = toeplitz(c, r)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    return result


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix: (A + A^T) / 2.

    Args:
        A: Square matrix of shape (n, n)

    Returns:
        Symmetric matrix of same shape
    """
    return (A + A.T) / 2.0


def make_sparse_mask(
    shape: tuple,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a boolean sparsity mask.

    Args:
        shape: Shape of the mask array
        sparsity: Fraction of entries set to True (connection probability)
        rng: NumPy random generator

    Returns:
        Boolean mask array of given shape
    """
    return rng.random(shape) < sparsity


def scale_spectral_radius(W: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale a matrix so its spectral radius equals target_radius.

    If the current spectral radius is zero, returns W unchanged.

    Args:
        W: Square matrix to rescale
        target_radius: Desired spectral radius (max absolute eigenvalue)

    Returns:
        Rescaled matrix with spectral radius equal to target_radius
    """
    eigvals = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigvals))
    if current_radius > 0:
        W = W * (target_radius / current_radius)
    return W


def sample_loguniform(
    low: float,
    high: float,
    size,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw samples from a log-uniform distribution over [low, high].

    Args:
        low: Lower bound (exclusive of 0)
        high: Upper bound
        size: Output shape, as accepted by rng.uniform
        rng: NumPy random generator

    Returns:
        Array of samples in [low, high] drawn log-uniformly
    """
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def alpha_kernel(t_ms: np.ndarray, tau_s: float) -> np.ndarray:
    """Compute alpha-function synaptic kernel values.

    α(t) = H(t) · (t / τ_s) · exp(-t / τ_s)

    where H(t) is the Heaviside step function.

    Args:
        t_ms: Time array in milliseconds (may include negative values)
        tau_s: Synaptic decay time constant in milliseconds

    Returns:
        Kernel values at each time point, shape same as t_ms
    """
    t = np.asarray(t_ms, dtype=float)
    t_norm = t / tau_s
    alpha = t_norm * np.exp(-t_norm)
    alpha[t < 0] = 0.0
    return alpha


def safe_normalize(x: np.ndarray, clip_min: float = 1e-12) -> np.ndarray:
    """Normalize array to sum to 1, clipping small values to avoid log(0).

    Useful for converting a power spectrum into a probability distribution
    before computing entropy.

    Args:
        x: Non-negative array to normalize
        clip_min: Minimum value after normalization (applied via np.clip)

    Returns:
        Normalized array clipped to [clip_min, 1.0], same shape as x.
        Returns zeros if x sums to zero.
    """
    total = np.sum(x)
    if total <= 0:
        return np.zeros_like(x, dtype=float)
    normed = x / total
    return np.clip(normed, clip_min, 1.0)
