"""System comparison pipeline."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance_nd

from neurosignature.inputs.input_generator import InputGenerator
from neurosignature.simulation.simulator import Simulator
from neurosignature.summaries.descriptors import VectorDescriptor

_VALID_MODES = ("cross", "matched")


class SystemComparator:
    """Compare descriptor distributions produced by two simulators.

    Both simulators receive independently drawn inputs from the same
    ``input_gen`` on each trial.  A single shared ``VectorDescriptor``
    maps each trial output to a descriptor vector ``z``.

    Args:
        input_gen: Input event stream generator shared by both systems.
        simulator_a: First simulator (system A).
        simulator_b: Second simulator (system B).
        descriptor: Shared ``VectorDescriptor`` applied to both systems.
    """

    def __init__(
        self,
        input_gen: InputGenerator,
        simulator_a: Simulator,
        simulator_b: Simulator,
        descriptor: VectorDescriptor,
    ):
        self.input_gen = input_gen
        self.simulator_a = simulator_a
        self.simulator_b = simulator_b
        self.descriptor = descriptor

    def run(self, n_trials: int, duration_ms: float) -> tuple:
        """Run ``n_trials`` for each system and collect descriptor matrices.

        Args:
            n_trials: Number of independent input realisations per system.
            duration_ms: Duration of each trial in milliseconds.

        Returns:
            Tuple ``(Z_A, Z_B)`` of descriptor matrices, each shape
            ``(n_trials, descriptor_dim)``.
        """
        Z_A, Z_B = [], []
        for _ in range(n_trials):
            ts = self.input_gen.generate(duration_ms)
            Z_A.append(self.descriptor.compute(self.simulator_a.run(ts)))
            ts = self.input_gen.generate(duration_ms)
            Z_B.append(self.descriptor.compute(self.simulator_b.run(ts)))
        return np.array(Z_A), np.array(Z_B)

    def euclidean_distances(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        mode: str = "cross",
    ) -> dict:
        """Euclidean distances between descriptor vectors in Z_A and Z_B.

        Args:
            Z_A: Descriptor matrix for system A, shape (n_A, dim).
            Z_B: Descriptor matrix for system B, shape (n_B, dim).
            mode: ``"cross"`` (all n_A × n_B pairs) or ``"matched"``
                (element-wise; requires n_A == n_B).

        Returns:
            Dict with keys ``"distances"`` (1-D array), ``"mean"``,
            and ``"std"``.
        """
        dists = self._pairwise(Z_A, Z_B, metric="euclidean", mode=mode)
        return {
            "distances": dists,
            "mean": float(dists.mean()),
            "std": float(dists.std()),
        }

    def cosine_distances(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        mode: str = "cross",
    ) -> dict:
        """Cosine distances between descriptor vectors in Z_A and Z_B.

        Args:
            Z_A: Descriptor matrix for system A, shape (n_A, dim).
            Z_B: Descriptor matrix for system B, shape (n_B, dim).
            mode: ``"cross"`` (all n_A × n_B pairs) or ``"matched"``
                (element-wise; requires n_A == n_B).

        Returns:
            Dict with keys ``"distances"`` (1-D array), ``"mean"``,
            and ``"std"``.
        """
        dists = self._pairwise(Z_A, Z_B, metric="cosine", mode=mode)
        return {
            "distances": dists,
            "mean": float(dists.mean()),
            "std": float(dists.std()),
        }

    def wasserstein_distance(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
    ) -> float:
        """n-D Wasserstein distance between the two descriptor clouds.

        Uses ``scipy.stats.wasserstein_distance_nd``, treating each row
        of ``Z_A`` / ``Z_B`` as a sample from the respective distribution.

        Args:
            Z_A: Descriptor matrix for system A, shape (n_A, dim).
            Z_B: Descriptor matrix for system B, shape (n_B, dim).

        Returns:
            Scalar Wasserstein distance.
        """
        return float(wasserstein_distance_nd(Z_A, Z_B))

    def compare(self, n_trials: int, duration_ms: float, mode: str = "cross") -> dict:
        """Run both systems and return all pairwise distance metrics.

        Args:
            n_trials: Number of independent trials per system.
            duration_ms: Trial duration in milliseconds.
            mode: Pairwise comparison mode — "cross" or "matched".

        Returns:
            Dict with keys:

            - ``"Z_A"``, ``"Z_B"``: descriptor matrices
            - ``"euclidean"``: dict with ``"distances"``, ``"mean"``, ``"std"``
            - ``"cosine"``:    dict with ``"distances"``, ``"mean"``, ``"std"``
            - ``"wasserstein"``: scalar float
        """
        Z_A, Z_B = self.run(n_trials, duration_ms)
        return {
            "Z_A": Z_A,
            "Z_B": Z_B,
            "euclidean": self.euclidean_distances(Z_A, Z_B, mode=mode),
            "cosine": self.cosine_distances(Z_A, Z_B, mode=mode),
            "wasserstein": self.wasserstein_distance(Z_A, Z_B),
        }

    def _pairwise(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        metric: str,
        mode: str,
    ) -> np.ndarray:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Must be one of {_VALID_MODES}.")
        if mode == "matched":
            if Z_A.shape[0] != Z_B.shape[0]:
                raise ValueError(
                    "matched mode requires equal n_trials, got "
                    f"{Z_A.shape[0]} and {Z_B.shape[0]}."
                )
            return np.array(
                [
                    float(cdist(Z_A[i : i + 1], Z_B[i : i + 1], metric=metric)[0, 0])
                    for i in range(Z_A.shape[0])
                ]
            )
        D = cdist(Z_A, Z_B, metric=metric)
        return D.ravel()
