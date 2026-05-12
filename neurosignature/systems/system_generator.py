"""Factory for generating dynamical systems with varying properties."""

import numpy as np
from typing import Optional, List
from .recurrent_system import ContinuousTimeRNN


class SystemGenerator:
    """Generate collections of dynamical systems with controlled variation.

    Supports generating systems with different:
    - Spectral radius
    - Sparsity levels
    - Timescale distributions
    - Modular structure
    """

    def __init__(
        self,
        n_hidden: int = 64,
        n_inputs: int = 25,
        n_outputs: int = 32,
        base_seed: int = 42,
    ):
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.base_seed = base_seed

    def generate_random_system(
        self,
        spectral_radius: float = 0.8,
        sparsity: float = 0.1,
        tau_range: tuple = (10.0, 100.0),
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Generate a single random system.

        Args:
            spectral_radius: Target spectral radius for recurrent weights
            sparsity: Sparsity level for recurrent connections
            tau_range: (min, max) timescale in ms
            seed: Random seed (auto-generated if None)

        Returns:
            Configured ContinuousTimeRNN instance
        """
        if seed is None:
            seed = self.base_seed + np.random.randint(0, 1000000)

        tau = np.random.uniform(tau_range[0], tau_range[1], size=self.n_hidden)

        return ContinuousTimeRNN(
            n_hidden=self.n_hidden,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            tau=tau,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            seed=seed,
        )

    def generate_spectral_radius_sweep(
        self,
        radii: List[float],
        sparsity: float = 0.1,
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying spectral radii.

        Args:
            radii: List of spectral radius values to test
            sparsity: Fixed sparsity level

        Returns:
            List of systems, one per radius value
        """
        systems = []
        for i, radius in enumerate(radii):
            system = self.generate_random_system(
                spectral_radius=radius,
                sparsity=sparsity,
                seed=self.base_seed + i,
            )
            systems.append(system)
        return systems

    def generate_sparsity_sweep(
        self,
        sparsities: List[float],
        spectral_radius: float = 0.8,
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying sparsity levels.

        Args:
            sparsities: List of sparsity values [0, 1]
            spectral_radius: Fixed spectral radius

        Returns:
            List of systems, one per sparsity value
        """
        systems = []
        for i, sparsity in enumerate(sparsities):
            system = self.generate_random_system(
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                seed=self.base_seed + 1000 + i,
            )
            systems.append(system)
        return systems

    def generate_timescale_sweep(
        self,
        tau_ranges: List[tuple],
        spectral_radius: float = 0.8,
        sparsity: float = 0.1,
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying timescale distributions.

        Args:
            tau_ranges: List of (min_tau, max_tau) tuples in ms
            spectral_radius: Fixed spectral radius
            sparsity: Fixed sparsity

        Returns:
            List of systems, one per timescale range
        """
        systems = []
        for i, tau_range in enumerate(tau_ranges):
            tau = np.random.uniform(tau_range[0], tau_range[1], size=self.n_hidden)
            system = ContinuousTimeRNN(
                n_hidden=self.n_hidden,
                n_inputs=self.n_inputs,
                n_outputs=self.n_outputs,
                tau=tau,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                seed=self.base_seed + 2000 + i,
            )
            systems.append(system)
        return systems

    def generate_ensemble(
        self,
        n_systems: int,
        spectral_radius_range: tuple = (0.5, 0.95),
        sparsity_range: tuple = (0.05, 0.3),
    ) -> List[ContinuousTimeRNN]:
        """Generate diverse ensemble of random systems.

        Args:
            n_systems: Number of systems to generate
            spectral_radius_range: (min, max) spectral radius
            sparsity_range: (min, max) sparsity level

        Returns:
            List of randomly varied systems
        """
        systems = []
        for i in range(n_systems):
            radius = np.random.uniform(*spectral_radius_range)
            sparsity = np.random.uniform(*sparsity_range)
            system = self.generate_random_system(
                spectral_radius=radius,
                sparsity=sparsity,
                seed=self.base_seed + 3000 + i,
            )
            systems.append(system)
        return systems
