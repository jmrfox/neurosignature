"""System comparison pipeline for computing pairwise distances."""

import numpy as np
from typing import List, Dict, Optional
from neurosignature.systems.recurrent_system import ContinuousTimeRNN
from neurosignature.simulation.simulator import Simulator
from neurosignature.summaries.descriptors import DescriptorAssembler
from neurosignature.metrics.distances import (
    compute_pairwise_distance_matrix,
    compute_distance_statistics,
)


class SystemComparator:
    """Compare multiple dynamical systems using functional descriptors.

    Pipeline:
    1. Generate or receive multiple systems
    2. Run simulations with shared or varied inputs
    3. Compute descriptors for each system
    4. Compute pairwise distances
    5. Analyze distance structure

    Args:
        simulator: Simulator instance for running systems
        descriptor_assembler: DescriptorAssembler for computing descriptors
        dt_ms: Time step in milliseconds
    """

    def __init__(
        self,
        simulator: Simulator,
        descriptor_assembler: DescriptorAssembler,
        dt_ms: float = 1.0,
    ):
        self.simulator = simulator
        self.descriptor_assembler = descriptor_assembler
        self.dt_ms = dt_ms

    def compare_with_shared_input(
        self,
        systems: List[ContinuousTimeRNN],
        input_currents: np.ndarray,
    ) -> Dict:
        """Compare systems using the same input realization.

        Args:
            systems: List of systems to compare
            input_currents: Input currents, shape (n_timesteps, n_inputs)

        Returns:
            Dictionary with descriptors, distance matrix, and statistics
        """
        # Run simulations and collect outputs
        outputs_list = []
        for system in systems:
            sim = Simulator(system, self.dt_ms)
            outputs, _ = sim.run(input_currents)
            outputs_list.append(outputs)

        # Stack into batch
        outputs_batch = np.array(outputs_list)

        # Compute descriptors
        descriptors = self.descriptor_assembler.compute_descriptors_batch(
            outputs_batch, self.dt_ms
        )

        # Compute distances
        distance_matrix = compute_pairwise_distance_matrix(
            descriptors, metric="euclidean"
        )

        # Compute statistics
        dist_stats = compute_distance_statistics(distance_matrix)

        return {
            "descriptors": descriptors,
            "distance_matrix": distance_matrix,
            "distance_statistics": dist_stats,
        }

    def compare_with_varied_inputs(
        self,
        systems: List[ContinuousTimeRNN],
        input_generator,
        n_realizations: int = 5,
    ) -> Dict:
        """Compare systems using multiple input realizations.

        Averages descriptors across realizations for robustness.

        Args:
            systems: List of systems to compare
            input_generator: Callable that generates input currents
            n_realizations: Number of input realizations per system

        Returns:
            Dictionary with averaged descriptors and distance matrix
        """
        n_systems = len(systems)

        # Collect descriptors across realizations
        all_descriptors = []

        for _ in range(n_realizations):
            input_currents = input_generator()

            # Run simulations
            outputs_list = []
            for system in systems:
                sim = Simulator(system, self.dt_ms)
                outputs, _ = sim.run(input_currents)
                outputs_list.append(outputs)

            # Compute descriptors
            outputs_batch = np.array(outputs_list)
            descriptors = self.descriptor_assembler.compute_descriptors_batch(
                outputs_batch, self.dt_ms
            )
            all_descriptors.append(descriptors)

        # Average descriptors across realizations
        descriptors_mean = np.mean(all_descriptors, axis=0)

        # Compute distances on averaged descriptors
        distance_matrix = compute_pairwise_distance_matrix(
            descriptors_mean, metric="euclidean"
        )

        return {
            "descriptors": descriptors_mean,
            "descriptor_std": np.std(all_descriptors, axis=0),
            "distance_matrix": distance_matrix,
            "distance_statistics": compute_distance_statistics(distance_matrix),
        }

    def compare_groups(
        self,
        group_a: List[ContinuousTimeRNN],
        group_b: List[ContinuousTimeRNN],
        input_currents: np.ndarray,
    ) -> Dict:
        """Compare two groups of systems.

        Computes within-group and between-group distances.

        Args:
            group_a: First group of systems
            group_b: Second group of systems
            input_currents: Shared input currents

        Returns:
            Dictionary with group comparison statistics
        """
        all_systems = group_a + group_b
        n_a = len(group_a)
        n_b = len(group_b)

        # Compute descriptors
        result = self.compare_with_shared_input(all_systems, input_currents)
        distance_matrix = result["distance_matrix"]

        # Extract within-group and between-group distances
        within_a = distance_matrix[:n_a, :n_a]
        within_b = distance_matrix[n_a:, n_a:]
        between = distance_matrix[:n_a, n_a:]

        # Get upper triangles (excluding diagonals)
        def upper_tri_values(mat):
            mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
            return mat[mask]

        within_a_vals = upper_tri_values(within_a)
        within_b_vals = upper_tri_values(within_b)
        between_vals = between.flatten()

        return {
            "within_group_a_mean": float(np.mean(within_a_vals)),
            "within_group_b_mean": float(np.mean(within_b_vals)),
            "between_groups_mean": float(np.mean(between_vals)),
            "within_group_a_std": float(np.std(within_a_vals)),
            "within_group_b_std": float(np.std(within_b_vals)),
            "between_groups_std": float(np.std(between_vals)),
            "descriptors": result["descriptors"],
            "distance_matrix": distance_matrix,
        }
