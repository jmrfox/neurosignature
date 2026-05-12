"""Parameter sweep experiments for descriptor characterization."""

import numpy as np
from typing import List, Dict, Callable, Optional
from neurosignature.systems.recurrent_system import ContinuousTimeRNN
from neurosignature.systems.system_generator import SystemGenerator
from neurosignature.simulation.simulator import Simulator
from neurosignature.summaries.descriptors import DescriptorAssembler


class ParameterSweeper:
    """Sweep parameter values and characterize descriptor trajectories.

    Args:
        system_generator: SystemGenerator for creating systems
        descriptor_assembler: DescriptorAssembler for computing descriptors
        simulator: Simulator for running systems
    """

    def __init__(
        self,
        system_generator: SystemGenerator,
        descriptor_assembler: DescriptorAssembler,
        simulator: Simulator,
    ):
        self.system_generator = system_generator
        self.descriptor_assembler = descriptor_assembler
        self.simulator = simulator

    def sweep_spectral_radius(
        self,
        radii: List[float],
        input_currents: np.ndarray,
    ) -> Dict:
        """Sweep spectral radius and record descriptor changes.

        Args:
            radii: List of spectral radius values to test
            input_currents: Input currents for simulations

        Returns:
            Dictionary with radii, descriptors, and outputs
        """
        systems = self.system_generator.generate_spectral_radius_sweep(radii)

        outputs_list = []
        descriptors_list = []

        for system in systems:
            outputs, _ = self.simulator.run(input_currents)
            outputs_list.append(outputs)

            descriptor = self.descriptor_assembler.compute_descriptor(
                outputs, self.simulator.dt_ms
            )
            descriptors_list.append(descriptor)

        return {
            "parameter_name": "spectral_radius",
            "parameter_values": radii,
            "systems": systems,
            "outputs": outputs_list,
            "descriptors": np.array(descriptors_list),
        }

    def sweep_sparsity(
        self,
        sparsities: List[float],
        input_currents: np.ndarray,
    ) -> Dict:
        """Sweep sparsity and record descriptor changes.

        Args:
            sparsities: List of sparsity values to test
            input_currents: Input currents for simulations

        Returns:
            Dictionary with sparsities, descriptors, and outputs
        """
        systems = self.system_generator.generate_sparsity_sweep(sparsities)

        outputs_list = []
        descriptors_list = []

        for system in systems:
            outputs, _ = self.simulator.run(input_currents)
            outputs_list.append(outputs)

            descriptor = self.descriptor_assembler.compute_descriptor(
                outputs, self.simulator.dt_ms
            )
            descriptors_list.append(descriptor)

        return {
            "parameter_name": "sparsity",
            "parameter_values": sparsities,
            "systems": systems,
            "outputs": outputs_list,
            "descriptors": np.array(descriptors_list),
        }

    def sweep_timescales(
        self,
        tau_ranges: List[tuple],
        input_currents: np.ndarray,
    ) -> Dict:
        """Sweep timescale ranges and record descriptor changes.

        Args:
            tau_ranges: List of (min_tau, max_tau) tuples
            input_currents: Input currents for simulations

        Returns:
            Dictionary with tau ranges, descriptors, and outputs
        """
        systems = self.system_generator.generate_timescale_sweep(tau_ranges)

        outputs_list = []
        descriptors_list = []

        for system in systems:
            outputs, _ = self.simulator.run(input_currents)
            outputs_list.append(outputs)

            descriptor = self.descriptor_assembler.compute_descriptor(
                outputs, self.simulator.dt_ms
            )
            descriptors_list.append(descriptor)

        return {
            "parameter_name": "timescale_range",
            "parameter_values": tau_ranges,
            "systems": systems,
            "outputs": outputs_list,
            "descriptors": np.array(descriptors_list),
        }

    def compute_descriptor_derivatives(
        self,
        sweep_result: Dict,
    ) -> np.ndarray:
        """Compute approximate derivatives of descriptors w.r.t. parameter.

        Uses finite differences between adjacent parameter values.

        Args:
            sweep_result: Output from a sweep method

        Returns:
            Derivative matrix, shape (n_params-1, descriptor_dim)
        """
        descriptors = sweep_result["descriptors"]
        params = np.array(sweep_result["parameter_values"])

        # Compute parameter differences
        d_params = np.diff(params)

        # Compute descriptor differences
        d_descriptors = np.diff(descriptors, axis=0)

        # Divide for derivatives
        derivatives = d_descriptors / d_params[:, np.newaxis]

        return derivatives
