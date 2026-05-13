"""Neurosignature: Functional system identification for neural operators."""

__version__ = "0.1.0"

# Public API - import key classes for convenient access
from neurosignature.systems import ContinuousTimeRNN, SystemGenerator
from neurosignature.simulation import Simulator
from neurosignature.inputs import PoissonGenerator, SynapticKernel, InputGenerator
from neurosignature.summaries import DescriptorAssembler, compute_descriptor
from neurosignature.experiments import SystemComparator
from neurosignature.metrics import (
    compute_pairwise_distance_matrix,
    compute_distance_statistics,
)
from neurosignature.math import relu, softplus, generate_random_matrix

__all__ = [
    "ContinuousTimeRNN",
    "SystemGenerator",
    "Simulator",
    "PoissonGenerator",
    "SynapticKernel",
    "InputGenerator",
    "DescriptorAssembler",
    "compute_descriptor",
    "SystemComparator",
    "compute_pairwise_distance_matrix",
    "compute_distance_statistics",
    "relu",
    "softplus",
    "generate_random_matrix",
]
