"""Neurosignature: Functional system identification for neural operators."""

__version__ = "0.1.0"

# Public API
from neurosignature.systems import ContinuousTimeRNN
from neurosignature.simulation import Simulator, BatchSimulator
from neurosignature.inputs import PoissonGenerator, InputGenerator
from neurosignature.summaries import (
    ScalarDescriptor,
    BatchScalarDescriptor,
    ReferenceMean,
    ReferenceStd,
    ReferenceSpectralCentroid,
    ResidualMean,
    ResidualStd,
    ResidualEnergy,
    ResidualSpectralCentroidMean,
    ResidualSpectralCentroidStd,
    ResidualParticipationRatio,
    ResidualMaxEigenvalue,
    ResidualEigenvalueEntropy,
    CrossCorrelationMean,
    TransmissionEfficiency,
    VectorDescriptor,
    BatchVectorDescriptor,
)
from neurosignature.experiments import Experiment, SystemComparator
from neurosignature.math import relu, softplus, generate_random_matrix

__all__ = [
    "ContinuousTimeRNN",
    "Simulator",
    "BatchSimulator",
    "PoissonGenerator",
    "InputGenerator",
    "ScalarDescriptor",
    "BatchScalarDescriptor",
    "ReferenceMean",
    "ReferenceStd",
    "ReferenceSpectralCentroid",
    "ResidualMean",
    "ResidualStd",
    "ResidualEnergy",
    "ResidualSpectralCentroidMean",
    "ResidualSpectralCentroidStd",
    "ResidualParticipationRatio",
    "ResidualMaxEigenvalue",
    "ResidualEigenvalueEntropy",
    "CrossCorrelationMean",
    "TransmissionEfficiency",
    "VectorDescriptor",
    "BatchVectorDescriptor",
    "Experiment",
    "SystemComparator",
    "relu",
    "softplus",
    "generate_random_matrix",
]
