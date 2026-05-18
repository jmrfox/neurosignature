"""Summary statistics and descriptor computation."""

from .statistics import compute_channel_statistics, compute_global_statistics
from .spectral import compute_spectral_statistics
from .scalar_descriptors import (
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
)
from .descriptors import VectorDescriptor, BatchVectorDescriptor

__all__ = [
    "compute_channel_statistics",
    "compute_global_statistics",
    "compute_spectral_statistics",
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
]
