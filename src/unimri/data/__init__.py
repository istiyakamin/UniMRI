"""The unified MRI data model.

Everything in UniMRI is expressed in terms of :class:`MRIData` and its
components. This subpackage has no dependencies on the I/O, operator, or
reconstruction layers -- it is the shared vocabulary they all import.
"""

from __future__ import annotations

from unimri.data.acquisition import (
    GYROMAGNETIC_RATIO_MHZ_PER_T,
    AcquisitionInfo,
    gamma_hz_per_t,
)
from unimri.data.encoding import EncodingSpace, FieldOfView, SamplingPattern
from unimri.data.metadata import Provenance, ProvenanceStep, ScannerMetadata
from unimri.data.mri_data import KNOWN_AXES, CoilInfo, MRIData
from unimri.data.trajectory import Trajectory, TrajectoryUnits

__all__ = [
    "MRIData",
    "KNOWN_AXES",
    "CoilInfo",
    "EncodingSpace",
    "FieldOfView",
    "SamplingPattern",
    "Trajectory",
    "TrajectoryUnits",
    "AcquisitionInfo",
    "gamma_hz_per_t",
    "GYROMAGNETIC_RATIO_MHZ_PER_T",
    "ScannerMetadata",
    "Provenance",
    "ProvenanceStep",
]
