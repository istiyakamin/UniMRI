"""UniMRI: a unified Python framework for MRI raw-data processing and reconstruction.

This package is in **pre-alpha**. The public interfaces are defined and tested,
but no reconstruction backends are implemented yet. See ``docs/roadmap.md``.
"""

from __future__ import annotations

from unimri.data import (
    AcquisitionInfo,
    CoilInfo,
    EncodingSpace,
    MRIData,
    Provenance,
    SamplingPattern,
    ScannerMetadata,
    Trajectory,
)
from unimri.exceptions import (
    OperatorShapeError,
    ReaderError,
    UniMRIError,
    UnsupportedFormatError,
    ValidationError,
)
from unimri.io import read
from unimri.operators import (
    CompositeOperator,
    IdentityOperator,
    LinearOperator,
    ScaledOperator,
)

# Single source of truth for the version. Hatchling reads this line
# (`[tool.hatch.version]`), and CITATION.cff is kept in sync by a test.
__version__ = "0.0.0"

__all__ = [
    "__version__",
    # data model
    "MRIData",
    "Trajectory",
    "EncodingSpace",
    "SamplingPattern",
    "CoilInfo",
    "AcquisitionInfo",
    "ScannerMetadata",
    "Provenance",
    # io
    "read",
    # operators
    "LinearOperator",
    "IdentityOperator",
    "ScaledOperator",
    "CompositeOperator",
    # exceptions
    "UniMRIError",
    "ReaderError",
    "UnsupportedFormatError",
    "OperatorShapeError",
    "ValidationError",
    # top-level entry points
    "reconstruct",
]


def reconstruct(data: MRIData, method: str = "fft", **kwargs: object) -> object:
    """Reconstruct an image from :class:`MRIData`.

    This is the intended top-level entry point. No reconstruction methods are
    implemented yet; this raises :class:`NotImplementedError` until the
    ``unimri.reconstruction`` backends land (see ``docs/roadmap.md``).

    Parameters
    ----------
    data:
        The raw data to reconstruct.
    method:
        Reconstruction method name, e.g. ``"fft"``, ``"sense"``, ``"cg"``.
    **kwargs:
        Method-specific options.
    """
    raise NotImplementedError(
        f"reconstruct(method={method!r}) is not implemented yet. "
        "UniMRI is pre-alpha; see https://github.com/istiyakamin/UniMRI/blob/main/docs/roadmap.md"
    )
