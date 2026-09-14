"""UniMRI: a unified Python framework for MRI raw-data processing and reconstruction.

This package is in **pre-alpha**. The data model, operator algebra, and
``reconstruct(method="adjoint"|"cg")`` work today, validated on real 3-D
radial data; vendor readers (``unimri.read``) are still stubs. See
``docs/roadmap.md``.
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
__version__ = "0.0.1a2"

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


def reconstruct(data: MRIData, method: str = "adjoint", **kwargs: object):
    """Reconstruct an image from :class:`MRIData`.

    The top-level entry point. Currently implemented:

    - ``"adjoint"`` -- centered inverse FFT (Cartesian) or density-compensated
      gridding via the adjoint NUFFT (non-Cartesian).
    - ``"cg"`` -- CG-SENSE: iterative reconstruction with coil sensitivities,
      the same solver for Cartesian and non-Cartesian data.

    Both non-Cartesian paths need the ``nufft`` extra
    (``pip install "unimri[nufft]"``). Planned: ``"sense"``, ``"grappa"``,
    ``"cs"`` -- see ``docs/roadmap.md``.

    Parameters
    ----------
    data:
        The raw data to reconstruct.
    method:
        Reconstruction method name.
    **kwargs:
        Method-specific options (e.g. ``coil_combine="rss"`` for ``"adjoint"``;
        ``n_iter``, ``l2``, ``sensitivity`` for ``"cg"``).

    Returns
    -------
    numpy.ndarray
        The reconstructed image (``(z, y, x)`` for 3-D, ``(y, x)`` for 2-D).
    """
    from unimri.reconstruction import run

    return run(data, method, **kwargs)
