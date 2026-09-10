"""Synthetic data and reference transforms for validating UniMRI.

Nothing here is for production reconstruction. It provides:

- :mod:`~unimri.testing.phantoms` -- analytic images with known ground truth
- :mod:`~unimri.testing.trajectories` -- k-space trajectory generators
- :mod:`~unimri.testing.ndft` -- a slow, exact non-uniform DFT reference
- :func:`synthetic_dataset` -- one valid :class:`~unimri.data.MRIData` per
  :class:`~unimri.data.SamplingPattern`, with the exact image it came from

Import cost: only NumPy.
"""

from __future__ import annotations

from unimri.testing.datasets import (
    AVAILABLE_PATTERNS,
    SyntheticDataset,
    synthetic_dataset,
)
from unimri.testing.ndft import ndft_adjoint, ndft_forward

__all__ = [
    "synthetic_dataset",
    "SyntheticDataset",
    "AVAILABLE_PATTERNS",
    "ndft_forward",
    "ndft_adjoint",
]
