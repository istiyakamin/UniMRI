"""Composable linear operators for reconstruction.

The algebraic core (:class:`LinearOperator` and friends) is implemented here.
Physical operators (``fourier``, ``nufft``, ``sampling``, ``coil``) are planned
-- see their module docstrings and ``docs/roadmap.md``.
"""

from __future__ import annotations

from unimri.operators.base import (
    AdjointOperator,
    CompositeOperator,
    IdentityOperator,
    LinearOperator,
    ScaledOperator,
)

__all__ = [
    "LinearOperator",
    "AdjointOperator",
    "IdentityOperator",
    "ScaledOperator",
    "CompositeOperator",
]
