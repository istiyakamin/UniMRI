"""Composable linear operators for reconstruction.

The algebraic core (:class:`LinearOperator` and friends) plus the physical
operators: :class:`FourierOperator` (Cartesian), :class:`NUFFTOperator`
(non-Cartesian, needs the ``nufft`` extra), and :class:`SensitivityOperator`
(coils). Planned: ``sampling``. See ``docs/roadmap.md``.
"""

from __future__ import annotations

from unimri.operators.base import (
    AdjointOperator,
    CompositeOperator,
    IdentityOperator,
    LinearOperator,
    ScaledOperator,
    UncheckedOperator,
    unchecked,
)
from unimri.operators.coil import SensitivityOperator
from unimri.operators.fourier import FourierOperator
from unimri.operators.nufft import NUFFTOperator

__all__ = [
    "LinearOperator",
    "AdjointOperator",
    "IdentityOperator",
    "ScaledOperator",
    "CompositeOperator",
    "UncheckedOperator",
    "unchecked",
    "FourierOperator",
    "NUFFTOperator",
    "SensitivityOperator",
]
