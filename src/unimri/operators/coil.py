"""Coil sensitivity operator.

``SensitivityOperator`` implements the ``S`` in ``y = P F S x``:

    forward:  single image    ->  per-coil images   (S)   -- multiply by maps
    adjoint:  per-coil images ->  single image        (Sᴴ) -- conj-map weighted sum

Sensitivity maps come from :mod:`unimri.calibration` (or are supplied
externally). Composing ``FourierOperator(shape) @ SensitivityOperator(maps)``
(or :class:`~unimri.operators.nufft.NUFFTOperator` for non-Cartesian) builds the
full multi-coil encoding operator used by CG-SENSE; see
:func:`unimri.reconstruction.methods.cg`.
"""

from __future__ import annotations

import numpy as np

from unimri._typing import Array
from unimri.operators.base import LinearOperator

__all__ = ["SensitivityOperator"]


class SensitivityOperator(LinearOperator):
    """Multiply/combine by per-coil sensitivity maps.

    Parameters
    ----------
    sensitivity:
        Complex array of shape ``(n_coils, *image_shape)``.
    """

    name = "S"

    def __init__(self, sensitivity: Array) -> None:
        self._sens = np.asarray(sensitivity, dtype=np.complex128)
        self.in_shape = self._sens.shape[1:]
        self.out_shape = self._sens.shape

    def _forward(self, x: Array) -> Array:
        return self._sens * np.asarray(x)[None, ...]

    def _adjoint(self, y: Array) -> Array:
        return (np.conj(self._sens) * np.asarray(y)).sum(axis=0)

    def _normal(self, x: Array) -> Array:
        # Sᴴ S x = (sum_c |sens_c|^2) * x -- exact, no FFT/NUFFT involved.
        return (np.abs(self._sens) ** 2).sum(axis=0) * np.asarray(x)
