"""Cartesian Fourier operator.

``FourierOperator`` implements a centered n-D DFT/adjoint-DFT as a
:class:`~unimri.operators.base.LinearOperator`:

    forward:  image  ->  Cartesian k-space   (F)
    adjoint:  k-space ->  image               (Fᴴ, the exact conjugate transpose)

Convention: **unnormalized**, matching :mod:`unimri.testing.ndft` and
:class:`~unimri.operators.nufft.NUFFTOperator` exactly (`forward` reproduces
``ndft_forward`` bit-for-bit on a Cartesian grid) -- so ``FourierOperator`` and
``NUFFTOperator`` are drop-in equivalents for `reconstruct(method=...)`, which
picks one or the other purely based on ``data.is_cartesian``, and amplitudes
stay comparable across both. ``Fᴴ F = N·I`` where ``N`` is the number of
transformed samples (not unitary -- the same non-orthonormal convention as a
plain adjoint NUFFT).

Backed by the array namespace of the input (NumPy today; CuPy/PyTorch once they
implement the array API's ``fft`` extension), so it runs wherever the data
lives. ``axes`` are given as negative indices (counted from the end), so the
same operator transparently handles an extra leading batch axis -- e.g. coils,
when composed as ``FourierOperator(shape) @ SensitivityOperator(maps)``.
"""

from __future__ import annotations

import math

import numpy as np

from unimri._typing import Array
from unimri.operators.base import LinearOperator

__all__ = ["FourierOperator"]


class FourierOperator(LinearOperator):
    """Centered, unnormalized DFT over ``image_shape``.

    Parameters
    ----------
    image_shape:
        The (single-instance) image shape, e.g. ``(ny, nx)`` or ``(nz, ny, nx)``.
    axes:
        Which axes to transform, as negative indices. Defaults to the last
        ``len(image_shape)`` axes, so a leading batch axis (coils, ...) is left
        untouched.
    """

    name = "F"

    def __init__(
        self, image_shape: tuple[int, ...], *, axes: tuple[int, ...] | None = None
    ) -> None:
        self.in_shape = tuple(int(s) for s in image_shape)
        self.out_shape = self.in_shape
        ndim = len(self.in_shape)
        self._axes = axes if axes is not None else tuple(range(-ndim, 0))
        self._n_transformed = math.prod(self.in_shape[a] for a in self._axes)

    def _forward(self, x: Array) -> Array:
        return np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(x, axes=self._axes), axes=self._axes, norm=None),
            axes=self._axes,
        )

    def _adjoint(self, y: Array) -> Array:
        return np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(y, axes=self._axes), axes=self._axes, norm="forward"),
            axes=self._axes,
        )

    def _normal(self, x: Array) -> Array:
        return self._n_transformed * np.asarray(x)  # exact: Fᴴ F = N·I
