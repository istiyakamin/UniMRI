"""Non-uniform Fourier operator, backed by FINUFFT.

``NUFFTOperator`` maps between a Cartesian image grid and samples on an arbitrary
:class:`~unimri.data.Trajectory`:

    forward:  image   ->  non-Cartesian k-space   (FINUFFT type 2)
    adjoint:  k-space  ->  image                    (FINUFFT type 1)

UniMRI does not ship its own gridder. This is a thin adapter over
`finufft <https://finufft.readthedocs.io>`_ (install with the ``nufft`` extra:
``pip install "unimri[nufft]"``). The interface is intentionally close to
``mri-nufft`` so a backend swap later is mechanical.

Density compensation is **not** applied here -- it is a separate diagonal
weighting handled by the reconstruction, so ``forward`` and ``adjoint`` stay
true adjoints (verified by ``dot_test``).

Trajectory units: coordinates are converted from
:class:`~unimri.data.TrajectoryUnits.NORMALIZED` (Nyquist window ``[-N/2, N/2)``)
to the radians-per-sample that FINUFFT expects, i.e. multiplied by ``2*pi / N``
per axis. Other unit enums are converted to normalized first.

Axis convention: ``trajectory.coords`` row ``d`` corresponds to image axis ``d``
(NumPy order). For a 3-D image ``(nz, ny, nx)`` that is ``coords = (kz, ky, kx)``;
for 2-D ``(ny, nx)`` it is ``(ky, kx)``. See ``docs/data-model.md``.
"""

from __future__ import annotations

import numpy as np

from unimri._typing import Array
from unimri.data.trajectory import Trajectory, TrajectoryUnits
from unimri.exceptions import BackendError
from unimri.operators.base import LinearOperator

try:  # optional dependency
    import finufft
except ImportError:  # pragma: no cover - exercised via the nufft extra
    finufft = None

__all__ = ["NUFFTOperator"]

_EPS = 1e-6


def _to_normalized(traj: Trajectory, matrix: tuple[int, ...]) -> np.ndarray:
    """Return trajectory coordinates in NORMALIZED units, shape ``(ndim, n_samples)``."""
    k = np.asarray(traj.coords, dtype=np.float64).reshape(traj.n_dims, -1)
    n = np.asarray(matrix[: traj.n_dims], dtype=np.float64)[:, None]
    u = traj.units
    if u is TrajectoryUnits.NORMALIZED:
        return k
    if u is TrajectoryUnits.CYCLES_PER_FOV:
        return k * n
    if u is TrajectoryUnits.RADIANS_PER_VOXEL:
        return k * n / (2.0 * np.pi)
    raise BackendError(f"NUFFTOperator cannot convert trajectory units {u.value!r}")


class NUFFTOperator(LinearOperator):
    """FINUFFT-backed non-uniform FFT for a fixed trajectory and image shape.

    Parameters
    ----------
    trajectory:
        The k-space sample locations. 2-D or 3-D.
    image_shape:
        Target image grid, in NumPy axis order -- ``(ny, nx)`` for 2-D,
        ``(nz, ny, nx)`` for 3-D. Trajectory row ``d`` pairs with image axis
        ``d`` (so ``(kz, ky, kx)`` for a 3-D image).
    eps:
        FINUFFT relative tolerance.
    """

    name = "NUFFT"

    def __init__(
        self,
        trajectory: Trajectory,
        image_shape: tuple[int, ...],
        *,
        eps: float = _EPS,
    ) -> None:
        if finufft is None:
            raise BackendError(
                'NUFFTOperator needs FINUFFT. Install it with: pip install "unimri[nufft]"'
            )
        ndim = trajectory.n_dims
        if ndim not in (2, 3):
            raise ValueError(f"NUFFTOperator supports 2-D and 3-D, not {ndim}-D")
        if len(image_shape) != ndim:
            raise ValueError(f"image_shape {image_shape} does not match {ndim}-D trajectory")

        self._ndim = ndim
        self._image_shape = tuple(int(s) for s in image_shape)
        self._eps = float(eps)
        self._n_samples = trajectory.n_samples

        # NORMALIZED coords -> radians per sample. Row d pairs with image axis d,
        # which is exactly the coordinate order FINUFFT's nufftNd* expect.
        kn = _to_normalized(trajectory, self._image_shape)
        scale = 2.0 * np.pi / np.asarray(self._image_shape, dtype=np.float64)[:, None]
        self._coords = [np.ascontiguousarray(row) for row in kn * scale]

        self.in_shape = self._image_shape
        self.out_shape = (self._n_samples,)

    # -- LinearOperator -------------------------------------------------

    def _forward(self, x: Array) -> Array:
        img = np.ascontiguousarray(np.asarray(x, dtype=np.complex128))
        fn = finufft.nufft2d2 if self._ndim == 2 else finufft.nufft3d2
        return fn(*self._coords, img, isign=-1, eps=self._eps)

    def _adjoint(self, y: Array) -> Array:
        c = np.ascontiguousarray(np.asarray(y, dtype=np.complex128).ravel())
        if self._ndim == 2:
            return finufft.nufft2d1(*self._coords, c, self._image_shape, isign=1, eps=self._eps)
        return finufft.nufft3d1(*self._coords, c, self._image_shape, isign=1, eps=self._eps)
