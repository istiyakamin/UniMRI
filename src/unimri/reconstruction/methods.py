"""Reconstruction methods, dispatched by name from :func:`unimri.reconstruct`.

Register a method with :func:`register_method`; call it via
``unimri.reconstruct(data, method="...")``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from unimri._typing import Array
from unimri.data import MRIData
from unimri.exceptions import UniMRIError

_METHODS: dict[str, Callable[..., Array]] = {}

Method = Callable[..., Array]


def register_method(name: str) -> Callable[[Method], Method]:
    def deco(fn: Method) -> Method:
        _METHODS[name] = fn
        return fn

    return deco


def available_methods() -> list[str]:
    return sorted(_METHODS)


def run(data: MRIData, method: str, **kwargs: object) -> Array:
    if method not in _METHODS:
        raise UniMRIError(
            f"unknown reconstruction method {method!r}; available: {available_methods()}"
        )
    data.validate()
    return _METHODS[method](data, **kwargs)


# ---------------------------------------------------------------------------


def image_shape_of(data: MRIData) -> tuple[int, ...]:
    """The reconstructed-image array shape: ``(ny, nx)`` or ``(nz, ny, nx)``."""
    nx, ny, nz = data.encoding.recon_matrix
    return (ny, nx) if data.encoding.n_dims == 2 else (nz, ny, nx)


def _combine(coil_images: np.ndarray, mode: str) -> np.ndarray:
    """coil_images: (n_coils, *image_shape) -> (*image_shape), or unchanged for "none"."""
    if mode == "rss":
        return np.sqrt((np.abs(coil_images) ** 2).sum(axis=0))
    if mode == "sum":
        return coil_images.sum(axis=0)
    if mode == "none":
        return coil_images
    raise UniMRIError(f"unknown coil_combine mode {mode!r}")


def _as_coil_first(data: MRIData, wanted: tuple[str, ...]) -> np.ndarray:
    """Transpose kspace to ("coil", *wanted); require every other axis to be singleton."""
    order = [data.axis("coil")] + [data.axis(a) for a in wanted]
    rest = [i for i in range(data.kspace.ndim) if i not in order]
    arr = np.transpose(np.asarray(data.kspace), order + rest)
    n_named = 1 + len(wanted)
    if any(arr.shape[n_named + i] != 1 for i in range(len(rest))):
        extra = [data.kspace_axes[i] for i in rest if data.kspace.shape[i] != 1]
        raise UniMRIError(f"reconstruction does not handle non-singleton axes yet: {extra}")
    return arr.reshape(arr.shape[:n_named])


def coil_images(data: MRIData, *, eps: float = 1e-6) -> np.ndarray:
    """Per-coil, single-pass images: ``Aᴴ(w·y)`` per coil (no combination).

    Cartesian: centered inverse FFT. Non-Cartesian: density-compensated
    gridding via the adjoint NUFFT. Building block for :func:`adjoint` and for
    :func:`unimri.calibration.estimate_sensitivity`.
    """
    image_shape = image_shape_of(data)

    if data.is_cartesian:
        from unimri.operators import FourierOperator

        spatial = tuple(a for a in ("kz", "ky", "kx") if a in data.kspace_axes)
        ks = _as_coil_first(data, spatial)  # (coil, [kz], ky, kx)
        return FourierOperator(image_shape)._adjoint(ks)

    from unimri.operators import NUFFTOperator

    traj = data.trajectory
    assert traj is not None  # guaranteed by validate() for non-Cartesian data
    ks = _as_coil_first(data, ("shot", "readout"))  # (coil, shot, readout)
    op = NUFFTOperator(traj, image_shape, eps=eps)
    dcf = traj.density_compensation
    w = None if dcf is None else np.asarray(dcf, dtype=np.float64).ravel()

    y = ks.reshape(ks.shape[0], -1).astype(np.complex128)
    if w is not None:
        y = y * w
    return op._adjoint(y)


@register_method("adjoint")
def adjoint(data: MRIData, *, coil_combine: str = "rss", eps: float = 1e-6) -> Array:
    """Single-pass reconstruction: gridding (non-Cartesian) or plain iFFT (Cartesian).

    Coils are combined by ``coil_combine`` ("rss", "sum", or "none"). Returns an
    array of shape ``recon_matrix`` in ``(z, y, x)`` order for 3-D / ``(y, x)`` for
    2-D — with a leading coil axis when ``coil_combine="none"``.
    """
    result = _combine(coil_images(data, eps=eps), coil_combine)
    data.provenance.record(
        "reconstruct", params={"method": "adjoint", "coil_combine": coil_combine}, backend="numpy"
    )
    return result


@register_method("cg")
def cg(
    data: MRIData,
    *,
    n_iter: int = 10,
    l2: float = 1e-4,
    sensitivity: np.ndarray | None = None,
    sensitivity_method: str = "rss",
    eps: float = 1e-6,
) -> Array:
    """CG-SENSE: iterative reconstruction with coil sensitivities, y = A x = F S x.

    Solves ``(Aᴴ A + l2·I) x = Aᴴ y`` by conjugate gradient (Pruessmann et al.,
    MRM 2001), where ``A`` is :class:`~unimri.operators.FourierOperator` for
    Cartesian data or :class:`~unimri.operators.NUFFTOperator` for non-Cartesian
    data, composed with :class:`~unimri.operators.SensitivityOperator`. The same
    solver code handles both, because only ``A`` changes with the trajectory.

    ``sensitivity`` (``(n_coils, *image_shape)``) is estimated automatically via
    :func:`unimri.calibration.estimate_sensitivity` if not supplied.
    """
    from unimri.operators import FourierOperator, NUFFTOperator, SensitivityOperator, unchecked
    from unimri.optimization import conjugate_gradient

    image_shape = image_shape_of(data)

    if sensitivity is None:
        from unimri.calibration import estimate_sensitivity

        sensitivity = estimate_sensitivity(data, method=sensitivity_method, eps=eps)

    S = SensitivityOperator(sensitivity)
    F: FourierOperator | NUFFTOperator
    if data.is_cartesian:
        spatial = tuple(a for a in ("kz", "ky", "kx") if a in data.kspace_axes)
        F = FourierOperator(image_shape)
        y = _as_coil_first(data, spatial)
    else:
        traj = data.trajectory
        assert traj is not None
        F = NUFFTOperator(traj, image_shape, eps=eps)
        y = _as_coil_first(data, ("shot", "readout")).reshape(data.n_coils, -1)

    encoding = unchecked(F @ S)
    img = conjugate_gradient(encoding, y.astype(np.complex128), n_iter=n_iter, l2=l2)

    data.provenance.record(
        "reconstruct", params={"method": "cg", "n_iter": n_iter, "l2": l2}, backend="numpy"
    )
    return img
