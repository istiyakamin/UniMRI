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


def _combine(coil_images: np.ndarray, mode: str) -> np.ndarray:
    """coil_images: (n_coils, *image_shape) -> (*image_shape), or unchanged for "none"."""
    if mode == "rss":
        return np.sqrt((np.abs(coil_images) ** 2).sum(axis=0))
    if mode == "sum":
        return coil_images.sum(axis=0)
    if mode == "none":
        return coil_images
    raise UniMRIError(f"unknown coil_combine mode {mode!r}")


def _ifftn_centered(kspace: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


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


@register_method("adjoint")
def adjoint(data: MRIData, *, coil_combine: str = "rss", eps: float = 1e-6) -> Array:
    """Single-pass reconstruction.

    Cartesian: centered inverse FFT over the k-space axes.
    Non-Cartesian: density-compensated gridding via the adjoint NUFFT
    (``Aᴴ (w · y)`` per coil).

    Coils are combined by ``coil_combine`` ("rss", "sum", or "none"). Returns an
    array of shape ``recon_matrix`` in ``(z, y, x)`` order for 3-D / ``(y, x)`` for
    2-D — with a leading coil axis when ``coil_combine="none"``.
    """
    nx, ny, nz = data.encoding.recon_matrix
    image_shape: tuple[int, ...] = (ny, nx) if data.encoding.n_dims == 2 else (nz, ny, nx)

    if data.is_cartesian:
        spatial = tuple(a for a in ("kz", "ky", "kx") if a in data.kspace_axes)
        ks = _as_coil_first(data, spatial)  # (coil, [kz], ky, kx)
        coil_images = _ifftn_centered(ks, axes=tuple(range(1, ks.ndim)))
    else:
        from unimri.operators.nufft import NUFFTOperator

        traj = data.trajectory
        assert traj is not None  # guaranteed by validate() for non-Cartesian data
        ks = _as_coil_first(data, ("shot", "readout"))  # (coil, shot, readout)
        op = NUFFTOperator(traj, image_shape, eps=eps)
        dcf = traj.density_compensation
        w = None if dcf is None else np.asarray(dcf, dtype=np.float64).ravel()

        coil_images = np.empty((ks.shape[0], *image_shape), dtype=np.complex128)
        for c in range(ks.shape[0]):
            y = np.ascontiguousarray(ks[c]).ravel().astype(np.complex128)
            coil_images[c] = op.adjoint(y * w if w is not None else y)

    result = _combine(coil_images, coil_combine)
    data.provenance.record(
        "reconstruct",
        params={"method": "adjoint", "coil_combine": coil_combine},
        backend="numpy",
    )
    return result
