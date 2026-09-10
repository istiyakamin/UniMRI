"""Brute-force non-uniform DFT -- a slow but exact reference.

This is *not* for production reconstruction. It exists so that fast NUFFT
backends (added later) can be checked against an implementation with no gridding,
kernel, or interpolation approximation, on small problem sizes.

Image pixels are assumed to sit on an integer grid centred at zero, matching the
``NORMALIZED`` trajectory convention (``k`` in samples of the encoded matrix).
The transform pair is

    forward:  s(k)  = sum_r  x(r) exp(-2j pi k . r / N)
    adjoint:  x(r) ~= sum_k  w(k) s(k) exp(+2j pi k . r / N)

with ``w`` the density compensation.
"""

from __future__ import annotations

import numpy as np


def _pixel_grid(shape: tuple[int, ...]) -> np.ndarray:
    axes = [np.arange(n) - n // 2 for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=0)  # (ndim, n_pixels)


def ndft_forward(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample ``image`` at k-space locations ``coords`` (shape ``(ndim, ...)``).

    Returns an array with the shape of ``coords[0]``.
    """
    ndim = coords.shape[0]
    if image.ndim != ndim:
        raise ValueError(f"coords is {ndim}-D but image is {image.ndim}-D")
    n = np.array(image.shape, dtype=float)
    k = coords.reshape(ndim, -1)  # (ndim, n_samples)
    r = _pixel_grid(image.shape)  # (ndim, n_pixels)
    phase = -2j * np.pi * ((k / n[:, None]).T @ r)  # (n_samples, n_pixels)
    samples = np.exp(phase) @ image.reshape(-1)
    return samples.reshape(coords.shape[1:])


def ndft_adjoint(
    samples: np.ndarray,
    coords: np.ndarray,
    shape: tuple[int, ...],
    dcf: np.ndarray | None = None,
) -> np.ndarray:
    """Density-compensated adjoint: k-space samples -> image on grid ``shape``."""
    ndim = coords.shape[0]
    k = coords.reshape(ndim, -1)
    s = samples.reshape(-1)
    if dcf is not None:
        s = s * dcf.reshape(-1)
    n = np.array(shape, dtype=float)
    r = _pixel_grid(shape)
    phase = 2j * np.pi * (r.T @ (k / n[:, None]))  # (n_pixels, n_samples)
    img = np.exp(phase) @ s
    return img.reshape(shape)
