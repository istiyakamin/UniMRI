"""Analytic phantoms with known ground truth.

Used to build synthetic datasets whose exact image is known, so a
reconstruction's error can be measured rather than eyeballed.
"""

from __future__ import annotations

import numpy as np

# (intensity, a, b, x0, y0, angle_deg) -- standard Shepp-Logan (Toft variant).
_SHEPP_LOGAN_2D = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
]


def shepp_logan(shape: tuple[int, int]) -> np.ndarray:
    """A 2-D Shepp-Logan phantom, values in roughly ``[0, 1]``, shape ``(ny, nx)``."""
    ny, nx = shape
    ygrid, xgrid = np.mgrid[-1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float64)
    for intensity, a, b, x0, y0, angle in _SHEPP_LOGAN_2D:
        t = np.deg2rad(angle)
        xr = (xgrid - x0) * np.cos(t) + (ygrid - y0) * np.sin(t)
        yr = -(xgrid - x0) * np.sin(t) + (ygrid - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += intensity
    return img


def ellipsoid_phantom(shape: tuple[int, int, int]) -> np.ndarray:
    """A simple 3-D phantom: a few nested/offset ellipsoids. Shape ``(nz, ny, nx)``."""
    nz, ny, nx = shape
    zg, yg, xg = np.mgrid[-1 : 1 : nz * 1j, -1 : 1 : ny * 1j, -1 : 1 : nx * 1j]
    img = np.zeros(shape, dtype=np.float64)
    specs = [
        (1.0, (0.75, 0.9, 0.9), (0.0, 0.0, 0.0)),
        (-0.6, (0.6, 0.75, 0.75), (0.0, 0.0, -0.02)),
        (0.3, (0.2, 0.25, 0.25), (0.0, 0.0, 0.3)),
        (0.3, (0.12, 0.12, 0.12), (0.22, 0.0, 0.0)),
        (0.3, (0.12, 0.12, 0.12), (-0.22, 0.0, 0.0)),
    ]
    for intensity, (az, ay, ax), (z0, y0, x0) in specs:
        mask = ((zg - z0) / az) ** 2 + ((yg - y0) / ay) ** 2 + ((xg - x0) / ax) ** 2 <= 1.0
        img[mask] += intensity
    return img
