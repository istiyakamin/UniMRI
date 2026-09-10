"""Analytic k-space trajectory generators for synthetic datasets.

All generators return ``coords`` in :class:`~unimri.data.TrajectoryUnits.NORMALIZED`
units (one unit = one sample of the encoded matrix, Nyquist window
``[-N/2, N/2)``) and, where a closed form exists, an analytic density
compensation function.
"""

from __future__ import annotations

import numpy as np

_GOLDEN_ANGLE_2D = np.pi * (3.0 - np.sqrt(5.0))  # ~111.25 deg
# 3-D: two golden ratios for a spiral-phyllotaxis distribution on the sphere.
_PHI1 = 0.4656  # 1 / golden ratio ^? -- Saff & Kuijlaars style increments
_PHI2 = 0.6823


def radial_2d(
    matrix: int, n_spokes: int, *, golden_angle: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """2-D radial spokes. Returns ``(coords (2, n_spokes, matrix), dcf (n_spokes, matrix))``."""
    kr = np.linspace(-matrix / 2, matrix / 2, matrix, endpoint=False)
    if golden_angle:
        angles = np.arange(n_spokes) * _GOLDEN_ANGLE_2D
    else:
        angles = np.linspace(0.0, np.pi, n_spokes, endpoint=False)
    kx = np.cos(angles)[:, None] * kr[None, :]
    ky = np.sin(angles)[:, None] * kr[None, :]
    coords = np.stack([kx, ky], axis=0)
    # ramp filter (|k|), normalised; the DC sample keeps a small finite weight.
    dcf = np.abs(kr)[None, :].repeat(n_spokes, axis=0)
    dcf = np.maximum(dcf, 0.5 / matrix)
    dcf = dcf / dcf.sum() * (n_spokes * matrix)
    return coords, dcf


def radial_3d(
    matrix: int, n_spokes: int, readout: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """3-D center-out projections on a spiral-phyllotaxis sphere distribution.

    Returns ``(coords (3, n_spokes, readout), dcf (n_spokes, readout))``.
    """
    readout = readout or matrix // 2
    kr = np.linspace(0.0, matrix / 2, readout, endpoint=False)
    i = np.arange(n_spokes)
    z = 1.0 - (2.0 * i + 1.0) / n_spokes
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = 2.0 * np.pi * ((i * _PHI1) % 1.0)
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=0
    )  # (3, n_spokes)
    coords = dirs[:, :, None] * kr[None, None, :]  # (3, n_spokes, readout)
    # center-out radial: dcf ~ k^2 (surface area of the shell).
    dcf = (kr**2)[None, :].repeat(n_spokes, axis=0)
    dcf = np.maximum(dcf, (0.5 / matrix) ** 2)
    dcf = dcf / dcf.sum() * (n_spokes * readout)
    return coords, dcf


def stack_of_stars(matrix: int, n_spokes: int, n_partitions: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial in-plane, Cartesian along z.

    Returns ``(coords (3, n_spokes*n_partitions, matrix), dcf (n_spokes*n_partitions, matrix))``.
    """
    coords2d, dcf2d = radial_2d(matrix, n_spokes, golden_angle=True)
    kz = np.linspace(-n_partitions / 2, n_partitions / 2, n_partitions, endpoint=False)
    shots = []
    for z in kz:
        c = coords2d.copy()  # (2, n_spokes, matrix)
        c3 = np.concatenate([c, np.full((1, n_spokes, matrix), z)], axis=0)
        shots.append(c3)
    coords = np.concatenate(shots, axis=1)  # (3, n_spokes*nz, matrix)
    dcf = np.tile(dcf2d, (n_partitions, 1))
    return coords, dcf


def cartesian(matrix: int, n_dims: int = 2) -> np.ndarray:
    """A fully-sampled Cartesian grid expressed as a trajectory (for testing NUFFT==FFT)."""
    ax = np.arange(matrix) - matrix // 2
    grids = np.meshgrid(*([ax] * n_dims), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=0)  # (n_dims, matrix**n_dims)
