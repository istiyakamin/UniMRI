"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from unimri.data import (
    AcquisitionInfo,
    CoilInfo,
    EncodingSpace,
    FieldOfView,
    MRIData,
    SamplingPattern,
    Trajectory,
    TrajectoryUnits,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def cartesian_mri_data(rng: np.random.Generator) -> MRIData:
    """A small, valid Cartesian dataset: 4 coils, 16x16x8."""
    nc, nz, ny, nx = 4, 8, 16, 16
    kspace = rng.standard_normal((nc, nz, ny, nx)) + 1j * rng.standard_normal((nc, nz, ny, nx))
    kspace = kspace.astype(np.complex64)
    return MRIData(
        kspace=kspace,
        kspace_axes=("coil", "kz", "ky", "kx"),
        encoding=EncodingSpace(
            recon_matrix=(nx, ny, nz),
            encoded_matrix=(nx, ny, nz),
            fov=FieldOfView(240.0, 240.0, 120.0),
            sampling=SamplingPattern.CARTESIAN,
            n_dims=3,
        ),
        acquisition=AcquisitionInfo(nucleus="1H", field_strength_t=3.0),
        coils=CoilInfo(n_channels=nc),
    )


@pytest.fixture
def radial_mri_data(rng: np.random.Generator) -> MRIData:
    """A small, valid 2D radial dataset: 2 coils, 33 spokes x 64 readout."""
    nc, n_spokes, n_ro = 2, 33, 64
    kspace = (
        rng.standard_normal((nc, n_spokes, n_ro)) + 1j * rng.standard_normal((nc, n_spokes, n_ro))
    ).astype(np.complex64)

    angles = np.pi * np.arange(n_spokes) * (np.sqrt(5.0) - 1.0) / 2.0
    radius = np.linspace(-n_ro / 2, n_ro / 2, n_ro, endpoint=False)
    kx = np.cos(angles)[:, None] * radius[None, :]
    ky = np.sin(angles)[:, None] * radius[None, :]
    coords = np.stack([kx, ky], axis=0)  # (2, n_spokes, n_ro)

    return MRIData(
        kspace=kspace,
        kspace_axes=("coil", "shot", "readout"),
        encoding=EncodingSpace(
            recon_matrix=(n_ro, n_ro, 1),
            encoded_matrix=(n_ro, n_ro, 1),
            fov=FieldOfView(200.0, 200.0, 5.0),
            sampling=SamplingPattern.RADIAL,
            n_dims=2,
        ),
        acquisition=AcquisitionInfo(nucleus="23Na", field_strength_t=3.0),
        coils=CoilInfo(n_channels=nc),
        trajectory=Trajectory(coords=coords, units=TrajectoryUnits.NORMALIZED),
    )
