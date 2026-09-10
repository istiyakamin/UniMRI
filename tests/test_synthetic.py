"""Tests for the synthetic-data / reference-transform toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from unimri.data import SamplingPattern
from unimri.testing import (
    AVAILABLE_PATTERNS,
    ndft_adjoint,
    ndft_forward,
    synthetic_dataset,
)
from unimri.testing import trajectories as T
from unimri.testing.phantoms import ellipsoid_phantom, shepp_logan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a).ravel() - np.abs(a).mean()
    b = np.abs(b).ravel() - np.abs(b).mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_phantoms_have_expected_shape_and_range() -> None:
    p = shepp_logan((32, 32))
    assert p.shape == (32, 32)
    assert p.min() >= -0.1 and p.max() <= 1.1
    assert ellipsoid_phantom((8, 16, 16)).shape == (8, 16, 16)


@pytest.mark.parametrize("pattern", AVAILABLE_PATTERNS, ids=lambda p: p.value)
def test_synthetic_dataset_is_valid(pattern: SamplingPattern) -> None:
    ds = synthetic_dataset(pattern, matrix=20)
    ds.data.validate()  # raises on any inconsistency
    assert ds.data.encoding.sampling is pattern
    assert ds.ground_truth.ndim in (2, 3)
    assert ds.data.n_coils == 1
    assert len(ds.data.provenance) >= 1


@pytest.mark.parametrize("pattern", AVAILABLE_PATTERNS, ids=lambda p: p.value)
def test_synthetic_dataset_multicoil_is_valid(pattern: SamplingPattern) -> None:
    ds = synthetic_dataset(pattern, matrix=16, n_coils=4)
    ds.data.validate()
    assert ds.data.n_coils == 4
    assert ds.coil_maps is not None and ds.coil_maps.shape[0] == 4


def test_cartesian_ndft_matches_fft() -> None:
    img = shepp_logan((24, 24)).astype(np.complex128)
    coords = T.cartesian(24, n_dims=2)
    got = ndft_forward(img, coords).reshape(24, 24)
    ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    assert np.allclose(got, ref, atol=1e-8)


def test_ndft_adjoint_passes_the_dot_test() -> None:
    """The exact operator-correctness guarantee: <Ax, y> == <x, Aᴴy>."""
    rng = np.random.default_rng(0)
    coords, _ = T.radial_2d(20, 30)
    x = rng.standard_normal((20, 20)) + 1j * rng.standard_normal((20, 20))
    y = rng.standard_normal(coords.shape[1:]) + 1j * rng.standard_normal(coords.shape[1:])
    lhs = np.vdot(ndft_forward(x, coords), y)
    rhs = np.vdot(x, ndft_adjoint(y, coords, (20, 20), dcf=None))
    assert np.isclose(lhs, rhs, rtol=1e-6)


def test_radial_adjoint_recovers_phantom() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=24, n_coils=1)
    traj = ds.data.trajectory
    recon = ndft_adjoint(ds.data.kspace[0], traj.coords, (24, 24), dcf=traj.density_compensation)
    # crude gridding of a hard phantom: recognisable, not publication-quality
    assert _corr(recon, ds.ground_truth) > 0.6
    # the recon must actually depend on the trajectory
    rng = np.random.default_rng(2)
    wrong = rng.uniform(-12, 12, traj.coords.shape)
    assert _corr(ndft_adjoint(ds.data.kspace[0], wrong, (24, 24)), ds.ground_truth) < 0.3


def test_stack_of_stars_adjoint_recovers_phantom() -> None:
    ds = synthetic_dataset(SamplingPattern.STACK_OF_STARS, matrix=16, n_coils=1)
    traj = ds.data.trajectory
    nz = ds.ground_truth.shape[0]
    recon = ndft_adjoint(
        ds.data.kspace[0], traj.coords, (nz, 16, 16), dcf=traj.density_compensation
    )
    assert _corr(recon, ds.ground_truth) > 0.55


def test_trajectory_units_and_coverage() -> None:
    coords, dcf = T.radial_2d(32, 40)
    assert coords.shape == (2, 40, 32)
    assert dcf.shape == (40, 32)
    # spokes span the Nyquist window [-N/2, N/2)
    assert abs(coords.max() - 16) < 1.0 and abs(coords.min() + 16) < 1.0

    c3, d3 = T.radial_3d(24, 100)
    assert c3.shape[0] == 3
    # center-out: radius starts at 0
    r = np.sqrt((c3**2).sum(0))
    assert r.min() < 1e-9
