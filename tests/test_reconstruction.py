"""Tests for the NUFFT operator and the ``adjoint`` reconstruction method."""

from __future__ import annotations

import numpy as np
import pytest

import unimri
from unimri.data import SamplingPattern, Trajectory, TrajectoryUnits
from unimri.exceptions import UniMRIError
from unimri.testing import ndft_forward, synthetic_dataset
from unimri.testing import trajectories as T

finufft = pytest.importorskip("finufft")

from unimri.operators import NUFFTOperator  # noqa: E402  (after importorskip)


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    r = np.abs(recon).ravel() / (np.abs(recon).max() + 1e-12)
    t = np.abs(truth).ravel() / (np.abs(truth).max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def test_nufft_forward_matches_reference_ndft_2d() -> None:
    rng = np.random.default_rng(0)
    n = 20
    img = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    coords, _ = T.radial_2d(n, 32)
    op = NUFFTOperator(Trajectory(coords, TrajectoryUnits.NORMALIZED), (n, n))
    got = op.forward(img)
    ref = ndft_forward(img, coords).ravel()
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4


def test_nufft_forward_matches_reference_ndft_3d() -> None:
    rng = np.random.default_rng(1)
    n = 12
    img = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    coords, _ = T.radial_3d(n, 80, readout=n)
    op = NUFFTOperator(Trajectory(coords, TrajectoryUnits.NORMALIZED), (n, n, n))
    got = op.forward(img)
    ref = ndft_forward(img, coords).ravel()
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-4


@pytest.mark.parametrize("shape,n_spokes", [((20, 20), 30), ((14, 14, 14), 90)])
def test_nufft_passes_dot_test(shape: tuple[int, ...], n_spokes: int) -> None:
    if len(shape) == 2:
        coords, _ = T.radial_2d(shape[0], n_spokes)
    else:
        coords, _ = T.radial_3d(shape[0], n_spokes, readout=shape[0])
    op = NUFFTOperator(Trajectory(coords, TrajectoryUnits.NORMALIZED), shape)
    n_samples = int(np.prod(coords.shape[1:]))
    assert op.dot_test(in_shape=shape, out_shape=(n_samples,), rtol=1e-4)


def test_nufft_rejects_shape_mismatch() -> None:
    coords, _ = T.radial_2d(16, 20)
    with pytest.raises(ValueError, match="does not match"):
        NUFFTOperator(Trajectory(coords, TrajectoryUnits.NORMALIZED), (16, 16, 16))


def test_reconstruct_radial_recovers_phantom() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=32, n_coils=4, noise_std=0.01, seed=2)
    img = unimri.reconstruct(ds.data, method="adjoint")
    assert img.shape == (32, 32)
    assert nrmse(img, ds.ground_truth) < 0.6
    assert any(s.operation == "reconstruct" for s in ds.data.provenance)


def test_reconstruct_stack_of_stars_recovers_phantom() -> None:
    ds = synthetic_dataset(SamplingPattern.STACK_OF_STARS, matrix=16, n_coils=2, seed=3)
    img = unimri.reconstruct(ds.data, method="adjoint")
    assert img.shape == ds.ground_truth.shape
    assert nrmse(img, ds.ground_truth) < 0.6


def test_reconstruct_cartesian_is_exact_ifft() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=24, n_coils=3, seed=4)
    img = unimri.reconstruct(ds.data, method="adjoint")
    assert nrmse(img, ds.ground_truth) < 1e-6


def test_reconstruct_coil_combine_none_keeps_coil_axis() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=20, n_coils=3, seed=5)
    img = unimri.reconstruct(ds.data, method="adjoint", coil_combine="none")
    assert img.shape == (3, 20, 20)


def test_reconstruct_rejects_non_singleton_extra_axis(cartesian_mri_data) -> None:
    import dataclasses

    ks = np.stack([cartesian_mri_data.kspace, cartesian_mri_data.kspace], axis=0)
    bad = dataclasses.replace(
        cartesian_mri_data, kspace=ks, kspace_axes=("echo", "coil", "kz", "ky", "kx")
    )
    with pytest.raises(UniMRIError, match="non-singleton"):
        unimri.reconstruct(bad, method="adjoint")
