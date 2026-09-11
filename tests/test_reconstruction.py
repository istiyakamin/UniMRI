"""Tests for the NUFFT operator and the ``adjoint`` reconstruction method."""

from __future__ import annotations

import numpy as np
import pytest

import unimri
from unimri.data import SamplingPattern, Trajectory, TrajectoryUnits
from unimri.exceptions import UniMRIError
from unimri.operators import FourierOperator, SensitivityOperator
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


# -- FourierOperator ---------------------------------------------------------


def test_fourier_operator_matches_reference_ndft_exactly() -> None:
    rng = np.random.default_rng(0)
    n = 10
    img = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    got = FourierOperator((n, n)).forward(img)
    ref = ndft_forward(img, T.cartesian(n, n_dims=2)).reshape(n, n)
    assert np.abs(got - ref).max() < 1e-9


def test_fourier_operator_dot_test_and_exact_normal() -> None:
    op = FourierOperator((12, 8))
    assert op.dot_test()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((12, 8)) + 1j * rng.standard_normal((12, 8))
    assert np.allclose(op.normal(x), 12 * 8 * x)


def test_fourier_operator_batches_over_leading_axis() -> None:
    op = FourierOperator((10, 10))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 10, 10)) + 1j * rng.standard_normal((4, 10, 10))
    batched = op._forward(x)
    stacked = np.stack([op.forward(x[c]) for c in range(4)])
    assert np.allclose(batched, stacked)


# -- SensitivityOperator ------------------------------------------------------


def test_sensitivity_operator_dot_test_and_exact_normal() -> None:
    rng = np.random.default_rng(3)
    sens = rng.standard_normal((5, 6, 6)) + 1j * rng.standard_normal((5, 6, 6))
    op = SensitivityOperator(sens)
    assert op.dot_test(in_shape=(6, 6), out_shape=(5, 6, 6))
    x = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    assert np.allclose(op.normal(x), (np.abs(sens) ** 2).sum(0) * x)


# -- CG-SENSE ------------------------------------------------------------


def test_reconstruct_cg_beats_adjoint_on_undersampled_radial() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=28, n_coils=4, noise_std=0.02, seed=8)
    img_adjoint = unimri.reconstruct(ds.data, method="adjoint")
    img_cg = unimri.reconstruct(ds.data, method="cg", n_iter=15, l2=1e-3)
    assert img_cg.shape == img_adjoint.shape
    assert nrmse(img_cg, ds.ground_truth) < nrmse(img_adjoint, ds.ground_truth)


def test_reconstruct_cg_cartesian_matches_adjoint_closely() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=24, n_coils=4, seed=7)
    img_cg = unimri.reconstruct(ds.data, method="cg", n_iter=15, l2=1e-6)
    assert nrmse(img_cg, ds.ground_truth) < 0.05


def test_reconstruct_cg_accepts_precomputed_sensitivity() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=20, n_coils=3, seed=9)
    assert ds.coil_maps is not None
    img = unimri.reconstruct(ds.data, method="cg", n_iter=10, sensitivity=ds.coil_maps)
    assert img.shape == ds.ground_truth.shape
    assert nrmse(img, ds.ground_truth) < 0.05


def test_reconstruct_cg_records_provenance() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=16, n_coils=2, seed=10)
    unimri.reconstruct(ds.data, method="cg", n_iter=5)
    steps = [s for s in ds.data.provenance if s.operation == "reconstruct"]
    assert steps and steps[-1].params["method"] == "cg"
