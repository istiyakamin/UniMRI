"""Tests for coil sensitivity estimation."""

from __future__ import annotations

import numpy as np
import pytest

from unimri.calibration import estimate_sensitivity
from unimri.data import SamplingPattern
from unimri.exceptions import UniMRIError
from unimri.testing import synthetic_dataset

finufft = pytest.importorskip("finufft")  # radial case needs the NUFFT operator


@pytest.mark.parametrize("pattern", [SamplingPattern.CARTESIAN, SamplingPattern.RADIAL])
def test_estimate_sensitivity_shape_and_normalization(pattern: SamplingPattern) -> None:
    ds = synthetic_dataset(pattern, matrix=20, n_coils=4, seed=1)
    sens = estimate_sensitivity(ds.data, method="rss")
    assert sens.shape == (4, *ds.ground_truth.shape)
    # RSS of the estimated maps is ~1 wherever the true image has signal
    rss = np.sqrt((np.abs(sens) ** 2).sum(axis=0))
    signal = np.abs(ds.ground_truth) > 0.1 * np.abs(ds.ground_truth).max()
    assert np.allclose(rss[signal], 1.0, atol=1e-6)


def test_estimate_sensitivity_matches_true_maps_up_to_phase() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=24, n_coils=4, seed=2)
    assert ds.coil_maps is not None
    sens = estimate_sensitivity(ds.data, method="rss")
    signal = np.abs(ds.ground_truth) > 0.2 * np.abs(ds.ground_truth).max()
    # both are already RSS-normalized coil-combination weights -> compare magnitudes
    got = np.abs(sens)[:, signal]
    true = np.abs(ds.coil_maps)[:, signal]
    assert np.corrcoef(got.ravel(), true.ravel())[0, 1] > 0.9


def test_estimate_sensitivity_rejects_unknown_method() -> None:
    ds = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=16, n_coils=2, seed=3)
    with pytest.raises(UniMRIError, match="unknown sensitivity method"):
        estimate_sensitivity(ds.data, method="espirit")
