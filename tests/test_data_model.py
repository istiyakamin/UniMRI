"""Contract tests for the unified data model."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unimri.data import (
    EncodingSpace,
    FieldOfView,
    MRIData,
    SamplingPattern,
    gamma_hz_per_t,
)
from unimri.exceptions import ValidationError


def test_cartesian_fixture_validates(cartesian_mri_data: MRIData) -> None:
    assert cartesian_mri_data.validate() is cartesian_mri_data
    assert cartesian_mri_data.is_cartesian
    assert cartesian_mri_data.n_coils == 4
    assert cartesian_mri_data.axis("kx") == 3
    assert "MRIData" in cartesian_mri_data.summary()


def test_radial_fixture_validates(radial_mri_data: MRIData) -> None:
    assert radial_mri_data.validate() is radial_mri_data
    assert not radial_mri_data.is_cartesian
    assert radial_mri_data.trajectory is not None
    assert radial_mri_data.trajectory.n_dims == 2
    assert radial_mri_data.trajectory.sample_shape == (33, 64)


def test_axis_lookup_raises_for_unknown(cartesian_mri_data: MRIData) -> None:
    with pytest.raises(KeyError):
        cartesian_mri_data.axis("shot")


def test_non_complex_kspace_rejected(cartesian_mri_data: MRIData) -> None:
    bad = dataclasses.replace(cartesian_mri_data, kspace=np.zeros((4, 8, 16, 16), dtype=np.float32))
    with pytest.raises(ValidationError, match="complex"):
        bad.validate()


def test_axis_count_mismatch_rejected(cartesian_mri_data: MRIData) -> None:
    bad = dataclasses.replace(cartesian_mri_data, kspace_axes=("coil", "kz", "ky"))
    with pytest.raises(ValidationError, match="axes"):
        bad.validate()


def test_unknown_axis_name_rejected(cartesian_mri_data: MRIData) -> None:
    bad = dataclasses.replace(cartesian_mri_data, kspace_axes=("coil", "kz", "ky", "frequency"))
    with pytest.raises(ValidationError, match="unknown axis"):
        bad.validate()


def test_neither_cartesian_nor_noncartesian_axes_rejected(cartesian_mri_data: MRIData) -> None:
    ks = cartesian_mri_data.kspace[..., 0]
    bad = dataclasses.replace(cartesian_mri_data, kspace=ks, kspace_axes=("coil", "kz", "ky"))
    with pytest.raises(ValidationError):
        bad.validate()


def test_coil_count_mismatch_rejected(cartesian_mri_data: MRIData) -> None:
    from unimri.data import CoilInfo

    bad = dataclasses.replace(cartesian_mri_data, coils=CoilInfo(n_channels=8))
    with pytest.raises(ValidationError, match="n_channels"):
        bad.validate()


def test_trajectory_sample_count_must_match_kspace(radial_mri_data: MRIData) -> None:
    short = radial_mri_data.trajectory.coords[:, :, :32]
    bad_traj = dataclasses.replace(radial_mri_data.trajectory, coords=short)
    bad = dataclasses.replace(radial_mri_data, trajectory=bad_traj)
    with pytest.raises(ValidationError, match="samples"):
        bad.validate()


def test_trajectory_with_cartesian_encoding_rejected(radial_mri_data: MRIData) -> None:
    enc = dataclasses.replace(radial_mri_data.encoding, sampling=SamplingPattern.CARTESIAN)
    bad = dataclasses.replace(radial_mri_data, encoding=enc)
    with pytest.raises(ValidationError, match="CARTESIAN"):
        bad.validate()


def test_bad_encoding_matrix_rejected() -> None:
    enc = EncodingSpace(
        recon_matrix=(0, 16, 16),
        encoded_matrix=(16, 16, 16),
        fov=FieldOfView(240.0, 240.0, 240.0),
    )
    data = MRIData(
        kspace=np.zeros((1, 4, 4, 4), dtype=np.complex64),
        kspace_axes=("coil", "kz", "ky", "kx"),
        encoding=enc,
    )
    with pytest.raises(ValidationError, match="recon_matrix"):
        data.validate()


def test_provenance_records_steps(cartesian_mri_data: MRIData) -> None:
    prov = cartesian_mri_data.provenance
    assert len(prov) == 0
    step = prov.record("remove_oversampling", params={"axis": "kx"}, backend="numpy")
    assert len(prov) == 1
    assert step.operation == "remove_oversampling"
    assert step.params == {"axis": "kx"}
    assert step.unimri_version  # populated from unimri.__version__


def test_multinuclear_fields(radial_mri_data: MRIData) -> None:
    acq = radial_mri_data.acquisition
    assert acq.is_multinuclear()
    assert acq.nucleus == "23Na"
    g = acq.gamma_hz_per_t
    assert g is not None and abs(g - 11.262e6) < 1e3
    assert acq.expected_larmor_hz == pytest.approx(abs(g) * 3.0)


def test_gamma_table_unknown_nucleus() -> None:
    assert gamma_hz_per_t("42Xx") is None
