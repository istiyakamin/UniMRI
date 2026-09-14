"""ISMRMRDReader: write a real ISMRMRD file with the ``ismrmrd`` package, read
it back with UniMRI, and check both the raw round-trip and the full
read -> reconstruct pipeline against known ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ismrmrd = pytest.importorskip("ismrmrd")
import ismrmrd.xsd as ismrmrd_xsd  # noqa: E402

import unimri  # noqa: E402
from unimri.data import MRIData, SamplingPattern  # noqa: E402
from unimri.exceptions import ReaderError  # noqa: E402
from unimri.io.ismrmrd import ISMRMRDReader  # noqa: E402
from unimri.testing import synthetic_dataset  # noqa: E402


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    """Scale-invariant error: reconstruct(method="adjoint") is a gridding/adjoint
    image, not amplitude-calibrated against the source, so compare shapes only."""
    r = np.abs(recon).ravel() / (np.abs(recon).max() + 1e-12)
    t = np.abs(truth).ravel() / (np.abs(truth).max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def _write_ismrmrd_cartesian(
    path: Path,
    data: MRIData,
    *,
    trajectory: str = "cartesian",
    n_slices: int = 1,
) -> None:
    """Write ``data`` (a valid 2-D Cartesian ``MRIData``) as a real ISMRMRD file.

    UniMRI does not ship an ISMRMRD *writer* -- this is test-only plumbing that
    exercises the reader against the reference library's own file format,
    rather than against UniMRI's own (possibly self-consistently wrong)
    assumptions about it.
    """
    n_coils, ny, nx = data.kspace.shape
    fov = data.encoding.fov

    hdr = ismrmrd_xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd_xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=127_728_000
        ),
        acquisitionSystemInformation=ismrmrd_xsd.acquisitionSystemInformationType(
            systemVendor="UniMRI-Test",
            systemModel="Synthetic",
            receiverChannels=n_coils,
            systemFieldStrength_T=data.acquisition.field_strength_t or 3.0,
            institutionName="UniMRI CI",
        ),
        measurementInformation=ismrmrd_xsd.measurementInformationType(
            protocolName="unimri_test", patientPosition="HFS"
        ),
        encoding=[
            ismrmrd_xsd.encodingType(
                encodedSpace=ismrmrd_xsd.encodingSpaceType(
                    matrixSize=ismrmrd_xsd.matrixSizeType(x=nx, y=ny, z=1),
                    fieldOfView_mm=ismrmrd_xsd.fieldOfViewMm(x=fov.x, y=fov.y, z=fov.z),
                ),
                reconSpace=ismrmrd_xsd.encodingSpaceType(
                    matrixSize=ismrmrd_xsd.matrixSizeType(x=nx, y=ny, z=1),
                    fieldOfView_mm=ismrmrd_xsd.fieldOfViewMm(x=fov.x, y=fov.y, z=fov.z),
                ),
                encodingLimits=ismrmrd_xsd.encodingLimitsType(
                    kspace_encoding_step_1=ismrmrd_xsd.limitType(
                        minimum=0, maximum=ny - 1, center=ny // 2
                    ),
                    slice=ismrmrd_xsd.limitType(minimum=0, maximum=n_slices - 1, center=0),
                ),
                trajectory=ismrmrd_xsd.trajectoryType(trajectory),
            )
        ],
    )
    xmlstring = ismrmrd_xsd.ToXML(hdr)
    if isinstance(xmlstring, bytes):
        xmlstring = xmlstring.decode("utf-8")

    if path.exists():
        path.unlink()
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    try:
        dset.write_xml_header(xmlstring)
        for sl in range(n_slices):
            for ky in range(ny):
                acq = ismrmrd.Acquisition()
                acq.resize(nx, n_coils)
                acq.idx.kspace_encode_step_1 = ky
                acq.idx.kspace_encode_step_2 = 0
                acq.idx.slice = sl
                acq.data[:] = data.kspace[:, ky, :]
                if ky == 0:
                    acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
                if ky == ny - 1:
                    acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
                dset.append_acquisition(acq)
        # A noise-only acquisition, to check it's correctly skipped.
        noise = ismrmrd.Acquisition()
        noise.resize(nx, n_coils)
        noise.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        dset.append_acquisition(noise)
    finally:
        dset.close()


@pytest.fixture
def cartesian_ismrmrd_file(tmp_path: Path) -> tuple[Path, MRIData, np.ndarray]:
    sd = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=24, n_coils=3, seed=1)
    path = tmp_path / "synthetic.mrd"
    _write_ismrmrd_cartesian(path, sd.data)
    return path, sd.data, sd.ground_truth


def test_ismrmrd_reader_is_registered_and_dispatches(cartesian_ismrmrd_file) -> None:
    path, _, _ = cartesian_ismrmrd_file
    data = unimri.read(path)
    assert isinstance(data, MRIData)


def test_ismrmrd_round_trips_kspace_exactly(cartesian_ismrmrd_file) -> None:
    path, original, _ = cartesian_ismrmrd_file
    data = ISMRMRDReader().read(path)
    assert data.kspace_axes == ("coil", "ky", "kx")
    assert data.n_coils == original.n_coils
    np.testing.assert_allclose(data.kspace, original.kspace)


def test_ismrmrd_reads_encoding_and_metadata(cartesian_ismrmrd_file) -> None:
    path, original, _ = cartesian_ismrmrd_file
    data = ISMRMRDReader().read(path)
    assert data.encoding.sampling is SamplingPattern.CARTESIAN
    assert data.encoding.n_dims == 2
    assert data.encoding.recon_matrix[:2] == original.encoding.recon_matrix[:2]
    assert data.metadata.vendor == "UniMRI-Test"
    assert data.metadata.receiver_channels == original.n_coils
    assert data.acquisition.protocol_name == "unimri_test"
    assert len(data.provenance) == 1
    assert data.provenance.steps[0].operation == "read_ismrmrd"


def test_ismrmrd_skips_noise_measurements(cartesian_ismrmrd_file) -> None:
    # The fixture appends one ACQ_IS_NOISE_MEASUREMENT acquisition; if it were
    # not skipped it would corrupt the last k-space line written.
    path, original, _ = cartesian_ismrmrd_file
    data = ISMRMRDReader().read(path)
    np.testing.assert_allclose(data.kspace, original.kspace)


def test_ismrmrd_end_to_end_reconstruction_matches_ground_truth(cartesian_ismrmrd_file) -> None:
    path, _, ground_truth = cartesian_ismrmrd_file
    data = unimri.read(path)
    image = unimri.reconstruct(data, method="adjoint")
    assert nrmse(image, ground_truth) < 1e-6


def test_ismrmrd_rejects_non_cartesian_trajectory(tmp_path: Path) -> None:
    sd = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=16, n_coils=1, seed=2)
    path = tmp_path / "radial.mrd"
    _write_ismrmrd_cartesian(path, sd.data, trajectory="radial")
    with pytest.raises(ReaderError, match="Cartesian"):
        ISMRMRDReader().read(path)


def test_ismrmrd_rejects_multi_slice(tmp_path: Path) -> None:
    sd = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=16, n_coils=1, seed=3)
    path = tmp_path / "multislice.mrd"
    _write_ismrmrd_cartesian(path, sd.data, n_slices=2)
    with pytest.raises(ReaderError, match="slice"):
        ISMRMRDReader().read(path)
