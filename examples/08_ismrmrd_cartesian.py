"""Read a real Cartesian ISMRMRD file end-to-end: `unimri.read()` -> both
reconstruction methods -> score against known ground truth.

No local data needed -- this writes its own small ISMRMRD file (using the
`ismrmrd` package directly, not UniMRI, since UniMRI does not ship a writer)
from a synthetic phantom with a known answer, then reads it back exactly the
way a real scanner-exported `.mrd` file would be read.

Needs `pip install "unimri[ismrmrd]"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import unimri
from unimri.data import SamplingPattern
from unimri.testing import synthetic_dataset


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    r = np.abs(recon).ravel() / (np.abs(recon).max() + 1e-12)
    t = np.abs(truth).ravel() / (np.abs(truth).max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def write_ismrmrd_cartesian(path: Path, sd) -> None:  # noqa: ANN001 - SyntheticDataset
    """Write a valid Cartesian `MRIData` as a real ISMRMRD file, using the
    reference `ismrmrd` library directly. This is what a `siemens_to_ismrmrd` /
    `ge_to_ismrmrd` converter (or a scanner's own export) would hand you."""
    import ismrmrd
    import ismrmrd.xsd as xsd

    data = sd.data
    n_coils, ny, nx = data.kspace.shape
    fov = data.encoding.fov

    hdr = xsd.ismrmrdHeader(
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=127_728_000),
        acquisitionSystemInformation=xsd.acquisitionSystemInformationType(
            systemVendor="UniMRI-Example", receiverChannels=n_coils, systemFieldStrength_T=3.0
        ),
        measurementInformation=xsd.measurementInformationType(
            protocolName="unimri_example", patientPosition="HFS"
        ),
        encoding=[
            xsd.encodingType(
                encodedSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=nx, y=ny, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=fov.x, y=fov.y, z=fov.z),
                ),
                reconSpace=xsd.encodingSpaceType(
                    matrixSize=xsd.matrixSizeType(x=nx, y=ny, z=1),
                    fieldOfView_mm=xsd.fieldOfViewMm(x=fov.x, y=fov.y, z=fov.z),
                ),
                encodingLimits=xsd.encodingLimitsType(
                    kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=ny - 1, center=ny // 2)
                ),
                trajectory=xsd.trajectoryType("cartesian"),
            )
        ],
    )
    xmlstring = xsd.ToXML(hdr)
    if isinstance(xmlstring, bytes):
        xmlstring = xmlstring.decode("utf-8")

    if path.exists():
        path.unlink()
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    try:
        dset.write_xml_header(xmlstring)
        for ky in range(ny):
            acq = ismrmrd.Acquisition()
            acq.resize(nx, n_coils)
            acq.idx.kspace_encode_step_1 = ky
            acq.data[:] = data.kspace[:, ky, :]
            dset.append_acquisition(acq)
    finally:
        dset.close()


def main() -> None:
    try:
        import ismrmrd  # noqa: F401
    except ImportError:
        print('ismrmrd not installed -- pip install "unimri[ismrmrd]"')
        return

    sd = synthetic_dataset(SamplingPattern.CARTESIAN, matrix=48, n_coils=4, noise_std=0.01, seed=6)
    path = Path("example_cartesian.mrd")
    write_ismrmrd_cartesian(path, sd)
    print(f"wrote {path} ({path.stat().st_size} bytes)\n")

    data = unimri.read(path)  # vendor-agnostic dispatch, same call as for any format
    print(data.summary())
    print()

    adjoint = unimri.reconstruct(data, method="adjoint")
    cg = unimri.reconstruct(data, method="cg", n_iter=15, l2=1e-6)
    print(f"adjoint (iFFT)   : {adjoint.shape}  NRMSE {nrmse(adjoint, sd.ground_truth):.4f}")
    print(f"CG-SENSE         : {cg.shape}  NRMSE {nrmse(cg, sd.ground_truth):.4f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    imgs = [
        ("ground truth", np.abs(sd.ground_truth)),
        ("adjoint", np.abs(adjoint)),
        ("CG-SENSE", np.abs(cg)),
    ]
    fig, ax = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3))
    for a, (title, im) in zip(ax, imgs):
        a.imshow(im, cmap="gray")
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig("ismrmrd_cartesian.png", dpi=110)
    print("saved ismrmrd_cartesian.png")


if __name__ == "__main__":
    main()
