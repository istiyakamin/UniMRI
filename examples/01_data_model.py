"""Build an MRIData by hand, validate it, and inspect it.

This is the representation every reader produces and every operator consumes.
"""

from __future__ import annotations

import numpy as np

from unimri.data import (
    AcquisitionInfo,
    CoilInfo,
    EncodingSpace,
    FieldOfView,
    MRIData,
    SamplingPattern,
    ScannerMetadata,
)


def main() -> None:
    # A tiny Cartesian dataset: 8 coils, 32 x 32 x 12 k-space.
    rng = np.random.default_rng(0)
    nc, nz, ny, nx = 8, 12, 32, 32
    kspace = rng.standard_normal((nc, nz, ny, nx)) + 1j * rng.standard_normal((nc, nz, ny, nx))
    kspace = kspace.astype(np.complex64)

    data = MRIData(
        kspace=kspace,
        # spatial axes come last; see docs/data-model.md
        kspace_axes=("coil", "kz", "ky", "kx"),
        encoding=EncodingSpace(
            recon_matrix=(nx, ny, nz),
            encoded_matrix=(nx, ny, nz),
            fov=FieldOfView(x=240.0, y=240.0, z=144.0),
            sampling=SamplingPattern.CARTESIAN,
            n_dims=3,
        ),
        acquisition=AcquisitionInfo(
            sequence_name="gre_3d",
            tr_s=8e-3,
            te_s=[3.5e-3],
            flip_angle_deg=12.0,
            nucleus="1H",
            field_strength_t=3.0,
        ),
        coils=CoilInfo(n_channels=nc),
        metadata=ScannerMetadata(vendor="Example", model="Sim 3T", field_strength_t=3.0),
    )

    # validate() raises ValidationError on any inconsistency and returns self.
    data.validate()

    # A processing step would record itself here; we do it manually to show the log.
    data.provenance.record("example_load", params={"note": "hand-built"}, backend="numpy")

    print(data.summary())
    print()
    print(data.provenance)
    print()
    print("is_cartesian :", data.is_cartesian)
    print("n_coils      :", data.n_coils)
    print("kx axis index:", data.axis("kx"))


if __name__ == "__main__":
    main()
