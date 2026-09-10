"""Add support for a new raw-data format by writing a Reader.

A Reader maps one file to one MRIData. Register it and `unimri.read()` picks it
up automatically by sniffing the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import unimri
from unimri.data import CoilInfo, EncodingSpace, FieldOfView, MRIData, SamplingPattern
from unimri.io import Reader, register_reader


class NpzCartesianReader(Reader):
    """Reads a trivial `.unpz` file: a JSON sidecar + a .npy k-space array.

    Real readers wrap `twixtools` / `ismrmrd`; this one is deliberately minimal.
    """

    format_name = "Toy NPZ Cartesian"
    extensions = (".unpz",)

    def can_read(self, path) -> bool:
        return Path(path).suffix.lower() == ".unpz"

    def read(self, path, **options) -> MRIData:
        p = Path(path)
        payload = json.loads(p.read_text())
        kspace = np.load(p.with_suffix(".npy"))
        nx, ny = payload["matrix"]
        data = MRIData(
            kspace=kspace.astype(np.complex64),
            kspace_axes=("coil", "ky", "kx"),
            encoding=EncodingSpace(
                recon_matrix=(nx, ny, 1),
                encoded_matrix=(nx, ny, 1),
                fov=FieldOfView(*payload["fov"]),
                sampling=SamplingPattern.CARTESIAN,
                n_dims=2,
            ),
            coils=CoilInfo(n_channels=kspace.shape[0]),
        )
        data.provenance.record("NpzCartesianReader.read", params={"path": str(p)})
        return data.validate()


def main() -> None:
    register_reader(NpzCartesianReader(), prepend=True)

    # Write a toy file.
    tmp = Path("toy_scan.unpz")
    ks = np.random.default_rng(0).standard_normal((4, 64, 64)) + 1j * np.random.default_rng(
        1
    ).standard_normal((4, 64, 64))
    np.save(tmp.with_suffix(".npy"), ks)
    tmp.write_text(json.dumps({"matrix": [64, 64], "fov": [220.0, 220.0, 5.0]}))

    try:
        data = unimri.read(tmp)  # dispatched to NpzCartesianReader
        print(data.summary())
        print()
        print(data.provenance)
    finally:
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".npy").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
