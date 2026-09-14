"""ISMRMRD reader, wrapping the reference ``ismrmrd`` Python package.

Scope (Milestone 1, see ``docs/roadmap.md``): a single Cartesian encoding
space, with exactly one slice / average / contrast / repetition / set /
segment. Multi-dimensional acquisitions (multi-slice, multi-average, ...)
and non-Cartesian ISMRMRD trajectories are not yet supported -- both raise a
clear :class:`~unimri.exceptions.ReaderError` rather than silently dropping
data. Noise-measurement acquisitions (``ACQ_IS_NOISE_MEASUREMENT``) are
skipped; pre-whitening is not yet applied.

ISMRMRD's header does not carry a nucleus label, so :attr:`AcquisitionInfo.nucleus`
is left at its "1H" default; multinuclear ISMRMRD files need it set manually
after reading (``data.acquisition.nucleus = "23Na"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from unimri._typing import PathLike
from unimri.data import (
    AcquisitionInfo,
    CoilInfo,
    EncodingSpace,
    FieldOfView,
    MRIData,
    SamplingPattern,
    ScannerMetadata,
)
from unimri.exceptions import BackendError, ReaderError
from unimri.io.base import Reader

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ISMRMRDReader(Reader):
    format_name = "ISMRMRD"
    extensions = (".mrd", ".h5", ".ismrmrd")

    def can_read(self, path: PathLike) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self.extensions:
            return False
        try:
            with p.open("rb") as fh:
                if fh.read(8) != _HDF5_MAGIC:
                    return False
        except OSError:
            return False
        return p.suffix.lower() != ".h5" or _looks_like_ismrmrd(p)

    def read(self, path: PathLike, **options: object) -> MRIData:
        """Read an ISMRMRD file.

        Options
        -------
        dataset_name:
            HDF5 group holding the dataset (default ``"dataset"``, the
            ISMRMRD convention).
        """
        try:
            import ismrmrd
            import ismrmrd.xsd as ismrmrd_xsd
        except ImportError as exc:
            raise BackendError(
                "ISMRMRDReader needs the 'ismrmrd' package: pip install \"unimri[ismrmrd]\""
            ) from exc

        dataset_name = str(options.get("dataset_name", "dataset"))
        p = Path(path)
        dset = ismrmrd.Dataset(str(p), dataset_name, create_if_needed=False)
        try:
            return self._read_open(p, dset, ismrmrd, ismrmrd_xsd, dataset_name)
        finally:
            dset.close()

    def _read_open(
        self, path: Path, dset: Any, ismrmrd: Any, xsd: Any, dataset_name: str
    ) -> MRIData:
        hdr = xsd.CreateFromDocument(dset.read_xml_header())
        if not hdr.encoding:
            raise ReaderError(f"{path}: ISMRMRD header has no <encoding> element")
        enc = hdr.encoding[0]

        trajectory_kind = getattr(enc.trajectory, "value", enc.trajectory)
        if trajectory_kind != "cartesian":
            raise ReaderError(
                f"{path}: ISMRMRDReader only supports Cartesian trajectories so far "
                f"(this file is {trajectory_kind!r}). Non-Cartesian ISMRMRD is tracked "
                "in docs/roadmap.md (Milestone 2/5)."
            )

        enc_matrix = enc.encodedSpace.matrixSize
        rec_matrix = enc.reconSpace.matrixSize
        enc_fov = enc.encodedSpace.fieldOfView_mm
        rec_fov = enc.reconSpace.fieldOfView_mm
        n_dims = 3 if enc_matrix.z > 1 else 2

        limits = enc.encodingLimits

        def _extent(lim: Any) -> int:
            return 1 if lim is None else lim.maximum - lim.minimum + 1

        extra_dims = {
            "slice": _extent(limits.slice),
            "average": _extent(limits.average),
            "contrast": _extent(limits.contrast),
            "repetition": _extent(limits.repetition),
            "set": _extent(limits.set),
            "segment": _extent(limits.segment),
        }
        multi = {k: v for k, v in extra_dims.items() if v > 1}
        if multi:
            raise ReaderError(
                f"{path}: ISMRMRDReader (Milestone 1) supports a single slice / average / "
                f"contrast / repetition / set / segment; this file has {multi}. "
                "Multi-dimensional acquisitions are tracked in docs/roadmap.md."
            )

        kspace: np.ndarray | None = None
        n_coils = 0
        for i in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(i)
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                continue
            samples = np.asarray(acq.data)  # (active_channels, number_of_samples)
            if kspace is None:
                n_coils = int(acq.active_channels)
                n_kx = int(acq.number_of_samples)
                shape = (
                    (n_coils, enc_matrix.z, enc_matrix.y, n_kx)
                    if n_dims == 3
                    else (
                        n_coils,
                        enc_matrix.y,
                        n_kx,
                    )
                )
                kspace = np.zeros(shape, dtype=np.complex64)
            ky = acq.idx.kspace_encode_step_1
            if n_dims == 3:
                kz = acq.idx.kspace_encode_step_2
                kspace[:, kz, ky, :] = samples
            else:
                kspace[:, ky, :] = samples

        if kspace is None:
            raise ReaderError(f"{path}: no imaging acquisitions found (only noise measurements?)")

        axes = ("coil", "kz", "ky", "kx") if n_dims == 3 else ("coil", "ky", "kx")

        asi = hdr.acquisitionSystemInformation
        exp = hdr.experimentalConditions
        meas = hdr.measurementInformation
        field_strength = getattr(asi, "systemFieldStrength_T", None) if asi is not None else None
        larmor_hz = float(exp.H1resonanceFrequency_Hz) if exp is not None else None

        data = MRIData(
            kspace=kspace,
            kspace_axes=axes,
            encoding=EncodingSpace(
                recon_matrix=(rec_matrix.x, rec_matrix.y, rec_matrix.z if n_dims == 3 else 1),
                encoded_matrix=(enc_matrix.x, enc_matrix.y, enc_matrix.z if n_dims == 3 else 1),
                fov=FieldOfView(rec_fov.x, rec_fov.y, rec_fov.z if n_dims == 3 else enc_fov.z),
                sampling=SamplingPattern.CARTESIAN,
                n_dims=n_dims,
            ),
            acquisition=AcquisitionInfo(
                protocol_name=getattr(meas, "protocolName", None) if meas is not None else None,
                sequence_name=getattr(meas, "sequenceName", None) if meas is not None else None,
                field_strength_t=field_strength,
                larmor_hz=larmor_hz,
            ),
            coils=CoilInfo(n_channels=n_coils),
            metadata=ScannerMetadata(
                vendor=getattr(asi, "systemVendor", None) if asi is not None else None,
                model=getattr(asi, "systemModel", None) if asi is not None else None,
                field_strength_t=field_strength,
                institution=getattr(asi, "institutionName", None) if asi is not None else None,
                receiver_channels=getattr(asi, "receiverChannels", None)
                if asi is not None
                else None,
            ),
        )
        data.provenance.record(
            "read_ismrmrd",
            params={"path": str(path), "dataset_name": dataset_name},
            backend="ismrmrd",
        )
        return data.validate()


def _looks_like_ismrmrd(path: Path) -> bool:
    """Best-effort check for the ISMRMRD group structure inside an HDF5 file."""
    try:
        import h5py
    except ImportError:
        return True  # cannot disprove it; let ``read`` give the real error
    try:
        with h5py.File(path, "r") as fh:
            return any("xml" in fh[key] for key in fh if hasattr(fh[key], "keys"))
    except Exception:  # noqa: BLE001
        return False
