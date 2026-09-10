"""ISMRMRD reader (interface only).

ISMRMRD is the natural interchange format for UniMRI: it already standardises
acquisitions, encoding spaces, and trajectories. This reader will be the first
one implemented (see ``docs/roadmap.md``, Milestone 1). Wrapping the ``ismrmrd``
package is preferred over re-parsing the HDF5 layout by hand.
"""

from __future__ import annotations

from pathlib import Path

from unimri._typing import PathLike
from unimri.data import MRIData
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
        # A full check would open the file and look for the '/dataset/xml'
        # ISMRMRD header. Deferred until the reader is implemented.
        return p.suffix.lower() != ".h5" or _looks_like_ismrmrd(p)

    def read(self, path: PathLike, **options: object) -> MRIData:
        raise NotImplementedError(
            "ISMRMRDReader.read is not implemented yet (Milestone 1). "
            "Track it at https://github.com/istiyakamin/UniMRI/blob/main/docs/roadmap.md"
        )


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
