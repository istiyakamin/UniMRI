"""Generic HDF5 reader (interface only).

For lab / research datasets stored as plain HDF5 with a UniMRI-defined layout
(k-space dataset + attributes describing axes, encoding, and trajectory). This
is also the format :meth:`MRIData` round-trips to for caching and test fixtures.
"""

from __future__ import annotations

from pathlib import Path

from unimri._typing import PathLike
from unimri.data import MRIData
from unimri.io.base import Reader

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class HDF5Reader(Reader):
    format_name = "UniMRI HDF5"
    extensions = (".h5", ".hdf5")

    def can_read(self, path: PathLike) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self.extensions:
            return False
        try:
            with p.open("rb") as fh:
                return fh.read(8) == _HDF5_MAGIC
        except OSError:
            return False

    def read(self, path: PathLike, **options: object) -> MRIData:
        raise NotImplementedError(
            "HDF5Reader.read is not implemented yet (Milestone 1). "
            "The on-disk schema is specified in docs/data-model.md."
        )
