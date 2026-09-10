"""Siemens TWIX (``.dat``) reader (interface only).

Will wrap ``twixtools`` (the same library RecoTwix uses) for parsing, then map
its 16-dimensional ``twix_array`` onto :class:`MRIData` -- including trajectory
extraction for radial / non-Cartesian sequences. See ``docs/roadmap.md``,
Milestone 2.
"""

from __future__ import annotations

import struct
from pathlib import Path

from unimri._typing import PathLike
from unimri.data import MRIData
from unimri.io.base import Reader


class TwixReader(Reader):
    format_name = "Siemens TWIX"
    extensions = (".dat",)

    def can_read(self, path: PathLike) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self.extensions:
            return False
        # VD/VE twix: first uint32 is 0, second is the measurement count (1..64).
        try:
            with p.open("rb") as fh:
                first, n_meas = struct.unpack("<II", fh.read(8))
        except (OSError, struct.error):
            return False
        return first == 0 and 1 <= n_meas <= 64

    def read(self, path: PathLike, **options: object) -> MRIData:
        raise NotImplementedError(
            "TwixReader.read is not implemented yet (Milestone 2). "
            "It will wrap twixtools; see docs/roadmap.md"
        )
