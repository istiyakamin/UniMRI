"""Reader registry and the top-level :func:`read` dispatcher."""

from __future__ import annotations

from pathlib import Path

from unimri._typing import PathLike
from unimri.data import MRIData
from unimri.exceptions import UnsupportedFormatError
from unimri.io.base import Reader

_READERS: list[Reader] = []


def register_reader(reader: Reader, *, prepend: bool = False) -> Reader:
    """Register a :class:`Reader` instance for use by :func:`read`.

    Parameters
    ----------
    reader:
        The reader instance.
    prepend:
        If ``True``, give this reader priority over already-registered ones.
    """
    if prepend:
        _READERS.insert(0, reader)
    else:
        _READERS.append(reader)
    return reader


def available_readers() -> list[Reader]:
    """Return the registered readers, in priority order."""
    return list(_READERS)


def find_reader(path: PathLike) -> Reader:
    """Return the first registered reader whose :meth:`Reader.can_read` accepts ``path``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    for reader in _READERS:
        try:
            if reader.can_read(p):
                return reader
        except Exception:  # noqa: BLE001 - a broken reader must not block others
            continue
    known = ", ".join(sorted({ext for r in _READERS for ext in r.extensions})) or "(none)"
    raise UnsupportedFormatError(
        f"no registered reader recognises {p.name!r}. Known extensions: {known}"
    )


def read(path: PathLike, **options: object) -> MRIData:
    """Read a raw-data file into :class:`MRIData`, choosing the reader automatically.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    UnsupportedFormatError
        If no registered reader recognises the file.
    """
    return find_reader(path).read(path, **options)


def _register_builtin_readers() -> None:
    """Register the readers that ship with UniMRI (interfaces only for now)."""
    from unimri.io.hdf5 import HDF5Reader
    from unimri.io.ismrmrd import ISMRMRDReader
    from unimri.io.twix import TwixReader

    for cls in (ISMRMRDReader, HDF5Reader, TwixReader):
        register_reader(cls())


_register_builtin_readers()
