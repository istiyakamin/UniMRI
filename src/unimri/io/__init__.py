"""Vendor-independent raw-data I/O.

Use :func:`read` for automatic format detection, or import a specific
:class:`Reader`. Register your own reader with :func:`register_reader`.
"""

from __future__ import annotations

from unimri.io.base import Reader
from unimri.io.hdf5 import HDF5Reader
from unimri.io.ismrmrd import ISMRMRDReader
from unimri.io.registry import (
    available_readers,
    find_reader,
    read,
    register_reader,
)
from unimri.io.twix import TwixReader

__all__ = [
    "Reader",
    "read",
    "find_reader",
    "register_reader",
    "available_readers",
    "ISMRMRDReader",
    "HDF5Reader",
    "TwixReader",
]
