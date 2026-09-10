"""The reader interface.

A :class:`Reader` turns one on-disk raw-data file into one :class:`MRIData`.
Readers are registered with :func:`unimri.io.registry.register_reader` and
selected automatically by :func:`unimri.read` via :meth:`Reader.can_read`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from unimri._typing import PathLike
from unimri.data import MRIData


class Reader(ABC):
    """Base class for all raw-data readers."""

    #: Human-readable format name, e.g. "Siemens TWIX".
    format_name: str = "unknown"
    #: File extensions this reader typically handles, lowercase, with dot.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def can_read(self, path: PathLike) -> bool:
        """Return ``True`` if this reader recognises ``path``.

        Implementations should be cheap: check the extension and, if needed,
        sniff a few magic bytes. Do not parse the whole file here.
        """

    @abstractmethod
    def read(self, path: PathLike, **options: object) -> MRIData:
        """Parse ``path`` and return a validated :class:`MRIData`.

        Implementations must call ``.validate()`` on the result and record a
        provenance step describing the read.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} format={self.format_name!r}>"
