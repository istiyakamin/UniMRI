"""Exception hierarchy for UniMRI.

All UniMRI-specific exceptions derive from :class:`UniMRIError` so that callers
can catch the whole family with a single ``except``.
"""

from __future__ import annotations


class UniMRIError(Exception):
    """Base class for all UniMRI errors."""


class ReaderError(UniMRIError):
    """A reader failed while parsing a raw-data file."""


class UnsupportedFormatError(ReaderError):
    """No registered reader recognises the given input."""


class ValidationError(UniMRIError):
    """An :class:`~unimri.data.MRIData` object (or a component) is inconsistent."""


class OperatorShapeError(UniMRIError):
    """An operator was applied to an array whose shape it does not accept."""


class BackendError(UniMRIError):
    """The requested array backend is unavailable or incompatible."""
