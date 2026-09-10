"""Shared type aliases and protocols.

UniMRI targets the Python array API standard so that operators can run on NumPy,
CuPy, or PyTorch arrays interchangeably. We deliberately avoid importing those
libraries here; ``Array`` is intentionally loose.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

# A path on disk, as accepted by ``open`` / ``pathlib.Path``.
PathLike = str | os.PathLike[str]

# An n-dimensional array. Kept as ``Any`` on purpose: the concrete type depends
# on the active backend (numpy.ndarray, cupy.ndarray, torch.Tensor, ...).
Array = Any


@runtime_checkable
class ArrayNamespace(Protocol):
    """The subset of the array-API namespace UniMRI relies on."""

    def asarray(self, obj: Any, /, **kwargs: Any) -> Array: ...
    def zeros(self, shape: Any, /, **kwargs: Any) -> Array: ...
    def conj(self, x: Array, /) -> Array: ...

    @property
    def pi(self) -> float: ...
