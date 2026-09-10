"""Minimal array-backend abstraction.

The goal is that operator and reconstruction code never imports ``numpy``
directly, but instead asks for the *array namespace* of whatever array it was
handed. That array might be a ``numpy.ndarray``, ``cupy.ndarray``, or
``torch.Tensor``.

This module is deliberately tiny for now. It will grow as real operators need
more of the array-API surface. It is not meant to be a full compatibility shim
like ``array-api-compat`` -- if we need that, we should depend on it.
"""

from __future__ import annotations

from typing import Any

from unimri._typing import Array, ArrayNamespace
from unimri.exceptions import BackendError


def get_namespace(*arrays: Array) -> ArrayNamespace:
    """Return the array-API namespace shared by ``arrays``.

    Falls back to NumPy when the arrays do not expose ``__array_namespace__``
    (e.g. plain Python lists, or older array libraries).

    Raises
    ------
    BackendError
        If two arrays come from incompatible backends.
    """
    namespaces = set()
    for arr in arrays:
        xp = getattr(arr, "__array_namespace__", None)
        if callable(xp):
            namespaces.add(xp())

    if len(namespaces) > 1:
        raise BackendError(
            f"arrays come from {len(namespaces)} different array backends; "
            "convert them to a common backend first"
        )
    if namespaces:
        return next(iter(namespaces))

    import numpy as np  # local import: numpy is a hard dep but keep this module import-light

    return np  # type: ignore[return-value]


def asarray(obj: Any, *, like: Array | None = None) -> Array:
    """Convert ``obj`` to an array in the same backend as ``like`` (or NumPy)."""
    xp = get_namespace(like) if like is not None else get_namespace()
    return xp.asarray(obj)
