"""Reconstruction methods.

Dispatched by name from :func:`unimri.reconstruct`:

- ``adjoint`` -- centered inverse FFT (Cartesian) or density-compensated
  gridding via the adjoint NUFFT (non-Cartesian). Implemented.

Planned (see ``docs/roadmap.md``): ``sense``, ``grappa``, ``cg``, ``cs``.

Each method is expressed via :mod:`unimri.operators` and, where iterative,
:mod:`unimri.optimization`.
"""

from __future__ import annotations

from unimri.reconstruction.methods import available_methods, register_method, run

__all__ = ["run", "register_method", "available_methods"]
