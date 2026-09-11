"""Optimization primitives.

Backend-agnostic solvers that operate purely through
:class:`~unimri.operators.base.LinearOperator`:

- ``conjugate_gradient`` -- Hermitian PSD normal-equation solver (CG-SENSE core). Implemented.

Planned (see ``docs/roadmap.md``): FISTA / proximal-gradient, ADMM, and
regularizers (L1, total variation, wavelet, locally-low-rank).
"""

from __future__ import annotations

from unimri.optimization.cg import conjugate_gradient

__all__ = ["conjugate_gradient"]
