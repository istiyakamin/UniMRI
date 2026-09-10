"""Optimization primitives (planned).

Backend-agnostic solvers that operate purely through :class:`LinearOperator`:
conjugate gradient (for ``AᴴA x = Aᴴ y``), FISTA / proximal-gradient, ADMM, and
regularizers (L1, total variation, wavelet, locally-low-rank).

See ``docs/roadmap.md`` Milestone 6.
"""

from __future__ import annotations

__all__: list[str] = []
