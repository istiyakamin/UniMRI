"""Sampling / masking operator (planned).

``SamplingOperator`` will apply a k-space sampling pattern (the ``P`` in
``y = P F S x``):

    forward:  full k-space   ->  acquired samples   (mask / gather)
    adjoint:  acquired samples ->  full k-space      (zero-fill / scatter)

For Cartesian data this is an undersampling mask; for non-Cartesian data the
"sampling" is folded into the NUFFT and this operator is usually the identity.

See ``docs/roadmap.md`` Milestone 4.
"""

from __future__ import annotations

__all__: list[str] = []
