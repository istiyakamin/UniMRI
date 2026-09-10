"""Coil sensitivity operator (planned).

``SensitivityOperator`` will implement the ``S`` in ``y = P F S x``:

    forward:  single image        ->  per-coil images   (multiply by maps)
    adjoint:  per-coil images     ->  single image       (conj-map weighted sum)

Sensitivity maps come from ``unimri.calibration`` (ESPIRiT / adaptive-combine /
externally supplied). A related ``CoilCompressionOperator`` will handle SVD /
geometric coil compression.

See ``docs/roadmap.md`` Milestone 4.
"""

from __future__ import annotations

__all__: list[str] = []
