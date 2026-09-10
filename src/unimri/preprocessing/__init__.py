"""Raw-data preprocessing (planned).

Vendor-neutral operations applied to :class:`~unimri.data.MRIData` before
reconstruction: oversampling removal, noise pre-whitening, ramp-sampling
regridding, gradient-delay / trajectory correction, coil compression, and
asymmetric-echo handling. Each records a provenance step.

See ``docs/roadmap.md`` Milestone 3.
"""

from __future__ import annotations

__all__: list[str] = []
