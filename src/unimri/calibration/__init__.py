"""Calibration.

Coil-sensitivity estimation -- currently a root-sum-of-squares baseline
(``method="rss"``); ESPIRiT and adaptive/Walsh combine are planned, wrapping
``sigpy.mri`` / BART rather than reimplementing them. GRAPPA kernel calibration
and density-compensation estimation (Pipe-Menon) are also planned.

See ``docs/roadmap.md`` Milestone 4-5.
"""

from __future__ import annotations

from unimri.calibration.sensitivity import estimate_sensitivity

__all__ = ["estimate_sensitivity"]
