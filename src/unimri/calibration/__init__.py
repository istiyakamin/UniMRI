"""Calibration (planned).

Coil-sensitivity estimation (ESPIRiT, adaptive/Walsh combine, low-res gridded
maps for non-Cartesian), GRAPPA kernel calibration, and density-compensation
estimation (analytic, Voronoi, Pipe-Menon iterative).

Where possible these wrap ``sigpy.mri`` / BART rather than reimplementing.
See ``docs/roadmap.md`` Milestone 4-5.
"""

from __future__ import annotations

__all__: list[str] = []
