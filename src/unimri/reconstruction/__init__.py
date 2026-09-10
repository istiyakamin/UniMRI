"""Reconstruction methods (planned).

Registered by name and dispatched from :func:`unimri.reconstruct`:

- ``fft``     -- direct Cartesian inverse FFT + coil combine (Milestone 3)
- ``adjoint`` -- density-compensated gridding for non-Cartesian (Milestone 5)
- ``sense``   -- Cartesian SENSE (Milestone 4)
- ``grappa``  -- k-space parallel imaging (Milestone 4)
- ``cg``      -- iterative CG-SENSE, trajectory-agnostic (Milestone 6)
- ``cs``      -- compressed sensing (L1 / TV / wavelet) (Milestone 6)

Each method is expressed via :mod:`unimri.operators` and, where iterative,
:mod:`unimri.optimization`.
"""

from __future__ import annotations

__all__: list[str] = []
