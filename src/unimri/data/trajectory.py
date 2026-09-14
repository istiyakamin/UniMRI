"""K-space trajectory container.

**Unit convention (important).** UniMRI stores trajectory coordinates in
*normalised k-space*: one unit of ``coords`` equals one sample of the encoded
matrix, and the fully-sampled Nyquist window spans ``[-N/2, N/2)`` along each
axis (equivalently ``[-0.5, 0.5)`` cycles per voxel after dividing by ``N``).
This matches the convention used by BART and ``mri-nufft``'s "unitless" mode.
Backends that want radians-per-voxel (e.g. ``torchkbnufft``) multiply by
``2*pi/N``; this conversion lives in the NUFFT operator, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from unimri._typing import Array


class TrajectoryUnits(str, Enum):
    #: Samples of the encoded matrix; Nyquist window ``[-N/2, N/2)``.
    NORMALIZED = "normalized"
    #: Cycles per FOV; Nyquist window ``[-0.5, 0.5)``.
    CYCLES_PER_FOV = "cycles_per_fov"
    #: Radians per voxel; Nyquist window ``[-pi, pi)``.
    RADIANS_PER_VOXEL = "radians_per_voxel"
    #: Physical 1/m.
    INVERSE_METERS = "inverse_meters"


@dataclass
class Trajectory:
    """Sample locations in k-space for a non-Cartesian (or arbitrary) acquisition.

    Attributes
    ----------
    coords:
        Array of shape ``(n_dims, ...)`` where the trailing axes index the
        readout samples (e.g. ``(n_dims, n_shots, n_readout)``). ``n_dims`` is
        2 or 3. **Row ``d`` corresponds to image axis ``d``** (NumPy order):
        ``(ky, kx)`` for a 2-D image ``(ny, nx)``, ``(kz, ky, kx)`` for 3-D
        ``(nz, ny, nx)``.
    units:
        Interpretation of ``coords`` -- see the module docstring.
    density_compensation:
        Optional array broadcastable to ``coords[0]``; per-sample DCF weights.
    dwell_time_s:
        Optional readout dwell time, for off-resonance / trajectory timing.
    """

    coords: Array
    units: TrajectoryUnits = TrajectoryUnits.NORMALIZED
    density_compensation: Array | None = None
    dwell_time_s: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def n_dims(self) -> int:
        return int(self.coords.shape[0])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of the readout-sample axes (everything after ``n_dims``)."""
        return tuple(int(s) for s in self.coords.shape[1:])

    @property
    def n_samples(self) -> int:
        n = 1
        for s in self.sample_shape:
            n *= s
        return n
