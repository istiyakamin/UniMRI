"""Encoding-space description: matrix sizes, field of view, acceleration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SamplingPattern(str, Enum):
    """How k-space was sampled. Drives which operators a reconstruction builds."""

    CARTESIAN = "cartesian"
    RADIAL = "radial"
    SPIRAL = "spiral"
    STACK_OF_STARS = "stack_of_stars"
    STACK_OF_SPIRALS = "stack_of_spirals"
    CONES = "cones"
    ROSETTE = "rosette"
    PROPELLER = "propeller"
    ARBITRARY = "arbitrary"

    @property
    def is_cartesian(self) -> bool:
        return self is SamplingPattern.CARTESIAN


@dataclass
class FieldOfView:
    """Field of view in millimetres along (x, y, z)."""

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class EncodingSpace:
    """Geometry and sampling of the acquisition.

    Attributes
    ----------
    recon_matrix:
        Target image matrix ``(nx, ny, nz)`` after reconstruction.
    encoded_matrix:
        Nominal fully-sampled k-space matrix ``(nx, ny, nz)`` (before removing
        oversampling, before acceleration). For non-Cartesian data this is the
        gridding target.
    fov:
        Field of view in mm.
    sampling:
        The :class:`SamplingPattern`.
    acceleration:
        Parallel-imaging acceleration per phase-encode axis, e.g. ``(2, 1)``.
    partial_fourier:
        Partial-Fourier fraction per axis (1.0 == full), ``(ro, pe1, pe2)``.
    n_dims:
        2 or 3.
    affine:
        Optional 4x4 voxel-to-world matrix (RAS+), if the reader computed one.
    """

    recon_matrix: tuple[int, int, int]
    encoded_matrix: tuple[int, int, int]
    fov: FieldOfView
    sampling: SamplingPattern = SamplingPattern.CARTESIAN
    acceleration: tuple[int, ...] = (1,)
    partial_fourier: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_dims: int = 3
    affine: list[list[float]] | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def is_cartesian(self) -> bool:
        return self.sampling.is_cartesian

    @property
    def is_accelerated(self) -> bool:
        return any(a > 1 for a in self.acceleration)

    @property
    def has_partial_fourier(self) -> bool:
        return any(f < 1.0 for f in self.partial_fourier)
