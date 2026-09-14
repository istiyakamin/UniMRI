"""The unified MRI raw-data container.

``MRIData`` is the pivot of the whole framework: every reader produces one, and
every operator / reconstruction / pipeline stage consumes and returns one (or an
image derived from one). Keeping this class small, explicit, and well-specified
is more important than any single algorithm built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unimri._typing import Array
from unimri.data.acquisition import AcquisitionInfo
from unimri.data.encoding import EncodingSpace
from unimri.data.metadata import Provenance, ScannerMetadata
from unimri.data.trajectory import Trajectory
from unimri.exceptions import ValidationError

#: Axis names UniMRI understands in ``MRIData.kspace_axes``. The three spatial /
#: readout axes must be the *last* axes, in one of these two groupings:
#:   Cartesian:      (..., "coil", "kz", "ky", "kx")
#:   non-Cartesian:  (..., "coil", "shot", "readout")
#: Leading axes describe everything else and may appear in any order.
KNOWN_AXES = frozenset(
    {
        "coil",
        "kx",
        "ky",
        "kz",
        "shot",
        "readout",
        "average",
        "contrast",
        "echo",
        "phase",
        "repetition",
        "set",
        "slice",
        "segment",
        "user",
    }
)


@dataclass
class CoilInfo:
    """Receive-coil information."""

    n_channels: int
    names: list[str] = field(default_factory=list)
    #: Optional ``(n_channels, n_channels)`` noise covariance for pre-whitening.
    noise_covariance: Array | None = None
    #: Optional ``(n_virtual, n_channels)`` coil-compression matrix already applied.
    compression_matrix: Array | None = None


@dataclass
class MRIData:
    """A vendor-independent representation of one MRI acquisition.

    Attributes
    ----------
    kspace:
        Complex k-space samples. See :data:`KNOWN_AXES` for the layout rules.
    kspace_axes:
        Names of the ``kspace`` axes, same length as ``kspace.ndim``.
    encoding:
        :class:`~unimri.data.EncodingSpace` -- geometry and sampling.
    acquisition:
        :class:`~unimri.data.AcquisitionInfo` -- sequence / contrast parameters.
    coils:
        :class:`CoilInfo`.
    trajectory:
        :class:`~unimri.data.Trajectory` for non-Cartesian data; ``None`` for
        Cartesian.
    metadata:
        :class:`~unimri.data.ScannerMetadata`.
    provenance:
        :class:`~unimri.data.Provenance` -- processing history.
    """

    kspace: Array
    kspace_axes: tuple[str, ...]
    encoding: EncodingSpace
    acquisition: AcquisitionInfo = field(default_factory=AcquisitionInfo)
    coils: CoilInfo | None = None
    trajectory: Trajectory | None = None
    metadata: ScannerMetadata = field(default_factory=ScannerMetadata)
    provenance: Provenance = field(default_factory=Provenance)

    # -- derived views -----------------------------------------------------

    @property
    def is_cartesian(self) -> bool:
        return self.trajectory is None and self.encoding.is_cartesian

    @property
    def n_coils(self) -> int:
        if "coil" not in self.kspace_axes:
            return 1
        return int(self.kspace.shape[self.kspace_axes.index("coil")])

    def axis(self, name: str) -> int:
        """Return the integer position of a named axis."""
        try:
            return self.kspace_axes.index(name)
        except ValueError as exc:
            raise KeyError(f"no axis {name!r} in {self.kspace_axes}") from exc

    # -- validation ------------------------------------------------------

    def validate(self) -> MRIData:
        """Check internal consistency. Returns ``self`` so calls can be chained.

        Raises
        ------
        ValidationError
            On the first inconsistency found.
        """
        ks = self.kspace

        if not hasattr(ks, "shape") or not hasattr(ks, "ndim"):
            raise ValidationError("kspace must be an array-like with .shape and .ndim")
        dtype = getattr(ks, "dtype", None)
        if dtype is not None and "complex" not in str(dtype):
            raise ValidationError(f"kspace must be complex, got dtype {dtype}")

        if len(self.kspace_axes) != ks.ndim:
            raise ValidationError(
                f"kspace_axes has {len(self.kspace_axes)} names but kspace is {ks.ndim}-D"
            )
        if len(set(self.kspace_axes)) != len(self.kspace_axes):
            raise ValidationError(f"duplicate axis names in {self.kspace_axes}")
        unknown = set(self.kspace_axes) - KNOWN_AXES
        if unknown:
            raise ValidationError(f"unknown axis names: {sorted(unknown)}")

        cart = {"kx", "ky"}.issubset(self.kspace_axes)
        noncart = {"shot", "readout"}.issubset(self.kspace_axes)
        if cart == noncart:
            raise ValidationError(
                "kspace_axes must contain either the Cartesian readout axes "
                "('kx','ky'[,'kz']) or the non-Cartesian axes ('shot','readout'), "
                "not both and not neither"
            )

        if noncart and self.trajectory is None:
            raise ValidationError("non-Cartesian kspace_axes require a trajectory")
        if self.trajectory is not None:
            if self.encoding.is_cartesian:
                raise ValidationError("a trajectory is set but encoding.sampling == CARTESIAN")
            traj_n = self.trajectory.n_samples
            ks_n = 1
            for name in ("shot", "readout"):
                ks_n *= int(ks.shape[self.axis(name)])
            if traj_n != ks_n:
                raise ValidationError(
                    f"trajectory has {traj_n} samples but kspace shot*readout = {ks_n}"
                )
            if self.trajectory.n_dims != self.encoding.n_dims:
                raise ValidationError(
                    f"trajectory is {self.trajectory.n_dims}-D but encoding.n_dims = "
                    f"{self.encoding.n_dims}"
                )

        if self.coils is not None and "coil" in self.kspace_axes:
            declared = self.coils.n_channels
            actual = int(ks.shape[self.axis("coil")])
            if declared != actual:
                raise ValidationError(
                    f"coils.n_channels ({declared}) != kspace coil axis ({actual})"
                )

        for label, mat in (
            ("recon_matrix", self.encoding.recon_matrix),
            ("encoded_matrix", self.encoding.encoded_matrix),
        ):
            if len(mat) != 3 or any(int(m) <= 0 for m in mat):
                raise ValidationError(f"encoding.{label} must be 3 positive ints, got {mat}")

        return self

    # -- display -------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"MRIData  {'Cartesian' if self.is_cartesian else self.encoding.sampling.value}",
            f"  kspace         : {tuple(self.kspace.shape)}  axes={self.kspace_axes}",
            f"  recon matrix   : {self.encoding.recon_matrix}",
            f"  fov (mm)       : {self.encoding.fov.as_tuple()}",
            f"  coils          : {self.n_coils}",
            f"  nucleus        : {self.acquisition.nucleus}",
        ]
        if self.trajectory is not None:
            lines.append(
                f"  trajectory     : {self.trajectory.n_dims}-D, "
                f"{self.trajectory.sample_shape} samples, units={self.trajectory.units.value}"
            )
        if self.metadata.vendor:
            lines.append(f"  scanner        : {self.metadata.vendor} {self.metadata.model or ''}")
        lines.append(f"  provenance     : {len(self.provenance)} step(s)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MRIData(kspace={tuple(self.kspace.shape)}, axes={self.kspace_axes}, "
            f"sampling={self.encoding.sampling.value}, coils={self.n_coils}, "
            f"nucleus={self.acquisition.nucleus!r})"
        )
