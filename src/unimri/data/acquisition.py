"""Sequence / contrast parameters, including multinuclear information."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Gyromagnetic ratios, MHz/T (gamma-bar = gamma / 2*pi). Values from the IAEA /
# Bruker almanac, rounded. UniMRI is multinuclear from the start -- this is cheap
# and X-nuclei imaging (23Na, 31P, ...) is a first-class use case.
GYROMAGNETIC_RATIO_MHZ_PER_T: dict[str, float] = {
    "1H": 42.577478518,
    "2H": 6.536,
    "13C": 10.7084,
    "17O": -5.7716,
    "19F": 40.0776,
    "23Na": 11.2620,
    "31P": 17.2510,
    "129Xe": -11.7768,
}


def gamma_hz_per_t(nucleus: str) -> float | None:
    """Return gamma-bar in Hz/T for a nucleus label (e.g. ``"23Na"``), or ``None``."""
    mhz = GYROMAGNETIC_RATIO_MHZ_PER_T.get(nucleus)
    return None if mhz is None else mhz * 1e6


@dataclass
class AcquisitionInfo:
    """Pulse-sequence and contrast metadata for a dataset.

    Times are in seconds, angles in degrees, unless noted. Lists hold one entry
    per contrast/echo where applicable.
    """

    sequence_name: str | None = None
    protocol_name: str | None = None

    tr_s: float | None = None
    te_s: list[float] = field(default_factory=list)
    ti_s: list[float] = field(default_factory=list)
    flip_angle_deg: float | None = None
    n_contrasts: int = 1
    n_averages: int = 1

    #: Nucleus label, e.g. "1H", "23Na", "31P".
    nucleus: str = "1H"
    #: Static field strength in tesla (mirrors ScannerMetadata for convenience).
    field_strength_t: float | None = None
    #: Larmor / center frequency in Hz, as reported by the scanner.
    larmor_hz: float | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def gamma_hz_per_t(self) -> float | None:
        """Gyromagnetic ratio (Hz/T) for :attr:`nucleus`, if known."""
        return gamma_hz_per_t(self.nucleus)

    @property
    def expected_larmor_hz(self) -> float | None:
        """Larmor frequency predicted from nucleus and field strength."""
        g = self.gamma_hz_per_t
        if g is None or self.field_strength_t is None:
            return None
        return abs(g) * self.field_strength_t

    def is_multinuclear(self) -> bool:
        return self.nucleus != "1H"
