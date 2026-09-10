"""Multinuclear metadata: UniMRI carries the nucleus from the start.

X-nuclei imaging (²³Na, ³¹P, ...) is a primary motivation -- it is where
Cartesian-only tools fall down, and it costs almost nothing at the data-model
level.
"""

from __future__ import annotations

from unimri.data import GYROMAGNETIC_RATIO_MHZ_PER_T, AcquisitionInfo, gamma_hz_per_t


def main() -> None:
    print("Gyromagnetic ratios (MHz/T):")
    for nucleus, mhz in GYROMAGNETIC_RATIO_MHZ_PER_T.items():
        print(f"  {nucleus:5s} {mhz:+10.5f}")
    print()

    b0 = 7.0
    for nucleus in ("1H", "23Na", "31P"):
        acq = AcquisitionInfo(nucleus=nucleus, field_strength_t=b0)
        g = gamma_hz_per_t(nucleus)
        print(
            f"{nucleus:5s} at {b0} T: gamma = {g / 1e6:8.4f} MHz/T, "
            f"Larmor ~ {acq.expected_larmor_hz / 1e6:8.3f} MHz, "
            f"multinuclear = {acq.is_multinuclear()}"
        )


if __name__ == "__main__":
    main()
