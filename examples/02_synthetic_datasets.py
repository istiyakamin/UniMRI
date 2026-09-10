"""Generate one valid dataset per sampling pattern, with known ground truth.

`unimri.testing.synthetic_dataset` simulates multi-coil k-space from an analytic
phantom via the reference NDFT, so the exact image is available for scoring a
reconstruction.
"""

from __future__ import annotations

from unimri.testing import AVAILABLE_PATTERNS, synthetic_dataset


def main() -> None:
    for pattern in AVAILABLE_PATTERNS:
        ds = synthetic_dataset(pattern, matrix=24, n_coils=4, noise_std=0.0)
        d = ds.data
        print(f"=== {pattern.value} ===")
        print(d.summary())
        print(f"  ground truth : {ds.ground_truth.shape}  (exact image)")
        if ds.coil_maps is not None:
            print(f"  coil maps    : {ds.coil_maps.shape}")
        if d.trajectory is not None:
            k = d.trajectory.coords
            print(f"  k range      : [{k.min():.1f}, {k.max():.1f}]")
        print()


if __name__ == "__main__":
    main()
