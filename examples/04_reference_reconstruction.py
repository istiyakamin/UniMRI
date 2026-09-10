"""Reconstruct a synthetic radial dataset two ways and compare.

- `unimri.reconstruct(data, method="adjoint")` -- the real path: FINUFFT-backed
  density-compensated gridding (needs `pip install "unimri[nufft]"`).
- `unimri.testing.ndft_adjoint` -- the slow, exact reference the operator is
  validated against.

Both are scored against the known ground truth.
"""

from __future__ import annotations

import numpy as np

import unimri
from unimri.data import SamplingPattern
from unimri.testing import ndft_adjoint, synthetic_dataset


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    r = np.abs(recon).ravel() / (np.abs(recon).max() + 1e-12)
    t = np.abs(truth).ravel() / (np.abs(truth).max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def main() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=32, n_coils=4, noise_std=0.01, seed=1)
    data = ds.data
    print(data.summary())
    print()

    try:
        fast = unimri.reconstruct(data, method="adjoint")
        print(f"NUFFT gridding   : {fast.shape}  NRMSE {nrmse(fast, ds.ground_truth):.3f}")
    except Exception as exc:  # noqa: BLE001 - finufft may be missing
        fast = None
        print(f"NUFFT gridding   : skipped ({exc})")

    traj = data.trajectory
    ny, nx, _ = data.encoding.recon_matrix
    ref = np.sqrt(
        sum(
            np.abs(
                ndft_adjoint(data.kspace[c], traj.coords, (ny, nx), dcf=traj.density_compensation)
            )
            ** 2
            for c in range(data.n_coils)
        )
    )
    print(f"reference NDFT   : {ref.shape}  NRMSE {nrmse(ref, ds.ground_truth):.3f}")
    print("(crude single-pass gridding of a hard phantom; iterative recon comes later)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    imgs = [("ground truth", np.abs(ds.ground_truth)), ("reference NDFT", ref)]
    if fast is not None:
        imgs.insert(1, ("NUFFT gridding", np.abs(fast)))
    fig, ax = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3))
    for a, (title, im) in zip(ax, imgs):
        a.imshow(im, cmap="gray")
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig("radial_recon.png", dpi=110)
    print("saved radial_recon.png")


if __name__ == "__main__":
    main()
