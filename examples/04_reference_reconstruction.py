"""Reconstruct a synthetic radial dataset with the reference transform.

`unimri.testing.ndft` is a slow, exact non-uniform DFT. It is not for production
use, but it lets you do a real density-compensated gridding reconstruction today
and score it against the known ground truth -- and later it is the reference a
fast NUFFT operator is checked against.
"""

from __future__ import annotations

import numpy as np

from unimri.data import SamplingPattern
from unimri.testing import ndft_adjoint, synthetic_dataset


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    r = np.abs(recon).ravel()
    t = np.abs(truth).ravel()
    r = r / (r.max() + 1e-12)
    t = t / (t.max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def main() -> None:
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=32, n_coils=4, noise_std=0.01, seed=1)
    data = ds.data
    traj = data.trajectory
    ny, nx, _ = data.encoding.recon_matrix
    print(data.summary())

    # Density-compensated gridding (adjoint) per coil, then root-sum-of-squares.
    coil_imgs = np.stack(
        [
            ndft_adjoint(data.kspace[c], traj.coords, (ny, nx), dcf=traj.density_compensation)
            for c in range(data.n_coils)
        ]
    )
    recon = np.sqrt((np.abs(coil_imgs) ** 2).sum(0))

    print()
    print(f"reconstruction : {recon.shape}")
    print(f"NRMSE vs truth : {nrmse(recon, ds.ground_truth):.3f}")
    print("(crude single-pass gridding of a hard phantom; iterative recon comes later)")

    # Optional: save a side-by-side PNG if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(np.abs(ds.ground_truth), cmap="gray")
    ax[0].set_title("ground truth")
    ax[1].imshow(recon, cmap="gray")
    ax[1].set_title("radial gridding recon")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig("radial_recon.png", dpi=110)
    print("saved radial_recon.png")


if __name__ == "__main__":
    main()
