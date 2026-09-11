r"""Reconstruct real 3-D density-adapted radial (DA-3DPR) sodium data.

This example needs local data that is **not** in the repository -- a
patient/phantom ``.dat`` is proprietary. It works from k-space and trajectory
that have already been extracted to MATLAB files (e.g. by a mapVBVD / gridding
pipeline). Point it at yours::

    python examples/sodium_radial.py \
        --kspace  path/to/data_unaveraged.mat \   # (n_coils, n_spokes, n_readout)
        --traj    path/to/trajectory.mat \        # kvec (3, n_spokes*n_readout), dk (n_readout,)
        --matrix  96

It is not run in CI. It shows that UniMRI's data model + NUFFT reconstruction
handle a real non-Cartesian multinuclear acquisition end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import unimri
from unimri.data import (
    AcquisitionInfo,
    CoilInfo,
    EncodingSpace,
    FieldOfView,
    MRIData,
    SamplingPattern,
    Trajectory,
    TrajectoryUnits,
)

DEFAULT_BASE = Path(r"C:\Users\isanto\Desktop\ResearchWork\SSDU_patients")


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    """Read one variable from a .mat file (v7.3/HDF5 or older v5)."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            a = np.asarray(f[key])
        if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
            a = a["real"] + 1j * a["imag"]
        return a
    except (OSError, KeyError):
        import scipy.io as sio

        return np.asarray(sio.loadmat(path, squeeze_me=True)[key])


def build_mri_data(kspace_mat: Path, traj_mat: Path, matrix: int) -> MRIData:
    kspace = _load_mat_array(kspace_mat, "data_unaveraged")  # (n_coils, n_spokes, n_readout)
    kvec = _load_mat_array(traj_mat, "kvec").astype(np.float64)  # (3, n_spokes*n_readout)
    dk = _load_mat_array(traj_mat, "dk").astype(np.float64).ravel()  # (n_readout,)

    n_coils, n_spokes, n_readout = kspace.shape
    kvec = kvec.reshape(3, n_spokes, n_readout)

    # MATLAB kvec rows are (kx, ky, kz); UniMRI wants (kz, ky, kx) = image axis order.
    coords = kvec[::-1]
    # Scale so the outermost sampled point sits at the Nyquist edge (+-matrix/2).
    rmax = np.sqrt((coords**2).sum(0)).max()
    coords = coords / rmax * (matrix / 2.0)
    dcf = np.broadcast_to(dk / dk.max(), (n_spokes, n_readout)).copy()

    data = MRIData(
        kspace=kspace.astype(np.complex64),
        kspace_axes=("coil", "shot", "readout"),
        encoding=EncodingSpace(
            recon_matrix=(matrix, matrix, matrix),
            encoded_matrix=(matrix, matrix, matrix),
            fov=FieldOfView(220.0, 220.0, 220.0),
            sampling=SamplingPattern.RADIAL,
            n_dims=3,
        ),
        acquisition=AcquisitionInfo(sequence_name="DA-3DPR", nucleus="23Na", field_strength_t=7.0),
        coils=CoilInfo(n_channels=n_coils),
        trajectory=Trajectory(coords, TrajectoryUnits.NORMALIZED, density_compensation=dcf),
    )
    data.provenance.record("load_da3dpr_mat", params={"kspace": kspace_mat.name})
    return data.validate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ks_default = DEFAULT_BASE / "matlab_data_unaveraged" / "matlab_data_unaveraged_SSDU_01.mat"
    tr_default = DEFAULT_BASE / "Trajectory" / "trajectory_SSDU_01.mat"
    ap.add_argument("--kspace", type=Path, default=ks_default)
    ap.add_argument("--traj", type=Path, default=tr_default)
    ap.add_argument("--matrix", type=int, default=96)
    ap.add_argument(
        "--max-spokes",
        type=int,
        default=0,
        help="subsample projections for a quick preview (0 = use all)",
    )
    ap.add_argument(
        "--method",
        choices=["adjoint", "cg"],
        default="adjoint",
        help="'adjoint' = single-pass gridding (fast); 'cg' = CG-SENSE (slower, sharper)",
    )
    ap.add_argument("--n-iter", type=int, default=8, help="CG iterations (--method cg)")
    ap.add_argument("--l2", type=float, default=1e-2, help="CG L2 regularization (--method cg)")
    args = ap.parse_args()

    if not args.kspace.exists() or not args.traj.exists():
        print("Data not found. Pass --kspace and --traj (see the module docstring).")
        print(f"  looked for: {args.kspace}\n              {args.traj}")
        sys.exit(0)

    data = build_mri_data(args.kspace, args.traj, args.matrix)
    if args.max_spokes:
        s = slice(0, args.max_spokes)
        data.kspace = data.kspace[:, s]
        data.trajectory.coords = data.trajectory.coords[:, s]
        data.trajectory.density_compensation = data.trajectory.density_compensation[s]
        data.validate()
    print(data.summary())
    print(f"\nexpected Larmor at 7 T: {data.acquisition.expected_larmor_hz / 1e6:.2f} MHz")

    if args.method == "adjoint":
        print("\nreconstructing (density-compensated NUFFT gridding)...")
        img = unimri.reconstruct(data, method="adjoint")  # (nz, ny, nx)
    else:
        print(f"\nreconstructing (CG-SENSE, {args.n_iter} iterations)...")
        img = unimri.reconstruct(data, method="cg", n_iter=args.n_iter, l2=args.l2)
    print(f"image: {img.shape}  dtype {img.dtype}")

    stem = f"sodium_da3dpr_{args.method}"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        np.save(f"{stem}.npy", img)
        print(f"saved {stem}.npy")
        return
    nz = img.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, z in zip(ax, (nz // 3, nz // 2, 2 * nz // 3)):
        a.imshow(np.abs(img[z]), cmap="gray")
        a.set_title(f"z = {z}")
        a.axis("off")
    fig.suptitle(f"23Na DA-3DPR, {args.method} reconstruction")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    print(f"saved {stem}.png")


if __name__ == "__main__":
    main()
