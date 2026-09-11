"""CG-SENSE: reconstruction as an inverse problem, y = F S x.

The same solver code reconstructs Cartesian and non-Cartesian data -- only the
encoding operator ``F`` changes (``FourierOperator`` vs. ``NUFFTOperator``).
Compares against the single-pass ``"adjoint"`` gridding reconstruction.
"""

from __future__ import annotations

import numpy as np

import unimri
from unimri.calibration import estimate_sensitivity
from unimri.data import SamplingPattern
from unimri.operators import FourierOperator, NUFFTOperator, SensitivityOperator, unchecked
from unimri.optimization import conjugate_gradient
from unimri.reconstruction import image_shape_of
from unimri.testing import synthetic_dataset


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    r = np.abs(recon).ravel() / (np.abs(recon).max() + 1e-12)
    t = np.abs(truth).ravel() / (np.abs(truth).max() + 1e-12)
    return float(np.linalg.norm(r - t) / (np.linalg.norm(t) + 1e-12))


def main() -> None:
    # An undersampled, noisy radial acquisition -- where CG-SENSE should
    # clearly beat single-pass gridding.
    ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=32, n_coils=6, noise_std=0.02, seed=1)
    data = ds.data
    print(data.summary())

    # -- one call -----------------------------------------------------------
    img_adjoint = unimri.reconstruct(data, method="adjoint")
    img_cg = unimri.reconstruct(data, method="cg", n_iter=20, l2=1e-3)
    print(f"\nadjoint gridding : NRMSE {nrmse(img_adjoint, ds.ground_truth):.3f}")
    print(f"CG-SENSE         : NRMSE {nrmse(img_cg, ds.ground_truth):.3f}")

    # -- or assemble the operator yourself, exactly as `reconstruct` does ---
    # This is the aspirational-API snippet from examples/99, made concrete.
    image_shape = image_shape_of(data)
    maps = estimate_sensitivity(data, method="rss")
    F = (
        NUFFTOperator(data.trajectory, image_shape)
        if not data.is_cartesian
        else FourierOperator(image_shape)
    )
    A = unchecked(F @ SensitivityOperator(maps))

    residuals = []
    y = data.kspace.reshape(data.n_coils, -1).astype(np.complex128)
    img_manual = conjugate_gradient(
        A, y, n_iter=20, l2=1e-3, callback=lambda i, x, r: residuals.append(r)
    )
    print(f"manual assembly  : NRMSE {nrmse(img_manual, ds.ground_truth):.3f}")
    n_iters = len(residuals) - 1
    print(f"residual norm    : {residuals[0]:.4g} -> {residuals[-1]:.4g} over {n_iters} iters")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for a, (title, im) in zip(
        ax, [("ground truth", ds.ground_truth), ("adjoint", img_adjoint), ("CG-SENSE", img_cg)]
    ):
        a.imshow(np.abs(im), cmap="gray")
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig("cg_sense.png", dpi=110)
    print("saved cg_sense.png")


if __name__ == "__main__":
    main()
