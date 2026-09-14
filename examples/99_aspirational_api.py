"""UniMRI quickstart -- partly ASPIRATIONAL.

This script shows the *intended* top-level API for a Siemens `.dat` file.
The reconstruction half works today (see examples/04, examples/07); reading
Cartesian *ISMRMRD* also works today (see examples/08_ismrmrd_cartesian.py).
What's still missing is `unimri.read` for Siemens TWIX specifically -- see
docs/roadmap.md, Milestone 2.
"""

from __future__ import annotations

import unimri

# ---------------------------------------------------------------------------
# 1. Read raw data -- format detected automatically (Siemens / ISMRMRD / HDF5).
#    NOT YET IMPLEMENTED: TwixReader.read is a stub (docs/roadmap.md, M2).
# ---------------------------------------------------------------------------
raw = unimri.read("measurement.dat")
print(raw.summary())

# ---------------------------------------------------------------------------
# 2. Simple reconstruction -- one call, trajectory-agnostic. WORKS TODAY.
# ---------------------------------------------------------------------------
image = unimri.reconstruct(raw, method="adjoint")

# ---------------------------------------------------------------------------
# 3. CG-SENSE: reconstruction as an inverse problem, y = F S x. WORKS TODAY --
#    see examples/07_cg_sense.py for a version you can actually run, and
#    examples/sodium_radial.py for it on real 3-D radial sodium data.
# ---------------------------------------------------------------------------
image = unimri.reconstruct(raw, method="cg", n_iter=20, l2=1e-4)

# Or assemble the operator yourself:
# from unimri.calibration import estimate_sensitivity
# from unimri.operators import FourierOperator, NUFFTOperator, SensitivityOperator, unchecked
# from unimri.optimization import conjugate_gradient
#
# maps = estimate_sensitivity(raw)
# F = (
#     NUFFTOperator(raw.trajectory, image_shape)
#     if not raw.is_cartesian
#     else FourierOperator(image_shape)
# )
# A = unchecked(F @ SensitivityOperator(maps))
# image = conjugate_gradient(A, raw.kspace, n_iter=20, l2=1e-4)

# ---------------------------------------------------------------------------
# 4. Everything that happened is recorded.
# ---------------------------------------------------------------------------
print(raw.provenance)
