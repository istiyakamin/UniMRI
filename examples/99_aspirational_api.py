"""UniMRI quickstart -- ASPIRATIONAL.

This script shows the *intended* API. It does NOT run yet: UniMRI is pre-alpha
and no readers or reconstruction methods are implemented. See docs/roadmap.md.
"""

from __future__ import annotations

import unimri

# ---------------------------------------------------------------------------
# 1. Read raw data -- format detected automatically (Siemens / ISMRMRD / HDF5).
# ---------------------------------------------------------------------------
raw = unimri.read("measurement.dat")
print(raw.summary())

# ---------------------------------------------------------------------------
# 2. Simple reconstruction -- one call, trajectory-agnostic.
# ---------------------------------------------------------------------------
image = unimri.reconstruct(raw, method="fft")

# ---------------------------------------------------------------------------
# 3. Advanced: reconstruction as an inverse problem  y = P F S x
#    The same expression works for Cartesian and non-Cartesian data;
#    only the F operator differs.
# ---------------------------------------------------------------------------
# from unimri.calibration import estimate_sensitivity, density_compensation
# from unimri.operators import FourierOperator, NUFFTOperator, SensitivityOperator
# from unimri.optimization import conjugate_gradient
#
# maps = estimate_sensitivity(raw, method="espirit")
# F = (
#     NUFFTOperator(raw.trajectory, raw.encoding)
#     if not raw.is_cartesian
#     else FourierOperator(raw.encoding)
# )
# A = F @ SensitivityOperator(maps)
# image = conjugate_gradient(A, raw.kspace, n_iter=30)

# ---------------------------------------------------------------------------
# 4. Everything that happened is recorded.
# ---------------------------------------------------------------------------
print(raw.provenance)
