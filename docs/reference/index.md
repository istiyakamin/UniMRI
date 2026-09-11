# API reference

Auto-generated from docstrings. UniMRI is pre-alpha; anything not listed here is
either an internal detail or a not-yet-implemented stub (see the
[roadmap](../roadmap.md)).

- [Data model](data-model.md) — `MRIData` and its components
- [I/O](io.md) — the reader interface and registry
- [Operators](operators.md) — the linear-operator algebra, `FourierOperator`, `NUFFTOperator`, `SensitivityOperator`
- [Calibration](calibration.md) — coil sensitivity estimation
- [Optimization](optimization.md) — the conjugate-gradient solver
- [Reconstruction](reconstruction.md) — `adjoint` and `cg` (CG-SENSE)
- [Testing utilities](testing.md) — synthetic data and reference transforms

::: unimri
    options:
      members: ["read", "reconstruct", "__version__"]
      show_root_heading: false
