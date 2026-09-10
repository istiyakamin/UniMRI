# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold: packaging, CI, documentation skeleton.
- Design specification (`docs/architecture.md`, `docs/data-model.md`).
- Public interfaces: `MRIData` and its components, `Reader` + reader registry,
  `LinearOperator` with `IdentityOperator` / `ScaledOperator` /
  `CompositeOperator` and an adjoint dot-test.
- Contract test suite for the data model, operator algebra, and reader registry.
- `unimri.testing`: analytic phantoms, trajectory generators (2D/3D radial,
  stack-of-stars, Cartesian), a brute-force reference NDFT, and
  `synthetic_dataset(pattern)` returning one valid `MRIData` per
  `SamplingPattern` with known ground truth.
- `examples/` — six runnable examples (data model, synthetic datasets, operator
  algebra, reference radial reconstruction, custom reader, multinuclear) plus
  the aspirational end-to-end API; `test_examples.py` runs them in CI.

### Not yet implemented
- All vendor readers (`ismrmrd`, `twix`, `hdf5`) — interfaces only.
- All operators beyond the algebraic base (`fourier`, `nufft`, `sampling`,
  `coil`) — docstring stubs only.
- All reconstruction, calibration, preprocessing, optimization, pipeline stages.

## [0.0.0]

- Reserved for the first tagged scaffold commit.
