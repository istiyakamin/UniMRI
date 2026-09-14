# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- `docs/index.md` rewritten: the status banner still said "pre-alpha (v0.0.0),
  no reconstruction backends implemented" -- badly stale. Now accurate, plus
  a real quickstart (ISMRMRD read+reconstruct, synthetic data, building
  `MRIData` by hand) instead of only a "why" pitch.
- `docs/reference/calibration.md` and `optimization.md` were bare
  mkdocstrings stubs with no surrounding context; added the same kind of
  intro prose + architecture cross-links the other reference pages have.
- `docs/reference/io.md` now documents `ISMRMRDReader` itself, not just the
  registry functions; added a class docstring to `ISMRMRDReader` so it
  renders (module docstrings aren't picked up by a class-level
  `::: unimri.io.ISMRMRDReader` directive).

## [0.1.0a1] - 2026-09-14

Milestone 1: `unimri.read()` reads real data for the first time.

### Added
- `ISMRMRDReader.read` — real, wrapping the `ismrmrd` package. Scope: a
  single Cartesian encoding space with exactly one slice / average /
  contrast / repetition / set / segment. Non-Cartesian ISMRMRD trajectories
  and multi-dimensional acquisitions raise a clear `ReaderError` naming what
  isn't supported, rather than silently mishandling data.
- `tests/test_io_ismrmrd.py` — end-to-end validation: write a real ISMRMRD
  file with the `ismrmrd` package (not UniMRI, which has no writer), read it
  back with `ISMRMRDReader`, reconstruct, and match known ground truth
  exactly (`< 1e-6` scale-invariant NRMSE). Also covers metadata mapping,
  noise-measurement skipping, and the two rejection paths above.
- `examples/08_ismrmrd_cartesian.py` — the same round trip as a runnable
  example, with both `"adjoint"` and `"cg"` reconstruction.
- `ismrmrd` added to the `dev` extra so CI exercises the reader, not just
  `importorskip`-skips it.

## [0.0.1a2] - 2026-09-14

### Fixed
- README images and doc links used repo-relative paths, which resolve on
  GitHub but 404 on PyPI (the long_description page has no repo base path).
  Images now point at raw.githubusercontent.com; doc links point at the
  MkDocs site or the github.com blob view. No code changes; re-released
  solely because PyPI freezes long_description per-version and 0.0.1a1's
  was already broken.

## [0.0.1a1] - 2026-09-14

First alpha release.

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
- `docs/acknowledgments.md` — credit for every library, standard, and method
  UniMRI uses or is built to use, with licenses.
- Project logo, architecture diagram, and README status badges.
- API reference pages (`docs/reference/`, mkdocstrings) in the docs nav.
- `SECURITY.md`, `RELEASING.md`, `.editorconfig`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`.
- Version single-sourced from `src/unimri/__init__.py` via `hatch.version`;
  a test keeps `CITATION.cff` in sync.
- CI now runs on Linux/Windows/macOS, builds and `twine check`s the wheel, and
  installs it in a clean env; a coverage floor of 80% is enforced.
- Workflows: `docs.yml` deploys the MkDocs site to GitHub Pages; `release.yml`
  publishes to PyPI via Trusted Publishing on a `v*` tag.
- `NUFFTOperator` — FINUFFT-backed non-uniform FFT (2-D and 3-D), passes the
  adjoint dot-test and matches the reference NDFT. Optional `nufft` extra.
- `unimri.reconstruct(data, method="adjoint")` — centered inverse FFT for
  Cartesian data, density-compensated NUFFT gridding for non-Cartesian, with
  RSS / sum / no coil combination.
- `examples/sodium_radial.py` — end-to-end reconstruction of real 3-D
  density-adapted radial ²³Na data; `04` now uses `unimri.reconstruct`.
- Trajectory axis convention pinned: `coords` row `d` pairs with image axis `d`
  (`(kz, ky, kx)` for 3-D). `unimri.testing` trajectories and the docs updated.
- `viz` extra (matplotlib) for the plotting examples.
- `FourierOperator` (Cartesian, unnormalized/exact-adjoint convention matching
  `NUFFTOperator` and the reference NDFT) and `SensitivityOperator` (coils),
  both composable via `@`; `UncheckedOperator`/`unchecked()` for batched
  (multi-coil) use through operators declared for a single instance.
- `unimri.calibration.estimate_sensitivity(method="rss")` — RSS-normalized
  coil sensitivity baseline.
- `unimri.optimization.conjugate_gradient` — CG for `(AᴴA + l2·I)x = Aᴴy`.
- `unimri.reconstruct(data, method="cg")` — CG-SENSE, trajectory-agnostic;
  beats `"adjoint"` on undersampled/noisy radial data.
- `examples/07_cg_sense.py`; `99_aspirational_api.py` trimmed to just the
  still-missing reader half.
- Fixed: `FourierOperator` previously used a unitary ("ortho") FFT convention
  inconsistent with `NUFFTOperator`/`ndft`'s unnormalized convention -- caught
  while wiring `reconstruct(method="cg")` to swap between them transparently.
- Fixed: an unanchored `data/` pattern in `.gitignore` matched `src/unimri/data/`
  (the `MRIData` data model) as well as the intended top-level scratch
  directory, so it had never actually been committed -- every CI job and a
  real wheel install failed on import. Anchored the pattern to the repo root.

### Not yet implemented
- All vendor readers (`ismrmrd`, `twix`, `hdf5`) — interfaces only.
- All operators beyond the algebraic base (`fourier`, `nufft`, `sampling`,
  `coil`) — docstring stubs only.
- All reconstruction, calibration, preprocessing, optimization, pipeline stages.

## [0.0.0]

- Reserved for the first tagged scaffold commit.
