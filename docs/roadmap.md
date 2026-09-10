# Roadmap

Milestone numbers are ordering, not dates. UniMRI is a community project; pace
depends on contributors.

## Guiding principle: design against real data

The interfaces in this scaffold (`MRIData`, `Reader`, `LinearOperator`) are a
*starting point*, not a frozen contract. Only after several real formats and
several real reconstructions work should the interfaces be declared stable (the
1.0 line).

The rule is: **build against real data, not ahead of it.** The `NUFFTOperator`
and `reconstruct(method="adjoint")` were built early (out of milestone order)
because real 3-D density-adapted radial ²³Na data was on hand to validate them
against — and doing so already pinned down the trajectory axis convention. The
`Trajectory` and `EncodingSpace` shapes may still shift as the vendor readers
land.

## Milestone 0 — Scaffold ✅ (this commit)

- Packaging, CI, contributor docs.
- Design spec: `architecture.md`, `data-model.md`, `io-format-support.md`.
- Interfaces: `MRIData` + components with `validate()`; `Reader` + registry;
  `LinearOperator` algebra with adjoint dot-test.
- Contract test suite (data model, operator algebra, reader registry).

## Milestone 1 — Core + first real read → `0.1.0`

- ✅ `unimri.testing` — synthetic phantoms, trajectories, reference NDFT,
  `synthetic_dataset(pattern)`.
- ✅ `reconstruct(method="adjoint")` — centered inverse FFT (Cartesian) and
  density-compensated NUFFT gridding (non-Cartesian), RSS coil combine.
- `ISMRMRDReader.read` — real, wrapping the `ismrmrd` package.
- `HDF5Reader` + `MRIData.to_hdf5()` round-trip (test fixtures, caching).
- `FourierOperator` as a proper `LinearOperator` (the current recon uses a bare
  centered FFT inline).
- End-to-end test: ISMRMRD file → image.
- **Revisit the data model** based on what ISMRMRD actually carries.

## Milestone 2 — Siemens TWIX → `0.1.x`

- `TwixReader.read` wrapping `twixtools`; map the 16-D array to `MRIData`.
- Oversampling removal, noise pre-whitening, ramp-sampling regrid as
  preprocessing stages.
- Trajectory extraction for radial / stack-of-stars sequences.

## Milestone 3 — Cartesian reconstruction basics → `0.2.0`

- `SamplingOperator` (mask / zero-fill), partial-Fourier (POCS / homodyne).
- Cropping, apodisation, zero-padding utilities.
- `Pipeline` stages for the above; provenance end-to-end.

## Milestone 4 — Parallel imaging → `0.3.0`

- `SensitivityOperator`, `CoilCompressionOperator`.
- Calibration: ESPIRiT and adaptive/Walsh combine (wrapping `sigpy.mri` / BART
  where available).
- `reconstruct(method="sense")`, `method="grappa")`.

## Milestone 5 — Non-Cartesian → `0.4.0`

- ✅ `NUFFTOperator` — FINUFFT-backed, 2-D and 3-D, passes the adjoint dot-test
  and matches the reference NDFT. Verified on real 3-D density-adapted radial
  (DA-3DPR) ²³Na data (`examples/sodium_radial.py`).
- ✅ `reconstruct(method="adjoint")` — density-compensated gridding.
- Swap/extend the backend to `mri-nufft` (GPU, more kernels) behind the same
  interface.
- Density compensation: analytic and Pipe–Menon iterative (currently the DCF
  must be supplied on the `Trajectory`).
- Broader trajectory coverage: 2-D radial ✅, stack-of-stars ✅, 3-D radial ✅,
  spiral, cones.
- GIRF / gradient-delay correction hook.

## Milestone 6 — Iterative reconstruction → `0.5.0`

- `unimri.optimization`: conjugate gradient, FISTA / proximal gradient, ADMM.
- Regularizers: L1, total variation, wavelet, locally-low-rank.
- `reconstruct(method="cg")` and `method="cs")` — trajectory-agnostic, so they
  work on Cartesian and non-Cartesian data unchanged.

## Milestone 7 — Multi-vendor → `0.6.0`

- GE and Philips readers.
- Broader ISMRMRD coverage and converter interop.
- This is where "unified" earns its name — the same script over three vendors.

## Milestone 8 — GPU → `0.7.0`

- Verified CuPy and PyTorch operator paths.
- Device-aware `reconstruct(..., device="cuda")`.
- Benchmark harness (runtime, memory, VRAM).

## Milestone 9 — Benchmarking & validation → `0.8.0`

- Reference datasets and expected outputs; regression tests on image values.
- Metrics: NRMSE, PSNR, SSIM, g-factor.
- Cross-check against BART / sigpy on the same inputs.

## Milestone 10 — `1.0.0`

Stable API · multi-vendor · documented data model · validated reconstruction ·
CPU + GPU · comprehensive tests · benchmarks · scientific examples · a software
paper (JOSS-style).

## Explicitly deferred

Deep-learning reconstruction (MoDL, variational networks, diffusion). The
operator layer is being designed so unrolled networks are natural to add later,
but no ML lands before 1.0.
