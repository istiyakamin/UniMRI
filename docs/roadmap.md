# Roadmap

Milestone numbers are ordering, not dates. UniMRI is a community project; pace
depends on contributors.

## Guiding principle: design against real data

The interfaces in this scaffold (`MRIData`, `Reader`, `LinearOperator`) are a
*starting point*, not a frozen contract. The single most important early task is
**Milestone 1**: get one real ISMRMRD dataset to reconstruct end-to-end with a
plain FFT. That exercise is expected to reshape the data model. Only after two
or three real formats and two or three real reconstructions work should the
interfaces be declared stable (the 1.0 line).

Do not build layers 3–6 broadly before layer 1–2 works on real data.

## Milestone 0 — Scaffold ✅ (this commit)

- Packaging, CI, contributor docs.
- Design spec: `architecture.md`, `data-model.md`, `io-format-support.md`.
- Interfaces: `MRIData` + components with `validate()`; `Reader` + registry;
  `LinearOperator` algebra with adjoint dot-test.
- Contract test suite (data model, operator algebra, reader registry).

## Milestone 1 — Core + first real read → `0.1.0`

- `ISMRMRDReader.read` — real, wrapping the `ismrmrd` package.
- `HDF5Reader` + `MRIData.to_hdf5()` round-trip (test fixtures, caching).
- `FourierOperator` (centered n-D FFT, array-API backed).
- `reconstruct(method="fft")` — inverse FFT + root-sum-of-squares coil combine.
- Synthetic phantom fixtures; end-to-end test: ISMRMRD file → image.
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

- `NUFFTOperator` adapter over `mri-nufft` (and/or `torchkbnufft`, `sigpy`).
- Density compensation: analytic, Voronoi, Pipe–Menon iterative.
- `reconstruct(method="adjoint")` — density-compensated gridding.
- First target trajectory: **3D density-adapted radial** (the sodium / X-nuclei
  workhorse). Then 2D radial, stack-of-stars, spiral.
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
