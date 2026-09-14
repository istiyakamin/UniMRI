# Roadmap

Milestone numbers are ordering, not dates. UniMRI is a community project; pace
depends on contributors.

## Guiding principle: design against real data

The interfaces in this scaffold (`MRIData`, `Reader`, `LinearOperator`) are a
*starting point*, not a frozen contract. Only after several real formats and
several real reconstructions work should the interfaces be declared stable (the
1.0 line).

The rule is: **build against real data, not ahead of it.** The `NUFFTOperator`,
`FourierOperator`, `SensitivityOperator`, `reconstruct(method="adjoint"|"cg")`,
and `conjugate_gradient` were built early (out of milestone order) because real
3-D density-adapted radial ²³Na data was on hand to validate them against — and
doing so already caught two real bugs: an inconsistent trajectory axis
convention, and `FourierOperator` using a different amplitude normalization
than `NUFFTOperator`/the reference NDFT (both fixed; see `docs/data-model.md`
and `unimri.operators.fourier`). The
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
- ✅ `FourierOperator` as a proper `LinearOperator` (also used by
  `reconstruct(method="cg")`).
- ✅ `ISMRMRDReader.read` — real, wrapping the `ismrmrd` package. Scope:
  single Cartesian encoding space, one slice/average/contrast/repetition/
  set/segment (see `docs/io-format-support.md`). Non-Cartesian ISMRMRD and
  multi-dimensional acquisitions raise a clear `ReaderError`, not silently
  mishandled data.
- ✅ End-to-end test: write a real ISMRMRD file with the `ismrmrd` package,
  read it with `ISMRMRDReader`, reconstruct, match ground truth
  (`tests/test_io_ismrmrd.py`).
- `HDF5Reader` + `MRIData.to_hdf5()` round-trip (test fixtures, caching).
- **Revisit the data model** based on what ISMRMRD actually carries: so far
  no changes were needed -- `MRIData`'s Cartesian axis/encoding shapes matched
  ISMRMRD's `encodedSpace`/`reconSpace`/`encodingLimits` directly.

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

- ✅ `SensitivityOperator` — composable coil operator (`Fᴴ` maps, `Fᴴ` combine),
  exact `_normal`.
- ✅ `estimate_sensitivity(method="rss")` — RSS-normalized baseline (crude but
  dependency-free).
- `CoilCompressionOperator`.
- Calibration: ESPIRiT and adaptive/Walsh combine (wrapping `sigpy.mri` / BART
  where available) -- the `"rss"` baseline is not a substitute for these.
- `reconstruct(method="sense")` (direct, non-iterative), `method="grappa")`.

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

- ✅ `unimri.optimization.conjugate_gradient` — solves `(AᴴA + l2·I)x = Aᴴy`
  against any `LinearOperator`; L2 (Tikhonov) regularization built in.
- ✅ `reconstruct(method="cg")` — CG-SENSE (`A = F @ SensitivityOperator(maps)`).
  The **same code** reconstructs Cartesian and non-Cartesian data (only `F`
  changes), and it measurably beats `"adjoint"` on undersampled/noisy radial
  data (synthetic and real ²³Na, see `examples/07_cg_sense.py`).
- FISTA / proximal gradient, ADMM.
- Regularizers beyond L2: L1, total variation, wavelet, locally-low-rank.
- `reconstruct(method="cs")` (compressed sensing).
- Better coil sensitivities (ESPIRiT, Milestone 4) and a DCF-based
  preconditioner should improve `"cg"` convergence and quality further; the
  current `"rss"` sensitivity + unpreconditioned CG is a correct but basic
  starting point.

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
