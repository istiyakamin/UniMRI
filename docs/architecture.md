# Architecture

UniMRI is organised into seven layers. Each depends only on the layers below it.
The design goal is that a reconstruction algorithm never contains a
`if vendor == ...` or `if trajectory == ...` branch — those decisions are
resolved once, when the data is loaded and when operators are assembled.

```text
┌─────────────────────────────────────────────────────────┐
│ 7  User API        unimri.read() / unimri.reconstruct()  │
│                    Python API · CLI · notebooks           │
├─────────────────────────────────────────────────────────┤
│ 6  Pipeline        ordered, reproducible Stage chains     │
├─────────────────────────────────────────────────────────┤
│ 5  Reconstruction  fft · sense · grappa · cg · cs · …     │
├─────────────────────────────────────────────────────────┤
│ 4  Operators       LinearOperator algebra: A, Aᴴ, AᴴA     │
│                    Fourier · NUFFT · Sampling · Coil       │
├─────────────────────────────────────────────────────────┤
│ 3  Calibration &   sensitivity maps · GRAPPA kernels ·    │
│    preprocessing   DCF · coil compression · noise         │
├─────────────────────────────────────────────────────────┤
│ 2  Data model      MRIData: kspace · trajectory ·         │
│                    encoding · coils · metadata · provenance│
├─────────────────────────────────────────────────────────┤
│ 1  I/O             Reader registry: Siemens · GE ·        │
│                    Philips · ISMRMRD · HDF5               │
└─────────────────────────────────────────────────────────┘
```

## Layer 1 — I/O

A `Reader` maps one on-disk file to one `MRIData`. Readers register themselves
and are selected by sniffing (`Reader.can_read`), so `unimri.read("scan.dat")`
just works. Vendor parsing is delegated to existing libraries
(`twixtools`, `ismrmrd`, `h5py`); UniMRI only owns the *mapping* onto the data
model. See [I/O and formats](io-format-support.md).

## Layer 2 — Data model

`MRIData` is the pivot of the framework and the part most worth getting right.
It is a plain dataclass; there is no hidden lazy-loading or vendor state. Its
full schema and — importantly — its axis and unit conventions are in
[data-model.md](data-model.md).

Reproducibility is built in: `MRIData.provenance` is an ordered log, and every
reader and transform appends a step to it.

## Layer 3 — Calibration & preprocessing

Vendor-neutral operations on `MRIData`: oversampling removal, noise
pre-whitening, ramp-sampling regridding, gradient-delay / trajectory correction,
coil compression, and the estimation problems — coil sensitivities (ESPIRiT,
adaptive combine), GRAPPA kernels, density-compensation functions. These wrap
`sigpy.mri` / BART where a good implementation already exists.

## Layer 4 — Operators

Reconstruction is treated as the inverse problem

$$
y = A x, \qquad A = P\,F\,S
$$

| symbol | meaning | UniMRI operator |
| --- | --- | --- |
| $x$ | the image we want | — |
| $S$ | coil sensitivities | `SensitivityOperator` |
| $F$ | Fourier encoding (uniform **or** non-uniform) | `FourierOperator` / `NUFFTOperator` |
| $P$ | k-space sampling / masking | `SamplingOperator` |
| $y$ | measured k-space | `MRIData.kspace` |

Every block is a `LinearOperator` exposing `forward` ($A$), `adjoint` ($A^H$),
and `normal` ($A^H A$). Operators compose with `@` and scale with `*`:

```python
A = SamplingOperator(...) @ FourierOperator(...) @ SensitivityOperator(...)
```

so a whole encoding model is one expression, and a solver only needs
`forward` / `adjoint` / `normal`. It is oblivious to whether $F$ was an FFT or a
NUFFT — which is exactly how the same CG-SENSE code reconstructs Cartesian and
radial data.

The algebraic core (`LinearOperator`, `IdentityOperator`, `ScaledOperator`,
`CompositeOperator`, the adjoint wrapper, and an adjoint **dot-test**) is
implemented today. Physical operators are stubs — see the
[roadmap](roadmap.md).

### The dot-test

Every operator must pass `op.dot_test()`, which checks
$\langle A x, y\rangle = \langle x, A^H y\rangle$ on random complex inputs. A
wrong adjoint silently corrupts every iterative reconstruction, so this is a
required part of the operator contract, enforced in the test suite.

## Layer 5 — Reconstruction

Named methods (`"fft"`, `"sense"`, `"cg"`, `"cs"`, …) dispatched from
`unimri.reconstruct(data, method=...)`. Each is a thin function that assembles
operators (layer 4) from calibration outputs (layer 3) and, if iterative, hands
them to a solver.

## Layer 6 — Pipeline

`Pipeline` is an ordered list of `Stage` objects threaded over one `MRIData`.
Each stage records provenance, so a pipeline run is fully auditable and
reproducible. Interface defined today; concrete stages arrive with layers 3–5.

## Layer 7 — User API

- `unimri.read(path)` → `MRIData`
- `unimri.reconstruct(data, method="fft", device="cuda")` → image
- the operator / optimization API for algorithm developers
- later: a small CLI and notebook helpers

## Backend abstraction

Core code targets the [Python array API](https://data-apis.org/array-api/):
operator and reconstruction code asks an array for its namespace
(`unimri.backend.get_namespace`) rather than importing NumPy. The same operator
then runs on `numpy`, `cupy`, or `torch` arrays. NumPy is the only required
runtime dependency; GPU and PyTorch are optional extras. UniMRI does not aim to
be a full array-compat shim — if that is needed, it will depend on
`array-api-compat` rather than grow one.

## Non-goals

- **Not** a DICOM / image-domain toolkit — UniMRI starts at k-space.
- **Not** a new raw-data file format — it interoperates with ISMRMRD.
- **Not** a from-scratch NUFFT or optimisation library — those are wrapped.
- **Not** (initially) a deep-learning reconstruction framework, though the
  operator layer is designed to make unrolled networks straightforward later.
