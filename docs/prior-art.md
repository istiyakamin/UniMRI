# Prior art

UniMRI overlaps with several mature projects. This page is an honest account of
that overlap, what UniMRI should **depend on** rather than reimplement, and
where a distinct niche plausibly exists. (Project capabilities described here
were current in early 2026 — verify before quoting.)

## The closest projects

### MRIReco.jl

A Julia framework with essentially the same thesis as UniMRI: a unified raw-data
type, multiple file-format readers (ISMRMRD, Siemens, Philips, Bruker,
GE via converters), a linear-operator model, and Cartesian + non-Cartesian
iterative reconstruction. It is the strongest evidence that the "unified
representation + operator model" idea works. The gap UniMRI addresses is
**language**: the Python MRI ecosystem (research code, DL tooling, teaching) is
much larger, and there is no single Python project that occupies MRIReco.jl's
position.

### mrpro

A modern **PyTorch** reconstruction package (PTB and collaborators, Apache-2.0).
Reads ISMRMRD and Siemens raw data, has a `KData` container, a `KTrajectory`
type, an operator framework (Fourier, sensitivity, density compensation, …), and
solvers including CG-SENSE. This is the most direct overlap.

Differences UniMRI is betting on:

- **Backend-agnostic core, not PyTorch-first.** mrpro's design centre is
  autograd tensors; UniMRI targets the array API so the core runs on plain
  NumPy with no deep-learning stack, and CuPy / PyTorch are optional.
- **I/O breadth as a first-class goal.** UniMRI treats the reader registry and
  multi-vendor mapping as the primary product; reconstruction methods are
  replaceable parts.
- **Provenance / reproducibility as a built-in**, not an add-on.

If mrpro already solves your problem, use mrpro. UniMRI's contribution only
matters if the backend-neutrality and I/O breadth are things you need.

## Components UniMRI wraps (does not reimplement)

| Need | UniMRI depends on |
| --- | --- |
| NUFFT | **`mri-nufft`** — a unified Python interface over finufft, cufinufft, gpuNUFFT, torchkbnufft, sigpy, … plus trajectory tools. UniMRI's `NUFFTOperator` is an adapter over this. |
| Cartesian PI / CS reference | **`sigpy.mri`** (ESPIRiT, App-based CG / L1-wavelet), optionally **BART** |
| Siemens raw parsing | **`twixtools`** (also used by RecoTwix) |
| ISMRMRD | **`ismrmrd`** Python package |
| Coil combination (adaptive) | `sigpy` / BART |
| Array-API compat, if needed | `array-api-compat` |

The rule (see `CONTRIBUTING.md`): if a good implementation exists, wrap it
behind a UniMRI interface.

## Adjacent projects, different scope

| Project | What it is | Relation to UniMRI |
| --- | --- | --- |
| **ISMRMRD + Gadgetron** | The standard raw-data format and a C++ streaming reconstruction framework with vendor converters | UniMRI interoperates with the format; it is a Python, in-process alternative to the Gadgetron pipeline model |
| **BART** | Reference C toolbox for calibration and iterative recon, CFL data format | A backend UniMRI can call; not a Python data model |
| **sigpy** | NumPy/CuPy signal-processing + `sigpy.mri` | A dependency and a design influence (its `Linop`); no vendor I/O or data model |
| **fastMRI** | Cartesian knee/brain datasets, subsampling utilities, PyTorch DL baselines | Data + benchmarks, not a general framework; UniMRI can consume fastMRI data |
| **PyQMRI** | Quantitative MRI fitting from k-space / image data, OpenCL | Focused on parameter mapping; UniMRI is the layer beneath that |
| **direct** | Deep-learning reconstruction (unrolled networks) | Model zoo; UniMRI defers DL and would sit below such a project |
| **torchkbnufft / PyNUFFT / gpuNUFFT** | Individual NUFFT implementations | Backends reached via `mri-nufft` |
| **twixtools / pymapVBVD** | Siemens `.dat` readers | Parsing dependencies |

## Where UniMRI's niche is

Stated plainly, UniMRI is worth building if — and only if — the following
combination is not available elsewhere in Python:

1. A documented, backend-neutral **unified raw-data model** that a plain-NumPy
   user and a GPU/PyTorch user share.
2. A **reader registry** designed for many vendor formats, with the mapping
   onto the data model as the maintained core.
3. An **operator + pipeline** layer where the same reconstruction code runs on
   Cartesian and non-Cartesian data, wrapping best-in-class backends rather
   than reimplementing them.
4. **Reproducibility** (provenance) as a default behaviour.

Positioning diagram:

```text
                         UniMRI
             (unified I/O · data model · pipeline glue)
                            │
     ┌──────────────┬───────┴───────┬──────────────┐
   ISMRMRD      mri-nufft         sigpy           BART
  (format)      (NUFFT)        (PI / CS ref)   (calibration)
```

If, during Milestone 1, it becomes clear that `mrpro` + `mri-nufft` already
cover a contributor's needs, contributing there is the higher-leverage choice
and this document should say so.
