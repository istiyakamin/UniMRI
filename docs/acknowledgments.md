# Acknowledgments

UniMRI stands on a large body of open-source scientific software. This page
credits every library, standard, and method the project uses or is built to use.
Licenses are noted where known; consult each project for the authoritative terms.

## Runtime dependencies

Libraries imported by the UniMRI core (`pip install unimri`):

| Library | Used for | License |
| --- | --- | --- |
| [NumPy](https://numpy.org) | n-dimensional arrays; the default compute backend | BSD-3-Clause |
| [SciPy](https://scipy.org) | signal processing, linear algebra, spatial transforms | BSD-3-Clause |
| [h5py](https://www.h5py.org) | reading/writing HDF5 (ISMRMRD, UniMRI's cache format) | BSD-3-Clause |

UniMRI targets the [Python Array API standard](https://data-apis.org/array-api/)
so that operators can run on NumPy, CuPy, or PyTorch arrays unchanged.

## Optional integrations

Enabled through extras (`pip install "unimri[...]"`):

| Library | Extra | Used for | License |
| --- | --- | --- | --- |
| [FINUFFT](https://github.com/flatironinstitute/finufft) | `nufft` | the non-uniform FFT behind `NUFFTOperator` and non-Cartesian reconstruction | Apache-2.0 |
| [Matplotlib](https://matplotlib.org) | `viz` | plotting in the examples | Matplotlib (BSD-style) |
| [ISMRMRD (Python)](https://github.com/ismrmrd/ismrmrd-python) | `ismrmrd` | reading ISMRMRD raw data | permissive (see project) |
| [PyTorch](https://pytorch.org) | `torch` | GPU / autograd array backend; unrolled-network reconstruction later | BSD-3-Clause |
| [CuPy](https://cupy.dev) | `gpu` | CUDA array backend | MIT |

## Planned dependencies

These will be wrapped behind UniMRI interfaces as the corresponding milestones
land (see the [roadmap](roadmap.md)); UniMRI does not reimplement them.

| Library | Will be used for | License |
| --- | --- | --- |
| [twixtools](https://github.com/pehses/twixtools) (P. Ehses) | parsing Siemens TWIX `.dat` files | MIT |
| [MRI-NUFFT](https://github.com/mind-inria/mri-nufft) | broader NUFFT backend (GPU, more kernels) behind the same `NUFFTOperator` interface | BSD-3-Clause |
| [cufinufft](https://github.com/flatironinstitute/finufft) | GPU NUFFT backend | Apache-2.0 |
| [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT) | GPU gridding backend (via MRI-NUFFT) | MIT |
| [torchkbnufft](https://github.com/mmuckley/torchkbnufft) (M. Muckley) | differentiable Kaiser–Bessel NUFFT (via MRI-NUFFT) | MIT |
| [SigPy](https://github.com/mikgroup/sigpy) | reference parallel-imaging / compressed-sensing recon, ESPIRiT | BSD-3-Clause |
| [BART](https://mrirecon.github.io/bart/) | optional reference calibration & iterative reconstruction | BSD-3-Clause |
| [array-api-compat](https://github.com/data-apis/array-api-compat) | array-backend compatibility, if a full shim becomes necessary | MIT |

## Development and documentation tooling

| Tool | Role | License |
| --- | --- | --- |
| [Hatch / hatchling](https://hatch.pypa.io) | build backend | MIT |
| [pytest](https://pytest.org) + [pytest-cov](https://github.com/pytest-dev/pytest-cov) | test runner and coverage | MIT |
| [Ruff](https://docs.astral.sh/ruff/) | linter and formatter | MIT |
| [mypy](https://mypy-lang.org) | static type checking | MIT |
| [MkDocs](https://www.mkdocs.org) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | documentation site | BSD-2-Clause / MIT |
| [mkdocstrings](https://mkdocstrings.github.io) | API docs from docstrings | ISC |
| [pre-commit](https://pre-commit.com) | git hook management | MIT |
| [GitHub Actions](https://github.com/features/actions) | continuous integration | — |

## Standards and community templates

- [ISMRMRD](https://ismrmrd.github.io) — the raw-data interchange format UniMRI
  interoperates with (Inati et al., *Magn Reson Med* 2017).
- [Python Array API standard](https://data-apis.org/array-api/) — the backend
  abstraction contract.
- [Contributor Covenant](https://www.contributor-covenant.org) — the Code of
  Conduct.
- [Keep a Changelog](https://keepachangelog.com) and
  [Semantic Versioning](https://semver.org) — changelog and version scheme.
- [Citation File Format](https://citation-file-format.github.io) — `CITATION.cff`.

## Methods and algorithms

UniMRI implements or will implement the following published methods; credit is to
their originators, with full citations added alongside each implementation:

- **Shepp–Logan phantom** — Shepp & Logan (1974); the modified (Toft) variant
  used by `unimri.testing.phantoms`.
- **Golden-angle radial sampling** — Winkelmann et al. (2007).
- **Density-adapted 3D projection reconstruction** — Nagel et al. (2009); the
  primary non-Cartesian / X-nuclei target.
- **Sampling density compensation (iterative)** — Pipe & Menon (1999).
- **ESPIRiT** coil-sensitivity calibration — Uecker et al. (2014).
- **POCS / homodyne** partial-Fourier reconstruction — Haacke et al.; Noll et al.
- **CG-SENSE** — Pruessmann et al. (2001).

## Inspiration and prior art

UniMRI's unified-data-model + operator-based design follows earlier work,
discussed in detail in [Prior art](prior-art.md):

- [MRIReco.jl](https://github.com/MagneticResonanceImaging/MRIReco.jl) — the same
  thesis, in Julia (Knopp & Grosser, *Magn Reson Med* 2021).
- [mrpro](https://github.com/PTB-MR/mrpro) — a PyTorch MR reconstruction
  framework with a closely related architecture.
- [Gadgetron](https://github.com/gadgetron/gadgetron) — streaming reconstruction
  and vendor converters (Hansen & Sørensen, *Magn Reson Med* 2013).
- [fastMRI](https://github.com/facebookresearch/fastMRI),
  [PyQMRI](https://github.com/IMTtugraz/PyQMRI),
  [DIRECT](https://github.com/NKI-AI/direct) — adjacent toolkits.
- [mapVBVD](https://github.com/CIC-methods/FID-A) / pymapVBVD — Siemens raw
  reading, the lineage `twixtools` comes from.

## Getting credit right

If your library or method is used by UniMRI and is missing, mis-licensed, or
mis-attributed here, please open an issue or pull request — corrections are
welcome and appreciated.
