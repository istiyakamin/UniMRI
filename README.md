<p align="center">
  <img src="https://raw.githubusercontent.com/istiyakamin/UniMRI/main/assets/logo.png" alt="UniMRI — one interface for MRI raw data" width="380">
</p>

<h1 align="center">UniMRI</h1>

<p align="center"><strong>One interface for MRI raw data.</strong></p>

<p align="center">
  <a href="https://github.com/istiyakamin/UniMRI/actions/workflows/ci.yml"><img src="https://github.com/istiyakamin/UniMRI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status: pre-alpha">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/istiyakamin/UniMRI/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <img src="https://img.shields.io/badge/types-typed-brightgreen.svg" alt="Typed">
  <a href="https://istiyakamin.github.io/UniMRI/"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/unimri/"><img src="https://img.shields.io/pypi/v/unimri.svg" alt="PyPI"></a>
</p>

<!-- Enable once these apply:
  <a href="https://pypi.org/project/unimri/"><img src="https://img.shields.io/pypi/dm/unimri.svg?label=PyPI%20downloads" alt="PyPI downloads"></a>  (once there's real download volume)
  <a href="https://doi.org/PLACEHOLDER"><img src="https://img.shields.io/badge/DOI-PLACEHOLDER-blue.svg" alt="DOI"></a>  (needs Zenodo integration)
  <a href="https://securityscorecards.dev/viewer/?uri=github.com/istiyakamin/UniMRI"><img src="https://api.securityscorecards.dev/projects/github.com/istiyakamin/UniMRI/badge" alt="OpenSSF Scorecard"></a>
-->

UniMRI is an open-source Python framework that provides a common computational
interface for MRI raw data across scanners, vendors, and acquisition strategies —
Cartesian, radial, spiral, and other non-Cartesian trajectories — so that
reconstruction algorithms can be written once and run on data from anywhere.

> **Status: pre-alpha (v0.1.0a1).** The data model (`MRIData`), the operator
> algebra, `FourierOperator`/`NUFFTOperator`/`SensitivityOperator`, and
> `reconstruct(method="adjoint"|"cg")` — single-pass gridding and CG-SENSE, the
> same code for Cartesian and non-Cartesian data — all work, validated on real
> 3-D radial ²³Na data. `unimri.read()` now reads real **Cartesian ISMRMRD**
> files end-to-end (single slice/average/contrast/repetition/set/segment);
> other vendor formats (Siemens TWIX, GE, Philips) and non-Cartesian ISMRMRD
> are still stubs. See the [roadmap](https://istiyakamin.github.io/UniMRI/roadmap/).

> **Name caveat:** `UniMRI` / `unimri` is a provisional working name. A full
> PyPI / GitHub / trademark clearance is still pending before any public release
> or PyPI upload.

## The idea

The core problem is fragmentation: every vendor format has its own loader, and
every reconstruction toolbox has its own data model. UniMRI normalizes vendor
raw data into one representation, then reconstructs from that.

<p align="center">
  <img src="https://raw.githubusercontent.com/istiyakamin/UniMRI/main/assets/idea.png" alt="UniMRI pipeline: raw data from any vendor (Siemens TWIX, GE P-file, Philips RAW, ISMRMRD, HDF5) flows through the I/O layer (format detection + reader registry), into the unified MRIData model (k-space, trajectory, encoding, coils, metadata, provenance), through the operator layer (reconstruction as an inverse problem, y = P F S x), through reconstruction (FFT, SENSE, GRAPPA, CG-SENSE, compressed sensing), to an MRI image — vendor- and trajectory-independent." width="560">
</p>

## Aspirational API

```python
import unimri

raw = unimri.read("scan.mrd")  # ISMRMRD, Cartesian -- WORKS TODAY (see examples/08)
image = unimri.reconstruct(raw, method="cg")  # trajectory-agnostic -- WORKS TODAY
```

Other vendor formats (Siemens TWIX, GE, Philips) and non-Cartesian ISMRMRD
are not read yet -- `unimri.read()` raises a clear error naming what's
missing rather than guessing. See
[I/O & format support](https://istiyakamin.github.io/UniMRI/io-format-support/)
for exactly what's supported.

```python
# advanced: reconstruction as an inverse problem -- WORKS TODAY (see examples/07_cg_sense.py)
from unimri.calibration import estimate_sensitivity
from unimri.operators import FourierOperator, NUFFTOperator, SensitivityOperator, unchecked
from unimri.optimization import conjugate_gradient

maps = estimate_sensitivity(raw)
F = (
    NUFFTOperator(raw.trajectory, image_shape)
    if not raw.is_cartesian
    else FourierOperator(image_shape)
)
A = unchecked(F @ SensitivityOperator(maps))
image = conjugate_gradient(A, raw.kspace, n_iter=30)
```

Reading Cartesian ISMRMRD data and reconstructing it (both `"adjoint"` and
`"cg"`) work today end-to-end (see `examples/`); other vendor formats and
non-Cartesian ISMRMRD are not read yet.

## Installation (development)

```bash
git clone https://github.com/istiyakamin/UniMRI
cd UniMRI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

GPU and vendor-specific support are optional extras: `pip install -e ".[gpu]"`,
`".[torch]"`, `".[ismrmrd]"`.

## Documentation

- [Architecture](https://istiyakamin.github.io/UniMRI/architecture/) — the layered design and the operator model
- [Data model](https://istiyakamin.github.io/UniMRI/data-model/) — `MRIData` schema and unit conventions
- [I/O & format support](https://istiyakamin.github.io/UniMRI/io-format-support/)
- [Roadmap](https://istiyakamin.github.io/UniMRI/roadmap/)
- [Prior art](https://istiyakamin.github.io/UniMRI/prior-art/) — how UniMRI relates to `mrpro`, `mri-nufft`,
  `sigpy`, BART, ISMRMRD/Gadgetron, and others
- [Acknowledgments](https://istiyakamin.github.io/UniMRI/acknowledgments/) — every library, standard, and
  method UniMRI uses, with credit

## Contributing

UniMRI is meant to grow as a community project. See
[CONTRIBUTING.md](https://github.com/istiyakamin/UniMRI/blob/main/CONTRIBUTING.md) and the
[good first issues](https://github.com/istiyakamin/UniMRI/labels/good%20first%20issue).

## Acknowledgments

UniMRI builds on the open-source scientific Python ecosystem — NumPy, SciPy,
h5py, and (as it grows) ISMRMRD, MRI-NUFFT, SigPy, BART, twixtools, PyTorch, and
others, plus published reconstruction methods credited to their originators. The
full list with licenses is in [docs/acknowledgments.md](https://istiyakamin.github.io/UniMRI/acknowledgments/).

## License

MIT — see [LICENSE](https://github.com/istiyakamin/UniMRI/blob/main/LICENSE).
