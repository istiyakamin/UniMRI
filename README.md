# UniMRI

**One interface for MRI raw data.**

UniMRI is an open-source Python framework that provides a common computational
interface for MRI raw data across scanners, vendors, and acquisition strategies —
Cartesian, radial, spiral, and other non-Cartesian trajectories — so that
reconstruction algorithms can be written once and run on data from anywhere.

> **Status: pre-alpha (v0.0.0).** This repository currently contains the design
> specification, the public interfaces (`MRIData`, `Reader`, `LinearOperator`),
> and the project scaffold. No reconstruction backends are implemented yet — see
> the [roadmap](docs/roadmap.md).

> **Name caveat:** `UniMRI` / `unimri` is a provisional working name. A full
> PyPI / GitHub / trademark clearance is still pending before any public release
> or PyPI upload.

## The idea

The core problem is fragmentation: every vendor format has its own loader, and
every reconstruction toolbox has its own data model. UniMRI normalizes vendor
raw data into one representation, then reconstructs from that.

```text
        Siemens TWIX     GE P-file     Philips RAW     ISMRMRD     HDF5
              │              │              │             │          │
              └──────────────┴──────┬───────┴─────────────┴──────────┘
                                    ▼
                          ┌──────────────────┐
                          │  UniMRI IO layer │      format detection + adapters
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │ Unified MRIData  │      k-space · trajectory ·
                          │   data model     │      encoding · coils · metadata
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  Operator layer  │      y = P F S x
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │  Reconstruction  │      FFT · SENSE · CG · CS · …
                          └────────┬─────────┘
                                   ▼
                               MRI image
```

## Aspirational API

```python
import unimri

raw = unimri.read("measurement.dat")  # vendor-agnostic
image = unimri.reconstruct(raw, method="fft")  # trajectory-agnostic
```

```python
# advanced: reconstruction as an inverse problem
from unimri.operators import FourierOperator, SensitivityOperator, SamplingOperator
from unimri.optimization import conjugate_gradient

A = SamplingOperator(raw.encoding) @ FourierOperator(raw.trajectory) @ SensitivityOperator(maps)
image = conjugate_gradient(A, raw.kspace, n_iter=30)
```

These examples describe the target interface. They do **not** work yet.

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

- [Architecture](docs/architecture.md) — the layered design and the operator model
- [Data model](docs/data-model.md) — `MRIData` schema and unit conventions
- [I/O & format support](docs/io-format-support.md)
- [Roadmap](docs/roadmap.md)
- [Prior art](docs/prior-art.md) — how UniMRI relates to `mrpro`, `mri-nufft`,
  `sigpy`, BART, ISMRMRD/Gadgetron, and others

## Contributing

UniMRI is meant to grow as a community project. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/istiyakamin/UniMRI/labels/good%20first%20issue).

## License

MIT — see [LICENSE](LICENSE).
