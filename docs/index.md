# UniMRI

**One interface for MRI raw data.**

UniMRI is an open-source Python framework that provides a common computational
interface for MRI raw data across scanners, vendors, and acquisition strategies —
Cartesian, radial, spiral, and other non-Cartesian trajectories — so that
reconstruction algorithms can be written once and run on data from anywhere.

!!! warning "Pre-alpha (v0.1.0a1)"
    The data model (`MRIData`), the operator algebra, `FourierOperator` /
    `NUFFTOperator` / `SensitivityOperator`, and
    `reconstruct(method="adjoint"|"cg")` all work today — validated on both
    synthetic phantoms and real 3-D radial ²³Na data. `unimri.read()` reads
    real **Cartesian ISMRMRD** files end-to-end. Siemens TWIX, GE, Philips,
    and non-Cartesian ISMRMRD are not read yet. See the
    [roadmap](roadmap.md) and [I/O & format support](io-format-support.md)
    for exactly what is and isn't implemented — this project tries hard not
    to claim more than the code does.

!!! note "Provisional name"
    `UniMRI` / `unimri` is a working name. Full trademark clearance is still
    pending.

## Why

MRI raw-data tooling is fragmented. Every vendor format has its own loader, and
every reconstruction toolbox has its own in-memory model. A researcher who wants
to try one reconstruction on data from two scanners ends up writing glue code
twice.

UniMRI's bet is that the **unified representation** — not any single algorithm —
is the useful contribution. Normalise vendor raw data into one `MRIData` object,
then reconstruct from that with a trajectory-agnostic operator model.

## Install

```bash
pip install unimri
```

The base install only needs NumPy, SciPy, and h5py. Everything else is an
optional extra, so you only pull in what you actually use:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `nufft` | [FINUFFT](https://finufft.readthedocs.io) | non-Cartesian `reconstruct()` (radial, spiral, ...) |
| `ismrmrd` | [ismrmrd](https://github.com/ismrmrd/ismrmrd-python) | `unimri.read()` on `.mrd` / ISMRMRD `.h5` files |
| `viz` | Matplotlib | the plotting in `examples/` |
| `torch` | PyTorch | GPU / autograd array backend (planned) |
| `gpu` | CuPy | CUDA array backend (planned) |

Combine what you need, e.g. `pip install "unimri[nufft,viz]"`. For
development (running the test suite, linting, building docs):

```bash
git clone https://github.com/istiyakamin/UniMRI
cd UniMRI
pip install -e ".[dev]"
pytest
```

## Quickstart

### Reading and reconstructing a real file

If you have a Cartesian **ISMRMRD** file (`.mrd`, or `.h5` in the ISMRMRD
layout — the output of `siemens_to_ismrmrd`, `ge_to_ismrmrd`, or a Gadgetron
export), this is the whole API:

```python
import unimri

data = unimri.read("scan.mrd")  # format detected automatically
print(data.summary())

image = unimri.reconstruct(data, method="cg")  # or method="adjoint" for single-pass gridding
```

`unimri.read` picks the right `Reader` by sniffing the file (extension + a
few magic bytes), so the call is the same regardless of format — only
Cartesian ISMRMRD actually works yet (see
[I/O & format support](io-format-support.md) for the exact scope and what
raises a clear error instead of guessing). `reconstruct` is likewise
trajectory-agnostic: `"adjoint"` and `"cg"` both run the same code path
whether `data` came from Cartesian or non-Cartesian k-space, dispatching
internally on `data.is_cartesian`.

See `examples/08_ismrmrd_cartesian.py` for a runnable version that writes its
own test ISMRMRD file first, so you can try this with no data of your own.

### No file on hand? Use synthetic data

`unimri.testing` builds a valid `MRIData` from an analytic phantom, with the
exact ground-truth image alongside it — useful for learning the API, for
testing your own code against UniMRI, or for exactly reproducing every
example and test in this repository:

```python
from unimri.data import SamplingPattern
from unimri.testing import synthetic_dataset
import unimri

ds = synthetic_dataset(SamplingPattern.RADIAL, matrix=64, n_coils=8)
image = unimri.reconstruct(ds.data, method="cg", n_iter=15)
# ds.ground_truth is the exact image `ds.data.kspace` was simulated from
```

`SamplingPattern.CARTESIAN`, `.RADIAL`, and `.STACK_OF_STARS` are all
available (`unimri.testing.AVAILABLE_PATTERNS`). See
`examples/02_synthetic_datasets.py` and `examples/04_reference_reconstruction.py`.

### Building `MRIData` by hand

Any array you can get k-space and a trajectory out of (a `.mat` export from
your own pipeline, a MATLAB struct, a proprietary lab format) can become a
`MRIData` without waiting for a dedicated `Reader`:

```python
from unimri.data import EncodingSpace, FieldOfView, MRIData, SamplingPattern

data = MRIData(
    kspace=my_kspace,  # complex array, e.g. (coil, ky, kx)
    kspace_axes=("coil", "ky", "kx"),
    encoding=EncodingSpace(
        recon_matrix=(256, 256, 1),
        encoded_matrix=(256, 256, 1),
        fov=FieldOfView(240.0, 240.0, 5.0),
        sampling=SamplingPattern.CARTESIAN,
        n_dims=2,
    ),
).validate()

image = unimri.reconstruct(data, method="adjoint")
```

`MRIData.validate()` raises `ValidationError` with a specific, actionable
message the first time something is inconsistent (wrong axis names, a
trajectory that doesn't match the sampling pattern, mismatched coil counts,
...) — see [the data model](data-model.md) for the exact schema and, in
particular, the axis-order and k-space-unit conventions that are the easiest
part to get subtly wrong. `examples/01_data_model.py` and
`examples/05_custom_reader.py` walk through this and through writing your
own `Reader` so `unimri.read()` picks up a new format.

## The shape of it

```text
   Siemens TWIX   GE P-file   Philips RAW   ISMRMRD   HDF5
         └─────────────┴───────┬──────┴─────────┴───────┘
                               ▼
                     UniMRI I/O layer            format detection + adapters
                     (ISMRMRD/Cartesian works;    ← unimri.io / unimri.read
                      others still stubs)
                               ▼
                     Unified MRIData             k-space · trajectory ·
                     data model                  encoding · coils · metadata
                               ▼
                     Operator layer              y = F S x
                     (works today)                ← unimri.operators
                               ▼
                     Reconstruction              adjoint (gridding) · CG-SENSE
                     (works today)                ← unimri.reconstruct
                               ▼
                          MRI image
```

## Where to start reading

- [Architecture](architecture.md) — the seven layers and the operator model,
  with an explicit "implemented today" note on every layer.
- [Data model](data-model.md) — the `MRIData` schema and, crucially, the unit
  and axis conventions (two real bugs were caught by getting these wrong
  early — the story is in the [roadmap](roadmap.md)).
- [I/O & format support](io-format-support.md) — exactly what `unimri.read`
  handles today, and the precise scope limits of `ISMRMRDReader`.
- [API reference](reference/index.md) — generated from docstrings.
- [Roadmap](roadmap.md) — what is being built, in what order, and what
  "done" means for each milestone.
- [Prior art](prior-art.md) — how UniMRI relates to `mrpro`, `mri-nufft`,
  `sigpy`, BART, ISMRMRD/Gadgetron, and others, and what it should depend on
  rather than reimplement.
- [Acknowledgments](acknowledgments.md) — every library, standard, and method
  UniMRI uses or is built to use, with credit and licenses.
- [`examples/`](https://github.com/istiyakamin/UniMRI/tree/main/examples) —
  runnable scripts for everything above; `examples/README.md` lists what
  each one shows and whether it needs local data.
