# I/O and format support

## The reader contract

```python
class Reader(ABC):
    format_name: str
    extensions: tuple[str, ...]

    def can_read(self, path) -> bool: ...  # cheap sniff — extension + magic bytes
    def read(self, path, **options) -> MRIData: ...  # full parse, validated, provenance recorded
```

- `can_read` must be cheap and side-effect-free: check the extension, and if
  needed read a few magic bytes. Never parse the whole file.
- `read` must return a `MRIData` that passes `.validate()` and has at least one
  `provenance` step describing the read (source path, reader, key options).
- A reader that raises inside `can_read` is skipped, not fatal.

## Dispatch

Readers register with `register_reader(MyReader())` (optionally
`prepend=True` for priority). `unimri.read(path)` calls `find_reader`, which
returns the first reader whose `can_read` accepts the file, else raises
`UnsupportedFormatError`. Built-in readers are registered at import time.

## Format status

| Format | Extension | Reader | Status | Backend |
| --- | --- | --- | --- | --- |
| ISMRMRD | `.mrd`, `.h5` | `ISMRMRDReader` | interface only — **Milestone 1** | `ismrmrd` |
| UniMRI HDF5 | `.h5`, `.hdf5` | `HDF5Reader` | interface only — **Milestone 1** | `h5py` |
| Siemens TWIX | `.dat` | `TwixReader` | interface only — **Milestone 2** | `twixtools` |
| GE | `.7`, ScanArchive | — | planned — Milestone 7 | TBD (Orchestra / `pfile`) |
| Philips | `.raw`/`.lab`/`.sin` | — | planned — Milestone 7 | TBD |
| Bruker | `fid`/`2dseq` | — | later | `brukerapi` |

"Interface only" means `can_read` is implemented (so detection works) but
`read` raises `NotImplementedError` pointing at the roadmap.

## Why ISMRMRD first

ISMRMRD already solved the hard part of the data model — it standardises
acquisitions, encoding spaces, and trajectories, and there is a large body of
public test data and vendor converters (`siemens_to_ismrmrd`, `ge_to_ismrmrd`,
`philips_to_ismrmrd`). Making `ISMRMRDReader` the first real reader means the
`MRIData` design is validated against a mature schema and real datasets before
the vendor-specific readers are written.

## Trajectory extraction

For non-Cartesian vendor data the reader is responsible for producing a
`Trajectory` in `NORMALIZED` units:

- **ISMRMRD** — use the stored `traj` array on each acquisition when present.
- **Siemens TWIX** — either the analytic trajectory computed from sequence
  special-card parameters (radial views, density-adapted readout profile, etc.),
  or a measured trajectory stored as a separate MDB category. Gradient-delay /
  GIRF correction is a preprocessing step, not the reader's job.

## Writing your own reader

Subclass `Reader`, implement the two methods, and register it:

```python
from unimri.io import Reader, register_reader


class MyLabReader(Reader):
    format_name = "MyLab binary"
    extensions = (".mlb",)

    def can_read(self, path): ...
    def read(self, path, **opts): ...


register_reader(MyLabReader())
```

Contributions of readers for additional formats are very welcome — see
`CONTRIBUTING.md`.
