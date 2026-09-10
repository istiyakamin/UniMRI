# UniMRI

**One interface for MRI raw data.**

UniMRI is an open-source Python framework that provides a common computational
interface for MRI raw data across scanners, vendors, and acquisition strategies —
Cartesian, radial, spiral, and other non-Cartesian trajectories — so that
reconstruction algorithms can be written once and run on data from anywhere.

!!! warning "Pre-alpha (v0.0.0)"
    This repository currently contains the **design specification** and the
    **public interfaces** (`MRIData`, `Reader`, `LinearOperator`) plus the
    project scaffold. No reconstruction backends are implemented yet. See the
    [roadmap](roadmap.md).

!!! note "Provisional name"
    `UniMRI` / `unimri` is a working name. A full PyPI / GitHub / trademark
    clearance is pending before any public release.

## Why

MRI raw-data tooling is fragmented. Every vendor format has its own loader, and
every reconstruction toolbox has its own in-memory model. A researcher who wants
to try one reconstruction on data from two scanners ends up writing glue code
twice.

UniMRI's bet is that the **unified representation** — not any single algorithm —
is the useful contribution. Normalise vendor raw data into one `MRIData` object,
then reconstruct from that with a trajectory-agnostic operator model.

## The shape of it

```text
   Siemens TWIX   GE P-file   Philips RAW   ISMRMRD   HDF5
         └─────────────┴───────┬──────┴─────────┴───────┘
                               ▼
                     UniMRI I/O layer            format detection + adapters
                               ▼
                     Unified MRIData             k-space · trajectory ·
                     data model                  encoding · coils · metadata
                               ▼
                     Operator layer              y = P F S x
                               ▼
                     Reconstruction              FFT · SENSE · CG · CS · …
                               ▼
                          MRI image
```

## Where to start reading

- [Architecture](architecture.md) — the seven layers and the operator model.
- [Data model](data-model.md) — the `MRIData` schema and, crucially, the unit
  and axis conventions.
- [Roadmap](roadmap.md) — what is being built, in what order.
- [Prior art](prior-art.md) — how UniMRI relates to `mrpro`, `mri-nufft`,
  `sigpy`, BART, ISMRMRD/Gadgetron, and others, and what it should depend on
  rather than reimplement.
