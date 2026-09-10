# The data model

`MRIData` is the one representation every reader produces and every operator,
reconstruction, and pipeline stage consumes. This page is the normative spec for
its contents. Frameworks most often trip over **axis order** and **k-space
units**, so those are pinned down here explicitly.

## `MRIData`

| field | type | notes |
| --- | --- | --- |
| `kspace` | array (complex) | the samples; layout rules below |
| `kspace_axes` | `tuple[str, ...]` | one name per `kspace` axis |
| `encoding` | `EncodingSpace` | geometry and sampling pattern |
| `acquisition` | `AcquisitionInfo` | sequence / contrast / nucleus |
| `coils` | `CoilInfo \| None` | channel count, names, noise covariance |
| `trajectory` | `Trajectory \| None` | `None` ⟺ Cartesian |
| `metadata` | `ScannerMetadata` | vendor, model, field strength |
| `provenance` | `Provenance` | ordered processing log |

`MRIData.validate()` checks internal consistency and raises `ValidationError`
on the first problem. Readers must return a validated object.

## k-space layout

`kspace` is an n-dimensional complex array. `kspace_axes` names every axis. The
**spatial / readout axes come last**, in exactly one of two groupings:

| sampling | trailing axes | example full layout |
| --- | --- | --- |
| Cartesian | `"coil", "kz", "ky", "kx"` (drop `"kz"` for 2D) | `("repetition", "coil", "kz", "ky", "kx")` |
| non-Cartesian | `"coil", "shot", "readout"` | `("echo", "coil", "shot", "readout")` |

`"shot"` indexes trajectory interleaves / spokes / projections; `"readout"`
indexes samples along one shot.

Leading axes describe everything else and may appear in any order. Recognised
names (`unimri.data.KNOWN_AXES`):

```
coil  kx ky kz  shot readout
average contrast echo phase repetition set slice segment user
```

A dataset is Cartesian iff it has **no** `trajectory` **and**
`encoding.sampling == CARTESIAN`.

## `Trajectory` and the unit convention

```python
Trajectory(coords, units=TrajectoryUnits.NORMALIZED, density_compensation=None)
```

`coords` has shape `(n_dims, *sample_axes)` — e.g. `(3, n_shots, n_readout)` —
where `n_dims` is 2 or 3, matching `encoding.n_dims`. The trailing axes must
multiply to the same count as `kspace`'s `shot * readout`.

**Axis order.** Row `d` of `coords` corresponds to **image axis `d`** in NumPy
order, i.e. `(ky, kx)` for a 2-D image `(ny, nx)` and `(kz, ky, kx)` for a 3-D
image `(nz, ny, nx)` — the same slowest-to-fastest order as the Cartesian
`kspace_axes` (`… kz, ky, kx`). `NUFFTOperator` and the reference `ndft` both
assume this.

**Units.** UniMRI's canonical unit is `NORMALIZED`: one unit of `coords` is one
sample of the *encoded matrix*, and the fully-sampled Nyquist window is
`[-N/2, N/2)` per axis. This matches BART and `mri-nufft`'s unitless mode.

| enum | Nyquist window | conversion from normalized (matrix `N`) |
| --- | --- | --- |
| `NORMALIZED` | `[-N/2, N/2)` | — |
| `CYCLES_PER_FOV` | `[-0.5, 0.5)` | `k / N` |
| `RADIANS_PER_VOXEL` | `[-π, π)` | `k · 2π / N` |
| `INVERSE_METERS` | physical | `k / FOV_mm · 1000` |

NUFFT backends that want radians-per-voxel (e.g. `torchkbnufft`) do the
conversion **inside the operator**; readers should store `NORMALIZED`.

Density compensation, when known analytically or supplied by the vendor, lives
in `Trajectory.density_compensation`. It is applied as a separate diagonal
operator during reconstruction so it never pollutes the $A/A^H$ adjoint
relationship.

## `EncodingSpace`

| field | meaning |
| --- | --- |
| `recon_matrix` | target image matrix `(nx, ny, nz)` |
| `encoded_matrix` | nominal fully-sampled k-space matrix `(nx, ny, nz)` |
| `fov` | `FieldOfView(x, y, z)` in **mm** |
| `sampling` | `SamplingPattern` enum (CARTESIAN, RADIAL, SPIRAL, STACK_OF_STARS, CONES, …) |
| `acceleration` | parallel-imaging factor per PE axis, e.g. `(2, 1)` |
| `partial_fourier` | fraction per axis `(ro, pe1, pe2)`, `1.0` = full |
| `n_dims` | 2 or 3 |
| `affine` | optional 4×4 voxel→world (RAS+) matrix |

Both matrices are always length-3 tuples of positive ints (`nz = 1` for 2D).

## `AcquisitionInfo` — including multinuclear

Times in seconds, angles in degrees.

| field | notes |
| --- | --- |
| `sequence_name`, `protocol_name` | |
| `tr_s`, `te_s` (list), `ti_s` (list), `flip_angle_deg` | |
| `n_contrasts`, `n_averages` | |
| `nucleus` | `"1H"`, `"23Na"`, `"31P"`, `"13C"`, `"2H"`, `"17O"`, `"19F"`, `"129Xe"` |
| `field_strength_t` | tesla |
| `larmor_hz` | scanner-reported centre frequency |

Derived: `gamma_hz_per_t` (from a built-in gyromagnetic table),
`expected_larmor_hz` (`|γ| · B₀`), `is_multinuclear()`. X-nuclei support is
first-class from day one — it is almost free at the data-model level and is a
primary motivation (sodium imaging is where Cartesian-only tools fail).

## `Provenance`

```python
data.provenance.record("remove_oversampling", params={"axis": "kx"}, backend="numpy")
```

Each `ProvenanceStep` stores the operation name, its parameters, the `unimri`
version, the backend, and a timestamp. A reconstruction result can therefore be
traced back to the raw file and every parameter in between — intended to be
pasted directly into a methods section.

## On-disk HDF5 schema (planned)

`HDF5Reader` / an `MRIData.to_hdf5()` writer will use a single layout:

```
/kspace                 complex dataset
  @axes                 JSON list of axis names
/trajectory/coords      float dataset            (optional)
/trajectory/dcf         float dataset            (optional)
  @units                "normalized" | …
/encoding               group of scalar attrs
/acquisition            group of scalar attrs
/metadata               group of scalar attrs
/provenance             JSON string attr
```

This is the format used for test fixtures and result caching, and is
deliberately not a competitor to ISMRMRD for interchange.
