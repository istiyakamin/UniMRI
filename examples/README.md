# Examples

Run any of these from the repo root after `pip install -e ".[dev]"`:

```bash
python examples/01_data_model.py
```

## Runnable today (pre-alpha)

| File | Shows |
| --- | --- |
| `01_data_model.py` | Build an `MRIData` by hand, `validate()` it, read its `summary()` and `provenance` |
| `02_synthetic_datasets.py` | `unimri.testing.synthetic_dataset(...)` — one valid dataset per sampling pattern, with known ground truth |
| `03_operator_algebra.py` | `LinearOperator` — compose with `@`, scale with `*`, take the adjoint `.H`, and check it with `dot_test()` |
| `04_reference_reconstruction.py` | Reconstruct a synthetic radial dataset with the brute-force reference NDFT (density-compensated gridding) and score it against ground truth |
| `05_custom_reader.py` | Write and register your own `Reader` so `unimri.read()` handles a new format |
| `06_multinuclear.py` | The gyromagnetic table; `AcquisitionInfo` for ²³Na vs ¹H |

`test_examples.py` in the test suite executes 01–06 in CI, so they stay working.

## Aspirational

| File | Shows |
| --- | --- |
| `99_aspirational_api.py` | The **intended** end-to-end API (`unimri.read` → `unimri.reconstruct`). Does **not** run yet — no readers or reconstruction backends exist. Tracks the target in `docs/roadmap.md`. |
