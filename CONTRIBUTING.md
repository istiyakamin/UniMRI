# Contributing to UniMRI

Thanks for your interest. UniMRI is an early-stage community project and the
architecture is still malleable — feedback on the interfaces is as valuable as
code.

## Ground rules

- Be respectful; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Discuss non-trivial changes in an issue before opening a large PR, especially
  anything that touches `unimri.data` (the data model) or `unimri.operators.base`
  (the operator contract). Those are load-bearing.
- Keep dependencies minimal. Vendor readers, NUFFT backends, and GPU support go
  behind optional extras — the core (`numpy`, `scipy`, `h5py`) stays small.
- Don't reimplement solved problems. Wrap `ismrmrd`, `twixtools`, `mri-nufft`,
  `sigpy`, BART, etc. behind UniMRI interfaces rather than porting their code.
  When you add a dependency or implement a published method, credit it in
  [`docs/acknowledgments.md`](docs/acknowledgments.md) in the same PR.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install        # optional but recommended
```

## Before you push

```bash
ruff check . && ruff format .
mypy src
pytest
```

CI runs the same on Python 3.10–3.13.

## Design principles

1. **The data model is the product.** Reconstruction algorithms are
   replaceable; a stable, well-documented `MRIData` is the contribution.
2. **Reconstruction is an inverse problem.** New reconstructions should be
   expressed via composable `LinearOperator`s, not monolithic functions.
3. **Backend-agnostic.** Core code targets the Python array API; it must not
   hard-assume NumPy where an operator could run on CuPy or PyTorch.
4. **Reproducible by default.** Every operation appends to `MRIData.provenance`.
5. **Interfaces are designed against real data.** Prefer a working end-to-end
   path on one real dataset over a speculative abstraction.

## Commit / PR conventions

- One logical change per PR.
- Reference the issue it closes.
- Add or update tests and docs in the same PR.
