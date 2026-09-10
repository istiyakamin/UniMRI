"""Smoke tests: the package imports cleanly and the public API is wired up."""

from __future__ import annotations

import pytest

import unimri


def test_version_string() -> None:
    assert isinstance(unimri.__version__, str)
    assert unimri.__version__.count(".") == 2


def test_public_api_surface() -> None:
    for name in unimri.__all__:
        assert hasattr(unimri, name), f"unimri.__all__ lists {name!r} but it is missing"


def test_reconstruct_dispatches_and_rejects_unknown_methods(cartesian_mri_data) -> None:
    from unimri.exceptions import UniMRIError

    img = unimri.reconstruct(cartesian_mri_data, method="adjoint")
    assert img.shape == (8, 16, 16)  # (nz, ny, nx)
    with pytest.raises(UniMRIError, match="unknown reconstruction method"):
        unimri.reconstruct(cartesian_mri_data, method="does-not-exist")


def test_read_and_reconstruct_are_exported() -> None:
    assert callable(unimri.read)
    assert callable(unimri.reconstruct)
