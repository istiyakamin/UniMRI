"""Coil sensitivity estimation.

``estimate_sensitivity`` currently implements one baseline method:

- ``"rss"`` -- normalize each coil's single-pass image
  (:func:`unimri.reconstruction.coil_images`) by the root-sum-of-squares
  combination. Fast, dependency-free, and a reasonable starting point for
  CG-SENSE -- but crude: no noise suppression or explicit support masking,
  unlike ESPIRiT.

Planned: ``"espirit"`` (wrapping ``sigpy.mri.app.EspiritCalib``) and
``"walsh"`` (adaptive combine), see ``docs/roadmap.md`` Milestone 4.
"""

from __future__ import annotations

import numpy as np

from unimri._typing import Array
from unimri.data import MRIData
from unimri.exceptions import UniMRIError

__all__ = ["estimate_sensitivity"]

_METHODS = ("rss",)


def estimate_sensitivity(data: MRIData, *, method: str = "rss", eps: float = 1e-6) -> Array:
    """Return coil sensitivity maps, shape ``(n_coils, *image_shape)``.

    Parameters
    ----------
    data:
        Multi-coil :class:`~unimri.data.MRIData`.
    method:
        ``"rss"`` (the only method implemented so far).
    eps:
        Regularization floor for the RSS normalization, relative to its max.
    """
    if method != "rss":
        raise UniMRIError(f"unknown sensitivity method {method!r}; available: {list(_METHODS)}")

    from unimri.reconstruction.methods import coil_images

    imgs = coil_images(data, eps=eps)
    rss = np.sqrt((np.abs(imgs) ** 2).sum(axis=0))
    floor = eps * (rss.max() + 1e-30)
    return imgs / np.maximum(rss, floor)
