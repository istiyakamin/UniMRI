"""Runtime configuration for UniMRI.

Access the singleton via :data:`config`. Values can also be set from the
environment with the ``UNIMRI_`` prefix (e.g. ``UNIMRI_DEFAULT_DEVICE=cuda``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("unimri")


@dataclass
class Config:
    """Global, process-wide options."""

    #: Preferred compute device for backends that support it ("cpu", "cuda", ...).
    default_device: str = os.environ.get("UNIMRI_DEFAULT_DEVICE", "cpu")

    #: Whether every operation should append a step to ``MRIData.provenance``.
    record_provenance: bool = os.environ.get("UNIMRI_RECORD_PROVENANCE", "1") != "0"

    #: Verbosity for progress reporting in long-running reconstructions.
    verbose: bool = os.environ.get("UNIMRI_VERBOSE", "1") != "0"


config = Config()
