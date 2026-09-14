"""Scanner metadata and reconstruction provenance."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScannerMetadata:
    """Information about the acquisition system.

    All fields are optional; readers fill in what the source format provides.
    Anything without a typed home goes in :attr:`extra`.
    """

    vendor: str | None = None
    model: str | None = None
    software_version: str | None = None
    field_strength_t: float | None = None
    institution: str | None = None
    system_id: str | None = None
    receiver_channels: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceStep:
    """A single recorded operation in a processing history."""

    operation: str
    params: dict[str, Any] = field(default_factory=dict)
    unimri_version: str = ""
    backend: str = ""
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"[{when}] {self.operation}({args})"


@dataclass
class Provenance:
    """Ordered history of everything done to a dataset.

    Reproducibility is a first-class feature of UniMRI: reading a file and every
    subsequent transform should append a :class:`ProvenanceStep` here.
    """

    steps: list[ProvenanceStep] = field(default_factory=list)

    def record(
        self,
        operation: str,
        *,
        params: dict[str, Any] | None = None,
        backend: str = "",
    ) -> ProvenanceStep:
        """Append a step and return it."""
        from unimri import __version__

        step = ProvenanceStep(
            operation=operation,
            params=dict(params or {}),
            unimri_version=__version__,
            backend=backend,
        )
        self.steps.append(step)
        return step

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if not self.steps:
            return "Provenance: (empty)"
        return "Provenance:\n" + "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
