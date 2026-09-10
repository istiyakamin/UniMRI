"""Chainable processing pipelines (interface sketch).

A :class:`Pipeline` is an ordered list of :class:`Stage` objects. Running it
threads an :class:`~unimri.data.MRIData` through each stage, and every stage
appends to ``data.provenance`` so the whole run is reproducible and inspectable.

Only the interface is defined here; concrete stages arrive with the
preprocessing / calibration / reconstruction layers (see ``docs/roadmap.md``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from unimri.data import MRIData


class Stage(ABC):
    """One step in a :class:`Pipeline`."""

    #: Name recorded in provenance.
    name: str = ""

    @abstractmethod
    def apply(self, data: MRIData) -> MRIData:
        """Transform ``data`` and return it (mutation or copy, stage's choice)."""

    def _record(self, data: MRIData, **params: Any) -> None:
        data.provenance.record(self.name or type(self).__name__, params=params)


class Pipeline:
    """An ordered, inspectable sequence of :class:`Stage` objects.

    Example (aspirational -- stages not implemented yet)::

        pipe = (
            Pipeline()
            .add(RemoveOversampling())
            .add(CompressCoils(n_virtual=8))
            .add(EstimateSensitivity(method="espirit"))
            .add(Reconstruct(method="sense"))
        )
        image = pipe.run(raw)
    """

    def __init__(self, stages: list[Stage] | None = None) -> None:
        self.stages: list[Stage] = list(stages or [])

    def add(self, stage: Stage) -> Pipeline:
        self.stages.append(stage)
        return self

    def run(self, data: MRIData) -> MRIData:
        for stage in self.stages:
            data = stage.apply(data)
        return data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = [s.name or type(s).__name__ for s in self.stages]
        return f"Pipeline({' -> '.join(names) or 'empty'})"
