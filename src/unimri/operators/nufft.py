"""Non-uniform Fourier operator (planned).

``NUFFTOperator`` will map between a Cartesian image grid and samples on an
arbitrary :class:`~unimri.data.Trajectory`:

    forward:  image        ->  non-Cartesian k-space   (type-2 NUFFT)
    adjoint:  k-space       ->  image                    (type-1 NUFFT)

UniMRI will **not** ship its own gridder. This operator is a thin adapter over an
established backend -- ``mri-nufft`` (which itself unifies finufft / cufinufft /
gpuNUFFT / torchkbnufft / ...), or ``torchkbnufft`` / ``sigpy`` directly -- chosen
via a ``backend=`` argument. Trajectory unit conversion (normalized ->
radians/voxel etc.) happens here, per ``unimri.data.trajectory``.

Density compensation is supplied by the trajectory or computed by
``unimri.calibration``; it is applied as a separate diagonal operator so it does
not pollute the adjoint relationship.

See ``docs/roadmap.md`` Milestone 5.
"""

from __future__ import annotations

__all__: list[str] = []
