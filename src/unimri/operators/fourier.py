"""Cartesian Fourier operator (planned).

``FourierOperator`` will implement a centered n-D FFT/IFFT over designated
k-space axes, with optional oversampling handling, as a :class:`LinearOperator`:

    forward:  image  ->  Cartesian k-space   (F)
    adjoint:  k-space ->  image               (Fᴴ, unitary so Fᴴ = F⁻¹)

Backed by the array namespace of the input (NumPy / CuPy / PyTorch), so it runs
wherever the data lives. See ``docs/roadmap.md`` Milestone 3.
"""

from __future__ import annotations

__all__: list[str] = []
