"""Compose linear operators, take adjoints, and verify them.

Reconstruction in UniMRI is an inverse problem y = A x built from composable
LinearOperators. A solver only ever needs forward / adjoint / normal, so it does
not care what A is made of.
"""

from __future__ import annotations

import numpy as np

from unimri.operators import IdentityOperator, LinearOperator


class Diagonal(LinearOperator):
    """Elementwise multiply by a fixed complex vector (a stand-in for, e.g., a
    coil-sensitivity or density-compensation operator)."""

    name = "Diag"

    def __init__(self, weights: np.ndarray) -> None:
        self._w = weights
        self.in_shape = weights.shape
        self.out_shape = weights.shape

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return self._w * x

    def _adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.conj(self._w) * y


class FFT2(LinearOperator):
    """A centred, unitary 2-D FFT (a stand-in for the Fourier encoding operator)."""

    name = "F"

    def __init__(self, shape: tuple[int, int]) -> None:
        self.in_shape = shape
        self.out_shape = shape

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

    def _adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))


def main() -> None:
    rng = np.random.default_rng(0)
    shape = (16, 16)
    # unit-magnitude weights, so S is unitary and AᴴA comes out exactly |2|² I
    sens = np.exp(1j * rng.uniform(0.0, 2 * np.pi, shape))

    S = Diagonal(sens)
    F = FFT2(shape)
    Id = IdentityOperator(shape)

    # Compose: (F @ S)(x) == F(S(x)). Scale with a scalar.
    A = 2.0 * (F @ S @ Id)
    print("operator      :", A)
    print("in/out shape  :", A.in_shape, A.out_shape)

    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    y = A.forward(x)
    x_back = A.H.forward(y)  # adjoint operator
    print("forward shape :", y.shape)
    print("adjoint shape :", x_back.shape)

    # Every operator must pass the adjoint dot-test: <A x, y> == <x, A^H y>.
    for op in (S, F, Id, F @ S, A):
        op.dot_test(in_shape=shape, out_shape=shape)
        print(f"dot_test OK   : {op}")

    # A^H A is available (default: adjoint(forward(x)); operators may override).
    normal = A.normal(x)
    print("A^H A (x) shape:", normal.shape)
    ratio = np.linalg.norm(normal) / np.linalg.norm(x)
    print(f"||A^H A x|| / ||x|| : {ratio:.6f}  (== |2|^2 = 4, since F and S are unitary)")


if __name__ == "__main__":
    main()
