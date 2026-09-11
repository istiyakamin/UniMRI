"""Tests for the conjugate-gradient solver."""

from __future__ import annotations

import numpy as np

from unimri.operators.base import LinearOperator
from unimri.optimization import conjugate_gradient


class Diagonal(LinearOperator):
    """AᴴA = diag(w)^2 -- lets us check CG against the exact solution."""

    def __init__(self, w: np.ndarray) -> None:
        self._w = w
        self.in_shape = w.shape
        self.out_shape = w.shape

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return self._w * x

    def _adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.conj(self._w) * y


def test_cg_recovers_exact_solution_for_diagonal_system() -> None:
    rng = np.random.default_rng(0)
    n = 12
    w = rng.uniform(0.5, 3.0, n) + 0j  # real, positive -> well-conditioned AᴴA
    op = Diagonal(w)
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = op.forward(x_true)

    x_hat = conjugate_gradient(op, y, n_iter=n, tol=0.0)
    assert np.allclose(x_hat, x_true, atol=1e-8)


def test_cg_converges_within_dimension_many_iterations() -> None:
    rng = np.random.default_rng(1)
    n = 8
    w = rng.uniform(1.0, 2.0, n) + 0j
    op = Diagonal(w)
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = op.forward(x_true)

    residuals = []
    conjugate_gradient(op, y, n_iter=n, tol=0.0, callback=lambda i, x, r: residuals.append(r))
    assert residuals[-1] < 1e-6
    # residual should be (roughly) monotonically decreasing
    assert residuals[-1] <= residuals[0]


def test_cg_zero_data_returns_zero() -> None:
    op = Diagonal(np.array([1.0, 2.0, 3.0]))
    y = np.zeros(3, dtype=complex)
    x = conjugate_gradient(op, y, n_iter=5)
    assert np.allclose(x, 0)


def test_cg_l2_regularization_shrinks_solution() -> None:
    rng = np.random.default_rng(2)
    n = 6
    w = rng.uniform(0.5, 1.5, n) + 0j
    op = Diagonal(w)
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = op.forward(x_true)

    x_reg = conjugate_gradient(op, y, n_iter=n, l2=10.0, tol=0.0)
    x_unreg = conjugate_gradient(op, y, n_iter=n, l2=0.0, tol=0.0)
    assert np.linalg.norm(x_reg) < np.linalg.norm(x_unreg)


def test_cg_respects_max_iterations() -> None:
    rng = np.random.default_rng(3)
    n = 50
    w = rng.uniform(0.1, 10.0, n) + 0j  # ill-conditioned -> needs many iterations
    op = Diagonal(w)
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    calls = []
    conjugate_gradient(op, y, n_iter=3, tol=0.0, callback=lambda i, x, r: calls.append(i))
    assert max(calls) == 3
