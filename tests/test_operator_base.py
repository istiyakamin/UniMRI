"""Contract tests for the linear-operator algebra."""

from __future__ import annotations

import numpy as np
import pytest

from unimri.exceptions import OperatorShapeError
from unimri.operators import (
    CompositeOperator,
    IdentityOperator,
    LinearOperator,
    ScaledOperator,
)
from unimri.operators.base import AdjointOperator


class Diagonal(LinearOperator):
    """A concrete test operator: elementwise multiply by a fixed complex vector."""

    name = "Diag"

    def __init__(self, weights: np.ndarray) -> None:
        self._w = weights
        self.in_shape = weights.shape
        self.out_shape = weights.shape

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return self._w * x

    def _adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.conj(self._w) * y


class Shift(LinearOperator):
    """Circular shift by one along axis 0 -- adjoint is the opposite shift."""

    name = "Shift"

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.in_shape = shape
        self.out_shape = shape

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return np.roll(x, 1, axis=0)

    def _adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.roll(y, -1, axis=0)


@pytest.fixture
def w(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))


def test_identity_roundtrip_and_dot_test() -> None:
    op = IdentityOperator((4, 4))
    x = np.arange(16).reshape(4, 4).astype(complex)
    assert np.array_equal(op.forward(x), x)
    assert np.array_equal(op.normal(x), x)
    assert op.dot_test()


def test_diagonal_dot_test(w: np.ndarray) -> None:
    assert Diagonal(w).dot_test()


def test_shift_dot_test() -> None:
    assert Shift((7, 3)).dot_test()


def test_adjoint_property_involutive(w: np.ndarray) -> None:
    op = Diagonal(w)
    assert isinstance(op.H, AdjointOperator)
    assert op.H.H is op
    y = np.ones_like(w)
    assert np.allclose(op.H.forward(y), op.adjoint(y))


def test_scaled_operator_conjugates_scalar_in_adjoint(w: np.ndarray) -> None:
    op = (2 + 1j) * Diagonal(w)
    assert isinstance(op, ScaledOperator)
    assert op.dot_test()
    x = np.ones_like(w)
    assert np.allclose(op.forward(x), (2 + 1j) * (w * x))


def test_composition_applies_right_to_left() -> None:
    shape = (5, 4)
    a = Shift(shape)
    d = Diagonal(np.full(shape, 2.0 + 0j))
    comp = a @ d
    assert isinstance(comp, CompositeOperator)
    x = np.arange(20).reshape(shape).astype(complex)
    assert np.allclose(comp.forward(x), np.roll(2.0 * x, 1, axis=0))
    assert comp.dot_test()


def test_composition_flattens_nested_composites() -> None:
    shape = (4, 4)
    ops = [Shift(shape), Diagonal(np.ones(shape, complex)), Shift(shape)]
    comp = (ops[0] @ ops[1]) @ ops[2]
    assert isinstance(comp, CompositeOperator)
    assert len(comp.operators) == 3


def test_shape_mismatch_raises(w: np.ndarray) -> None:
    op = Diagonal(w)
    with pytest.raises(OperatorShapeError):
        op.forward(np.ones((2, 2), dtype=complex))
    with pytest.raises(OperatorShapeError):
        op.adjoint(np.ones((2, 2), dtype=complex))


def test_dot_test_needs_concrete_shapes() -> None:
    class Flexible(LinearOperator):
        def _forward(self, x: np.ndarray) -> np.ndarray:
            return x

        def _adjoint(self, y: np.ndarray) -> np.ndarray:
            return y

    with pytest.raises(ValueError, match="fully-specified"):
        Flexible().dot_test()
    assert Flexible().dot_test(in_shape=(3,), out_shape=(3,))
