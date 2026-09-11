"""The linear-operator algebra at the heart of UniMRI reconstruction.

UniMRI treats reconstruction as an inverse problem

    y = A x           with   A = P F S    (sampling . Fourier . sensitivity)

Every building block is a :class:`LinearOperator` exposing ``forward`` (``A``),
``adjoint`` (``Aᴴ``), and ``normal`` (``AᴴA``). Operators compose with ``@`` and
scale with ``*``, so a reconstruction is just an expression, and a solver only
needs ``forward`` / ``adjoint`` / ``normal`` -- it never cares whether the data
was Cartesian or radial.

This module provides the abstract base and the algebraic operators
(:class:`IdentityOperator`, :class:`ScaledOperator`, :class:`CompositeOperator`,
and the adjoint wrapper). Physical operators live in the sibling modules
``fourier``, ``nufft``, ``sampling``, ``coil``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from unimri._typing import Array
from unimri.exceptions import OperatorShapeError

Shape = tuple[int | None, ...] | None


def _shape_ok(declared: Shape, actual: tuple[int, ...]) -> bool:
    if declared is None:
        return True
    if len(declared) != len(actual):
        return False
    return all(d is None or d == a for d, a in zip(declared, actual))


class LinearOperator(ABC):
    """Abstract linear operator ``A`` with an adjoint ``Aᴴ``.

    Subclasses implement :meth:`_forward` and :meth:`_adjoint`. Declare
    ``in_shape`` / ``out_shape`` (tuples, with ``None`` for free axes) to get
    automatic shape checking; leave them ``None`` to skip it.
    """

    in_shape: Shape = None
    out_shape: Shape = None
    name: str = ""

    # -- to implement in subclasses --------------------------------------

    @abstractmethod
    def _forward(self, x: Array) -> Array: ...

    @abstractmethod
    def _adjoint(self, y: Array) -> Array: ...

    def _normal(self, x: Array) -> Array:
        return self._adjoint(self._forward(x))

    # -- public API ----------------------------------------------------

    def forward(self, x: Array) -> Array:
        """Apply ``A`` to ``x``."""
        if not _shape_ok(self.in_shape, tuple(x.shape)):
            raise OperatorShapeError(
                f"{self!r}.forward expected input shape {self.in_shape}, got {tuple(x.shape)}"
            )
        return self._forward(x)

    def adjoint(self, y: Array) -> Array:
        """Apply ``Aᴴ`` to ``y``."""
        if not _shape_ok(self.out_shape, tuple(y.shape)):
            raise OperatorShapeError(
                f"{self!r}.adjoint expected input shape {self.out_shape}, got {tuple(y.shape)}"
            )
        return self._adjoint(y)

    def normal(self, x: Array) -> Array:
        """Apply ``AᴴA`` to ``x`` (override :meth:`_normal` for a fast path)."""
        if not _shape_ok(self.in_shape, tuple(x.shape)):
            raise OperatorShapeError(
                f"{self!r}.normal expected input shape {self.in_shape}, got {tuple(x.shape)}"
            )
        return self._normal(x)

    __call__ = forward

    @property
    def H(self) -> LinearOperator:  # noqa: N802 - matches numpy / scipy convention
        """The adjoint operator ``Aᴴ``. ``A.H.H is A``."""
        return AdjointOperator(self)

    # -- algebra ------------------------------------------------------

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """``(A @ B)(x) == A(B(x))``."""
        if not isinstance(other, LinearOperator):
            return NotImplemented
        left = self.operators if isinstance(self, CompositeOperator) else [self]
        right = other.operators if isinstance(other, CompositeOperator) else [other]
        return CompositeOperator([*left, *right])

    def __mul__(self, scalar: complex) -> LinearOperator:
        return ScaledOperator(self, scalar)

    __rmul__ = __mul__

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        label = self.name or type(self).__name__
        return f"<{label}>"

    # -- diagnostics -------------------------------------------------

    def dot_test(
        self,
        in_shape: tuple[int, ...] | None = None,
        out_shape: tuple[int, ...] | None = None,
        *,
        n_trials: int = 3,
        rtol: float = 1e-5,
        seed: int = 0,
    ) -> bool:
        """Verify the adjoint via ``⟨A x, y⟩ ≈ ⟨x, Aᴴ y⟩`` on random complex data.

        NumPy-only diagnostic. Provide concrete ``in_shape`` / ``out_shape`` if
        the operator declares free (``None``) axes.
        """
        import numpy as np

        raw_ish = in_shape or self.in_shape
        raw_osh = out_shape or self.out_shape
        if (
            raw_ish is None
            or raw_osh is None
            or any(d is None for d in raw_ish)
            or any(d is None for d in raw_osh)
        ):
            raise ValueError("dot_test needs fully-specified in_shape and out_shape")
        ish: tuple[int, ...] = tuple(int(d) for d in raw_ish if d is not None)
        osh: tuple[int, ...] = tuple(int(d) for d in raw_osh if d is not None)

        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            x = rng.standard_normal(ish) + 1j * rng.standard_normal(ish)
            y = rng.standard_normal(osh) + 1j * rng.standard_normal(osh)
            lhs = np.vdot(self.forward(x), y)
            rhs = np.vdot(x, self.adjoint(y))
            if not np.isclose(lhs, rhs, rtol=rtol, atol=0.0):
                raise AssertionError(
                    f"{self!r} failed the adjoint dot-test: ⟨Ax,y⟩={lhs:.6g} vs ⟨x,Aᴴy⟩={rhs:.6g}"
                )
        return True


class AdjointOperator(LinearOperator):
    """Wraps an operator to swap ``forward`` and ``adjoint``."""

    def __init__(self, base: LinearOperator) -> None:
        self._base = base
        self.in_shape = base.out_shape
        self.out_shape = base.in_shape
        self.name = f"{base.name or type(base).__name__}ᴴ"

    def _forward(self, x: Array) -> Array:
        return self._base._adjoint(x)

    def _adjoint(self, y: Array) -> Array:
        return self._base._forward(y)

    @property
    def H(self) -> LinearOperator:
        return self._base


class IdentityOperator(LinearOperator):
    """``I``. Useful as a neutral element and in tests."""

    name = "I"

    def __init__(self, shape: Shape = None) -> None:
        self.in_shape = shape
        self.out_shape = shape

    def _forward(self, x: Array) -> Array:
        return x

    def _adjoint(self, y: Array) -> Array:
        return y

    def _normal(self, x: Array) -> Array:
        return x


class ScaledOperator(LinearOperator):
    """``c * A`` for a (possibly complex) scalar ``c``."""

    def __init__(self, base: LinearOperator, scalar: complex) -> None:
        self._base = base
        self._c = scalar
        self.in_shape = base.in_shape
        self.out_shape = base.out_shape
        self.name = f"{scalar}*{base.name or type(base).__name__}"

    def _forward(self, x: Array) -> Array:
        return self._c * self._base._forward(x)

    def _adjoint(self, y: Array) -> Array:
        c = self._c.conjugate() if isinstance(self._c, complex) else self._c
        return c * self._base._adjoint(y)


class CompositeOperator(LinearOperator):
    """A chain ``A₁ A₂ … Aₙ`` applied right-to-left; adjoint reverses the chain."""

    def __init__(self, operators: list[LinearOperator]) -> None:
        if not operators:
            raise ValueError("CompositeOperator needs at least one operator")
        self.operators = list(operators)
        self.in_shape = self.operators[-1].in_shape
        self.out_shape = self.operators[0].out_shape
        self.name = " @ ".join(op.name or type(op).__name__ for op in self.operators)

    def _forward(self, x: Array) -> Array:
        for op in reversed(self.operators):
            x = op._forward(x)
        return x

    def _adjoint(self, y: Array) -> Array:
        for op in self.operators:
            y = op._adjoint(y)
        return y


class UncheckedOperator(LinearOperator):
    """Wrap ``base``, skipping the public input/output shape validation.

    Some operators (e.g. :class:`~unimri.operators.fourier.FourierOperator` and
    :class:`~unimri.operators.nufft.NUFFTOperator`) declare the shape for a
    *single* image but their ``_forward``/``_adjoint`` also accept a batched
    input with a leading axis (e.g. coils) -- that is how
    ``FourierOperator(...) @ SensitivityOperator(maps)`` becomes a full SENSE
    encoding operator. The public :meth:`forward`/:meth:`adjoint`/:meth:`normal`
    would reject the batched shape; this wrapper calls the unchecked internals
    directly, for use once you know the batching is intentional (e.g. inside a
    solver such as :func:`unimri.optimization.conjugate_gradient`).
    """

    def __init__(self, base: LinearOperator) -> None:
        self._base = base
        self.in_shape = None
        self.out_shape = None
        self.name = f"unchecked({base.name or type(base).__name__})"

    def _forward(self, x: Array) -> Array:
        return self._base._forward(x)

    def _adjoint(self, y: Array) -> Array:
        return self._base._adjoint(y)

    def _normal(self, x: Array) -> Array:
        return self._base._normal(x)


def unchecked(op: LinearOperator) -> UncheckedOperator:
    """Shorthand for :class:`UncheckedOperator`."""
    return UncheckedOperator(op)
