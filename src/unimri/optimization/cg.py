"""Conjugate gradient for Hermitian positive-(semi)definite normal equations.

Solves ``(Aᴴ A + l2 I) x = Aᴴ y`` for ``x``, given only ``operator.adjoint`` and
``operator.normal`` (``AᴴA``, or ``Aᴴ A + l2 I`` computed manually here). This is
the trajectory-agnostic core of CG-SENSE (Pruessmann et al., MRM 2001): the same
code reconstructs Cartesian and non-Cartesian data, because the trajectory only
affects what ``A`` -- e.g. ``FourierOperator @ SensitivityOperator`` or
``NUFFTOperator @ SensitivityOperator`` -- computes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from unimri._typing import Array
from unimri.operators.base import LinearOperator


def conjugate_gradient(
    operator: LinearOperator,
    y: Array,
    *,
    n_iter: int = 10,
    l2: float = 0.0,
    x0: Array | None = None,
    tol: float = 1e-10,
    callback: Callable[[int, Array, float], None] | None = None,
) -> Array:
    """Solve ``(Aᴴ A + l2 I) x = Aᴴ y`` by linear conjugate gradient.

    Parameters
    ----------
    operator:
        Exposes ``.adjoint(y)`` and ``.normal(x)`` (``AᴴA x``).
    y:
        Measured data (e.g. multi-coil k-space).
    n_iter:
        Maximum number of iterations.
    l2:
        Tikhonov regularization strength (``0`` = none).
    x0:
        Initial guess; defaults to zero.
    tol:
        Stop early once ``||r|| < tol``.
    callback:
        Optional ``callback(iteration, x, residual_norm)``, called each step.
    """
    b = operator.adjoint(y)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=b.dtype, copy=True)

    def apply(v: Array) -> Array:
        Av = operator.normal(v)
        return Av + l2 * v if l2 else Av

    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.real(np.vdot(r, r)))
    if callback is not None:
        callback(0, x, rs_old**0.5)
    if rs_old**0.5 < tol:
        return x

    for i in range(1, n_iter + 1):
        Ap = apply(p)
        pAp = float(np.real(np.vdot(p, Ap)))
        alpha = rs_old / (pAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.real(np.vdot(r, r)))
        if callback is not None:
            callback(i, x, rs_new**0.5)
        if rs_new**0.5 < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x
