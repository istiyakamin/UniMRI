# Optimization

Generic iterative solvers that work against any `LinearOperator` — they know
nothing about MRI, k-space, or coils. See
[Architecture → Layer 6, Pipeline](../architecture.md#layer-6-pipeline) for
where solvers sit relative to the operator and reconstruction layers.

`conjugate_gradient` solves the normal equations

$$
(A^H A + \lambda I)\,x = A^H y
$$

for any `LinearOperator` `A` and data `y`, using only `A.adjoint` and
`A.normal` — it never calls `A.forward` directly. This is the solver behind
`reconstruct(method="cg")` (CG-SENSE, Pruessmann et al., *MRM* 2001), where
`A = F @ S` (a `FourierOperator` or `NUFFTOperator` composed with a
`SensitivityOperator`): the same solver code runs on Cartesian and
non-Cartesian data because only `A` changes with the trajectory, not the
optimization.

```python
from unimri.operators import FourierOperator, SensitivityOperator, unchecked
from unimri.optimization import conjugate_gradient

A = unchecked(FourierOperator(image_shape) @ SensitivityOperator(coil_maps))
image = conjugate_gradient(A, kspace, n_iter=15, l2=1e-4)
```

`l2` is Tikhonov regularization toward zero (helps on undersampled /
noisy data at the cost of some bias); `tol` stops early once the residual
norm is small enough; `callback(i, x, residual_norm)` runs after every
iteration, useful for convergence plots or early stopping on your own
criterion.

::: unimri.optimization.conjugate_gradient
