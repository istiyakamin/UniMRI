# Calibration

Vendor-neutral estimation problems that turn raw `MRIData` into the inputs an
operator (Layer 4) needs — currently just coil sensitivity maps, used by
`reconstruct(method="cg")` when `sensitivity` isn't supplied explicitly. See
[Architecture → Calibration & preprocessing](../architecture.md#layer-3-calibration-preprocessing).

`estimate_sensitivity(method="rss")` is a crude, dependency-free baseline:
per-coil images from a single-pass gridding/iFFT (the same `coil_images()`
building block `reconstruct(method="adjoint")` uses), normalized by their
root-sum-of-squares combination. It is *not* a substitute for ESPIRiT or an
adaptive/Walsh combine — those are planned (Milestone 4, wrapping `sigpy.mri`
/ BART rather than reimplementing them) and will be selectable via the same
`method=` argument.

::: unimri.calibration.estimate_sensitivity
