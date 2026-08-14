# Imaging likelihood hazards

This directory contains imaging-specific fixtures and cells that wrap
the reusable detectors in [`scripts/misc/hazards/`](../../misc/hazards/README.md)
around a complete likelihood.

`pixelization.py` is the first tier-2 cell. It uses an in-memory 7x7 imaging
dataset, an Isothermal lens and a 3x3 rectangular source with constant
regularization. The bounded scan measures active-set support transitions,
matrix-floor scale, NumPy/JAX backend divergence, and the circular-profile
orientation degeneracy. The conditioning leg also runs an unfloored control and
a scale-aware counterfactual calibrated to the packaged absolute floor at the
reference noise scale; this is profiling evidence only and does not change the
PyAutoArray default. Floor fractions use only the curvature-diagonal entries in
`no_regularization_index_list`, because those are the entries the policy
actually modifies. Reusable detector logic remains in `misc/hazards`.

The solver-diagnostic leg re-solves each NumPy- and JAX-built NNLS system with
both positive solvers. It also samples the one-ULP neighbourhood around the
largest native-path difference and records objective and scale-normalized KKT
residuals. This separates solver convergence from backend differences already
present in the curvature-regularization matrix and data vector.

The border-relocator leg continues that diagnosis through traced source grids,
PCA ellipse parameters, relocated grids, mesh extents, mappings and likelihood
systems. It includes an axis-stable counterfactual for a near-isotropic border
covariance, where PCA eigenvectors are otherwise mathematically undefined and
backend eigensolvers may choose different orientations.
The stable backend-divergence finding is emitted only when a complete
likelihood output exceeds a `1e-8` relative parity tolerance; the finding ID
therefore returns unchanged if this resolved source behavior regresses.
The curvature-floor documentation finding likewise remains stable but is
emitted only when the helper and `Settings` docstrings do not both name the
live packaged default.

Run the cell directly to write its raw probe, or run the shared scanner to
write semantic findings:

```bash
python scripts/imaging/hazards/pixelization.py
python scripts/misc/hazards/scan.py --subject likelihood
```
