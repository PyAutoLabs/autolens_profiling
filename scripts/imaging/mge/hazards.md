# MGE imaging likelihood hazards

`hazards_nnls_capture.py` captures the positive-only linear systems the JAX
likelihood hands to `reconstruction_positive_only_from` for the SLaM
`source_lp[1]` MGE model (2 x 20 lens Gaussians + 20 source Gaussians, free
Isothermal + ExternalShear) at 48 seeded near-truth vectors on the HST dataset,
and records whether the Jacobi-preconditioned PDIP solve converges at caps 50
and 200 against `fnnls_cholesky` (PyAutoArray#571). It writes
`results/hazards/component/mge/nnls_capture_slam_hst_v<version>.json` plus an
8-system `.npz` that PyAutoArray uses as a regression fixture. The unlabelled
JSON is the pre-fix capture (14/48 unconverged at cap 50, 5/48 at cap 200).
`--label postfix` re-runs it against the fixed library (mapper-less inversions
use the raw-forward PDIP mode) and writes
`nnls_capture_slam_hst_v<version>_postfix.json`: 0/48 unconverged and a JAX vs
NumPy log-likelihood difference of at most 6e-7 on all 48 vectors.

Run from the repository root:

```bash
python scripts/imaging/mge/hazards_nnls_capture.py --label postfix
```

Dataset-free MGE mass-deflection probes are documented in the
[shared hazards guide](../../misc/hazards/README.md); they are not findings
about this MGE light-model likelihood.
