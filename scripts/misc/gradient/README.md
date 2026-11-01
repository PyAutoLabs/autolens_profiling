# gradient

Likelihood lifecycle phase 6 (autolens_profiling#505): what a gradient evaluation of a
likelihood costs beside the forward call, and whether the gradient is correct.

- `harness.py` times the forward call and `jax.value_and_grad` (first call, steady single-call
  median, vmap per call), runs the pre-registered finite-difference check and writes the cell's
  `gradient.json` (`../tooling/gradient_witness.py`: the convention, the FD rule, the hazard rule
  and a `--check` validator).
- Each likelihood's `scripts/<dataset_type>/<variant>/gradient_breakdown.py` builds the cell's
  fused likelihood exactly as its runtime script does and calls the harness. The pilot is
  `scripts/imaging/mge/gradient_breakdown.py`.
- `build_catalogue.py` reads each `gradient.json` as a `measurement: gradient` record (headline
  `value_and_grad.single_call_s`), with the FD status, grad/forward and the hazard in its
  `gradient` block.

FD rule (human ruling 2026-11-01): central differences in fp64 along 3 seeded random unit
directions, at steps h and h/2 combined by Richardson extrapolation; pass when
`|FD - grad.v| / max(|grad.v|, 1e-8) <= 1e-4` in every direction. Hazard: value_and_grad over
forward above 4x (single call), or a failed FD check.

Tests: [`test/`](test/test_harness.py) (needs JAX).
