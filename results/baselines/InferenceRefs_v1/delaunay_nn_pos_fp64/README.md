# InferenceRefs_v1 — `delaunay_nn_pos_fp64`

Reference baseline for the `delaunay_nn_pos_fp64` target
(`_targets.TARGETS["delaunay_nn_pos_fp64"]` — `imaging/delaunay_nn/hst`,
positions **on** (fixed threshold 0.3), fp64), certified from task **t2** of
SLURM array job **342091** on RAL under PyAutoCortex ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md).

## Files

- `reference.json` — the certified run's full results JSON (schema v2, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`,
  `target_key`, `note`).
- `target.json` — the run's own schema-v2 `target` block
  (`target_id: sha256:1e007f224db6`, `priors_ref: _targets.py@7f2f14765de7`).
  **Registry check: MATCHED.** Recomputing the block now from
  `scripts/misc/searches/_targets.py` for the registry key
  `delaunay_nn_pos_fp64` — via `scripts/misc/searches/restamp_target_block.py`
  (report mode) and directly via `_targets.target_block(...)` with the row's own
  recorded `positions` block — reproduces `sha256:1e007f224db6` and the same
  `priors_ref`, so the row targets exactly what the registry defines today.

## Provenance

- Source: `results/searches/nautilus/imaging/delaunay_nn/hst/hpc_hpc_a100_fp64_ref_pos_t0.3_f1e8.json`
- Version: `2026.8.17.1`
- Run: job `342091` task `2`; mirror run dir
  `output/searches/nautilus/imaging/delaunay_nn/hst/pos_t0.3_f1e8/hpc_a100_fp64_ref_pos_t0.3_f1e8/112548c8d02373f3abf470ecdaa3a941`
- Sampler: `af.Nautilus`, `n_live=300`, `n_batch=100`, `seed=0`,
  `iterations_per_update=900`, `number_of_cores=1`, JAX vmap on, A100 fp64
- Positions: enabled, threshold mode `fixed`, value `0.3`, factor `1e8`,
  source `simulator_truth_positions` (idealised — truth positions, not
  re-solved from a completed search)
- Headline: `max_log_likelihood=31351.39`, `log_evidence=31275.23`,
  `sampler_wall_s=3113.59` (`total_wall_s=3170.11`),
  `likelihood_evals=50900`, `posterior_samples=50900`,
  `delta_max_ll_vs_truth=-170.35`

## Why this row is citable

Certified by ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md),
which accepted the inference-refs-v1-redo wave **scoped to its four
positions-on rows**. The binding rule: *a mesh / pixelization run without a
`positions.info` file is unreliable and cannot be used or cited.* This row
carries `positions.info`; its positions-off twin (t1, same cell) does not and
sits 700.6 nats below it — a penalty cannot raise a likelihood, so the
positions-off search ended in the lower, demagnified basin. Task t2 is the
one fixed-threshold row of the accepted four; the other three use
`pos_tauto0.2_f1e8`, the ruling's confirmed physical configuration for mesh
sources.
