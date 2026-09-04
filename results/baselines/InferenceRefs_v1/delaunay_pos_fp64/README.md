# InferenceRefs_v1 — `delaunay_pos_fp64`

Reference baseline for the `delaunay_pos_fp64` target
(`_targets.TARGETS["delaunay_pos_fp64"]` — `imaging/delaunay/hst`, positions
**on** (auto threshold), fp64), certified from task **t10** of SLURM array
job **342091** on RAL under PyAutoCortex ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md).
It is the positions-on companion to the existing positions-off
`delaunay_fp64` baseline in this directory.

## Files

- `reference.json` — the certified run's full results JSON (schema v2, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`,
  `target_key`, `note`).
- `target.json` — the run's own schema-v2 `target` block
  (`target_id: sha256:dcbcb9627b78`, `priors_ref: _targets.py@7f2f14765de7`).
  **Registry check: MATCHED.** Recomputing the block now from
  `scripts/misc/searches/_targets.py` for the registry key
  `delaunay_pos_fp64` — via `scripts/misc/searches/restamp_target_block.py`
  (report mode) and directly via `_targets.target_block(...)` with the row's
  own recorded `positions` block — reproduces `sha256:dcbcb9627b78` and the
  same `priors_ref`.

## Provenance

- Source: `results/searches/nautilus/imaging/delaunay/hst/hpc_hpc_a100_fp64_ref_pos_tauto0.2_f1e8.json`
- Version: `2026.8.17.1`
- Run: job `342091` task `10`; mirror run dir
  `output/searches/nautilus/imaging/delaunay/hst/pos_tauto0.2_f1e8/hpc_a100_fp64_ref_pos_tauto0.2_f1e8/ebdb103586dc39c77e805aa75c9881b9`
- Sampler: `af.Nautilus`, `n_live=300`, `n_batch=16`, `seed=0`,
  `iterations_per_update=900`, `number_of_cores=1`, JAX vmap on, A100 fp64
- Positions: enabled, threshold mode `auto` (`auto_factor=3.0`,
  `auto_minimum_threshold=0.2`, resolved value `0.2`), factor `1e8`, source
  `simulator_truth_positions` (idealised)
- Headline: `max_log_likelihood=31338.43`, `log_evidence=31264.83`,
  `sampler_wall_s=3141.26` (`total_wall_s=3197.64`),
  `likelihood_evals=50832`, `posterior_samples=50832`,
  `delta_max_ll_vs_truth=-183.31`

## Why this row is citable

Certified by ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md),
which accepted the inference-refs-v1-redo wave **scoped to its four
positions-on rows**. The binding rule: *a mesh / pixelization run without a
`positions.info` file is unreliable and cannot be used or cited.* This cell
has no positions-off twin in 342091 — its positions-off baseline is the
separately certified `delaunay_fp64` row — but the same rule applies: this is
the arm that carries `positions.info`, and `pos_tauto0.2_f1e8` is the ruling's
confirmed physical configuration for a mesh source.
