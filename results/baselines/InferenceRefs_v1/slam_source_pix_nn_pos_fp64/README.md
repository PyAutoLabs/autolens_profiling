# InferenceRefs_v1 — `slam_source_pix_nn_pos_fp64`

Reference baseline for the `slam_source_pix_nn_pos_fp64` target
(`_targets.TARGETS["slam_source_pix_nn_pos_fp64"]` —
`imaging/slam_source_pix_nn/hst`, positions **on** (auto threshold, SLaM
convention), fp64), certified from task **t6** of SLURM array job **342091**
on RAL under PyAutoCortex ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md).

## Files

- `reference.json` — the certified run's full results JSON (schema v2, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`,
  `target_key`, `note`).
- `target.json` — the run's own schema-v2 `target` block
  (`target_id: sha256:8021b4b697ff`, `priors_ref: _targets.py@7f2f14765de7`).
  **Registry check: MATCHED.** Recomputing the block now from
  `scripts/misc/searches/_targets.py` for the registry key
  `slam_source_pix_nn_pos_fp64` — via
  `scripts/misc/searches/restamp_target_block.py` (report mode) and directly
  via `_targets.target_block(...)` with the row's own recorded `positions`
  block — reproduces `sha256:8021b4b697ff` and the same `priors_ref`.

## Provenance

- Source: `results/searches/nautilus/imaging/slam_source_pix_nn/hst/hpc_hpc_a100_fp64_ref_pos_tauto0.2_f1e8.json`
- Version: `2026.8.17.1`
- Run: job `342091` task `6`; mirror run dir
  `output/searches/nautilus/imaging/slam_source_pix_nn/hst/pos_tauto0.2_f1e8/hpc_a100_fp64_ref_pos_tauto0.2_f1e8/35b83ae5f6eab98575809e7c5dc5f3af`
- Sampler: `af.Nautilus`, `n_live=300`, `n_batch=100`, `seed=0`,
  `iterations_per_update=900`, `number_of_cores=1`, JAX vmap on, A100 fp64
- Positions: enabled, threshold mode `auto` (`auto_factor=3.0`,
  `auto_minimum_threshold=0.2`, resolved value `0.2`), factor `1e8`, source
  `simulator_truth_positions` (idealised)
- Headline: `max_log_likelihood=31405.63`, `log_evidence=31324.84`,
  `sampler_wall_s=4330.27` (`total_wall_s=4387.96`),
  `likelihood_evals=71100`, `posterior_samples=71079`,
  `delta_max_ll_vs_truth=-116.12`

## Why this row is citable

Certified by ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md),
which accepted the inference-refs-v1-redo wave **scoped to its four
positions-on rows**. The binding rule: *a mesh / pixelization run without a
`positions.info` file is unreliable and cannot be used or cited.* This row
carries `positions.info`; its positions-off twin (t5, same cell) does not and
sits 654.6 nats below it — the demagnified basin. Note the registry's standing
`delaunay_nn` caveat on this target (`_targets._DELAUNAY_NN_RESAMPLE_NOTE`):
an elevated resample rate at broad, untuned prior draws is a Phase-1 finding,
not a target-definition bug.
