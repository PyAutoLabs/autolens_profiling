# InferenceRefs_v1 — `knn_pos_fp64`

Reference baseline for the `knn_pos_fp64` target
(`_targets.TARGETS["knn_pos_fp64"]` — `imaging/knn/hst`, positions **on**
(auto threshold, `pos_tauto0.2_f1e8`), fp64), certified from task **t12** of
SLURM array job **342241** on RAL under PyAutoCortex ruling
[R-20260904-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260904-01.md).
It is the positions-on replacement for SUBMIT_LIST row 7 (`knn_fp64`, positions off), which
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md) struck.

## Files

- `reference.json` — the certified run's full results JSON (schema v2, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`,
  `target_key`, `note`).
- `target.json` — the run's own schema-v2 `target` block
  (`target_id: sha256:d06e54bad6c0`, `priors_ref: _targets.py@7f2f14765de7`).
  **Registry check: MATCHED.** Recomputing the block now from
  `scripts/misc/searches/_targets.py` via
  `scripts/misc/searches/restamp_target_block.py` (report mode, the three
  rows together: 0 changed, 3 unchanged, 0 refused) reproduces
  `sha256:d06e54bad6c0` and the same `priors_ref`.

## Provenance

- Source: `results/searches/nautilus/imaging/knn/hst/hpc_hpc_a100_fp64_ref_pos_tauto0.2_f1e8.json`
- Version: `2026.8.17.1`
- Run: job `342241` task `12`; mirror run dir
  `output/searches/nautilus/imaging/knn/hst/pos_tauto0.2_f1e8/hpc_a100_fp64_ref_pos_tauto0.2_f1e8/bbad3b14bf13a2bd0a4924bb5feb726c`
- Sampler: `af.Nautilus`, `n_live=300`, `n_batch=100`, `seed=0`,
  `iterations_per_update=900`, `number_of_cores=1`, JAX vmap on, A100 fp64
- Positions: enabled, threshold mode `auto` (`auto_factor=3.0`,
  `auto_minimum_threshold=0.2`, resolved value `0.2`), factor `1e8`, source
  `simulator_truth_positions` (idealised)
- Headline: `max_log_likelihood=30923.14`, `log_evidence=30838.39`,
  `sampler_wall_s=3257.46` (`total_wall_s=3315.64`),
  `likelihood_evals=67600`, `posterior_samples=67600`,
  `delta_max_ll_vs_truth=-598.60`

## Why this row is citable

Certified by ruling [R-20260904-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260904-01.md), which accepted the
refs-v1-positions-on-completion phase (Cortex phase 12). The binding rule of
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md) is met on its own terms: the run dir carries
`positions.info` and `.completed`, the positions penalty at the best point is
exactly `0.0` (the best model does not violate the threshold), and the
recovered Einstein radius is physical (`1.60`) — this is not the demagnified
basin. `pos_tauto0.2_f1e8` is that ruling's confirmed physical configuration
for a mesh source.

This cell's positions-off row (342091 t7) was struck by R-20260902-01 and lives in `output/legacy_wrong/`. R-20260902-01 also withdrew the old "stale-knn" reading, which compared this cell against a Phase-8B Prodigy arm that is itself in `legacy_wrong`; knn's standing is judged on this positions-on row and nothing else.
