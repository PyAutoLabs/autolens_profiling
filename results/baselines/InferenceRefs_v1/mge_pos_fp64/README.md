# InferenceRefs_v1 — `mge_pos_fp64`

Reference baseline for the `mge_pos_fp64` target
(`_targets.TARGETS["mge_pos_fp64"]` — `imaging/mge/hst`, positions **on**
(auto threshold), fp64), certified from task **9** of SLURM array job
**340210** on RAL under PyAutoCortex ruling
[R-20260901-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260901-01.md).
It is the positions-on companion to the existing positions-off `mge_fp64`
baseline in this directory.

**Ruling 2026-09-01, disk adoption 2026-09-02.** R-20260901-01 was passed on
2026-09-01 and directed the laptop actions — cite the run and `mv` its run dir
from `output/legacy/searches/…` back into the active tree on RAL **and** the
mirror. Those moves and this directory were both executed on **2026-09-02**
(hence `certified_at: "2026-09-02"` in `reference.json`), in the same
Phase-1 close-out as the four R-20260902-01 rows.

## Files

- `reference.json` — the certified run's full results JSON (schema v2, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`,
  `target_key`, `note`).
- `target.json` — the run's own schema-v2 `target` block
  (`target_id: sha256:6b93f0e52ecd`, `priors_ref: _targets.py@7f2f14765de7`).
  **Registry check: MATCHED.** Recomputing the block now from
  `scripts/misc/searches/_targets.py` for the registry key `mge_pos_fp64` —
  via `scripts/misc/searches/restamp_target_block.py` (report mode: 1
  unchanged, 0 refused) and directly via `_targets.target_block(...)` with the
  row's own recorded `positions` block — reproduces `sha256:6b93f0e52ecd` and
  the same `priors_ref`.

## Provenance

- Source: `results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64_ref_pos_tauto0.2_f1e8.json`
- Version: `2026.8.17.1`
- Run: job `340210` task `9` (2026-08-25); run dir
  `output/searches/nautilus/imaging/mge/hst/pos_tauto0.2_f1e8/hpc_a100_fp64_ref_pos_tauto0.2_f1e8/dc42087fb0524e78fd43eced7706b365`
  — moved back from `output/legacy/searches/…` into the active tree on RAL and
  on the mirror on 2026-09-02
- Sampler: `af.Nautilus`, `n_live=400`, `n_batch=64`, `seed=0`,
  `iterations_per_update=1200`, `number_of_cores=1`, JAX vmap on, A100 fp64
- Positions: enabled, threshold mode `auto` (`auto_factor=3.0`,
  `auto_minimum_threshold=0.2`, resolved value `0.2`), factor `1e8`, source
  `simulator_truth_positions` (idealised)
- Headline: `max_log_likelihood=31786.80`, `log_evidence=31690.42`,
  `sampler_wall_s=939.09` (`total_wall_s=1015.28`), `likelihood_evals=96704`,
  `posterior_samples=96704`, `delta_max_ll_vs_truth=+265.05`

## Why this row is citable

R-20260901-01 accepted the `mge_pos_ref_reuse` phase: the run was quarantined
under the 2026-08-31 REWIND but is **reusable** under the MGE reuse rule (MGE
rows were not implicated in the rewind), provided it is cited and its run dir
is returned to the active tree. Array task 9 stays excluded from any refs
rerun array, so this row is the reference for the cell and is not re-derived
by job 342091.

The mesh positions rule of
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md)
— *a mesh / pixelization run without a `positions.info` file is unreliable and
cannot be used or cited* — does not gate this row: `mge` is an analytic
(mesh-free) source target. The run is a positions-on `pos_tauto0.2_f1e8` arm
regardless, and carries its own `positions.info`.
