# InferenceRefs_v1 — `mge_fp64`

Reference baseline for the `mge_fp64` target (`_targets.TARGETS["mge_fp64"]`
— `imaging/mge/hst`, positions off, fp64), adopted **retroactively**
(`certified_by: "retro"` in `reference.json`) from an existing Nautilus
run rather than a fresh long-run re-fit.

## Files

- `reference.json` — the adopted run's full results JSON (v1 schema, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`).
- `target.json` — the schema-v2 `target` block for `mge_fp64`, computed
  **now** from the current `scripts/misc/searches/_targets.py` registry
  (`target_id: sha256:770ccd47439d`). The adopted run predates `target_id`
  and carries none of its own; this file is what a future run's `target_id`
  should be compared against to confirm it is targeting the same thing.

## Provenance

- Source: `results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64.json`
- Version: `2026.8.17.1`
- Sampler: `af.Nautilus`, `n_live=200`, `n_batch=64`, seed unset (library
  default)
- Headline: `max_log_likelihood=31786.63`, `log_evidence=31690.50`,
  `sampler_wall_s=706.50`, `likelihood_evals=62208`

## Why retroactive adoption (not a fresh reference re-fit)

This run already exists, was produced on the current stack (`2026.8.17.1`,
same major version as this Phase-1 commit), and reproduces the Nautilus
truth bar the inference programme has used as its MGE reference throughout
Phase 0-3 (`PROGRAMME.md` §1-3, `DECISIONS.md` 2026-08-24 Gate A entry:
"Nautilus truth bars reaffirmed to 2 dp; the 523 s Nautilus wall is retired
as a reference (707-773 s reproduces)"). Re-deriving a fresh reference at
this stage would duplicate that history rather than add to it. A future
Phase-1 refresh (see `SUBMIT_LIST.md` for the still-needed rows) should
still run at >= 2x this row's `n_live` for a tighter reference posterior.
