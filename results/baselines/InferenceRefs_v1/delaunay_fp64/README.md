# InferenceRefs_v1 — `delaunay_fp64`

Reference baseline for the `delaunay_fp64` target
(`_targets.TARGETS["delaunay_fp64"]` — `imaging/delaunay/hst`, positions
off, fp64), adopted **retroactively** (`certified_by: "retro"` in
`reference.json`) from an existing Nautilus run rather than a fresh
long-run re-fit.

## Files

- `reference.json` — the adopted run's full results JSON (v1 schema, copied
  verbatim under the `reference` key) plus certification metadata
  (`certified_by`, `certified_at`, `certifying_git_sha`, `source_artifact`).
- `target.json` — the schema-v2 `target` block for `delaunay_fp64`, computed
  **now** from the current `scripts/misc/searches/_targets.py` registry
  (`target_id: sha256:6b016e5752ac`). The adopted run predates `target_id`
  and carries none of its own; this file is what a future run's `target_id`
  should be compared against to confirm it is targeting the same thing.

## Provenance

- Source: `results/searches/nautilus/imaging/delaunay/hst/hpc_hpc_a100_fp64.json`
- Version: `2026.8.17.1`
- Sampler: `af.Nautilus`, `n_live=150`, `n_batch=16`, seed unset (library
  default)
- Headline: `max_log_likelihood=30623.20`, `log_evidence=30562.20`,
  `sampler_wall_s=1891.32`, `likelihood_evals=30240`

## Why retroactive adoption (not a fresh reference re-fit), and why NOT the May pixelization row

This Delaunay run is a same-stack (`2026.8.17.1`) re-baseline explicitly
produced to refresh the inference programme's Nautilus truth bar after the
fork-era vs current-stack version gap (`hpc/batch_gpu/
submit_search_nautilus_imaging_delaunay_a100_hst_fp64`'s own docstring: "the
Nautilus truth bars the inference programme compares every sampler against
are fork-era rows... re-running the trio on the current stack refreshes all
three truth bars at v2026.8.x with the current artifact schema"). It is
exactly the kind of row this baseline directory exists to certify.

By contrast, `results/searches/nautilus/imaging/pixelization/hst/
hpc_a100_fp64.json` is version `2026.5.21.1` — three months older, from
BEFORE the same version-gap refresh the Delaunay/MGE rows already got. It is
deliberately **not** adopted here; `SUBMIT_LIST.md` lists `pixelization off`
as a still-needed fresh reference row rather than retroactively certifying
the stale one.
