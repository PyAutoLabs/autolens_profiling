# InferenceRefs_v1

Named reference baselines for the Phase 1 targets registry (W4 / issue
#161, `scripts/misc/searches/_targets.py`). Each `<target_key>/` directory
holds `reference.json` (the certified run's results, verbatim, plus
certification metadata), `target.json` (the target's current schema-v2
`target` block, including `target_id`) and a `README.md` explaining the
adoption. See `results/notes/inference/PROGRAMME.md` §"Phase 1" and
§5 "Benchmark & result schema (v2)" for the design this directory
implements, and the per-target tolerances a run is judged against in
`results/notes/inference/targets/TOLERANCES.md`.

`INDEX.json` is the machine-readable form of the table below.

## Certified baselines

| Target key | Cell | Certified by | target_id |
|---|---|---|---|
| `mge_fp64` | `imaging/mge/hst` | retro | `sha256:770ccd47439d` |
| `delaunay_fp64` | `imaging/delaunay/hst` | retro | `sha256:6b016e5752ac` |

Both were adopted **retroactively** from existing same-stack (`2026.8.17.1`)
Nautilus runs rather than re-run for this Phase 1 commit — see each
target's own `README.md` for why. The `pixelization` target's existing
`2026.5.21.1` row was deliberately **not** adopted (it predates the
version-gap refresh the other two rows already got); a fresh
`pixelization off` reference row is listed in `SUBMIT_LIST.md` instead.

## Still needed

See `SUBMIT_LIST.md` for the full list of Nautilus fp64 reference rows this
directory does not yet have a baseline for, and
`hpc/batch_gpu/submit_search_nautilus_inference_refs_v1_array.sh` for the
(not-yet-submitted) SLURM array script that would produce them.
