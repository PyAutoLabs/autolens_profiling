# InferenceRefs_v1

Named reference baselines for the Phase 1 targets registry (W4 / issue
#161, `scripts/misc/searches/_targets.py`). Each `<target_key>/` directory
holds `reference.json` (the certified run's results, verbatim, plus
certification metadata), `target.json` (the target's schema-v2 `target`
block, including `target_id`) and a `README.md` explaining the adoption.
See `results/notes/inference/PROGRAMME.md` §"Phase 1" and
§5 "Benchmark & result schema (v2)" for the design this directory
implements, and the per-target tolerances a run is judged against in
`results/notes/inference/targets/TOLERANCES.md`.

`INDEX.json` is the machine-readable form of the table below.

## Certified baselines

| Target key | Cell | Certified by | target_id |
|---|---|---|---|
| `mge_fp64` | `imaging/mge/hst` | retro | `sha256:770ccd47439d` |
| `delaunay_fp64` | `imaging/delaunay/hst` | retro | `sha256:6b016e5752ac` |
| `delaunay_nn_pos_fp64` | `imaging/delaunay_nn/hst` | R-20260902-01 | `sha256:1e007f224db6` |
| `slam_source_pix_pos_fp64` | `imaging/slam_source_pix/hst` | R-20260902-01 | `sha256:b29616db92cf` |
| `slam_source_pix_nn_pos_fp64` | `imaging/slam_source_pix_nn/hst` | R-20260902-01 | `sha256:8021b4b697ff` |
| `delaunay_pos_fp64` | `imaging/delaunay/hst` | R-20260902-01 | `sha256:dcbcb9627b78` |
| `mge_pos_fp64` | `imaging/mge/hst` | R-20260901-01 | `sha256:6b93f0e52ecd` |

The first two were adopted **retroactively** from existing same-stack
(`2026.8.17.1`) Nautilus runs rather than re-run for the Phase 1 commit —
see each target's own `README.md` for why. The `pixelization` target's
existing `2026.5.21.1` row was deliberately **not** adopted (it predates the
version-gap refresh the other two rows already got).

## 2026-09-02 — the four positions-on rows (R-20260902-01)

The `inference_refs_v1` array was submitted to RAL as job **342091** (10
tasks, all delivered). PyAutoCortex ruling
[R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md)
accepted the wave **scoped to its four positions-on rows** — tasks t2, t4, t6
and t10 — which are the last four entries in the table above. Its binding
rule: *a mesh / pixelization run without a `positions.info` file is
unreliable and cannot be used or cited*, so the six positions-off rows of
342091 (t0 `pixelization`, t1 `delaunay_nn`, t3 `slam_source_pix`, t5
`slam_source_pix_nn`, t7 `knn`, t8 `delaunay_matern`) are committed as result
data but are **not citable**, and their run dirs live under
`output/legacy_wrong/`. In every paired cell the positions-on run sits 520–700
nats *above* its positions-off twin; a penalty cannot raise a likelihood, so
the positions-off searches ended in the demagnified basin. All four adopted
rows' `target_id`s were re-derived from the current `_targets.py` registry and
matched (`restamp_target_block.py`, report mode: 4 unchanged, 0 refused).

## 2026-09-02 — `mge_pos_fp64` (R-20260901-01)

The positions-on MGE reference is **not** a 342091 row. PyAutoCortex ruling
[R-20260901-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260901-01.md)
(2026-09-01) accepted the `mge_pos_ref_reuse` phase: the quarantined-but-reusable
run **340210_9** (2026-08-25) is adopted as the `mge_pos_fp64` reference under
the MGE reuse rule, its run dir cited and moved from `output/legacy/searches/…`
back into the active tree on RAL and on the mirror. Array task 9 stays excluded
from any refs rerun array. Ruling 2026-09-01, **disk adoption 2026-09-02**; the
row's `target_id` was re-derived from the current `_targets.py` registry and
matched (`sha256:6b93f0e52ecd`; `restamp_target_block.py` report mode: 1
unchanged, 0 refused). MGE is a mesh-free target, so R-20260902-01's mesh
positions rule does not gate it.

## Still needed

The `hpc/batch_gpu/submit_search_nautilus_inference_refs_v1_array.sh` array
script **has been submitted** (job 342091, 2026-09-02); what remains is the
positions-on replacement wave for the three cells whose positions-off rows
R-20260902-01 struck — `pixelization_pos_fp64`, `knn_pos_fp64` and
`delaunay_matern_pos_fp64` — which need a **new phase**, not a rerun. See
`SUBMIT_LIST.md` rows 11–13 for the full list of reference rows this
directory still has no baseline for.
