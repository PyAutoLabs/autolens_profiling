# InferenceRefs_v1 — first reference-baseline harvest (W4, issue #161)

RAL array job **340210** (11 tasks, `submit_search_nautilus_inference_refs_v1_array.sh`),
submitted 2026-08-24, harvested 2026-08-26. Nautilus, fp64, `n_live=300`
(2x the 150 fiducial), A100.

**7 of 11 rows landed.** Four tasks died, for two distinct reasons — both
recorded below because both cost reference rows, and one of them is a library
defect that is still open.

## Rows certified

| cell | config | wall | logZ | maxL | target_id |
|---|---|---:|---:|---:|---|
| imaging/mge/hst | ref_pos_tauto0.2_f1e8 | 1,015 s | 31690.42 | 31786.80 | `sha256:bf3d096fda76` |
| imaging/delaunay/hst | ref_pos_tauto0.2_f1e8 | 3,265 s | 31264.84 | 31338.97 | `sha256:6d52f9dcfbd0` |
| imaging/delaunay_nn/hst | ref | 3,230 s | 30591.09 | 30650.77 | `sha256:6a13b9a4e64b` |
| imaging/delaunay_nn/hst | ref_pos_t0.3_f1e8 | 3,111 s | 31275.23 | 31347.88 | `sha256:1e007f224db6` |
| imaging/pixelization/hst | ref | 6,535 s | 29590.70 | 29670.33 | `sha256:801ba27b970d` |
| imaging/slam_source_pix/hst | ref | 5,663 s | 30718.02 | 30809.46 | `sha256:37150b628da1` |
| imaging/slam_source_pix/hst | ref_pos_tauto0.2_f1e8 | 5,387 s | 31305.06 | 31411.38 | `sha256:b29616db92cf` |

All seven carry version stamp 2026.8.17.1, walls consistent with their cell's
recorded cost, and no "Fit Already Completed" line — the three checks
`../PROGRAMME.md` requires before an artifact is trusted.

## Rows lost — cause 1: missing `_N_LIVE` presets (FIXED here)

Tasks 7 (`imaging/knn`) and 8 (`imaging/delaunay_matern`) raised
`KeyError: No n_live preset for (...)` from `n_live_for`
(`scripts/misc/searches/_samplers.py`, via `_runner.py:462`) and died in ~3 s
each. Both cells were reachable from the submit list but had never been given
a row.

Fixed in this change: both added at the 150 fiducial every other imaging mesh
cell uses. **These two rows need a resubmit** — they are not in the table
above.

## Rows lost — cause 2: Sibson mesh is not jit-safe (OPEN)

Tasks 5 and 6 (`imaging/slam_source_pix_nn`) died after ~52 s with
`jax.errors.TracerArrayConversionError`: `scipy.spatial.Delaunay.__init__`
called on a traced `float64[1500,2]`, traced from
`autofit/non_linear/fitness.py:205`. Call site is
`PyAutoArray/autoarray/inversion/mesh/interpolator/sibson.py:555`.

This is a library defect, not a submit error — filed in PyAutoMind as
`draft/bug/autoarray/sibson_mesh_scipy_delaunay_under_jit.md`. Note that plain
`delaunay_nn` completed normally in the same array (rows above), because
`interpolator/delaunay.py:170` carries a JAX point-location routine that
Sibson lacks. **`slam_source_pix_nn` cannot be certified until that is fixed.**

## Status

InferenceRefs_v1 stands at 7 certified rows of the 13 targets in the registry
(plus the 2 retro-certified mge/delaunay rows recorded at W4 ship time).
Two more are a resubmit away; two are blocked on the Sibson fix.
