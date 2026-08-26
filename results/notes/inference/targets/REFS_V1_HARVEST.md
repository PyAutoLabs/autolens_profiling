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

## Rows lost — cause 2: `reg.Adapt` on a Delaunay-family mesh (OPEN)

Tasks 5 and 6 (`imaging/slam_source_pix_nn`) died after ~52 s with
`jax.errors.TracerArrayConversionError`. The frames enter through the
**regularization**, not the mesh interpolator:

    linear_obj.py:171            regularization_matrix
      regularization/adapt.py:251      regularization_matrix_from
        mesh/mesh_geometry/delaunay.py:151   neighbors
          scipy/spatial/_qhull.pyx:1874      Delaunay.__init__

PyAutoArray documents the constraint in
`autoarray/inversion/regularization/constant.py`: on the Delaunay mesh family
(`Delaunay`, `DelaunayNN`, `KNearestNeighbor`, `KNNBarycentric`) the neighbors
come from a direct `scipy.spatial.Delaunay` call on the traced source-plane
mesh grid, so a non-split scheme cannot be traced — a split-family scheme must
be used there.

`_slam_source_pix_nn_model` (`scripts/misc/searches/_setup.py:1425`) pairs
`al.mesh.DelaunayNN` with `af.Model(al.reg.Adapt)` and is the only target that
breaks the rule. Every sibling obeys it: `delaunay_nn` is DelaunayNN +
`ConstantSplit` (row certified above), `knn` is KNearestNeighbor + free
`AdaptSplit`, and `slam_source_pix`'s `reg.Adapt` is safe only because the
rectangular family has analytic neighbors.

**This is a target-configuration error in this repo, not a library defect.**
**FIXED in this PR**: `_slam_source_pix_nn_model` now uses
`af.Model(al.reg.AdaptSplit)`. The model builds at 14 free parameters — the
same count the failed run reported — so the change is in the regularization's
form, not its dimensionality. The two reference rows still need a resubmit.

It forces a science call first. `_setup.py:1428` records the W4 intent as
"same free `al.reg.Adapt` regularization ... so the mesh choice is isolated
with the regularization scheme held fixed" — but `AdaptSplit` changes mesh AND
regularization, so the RTU-vs-DelaunayNN pair no longer isolates the mesh.
Either `slam_source_pix` gains a matching `AdaptSplit` variant, or the confound
is accepted and recorded on both targets' `notes`.

**Human call 2026-08-26: record the confound, do not fix it.** The pair was
never a clean mesh comparison anyway — `RectangularRTUAdaptImage` carries two
free adapt-weighting priors (`weight_power`, `weight_floor`) and `DelaunayNN`
carries none, so `slam_source_pix` is 16 free parameters against this cell's
14. The regularization was the matched half; the mesh never was. A matching
`AdaptSplit` variant of `slam_source_pix` would restore one axis and leave the
mesh gap, so it was judged not worth another reference row and A100 bake. Cite
the caveat with any RTU-vs-DelaunayNN reading.

**Pilot in flight.** `AdaptSplit` on a Delaunay-family mesh is the combination
`_delaunay_adapt_split_model` documents as "the cell where the NaN wall
actually lives", and this cell already had the registry's worst broad-prior
record even under `reg.Adapt` (1/8 finite, 2/8 NaN, 5/8 FitException). But all
of that was measured under GRADIENT search, where a NaN traps a lane; this
target runs under Nautilus, which rejects the draw instead, and no Nautilus run
on a free-AdaptSplit cell existed to settle it. So the positions-off row went
up alone as **RAL job 341908** rather than spending both rows at once. If it
converges the positions-on row follows; if it thrashes on resamples, that is a
Phase-1 finding to record — DECISIONS.md 2026-08-24 is explicit that
DelaunayNN's resample behaviour is a finding, not something to engineer
around.

## Status

InferenceRefs_v1 stands at 7 certified rows of the 13 targets in the registry
(plus the 2 retro-certified mge/delaunay rows recorded at W4 ship time).
Two more are a resubmit away; two are blocked on the Sibson fix.
