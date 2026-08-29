# InferenceRefs_v1 — first reference-baseline harvest (W4, issue #161)

RAL array job **340210** (11 tasks, `submit_search_nautilus_inference_refs_v1_array.sh`),
submitted 2026-08-24, harvested 2026-08-26. Nautilus, fp64, `n_live=300`
(2x the 150 fiducial), A100.

**7 of 11 rows landed.** Four tasks died, for two distinct reasons — both
recorded below because both cost reference rows.

**Update 2026-08-27 (RAL 341879, tasks 7/8):** the two rows lost to cause 1
have landed, taking InferenceRefs_v1 to **9 certified rows of 13 targets**.
The `slam_source_pix_nn` pilot (341908) came back with an answer of its own:
it thrashes. Both are recorded below.

**CORRECTION 2026-08-29 (autolens_profiling#196) — the pilot answer above is
WRONG, and the knn row below is suspect.** 341908_5 did not thrash and did not
make zero calls; it made **90,000**, and was killed by a likelihood-overflow
flood in the lambda^4 `AdaptSplit` regularization. The knn row's 480-nat
deficit is the same pathology at lower amplitude. Both are detailed in the
"Pilot answer" section below and in DECISIONS.md 2026-08-29. **Nine rows are
certified; eight of them stand.** The knn row is withdrawn as a bar pending a
re-run, and every adapt-split row here is on the far side of a target/library
stack boundary from anything measured after 2026-08-29.

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
| imaging/knn/hst (341879_7) — **WITHDRAWN 2026-08-29, do not score against it** | ref | 3,343.6 s | 30010.170 | 30077.028 | `sha256:84c0d88d3032` (legacy class) |
| imaging/delaunay_matern/hst (341879_8) | ref | 3,351.7 s | 30614.972 | 30676.594 | `sha256:3f17a37225f9` |

All nine carry version stamp 2026.8.17.1, walls consistent with their cell's
recorded cost, and no resume marker — the three provenance checks
`../PROGRAMME.md` §3 requires before an artifact is trusted. Both marker
forms are checked: `Fit Already Completed` (nested searches) and
`Resuming .* previous samples found` (MultiStartGradient); the second was
added to the rule on 2026-08-27 after a 68 s silent Prodigy resume passed
the single-string check (RAL 341892 task 0).

The two 2026-08-27 rows ran Nautilus `n_live=300`, fp64, A100; the walls
quoted are `total_wall_s` (sampler walls 3,287.4 s and 3,299.3 s), Kish ESS
**4,848.8** (knn) and **5,728.2** (delaunay_matern).

**What "certified" does and does not mean.** It is these three human
provenance checks and nothing more: there is **no certification function and
no coded tolerance** anywhere in the framework. Two known blemishes ride
with the table: the **mge row ran `n_live=400`**, not the campaign's 300,
and sits **+265 nats** from its truth bar against the 2-nat registry
tolerance; and `results/baselines/InferenceRefs_v1/INDEX.json` **still lists
only the 2 retro-certified rows**, so the index is not the registry's state.

**FLAG — do not score against the knn reference.** Its maxL (30077.028)
sits **480 nats below** a same-`target_id` Phase 8B Prodigy `log_reg` arm
(30557.03 — `../phase_08_regularization/RESULTS.md` "8B — run history and
harvest"). A reference row that a gradient searcher beats by 480 nats is not
a bar; the discrepancy must be explained before this row is used as one.

## Rows lost — cause 1: missing `_N_LIVE` presets (FIXED here)

Tasks 7 (`imaging/knn`) and 8 (`imaging/delaunay_matern`) raised
`KeyError: No n_live preset for (...)` from `n_live_for`
(`scripts/misc/searches/_samplers.py`, via `_runner.py:462`) and died in ~3 s
each. Both cells were reachable from the submit list but had never been given
a row.

Fixed in this change: both added at the 150 fiducial every other imaging mesh
cell uses. ~~These two rows need a resubmit~~ — **resubmitted as RAL 341879
tasks 7/8 and certified 2026-08-27**; both are in the table above (they ran
at `n_live=300`, the campaign preset, in ~56 min each).

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

**Pilot answer 2026-08-27: "it thrashes".** 341908 compiled in 20 s and its
`.out` then sat at `Calls | 0` for six hours before hitting the wall-clock
limit with nothing to harvest. Read at the time as: free `AdaptSplit` on a
Delaunay-family mesh is not merely a gradient hazard, it is unaffordable under
nested sampling too.

**CORRECTED 2026-08-29 (#196): that reading was wrong in every part.** The
`checkpoint.hdf5` — the only artifact the run left — records:

| | |
|---|---|
| likelihood calls | **90,000** (not 0) |
| bounds | 29 |
| `explored` | **FALSE** |
| max log L | **30,701.3** (ABOVE the certified `delaunay_nn` ref, 30,650.77) |
| per-eval | 0.239 s |
| MaxRSS | 3.66 GB |
| exit | TIMEOUT at the 6 h wall |
| NaN / -inf draws | **zero of each** |

Three separate things had to be true for that to read as "0 calls / thrashes",
and all three are now fixed:

1. **The `.out` was empty because stdout was block-buffered.** A SLURM `.out`
   is a file, not a tty, so Python buffers at 8 KiB; every progress line the
   driver printed was in an unflushed buffer that the wall-clock kill
   discarded. `activate.sh` now exports `PYTHONUNBUFFERED=1` inside its
   `$SLURM_JOB_ID` block, covering every submit at once.
2. **It was not thrashing on resamples.** Zero NaN and zero `-inf` draws: there
   was nothing to resample. The DelaunayNN resample hazard the 2026-08-24 entry
   anticipated did not fire.
3. **It was drowning in a likelihood-overflow flood.** `AdaptSplit` squares its
   coefficient twice, so under `LogUniform(1e-6, 1e6)` the regularization term
   reaches 1e24; the curvature-plus-regularization matrix goes non-PD from
   c ~ 1e4, and the fp64 Cholesky there returns **finite garbage** rather than
   NaN. `log_l` up to **3e+303** appears in shells 14/23/24/26/28. PyAutoFit's
   `Fitness` passed the finite value through, Nautilus accepted it as the best
   point, `shell_log_l` reached **1e56** at `shell_n_eff` ~ 1, and `f_live`
   could never fall below its termination threshold. The run could only ever
   end at the wall clock. A NaN would have been rejected by every search in the
   stack; a finite 3e+303 was accepted.

**The cell is affordable. The objective was broken.** It reached a maximum
likelihood above the certified `delaunay_nn` reference in 6 h — this is not an
unaffordable target, it is a target that could not be told it had converged.

**What changed (all merged 2026-08-29).** `al.reg.AdaptSplitPower` squares the
coefficient once (`power` is a `Constant`, never sampled, so the model
dimension is unchanged); `_setup._free_adapt_split` caps the coefficient priors
at `LogUniform(1e-6, 1e4)`, below the measured non-PD onset; PyAutoFit's
`Fitness` rejects implausibly large finite log-likelihoods
(`general.test.log_likelihood_ceiling`, default 1e20) as a backstop.

**This changes the `target_id`s**, so the corrected cells are NEW targets:

| target | before | after |
|---|---|---|
| `knn_fp64` | `sha256:84c0d88d3032` | `sha256:ccafb8b191bc` |
| `knn_pos_fp64` | `sha256:f2bebfcc525f` | `sha256:d06e54bad6c0` |
| `slam_source_pix_nn_fp64` | `sha256:1721493bba6b` | `sha256:ad291b57fc62` |
| `slam_source_pix_nn_pos_fp64` | `sha256:6befb71d64ee` | `sha256:8021b4b697ff` |

The recorded rows keep their OLD ids — they really did run the legacy target,
and re-stamping them would erase exactly the boundary this correction draws.
`restamp_target_block.py` refuses them of its own accord (its reproduction
control fires: the environment computes the same id both the old way and the
corrected way, and neither matches the row), which is the right answer.

**Consequences.** Tasks 5 and 6 go back up on the corrected target
(`--array=5,6,7`, after `HPCPullPyAuto`). Task 7 (`knn`) joins them: its
certified row's maxL sits 480 nats below a same-`target_id` Prodigy arm, which
is the same overflow signature at lower amplitude, so it is **withdrawn as a
bar** rather than trusted. `slam_source_pix_nn` stays uncertified until its new
rows land. DECISIONS.md 2026-08-29.

## Status

InferenceRefs_v1 stands at **9 certified rows of the 13 targets** in the
registry as of 2026-08-27 (the 7 from 340210 plus knn and delaunay_matern
from 341879; the 2 retro-certified mge/delaunay rows recorded at W4 ship
time are counted separately and are the only rows `INDEX.json` lists).

The remaining gap is the `slam_source_pix_nn` pair. It was never a library
(Sibson) defect — it is the target-configuration error fixed above (commit
512b805) — and the fixed target's pilot thrashed, so the two rows are
blocked on a science decision about the target, not on a resubmit.

**Superseded 2026-08-29 (#196).** The pilot did not thrash (see the correction
above), so the `slam_source_pix_nn` pair was never blocked on a science
decision — it is a resubmit, and it is queued. The count moves to **8 rows that
stand of 13 targets**: the knn row is withdrawn as a bar until it is re-run,
and three tasks (`--array=5,6,7`) go back up on the corrected targets once
`HPCPullPyAuto` has landed the post-2026-08-29 stack on RAL. Those three rows
will carry NEW `target_id`s and belong to the post-boundary registry; the eight
that stand are pre-boundary and stay comparable only with each other.
