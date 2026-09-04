# InferenceRefs_v1 — still-needed reference rows

Nautilus fp64 reference rows the Phase 1 targets registry (W4 / issue #161)
still needs a certified `InferenceRefs_v1/<target_key>/` baseline for.
Every row below is a separate long-run Nautilus fit at `n_live >= 2x` the
`_samplers._N_LIVE` fiducial (`SEARCHES_NAUTILUS_N_LIVE`), seed 0, config
name `hpc_a100_fp64_ref` — driven by
`hpc/batch_gpu/submit_search_nautilus_inference_refs_v1_array.sh` (an array
job, one task per row below). **That script has NOT been submitted** — see
its own header for why (it is prep, not an executed campaign step; a human
decision to run it is a separate act from writing it). **Superseded
2026-09-02:** the array *was* submitted as job **342091** (all 10 tasks
delivered); see the ruling note below for which rows that leaves standing.

> **2026-09-02 — ruling [R-20260902-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260902-01.md) (autolens_profiling#201).**
> The positions-off mesh reference design is **retired**. A mesh / pixelization
> run without a `positions.info` file is unreliable and **not citable** — its
> search ends in the demagnified basin (job 342091: in every paired cell the
> positions-on run sits 520–700 nats *above* its positions-off twin). Rows
> **0, 1, 3, 5, 7, 8 are struck as designed**; their run dirs have been moved
> to `output/legacy_wrong/`. `pos_tauto0.2_f1e8` (threshold `auto`) is the
> confirmed physical configuration for mesh sources, so the three struck cells
> that have no positions-on row yet (`pixelization`, `knn`, `delaunay_matern`)
> reappear below as positions-on replacements needing a **new phase**, not a
> rerun.

> **2026-09-04 — ruling [R-20260904-01](https://github.com/PyAutoLabs/PyAutoCortex/blob/main/rulings/2026/09/R-20260904-01.md) (Cortex phase 12).**
> The replacement wave landed as job **342241**, array tasks **11–13**, and all
> three rows are accepted and adopted. **This list is closed**: every row that
> was not struck is certified, and `InferenceRefs_v1` stands at 9 baselines.
> The programme that commissioned it (`jax-inference-profiling`) is retired by
> the same check-in; new baselines are commissioned by its successor
> `gradient-slam-baseline` (`results/notes/gradient_slam/LEDGER.md`), not here.

| # | Target key | Cell | Positions | Threshold | Why |
|---|---|---|---|---|---|
| ~~0~~ | ~~`pixelization_fp64`~~ | ~~`imaging/pixelization/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; replaced by row 11 |
| ~~1~~ | ~~`delaunay_nn_fp64`~~ | ~~`imaging/delaunay_nn/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; row 2 is the positions-on row for this cell |
| 2 | `delaunay_nn_pos_fp64` | `imaging/delaunay_nn/hst` | on | fixed 0.3 | new target, positions arm — **LANDED 342091 — adopted 2026-09-02 into `InferenceRefs_v1/delaunay_nn_pos_fp64/` under R-20260902-01** |
| ~~3~~ | ~~`slam_source_pix_fp64`~~ | ~~`imaging/slam_source_pix/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; row 4 is the positions-on row for this cell |
| 4 | `slam_source_pix_pos_fp64` | `imaging/slam_source_pix/hst` | on | auto (SLaM convention) | new target, positions arm — **LANDED 342091 — adopted 2026-09-02 into `InferenceRefs_v1/slam_source_pix_pos_fp64/` under R-20260902-01** |
| ~~5~~ | ~~`slam_source_pix_nn_fp64`~~ | ~~`imaging/slam_source_pix_nn/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; row 6 is the positions-on row for this cell |
| 6 | `slam_source_pix_nn_pos_fp64` | `imaging/slam_source_pix_nn/hst` | on | auto (SLaM convention) | new target, positions arm — **LANDED 342091 — adopted 2026-09-02 into `InferenceRefs_v1/slam_source_pix_nn_pos_fp64/` under R-20260902-01** |
| ~~7~~ | ~~`knn_fp64`~~ | ~~`imaging/knn/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; replaced by row 12 |
| ~~8~~ | ~~`delaunay_matern_fp64`~~ | ~~`imaging/delaunay_matern/hst`~~ | ~~off~~ | ~~—~~ | **struck 2026-09-02 (R-20260902-01)** — positions-off mesh row, not citable; replaced by row 13 |
| 9 | `mge_pos_fp64` | `imaging/mge/hst` | on | auto | `mge_fp64` (off) is certified; the positions-on arm is not — **adopted 2026-09-02 into `InferenceRefs_v1/mge_pos_fp64/` under R-20260901-01** (run 340210_9, not a 342091 row) |
| 10 | `delaunay_pos_fp64` | `imaging/delaunay/hst` | on | auto | `delaunay_fp64` (off) is certified; the positions-on arm is not — **LANDED 342091 — adopted 2026-09-02 into `InferenceRefs_v1/delaunay_pos_fp64/` under R-20260902-01** |
| 11 | `pixelization_pos_fp64` | `imaging/pixelization/hst` | on | auto (`pos_tauto0.2_f1e8`) | positions-off row 0 struck; positions-on replacement — **LANDED 342241 t11 — adopted 2026-09-04 into `InferenceRefs_v1/pixelization_pos_fp64/` under R-20260904-01** |
| 12 | `knn_pos_fp64` | `imaging/knn/hst` | on | auto (`pos_tauto0.2_f1e8`) | positions-off row 7 struck; positions-on replacement — **LANDED 342241 t12 — adopted 2026-09-04 into `InferenceRefs_v1/knn_pos_fp64/` under R-20260904-01** |
| 13 | `delaunay_matern_pos_fp64` | `imaging/delaunay_matern/hst` | on | auto (`pos_tauto0.2_f1e8`) | positions-off row 8 struck; positions-on replacement — **LANDED 342241 t13 — adopted 2026-09-04 into `InferenceRefs_v1/delaunay_matern_pos_fp64/` under R-20260904-01** |

Threshold "auto" rows need `SEARCHES_POSITIONS_THRESHOLD=auto` explicitly —
the env-var-driven leaf-script path (these submits run the plain
`scripts/imaging/searches/nautilus/*.py` leaf scripts, not
`build_for_cell(target=...)`) does not read a target's own positions
convention, so the array script sets it directly per row. `#4`/`#6`
(`slam_source_pix(_nn)`) use auto because that is this target's own SLaM
convention (`_setup.build_for_cell`'s target-driven default, see
`DECISIONS.md` 2026-08-24); `#9`/`#10` (`mge`/`delaunay`) use auto because
that is the deliberate choice for these two reference rows specifically
(PROGRAMME.md Phase 1 spec), not because it is `mge`/`delaunay`'s general
default (which stays fixed 0.3 elsewhere, e.g. `delaunay_nn_pos_fp64` above
keeps the fixed default). Rows `#11`–`#13` use auto under
R-20260902-01: `pos_tauto0.2_f1e8` is the confirmed physical configuration for
a mesh source, so every mesh replacement row is a threshold-`auto` positions-on
run.

Every row that was going to be run has been run and adopted into
`InferenceRefs_v1/<target_key>/` following the `mge_fp64`/`delaunay_fp64`
pattern (`reference.json` + `target.json` + `README.md`), with
`INDEX.json`+`INDEX.md` updated. Nothing on this list is outstanding; it is
kept as the record of what was commissioned, struck and certified.

## 2026-08-31 — Phase 1 redo dispatch (rewind, #200)

**Job 342091**, `sbatch --array=0-8,10 --requeue
submit_search_nautilus_inference_refs_v1_array.sh` from
`hpc/batch_gpu/`. Ten of the eleven rows are resubmitted **fresh**: the
2026-08-31 rewind quarantined all pre-rewind mesh/pix evidence in
`output/legacy_wrong/` (never-cite side), so nothing from the 340210 /
341879 / 341908 waves may be reused for those rows. Tasks 5/6/7 run the
corrected `AdaptSplitPower` targets (see "RESUBMIT 2026-08-29" in the array
script header). At T+2 min: task 0 RUNNING on `euclid-ral-gpu-2`, tasks
1-8,10 PENDING (Resources).

**Task 9 (`mge_pos_fp64`) was deliberately excluded** under the rewind's
MGE reuse rule. Its earlier run **340210_9** COMPLETED 2026-08-25 (17:07
elapsed) and sits on the *reusable* side of the quarantine:

    output/legacy/searches/nautilus/imaging/mge/hst/
      pos_tauto0.2_f1e8/hpc_a100_fp64_ref_pos_tauto0.2_f1e8/
      dc42087fb0524e78fd43eced7706b365/

(identical on RAL and on the mirror
`/mnt/c/Users/Jammy/Science/inference_programme/output/legacy/...`; repo
artifact `results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64_ref_pos_tauto0.2_f1e8.json`).
It is an EXACT match to the refs spec — n_live 400 (2x the fiducial 200),
seed 0, config `hpc_a100_fp64_ref`, positions on, threshold auto (collapses
to the 0.2 floor), factor 1e8, version 2026.8.17.1, `.completed` present,
`priors_ref` `_targets.py@7f2f14765de7` which IS the current `_targets.py`
hash. Readout: sampler_wall **939.09 s**, maxLL **31786.7965**, logZ
**31690.4174**, **96,704** evals, kish_ess **7586.1**, target_id
`sha256:6b93f0e52ecd`. Adoption of this row is a human ruling, routed to the
batch 2026-08-31-pm review packet.

**Three rulings are pending in that packet:**

1. `mge_pos_fp64` — adopt the 340210_9 legacy row as the reference, or
   reject and queue array task 9 fresh.
2. `mge_fp64` — the retro-adopted baseline is OFF-SPEC against the refs
   standard (n_live 200 not 400, seed null not 0, config `hpc_a100_fp64`
   not `_ref`, `target.json` pins the STALE `_targets.py@bf2c8742c334`).
   Its source run `mge/hst/hpc_a100_fp64/181b13114ba3c2298191185ff74f90d8`
   is in `legacy/` (reusable). Keep as-is, or queue a fresh mge-off `_ref`
   run — for which **no array task exists**.
3. `delaunay_fp64` — the retro-adopted baseline is also off-spec (n_live
   150, seed null, stale `priors_ref` bf2c8742c334) AND its source run
   `delaunay/hst/hpc_a100_fp64` is in **`legacy_wrong/`** (never-cite), so
   it rests on rejected evidence. There is **no array task** for a
   delaunay positions-off ref run; adding one (task 11) is an option.

**Task 10 footnote.** 340210_10 (`delaunay_pos_fp64`) COMPLETED and is
spec-conformant (n_live 300, seed 0, current `priors_ref`, maxLL
31338.9667, logZ 31264.8427, wall 3206.5 s) — but it sits in
`legacy_wrong/` and is therefore doctrine-blocked from reuse. A fresh run
was dispatched as **342091_10**. Recorded so the cost of the doctrine call
is visible.

> **Historical.** Everything the 2026-08-31 section above lists as pending has
> since been ruled: `mge_pos_fp64` adopted (R-20260901-01), `mge_fp64` kept
> (R-20260902-01/R-20260901-02), `delaunay_fp64` dropped (R-20260901-03) and
> replaced by `delaunay_pos_fp64`, and rows 11-13 certified (R-20260904-01).
> The section is kept as the dispatch record, not as an open queue.
