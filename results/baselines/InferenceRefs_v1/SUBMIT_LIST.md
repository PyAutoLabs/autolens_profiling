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
