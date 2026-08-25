# InferenceRefs_v1 — still-needed reference rows

Nautilus fp64 reference rows the Phase 1 targets registry (W4 / issue #161)
still needs a certified `InferenceRefs_v1/<target_key>/` baseline for.
Every row below is a separate long-run Nautilus fit at `n_live >= 2x` the
`_samplers._N_LIVE` fiducial (`SEARCHES_NAUTILUS_N_LIVE`), seed 0, config
name `hpc_a100_fp64_ref` — driven by
`hpc/batch_gpu/submit_search_nautilus_inference_refs_v1_array.sh` (an array
job, one task per row below). **That script has NOT been submitted** — see
its own header for why (it is prep, not an executed campaign step; a human
decision to run it is a separate act from writing it).

| # | Target key | Cell | Positions | Threshold | Why |
|---|---|---|---|---|---|
| 0 | `pixelization_fp64` | `imaging/pixelization/hst` | off | — | existing row is `2026.5.21.1`, pre-refresh (see `delaunay_fp64/README.md`) — not adopted |
| 1 | `delaunay_nn_fp64` | `imaging/delaunay_nn/hst` | off | — | new target (W4), no prior artifact at all |
| 2 | `delaunay_nn_pos_fp64` | `imaging/delaunay_nn/hst` | on | fixed 0.3 | new target, positions arm |
| 3 | `slam_source_pix_fp64` | `imaging/slam_source_pix/hst` | off | — | new target, no prior artifact |
| 4 | `slam_source_pix_pos_fp64` | `imaging/slam_source_pix/hst` | on | auto (SLaM convention) | new target, positions arm |
| 5 | `slam_source_pix_nn_fp64` | `imaging/slam_source_pix_nn/hst` | off | — | new target, no prior artifact |
| 6 | `slam_source_pix_nn_pos_fp64` | `imaging/slam_source_pix_nn/hst` | on | auto (SLaM convention) | new target, positions arm |
| 7 | `knn_fp64` | `imaging/knn/hst` | off | — | registered target, no Nautilus reference row exists (only MultiStart* history) |
| 8 | `delaunay_matern_fp64` | `imaging/delaunay_matern/hst` | off | — | registered target, no Nautilus reference row exists |
| 9 | `mge_pos_fp64` | `imaging/mge/hst` | on | auto | `mge_fp64` (off) is certified; the positions-on arm is not |
| 10 | `delaunay_pos_fp64` | `imaging/delaunay/hst` | on | auto | `delaunay_fp64` (off) is certified; the positions-on arm is not |

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
keeps the fixed default).

Once run, each row's artifact should be adopted into
`InferenceRefs_v1/<target_key>/` following the `mge_fp64`/`delaunay_fp64`
pattern (`reference.json` + `target.json` + `README.md`), and this file's
row removed / `INDEX.json`+`INDEX.md` updated.
