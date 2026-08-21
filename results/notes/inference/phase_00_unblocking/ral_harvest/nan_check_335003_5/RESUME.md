# multistart-nan-step-diagnostics — verification checkpoint

Written 2026-08-15 ~02:20 BST. Verifies the merged feature
`PyAutoMind/complete/2026/08/multistart-nan-step-diagnostics.md`
(PyAutoFit PR#1473 `fbfcece3a`, autolens_profiling PR#127 `a34d619`).

## ANSWERED ALREADY — the headline result

`search.summary` DOES carry the new block. Verified on disk, RAL, against
merged `main`. Job 335002, `delaunay_matern`, 10 steps, 745 s:

```
Total Samples = 17
Resurrections = 0
Value-NaN Lane-Steps = 0
Gradient-NaN Lane-Steps = 0
Value-NaN Lane-Step Rate = 0.0
Gradient-NaN Lane-Step Rate = 0.0
Time To Run = 0:11:41.306727
```

Cross-checked against `files/samples_info.json`, NOT just read off the summary:
`n_value_nan_lane_steps=0`, `n_grad_nan_lane_steps=0`, `n_resurrections=0`,
`n_starts=16`, `total_steps=10` -> `lane_steps=160`, both rates recompute to
`0.0`. Part 1 (persistence) and Part 2 (normalised rates) both real.

Zeros are expected at 160 lane-steps with `alive 16/16` throughout. It proves
the block EMITS; it does not prove it COUNTS. That is what the overnight arms
are for.

## RUNNING OVERNIGHT — do these first thing

Three 500-step arms, submitted 02:10 BST, 24 h wall, all on
`euclid-ral-compute-1`, partition `ral` (CPU). ~59 s/step measured => ~8 h each,
so expect completion ~10:00-11:00 BST.

| Job | Arm | Mesh + reg | Prediction |
|---|---|---|---|
| 335003 | `scripts/imaging/searches/multi_start_prodigy/delaunay.py` | Delaunay + free Matern | gradient-NaN |
| 335004 | `nan_check/delaunay_plain.py` (ad-hoc leaf) | Delaunay + ConstantSplit | gradient-NaN |
| 335005 | `scripts/imaging/searches/multi_start_prodigy/knn.py` | Wendland-C4 KNN + free AdaptSplit | value-NaN |

The split IS the experiment: if both Delaunay arms show gradient-NaN and KNN
shows only value-NaN, the counter is reading mesh geometry rather than
regularization. Mechanism (see the memory note): Delaunay's
`area_weights = areas_factor * jnp.sqrt(areas)` in
`autoarray/inversion/mesh/interpolator/delaunay.py:323` — a degenerate/flipping
triangle sends a vertex dual area to 0, `sqrt(0)` is a finite `0.0` with an
`inf` derivative, `inf * 0` -> NaN gradient on a finite likelihood. KNN never
touches `barycentric_dual_area_from`.

### Check status

```bash
ssh euclid_jump
squeue -u jnightin -o "%.10i %.24j %.8T %.10M %.12L %R"
for j in 335003 335004 335005; do
  echo "--- $j ---"
  grep -o "prodigy step [0-9]*/[0-9]*.*" /mnt/ral/jnightin/nan_check/output/output.$j.out | tail -1
done
```

### Read the results

Search output root (autofit), config-name `cpu_nan_check`:

```
/mnt/ral/jnightin/autolens_profiling_delaunay/output/searches/multi_start_prodigy/imaging/<model>/hst/cpu_nan_check/<hash>/
```
where `<model>` is `delaunay_matern`, `delaunay`, `knn`.

```bash
AP=/mnt/ral/jnightin/autolens_profiling_delaunay/output/searches/multi_start_prodigy/imaging
for m in delaunay_matern delaunay knn; do
  echo "=== $m ==="
  cat $AP/$m/hst/cpu_nan_check/*/search.summary
done
```

Always ALSO recompute from `files/samples_info.json` — do not trust the
printed rates alone (that is how the 10-step run was verified).

Profiling JSON/PNG per arm: `/mnt/ral/jnightin/nan_check/results/<model>/`.

## STATE ALREADY SET UP (no need to redo)

- RAL PyAuto stack pulled: PyAutoFit `aea9a40ce` -> **`fbfcece3a`** (the feature).
  PyAutoNerves/Array/Galaxy/Lens also on `main`.
- `/mnt/ral/jnightin/autolens_profiling_delaunay` moved from
  `feature/delaunay-nn-laptop-gpu-profile` to **`main` @ `a34d619`**.
  Its two untracked A100 delaunay_nn result files were moved to
  `/mnt/ral/jnightin/_untracked_backup_delaunay_nn` and verified byte-identical
  to the committed copies — safe to delete whenever.
- `/mnt/ral/jnightin/autolens_profiling` (the OTHER checkout) was deliberately
  NOT touched: modified tracked files on `feature/point-source-defaults-campaign`.
- Submits + ad-hoc leaf live in `/mnt/ral/jnightin/nan_check/` (outside any git
  checkout on purpose). Regenerate with `bash mk_submit.sh`.

## TRAPS ALREADY HIT (don't re-learn these)

- `imaging:delaunay` (plain) is in NEITHER `_MULTI_START_N_STARTS_BY_CELL` nor
  `_MULTI_START_BATCH_BY_CELL` in `scripts/misc/searches/_samplers.py`, so it
  falls back to 64 unbatched starts = the ~58 GB jvp fusion. Job 335004 forces
  `SEARCHES_N_STARTS=16` / `SEARCHES_BATCH_SIZE=4`.
- Use a FRESH `--config-name`. Re-running a completed named search returns the
  cached result via `.completed` and will NOT rewrite `search.summary`.
- `JAX_ENABLE_X64=True` must be set explicitly in the submit — not inherited
  under `sbatch`.
- A completed search deletes `search_internal` and zips the output dir; the
  10-step run happened to leave the unzipped dir too, but read the `.zip` if not.

## OPEN — needs a decision from you

1. **The A100 row from autolens_profiling#127 is still unfilled.** These runs
   are CPU and cannot stand in for it. As of 2026-08-14 the `gpu` partition had
   0/8 A100s free with 83 pending array tasks from user `c4072114`; Slurm's
   estimated start for a new 1-GPU job was **2026-08-19**. You chose CPU-only
   for tonight. If you want that row, queueing the A100 job early costs nothing
   and it accrues priority while waiting — say the word and I'll submit it.
2. **500 steps is a sixth of the calibrated 3000-step budget.** A NON-zero
   result from these arms is solid. A ZERO is weak evidence — the #117 campaign
   saw knn's mode crossing at step ~1300. Decide whether a zero warrants a
   longer run (3000 steps on CPU is ~49 h, so that one really wants the A100).
