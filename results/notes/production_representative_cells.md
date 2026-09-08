# The CPU numba imaging cells now measure what production runs (2026-09-08)

Until this change the default imaging cells measured a configuration nothing in
production ran. Every cell hard-coded `over_sample_size_pixelization=1`, used a
fixed regularization coefficient, a 1250/1500-vertex mesh with no zeroed edge,
no positions penalty, a 60-Gaussian single-basis MGE, never pinned threads, and
repeated **one prior-median instance ten times** with the NNLS cross-evaluation
warm-start memo silently on — so repeats 2..10 seeded themselves from a
100 %-correct previous solve. The Euclid headline row
`imaging/delaunay_numba | euclid | 1.19 s` was that memo-self-seeded warm mean.

`autolens_profiling#235` replaced the four CPU numba cells' fiducial with the
production configuration per instrument, encoded field by field with file:line
provenance in [`_production_config.py`](../../_production_config.py).

## What production actually runs

Two references, because no single cell can match both. The Euclid cell is the
Euclid DR1 `vis_pix` stage (job 342301); the HST cell is subhalo-validation's
`source_pix[2]` / `rect_adapt` (job 342311).

| Field | Euclid `vis_pix` | subhalo `source_pix[2]` | Pre-#235 cell |
|---|---|---|---|
| Delaunay mesh | Hilbert 500 + 30 edge = 530 vertices, 30 zeroed | Hilbert 1250 + 30 = 1280, 30 zeroed | 1250, **0 zeroed** |
| Hilbert weights | `weight_power=3.5`, `weight_floor=0.01` | same | **1.0 / 0.0** |
| Rectangular mesh | (no rectangular production stage) | `(32, 32)` adapt, `Adapt` reg | **(28, 28)**, `Constant(1.0)` |
| Regularization | **free** `AdaptSplit`, LogUniform 1e-6..1e6 / Uniform 0..1 | same | fixed `ConstantSplit(1.0)` / `Constant(1.0)` |
| Adapt-image S/N cap | none | **3.0** | none |
| `over_sample_size_lp` | `[4, 2, 2]` at `[0.1, 0.3]` | same | `[4, 2, 1]` at `[0.3, 0.6]` |
| `over_sample_size_pixelization` | `4` where source S/N > 3, else `2` | same | **flat `1`** |
| Sparse operator | re-applied after every over-sampling change | same | applied once |
| Positions penalty | `factor=3.0, minimum_threshold=0.2` | none | none |
| MGE lens light | 20 Gaussians x 2 bases | 30 x 2 | **60 x 1** |
| Thread env | all five vars `=1`, 8 cores | same | **never set** |
| Instance stream | Nautilus samples every free parameter | same | **one prior median, x10** |
| NNLS memo | library default (`true`), unset by both projects | same | on, **unrecorded** |

## What the cells now do

- `--variant production` (default) resolves the instrument's preset;
  `--variant legacy` rebuilds the pre-#235 configuration and writes to its own
  `_legacy` basename. Legacy does **not** restore `[4, 2, 1]` — see below.
- Threads are pinned to `1` before numpy imports (OpenBLAS/MKL read those
  variables once, when the shared library loads), and the resolved environment
  — including anything overridden — is recorded under
  `configuration.thread_env`.
- The instance stream is `--n-instances` (default 20) seeded iid draws from the
  central 20 % of every prior, so no evaluation can seed the next.
- The memo is **off** by default through both gates
  (`al.Settings(nnls_warm_start_memo=False)` *and*
  `AUTOARRAY_NNLS_WARM_START=0`), and the flag is read back off the constructed
  `Settings` object so `configuration.memo_provenance` records what took effect
  rather than what was asked for. `--memo on` measures the memo.
- Timing protocol: instance 0 is a discarded warm-up (numba compile, plus the
  cold-cache first-call hazard); the next `--cold-evals` (default 3) are **cold
  evals**; the rest are **warm iid evals**, reported as median and mean. With
  the memo off every evaluation is cold in the memo sense, so cold and warm
  differ only by cache warmth and draw-to-draw variance.

### Why the cold eval is the comparable quantity

PyAutoFit logs one timed evaluation of the max-likelihood parameters as "Log
Likelihood Function Evaluation Time"
(`autofit/non_linear/search/updater.py:311-317`). A production `search.summary`
also records "Time Per Sample" (pooled across 8 Nautilus workers) and a "Speed
Up Factor"; the single-process quantity is `s/sample x speed-up`, i.e. the cold
eval — not the pooled `s/sample`.

| run | stage | s/sample | cold eval (s) | speed-up |
|---|---|---|---|---|
| 342301_2 Euclid | `vis_pix` | 0.2201 | 0.692 | 3.15 |
| 342301_0 | `vis_pix` | 0.2276 | 0.803 | 3.53 |
| 342301_4/5/6/9 | `vis_pix` | 0.210/0.326/0.279/0.227 | 0.912/1.130/1.047/0.785 | 4.35/3.47/3.75/3.46 |
| 342311_0 HST | `sp[1]`/`sp[2]` | 0.2260/0.2327 | 0.704/0.806 | — |
| 342311_1 | | 0.1707/0.1723 | 0.557/0.556 | — |
| 342311_2 | | 0.1678/0.1719 | 0.516/0.534 | — |

Witness (decision 2, human, 2026-09-08): a production-mode cold eval must land
within 1.5x of 0.69–1.13 s (Euclid) / 0.52–0.81 s (HST).

## The numbers (2026-09-08)

Local WSL host, Intel i9-10885H (8 logical cores), fp64 numba sparse path,
`--n-instances 20 --cold-evals 3` (seed 235), all five thread variables pinned
to `1`, NNLS memo **off** (`Settings.nnls_warm_start_memo=False` **and**
`AUTOARRAY_NNLS_WARM_START=0`, both read back and recorded). Libraries:
PyAutoArray `47a00e8c`, PyAutoFit `08207bad0`, PyAutoGalaxy `ec5ce75d`,
PyAutoLens `08a05858a`, PyAutoNerves `0e7163b`; `autolens 2026.8.17.1`.

| cell | instrument | cold eval median (s) | warm iid median (s) | warm iid mean (s) | n warm | witness |
|---|---|---|---|---|---|---|
| `likelihood_runtime/delaunay_numba` | euclid | 0.239 | 0.232 | 0.233 | 16 | below range |
| `likelihood_breakdown/delaunay_numba` | euclid | 0.271 | 0.275 | 0.270 | 16 | below range |
| `likelihood_runtime/pixelization_numba` | euclid | 0.715 | 0.530 | 0.551 | 16 | **PASS** |
| `likelihood_breakdown/pixelization_numba` | euclid | 0.693 | 0.588 | 0.580 | 16 | **PASS** |
| `likelihood_runtime/delaunay_numba` | hst | 1.463 | 1.969 | 1.938 | 16 | above range |
| `likelihood_breakdown/delaunay_numba` | hst | 1.642 | 2.272 | 2.460 | 16 | above range |
| `likelihood_runtime/pixelization_numba` | hst | 1.044 | 1.419 | 1.416 | 16 | **PASS** |
| `likelihood_breakdown/pixelization_numba` | hst | 0.829 | 1.060 | 1.040 | 16 | **PASS** |

Pinned log-likelihoods (the last instance of the seeded sequence, so a
deterministic function of model + seed + `n_instances`): Delaunay euclid
`5817.7313621849535`, hst `23996.231413329842`; rectangular euclid
`4242.698962741273`, hst `22677.756578185603`. The breakdown and runtime cells
of each family agree to the last digit, which is the cross-check that they build
the same model.

### Reading the witness

Four of eight rows land inside 1.5x of the production cold-eval range. The four
that do not are both Delaunay rows, and they miss in **opposite directions**,
which is the tell that what remains is dataset size and host speed, not
configuration:

- **Euclid Delaunay 0.24 s is ~2.9x *faster* than production's 0.69–1.13 s.**
  The profiling euclid dataset is a simulated 3.5" mask — 3841 masked pixels,
  15424 over-sampled — where a real Euclid VIS cut-out for job 342301 is
  substantially larger. Everything the preset controls (530 vertices, 30 zeroed,
  Hilbert 3.5/0.01, free `AdaptSplit`, MGE 20x2, S/N>3 4/2 over-sampling, the
  3.0/0.2 positions penalty) now matches; the residual is how much data each
  evaluation touches.
- **HST Delaunay 1.46 s is ~1.8x *slower* than production's 0.52–0.81 s.** Same
  arithmetic in the other direction: 15361 masked pixels, 61600 over-sampled and
  1280 vertices on a laptop core, against an 8-core RAL node.

The two rectangular rows pass on both instruments, so the protocol itself is
sound. Closing the Delaunay gap means matching the *dataset*, not the model, and
is left as a follow-up: the honest statement today is that the cells measure the
production **configuration**, and the pooled-throughput comparison
(`s/sample x speed-up`) still needs the production mask.



## The lp `[4, 2, 2]` sweep and the pins it moved

`sub_size_list=[4, 2, 1]` leaves the outermost radial bin un-over-sampled, which
causes gradient issues. It is retired everywhere: every cell in this repo, the
Euclid pipeline (`util.py` + three stage scripts, `[16, 4, 1]` -> `[16, 4, 2]`
where that recipe was used) and `subhalo_validation/scripts/imaging.py`. The
`euclid_dr1_prelim` checkout is a clone of the pipeline repo and picks it up on
its next pull. Four `over_sample_size_lp=1` sites became `2`.

This changes the over-sampled light-profile grid, so it moves pinned
likelihoods. Re-measured on this host from one eager run each:

| cell | old | new |
|---|---|---|
| `likelihood_breakdown/mge.py` | 27379.38890685539 | 27373.152646517723 |
| `likelihood_runtime/mge.py` | 27379.38890685539 | 27373.152646517723 |
| `likelihood_runtime/mge_mass_jax.py` | -56107.56407588643 | -56107.03327291309 |
| `likelihood_runtime/pixelization_numba_mge_mass.py` | -56107.564075886374 | -56107.03327291303 |

`likelihood_runtime/pixelization.py` re-ran clean and both pins (bilinear, rtu)
**passed unchanged** — the shift is below its `rtol=1e-4`.

Two pins could not be re-measured and say so in place:

- `likelihood_breakdown/pixelization.py` raises at step 5 on `main` as well,
  importing `autoarray.inversion.mesh.mesh.rectangular_adapt_density`, a module
  PyAutoArray has since split into `rectangular_bilinear_adapt_density` /
  `rectangular_rtu_adapt_density`. Unrelated to this task; filed as a follow-up.
- The four #232 JAX Delaunay cells' `EXPECTED_LOG_EVIDENCE` — see below.

The four converted cells' pins are **gone, not moved**: their configuration is a
different model, so the old values describe something that no longer exists.
Recorded here for the historic rows:

| cell | instrument / scheme | pre-#235 pinned value |
|---|---|---|
| `likelihood_runtime/delaunay_numba.py` | euclid / `constant_split` | 7215.3687893658935 |
| | hst / `constant_split` | 29090.527192092646 |
| | euclid / `adapt_split` (#232) | 5579.104036561161 |
| | hst / `adapt_split` (#232) | 29212.44050977029 |
| `likelihood_breakdown/delaunay_numba.py` | euclid | 7215.3687893658935 |
| | hst | 29090.527192092646 |
| `likelihood_runtime/pixelization_numba.py` | hst / bilinear | 27661.910133665442 |
| | hst / rtu | 27180.704715698186 |
| `likelihood_breakdown/pixelization_numba.py` | hst / bilinear | 27661.910133665442 |
| | hst / rtu | 27180.704715698186 |

`--variant legacy` is the closest thing that remains, and it does not reproduce
those values exactly either: the lp radial-bin change is unconditional, so the
legacy variant runs `[4, 2, 2]` too. What legacy does reproduce is the *cost
structure* — the mesh, regularization, over-sampling, MGE basis and repeated
instance the historic timings were measured with.

### The deflection axis, re-pinned (2026-09-08)

`scripts/lens/deflections/_driver.py` carried the last `[4, 2, 1]` in the repo;
its `SUB_SIZE_LIST` is now `[4, 2, 2]` as well, so the axis is retired
everywhere without exception. The over-sampled `Grid2DIrregular` grows
accordingly (hst 17980 -> 62752 points), which changes the `irregular_s` column
but not what any profile computes: the deflection pins are taken on
`dataset.grids.pixelization` (still `over_sample_size_pixelization=1`) and on a
dedicated fixed-coordinate `Grid2DIrregular`, neither of which the lp recipe
touches.

All four cells were nonetheless re-pinned on both instruments with `--repin`
(reason recorded as `pin_provenance`), and every value held: the **largest
relative move across all eight runs was 3.284e-11** (`total` / `PowerLaw`
sample, both instruments; `basis` moved by exactly zero). No re-pin came near
the `--repin-max-shift` guard, so `--repin-force` was never used. Each cell was
then re-run without `--repin` and all eight pin checks PASSED at rtol 1e-6.

## What deliberately did not change

- **The JAX / A100 Delaunay cells** (`likelihood_{breakdown,runtime}/delaunay.py`,
  `delaunay_nn.py`) keep their over-sampling, mesh and #232 A/B: they carry every
  A100 pin, and GPU production-representativeness is a separate task
  (decision 3, human, 2026-09-08). They gain the lp `[4, 2, 2]` change and
  thread / memo provenance fields only.
- **`over_sample_size_pixelization=1`** in the non-default, misc and deflection
  cells, which are documented legacy.
- **The library's memo default** (`nnls_warm_start_memo: true`). This task turns
  it off *in the profiling cells* and records the flag; whether production
  should change is a separate library prompt. The evidence that set the default
  now lives in [`scripts/misc/nnls_warm_start/`](../../scripts/misc/nnls_warm_start/README.md)
  with its two notes under [`results/nnls_warm_start/`](../nnls_warm_start/),
  so no default cell or README table cites a memo-on number.

## Related

- [`nnls_warm_start_memo.md`](../nnls_warm_start/nnls_warm_start_memo.md) and
  [`nnls_warm_start_memo_matrix.md`](../nnls_warm_start/nnls_warm_start_memo_matrix.md)
  — the memo experiment, moved out of the default breakdown package.
- [`numba_curvature_matrix_f_split.md`](./numba_curvature_matrix_f_split.md) —
  the per-step F/MGE decomposition these cells report.
- autolens_profiling#232 — the Delaunay cells' `AdaptSplit` A/B, merged first.
