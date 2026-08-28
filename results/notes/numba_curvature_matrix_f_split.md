# Numba CPU curvature matrix F — sub-block split (PyAutoArray#505)

Step 0 (the baseline below) measured 2026-08-28 on the local WSL host, `OMP_NUM_THREADS=1`,
`AUTOARRAY_NUMBA_OPERATED_MEMO=0`, PyAutoArray branch
`feature/numba-hst-curvature-matrix-speedup`, autolens `v2026.8.17.1`,
`n_repeats = 10`.

Cells (all four re-run from scratch, not hand-copied):

```
python scripts/imaging/likelihood_breakdown/pixelization_numba.py --instrument hst
python scripts/imaging/likelihood_breakdown/pixelization_numba.py --instrument euclid
python scripts/imaging/likelihood_breakdown/pixelization_numba.py --instrument hst --rect-mesh rtu
python scripts/imaging/likelihood_breakdown/delaunay_numba.py --instrument hst
```

## How F is split (and why the F row is a residual)

`InversionImagingSparseNumba.curvature_matrix` assembles F from three blocks —
now three separately callable private helpers in
`autoarray/inversion/inversion/imaging_numba/sparse.py`:

| block | helper | kernel |
|---|---|---|
| mapper x mapper | `_curvature_matrix_mapper_diag` | numba sparse-operator (w-tilde) contraction |
| mapper x linear-func | `_curvature_matrix_mapper_func_blocks_from` | dense sliding-window correlation over `ny x nx x ky x kx x 60` |
| linear-func x linear-func | `_curvature_matrix_func_func_blocks_from` | BLAS `dot` |

followed by a global mirror and the no-regularization diagonal add.

The breakdown harness times steps as the **incremental cost of touching the
next lazy attribute**. The three helpers are plain (uncached) methods, so
calling them does *not* prime the `curvature_matrix` cached property: touching
`curvature_matrix` afterwards recomputes all three blocks and then mirrors and
adds the diagonal.

Rather than let that double-count (which would inflate `total_step_by_step` by
~2x F and break comparability with pre-instrumentation artifacts), the F row is
reported as a **residual**:

```
F residual = t(inversion.curvature_matrix) - (t_mapper_mapper + t_mapper_func + t_func_func)
```

so the four F rows sum to exactly the un-split `curvature_matrix` step — which
each result JSON also records directly as `curvature_matrix_f_total_s`. The
residual is the mirror + diagonal add + assembly overhead, estimated by
subtracting the three isolated block timings from the composite. It is a
difference of averages and therefore carries the noise of all four
measurements; at the ~1e-2 s level it can come out slightly negative (the
recompute inside the cached property runs on warmer caches than the isolated
first touch). It is recorded as measured, never clipped.

## The split (seconds per evaluation)

| cell | direct eval | F total | mapper x mapper | mapper x linear-func | l-func x l-func | F residual |
|---|---|---|---|---|---|---|
| hst, rectangular bilinear | 1.738 | 1.275 | 0.316 | **0.953** | 0.0222 | -0.0156 |
| euclid, rectangular bilinear | 0.444 | 0.284 | 0.0633 | **0.198** | 0.00447 | 0.0181 |
| hst, rectangular RTU | 8.501 | 1.250 | 0.304 | **0.929** | 0.0222 | -0.00511 |
| hst, Delaunay-1250 | 1.455 | 1.055 | 0.130 | **0.899** | 0.0214 | 0.00432 |

Next two largest non-F steps:

| cell | 2nd | 3rd |
|---|---|---|
| hst bilinear | MGE operated mapping matrix 0.260 | inversion build 0.0303 |
| euclid bilinear | MGE operated mapping matrix 0.0949 | inversion build 0.0171 |
| hst RTU | mapper sparse triplets **7.88** | MGE operated mapping matrix 0.259 |
| hst Delaunay-1250 | MGE operated mapping matrix 0.235 | regularization matrix H (ConstantSplit) 0.0754 |

## Verdict (the step-0 checkpoint)

**The dense mapper x linear-func convolution dominates F on every cell** — 75 %
of F at hst bilinear, 70 % at euclid, 74 % at hst RTU, 85 % on Delaunay-1250.
The mapper x mapper sparse-operator block, which the symmetry work already
halved, is second at 10-25 %; the BLAS linear-func block and the mirror are
noise. The FFT lever (issue step 2 — route the 60 curvature-weight columns
through the existing batched `Convolver` with the flipped kernel) is therefore
the right phase-1 target, and it is the *only* term large enough to reach the
2x goal on its own.

Note the block does **not** scale with the source mesh: 0.90-0.95 s at hst
whether the source is a 784-cell rectangular mesh or a 1250-vertex Delaunay,
and it tracks the image-pixel count instead (0.198 s at euclid's 3841 masked
pixels vs 0.95 s at hst's 15361). That is the signature of the dense native-grid
expansion, not of the mapper contraction.

## Two things the re-baseline surfaced

1. **hst rectangular RTU is not a 1.3 s cell — it is an 8.5 s cell.** The
   "mapper sparse triplets" step costs **7.88 s** on the RTU (kernel-CDF)
   adaptive rectangular family, against 0.006 s on Bilinear and 0.019 s on
   Delaunay. RTU is GPU-only by the 2026-08-28 decision and out of scope for
   this issue, but the committed `_rtu` artifact now records the cost.
2. **The Delaunay-1250 cell is much cheaper than the issue's opening table**
   (1.455 s here vs 3.22 s): the reconstruction solve is 0.055 s (was 0.54 s)
   and the inversion build 0.030 s (was ~0.5 s), i.e. the NNLS warm-start memo
   and PyAutoArray#462 landed. The hst rectangular cell reads *higher* than the
   quoted 1.28 s (1.738 s here, F 1.275 s vs 0.997 s) with every numba step
   scaled up by a similar ~30 %, which is host/session variance rather than a
   regression — the block *proportions*, which is what step 0 exists to
   establish, are unaffected.

Pinned log-likelihood checks (rtol 1e-6): hst bilinear **PASSED**
(27661.910133664103), hst RTU **PASSED** (27180.704715696862), hst Delaunay
**PASSED** (29090.52721044813). euclid has no pinned value and is skipped by
design; this run measured 6213.306873885871.


---

# Steps 1-2: the result (PyAutoArray#505)

Measured 2026-08-28, same host and settings, `n_repeats = 10`, PyAutoArray
`feature/numba-hst-curvature-matrix-speedup` at the step-2 commit.

## What changed in the library

**Step 1 — three redundant passes over F, none of which changed a value:**
the global `curvature_matrix_mirrored_from` (every block was already symmetric
or had a known transpose, so the two off-diagonal writers now place the
transpose themselves); the `np.array(...)` copies of the three
`mapper.unique_mappings` arrays; and the per-mapper re-derivation and second
copy of `operated_mapping_matrix / noise_map ** 2`. Verified bit-identical:
`np.array_equal` on F, `True` on both euclid and hst.

**Step 2 — the FFT lever.** The mapper x linear-func block's dense sliding
window is a *correlation* with the PSF, which is exactly a convolution with the
PSF reversed along both axes. It now runs through the existing batched
`Convolver` (new cached `Convolver.reversed_kernel`), once per linear func
rather than once per (mapper, linear func) pair, and only the sparse scatter
onto source pixels stays in numba
(`curvature_matrix_off_diags_via_mapper_and_blurred_curvature_weights_from`).
The dense kernel is kept as the reference the FFT path is asserted against in
the unit tests, with asymmetric non-square PSFs so a missing reversal fails.

## Before / after (seconds per evaluation)

The step-0 table above and this one were measured in different sessions, and
this host carries ~20-30 % session-to-session variance, so the before column
here is a **fresh re-measurement of the step-0 commit taken back-to-back with
the after run**, not the step-0 table. (The step-0 numbers for the same cells
were 1.738 / 0.444 / 1.455 s.)

| cell | eval | F total | mapper x mapper | mapper x l-func | l-func x l-func |
|---|---|---|---|---|---|
| hst, rectangular bilinear | 1.562 -> **0.595** | 1.195 -> **0.359** | 0.295 -> 0.277 | 0.858 -> **0.054** | 0.0209 -> 0.0215 |
| euclid, rectangular bilinear | 0.349 -> **0.249** | 0.256 -> **0.097** | 0.060 -> 0.065 | 0.182 -> **0.023** | 0.0043 -> 0.0045 |
| hst, Delaunay-1250 | 1.367 -> **0.758** | 1.077 -> **0.184** | 0.122 -> 0.123 | 0.854 -> **0.052** | 0.0215 -> 0.0192 |
| hst, rectangular RTU | (step-0: 8.501) -> 7.479 | (1.250) -> **0.334** | (0.304) -> 0.249 | (0.929) -> **0.061** | 0.0222 -> 0.0193 |

RTU has no same-session before column: it is an 8.5 s cell dominated by the
7.1 s "mapper sparse triplets" step and is GPU-only by the 2026-08-28 decision,
so it was re-run once for currency rather than paired.

Speed-ups: **~16x on the mapper x linear-func block**, **3.3x on F** (5.9x on
Delaunay, where F was 74 % of the evaluation), and **2.6x on the whole HST
evaluation**. The 2x goal for the issue is met on the HST rectangular and
Delaunay cells.

## Where the time went

F is no longer the dominant term at HST resolution. After the change the
largest steps on hst bilinear are the mapper x mapper sparse-operator block
(0.277 s) and the MGE operated mapping matrix (0.224 s); the mapper x
linear-func block has fallen from 72 % of F to 15 %. The mapper x mapper block
is now the phase-2 candidate.

## Correctness

F agrees with the sliding-window result to **3e-18 relative** on both euclid
and hst (the FFT is not bit-identical to the direct sum, as expected). Pinned
log-likelihood checks **PASSED** on all three pinned cells: hst bilinear
27661.91013366411 (pinned 27661.910133665442), hst RTU 27180.70471569685
(pinned 27180.704715698186), hst Delaunay 29090.527210448134 (pinned
29090.527192092646). euclid, which has no pin, measured 6213.306873885871 —
unchanged to every recorded digit.

## Harness change

The `F: mapper x linear-func block` step label was `[dense conv]` and is now
`[FFT conv + scatter]`, since that is what it now times. Nothing else about the
step list, the residual design or the pins changed.

## Pool run (multiprocessing oversubscription check)

The harness measures one process at `OMP_NUM_THREADS=1`. Nautilus fits run one
process per core, so a single-thread win only counts if it survives the real
pool — and an FFT that quietly spins up its own threads would show up as a pool
regression even with the microbenchmark improving. This section is that check.

**Setup.** `autolens_workspace/scripts/imaging/features/pixelization/cpu_fast_modeling.py`,
first (non-SLaM) fit, run from the workspace CWD on an 8-core host, once against
canonical `PyAutoArray` `main` (caabe2d4) and once with `PYTHONPATH` shadowing
the task worktree; `autoarray.__file__` was asserted on both. `number_of_cores`
was raised 2 -> 8. Two deliberate departures from the smoke profile, both
forced:

- `PYAUTO_TEST_MODE=1`, not the smoke default `2`. Level 2 bypasses the sampler
  altogether, so the smoke profile as written runs **no pool at all**. Level 1
  sets Nautilus `n_like_max = 1`, which runs exactly one exploration batch:
  100 likelihood evaluations, so per-evaluation time is wall/100.
- `PYAUTO_SMALL_DATASETS` left unset. Under the 15x15 cap the fit sees 256
  masked image-pixels and the curvature matrix is too small to time; unset it is
  2828, the simulator's full resolution.

Everything else is the smoke default (`PYAUTO_SKIP_FIT_OUTPUT/VISUALIZATION/CHECKS=1`,
`PYAUTO_DISABLE_JAX=1`). The shell profile already exports
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`; that was left as-is,
since it is what a real user on this machine runs.

**Config A — the script as written.** Its lens is mass + shear with no light,
so `linear_obj_list` holds the mapper alone and the block this task moved to the
FFT does no work. As expected, nothing moves (s/eval, pool of 8, three runs):

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| main | 0.0756 | 0.0625 | 0.0610 | 0.0625 |
| branch | 0.0677 | 0.0682 | 0.0628 | 0.0677 |

**Config B — the same fit plus linear light.** A 60-component linear MGE bulge
(the `features/multi_gaussian_expansion/modeling.py` recipe: 30 Gaussians x 2
bases) added to the lens, so the mapper x linear-func block is non-empty. This
is the configuration the change actually touches:

| | run 1 | run 2 | run 3 | median |
|---|---|---|---|---|
| main, pool of 8 | 0.1706 | 0.2124 | 0.1919 | 0.1919 |
| branch, pool of 8 | 0.1601 | 0.1747 | 0.1767 | 0.1747 |
| main, serial (`number_of_cores=1`) | 0.4845 | | | 0.4845 |
| branch, serial (`number_of_cores=1`) | 0.4489 | | | 0.4489 |

**Verdict — no oversubscription.** The pool improves by 9 % and the single
process by 7 %: the pool gain tracks the single-thread gain rather than eroding
it. The parallel speed-up ratio is flat across the change — 0.4845/0.1919 =
**2.52x** on 8 cores before, 0.4489/0.1747 = **2.57x** after — which is the
number that would drop if the FFT had introduced hidden threads. (That ~2.5x on
8 cores is Nautilus's own batching and serialisation overhead; it is identical
before and after and is not something this task touched.)

This matches the code: `Convolver`'s FFT path uses `np.fft.rfft2`, which has no
`workers=` parameter and is single-threaded; `scipy.fft` appears only as
`next_fast_len`, a shape calculation, not a transform.

**Read this as a regression check, not a speed-up measurement.** The workspace
dataset is a 2828-pixel 0.1"/pixel simulation with a small rectangular mesh, so
F is a much smaller share of the evaluation than in the HST harness cells and
the end-to-end gain is correspondingly smaller than the 2.6x measured there.
