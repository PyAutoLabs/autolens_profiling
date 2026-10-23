# JAX Delaunay walk — early exit + single concatenated locate (A100, 2026-09-07)

PyAutoArray issue #530 / PR #531, branch `feature/delaunay-walk-early-exit` (`6152bdf0`),
against its merge base `bcd15cd9` (PyAutoArray `main`).

## What changed

The JAX point locator `pix_indexes_delaunay_walk_from` replaced its fixed-trip
`jax.lax.fori_loop` of `DELAUNAY_WALK_STEPS = 128` with a `jax.lax.while_loop` that exits
as soon as every query is located or outside the hull (typical resolution is under ten
steps), and stopped pushing the walk itself through the `DELAUNAY_LOCATE_CHUNK = 1024`
`lax.map` — only the nearest-vertex seed argmin is still chunked, which is all the chunking
was ever for (bounding the `(chunk, N)` intermediate under `vmap`). `jax_delaunay` now
locates the data grid and the 6,000 `ConstantSplit` cross points in **one** concatenated
call instead of two. Point location is integer-valued, so the walk's float inputs are
wrapped in `stop_gradient` (`while_loop` has no reverse-mode rule); every differentiable
quantity is still computed from the traced arrays.

## Verdict

On the A100, the HST / Hilbert-1500 / MGE-60 / `ConstantSplit` Delaunay imaging likelihood's
**params→H prefix — the aggregate that is invariant to the attribution shift below — falls
45.15 ms → 7.23 ms unbatched (−37.9 ms, 6.25×)** and 7.56 → 5.14 ms per call at `vmap` 16
(−2.4 ms, 1.47×). The witness asked for a ≥ 18 ms unbatched drop; the measured drop is
2.1× that. Whole-likelihood single-JIT runtime goes **93.46 → 60.36 ms (1.55×)** and the
step-by-step sum 99.46 → 62.05 ms. Both eager log-evidence pins held exactly, on both cells.
The DelaunayNN sibling, which shares the walk but not the concatenation, improves more
modestly (params→H prefix 178.97 → 144.79 ms, 1.24×).

## Provenance

| Job | Cell | Leg | Node | Start (2026-09-07) | Elapsed | State |
|---|---|---|---|---|---:|---|
| 342315 | `likelihood_breakdown/imaging/delaunay` | control | `euclid-ral-gpu-2` | 23:22:51 | 1:57 | COMPLETED |
| 342316 | `likelihood_breakdown/imaging/delaunay` | feature | `euclid-ral-gpu-2` | 23:24:49 | 1:47 | COMPLETED |
| 342317 | `likelihood_breakdown/imaging/delaunay_nn` | control | `euclid-ral-gpu-2` | 23:26:37 | 2:29 | COMPLETED |
| 342318 | `likelihood_breakdown/imaging/delaunay_nn` | feature | `euclid-ral-gpu-2` | 23:29:07 | 2:28 | COMPLETED |
| 342319 | `likelihood_runtime/imaging/delaunay` | control | `euclid-ral-gpu-2` | 23:33:58 | 1:02 | COMPLETED |
| 342320 | `likelihood_runtime/imaging/delaunay` | feature | `euclid-ral-gpu-2` | 23:35:01 | 1:03 | COMPLETED |

- Breakdown flags `--split-setup --vmap-batch 16`; runtime cell full timing only (no
  `--vmap-probe`), `vmap` batch 16 in both legs. `PyAutoLens 2026.8.17.1`, fp64
  (`JAX_ENABLE_X64=True`), dense inversion path, NVIDIA A100 80GB PCIe.
- Recorded `xla_flags` on every row: `--xla_disable_hlo_passes=constant_folding
  --xla_gpu_autotune_level=0` — identical across the six jobs, so these rows are
  comparable with the 2026-09-05 rows in `delaunay_nn_breakdown.md` and **not** with the
  2026-07-10 rows in `preopt_breakdown_baseline.md`.
- **All six jobs ran on the same node in one 14-minute window**, so the A/B is a
  same-session comparison, not a comparison against a recorded baseline.

### The A/B differs in PyAutoArray and nothing else

| Repo | Revision |
|---|---|
| PyAutoArray — control leg | `bcd15cd9d599c2e3fb7a7c3407d362fb39507c53` |
| PyAutoArray — feature leg | `6152bdf0506229dce36666db8b6b9bef636985fe` (`feature/delaunay-walk-early-exit`) |
| PyAutoNerves | `fc9c474ba36cca0aaf0d40351e65e791b99f9a35` |
| PyAutoFit | `f6a9915045550ac5fb6930c6076cd9242d014f1e` |
| PyAutoGalaxy | `6d216c151c914b2ce79af18fbc0348a6fc5425d7` |
| PyAutoLens | `146a3d7254418eac964edd8b89ada7281ac778e7` |
| autolens_profiling (`AP_ROOT`) | `d51b94ae9bdd612c0607ea27fc9a04b25a0f4418` |

The **shared** `/mnt/ral/jnightin/PyAuto` install was deliberately not touched — the
subhalo-validation and Euclid DR1 CPU runs (`342299`, `342301`, `342311`, `342314`) were
live on it throughout. Both legs instead prepend a private PyAutoArray checkout to the
`PYTHONPATH` that `activate.sh` exports:
`/mnt/ral/jnightin/PyAuto_wt/delaunay-walk-early-exit/PyAutoArray_{control,feature}`, and
each job prints `autoarray.__file__` and the checkout's `git rev-parse HEAD` into its
SLURM log. Note the shared install sits at `e36a5af4`, **eight commits behind** the
branch's merge base, which is why the control leg is a private `bcd15cd9` checkout rather
than the shared tree: running the control on the shared install would have folded eight
unrelated commits into the difference. Submits:
`hpc/batch_gpu/submit_{breakdown_imaging_delaunay,breakdown_imaging_delaunay_nn,runtime_imaging_delaunay}_a100_hst_fp64_{control,walk_early_exit}`,
run from `/mnt/ral/jnightin/autolens_profiling_wt/delaunay-walk-early-exit` (detached at
`d51b94a`).

### Pins

`EXPECTED_LOG_EVIDENCE_HST` passed **unchanged on every leg** — `29110.920858` for
Delaunay (342315/342316, and `pinned_drift: []` on the two runtime rows) and
`29144.581944` for DelaunayNN (342317/342318). The mapping is bit-identical to the eager
NumPy reference on both sides of the change.

## Delaunay — control vs feature

All values ms per likelihood call. `vmap/16` is `jax.jit(jax.vmap(fn))` over a params
pytree broadcast to batch 16, reported as batch time / 16; only the combined
inversion-setup block, the four `--split-setup` prefixes and the params→H prefix are
re-timed under `vmap`, so the remaining rows read `—` there rather than zero.

| Step | control | feature | Δ | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|---:|
| Ray-trace data grid | 0.188 | 0.182 | −0.006 | — | — |
| Ray-trace mesh grid | 0.226 | 0.172 | −0.054 | — | — |
| Lens light images (pre-PSF) | 0.161 | 0.136 | −0.025 | — | — |
| Blurred image (PSF convolution) | 0.852 | 0.846 | −0.006 | — | — |
| Profile-subtracted image | 0.158 | 0.153 | −0.005 | — | — |
| **Inversion setup (steps 5–8 combined)** | **44.650** | **20.430** | **−24.220** | **15.451** | **13.806** |
| Data vector (D) | 0.342 | 0.324 | −0.018 | — | — |
| Curvature matrix (F) | 4.808 | 4.798 | −0.010 | — | — |
| **Regularization matrix (H)** | **13.555** | **0.225** | **−13.330** | **0.741** | **0.070** |
| Regularized reconstruction | 32.281 | 32.513 | +0.232 | — | — |
| Mapped recon + log evidence | 2.240 | 2.267 | +0.027 | — | — |
| **Total step-by-step (unbatched)** | **99.463** | **62.045** | **−37.418** | — | — |

### Four-way setup split and the prefixes

| Piece | control | feature | Δ | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.031 | 1.032 | +0.001 | 0.099 | 0.062 |
| **Triangulation + interpolation** | **30.563** | **5.970** | **−24.593** | **6.715** | **5.007** |
| Mapping matrix | 0.333 | 0.195 | −0.138 | −0.001 | 0.121 |
| Blurred mapping matrix (PSF) | 8.050 | 11.928 | +3.878 | 8.399 | 8.343 |
| *Interpolator prefix (params→step 6)* | *31.594* | *7.003* | *−24.591* | *6.814* | *5.069* |
| **params→H prefix (the witness)** | **45.149** | **7.228** | **−37.921** | **7.555** | **5.139** |

The **+3.878 ms "Blurred mapping matrix (PSF)" row is a by-difference artifact, not a
regression**: it is prefix 8 minus prefix 7, both of which shrank, and the same row under
`vmap` 16 — where the prefixes are re-measured independently — is flat (8.399 → 8.343 ms).
The combined-block row (44.650 → 20.430 ms) and the whole-likelihood single-JIT runtime
below both fall by more than the sum of the split, so no work moved into the PSF step.

### Whole-likelihood runtime cell (`likelihood_runtime/imaging/delaunay`)

| | control | feature | Δ |
|---|---:|---:|---:|
| Full pipeline (single JIT) | 93.463 | 60.355 | **−33.108 (1.55×)** |
| `vmap` batch 16, per call | 41.945 | 39.520 | −2.425 (1.06×) |

This is the number a production evaluation actually pays. The unbatched saving (−33.1 ms)
tracks the params→H prefix drop (−37.9 ms) to within the fusion slack expected when the
whole pipeline is compiled as one program; the batched saving is small because at batch 16
the walk was already amortising and the dense linear algebra dominates.

## The H-row attribution caveat (read this before comparing rows across versions)

**The H row moved meaning with this change; the params→H prefix did not.** The breakdown
attributes step 11 as `t(params → H) − t(params → interpolator outputs)`. Before #531 a
prefix stopping at step 6 never asked for the `ConstantSplit` split points, so XLA
dead-code-eliminated that second walk out of the interpolator prefix and its whole cost
surfaced in the H row's subtraction. The feature concatenates both walks into one locate
call, so the split walk now runs **inside** the step-6 prefix — inside "Triangulation +
interpolation" — and the subtraction leaves only the regularization assembly. Hence
**H: 13.555 → 0.225 ms**, and a small negative would have been equally legitimate.

Two consequences:

1. Comparing the "Triangulation + interpolation" row alone *understates* the change
   (30.563 → 5.970 ms is a 5.1× improvement measured across a boundary that moved work
   *into* the numerator), and comparing the H row alone *overstates* it.
2. The aggregate to quote across library versions is **params→H (Tri+interp + H)**, i.e.
   `regularization_matrix_prefix_s` in the JSON. A short note to that effect has been added
   to `scripts/imaging/likelihood_breakdown/delaunay.py`'s module docstring and to the
   comment above `_setup_prefix_fn`.

The NumPy/scipy path (`scipy_delaunay`, the numba likelihood) is untouched by #531 and
still charges the split walk to H.

## DelaunayNN sibling

`DelaunayNN` (Sibson natural-neighbour, cap 32) shares the locator but **not** the
concatenation: `sibson.py:522` and `sibson.py:648` still call
`pix_indexes_delaunay_walk_from` twice, once for the data grid and once for the split
points. It therefore gets the early exit and the unchunked walk, but not the single-call
saving — and its H row stays large rather than collapsing.

| Row | control | feature | Δ | control vmap/16 | feature vmap/16 |
|---|---:|---:|---:|---:|---:|
| Inversion setup (steps 5–8 combined) | 135.428 | 111.484 | −23.944 | 22.264 | 22.944 |
| Triangulation + interpolation | 122.450 | 98.060 | −24.390 | 15.951 | 11.610 |
| Regularization matrix (H) | 55.467 | 45.663 | −9.804 | 12.742 | 12.749 |
| Interpolator prefix (params→step 6) | 123.504 | 99.126 | −24.378 | 16.015 | 11.675 |
| **params→H prefix** | **178.971** | **144.789** | **−34.182 (1.24×)** | **28.757** | **24.424** |
| Total step-by-step (unbatched) | 230.573 | 196.963 | −33.610 | — | — |

Both DelaunayNN legs held the `29144.581944` pin. The `−0.617` / `−1.912` ms "Mapping
matrix" rows are the same below-noise by-difference artifact documented in
`delaunay_nn_breakdown.md`.

## Phase 2 recommendation

**Do not start Phase 2 (static image-plane seed + fan test) on the Delaunay path; extend
the single concatenated locate to the Sibson path first.** The number this rests on: after
#531 the Delaunay params→H prefix is **7.23 ms unbatched and 5.14 ms per call at `vmap` 16**
— about 13% of the 39.5 ms batched per-evaluation cost — so Phase 2's remaining headroom on
this cell is at most a few ms per call, for a change that reaches into Mapper/AdaptImages
plumbing. DelaunayNN, by contrast, still spends **144.8 ms unbatched / 24.4 ms per call at
`vmap` 16** in the same prefix, and the cheapest available cut there is the one Delaunay
already took: concatenate `sibson.py`'s two walk calls into one. That is a much smaller
change than Phase 2 and targets the cell with 20× the remaining cost.

## Artifacts

- `results/breakdown/imaging/delaunay_hpc_a100_fp64_control_2026_09_07.{json,png}`
- `results/breakdown/imaging/delaunay_hpc_a100_fp64_walk_early_exit.{json,png}`
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_control_2026_09_07.{json,png}`
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_walk_early_exit.{json,png}`
- `results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_control_2026_09_07.json`
- `results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_walk_early_exit.json`
- `hpc/batch_gpu/submit_breakdown_imaging_delaunay_a100_hst_fp64_{control,walk_early_exit}`
- `hpc/batch_gpu/submit_breakdown_imaging_delaunay_nn_a100_hst_fp64_{control,walk_early_exit}`
- `hpc/batch_gpu/submit_runtime_imaging_delaunay_a100_hst_fp64_{control,walk_early_exit}`

The pre-existing `delaunay_hpc_a100_fp64.json` / `delaunay_nn_hpc_a100_fp64.json` rows
(2026-09-05, job 342277/342278, PyAutoArray `e36a5af4`) are left untouched; the control
legs here supersede them for any comparison against this branch.
