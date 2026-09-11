# A100 fp64 pixelized likelihood baseline — dense vs sparse (2026-09-10)

autolens_profiling issue [#241](https://github.com/PyAutoLabs/autolens_profiling/issues/241),
branch `feature/a100-pixelized-baseline`, code commit `59c0f84`. Twelve jobs
(342617–342628), `{rectangular bilinear, Delaunay, DelaunayNN} × {dense, sparse} ×
{likelihood_runtime, likelihood_breakdown}`, all on `euclid-ral-gpu-2` inside one
17-minute window. This is the reference set the **matrix-free CG + SLQ** work is to be
judged against; it replaces every A100 pixelized/sparse row taken before
PyAutoNerves#162 (`--xla_gpu_enable_triton_gemm=false`) and PyAutoArray #531/#533/#537.

## Scope — read this before quoting a number

- **Fiducial tier, not the production preset.** HST, mask 3.5″, 15,361 masked pixels,
  62,752 over-sampled pixels, source 1500 (Delaunay vertices) / 1521 (39×39 rectangular),
  MGE-60 lens light, `over_sample_size_lp` sub-size `[4, 2, 2]`, **pixelization
  over-sampling 1**. The #235 production preset (`production_representative_cells.md`,
  decision 3 still open) is a different, heavier configuration; do not mix the two.
- **The sparse rows are the "half matrix-free" comparator, not a production GPU path.**
  Production GPU runs fit the plain dataset — the JAX path never applies the sparse
  operator (`feedback_jax_path_never_applies_sparse_operator`). These legs call
  `Imaging.apply_sparse_operator(batch_size=128)` explicitly to measure what a w-tilde
  formulation costs when the dense mapping-matrix chain is replaced but the direct
  Cholesky solve is kept.
- **Regularization** is `Constant(1.0)` on the rectangular cell and `adapt_split`
  (0.1 / 10.0 / 0.1) on the Delaunay family, matching each cell's shipped default.

## Provenance and gate

| Job | Cell | Path | Task | Start (2026-09-10 BST) | Elapsed |
|---|---|---|---|---|---:|
| 342617 | pixelization | dense | runtime | 20:32:08 | 1:19 |
| 342618 | delaunay | dense | runtime | 20:33:28 | 0:57 |
| 342619 | delaunay_nn | dense | runtime | 20:34:25 | 1:17 |
| 342620 | pixelization | sparse | runtime | 20:35:43 | 0:58 |
| 342621 | delaunay | sparse | runtime | 20:36:41 | 1:06 |
| 342622 | delaunay_nn | sparse | runtime | 20:37:48 | 1:24 |
| 342623 | pixelization | dense | breakdown | 20:39:13 | 1:25 |
| 342624 | delaunay | dense | breakdown | 20:40:38 | 1:32 |
| 342625 | delaunay_nn | dense | breakdown | 20:42:11 | 2:23 |
| 342626 | pixelization | sparse | breakdown | 20:44:34 | 1:35 |
| 342627 | delaunay | sparse | breakdown | 20:46:10 | 1:44 |
| 342628 | delaunay_nn | sparse | breakdown | 20:47:55 | 2:30 |

All twelve `COMPLETED` on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe), fp64, `jax 0.10.2`,
`PyAutoLens 2026.8.17.1`. Library revisions (shared RAL install, identical on all jobs):
PyAutoNerves `0e7163bc`, PyAutoFit `66f9f8d5`, PyAutoArray `35aa681f`,
PyAutoGalaxy `99cf7429`, PyAutoLens `7d1b04de`.

Every leg carries
`--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`,
a **fresh per-job** `JAX_COMPILATION_CACHE_DIR` (`output/jax_cache/baseline_<jobid>`),
`autotune_cache_entries_at_start: 0`, `cache_fresh: true`, and
`AUTOTUNE_ENTRIES count=0` at exit — i.e. no leg inherited a seeded autotune cache, and
the cuBLASLt lowering came from the Triton-off flag rather than from cache state.

| Leg | pin | pin result | AUTOTUNE_ENTRIES | full-pipeline compile s |
|---|---|---|---:|---:|
| 342617 rect dense | log_evidence 28621.128714 | PASSED, `pinned_drift: []` | 0 | 13.86 |
| 342618 del dense | 29140.295896 | PASSED, `pinned_drift: []` | 0 | 13.69 |
| 342619 delnn dense | 29277.464594 | PASSED, `pinned_drift: []` | 0 | 17.45 |
| 342620 rect sparse | 28621.128714 | PASSED, `pinned_drift: []` | 0 | 13.32 |
| 342621 del sparse | 29140.295897 | PASSED, `pinned_drift: []` | 0 | 16.80 |
| 342622 delnn sparse | 29277.464595 | PASSED, `pinned_drift: []` | 0 | 17.64 |
| 342623 rect dense bd | 28621.128714 | eager regression + inversion-matrix checks PASSED | 0 | see below |
| 342624 del dense bd | 29140.295882 | both assertions PASSED | 0 | — |
| 342625 delnn dense bd | 29277.464588 | both assertions PASSED | 0 | — |
| 342626 rect sparse bd | 28621.128714 | + sparse F/D vs `fit.inversion` to 1e-8 PASSED | 0 | — |
| 342627 del sparse bd | 29140.295882 | + sparse F/D 1e-8 PASSED | 0 | — |
| 342628 delnn sparse bd | 29277.464588 | + sparse F/D 1e-8 PASSED | 0 | — |

The three `delaunay*` breakdown JSONs carry no `pinned_drift` key (only the rectangular
cell writes one); their pins were audited from the SLURM stdout, which is where the
`Assertion PASSED` lines above come from. **No gate failure**: dense
`Curvature matrix (F)` is 4.826 / 4.824 / 4.830 ms on rectangular / Delaunay / DelaunayNN
(threshold < 6 ms), no `cache_fresh: false`, no pin drift.

One pin bookkeeping wrinkle, not a gate failure: the **rectangular runtime** cell still
pins `28622.397322591198` (measured 2026-08-26, pre-#235), while the breakdown cell was
re-measured on 2026-09-08 to `28621.128714095972` after the light-profile radial bins
moved to `[4, 2, 2]`. Both runtime legs measured 28621.1287 and PASSED, because the
4.4e-5 relative offset sits inside `check_pinned`'s `rtol=1e-4` — the difference is
documented in `likelihood_breakdown/pixelization.py`. The runtime constant should be
re-measured so the pin describes what the cell computes today; file as a follow-up.

`sparse_equivalence` is identical on all three sparse legs — the w-tilde F and D are the
same matrices as the dense ones to round-off:

| quantity | max rel. diff | tolerance |
|---|---:|---:|
| curvature matrix F | 3.92e-15 | 1e-8 |
| data vector D | 3.33e-15 | 1e-8 |

## Runtime rows (full-pipeline JIT)

| Cell | Path | single-JIT ms | per-call @ `vmap` 16 ms | compile s | lower s | nvidia-smi used |
|---|---|---:|---:|---:|---:|---:|
| Rectangular 1521 | dense | 50.63 | **32.08** | 13.86 | 4.62 | 35,337 MiB |
| Rectangular 1521 | sparse | 57.63 | **36.65** | 13.32 | 3.19 | 4,877 MiB |
| Delaunay 1500 | dense | 64.95 | **42.53** | 13.69 | 3.43 | 35,339 MiB |
| Delaunay 1500 | sparse | 70.75 | **48.04** | 16.80 | 4.59 | 4,875 MiB |
| DelaunayNN 1500 | dense | 73.41 | **46.44** | 17.45 | 3.71 | 35,339 MiB |
| DelaunayNN 1500 | sparse | 80.09 | **49.25** | 17.64 | 3.84 | 5,639 MiB |

**Sparse costs +14.3 % / +13.0 % / +6.1 % per call at `vmap` 16** (+13.8 % / +8.9 % /
+9.1 % single-JIT) and uses ~7× less device memory (the `nvidia-smi` column is the used
figure at capture, indicative rather than a per-replica measurement — the per-replica
numbers in `sparse_vs_dense_inversion_path.md` remain the calibrated ones). The 2026-07
CPU-era verdict (sparse for memory, not speed) survives unchanged after the Triton-off
flag and the Delaunay speed-ups.

## Breakdown — unbatched per-call (ms)

Dense path:

| Step | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| Ray-trace (data grid) | 0.211 | 0.180 | 0.223 |
| Ray-trace (mesh grid) | — | 0.178 | 0.201 |
| Lens light images (pre-PSF) | 0.145 | 0.147 | 0.144 |
| Blurred image (PSF convolution) | 1.280 | 0.876 | 0.942 |
| Profile-subtracted image | 0.156 | 0.139 | 0.146 |
| Overlay grid (source pixel centres) | 0.167 | — | — |
| Inversion setup (combined) | 14.148 | 20.983 | 27.632 |
| Data vector (D) | 0.374 | 0.305 | 0.329 |
| **Curvature matrix (F)** | **4.826** | **4.824** | **4.830** |
| Regularization matrix (H) | −0.568 ⚠ | 0.234 | 0.869 |
| **Regularized reconstruction** | **36.936** | **37.394** | **37.310** |
| Mapped recon + log evidence | 2.344 | 2.214 | 2.275 |
| step-sum | 60.0 | 67.5 | 74.9 |

Sparse (w-tilde) path:

| Step | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| Ray-trace (data grid) | 0.176 | 0.189 | 0.203 |
| Ray-trace (mesh grid) | — | 0.158 | 0.191 |
| Lens light images (pre-PSF) | 0.143 | 0.119 | 0.132 |
| Blurred image (PSF convolution) | 1.230 | 1.327 | 0.846 |
| Profile-subtracted image | 0.136 | 0.129 | 0.127 |
| Overlay grid (source pixel centres) | 0.164 | — | — |
| Inversion setup (sparse, combined) | 9.488 | 13.628 | 20.160 |
| Data vector (D, w-tilde) | 0.221 | 0.190 | 0.149 |
| **Curvature matrix (F, w-tilde)** | **20.437** | **17.870** | **19.801** |
| Regularization matrix (H) | −0.123 ⚠ | 0.232 | 2.972 |
| **Regularized reconstruction** | **36.657** | **37.608** | **37.925** |
| Mapped recon + log evidence (sparse) | 2.280 | 2.288 | 2.226 |
| step-sum | 70.8 | 73.7 | 84.7 |

The `--split-setup` prefix rows (`setup_split`, unbatched ms):

| Prefix row | Rect D | Rect S | Del D | Del S | DelNN D | DelNN S |
|---|---:|---:|---:|---:|---:|---:|
| Border relocation | 1.088 | 1.022 | 1.072 | 1.114 | 1.073 | 1.069 |
| Overlay grid + interpolation / Triangulation + interpolation | 1.019 | 0.641 | 6.074 | 6.125 | 17.472 | 13.401 |
| Mapping matrix (dense) | 0.175 | — | 0.409 | — | −0.194 ⚠ | — |
| Blurred mapping matrix, PSF (dense) | 7.492 | — | 8.321 | — | 4.496 | — |
| Sparse triplets | — | −0.098 ⚠ | — | 0.083 | — | 1.523 |
| MGE operated basis + PSF-weighted data | — | 7.923 | — | 6.305 | — | 4.167 |

The sparse legs also report the three setup rows directly (`steps_sparse_setup_rows`,
independently compiled, not differenced — these are the numbers to quote):

| Row | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| Sparse triplets (data + curvature) | 0.170 | 0.167 | 0.180 |
| MGE operated basis (params prefix, 60 PSF-convolved columns) | 5.171 | 5.151 | 4.996 |
| PSF-weighted data | 0.156 | 0.193 | 0.136 |
| one-off `apply_sparse_operator` build (eager, not per-call) | 630 ms | 643 ms | 733 ms |
| `sparse_nnz` | 61,444 | 46,083 | **491,552** |

## Per-call at `vmap` 16

Only the combined inversion-setup row and the params→H prefix are re-timed under
`vmap` (`steps_vmap_per_call`, `setup_split_vmap`); the post-setup steps (D, F, solve,
log-evidence) have **no batched row in this grid** — for those, the runtime table's
`vmap` per-call is the only batched measurement.

| Row (per call @ `vmap` 16, ms) | Rect D | Rect S | Del D | Del S | DelNN D | DelNN S |
|---|---:|---:|---:|---:|---:|---:|
| Inversion setup (combined) | 9.109 | **0.751** | 13.957 | **5.797** | 15.319 | **7.154** |
| params→H prefix (`regularization_matrix_prefix`) | 0.180 | 0.174 | 5.481 | 5.256 | 9.961 | 7.376 |
| Border relocation | 0.093 | 0.065 | 0.068 | 0.064 | 0.065 | 0.066 |
| Overlay / Triangulation + interpolation | 0.047 | 0.078 | 5.070 | 5.068 | 9.006 | 6.458 |
| Mapping matrix (dense) | 0.195 | — | 0.121 | — | −2.457 ⚠ | — |
| Blurred mapping matrix, PSF (dense) | 8.501 | — | 8.358 | — | 8.427 | — |
| MGE operated basis + PSF-weighted data | — | 0.607 | — | 0.627 | — | 0.613 |

**The one place the sparse path is dramatically better is batched setup**: the
rectangular sparse inversion setup is 0.75 ms per call against 9.11 ms dense (12.1×),
because the 60-column MGE operated basis and the triplets amortise across the batch
while the dense (15,361 × 1521) blurred mapping matrix does not (8.50 ms per call at
`vmap` 16 — it does not amortise at all). That win is then given back, and more, by the
w-tilde F.

## The matrix-free "must beat" columns

What a matrix-free CG + SLQ formulation would have to replace, unbatched, per call:

**Dense** (mapping matrix + blurred mapping matrix + F + reconstruction + model image):

| Component | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| Mapping matrix | 0.175 | 0.409 | −0.194 ⚠ |
| Blurred mapping matrix (PSF) | 7.492 | 8.321 | 4.496 |
| Curvature matrix (F) | 4.826 | 4.824 | 4.830 |
| Regularized reconstruction (NNLS + Cholesky) | 36.936 | 37.394 | 37.310 |
| Mapped recon + log evidence (model image) | 2.344 | 2.214 | 2.275 |
| **total** | **51.77** | **53.16** | **48.72** |

**Sparse** (triplets + MGE basis + PSF-weighted data + w-tilde F + reconstruction + model image):

| Component | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| Sparse triplets | 0.170 | 0.167 | 0.180 |
| MGE operated basis | 5.171 | 5.151 | 4.996 |
| PSF-weighted data | 0.156 | 0.193 | 0.136 |
| Curvature matrix (F, w-tilde) | 20.437 | 17.870 | 19.801 |
| Regularized reconstruction (NNLS + Cholesky) | 36.657 | 37.608 | 37.925 |
| Mapped recon + log evidence (sparse) | 2.280 | 2.288 | 2.226 |
| **total** | **64.87** | **63.28** | **65.26** |
| vs dense | **+13.10** | **+10.11** | **+16.55** |

w-tilde F sub-rows (`steps_sparse_sub_rows`, each independently compiled, so they sum to
slightly more than the fused `Curvature matrix (F, w-tilde)` row):

| Sub-row | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| F diag (FFT blocks) | 20.826 | 18.442 | 19.908 |
| F off-diag (mapper × MGE) | 1.016 | 0.671 | 0.728 |
| F MGE × MGE GEMM | 0.443 | 0.400 | 0.415 |

**The trade the sparse path makes**: the w-tilde F (17.9–20.4 ms, essentially all of it
the FFT-block diagonal) replaces `blurred mapping matrix + dense F`, which is 12.32 /
13.15 / 9.33 ms on rectangular / Delaunay / DelaunayNN. That is **+8.12 / +4.72 /
+10.48 ms**, and it is the whole of the sparse per-call loss. Everything else on the
sparse side is neutral or better.

**The number that dominates both paths is the solve, not the matrix build.** The
regularized reconstruction (NNLS + Cholesky on the 1560/1581-parameter reduced system)
is 36.7–37.9 ms on every one of the six rows — 61–71 % of the "must beat" total, and
identical dense vs sparse to within 1.3 %, because it is byte-for-byte the same code on
the same reduced matrices. Any matrix-free scheme that leaves the solve alone can win at
most the remaining ~15 ms (dense) / ~26 ms (sparse) per call, and matrix-free CG replaces
exactly the part of the solve that is currently cheap relative to that NNLS.

## Log-det values

**Not recorded.** Neither the breakdown JSONs nor the SLURM stdout carry the Cholesky
log-det terms per row — the cells emit only the assembled `log_evidence` (the pin) and
step timings. If per-term log-det values are wanted for the matrix-free / SLQ comparison
(the natural place to check SLQ noise against exact Cholesky) the breakdown cells need a
new emission; file it as a follow-up rather than reading it out of this grid.

*Now recorded (2026-09-11, issue #243) — the per-term values are in
[Reconstruction split](#reconstruction-split-2026-09-11); this paragraph is kept as the
history of why they were missing from the 2026-09-10 grid.*

## Compile times

Full-pipeline compile from the runtime legs is in the runtime table above (13.3–17.6 s).
Per-step compile (`jit_phases[*].compile_s`, seconds; only steps over 0.4 s shown):

| Phase | Rect D | Rect S | Del D | Del S | DelNN D | DelNN S |
|---|---:|---:|---:|---:|---:|---:|
| inversion setup | 11.48 | 11.74 | 10.39 | 10.18 | 12.16 | 12.14 |
| `mge_operated_basis` | — | 10.95 | — | 11.05 | — | 8.67 |
| setup prefixes 5/6/(6s)/7/8 | 7.90 | 5.45 | 6.60 | 4.76 | 14.80 | 10.98 |
| regularization matrix | 2.32 | 2.53 | 2.99 | 3.39 | 4.47 | 3.78 |
| curvature matrix | 0.06 | 0.52 | 0.06 | 0.66 | 0.06 | 0.71 |
| reconstruction | 0.46 | 0.43 | 0.47 | 0.49 | 0.43 | 0.48 |
| **total over all `jit_phases`** | **23.3** | **33.6** | **21.9** | **32.7** | **33.4** | **39.1** |

The sparse legs cost ~10 s more to compile, essentially all of it the
`mge_operated_basis` step (8.7–11.0 s to compile the 60 PSF-convolved basis columns).

## ⚠ Negative differenced cells — artifact, not saving

Five cells above are negative. Every one of them is a `--split-setup` **prefix
difference** (`prefix_n − prefix_{n−1}`, each prefix independently compiled), so XLA's
DCE and fusion boundaries move work between adjacent prefixes and a cell can come out
below zero. They are attribution artifacts and are **not** to be "fixed" or read as
savings:

- rectangular `Regularization matrix (H)` −0.568 ms dense / −0.123 ms sparse
  (`prefix11 − prefix6`),
- DelaunayNN `Mapping matrix` −0.194 ms unbatched and −2.457 ms at `vmap` 16,
- rectangular sparse `Sparse triplets` −0.098 ms.

Judge the H row on the **params→H prefix** instead (`regularization_matrix_prefix_s`,
independently compiled): 1.540 / 1.540 ms unbatched and 0.180 / 0.174 ms per call at
`vmap` 16 (rect dense / sparse); 7.381 / 7.472 and 5.481 / 5.256 (Delaunay);
19.415 / 17.442 and 9.961 / 7.376 (DelaunayNN). Judge the triplets on
`steps_sparse_setup_rows` (0.167–0.180 ms), which is a direct timing.

## What this baseline says to the matrix-free work

1. **The target is the solve.** 37 ms of a ~50–65 ms per-call cost is NNLS +
   Cholesky on the reduced system, unchanged by the dense/sparse choice.
   *Refined 2026-09-11 — see [Reconstruction split](#reconstruction-split-2026-09-11):
   the target is the NNLS **iterations** (21–22 × ~1.7 ms), not the linear solve; one
   exact unconstrained solve of the same system is only 1.5–1.6 ms.*
2. **F is no longer the villain.** With `--xla_gpu_enable_triton_gemm=false` the dense
   F is 4.83 ms flat on all three meshes — 5.3× cheaper than the pre-flag 25.6 ms that
   originally motivated a matrix-free F. A matrix-free F has ~4.8 ms (dense) to beat,
   not ~25 ms.
3. **Sparse is still the memory lever, not the speed lever** — +6 to +14 % per call at
   `vmap` 16, ~7× less device memory, and 12× cheaper batched setup on the rectangular
   mesh. Its cost is concentrated in the FFT-block diagonal of the w-tilde F.
4. **DelaunayNN's `sparse_nnz` is 491,552** against 46,083 (Delaunay) and 61,444
   (rectangular) — 8–11× more non-zeros for the same 1500 vertices, because the Sibson
   stencil is far wider than the barycentric one. Any sparse/matrix-free costing that
   extrapolates from Delaunay to DelaunayNN by vertex count will be wrong.

## Artifacts

```
results/breakdown/imaging/{pixelization,delaunay,delaunay_nn}_hpc_a100_fp64{,_sparse}.{json,png}
results/runtime/imaging/pixelization/pixelization_hpc_a100_fp64{,_sparse}.json
results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64{,_sparse}.json
results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64{,_sparse}.json
```

Superseded by this note: recommendation #4 of
[`sparse_vs_dense_inversion_path.md`](./sparse_vs_dense_inversion_path.md) (defer
matrix-free) and every A100 pixelized/sparse row in that note's tables. The synthetic
GEMM probe in [`xla_autotune_triton_gemm.md`](./xla_autotune_triton_gemm.md) (arm E,
4.79 ms) is **confirmed** rather than superseded: the live dense F rows here are
4.824–4.830 ms on all three meshes, so nothing needs folding in.

## Reconstruction split (2026-09-11)

autolens_profiling issue [#243](https://github.com/PyAutoLabs/autolens_profiling/issues/243),
branch `feature/reconstruction-row-split`, code commit `353b9cf`. Four jobs re-run the
**dense** legs of the grid above with the reconstruction row instrumented: overlapping
sub-rows around steps 12/13, the PDIP iteration count threaded out of `jax_nnls`, and
every `log_evidence` term emitted. Written under the `_recon_split` output tag so the
2026-09-10 baseline JSONs above stay canonical.

### Provenance and gate

| Job | Cell | Task | Start (2026-09-11 BST) | Elapsed |
|---|---|---|---|---:|
| 342643 | pixelization | breakdown, dense | 00:04:28 | 1:45 |
| 342644 | delaunay | breakdown, dense | 00:06:14 | 1:36 |
| 342645 | delaunay_nn | breakdown, dense | 00:07:50 | 2:27 |
| 342646 | pixelization | runtime, dense | 00:10:18 | 0:55 |

All four `COMPLETED` on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe), fp64,
`PyAutoLens 2026.8.17.1`, same three XLA flags (Triton GEMM off), a **fresh per-job**
`JAX_COMPILATION_CACHE_DIR` (`output/jax_cache/recon_split_<jobid>`),
`autotune_cache_entries_at_start: 0`, `cache_fresh: true`, `AUTOTUNE_ENTRIES count=0` at
exit. Gate: dense `Curvature matrix (F)` 4.821 / 4.860 / 4.820 ms (threshold < 6 ms), no
`cache_fresh: false`, `pinned_drift: []` on both rectangular legs, `nnls.converged: true`
on all three, and the cell-driven NNLS reconstruction matches the library's to
4.7e-10 / 7.6e-10 / 1.2e-9 (tolerance 1e-8). The eager regression pins PASSED at
28621.128714 (rect, relative difference 2.0e-14), 29140.295882, 29277.464588; the
rectangular **runtime** pin is now `28621.128714095972` and PASSED, closing the
bookkeeping wrinkle flagged above. Runtime leg: single-JIT 50.66 ms, 32.10 ms per call at
`vmap` 16 (baseline 50.63 / 32.08).

The shared RAL library install was refreshed between 2026-09-10 and this re-run
(PyAutoFit `e354dbb6`, PyAutoArray `667deed3`, PyAutoGalaxy `6640a749`,
PyAutoLens `0da06de6`; PyAutoNerves unchanged at `0e7163bc`). Every step row lands within
0.7 % of the 2026-09-10 dense numbers, so the split below is directly comparable to the
tables above.

### The split — unbatched per-call (ms)

**These sub-rows overlap the reconstruction row; they are not a partition of it.** Each
is an independently compiled `jit` over the *same* `F + λH` / `D` the fused step-12 unit
receives, so they are comparators ("what would this piece cost on its own?"), never
addends. Step 12 remains `reconstruction_positive_only_from` and is unchanged; the
cell-driven `solve_nnls` row is its instrumented twin.

| Row | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| **Regularized reconstruction** (step 12, timed) | **36.962** | **37.386** | **37.569** |
| NNLS PDIP (cell-driven, max_iter 50) | 37.124 | 37.316 | 37.304 |
| — PDIP iterations to convergence | 21 | 22 | 22 |
| — ms per PDIP iteration | 1.768 | 1.696 | 1.696 |
| NNLS PDIP, one iteration (upper bound) | 3.318 | 3.166 | 3.184 |
| NNLS PDIP @ `vmap` 16, identical lanes | 21.874 | 21.926 | 21.911 |
| Cholesky (F+λH) | 1.263 | 1.194 | 1.235 |
| Cholesky solve (unconstrained, factorise + 2 triangular solves) | 1.556 | 1.576 | 1.522 |
| Log det Cholesky (F+λH reduced) | 1.271 | 1.156 | 1.164 |
| Log det Cholesky (H reduced) | 1.191 | 1.147 | 1.184 |
| **Mapped recon + log evidence** (step 13, timed) | **2.412** | **2.233** | **2.261** |

**The "one iteration" row is an upper bound, not the marginal cost.**
`solve_nnls(..., max_iter=1)` pays the whole solver entry — Jacobi scaling of the
returned arrays, the initial interior point, and the first KKT factorisation — which the
`lax.while_loop` pays once and then amortises. It is ~1.9× the amortised 1.70–1.77 ms per
iteration for exactly that reason. Read `row / iterations` as the per-iteration cost and
the one-iteration row as the ceiling on what removing a single iteration buys.

**The `vmap` 16 row is the best case for batching, not a representative one.** All 16
lanes are `broadcast_to` copies of the same `(Q_pc, q_pc)`, so they converge on the same
PDIP iteration and the `lax.while_loop` never runs on for a straggler. A real batch of
distinct parameter draws runs until its slowest lane converges. Even in this best case the
NNLS amortises only **1.70×** (37.1 → 21.9 ms per call), against 12× for the batched
sparse inversion setup.

### `log_evidence_terms` — the exact Cholesky log-dets

Now emitted per leg (`log_evidence_terms` in each breakdown JSON). The two log-det terms
are the exact dense-Cholesky values an SLQ estimator is to be checked against:

| Term | Rectangular | Delaunay | DelaunayNN |
|---|---:|---:|---:|
| χ² | 19992.673552 | 19821.951111 | 19745.505861 |
| regularization term | 204.864985 | 928.937527 | 800.447160 |
| log det (F+λH) | 3888.258090 | 8360.401763 | 7224.568778 |
| log det (H) | 1692.786817 | 7756.614959 | 6690.183752 |
| noise normalization | −79635.267237 | −79635.267237 | −79635.267237 |
| **log evidence** | **28621.128714** | **29140.295898** | **29277.464595** |

The rectangular log-dets are ~2× smaller than the Delaunay family's because its
regularization is `Constant(1.0)` while the Delaunay cells use `adapt_split`
(0.1 / 10.0 / 0.1) — compare terms within a mesh, not across.

### What this says to the matrix-free work

**The 37 ms is not a linear solve — it is 21–22 PDIP iterations at ~1.7 ms each, and
every one of them is a fresh dense KKT Cholesky inside `jaxnnls`.** A single Cholesky of
the same `F + λH` is **1.19–1.26 ms**, and a full *exact* unconstrained solve (that
factorisation plus two triangular solves) is **1.52–1.58 ms**. The positivity constraint
therefore costs about **24×** the exact solve it replaces.

The consequence for matrix-free CG + SLQ is blunt: **a CG line that only replaces the
linear solve has ~1.6 ms per call to beat, not 37 ms** — plus the ~2.4 ms of the two
log-det Choleskys (1.15–1.27 ms each), which is the part SLQ actually targets. Even a CG
solve that were free would take ~4 ms off a ~50–65 ms call, and would leave the 37 ms
untouched.

The lever on the reconstruction row is the **NNLS iteration count**, not the cost of a
factorisation: warm starts across likelihood evaluations, a looser `solver_tol`, a lower
`max_iter`, or a different treatment of positivity altogether (project, penalise, or drop
it and justify the drop). Batching does not rescue it either — under `vmap` 16 with
*identical* lanes the NNLS amortises only 1.7× (21.9 ms per call), so a prototype must
either attack the iteration count or make the case for dropping positivity before any CG
work is worth doing. This supersedes item 1 of "What this baseline says to the
matrix-free work" above: the target is the NNLS, not the solve.

### Artifacts

```
results/breakdown/imaging/{pixelization,delaunay,delaunay_nn}_hpc_a100_fp64_recon_split.{json,png}
results/runtime/imaging/pixelization/pixelization_hpc_a100_fp64_recon_split.json
```

New JSON keys on the breakdown legs: `steps_reconstruction_sub_rows`, `nnls`,
`log_evidence_terms`, `steps_reconstruction_vmap_note`; new `jit_phases` entries
`cholesky_curvature_reg_jit`, `cholesky_solve_jit`, `nnls_pdip_jit`,
`nnls_pdip_one_iteration_jit`, `nnls_pdip_vmap16`, `log_det_curvature_reg_jit`,
`log_det_regularization_jit`. Sub-row compile is cheap (0.07–0.10 s for the Cholesky
rows, 0.43–0.46 s for each NNLS row — the same order as the step-12 `reconstruction_jit`
at 0.45 s), so the instrumentation adds ~1.1 s of compile and no per-call cost to the
timed steps. `log_det_regularization_jit.compile_s` is ~1e-5 s on every leg: it is a
cache hit on the identically shaped `log_det_cholesky` traced one row earlier, not a
mis-measurement.
