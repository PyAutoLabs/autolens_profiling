# Numba CPU vs JAX-CPU interferometer curvature — the verdict (autolens_profiling#226)

Phase 2 of the `numba-interferometer-revisit` epic. Phase 1 (#223 / PR #225) recovered the
deleted numba w-tilde interferometer likelihood into `scripts/misc/numba_interferometer/`
with parity bit-identical to the JAX sparse path, and showed the
`F: mapper×mapper` row is 45–82 % of a CPU evaluation. This phase asks whether that row can
be made fast enough to justify reinstating a numba CPU interferometer path in PyAutoArray.

**Every number below is read from a committed JSON under
`results/breakdown/interferometer/`.** Nothing is quoted from the design review that
motivated the task; where the two disagree it is said so.

## Recommendation

**(d) and (a), in that order.**

**(d) — do this regardless of anything else.**
`InterferometerSparseOperator.apply_operator` pads a *real* input to `(2y, 2x)` and calls
complex `jnp.fft.fft2` / `ifft2`. On identical inputs, the real-transform route
(`rfft2`/`irfft2`) costs **1.27–1.61× less** than the complex one on the same stack
(`fft2_numpy` ÷ `rfft2_numpy`: 1.38 / 1.48 / 1.61 at sma / alma / alma_high on the Delaunay
mesh, 1.38 / 1.27 / 1.43 on the rectangular one). It is a ~10-line change, it is exact
rather than approximate, and it helps **every** user on **every** backend, GPU included. It
is the fair baseline this whole comparison should have been run against, not a lever.

**(a) — reinstate a numba CPU curvature kernel, but the new one, and geometry-gated.**
The *recovered* kernel is not worth reinstating: of the six synthetic cells it beats a plain
NumPy `rfft2` convolution at exactly one (sma Delaunay, 1.86×), loses the three other
measured cells, and at alma_high its extrapolated cost is 19–43× the FFT's. What is worth reinstating is the
**`direct_conv` extent-grid convolution written here**, which beats the JAX/FFT path's F row
by **7.0× (sma Delaunay), 4.1× (sma rectangular), 2.0× (alma Delaunay) and 1.10× (alma
rectangular)** in situ, and beats the recovered kernel by 2.9–4.8×. It should be selected by
a measured rule — non-zeros per source column below ≈ 60–80 — not unconditionally, because
above that the FFT wins and by alma_high it wins by 1.6–3.7×.

A NumPy `rfft2` application path (no JAX) is worth having for the other side of that
crossover: `rfft2_numpy` already beats the library's own JAX-CPU route by **1.10–1.30×** at
every cell measured, and it removes JAX from the CPU-only user's critical path entirely.

**What would change this verdict:** the control variable is `nnz / source column`
= `N_pix · P / S` — masked pixels times mapper neighbours per pixel, divided by source
pixels. The fiducials measured here sit at 7.7 (sma Delaunay) to 314 (alma_high
rectangular). If typical users run much larger real-space grids at fixed source-pixel count
(a wider mask, or a finer pixel scale), they move up that axis and the numba lever
evaporates; if they run more source pixels on the same grid they move down it and the lever
grows. Nothing else in the measurement is close to the decision boundary.

## How this was measured

| | |
|---|---|
| Host | i9-10885H, WSL2 (`Linux-5.10.16.3-microsoft-standard-WSL2`), 8 physical cores, 15 GB |
| Single-thread arms | `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` |
| JAX arms | additionally `JAX_PLATFORMS=cpu`, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"` (PyAutoNerves appends `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0`; recorded verbatim in every JSON) |
| Versions | autoarray/autolens `2026.8.17.1`, numba `0.62.1`, scipy `1.17.1`, jax `0.10.2`, numpy `2.2.6` |
| Protocol | 5 rounds per cell, **round-robin across kernels** (never one arm's rounds batched), round 0 discarded, median of the rest |
| Pin | every kernel vs the reference kernel's `F` at `rtol=1e-10`, `atol = 1e-10·max|F_ref|`, **before** any of its timings are recorded; a ×1.01 control must fail the same pin (it does, in all six cells) |
| Controls | a fixed 1500×1500 `dgemm` at the head and tail of every in-situ arm and every bake-off cell; a Cholesky of the arm's own `F + H` |

### The machine drifts by 2.5×; the ratios do not

This laptop throttles hard under sustained load, and the bake-off runs for the best part of
an hour. `bakeoff_machine_drift_sma_v2026.8.17.1.json` re-ran the whole `sma` cell on a quiet
machine: **every one of the nine kernels came out 2.37–2.83× faster** (mean 2.66×), while the
ratio that decides the verdict moved by 8 % (`rfft2 / direct_conv` 5.93× → 5.49×). Round-robin
interleaving is what buys that: within a round every kernel sees the same clock.

Read the **ratios** as the result. Absolute seconds are only comparable within a cell, and
the quiet-machine `sma` column is the one to compare against another host.

## 1. Synthetic bake-off (`bakeoff_v2026.8.17.1.json`)

Geometries are the real `instruments/interferometer.py` masks (`aa.Mask2D.circular`); the
`W~` preload and the mapper triplets are seeded synthetic arrays **shared by every kernel**,
so a timing difference is a kernel difference. `W~` has no compact support for an
interferometer, so its values cannot change any kernel's work — only the geometry can.

Median seconds per curvature build, single thread. `×` is `rfft2_numpy / kernel`, so >1 means
the kernel beats the NumPy real-FFT baseline.

### Delaunay mesh — S = 1500 source pixels, P = 3 neighbours per pixel

| kernel | sma (N=3852, M=4900, nnz/col 7.7) | alma (N=15380, M=19600, nnz/col 30.8) | alma_high (N=61572, M=78400, nnz/col 123.1) |
|---|---|---|---|
| `reference` (recovered) | 0.6789 s — 1.86× | 12.9364 s — 0.34× | *skipped, extrapolated ~207 s* |
| `hoisted` | 0.6519 s — 1.94× | 11.5540 s — 0.38× | *skipped, extrapolated ~185 s* |
| `symmetric` | 0.4076 s — 3.11× | 7.0574 s — 0.62× | 77.3487 s — 0.14× |
| `two_stage` | 0.2363 s — 5.36× | 4.4416 s — 0.99× | 34.3916 s — 0.32× |
| **`direct_conv`** | **0.2135 s — 5.93×** | **2.8522 s — 1.54×** | 17.4024 s — 0.63× |
| `source_loop` (recovered `…_from_2`) | 0.5083 s — 2.49× | 8.7508 s — 0.50× | 91.8143 s — 0.12× |
| `rfft2_numpy` | 1.2660 s — 1.00× | 4.3951 s — 1.00× | **11.0346 s — 1.00×** |
| `fft2_numpy` (= the library's algorithm) | 1.7479 s — 0.72× | 6.5003 s — 0.68× | 17.7362 s — 0.62× |
| `fft2_jax` (= the library) | 1.5239 s — 0.83× | 5.5156 s — 0.80× | 12.9655 s — 0.85× |

### Rectangular mesh — S = 784 source pixels, P = 4 neighbours per pixel

| kernel | sma (nnz/col 19.7) | alma (nnz/col 78.5) | alma_high (nnz/col 314.1) |
|---|---|---|---|
| `reference` | 1.1004 s — 0.67× | 13.0487 s — 0.20× | *skipped, extrapolated ~209 s* |
| `hoisted` | 1.0708 s — 0.69× | 16.8513 s — 0.16× | *skipped, extrapolated ~270 s* |
| `symmetric` | 0.6746 s — 1.09× | 10.7362 s — 0.25× | 63.1395 s — 0.08× |
| `two_stage` | 0.3210 s — 2.29× | 4.4579 s — 0.60× | 30.6292 s — 0.16× |
| **`direct_conv`** | **0.2598 s — 2.83×** | 2.6843 s — 0.99× | 17.9475 s — 0.27× |
| `source_loop` | 0.8051 s — 0.91× | 15.6739 s — 0.17× | *skipped, extrapolated ~251 s* |
| `rfft2_numpy` | 0.7361 s — 1.00× | **2.6557 s — 1.00×** | **4.8414 s — 1.00×** |
| `fft2_numpy` | 1.0124 s — 0.73× | 3.3786 s — 0.79× | 6.9360 s — 0.70× |
| `fft2_jax` | 0.8510 s — 0.86× | 2.9317 s — 0.91× | 6.2817 s — 0.77× |

At `alma_high` the `O(N² P²)` pair loops were extrapolated from the measured `alma` value by
`N²` first and only run when the estimate fitted a 180 s budget; `reference` and `hoisted`
did not, so they are recorded as `skipped: extrapolated` with the estimate rather than as a
number nobody waited for. The `alma_high` cells are pinned against `direct_conv`, which is
itself pinned against `reference` at both smaller geometries.

**Levers that did not pay.** `hoisted` — the reference pair loop with the destination row,
the `ip0` weight and the mapper gathers hoisted out of the inner loop — is worth
**0.77–1.12×** across the four cells where it ran, i.e. nothing, and it is *slower* than the
reference at alma rectangular (16.85 s vs 13.05 s). Symmetric halving gives **1.22–1.83×**,
not the 2× the operation count implies, because the extra loop bound and the `A + Aᵀ` mirror
eat the rest. Both apply only to the pair-loop form, which is the form that loses anyway.

## 2. The measured crossover

The controlling ratio is not `N`, `M` or `S` separately. Direct convolution costs
`O(nnz·M + S·nnz)` and the FFT costs `O(S·M·log M)`, so

```
    FFT time / direct-conv time  ∝  (S · M log M) / (nnz · M)  =  log M / (nnz/S)
```

— `M` cancels except inside the logarithm, and the control variable is **non-zeros per source
column**, `nnz/S = N_pix·P/S`. Log-interpolating the measured `rfft2 / direct_conv` ratios
through 1.0 gives the crossover at

| mesh | measured points | crossover |
|---|---|---|
| Delaunay | 7.7 → 5.93×, 30.8 → 1.54×, 123.1 → 0.63× | **nnz/col ≈ 60** |
| Rectangular | 19.7 → 2.83×, 78.5 → 0.99×, 314.1 → 0.27× | **nnz/col ≈ 77** |

So: **the FFT wins once a source column touches more than roughly 60–80 image pixels.** The
2026-09-07 design review put this at 35–45; the measurement puts it about 1.7× higher, which
moves alma from "roughly level" into "numba by 1.5×" on the Delaunay mesh.

### Measured constants

Seconds per multiply-accumulate, from the `operations` field beside each timing
(`median_s / operations`, in ns):

| kernel | sma D | alma D | alma_high D | what the constant is |
|---|---|---|---|---|
| `reference` | 5.08 | 6.08 | — | scatter with 2-D gathers |
| `symmetric` | 6.10 | 6.63 | 4.53 | as above, plus a branch and a mirror |
| `two_stage` | 3.82 | 5.70 | 2.95 | dense `acc[S]` accumulate + row AXPY |
| `direct_conv` | 2.89 | 2.93 | 1.18 | contiguous AXPY over a preload row |
| `source_loop` | 7.61 | 8.22 | 5.38 | two-level indirect gather |
| `rfft2_numpy` | 2.83 | 2.17 | 1.22 | pocketfft butterflies |

`direct_conv` is not winning on operation count — at sma Delaunay it does 7.40e7 MACs against
the reference's 1.34e8, only 1.8× fewer — it is winning because its inner loop is a
contiguous `u[base+jx] += w · preload[dy, off+jx]` AXPY that vectorises, at **2.9 ns/op
against the reference's 5.1**, and because it replaces the `N²` pixel-pair space with the
`M`-cell extent rectangle. That is the same reason the imaging two-stage accumulator won,
transplanted to a kernel with no compact support.

## 3. Thread scaling (`bakeoff_threads_{1,8}t_v2026.8.17.1.json`)

Delaunay mesh, 5 rounds, all four thread variables set explicitly and recorded.

| cell | arm | 1 thread | 8 threads | speed-up |
|---|---|---|---|---|
| sma | `direct_conv` serial (control) | 0.0809 s | 0.0933 s | 0.87× |
| sma | `direct_conv_parallel` (`prange`) | 0.0816 s | 0.0232 s | **3.52×** |
| sma | `rfft2_numpy` (`workers=`) | 0.4868 s | 0.3246 s | 1.50× |
| alma | `direct_conv` serial (control) | 0.9068 s | 0.9139 s | 0.99× |
| alma | `direct_conv_parallel` (`prange`) | 0.9588 s | 0.2072 s | **4.63×** |
| alma | `rfft2_numpy` (`workers=`) | 2.0266 s | 0.8079 s | 2.51× |

The serial rows are the control: they must not move, and they do not. The `prange` loop is
over source columns, each owning its own accumulator and writing only its own row of `F`, so
it needs neither a reduction nor a lock — which is why it scales better than the FFT's
`workers=`.

> **Pool caveat, recorded verbatim in both JSONs:** *"kernel threads are unavailable under a
> Nautilus multiprocessing pool; the single-thread number is the production number"*.

A model fit runs the likelihood inside a process pool that already owns every core, so the
8-thread column is a ceiling for an interactive single evaluation, not a number to plan a
search around.

## 4. In-situ (per-kernel `*_breakdown_*` JSON/PNG)

The real likelihood: `delaunay_numba.py` (Hilbert-1500 → Delaunay + `ConstantSplit`) and
`pixelization_numba.py` (32×32 rectangular + `Constant`), `Isothermal` + `ExternalShear`,
`n_repeats` 10 at sma and 3 at alma. Arms are interleaved and each is run twice; the first
pass after every arm switch is discarded (numba compile, cold pages) and the second is what
is committed. `--kernel jax` swaps the whole inversion for `InversionInterferometerSparse`
on the same fit.

**Every arm at a given (instrument, mesh) returns the identical log evidence** — sma Delaunay
`-3169.6493766794806` (the phase-1 pinned value), sma rectangular `-3168.280345651575`, alma
Delaunay `-12050107.615448244`, alma rectangular `-12049070.902051926`. The Cholesky control
is arm-invariant (0.039–0.046 s at S=1500, 0.0145–0.0194 s at S=784) and the head/tail
`dgemm` controls agree to within 25 % on every arm (drift 0.75–1.20×).

| instrument | mesh | arm | whole likelihood | `F` row | `F` row, jit-warm |
|---|---|---|---|---|---|
| sma | Delaunay | `reference` | 0.5167 s | 0.2475 s | — |
| sma | Delaunay | `symmetric` | 0.4463 s | 0.1608 s | — |
| sma | Delaunay | `source_loop` | 0.4424 s | 0.1674 s | — |
| sma | Delaunay | `two_stage` | 0.3806 s | 0.0982 s | — |
| sma | Delaunay | **`direct_conv`** | **0.3642 s** | **0.0844 s** | — |
| sma | Delaunay | `jax` (sparse-op FFT) | 2.2031 s | 0.6134 s | 0.5917 s |
| sma | rect | `reference` | 0.4727 s | 0.3967 s | — |
| sma | rect | **`direct_conv`** | **0.1736 s** | **0.0998 s** | — |
| sma | rect | `jax` | 1.7992 s | 0.5607 s | 0.4101 s |
| alma | Delaunay | `reference` | 4.4195 s | 4.0710 s | — |
| alma | Delaunay | **`direct_conv`** | **1.4336 s** | **1.2250 s** | — |
| alma | Delaunay | `jax` | 8.1049 s | 3.0947 s | 2.4297 s |
| alma | rect | `reference` | 6.5208 s | 7.0205 s | — |
| alma | rect | **`direct_conv`** | **1.8367 s** | **1.4631 s** | — |
| alma | rect | `jax` | 6.5739 s | 2.1783 s | 1.6110 s |

### The JAX arm's whole-likelihood column is not a production number

`InversionInterferometerSparse` exposes `curvature_matrix` and `data_vector` as plain
(uncached) properties, so an eager `FitInterferometer.figure_of_merit` rebuilds `F` once for
`reconstruction`, again for `fast_chi_squared` and again for the log-determinant. Under
`jax.jit` — which is how production runs it — those fold into one. The `F` row is therefore
the like-for-like comparator, and it is given both eagerly and jit-warm. The two are within
4–37 % of each other, because the operator's `lax.fori_loop` is already staged into a single
XLA computation even outside `jit`; the jit-warm number is the one used below.

Substituting the jit-warm `F` into the numba arm's measured non-`F` cost gives a fair
whole-evaluation estimate:

| cell | non-`F` cost | `direct_conv` | JAX-CPU (jit `F`) | ratio |
|---|---|---|---|---|
| sma Delaunay | 0.269 s | 0.354 s (measured 0.364) | 0.861 s | **2.43×** |
| sma rect | 0.076 s | 0.176 s (measured 0.174) | 0.486 s | **2.76×** |
| alma Delaunay | 0.349 s | 1.574 s (measured 1.434) | 2.778 s | **1.77×** |
| alma rect | 0.292 s | 1.755 s (measured 1.837) | 1.903 s | **1.08×** |

> **Reading the dashboard tables:** the auto-generated headline rows report
> `total_step_by_step`, and for the `*_numba_jax` rows that is a **partial** total by
> construction (the arm records only the steps it can measure directly — see the paragraph
> above and `steps_are_partial: true` in those JSONs). The `F` row and the substituted
> whole-evaluation estimate in this section are the numbers to compare across arms; the
> `*_numba_jax` dashboard entry is not the JAX arm's whole likelihood.

### What the in-situ `F` rows include

The variant kernels consume flat CSR/CSC arrays; the pack marshals the mapper's
`[N_pix, P]` triplets into that layout inside the `F` step (`kernel_index_arrays`), so the
numbers above **charge each variant its own input preparation**. The marshalling is fully
vectorised NumPy, but a library implementation would not pay it at all — the mapper can emit
the layout directly, exactly as `_sparse_triplets_curvature_from` already emits COO for the
FFT path. The in-situ ratios are therefore conservative for `direct_conv`; the bake-off
ratios (inputs prebuilt) are the pure-kernel ones.

## 5. Kill gate

> *If no numba kernel beats `rfft2_numpy` by >1.3× at sma **or** alma, stop the numba lever
> work.*

**Passed**, at both geometries and on both meshes' best case:

| cell | best numba | median | `rfft2_numpy` | ratio |
|---|---|---|---|---|
| sma / Delaunay | `direct_conv` | 0.2135 s | 1.2660 s | **5.93×** |
| sma / rect | `direct_conv` | 0.2598 s | 0.7361 s | **2.83×** |
| alma / Delaunay | `direct_conv` | 2.8522 s | 4.3951 s | **1.54×** |
| alma / rect | `direct_conv` | 2.6843 s | 2.6557 s | 0.99× |

## 6. Where this leaves the epic

- The recovered numba kernel is **not** the thing to reinstate. It beats a NumPy `rfft2`
  convolution only at sma Delaunay, and `direct_conv` beats it by 2.9–4.8× on the in-situ
  `F` row and by 3.2–4.9× in the bake-off.
- The library's CPU story is currently "JAX or nothing":
  `InterferometerSparseOperator.curvature_matrix_*_from` imports `jax.numpy` in the method
  body, so a no-GPU user has no NumPy application path at all. Two of the three
  recommendations above (the `rfft2` change and a NumPy application path) are about that, and
  neither depends on numba.
- Phase 3 (`interferometer_preload_cpu.md`) is unaffected and now better motivated: the
  `O(N_pix·K)` preload took **≈33 minutes** single-threaded at alma (1 M visibilities,
  280×280 offsets) in this campaign — longer than every likelihood measurement in this note
  put together. The pack now caches it beside the dataset (git-ignored) so a paired sweep
  pays it once; every in-situ JSON records `preload_cache_hit`.

## Reproducing

```bash
source activate.sh   # worktree root
cd autolens_profiling

python -m pytest scripts/misc/numba_interferometer/ -q            # the pins

OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1" \
  JAX_PLATFORMS=cpu python scripts/misc/numba_interferometer/bakeoff.py --reps 5

OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  python scripts/misc/numba_interferometer/bakeoff.py --mode threads --threads 8 \
  --geometries sma,alma --meshes delaunay --reps 5

for k in reference symmetric two_stage direct_conv source_loop jax; do
  OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python scripts/interferometer/likelihood_breakdown/delaunay_numba.py --kernel "$k"
done
```
