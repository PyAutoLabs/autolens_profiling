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

## 7. The preload

Phase 3 of the epic (autolens_profiling#229). Every number below is read from
`results/breakdown/interferometer/preload_breakdown_{sma,alma,alma_high}_v2026.8.17.1.json`,
written by `scripts/interferometer/likelihood_breakdown/preload_numba.py`.

Section 6 left the preload as the epic's largest unmeasured cost: the `O(N_pix·K)` build
every w-tilde arm pays once per dataset, numba and JAX alike, before a single likelihood is
evaluated. It is not a per-evaluation step, so no breakdown cell had ever timed it. This
phase times it four ways and tests the design review's hypothesis that the array is exactly
`Re` of a **type-1 (adjoint) NUFFT**. It is — and that is worth two to three orders of
magnitude.

### The construction

The brute-force builders compute, over the mask's bounding extent `(Ny, Nx)`,
`P[i, j] = Σ_k w_k cos(2π(dx·u_k + dy·v_k))` with `w_k = 1/σ_k²`, `dx = −j·Δ_rad`,
`dy = +i·Δ_rad` on autoarray's radian grid, the four quadrants filled from the four corners
into a `[2Ny, 2Nx]` wraparound array with the middle row `Ny` and column `Nx` left zero as
FFT padding. With the transformer's own scaled frequencies `x_k = 2π u_k Δ_rad`,
`y_k = 2π v_k Δ_rad` (`transformer.py:334-335`) that is `P[i,j] = Re Σ_k w_k exp(i(−j·x_k + i·y_k))`,
i.e.

```
f = nufftax.nufft2d1(−x, y, w, n_modes=(2Nx, 2Ny), eps, isign=+1)
P = ifftshift(Re f);   P[Ny, :] = 0;   P[:, Nx] = 0
```

Pinned empirically at sma rather than assumed
(`test_parity.py::test_preload_via_nufft_matches_modern_np_builder`). Of the eight candidate
mappings (axis swap × sign of `x` × sign of `y`) exactly **two** reproduce the brute force —
`(−x, +y)` and `(+x, −y)` — at `max|Δ| = 1.7e-17`, i.e. `8.7e-14` of the peak `P[0,0]`. The
other six are wrong by `2.1e-1` of the peak. The identification is not marginal: thirteen
orders of magnitude separate right from wrong. The two survivors are the same construction
(`w` is real, so the two transforms are conjugates and their real parts are equal), which is
also why `P[i,j] == P[−i,−j]`. `fftshift` vs `ifftshift` is not a choice here either: both
axes have even length `2N`, for which the two shifts are the same permutation. The padding
row/column is at index `N`, not `N−1` — after `ifftshift` that index carries the Nyquist
mode, which the brute force never evaluates and the NUFFT does, so it is zeroed explicitly.

Pinned separately: the exactly-zero padding row/column (and that its neighbours are *not*
zero, so the pin is not vacuous), the `P[i,j] == P[−i,−j]` evenness, invariance to
visibility chunking, a square-pixel guard, and a control in which a wrong permutation must
fail the parity rule.

### The four builders

Wall seconds, with CPU-seconds in brackets. `parity` is `max|Δ| / P[0,0]` against the NumPy
builder's array. sma is the median of 3 repeats after a discarded warm-up; alma's
brute-force trio is one build each (a repeat is half an hour) and its NUFFT row is 3
repeats.

| instrument | `K` | `N_pix` | preload | `numba` | `numpy` (library default) | `jax_cpu` | **`nufft`** (`eps=1e-12`) |
|---|---|---|---|---|---|---|---|
| sma | 190 | 3 852 | 140×140 | 0.2537 s (0.254) | 0.1643 s (0.164) | 0.7360 s (1.891) | **0.0210 s (0.031)** |
| alma | 1 000 000 | 15 380 | 280×280 | 3258.6 s (3235.7) | 2101.5 s (2097.1) | 1046.2 s (2705.2) | **7.278 s (18.84)** |
| alma_high | 5 000 000 | 61 572 | 560×560 | refused | refused | refused | **22.15 s (89.99)** |

**At alma the NUFFT builder replaces a 35-minute build with 7 seconds** — `289×` on wall
clock, `111×` on CPU-seconds against the library's own NumPy default. The recovered numba
builder is the *slowest* of the four (54.3 min, 1.55× the NumPy one); `jax_cpu` is the
fastest brute force at 17.4 min but only by spending 2.59 cores.

| instrument | `numba` parity | `numpy` parity | `jax_cpu` parity | `nufft` parity | pin |
|---|---|---|---|---|---|
| sma | 2.3e-15 | 0 (reference) | 3.3e-16 | 8.7e-14 | all four **pass** |
| alma | **2.2e-11 — FAILS** | 0 (reference) | 1.4e-16 | 1.9e-14 | three of four pass |
| alma_high | — | — | — | — | no affordable reference; eps self-consistency only |

**The one pin that fails is the recovered numba builder at alma**, at `2.195e-11` of the
peak against a bound of `1e-11` (`10·eps·P[0,0]`, `eps=1e-12`). It is recorded, not
loosened. What it means: the numba kernel accumulates all `10⁶` visibilities into one
running scalar per offset, while the NumPy reference sums them in `chunk_k = 2048` blocks,
so the two differ by summation-order round-off that grows with `K` — `2e-15` at sma's
`K = 190`, `2e-11` at alma's `K = 10⁶`, on 11 of 78 400 entries. That is a floating-point
property of the recovered kernel, not an algebraic disagreement, and it is not the NUFFT's
problem: the **NUFFT agrees with the reference a thousand times better than the numba brute
force does** (`1.9e-14` vs `2.2e-11`). It does mean the eps-scaled bound is the wrong ruler
for an exact-but-unchunked builder, and that a `10⁶`-visibility naive accumulation is worth
knowing about on its own.

`alma_high` hard-refuses the brute-force builders: `O(61572 × 5e6)` is hours per builder, and
nothing here needs it — the NUFFT builder is pinned where a pin is affordable.

### Threading — read both columns

`OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1` pin NumPy's BLAS and numba to one core, but XLA's CPU
runtime keeps its own intra-op pool that `--xla_cpu_multi_thread_eigen=false` does not close.
The two JAX-backed builders therefore run multi-core (`jax_cpu` 2.57× at sma and 2.59× at
alma; `nufft` 1.48× at sma, 2.59× at alma, 4.06× at alma_high) while the NumPy and numba
ones are measured at 1.00× and 0.99×. Every row records `build_cpu_s` (`time.process_time()`, summing all threads) and
`threads_effective`; quote the CPU-seconds column before claiming a speed-up over a
single-threaded loop.

### Accuracy: which `eps` holds the pin

Each `eps`'s preload is pushed through `apply_sparse_operator` →
`NumbaPreload.from_curvature_preload` → `InversionInterferometerNumba(kernel="direct_conv")`
→ `fit_util.log_evidence_from` and compared with the same chain on the brute-force preload.

| `eps` | build (sma / alma / alma_high) | peak-scaled parity (sma / alma) | log-evidence rel diff (sma / alma) | pin `rtol=1e-6` |
|---|---|---|---|---|
| `1e-6`  | 0.0195 / 2.26 / 9.30 s | 4.8e-08 / 5.7e-09 | 1.8e-11 / 5.2e-12 | **holds** |
| `1e-9`  | 0.0138 / 3.19 / 16.12 s | 6.6e-10 / 4.7e-11 | 2.0e-14 / 3.9e-14 | **holds** |
| `1e-12` | 0.0144 / 5.46 / 22.73 s | 8.7e-14 / 1.9e-14 | 1.4e-16 / **0** | **holds** |

**Every `eps` in the sweep holds the `rtol=1e-6` log-evidence pin**, by five orders of margin
even at the loosest. At `eps=1e-12` and alma the log evidence is *bit-identical* to the brute
force's (`-12050103.936303042`). The recommendation is nonetheless `eps=1e-12`: it is the
only value that also holds the *array* pin at sma, it saturates fp64 (`eps=1e-14` moves
`max|Δ|` only from `1.7e-17` to `1.4e-17`), and it costs seconds. A preload is reused by
every likelihood call in a fit; error budget spent here is spent for the whole run.

The two parity rules are both reported for every row, so neither has to be taken on trust.
At sma the rule is **mixed** — `allclose(rtol=1e-10, atol=1e-10·P[0,0])` — because a type-1
NUFFT bounds its error against `Σ_k|c_k|`, i.e. against the peak, so the preload's near-zero
entries (five orders below it, at the fp64 noise floor) carry no relative guarantee and a
relative-only test there measures round-off rather than the builder; at `eps=1e-12` the
strict elementwise relative error still reaches `6.0e-10` on 32 of 19 600 entries. At alma
the rule is the peak-scaled `max|Δ| ≤ 10·eps·P[0,0]`, and the JSON also records
`parity_max_rel_elementwise` and the count of entries over `rtol=1e-10` in both regimes.

### Scaling

The brute force evaluates one cosine per (offset, visibility) pair over four quadrants:
`4·Ny·Nx·K`, linear in both the extent and the visibility count. The NUFFT spreads each
visibility onto a fine grid with a separable kernel of width `nspread ≈ 14` per axis
(at `eps=1e-12`) and takes one FFT of the doubled grid: `K·nspread² + M log M`, `M = 4·Ny·Nx`.
Predicted ratios over the NumPy builder: **12× (sma)**, **398× (alma)**, **1590×
(alma_high)**. Measured (wall clock): **7.8× (sma)**, **289× (alma)** — the flop model is
right to within ~1.4×, which is as much as a flop count can claim across a numba loop, a
chunked NumPy broadcast and a spread-and-FFT. sma's ratio is small only because `K = 190` is
too few visibilities for the `K·nspread²` term to dominate the fixed `M log M` FFT; the gap
widens with instrument size because only one of the two costs is linear in `K`. On
CPU-seconds the alma win is `111×` rather than `289×` — the model counts flops, not cores.

### Verdict — a library follow-up is justified (architect's call)

**Yes — file the PyAutoArray follow-up.** `nufft_precision_operator_from` should build via
the transformer's own type-1 NUFFT, with the brute-force builders kept as the reference the
new path is pinned against. The case:

- It is the *same array*, not an approximation of a different one: pinned
  elementwise-with-an-absolute-floor at sma, peak-scaled at alma, and the log evidence it
  produces at alma is identical to every digit (`rel diff 0.0`).
- The saving is the largest single one in this epic — 35 minutes down to 7 seconds at alma,
  more than every likelihood lever in sections 1–5 put together, because it is a cost every
  arm pays and no arm was measuring.
- It removes a scaling wall rather than a constant: the brute force is linear in `K`, the
  NUFFT is `K·nspread²` plus a fixed FFT, so the gap widens with instrument size (predicted
  `1590×` at alma_high, where the brute force is simply not runnable here).
- It costs the library nothing new. `nufftax` is already a hard dependency of
  `TransformerNUFFT`, the mapping uses that transformer's own `_x`/`_y` convention, and the
  same call runs on GPU — so it also gives the GPU path a preload it does not currently have.

Caveats the follow-up must carry:

- **`eps` is a real dial.** `eps=1e-12` is the recommendation: it saturates fp64
  (`eps=1e-14` moves `max|Δ|` only from `1.7e-17` to `1.4e-17` at sma) and is the only value
  that also holds the array pin at sma. Every value tested holds the `rtol=1e-6`
  log-evidence pin with at least five orders of margin, so the dial is not load-bearing for
  correctness — but a preload is reused by every likelihood call in a fit, and error budget
  spent here is spent for the whole run.
- **Chunking is required, not optional.** The spreader's gather buffer is `K·nspread²`
  complex128; alma_high's 5 M visibilities need ~15 GB in one shot and were OOM-killed here
  before chunking was added. The library already has the right knob — the transformer's
  `chunk_size` (PyAutoArray#330) — and the chunked and one-shot arrays are pinned equal.
- **The mapping is convention-bound.** Six of the eight index permutations are wrong by 21 %
  of the peak and all eight are structurally plausible, so a library implementation must
  carry the permutation pin, not just a smoke test.
- **The wall-clock ratio is not a single-core ratio.** The NUFFT builder ran at 2.6× cores
  at alma; on CPU-seconds the win is `111×`, not `289×`. Both are in the JSON. The machine
  also drifted between the two measurements (`control_dgemm_s` 0.2550 for the brute-force
  builds, 0.3162 for the assembling run that timed the NUFFT), which makes the reported
  ratio conservative rather than flattering.

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

# Section 7 — the preload bake-off. Same pinning for all three.
PIN='OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"

# sma — all four builders, 3 repeats (seconds).
python scripts/interferometer/likelihood_breakdown/preload_numba.py \
  --instrument sma --builders all --reps 3

# alma — the brute-force trio first (~30 min each, isolated + 90-min capped, each
# result cached beside the dataset the moment it finishes), then the assembling run,
# which hits those caches and times the NUFFT builder and the eps sweep fresh.
python scripts/interferometer/likelihood_breakdown/preload_numba.py \
  --instrument alma --builders numba,numpy,jax_cpu --eps 1e-12 --reps 1
python scripts/interferometer/likelihood_breakdown/preload_numba.py \
  --instrument alma --builders all --reps 3

# alma_high — NUFFT only; the brute-force builders are hard-refused there.
python scripts/interferometer/likelihood_breakdown/preload_numba.py \
  --instrument alma_high --builders all --reps 3
```

`--builders all` at alma works in one shot too; the trio is split out above only so a
90-minute build can be driven in the background and re-assembled from its cache.
