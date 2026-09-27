# Interferometer mesh likelihood breakdown — Delaunay-1500 and rectangular on the CPU, numba direct_conv vs NumPy FFT through the library dispatch, crossover + ranked levers (2026-09)

autolens_profiling issue [#332](https://github.com/PyAutoLabs/autolens_profiling/issues/332),
branch `feature/interferometer-mesh-numba-p2`, epic `interferometer-likelihood-campaign`
(3/3, phase 2). This is the CPU sibling of the A100 note
[`interferometer_mesh_a100_breakdown_2026_09.md`](./interferometer_mesh_a100_breakdown_2026_09.md)
(2/3, #320). Phase 1 (#326 / #328) built the library-dispatch harness and the r3.5 rows. This
phase re-runs those rows on the cached library, measures the numba / FFT crossover in situ
across mask radius, and ranks the CPU levers. The crossover verdict for the packaged gate is
also section 8 of [`numba_interferometer_verdict.md`](./numba_interferometer_verdict.md).

**Headline.** One single-threaded alma (1M visibilities) likelihood costs **1.26 s
(Delaunay-1500) / 1.83 s (rectangular 39×39)** on the numba route. The NumPy FFT route costs
2.59 / 2.94 s on the same inputs. The curvature matrix F is 70–74 % of the numba call at alma
and 95 % at alma_high. The fnnls solve is most of the rest.

- **Caching fix.** PyAutoArray#582 (`e281abf3`) caches F and D. That cut every NumPy full call
  to **0.51–0.88×** of its phase-1 value. Phase-1 rows computed F twice and D four times per
  evaluation.
- **Crossover.** numba stops winning at **nnz/col ≈ 66 (Delaunay) / ≈ 72 (rectangular)**.
  Each is bracketed by two measured alma radii.
- **Gate.** The packaged gate of 60 already routes every measured cell to the faster arm. One
  gate of about 70 would be slightly better for both meshes (see the gate verdict below).

Cells: `scripts/interferometer/likelihood_breakdown/{delaunay,pixelization}_numba.py`, harness
`scripts/misc/likelihood_breakdown/interferometer_pixelized_numpy.py`. SLURM:
`hpc/batch_cpu/submit_breakdown_interferometer_{delaunay,pixelization}_numba_ral_crossover_fp64`
(arrays 358985 / 358986) and `hpc/batch_cpu/submit_breakdown_interferometer_numba_levers_ral_alma_fp64`
(job 359000). Every number below is read from the committed JSONs:

| Leg | JSON (under `results/breakdown/interferometer/`) |
|---|---|
| r3.5 re-runs (sma / alma / alma_high) | `{sma/,,alma_high/}{delaunay,pixelization}_numba_hpc_ral_cpu_fp64.json` |
| alma radius sweep | `{delaunay,pixelization}_numba_hpc_ral_cpu_fp64_r{2.0,4.25,5.0,6.0}.json` |
| levers (alma r3.5, numba arm) | `{delaunay,pixelization}_numba_hpc_ral_cpu_fp64_levers.json` |

## Scope — read this before quoting a number

- **Model and meshes.** Same as the A100 note. Isothermal + ExternalShear near the truth, no
  lens light.
  - Delaunay is Hilbert-1500 with `AdaptSplit(0.1, 10, 0.1)`; all 1500 pixels are solved.
  - Rectangular is `RectangularBilinearAdaptImage` 39×39 with `Constant(1.0)`. It is
    edge-zeroed, so 1369 of 1521 pixels are solved.
  - The adapt image is the May-18 `lensed_source.fits` cache at every radius. Its md5 is
    recorded in every row's `adapt_image`: `a978…` sma, `6bf5…` alma, `c3d2…` alma_high.
    `regenerated` is false everywhere.
  - At r > 3.5 the adapt image is zero outside the r3.5 disc. It only places the source-plane
    mesh, so every radius sees the same source mesh construction.
- **Path.** `FitInterferometer(..., xp=np)` through `inversion_interferometer_from`. Two arms
  run in one process on identical inputs, selected only by
  `Settings.interferometer_numba_nnz_per_source_max`:
  - numba, gate forced open: `InversionInterferometerSparseNumba`, the `direct_conv` kernel;
  - NumPy FFT, gate 0: `InversionInterferometerSparse(xp=np)`.

  Each JSON asserts the class it built. `configuration.inversion_path` reads
  `InversionInterferometerSparseNumba` on every row.
- **Discipline (#235).** One thread (BLAS family + `NUMBA_NUM_THREADS=1`), NNLS memo **off**,
  and an iid stream of 20 instances (instance 0 is a warm-up). The arm order alternates each
  instance (ABBA). A dgemm control runs at the head and tail. The sma rows are log-evidence
  pinned.
- **Mask sweep.** alma pixel scale 0.05″, radius 2.0 / 3.5 / 4.25 / 5.0 / 6.0″. The masked
  pixel count M and the operator extent (bounding box, `mask_2d.py:746-776`) grow with the
  radius, and so does nnz/col.
  - nnz/col is measured in situ from `mapper.pix_sizes_for_sub_slim_index`.
  - On the rectangular mesh it is 4·M / 1521 to within 1 % (alma r3.5: 40.4). The #226 rect
    fiducials (19.7 / 78.5 / 314) were a different mesh size (S = 784) and are not
    comparable.
- **JAX-CPU arm.** This is the pointer row `arms.jax_cpu_fft` from the phase-1 RAL JAX leg
  (same geometry, `same_geometry: true`). It was not re-run: JAX's CSE already merged the
  duplicate F / D, so #582 does not change it.

## Provenance and gate

- **Host.** RAL `euclid-ral-gpu-{1,2}` (AMD EPYC 7702), CPUs only, with no `--gres`.
  - The 14 crossover tasks ran concurrently on gpu-1 (56 of its 124 CPUs, no other jobs). The
    lever job ran alone on gpu-2.
  - All jobs `COMPLETED`, with 0 Tracebacks in every `.err`. Walls were 1.6–22 min.
- **Libraries.** The RAL mirror was refreshed with `HPCPullPyAuto` on 2026-09-27. Its HEADs
  equal the local mains, and every row's `source_revisions` reads:
  - PyAutoNerves `bf104102`, PyAutoFit `c156a9d8`, **PyAutoArray `e281abf3`** (#582),
    PyAutoGalaxy `ba8a08fa`, PyAutoLens `dcbd4b71`;
  - branch `92a2061`.

  numba 0.65.1, nufftax 0.6.1 (at the 0.6.1 floor).
- **Contention.** The dgemm control on gpu-1 read 0.150–0.166 s at the head and 0.141–0.154 s
  at the tail. For comparison it read 0.142 s on the idle gpu-2 and 0.141–0.147 s in phase 1.
  - So BLAS ran up to ~13 % slow while all 14 tasks overlapped. The host load average was ≤ 12.8
    on 124 cores.
  - The numba and FFT F kernels did not move against phase 1. alma Delaunay F-alone was 924
    vs 913 ms numba and 2174 vs 2208 ms FFT; alma_high was 13049 vs 13139 ms numba.
  - The arms alternate ABBA inside each task, so the ratios are insensitive to the overlap.
    Judged **not contended**; no leg was re-run.
- **Witness checks, all 14 rows:**
  - `evaluations_per_figure_of_merit` = {1, 1} on both arms;
  - numba vs FFT Δlog-evidence ≤ 7.5e-9 nat (bar 0.5);
  - step sum / full call 0.993–1.022 (band 0.9–1.1);
  - sma pins held with no drift: −3162.627234657415 Delaunay, −3168.595092417778 rect.

## Caching fix — before / after (PyAutoArray#582, `previous_row`)

Each r3.5 row carries the phase-1 row it replaced (`previous_row`). Phase 1 counted
`curvature_matrix_diag` **2** and `data_vector` **4** computations per `figure_of_merit`; now
both are 1.

| Row | numba full call (phase 1 → now) | FFT full call (phase 1 → now) |
|---|---|---|
| sma Delaunay | 358 → 278 ms (**0.78×**) | 1442 → 873 ms (**0.61×**) |
| sma rect | 933 → 823 ms (0.88×) | 1973 → 1389 ms (0.70×) |
| alma Delaunay | 2294 → 1262 ms (**0.55×**) | 4891 → 2595 ms (**0.53×**) |
| alma rect | 3169 → 1830 ms (0.58×) | 5148 → 2944 ms (0.57×) |
| alma_high Delaunay | 27324 → 13827 ms (**0.51×**) | 17932 → 9390 ms (0.52×) |
| alma_high rect | 37448 → 19044 ms (0.51×) | 18255 → 9587 ms (0.53×) |

The drop is largest where F is most of the call, because F was being computed twice. At sma
rect the fnnls solve dominates, so the drop is smallest there.

## RAL CPU baseline — cached library, r3.5, single thread

| Instrument | Mesh | nnz/col | M (masked) | extent | numba full | NumPy FFT full | numba / FFT | JAX-CPU FFT (steps, phase 1) | numba F share |
|---|---|---|---|---|---|---|---|---|---|
| sma | Delaunay | 6.2 | 3852 | 70² | **278 ms** | 873 ms | 0.32 | 1121 ms | 30 % |
| sma | rect | 10.1 | 3852 | 70² | **823 ms** | 1389 ms | 0.59 | 980 ms | 15 % (solve 76 %) |
| alma | Delaunay | 29.1 | 15380 | 140² | **1262 ms** | 2595 ms | 0.49 | 2876 ms | 74 % |
| alma | rect | 40.4 | 15380 | 140² | **1830 ms** | 2944 ms | 0.62 | 2425 ms | 70 % |
| alma_high | Delaunay | 117.6 | 61572 | 280² | 13827 ms | **9390 ms** | 1.47 | 8452 ms | 95 % |
| alma_high | rect | 161.9 | 61572 | 280² | 19044 ms | 9587 ms | 1.99 | **7908 ms** | 95 % |

- At alma_high the JAX-CPU route is the fastest CPU path, as in phase 1. It is a phase-1 row,
  but one with no redundant F.
- The one-off W~ preload with dataset load costs 1.9 s (sma), 10–15 s (alma, any radius) and
  18 s (alma_high) on a cache hit / miss. Peak RSS is 0.8 / 2.4 / 8.9 GB.

## The crossover (alma mask-radius sweep)

| r (″) | M | extent | Delaunay nnz/col | Delaunay numba / FFT | rect nnz/col | rect numba / FFT |
|---|---|---|---|---|---|---|
| 2.0 | 5024 | 80² | 7.3 | 0.412 (448 / 1085 ms) | 13.2 | 0.562 (783 / 1393 ms) |
| 3.5 | 15380 | 140² | 29.1 | 0.486 (1262 / 2595 ms) | 40.4 | 0.622 (1830 / 2944 ms) |
| 4.25 | 22704 | 170² | 42.6 | 0.564 (2310 / 4097 ms) | 59.7 | **0.749** (3248 / 4337 ms) |
| 5.0 | 31428 | 200² | 55.6 | **0.886** (3789 / 4277 ms) | 82.7 | **1.225** (5699 / 4653 ms) |
| 6.0 | 45244 | 240² | 81.4 | **1.169** (7384 / 6317 ms) | 119.0 | 1.639 (10738 / 6550 ms) |

**Method.** The crossover is the linear interpolation of ln(numba / FFT full call) against
nnz/col between the two measured points that straddle 1.

- Plain linear interpolation of the ratio gives the same answer to ±1.
- Interpolating the F-alone sub-row ratio instead gives 65.7 / 72.4.
- So the rest of the call (build, solve, log-dets) is shared between the arms and does not
  move the crossover.

| Mesh | Bracket (measured) | Crossover nnz/col |
|---|---|---|
| Delaunay | r5.0 (55.6, 0.886) → r6.0 (81.4, 1.169) | **≈ 66–67** |
| Rectangular | r4.25 (59.7, 0.749) → r5.0 (82.7, 1.225) | **≈ 72–73** |

- **The two meshes cross at different nnz/col.** The rectangular kernel has a higher cost per
  non-zero (bilinear, 4 weights per pixel). But its fnnls solve is shared and larger, which
  dilutes the F difference in the full call.
- **The crossover moves up with the extent at fixed nnz.** The alma_high rows (280² extent) are
  above crossover, as they should be. Proportional extrapolation from them gives about 80 on
  both meshes, against 66 / 72 at alma. That is an extrapolation, not a measurement: the
  FFT's log M factor grows with the extent while direct_conv's per-MAC constant falls (the
  #226 constants table). So one nnz/col gate is geometry-dependent, and alma is the reference
  geometry here.

### Gate verdict (`general.yaml` `interferometer_numba_nnz_per_source_max`)

The gate is **one number for both meshes** (`config/general.yaml:21` = 60.0, read at
`inversion/inversion/factory.py:284`). The data supports a single default of **≈ 70**. The cost
of each choice on each mesh comes from the interpolated alma ratio at the gate:

| Gate | Delaunay (crossover ≈ 66) | Rectangular (crossover ≈ 72) |
|---|---|---|
| 60 (packaged) | nnz 60–66 goes to FFT, up to **1.08×** slower than numba (ratio 0.93 at 60) | nnz 60–72 goes to FFT, up to **1.33×** slower (ratio 0.75 at 60) |
| 66 | optimal | nnz 66–72 on FFT, up to 1.17× slower |
| **70** | nnz 66–70 on numba, up to **1.03×** slower | nnz 70–72 on FFT, up to **1.07×** slower |
| 72 | up to 1.06× slower | optimal |
| 77 (#226 rect) | up to 1.12× slower | up to 1.09× slower (numba past 72) |

- **Every measured cell routes to its faster arm at both 60 and 70.** The nearest measured
  cells are rect 59.7 (numba, 0.749) and Delaunay 55.6 (numba) against Delaunay 81.4 / rect
  82.7 (FFT). The 60 → 70 move rests on interpolation inside those brackets, not on a
  misrouted measured cell.
- Only models whose nnz/col falls in 60–72 are affected. At alma pixel scale with 1500 source
  pixels that is about r4.25–r5.0 for rect and r5.0–r5.4 for Delaunay.
- A retune prompt is drafted: gate 70, with witness cells measured inside the band first. It
  is a small, machine-dependent gain and is not urgent. The `general.yaml` comment's "~60 /
  ~77" should be re-quoted as "~66 / ~72 in situ (#332)" either way.

## Ranked levers

Production CPU modelling runs one single-threaded likelihood per pool worker, so gains are
quoted per worker, single-threaded, unless the row says otherwise.

| Rank | Lever | Step attacked | Structural bound | Measured gain | Follow-up |
|---|---|---|---|---|---|
| 1 | fnnls warm-start memo: a guard that stops it hurting on scattered streams | solve (fnnls) | solve share 17 % Delaunay / 22 % rect at alma; 54 % / 76 % at sma | **measured**, alma r3.5: local walk solve 0.68× Delaunay / **0.13× rect**; iid solve **2.17× slower** Delaunay / 1.08× rect | research prompt (PyAutoArray) |
| 2 | Gate retune 60 → ~70 | routing | cells with nnz/col 60–72 | **interpolated**: up to 1.33× (rect) / 1.08× (Delaunay) on those cells; costs ≤ 1.03–1.07× at the band edges | feature prompt (PyAutoArray `general.yaml`) |
| 3 | `prange` kernel load balance (single-process / interactive runs only) | F (direct_conv) | F share 70–74 % alma | **measured**, gpu-2 idle: rect F 1.96× / 2.53× at 2 / 4 threads; Delaunay **1.02×** / 1.92×; implied full call 1.54× / 1.75× at 4 threads | research prompt (PyAutoArray) |
| 4 | Cholesky reuse for the edge-zeroed rect log-det | log det (F + H) | rect 50 ms = 2.7 % of the alma call | **measured miss**: rect reused 47.7 ms = dense 47.7 ms (fast path not taken); Delaunay reuse works (15 vs 38 ms, already shipped) | small feature prompt (PyAutoArray) |
| — | `kernel_index_arrays` marshalling | F marshalling | 5.8–6.1 ms = 0.35–0.47 % of the alma call | none worth taking; the instance-independent part is 0.13–0.15 ms | not filed |
| — | MGE + mesh numba route | F | unmeasured | static read only; not routed today | not filed (needs an MGE + mesh CPU cell first) |

### 1. fnnls warm-start memo — a guard that works on scattered streams

- **Step attacked.** The positive-only reconstruction, `reconstruction_positive_only_from`
  (PyAutoArray `inversion/inversion/inversion_util.py:291`). The memo seeds each fnnls solve
  from the previous evaluation's final passive set (`nnls_memo.py`). It is **on by default**
  (`general.yaml:12`, #498); this campaign's headline rows run it off.
- **Measured** (`levers.memo`, alma r3.5, 16 instances per stream, memo cleared per pass,
  first solve excluded):

  | Stream | Delaunay solve off → on | rect solve off → on |
  |---|---|---|
  | local walk (unit step 0.002) | 163 → 111 ms (**0.68×**) | 324 → 41 ms (**0.13×**) |
  | iid (central 20 % of every prior) | 164 → 356 ms (**2.17× slower**) | 329 → 356 ms (1.08× slower) |

  The figure of merit is identical on and off (Δ 0.0 nat). On the local walk that is −52 ms
  (−4 %) of the Delaunay call and −283 ms (−15 %) of the rect call. On the iid stream the
  Delaunay call is **+192 ms (+15 %)** with the memo on.
- **Why the iid stream regresses.** The fallback guard (`inversion_util.py:598-630`) judges a
  memo seed only after the seeded solve has already run.
  - It drops the entry when the seed's error fraction exceeds `nnls_warm_start_error_tolerance`
    × the dense-sign reference. The next solve then restarts dense and re-seeds.
  - On a scattered stream every other solve pays for a bad seed, so the guard bounds the damage
    to about half the solves but does not prevent it.
- **Lever.** A seed check that runs *before* the seeded solve. Or a guard that backs off the
  memo for a key after repeated fallbacks. Either would keep the local-walk gain without the
  iid regression.
- **To measure first.** Which stream a real sampler presents: Nautilus's early live-point
  phase is iid-like, its late phase walk-like. The research prompt measures a real sampler's
  evaluation sequence before any library change.

### 2. Gate retune 60 → ~70

See the gate verdict above. Bound: only cells with nnz/col 60–72 change route, gaining up to
1.33× (rect) / 1.08× (Delaunay) on those cells. The retune prompt's witness is a measured cell
inside the band per mesh: rect at alma r4.6 (predicted nnz ≈ 70) and Delaunay at alma r5.4
(predicted nnz ≈ 66; nnz/col ∝ M at fixed pixel scale). The routing at the new gate must
match the faster arm at every measured cell.

### 3. `prange` kernel load balance (interactive / single-process runs)

- **Step attacked.** F by `direct_conv_parallel_kernel`
  (`interferometer_numba/inversion_interferometer_numba_util.py:109-160`). It is selected by
  `general.yaml` `numba.parallel` (`:26`, default false) through
  `interferometer_numba/sparse.py:202-203`. The `prange` runs over source columns
  (`util.py:157`).
- **Measured** (`levers.threads`, gpu-2 idle, load 0.2–1.1, pool of 4, BLAS at 1):

  | Mesh | serial kernel | 1 thread | 2 threads | 4 threads | implied full call at 4 threads |
  |---|---|---|---|---|---|
  | Delaunay | 895 ms | 914 ms (0.98×) | 874 ms (**1.02×**) | 466 ms (1.92×) | 1228 → 799 ms (1.54×) |
  | rect | 1248 ms | 1182 ms (1.06×) | 637 ms (1.96×) | 493 ms (2.53×) | 1756 → 1002 ms (1.75×) |

  The parallel F is bit-identical to the serial one (max |Δ| = 0).
- **Why Delaunay stalls at 2 threads.** The Hilbert mesh orders source columns along the adapt
  image, and a column's cost is its nnz. A static split of `prange(pix_pixels)` gives one
  thread the dense half. Rect columns are near-uniform, so they split evenly.
- **Lever.** Order or chunk the columns by nnz (from `cscptr`) so threads get equal work.
- **Scope caveat.** A Nautilus pool already owns every core, so this is a ceiling for
  interactive single evaluations and single-process fits. It is not a per-worker production
  gain (the #226 pool caveat). #226's synthetic bake-off got 4.63× at 8 threads on uniform
  columns; the in-situ AdaptSplit mesh does not.

### 4. Cholesky reuse for the edge-zeroed rectangular log det

- **Step attacked.** `log_det_curvature_reg_matrix_term` (`inversion/inversion/abstract.py:998-1078`).
  It reads the log det off the fnnls Cholesky factor, but only when
  `_nnls_factor_ids_cover` (`:1081-1096`) sees the solve cover the whole reduced system.
- **Measured** (`levers.logdet`):
  - Delaunay: reused 15.2 ms against 38.3 ms for a fresh dense Cholesky. It is the same value
    (Δ 0), with 1306 of 1500 columns passive. The shipped reuse works.
  - rect: reused **47.7 ms = dense 47.7 ms**. The fast path is not taken. The edge-zeroed
    solve covers 1369 of 1521 ids, so the cover check falls through to the dense route.
- **Bound.** At most the 47–50 ms rect row, 2.7 % of the alma call. Only 947 of the solved
  columns are passive, so the Schur complement is larger than on Delaunay and the realised
  gain is probably about half that. It is small and cheap to try, so it is drafted as a small
  PyAutoArray prompt.

### Not levers (evaluated, not filed)

- **`kernel_index_arrays` marshalling.** 5.8 / 6.1 ms (Delaunay / rect) at alma, 0.47 / 0.35 %
  of the call. The instance-independent part (the extent-index gather + iy / ix) is
  0.13–0.15 ms. The mask already caches `extent_index_for_masked_pixel` (3.5 ms cold). The rest
  is the per-instance CSR / CSC build from the mapper's triplets, which changes every
  evaluation. Preloading would save < 0.02 % of the call.
- **MGE + mesh numba route (static read).** The factory gate `_use_interferometer_numba`
  (`inversion/inversion/factory.py:295-300`) requires exactly one `Mapper` and no other linear
  object. So any MGE + mesh model takes the FFT route on CPU, whatever its nnz/col.
  - A numba route would need the func-list × mapper off-diagonal blocks as well as the
    `direct_conv` mapper block.
  - The mapper × mapper block is the F row measured here, so the gain on such a model is
    bounded by the mapper block's numba / FFT ratio (0.32–0.62 at sma / alma).
  - Not filed: no MGE + mesh CPU cell exists yet. The campaign's phase-4 decision matrix is
    where one lands.
- **F on the FFT route.** This is F-bound (81–93 % of the FFT call at alma+). The
  pruned-transform half of the A100 note's lever 2 (`interferometer_w_tilde_fft_size_levers`)
  applies to the NumPy `rfft2` blocks as well. It is already drafted, so not re-filed.

## Follow-ups

Drafted as PyAutoMind prompts (`Status: draft`, epic `interferometer-likelihood-campaign`),
filed by the main session. None is implemented here.

| Lever | Draft | Target |
|---|---|---|
| 1 — memo guard on scattered streams | `interferometer_nnls_memo_scattered_stream_guard` | autolens_profiling → PyAutoArray |
| 2 — gate retune 60 → 70 | `interferometer_numba_gate_retune_70` | PyAutoArray |
| 3 — prange load balance | `interferometer_direct_conv_prange_load_balance` | PyAutoArray |
| 4 — edge-zeroed Cholesky reuse | `edge_zeroed_log_det_cholesky_reuse` | PyAutoArray |

Next in the campaign: phase 3 (A100 mask-radius sweep, r2.0 / r5.0), then phase 4 (the decision
matrix).
