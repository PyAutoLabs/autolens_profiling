# Interferometer likelihood decision matrix — which likelihood on CPU vs A100, by N_vis × mask × source (2026-09)

autolens_profiling issue [#356](https://github.com/PyAutoLabs/autolens_profiling/issues/356),
branch `feature/interferometer-decision-matrix`, epic `interferometer-likelihood-campaign`
(3/3, phase 4 — the epic deliverable). This note answers one question: **for an
interferometer fit of a given size, which likelihood path do I run, on which device, and
what does one fit cost?** It assembles the committed rows of the three campaign ledgers
([MGE](./interferometer_mge_breakdown_2026_09.md), [mesh A100](./interferometer_mesh_a100_breakdown_2026_09.md),
[mesh CPU](./interferometer_mesh_cpu_breakdown_2026_09.md), plus the numba
[verdict](./numba_interferometer_verdict.md)) and adds the cells they left empty:

- **8 CPU radius-gap cells:** sma and alma_high × mask r2.0 / r5.0 × Delaunay-1500 and
  rectangular 39², numba `direct_conv` and NumPy FFT arms, RAL CPU, single thread.
- **An sdp81 row:** the real ALMA SDP.81 uv coverage (108,384 visibilities) at r3.5, for
  Delaunay-1500, rect 39² (CPU + A100) and MGE-20 (A100 dense and W~ sparse).

**Status (2026-10-04): all 14 new cells are measured** (8 CPU radius gaps + 6 sdp81). 13 shipped on 2026-09-30 in #358 and pass their witness gates.

The last cell is CPU rect 39² at alma_high r5.0. It was filled on 2026-10-04 (issue #369) from RAL 375978_3 (COMPLETED, 01:11:36) into `alma_high/pixelization_numba_hpc_ral_cpu_fp64_r5.0.json`. It passes every gate except CPU vs A100:
- **CPU vs A100: 1.59e-3 nats.** CPU log-evidence −60244101.501766354 (fnnls); A100 `full_pipeline.figure_of_merit` −60244101.503356226 (PDIP, on PyAutoArray `9428eca2`). That misses the 1e-3 bar.
- **Accepted by the human on 2026-10-04.** The reasons:
  - It is the same failure mode as the alma_high r3.5 rect cell, which missed at 1.5e-3.
  - The A100 row is PDIP on revisions from before PyAutoArray#595, while the CPU row is fnnls. This is a solver/revision mismatch on the edge-zeroed subset, not a precision loss.
  - It is a relative 2.6e-11, and 310× inside the 0.5-nat evidence bar.
- The A100 row's own `solver_ab` puts its certified active-set solve within 0.0 nats of its PDIP solve.
- The gap is recorded as a miss, not rounded away. An A100 re-run on post-#595 revisions would close the question; it was not run.

## Rules (draft)

A draft rule set for the architect to review; each rule cites the rows it rests on.

1. **Mask extent, not N_vis, sets the per-likelihood cost on the sparse (W~) path, on both
   devices.** The visibilities are folded into the W~ preload once; per call the work is F on the
   masked-pixel extent plus the solve. alma r5.0 (31,428 masked px, 1M vis) costs more than
   alma_high r2.0 (20,108 px, 5M vis): CPU Delaunay 3.79 s vs 2.34 s, A100 70.2 vs 54.1 ms.
   The sdp81 row is the direct test: the same 15,380 masked pixels as alma r3.5 with 9× fewer visibilities costs the same on the A100 (Delaunay 53.7 vs 49.5 ms, rect 45.2 vs 44.5 ms) and on CPU (rect 1.85 vs 1.83 s; Delaunay 1.50 vs 1.26 s on a host at load ≈ 60 vs ≈ 10). Size a fit by its masked-pixel count.
2. **Meshes on the A100: always the W~ sparse path, never the dense mapping path.** The dense
   arm's `T` is N_vis × S complex128 and OOMs from 1e6 visibilities at every radius; where it
   runs it is slower — ~2× at sma, 17–21× at sdp81 (918 / 941 ms vs 53.7 / 45.2 ms).
3. **Meshes on CPU: trust the library gate.** numba `direct_conv` beats the NumPy FFT route below
   nnz/col ≈ 66 (Delaunay) / ≈ 72 (rect) and loses above; the packaged gate
   (`interferometer_numba_nnz_per_source_max = 60`) sent every measured cell in this matrix to the
   faster arm (no "(slower arm)" flags in the tables). At alma_high r3.5 the phase-1 JAX-CPU route is faster still
   (8.3 / 7.6 s vs FFT 9.4 / 9.6 s).
4. **The A100 wins every mesh cell, and the margin grows with the mask.** Best-CPU ÷ A100 is
   8–33× at sma, 28–41× at sdp81, 11–69× at alma and 43–114× at alma_high (single thread vs one
   JIT; rechecked 2026-10-04 with the last cell: rect alma_high r5.0 is 89×, inside the range). The CPU is
   still viable where the per-call cost is about a second or less (sma at any radius, alma
   r2.0 on both meshes, alma r3.5 Delaunay at 1.26 s): hours per fit on one core, minutes over a node's cores (Indicative time per fit).
   Above ~20,000 masked pixels a CPU fit is days per core; use the A100.
5. **MGE: always `apply_sparse_operator()` (the W~ route), on both devices.** The dense MGE chain
   is O(N_vis) per call (a NUFFT of the mapping matrix every evaluation) and OOMs the A100 from
   1e6 visibilities; the W~ route is ms-scale everywhere it ran (A100 2.7–20 ms, laptop CPU 35 ms at
   alma vs 13 s dense). At sdp81 (1e5 visibilities) the dense library path does fit the A100 (28.3 ms, after #577), and W~ is still 7.5× faster (3.79 ms).

## Scope — read this before quoting a number

- **Model.** Isothermal + ExternalShear near the simulator truth, no lens light, as in every
  campaign ledger. Sources: MGE-20 (20 linear Gaussians, PDIP NNLS), Delaunay-1500 (Hilbert
  image mesh on the adapt image, `AdaptSplit(0.1, 10, 0.1)`), rectangular 39²
  (`RectangularBilinearAdaptImage`, `Constant(1.0)`).
- **Instruments.** The presets in `instruments/interferometer.py`: sma (190 vis, 256² @ 0.1″),
  alma (1M, 800² @ 0.05″), alma_high (5M, 800² @ 0.025″), jvla (25M, 800² @ 0.01″). The pixel
  scale differs per instrument, so the same mask radius is a different number of masked
  pixels: masked pixels, not arcsec, set the curvature cost.
- **sdp81 is "real sdp81 uv coverage, simulated source".** The new `sdp81` preset loads the
  real SDP.81 `uv_wavelengths` (copied into `instruments/uv_coverage/sdp81_uv_wavelengths.fits`
  from `autolens_workspace/dataset/interferometer/sdp81/`) and simulates the standard lens +
  source on the alma grid, so between the sdp81 and alma rows **only N_vis and the uv
  distribution change** (same 800² @ 0.05″ grid, same r3.5 mask, 15,380 masked pixels). The real
  SDP.81 visibilities are not fitted (no adapt image or mask exists for them). The simulated
  dataset is tracked (`dataset/interferometer/sdp81/`, like alma) so CPU and A100 fit
  byte-identical data.
- **CPU = one thread.** Every RAL CPU cell runs one process with `NUMBA_NUM_THREADS=1` and the
  BLAS family at 1, NNLS memo off, iid instance stream (the #235 discipline), on the `gpu`
  partition without `--gres`. A production CPU fit runs one likelihood per core, so the
  per-core number is the one that scales; the thread-scaling story is in the
  [verdict](./numba_interferometer_verdict.md) §3.
- **A100 = one JIT call.** "A100 W~ sparse" is the library `FitInterferometer` on the sparse
  (W~) path under one `jax.jit` (`full_pipeline_single_jit`, PDIP), no vmap. Batched per-call
  numbers are in the A100 ledger's vmap columns and are lower.
- **Peak VRAM is not recorded** in any row: the A100 JSONs carry an `nvidia-smi` snapshot, not
  a peak. Peak RSS below is host memory.

## Main matrix — Delaunay-1500, per-likelihood ms

Bold = the faster CPU arm. "gate-60 route" is where the packaged gate
(`interferometer_numba_nnz_per_source_max = 60`) sends the cell; "(slower arm)" flags a
cell the gate routes to the slower arm. A100 speed-up = best CPU arm ÷ A100 W~.

| N_vis | instrument | mask r | masked px | nnz/col | CPU numba | CPU NumPy FFT | gate-60 route | CPU JAX (r3.5, ph.1) | A100 W~ sparse | A100 dense arm (step sum) | A100 speed-up vs best CPU |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 190 | sma | 2.0 | 1,264 | 1.8 | **244.9** | 439.3 | numba | — | 31.4 | 68.6 | 8× |
| 190 | sma | 3.5 | 3,852 | 6.2 | **277.7** | 872.8 | numba | 1,211 | 32.6 | 67.1 | 9× |
| 190 | sma | 5.0 | 7,860 | 8.8 | **450.6** | 1,523 | numba | — | 35.8 | 65.8 | 13× |
| 1.08e5 | sdp81 | 3.5 | 15,380 | 29.1 | **1,503** | 2,981 | numba | — | 53.7 | 918.2 | 28× |
| 1e6 | alma | 2.0 | 5,024 | 7.3 | **447.5** | 1,085 | numba | — | 39.1 | OOM (25.6 GiB) | 11× |
| 1e6 | alma | 3.5 | 15,380 | 29.1 | **1,262** | 2,595 | numba | 2,696 | 49.5 | OOM (25.7 GiB) | 25× |
| 1e6 | alma | 5.0 | 31,428 | 55.6 | **3,789** | 4,277 | numba | — | 70.2 | OOM (25.9 GiB) | 54× |
| 5e6 | alma_high | 2.0 | 20,108 | 29.5 | **2,341** | 4,241 | numba | — | 54.1 | skipped (T > VRAM) | 43× |
| 5e6 | alma_high | 3.5 | 61,572 | 117.6 | 13,827 | **9,390** | FFT | 8,328 | 101.5 | skipped (T > VRAM) | 93× |
| 5e6 | alma_high | 5.0 | 125,676 | 225.4 | 61,315 | **21,821** | FFT | — | 190.9 | skipped (T > VRAM) | 114× |
| 2.5e7 | jvla | 3.5 | 384,852 | — | not measured | not measured | — | — | 563.3 | skipped (T > VRAM) | — |

## Main matrix — rectangular 39², per-likelihood ms

| N_vis | instrument | mask r | masked px | nnz/col | CPU numba | CPU NumPy FFT | gate-60 route | CPU JAX (r3.5, ph.1) | A100 W~ sparse | A100 dense arm (step sum) | A100 speed-up vs best CPU |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 190 | sma | 2.0 | 1,264 | 3.3 | **233.7** | 379.9 | numba | — | 22.4 | 59.4 | 10× |
| 190 | sma | 3.5 | 3,852 | 10.1 | **823.2** | 1,389 | numba | 982.9 | 28.1 | 63.0 | 29× |
| 190 | sma | 5.0 | 7,860 | 20.7 | **1,111** | 1,950 | numba | — | 33.8 | 63.2 | 33× |
| 1.08e5 | sdp81 | 3.5 | 15,380 | 40.4 | **1,851** | 2,960 | numba | — | 45.2 | 940.7 | 41× |
| 1e6 | alma | 2.0 | 5,024 | 13.2 | **783.2** | 1,393 | numba | — | 30.6 | OOM (23.7 GiB) | 26× |
| 1e6 | alma | 3.5 | 15,380 | 40.4 | **1,830** | 2,944 | numba | 2,908 | 44.5 | OOM (23.7 GiB) | 41× |
| 1e6 | alma | 5.0 | 31,428 | 82.7 | 5,699 | **4,653** | FFT | — | 67.2 | OOM (23.7 GiB) | 69× |
| 5e6 | alma_high | 2.0 | 20,108 | 52.9 | **3,739** | 4,447 | numba | — | 48.9 | skipped (T > VRAM) | 77× |
| 5e6 | alma_high | 3.5 | 61,572 | 161.9 | 19,044 | **9,587** | FFT | 7,600 | 95.9 | skipped (T > VRAM) | 100× |
| 5e6 | alma_high | 5.0 | 125,676 | 330.5 | 75,321 | **17,649** | FFT | — | 198.6 | skipped (T > VRAM) | 89× |
| 2.5e7 | jvla | 3.5 | 384,852 | — | not measured | not measured | — | — | 564.4 | skipped (T > VRAM) | — |

## Main matrix — MGE-20, per-likelihood ms (r3.5 only)

MGE-20 has no mesh, so there is no numba arm and no nnz/col; the two paths are the **dense**
mapping chain (NUFFT of the 20-column mapping matrix every call) and the **W~ sparse** route
(PyAutoArray#575: `dataset.apply_sparse_operator()` turns an MGE-only fit onto
`InversionInterferometerSparse`). CPU rows are the laptop JAX-CPU `mge.py` legs (one thread,
shared host; not RAL numba), from the [MGE ledger](./interferometer_mge_breakdown_2026_09.md).
Mask radius r2.0 / r5.0 was not measured for MGE at any instrument.

| N_vis | instrument | CPU dense (laptop) | CPU W~ sparse (laptop) | A100 dense | A100 W~ sparse | source |
|---:|---|---:|---:|---:|---:|---|
| 190 | sma | 82.1 ms (full pipeline) | **7.77 ms** | 3.39 ms (after #577 real scatter; 854.6 ms before) | **2.4 ms** (#308 W~ arm 3-8; the library route was not run on the A100 at sma) | MGE ledger, "Library W~ route" round 2 + "Lever 3" |
| 1.08e5 | sdp81 | blocked on the laptop (OOM-kill in the eager `xp=np` reference, see Blocked); the same eager reference ran untimed on the RAL A100 host | not measured | 28.3 ms (library one-shot path, full pipeline single JIT: it fits); chunked step sum 975 ms | **3.79 ms** | this note, `sdp81/mge_hpc_a100_fp64{,_sparse}.json` (RAL 375983 / 375984) |
| 1e6 | alma | 13.10 s (full pipeline; #308: 41.1 s) | **34.7 ms** | OOM (asks 61.4 GiB); chunked arm 939.6 ms | **2.69 ms** | MGE ledger round 2 |
| 5e6 | alma_high | 212.19 s (#308 step sum) | blocked: laptop OOM-kill at 10.8 GB in `apply_sparse_operator()` | OOM (asks 300.3 GiB); chunked arm 3.94 s | **19.6 ms** (round 1) | MGE ledger |
| 2.5e7 | jvla | not run | not run | OOM (asks 1.46 TiB); chunked arm 24.20 s | **16.7 ms** | MGE ledger round 2 |

The W~ operator build is one-off: 1.07 s (sma, A100), 1.6 s (sdp81, A100), 3.0–3.1 s (alma, A100), 3.7 s
(alma_high, A100), 8.2 s (jvla, A100), 18.5 s (alma, laptop CPU).


sdp81 witness: the library W~ route matches the eager NumPy dense reference to 3.8e-7 nats in
log L (`library_sparse.witness_pass = true`; bar 1e-6), F~ vs dense F max rel diff 1.5e-13.

## Setup cost and host memory

The one-off cost is paid once per dataset + mask geometry, not per likelihood: on CPU the
W~ preload (a type-1 NUFFT, cached beside the dataset per shape × radius) plus the dirty image;
on the A100 the operator build. Cache hits time only the dirty image and the wrap.

| mesh | instrument | r | CPU W~ preload + dirty image (s) | CPU peak RSS (GB) | A100 operator build (s) | A100 host peak RSS (GB) |
|---|---|---:|---:|---:|---:|---:|
| Delaunay-1500 | sma | 2.0 | 3.4 | 0.8 | 2.5 | 1.7 |
| Delaunay-1500 | sma | 3.5 | 1.9 | 0.8 | 2.8 | 1.7 |
| Delaunay-1500 | sma | 5.0 | 3.0 | 1.0 | 2.7 | 1.7 |
| Delaunay-1500 | sdp81 | 3.5 | 3.2 | 2.3 | 4.0 | 2.2 |
| Delaunay-1500 | alma | 2.0 | 12.1 | 2.2 | 5.1 | 2.5 |
| Delaunay-1500 | alma | 3.5 | 9.7 | 2.3 | 3.2 | 3.3 |
| Delaunay-1500 | alma | 5.0 | 13.0 | 2.3 | 5.1 | 2.6 |
| Delaunay-1500 | alma_high | 2.0 | 29.7 | 8.7 | 4.9 | 3.6 |
| Delaunay-1500 | alma_high | 3.5 | 17.8 | 8.7 | 4.3 | 3.6 |
| Delaunay-1500 | alma_high | 5.0 | 32.7 | 8.7 | 4.5 | 4.1 |
| Delaunay-1500 | jvla | 3.5 | — | — | 7.4 | 9.2 |
| rect 39² | sma | 2.0 | 2.4 | 0.8 | 2.7 | 1.6 |
| rect 39² | sma | 3.5 | 1.9 | 0.9 | 2.4 | 1.6 |
| rect 39² | sma | 5.0 | 2.3 | 1.0 | 2.5 | 1.6 |
| rect 39² | sdp81 | 3.5 | 3.2 | 2.3 | 4.0 | 2.4 |
| rect 39² | alma | 2.0 | 15.0 | 2.2 | 3.0 | 2.6 |
| rect 39² | alma | 3.5 | 11.7 | 2.3 | 3.0 | 3.3 |
| rect 39² | alma | 5.0 | 13.8 | 2.3 | 3.0 | 2.7 |
| rect 39² | alma_high | 2.0 | 16.5 | 8.6 | 3.6 | 3.6 |
| rect 39² | alma_high | 3.5 | 17.8 | 8.6 | 3.7 | 3.7 |
| rect 39² | alma_high | 5.0 | 16.7 | 8.6 | 3.1 | 4.0 |
| rect 39² | jvla | 3.5 | — | — | 7.5 | 10.1 |

## Verification

The witness gates for this phase, per new cell: the numba arm's
`configuration.inversion_path == "InversionInterferometerSparseNumba"`; numba vs NumPy FFT
log-evidence within 0.5 nats; the step sum within 10 % of the full call; the CPU log-evidence
within 1e-3 nats of the matching A100 row (same instrument, radius, mesh).

| mesh | instrument | r | inversion_path (numba arm) | numba vs FFT Δ nats | step sum / full (numba, FFT) | CPU vs A100 Δ log-evidence (nats) | CPU PyAutoArray / A100 PyAutoArray |
|---|---|---:|---|---:|---|---:|---|
| Delaunay-1500 | sma | 2.0 | `InversionInterferometerSparseNumba` | 2.7e-12 | 0.958, 0.999 | 2.4e-05 | `7a89e19a` / `9428eca2` |
| Delaunay-1500 | sma | 3.5 | `InversionInterferometerSparseNumba` | 9.1e-13 | 1.004, 1.008 | 2.2e-05 | `e281abf3` / `14d63360` |
| Delaunay-1500 | sma | 5.0 | `InversionInterferometerSparseNumba` | 9.1e-13 | 0.995, 0.987 | 1.3e-05 | `7a89e19a` / `9428eca2` |
| Delaunay-1500 | sdp81 | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 0.993, 1.002 | 3.4e-06 | `7a89e19a` / `7a89e19a` |
| Delaunay-1500 | alma | 2.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 0.999, 1.022 | 3.3e-06 | `e281abf3` / `9428eca2` |
| Delaunay-1500 | alma | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.003, 1.009 | 3.3e-06 | `e281abf3` / `14d63360` |
| Delaunay-1500 | alma | 5.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.002, 1.005 | 9.6e-07 | `e281abf3` / `9428eca2` |
| Delaunay-1500 | alma_high | 2.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.006, 1.007 | 1.0e-06 | `7a89e19a` / `9428eca2` |
| Delaunay-1500 | alma_high | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.000, 1.003 | 2.8e-07 | `e281abf3` / `14d63360` |
| Delaunay-1500 | alma_high | 5.0 | `InversionInterferometerSparseNumba` | 7.5e-09 | 1.003, 1.008 | 1.6e-07 | `7a89e19a` / `9428eca2` |
| rect 39² | sma | 2.0 | `InversionInterferometerSparseNumba` | 4.5e-13 | 1.016, 1.003 | 9.2e-09 | `7a89e19a` / `9428eca2` |
| rect 39² | sma | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.001, 0.993 | 9.2e-09 | `e281abf3` / `14d63360` |
| rect 39² | sma | 5.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.001, 0.992 | 9.2e-09 | `7a89e19a` / `9428eca2` |
| rect 39² | sdp81 | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 0.998, 1.000 | 7.8e-04 | `7a89e19a` / `7a89e19a` |
| rect 39² | alma | 2.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.019, 0.996 | 9.3e-09 | `e281abf3` / `9428eca2` |
| rect 39² | alma | 3.5 | `InversionInterferometerSparseNumba` | 0.0e+00 | 1.002, 1.000 | 2.6e-05 | `e281abf3` / `14d63360` |
| rect 39² | alma | 5.0 | `InversionInterferometerSparseNumba` | 1.9e-09 | 0.996, 1.000 | 9.3e-04 | `e281abf3` / `9428eca2` |
| rect 39² | alma_high | 2.0 | `InversionInterferometerSparseNumba` | 0.0e+00 | 0.998, 1.003 | 1.5e-08 | `7a89e19a` / `9428eca2` |
| rect 39² | alma_high | 3.5 | `InversionInterferometerSparseNumba` | 7.5e-09 | 0.999, 0.999 | 1.5e-03 | `e281abf3` / `14d63360` |
| rect 39² | alma_high | 5.0 | `InversionInterferometerSparseNumba` | 7.5e-09 | 1.002, 1.008 | 1.6e-03 (accepted, see Status) | `7a89e19a` / `9428eca2` |

- **Every measured CPU cell runs the numba arm on `InversionInterferometerSparseNumba`** (the
  harness forces the gate open on that arm) and the FFT arm on `InversionInterferometerSparse`.
  numba and FFT agree to ≤ 7.5e-9 nats (bar 0.5); every step sum is within 0.96–1.03 of its full
  call (band 0.9–1.1); every new row reads `evaluations_per_figure_of_merit` {1, 1}.
- **CPU vs A100 log-evidence.** The CPU arms (fnnls) and the A100 (PDIP) fit the same data at the
  same instance. Every Delaunay cell agrees to ≤ 2.4e-5 nats. The rectangular cells, which solve
  on the edge-zeroed subset (`solve_subset_edge_zeroed`), are looser: sma, alma r2.0 / r3.5 and alma_high
  r2.0 agree to ≤ 2.6e-5, **sdp81 r3.5 to 7.8e-4 and alma r5.0 to 9.3e-4** (inside the 1e-3 bar),
  and **alma_high r3.5 to 1.5e-3 nats and alma_high r5.0 to 1.59e-3 nats, which both miss the 1e-3 bar**. alma_high r3.5 and alma r5.0
  are phase-2/3 rows; sdp81 is new and passes. alma_high r5.0 is new and misses. The human accepted that miss on
  2026-10-04 on the r3.5 precedent: the A100 is PDIP on pre-#595 PyAutoArray `9428eca2` and the CPU is fnnls on `7a89e19a`. On the edge-zeroed subset the A100's PDIP
  (converged, 14–15 iterations) and the CPU's fnnls stop at slightly different iterates;
  1.5e-3 nats on log L = −6.0e7 is a relative 2.5e-11 and 330× inside the 0.5-nat evidence bar,
  so it changes no decision here, but it is recorded as a miss, not rounded away.
- **sdp81, all paths.** Delaunay: numba, FFT and A100 sparse within 3.4e-6 nats, A100 dense
  within 4.7e-10 of A100 sparse. rect: numba = FFT (0.0), CPU vs A100 7.8e-4, A100 dense within
  4.7e-10 of sparse. MGE: A100 W~ vs eager dense 3.8e-7. Every sdp81 path agrees far inside the
  0.5-nat bar.

## Indicative time per fit

**Indicative only.** No interferometer Nautilus run has recorded its evaluation count, so this
multiplies one likelihood by an **imaging** Nautilus count and ignores sampler overhead,
batching (vmap) and parallel lanes:

- **73,215** — the Q1 example lens `vis_pix` stage (RAL job 343391, A100, Nautilus sampling
  finished at 73,215 calls, log Z +16607.12; the `vis_lp` stage took 71,900). Copied from
  `tmp/witt_wynne_q1_email/q1_summary.md` §4 (untracked): *"Nautilus finished sampling at
  15:24:09 (`vis_lp`, 71 900 calls, log Z +16668.07) and 15:56:09 (`vis_pix`, 73 215 calls,
  log Z +16607.12)"*.
- For a SLaM pipeline, the HST Delaunay-1250 A100 run
  (`autolens_inference/results/slam/imaging/hst/delaunay_1250/hpc_a100_jax_gpu_sparse_fp64/stages_seed0.json`)
  records `likelihood_evals` per stage: source_lp[1] 89,350 (parametric), source_pix[1]
  **51,420**, source_pix[2] 2,200, light[1] 10,340, mass_total[1] 27,180. The four stages with a
  pixelized source sum to 91,140, i.e. ~1.24× the single-stage column below.

| mesh | instrument | r | best CPU (1 thread) ms | → per fit × 73,215 | A100 W~ ms | → per fit × 73,215 |
|---|---|---:|---:|---:|---:|---:|
| Delaunay-1500 | sma | 2.0 | 244.9 | 5.0 h | 31.4 | 38 min |
| Delaunay-1500 | sma | 3.5 | 277.7 | 5.6 h | 32.6 | 40 min |
| Delaunay-1500 | sma | 5.0 | 450.6 | 9.2 h | 35.8 | 44 min |
| Delaunay-1500 | sdp81 | 3.5 | 1,503 | 30.6 h | 53.7 | 1.1 h |
| Delaunay-1500 | alma | 2.0 | 447.5 | 9.1 h | 39.1 | 48 min |
| Delaunay-1500 | alma | 3.5 | 1,262 | 25.7 h | 49.5 | 1.0 h |
| Delaunay-1500 | alma | 5.0 | 3,789 | 3.2 d | 70.2 | 1.4 h |
| Delaunay-1500 | alma_high | 2.0 | 2,341 | 47.6 h | 54.1 | 1.1 h |
| Delaunay-1500 | alma_high | 3.5 | 9,390 | 8.0 d | 101.5 | 2.1 h |
| Delaunay-1500 | alma_high | 5.0 | 21,821 | 18.5 d | 190.9 | 3.9 h |
| Delaunay-1500 | jvla | 3.5 | not measured | — | 563.3 | 11.5 h |
| rect 39² | sma | 2.0 | 233.7 | 4.8 h | 22.4 | 27 min |
| rect 39² | sma | 3.5 | 823.2 | 16.7 h | 28.1 | 34 min |
| rect 39² | sma | 5.0 | 1,111 | 22.6 h | 33.8 | 41 min |
| rect 39² | sdp81 | 3.5 | 1,851 | 37.6 h | 45.2 | 55 min |
| rect 39² | alma | 2.0 | 783.2 | 15.9 h | 30.6 | 37 min |
| rect 39² | alma | 3.5 | 1,830 | 37.2 h | 44.5 | 54 min |
| rect 39² | alma | 5.0 | 4,653 | 3.9 d | 67.2 | 1.4 h |
| rect 39² | alma_high | 2.0 | 3,739 | 3.2 d | 48.9 | 60 min |
| rect 39² | alma_high | 3.5 | 9,587 | 8.1 d | 95.9 | 1.9 h |
| rect 39² | alma_high | 5.0 | 17,649 | 15.0 d | 198.6 | 4.0 h |
| rect 39² | jvla | 3.5 | not measured | — | 564.4 | 11.5 h |

- The CPU column is **one core**. Nautilus evaluates a batch of points per iteration; spread
  over `n` cores the wall divides by up to `n` (a 64-core node turns the alma r3.5 Delaunay
  25.7 h into ~25 min of sampling, before overhead). The A100 column is one GPU, one JIT call
  per point; batching under vmap lowers it further (Caveats).
- Real interferometer fits spend some calls on parametric stages that do not use the mesh, and
  their call count depends on the data; treat the column as an order-of-magnitude budget.

## Blocked and not-measured cells

Every empty cell in the tables above is one of these. "Not measured" means no job was run for
it; "blocked" means a run was attempted and could not complete.

| Cell | State | Reason | Citation |
|---|---|---|---|
| MGE-20, any instrument, r2.0 / r5.0 | not measured | the MGE campaign ran at the preset r3.5 only; MGE cost is O(N_vis) dense / O(extent) on W~, so the mesh radius rows are the guide | [MGE ledger](./interferometer_mge_breakdown_2026_09.md) Scope |
| MGE-20 dense, A100, alma / alma_high / jvla | blocked (OOM) | the one-shot `transform_mapping_matrix` asks 61.4 GiB / 300.3 GiB / 1.46 TiB; the chunked-transform measurement arm runs (939.6 ms / 3.94 s / 24.20 s) | MGE ledger, "Library W~ route" |
| MGE-20 W~, CPU, alma_high | blocked (laptop OOM-kill) | killed at 10.8 GB anon RSS inside `apply_sparse_operator()` (1M-visibility NUFFT chunk) | MGE ledger, "Not run" |
| MGE-20 dense + W~, CPU, sdp81 | blocked (laptop OOM-kill) | the eager `xp=np` `FitInterferometer` reference (used below 200k visibilities) materialises the nufftax type-2 gather `(20, N_vis, 169)` complex128 ≈ 5.9 GB; the laptop kernel killed the run (exit 137) before any row was written, 2026-09-30. The A100 jobs ran the same eager reference on the RAL host without trouble but do not time it | this note |
| MGE-20, CPU / A100, jvla dense on CPU | not measured | CPU jvla never attempted (dense T alone 559 GiB) | MGE ledger |
| Delaunay / rect, CPU, jvla (any r) | not measured | not run on CPU in any phase (25M visibilities; the RAL CPU numba/FFT rows stop at alma_high) | CPU ledger |
| Delaunay / rect, A100, jvla r2.0 / r5.0 | not measured | phase 3 swept sma / alma / alma_high only | A100 ledger, mask-radius sweep |
| Delaunay / rect, A100 dense, alma (all r) | blocked (OOM) | `T` = N_vis × S complex128 is 22.4 GiB; the chunked transform asks 25.6–27.6 GB | A100 ledger, "Dense (mapping) arm" |
| Delaunay / rect, A100 dense, alma_high / jvla | skipped | `T` alone 112 / 559 GiB > 80 GB (`--dense-max-vis`) | A100 ledger |
| Delaunay / rect, CPU JAX, r2.0 / r5.0 and sdp81 | not measured | the JAX-CPU FFT route was measured in phase 1 at r3.5 only; the NumPy FFT arm is the CPU FFT column | CPU ledger, RAL CPU baseline |
| Delaunay / rect, sdp81 r2.0 / r5.0 | not measured | the sdp81 row is r3.5 only by plan (#356) | this note |

## Caveats

- **Mixed library revisions across columns.** The r3.5 A100 column is the #324 baseline on
  PyAutoArray `14d63360`; the r2.0 / r5.0 A100 rows (phase 3) ran on `9428eca2`; the phase-2 CPU
  rows on `e281abf3`; the new phase-4 cells on the library mains `7a89e19a` (PyAutoFit
  `b13169e2`, PyAutoGalaxy `4c834ced`, PyAutoLens `efd13c4c`, PyAutoNerves `1ec1c82`) from a
  private clone on RAL (see Provenance). Between `e281abf3` and `7a89e19a` the sparse
  interferometer path gained the array-free / precomputed-data-term streaming work (#588–#593)
  and PDIP's raw-forward polish (#595); the per-call CPU work of these cells is the same F / D /
  fnnls chain, and the log-evidences agree with the older A100 rows to the tolerances in the
  Verification table, so the columns are compared as-is. Compare timings within one row with
  that in mind.
- **The phase-1 alma N-sweep rows are on the uncached library** (F twice, D four times per
  evaluation, pre-#582) and are not used in this matrix; every CPU cell here is on the cached
  library ({1, 1} evaluations).
- **Host contention.** RAL CPU cells share `euclid-ral-gpu-{1,2}` with other jobs; the new cells
  ran while both nodes' 124 CPUs were fully allocated by other CPU jobs (load average ≈ 70 at the
  end of the sma tasks, recorded per JSON in `configuration.host_load_avg_*`; the dgemm controls
  are in `control_dgemm_{head,tail}_s`). Single-thread cells on a busy node read slower than on a
  quiet one; the numba / FFT *ratio* within a JSON is the robust quantity.
- **CPU MGE rows are laptop JAX-CPU**, one thread, on a shared host, not RAL. They set the order
  of magnitude (dense is O(N_vis) per call, W~ is not), not a like-for-like device ratio with
  the RAL mesh rows.
- **A100 per-call is one JIT, no vmap.** A sampler that batches likelihoods under `jax.vmap`
  gets the A100 ledger's vmap per-call numbers (e.g. alma Delaunay r3.5 29.4 ms at b64 vs 49.5
  ms single). The fit-time column therefore overstates the A100 fit time for batched samplers.
- **The alma A100 dense arm OOMs at every radius** because `T` is N_vis × S (22.4 GiB complex128
  at alma), independent of the mask; at sdp81 `T` is ~2.4 GiB.
- **alma rows at r4.25 / r6.0** (phase 2) exist on CPU only and are left out of the r2.0 / 3.5 /
  5.0 grid; they bracket the numba / FFT crossover and are in the
  [CPU ledger](./interferometer_mesh_cpu_breakdown_2026_09.md#the-crossover-alma-mask-radius-sweep).
- **Nautilus counts are imaging counts** (see Indicative time per fit).

## Provenance

- **Submits (new):**
  `hpc/batch_cpu/submit_breakdown_interferometer_{delaunay,pixelization}_numba_ral_radius_gaps_fp64`
  (arrays: `0 sma r2.0 | 1 sma r5.0 | 2 alma_high r2.0 | 3 alma_high r5.0`, copied from the phase-2
  `*_crossover_fp64` arrays), `hpc/batch_cpu/submit_breakdown_interferometer_{delaunay,pixelization}_numba_ral_sdp81_fp64`,
  and `hpc/batch_gpu/submit_breakdown_interferometer_{delaunay,pixelization,mge}_a100_sdp81_fp64` +
  `..._mge_a100_sdp81_fp64_sparse` (copied from the alma A100 submits; the Delaunay one drops the
  ConstantSplit bridge leg, the MGE sparse one drops the #575 branch-library copy).
- **RAL jobs:** CPU arrays **375977** (Delaunay gaps, tasks 0–3: 1:06 / 1:50 / 6:25 / 1:02:23) and **375978** (rect gaps, tasks 0–3: 0:48 / 2:40 / 7:04 / 1:11:36), CPU **375979** / **375980** (sdp81 Delaunay / rect, 4:02 / 4:03), A100 **375981** / **375982** / **375983** / **375984** (sdp81 Delaunay / rect / MGE dense / MGE W~, 2:33 / 1:59 / 1:46 / 2:23). All `COMPLETED` (375978_3 confirmed by `sacct` on 2026-10-04); every `.err` has 0 Tracebacks, 0 `RESOURCE_EXHAUSTED` and 0 float32 truncations. The CPU tasks ran on `euclid-ral-gpu-2` with the node's CPUs fully allocated by other jobs (load ≈ 48–74). The exception is 375978_3, which ran on `euclid-ral-gpu-1` (load 65–75 at import, 2.7 at write). The A100 jobs pended ~70 min because other jobs' CPU-only tasks held every CPU on both gpu nodes (see the RAL partition rule in `hpc/README.md`).
- **Libraries.** The shared RAL mirror `/mnt/ral/jnightin/PyAuto` was 7–16 commits behind the
  library mains and in use by another campaign's running jobs, so it was **not** pulled. The
  phase-4 submits take `PYAUTO_LIB_BASE` (default: the shared mirror, i.e. exactly the copied
  submit) and were run with `sbatch --export=ALL,PYAUTO_LIB_BASE=/mnt/ral/jnightin/PyAuto_branch/interferometer-decision-matrix`,
  a private clone of the mains at PyAutoNerves `1ec1c82`, PyAutoFit `b13169e2`, PyAutoArray
  `7a89e19a`, PyAutoGalaxy `4c834ced`, PyAutoLens `efd13c4c`. Each job asserts `autoarray` /
  `autolens` import from that base and records the SHAs in `source_revisions`.
- **RAL venv:** nufftax 0.6.1 (floor ≥ 0.6.1), numba 0.65.1, jax 0.10.2.
- **RAL worktree:** `/mnt/ral/jnightin/autolens_profiling_wt/interferometer-decision-matrix` on
  the pushed branch. The sma / alma_high `lensed_source.fits` adapt caches were copied from the
  phase-2 worktree (md5 `a978d8b1…` / `c3d20edc…`, the same files every earlier CPU and A100 row
  used); the sdp81 one from the laptop simulation (md5 `384c729a…`).
- **sdp81 dataset:** simulated on the laptop with `scripts/misc/simulators/interferometer.py
  --instrument sdp81` (82 s; summary `results/simulators/interferometer_sdp81_summary_v2026.8.17.1.json`),
  `data.fits` md5 `1773239a…`, `uv_wavelengths.fits` md5 `26cb8b7e…` (identical values to the
  workspace SDP.81 export). Laptop smoke of the Delaunay numba cell on it before submit (2
  instances): numba 2.61 s vs FFT 4.80 s per call (laptop, loaded), nnz/col 29.1, 0.0 nats apart.
