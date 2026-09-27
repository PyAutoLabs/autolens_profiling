# autolens_profiling

Profiling and run-time tracking for [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) likelihood functions and simulators across CPU, laptop GPU, and HPC GPU.

> **Likelihood timing only.** Science runs are the Cortex's and inference benchmarking lives in
> [`autolens_inference`](https://github.com/PyAutoLabs/autolens_inference). The retired inference
> programme's tree (searches framework, `InferenceRefs_v1`, notes) is PyAutoGut ref
> `refs/heads/archive/condemned/autolens-profiling/inference-programme` @
> `c8b605801068ec3de04314b47da8f7272a038ba1` — never cite it.

## Vision

This repository is the single home for PyAutoLens performance measurement. It exists so that the run-times that matter for science — evaluating a real lens likelihood, simulating an Euclid-resolution dataset — are visible, reproducible, and versioned across PyAutoLens releases.

**What is profiled:**

- **Likelihood functions** — imaging, interferometer, point-source, and datacube paths, across the MGE, pixelization, and Delaunay model compositions used in real science cases.
- **Mesh kernels** — full mapper-table profiling for Delaunay and Sibson natural-neighbour `DelaunayNN`, including static-cap scaling and split regularization.
- **Simulators** — run-time tracking for the imaging, interferometer, point-source, cluster, group, and multi-plane simulators.
- **Numerical hazards** — saturations, non-finite gradients, backend divergence, and scale-dependent conditioning mechanisms that shape sampler behavior.

**Hardware tiers covered:**

- CPU (single-machine, numpy backend).
- Laptop GPU (consumer-class, JAX backend).
- HPC GPU (A100 and similar, JAX backend).

**Dataset framing:**

Results are framed by **astronomy instrument** (HST, Euclid, JWST, …) rather than by raw pixel counts. Pixel counts are recorded too, but the headline numbers a reader sees first are the ones that map onto a real observing programme.

## Latest run-times

<!-- BEGIN auto-table:headline -->

**Likelihood runtime** — full-pipeline per-call cost per cell × config:

| Cell | local_cpu_fp64 | local_cpu_mp | local_cpu_fp64_sparse | local_cpu_mp_sparse | PreOptimizationTimes |
|---|---|---|---|---|---|
| `datacube/delaunay/sma` | — | — | — | — | — |
| `imaging/delaunay/hst` | 16.73 s | 17.84 s | 4.14 s | 4.49 s | 4.49 s |
| `imaging/delaunay/jwst` | 48.81 s | 22.43 s | 10.42 s | 14.05 s | 14.05 s |
| `imaging/mge/ao` | 3.11 s | 5.71 s | — | — | 5.71 s |
| `imaging/mge/hst` | 117.7 ms | 164.3 ms | 256.1 ms | 140.5 ms | 140.5 ms |
| `imaging/mge/jwst` | 716.2 ms | 678.5 ms | 387.3 ms | 488.7 ms | 488.7 ms |
| `imaging/pixelization/hst` | 13.72 s | 14.78 s | 5.79 s | 5.25 s | 5.25 s |
| `imaging/pixelization/jwst` | 21.78 s | 43.58 s | 9.57 s | 9.42 s | 9.42 s |
| `interferometer/delaunay/alma` | **GPU-only** | 6.51 s | — | — | 6.51 s |
| `interferometer/delaunay/sma` | 2.58 s | 3.34 s | — | — | 3.34 s |
| `interferometer/mge/sma` | 230.7 ms | 231.5 ms | — | — | 231.5 ms |
| `interferometer/pixelization/sma` | 2.04 s | 2.39 s | — | — | 2.39 s |

**Likelihood breakdown** — latest per-step decompositions:

| Cell | Instrument | Platform | Inversion path | Step-sum total | PyAutoLens version |
|------|------------|----------|----------------|----------------|--------------------|
| `cluster/image_plane` | — | local_cpu_fp64 | dense (mapping) | 4.81 s | v2026.8.17.1 |
| `cluster/image_plane` | — | hpc_ral_cpu_fp64 | dense (mapping) | 4.53 s | v2026.8.17.1 |
| `cluster/source_plane` | — | local_cpu_fp64 | dense (mapping) | 4.5 ms | v2026.7.23.1 |
| `datacube/delaunay` | alma_high | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/delaunay` | alma_high | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `imaging/delaunay` | hst | local_cpu_fp64 | dense (mapping) | 3.34 s | v2026.8.17.1 |
| `imaging/delaunay` | hst | local_cpu_fp64 | sparse (w-tilde) | 8.81 s | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_fp64 | dense (mapping) | 67.5 ms | v2026.8.17.1 |
| `imaging/delaunay` | hst | hpc_a100_fp64 | sparse (w-tilde) | 73.7 ms | v2026.8.17.1 |
| `imaging/delaunay` | hst | hpc_a100_mp | dense (mapping) | 96.8 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_mp | sparse (w-tilde) | 95.5 ms | v2026.7.6.649 |
| `imaging/delaunay_nn` | hst | local_cpu_fp64 | dense (mapping) | 4.49 s | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | hpc_a100_fp64 | dense (mapping) | 74.9 ms | v2026.8.17.1 |
| `imaging/delaunay_nn` | hst | hpc_a100_fp64 | sparse (w-tilde) | 84.7 ms | v2026.8.17.1 |
| `imaging/delaunay_numba` | euclid | local_cpu_fp64 | sparse (numba) | 255.0 ms | v2026.8.17.1 |
| `imaging/delaunay_numba` | hst | local_cpu_fp64 | sparse (numba) | 2.04 s | v2026.8.17.1 |
| `imaging/mge` | hst | local_cpu_fp64 | dense (mapping) | 40.3 ms | v2026.8.17.1 |
| `imaging/mge` | hst | hpc_a100_fp64 | dense (mapping) | 7.8 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | local_cpu_fp64 | dense (mapping) | 3.40 s | v2026.8.17.1 |
| `imaging/pixelization` | hst | local_cpu_fp64 | sparse (w-tilde) | 10.17 s | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_fp64 | dense (mapping) | 60.0 ms | v2026.8.17.1 |
| `imaging/pixelization` | hst | hpc_a100_fp64 | sparse (w-tilde) | 70.8 ms | v2026.8.17.1 |
| `imaging/pixelization` | hst | hpc_a100_mp | dense (mapping) | 56.4 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_mp | sparse (w-tilde) | 55.5 ms | v2026.7.6.649 |
| `imaging/pixelization_numba` | euclid | local_cpu_fp64 | sparse (numba) | 611.9 ms | v2026.8.17.1 |
| `imaging/pixelization_numba` | hst | local_cpu_fp64 | sparse (numba) | 1.30 s | v2026.8.17.1 |
| `interferometer/delaunay_numba_direct_conv` | alma | local_cpu_fp64 | sparse (numba) | 1.80 s | v2026.8.17.1 |
| `interferometer/delaunay_numba_direct_conv` | sma | local_cpu_fp64 | sparse (numba) | 378.2 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_jax` | alma | local_cpu_fp64 | sparse (w-tilde) | 3.28 s | v2026.8.17.1 |
| `interferometer/delaunay_numba_jax` | sma | local_cpu_fp64 | sparse (w-tilde) | 727.7 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_reference` | alma | local_cpu_fp64 | sparse (numba) | 4.54 s | v2026.8.17.1 |
| `interferometer/delaunay_numba_reference` | sma | local_cpu_fp64 | sparse (numba) | 535.9 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_source_loop` | sma | local_cpu_fp64 | sparse (numba) | 461.7 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_symmetric` | sma | local_cpu_fp64 | sparse (numba) | 443.1 ms | v2026.8.17.1 |
| `interferometer/delaunay_numba_two_stage` | sma | local_cpu_fp64 | sparse (numba) | 388.7 ms | v2026.8.17.1 |
| `interferometer/mge` | alma | local_cpu_fp64 | dense (mapping) | 41.19 s | v2026.8.17.1 |
| `interferometer/mge` | alma | local_cpu_fp64 | sparse (w-tilde) | 32.7 ms | v2026.8.17.1 |
| `interferometer/mge` | alma | hpc_a100_fp64 | dense (mapping) | 943.2 ms | v2026.8.17.1 |
| `interferometer/mge` | alma | hpc_a100_fp64 | sparse (w-tilde) | 2.7 ms | v2026.8.17.1 |
| `interferometer/mge` | alma_high | local_cpu_fp64 | dense (mapping) | 212.19 s | v2026.8.17.1 |
| `interferometer/mge` | alma_high | hpc_a100_fp64 | dense (mapping) | 3.88 s | v2026.8.17.1 |
| `interferometer/mge` | alma_high | hpc_a100_fp64 | sparse (w-tilde) | 19.6 ms | v2026.8.17.1 |
| `interferometer/mge` | alma_high | hpc_a100_mp | dense (mapping) | 3.93 s | v2026.8.17.1 |
| `interferometer/mge` | jvla | hpc_a100_fp64 | dense (mapping) | 23.81 s | v2026.8.17.1 |
| `interferometer/mge` | jvla | hpc_a100_fp64 | sparse (w-tilde) | 16.7 ms | v2026.8.17.1 |
| `interferometer/mge` | jvla | hpc_a100_mp | dense (mapping) | 23.72 s | v2026.8.17.1 |
| `interferometer/mge` | sma | local_cpu_fp64 | dense (mapping) | 84.0 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | local_cpu_fp64 | sparse (w-tilde) | 11.8 ms | v2026.8.17.1 |
| `interferometer/mge` | sma | hpc_a100_fp64 | dense (mapping) | 856.2 ms | v2026.8.17.1 |
| `interferometer/mge_dft` | sma | local_cpu_fp64 | dense (mapping) | 32.9 ms | v2026.8.17.1 |
| `interferometer/pixelization_numba_direct_conv` | alma | local_cpu_fp64 | sparse (numba) | 1.76 s | v2026.8.17.1 |
| `interferometer/pixelization_numba_direct_conv` | sma | local_cpu_fp64 | sparse (numba) | 184.5 ms | v2026.8.17.1 |
| `interferometer/pixelization_numba_jax` | alma | local_cpu_fp64 | sparse (w-tilde) | 2.28 s | v2026.8.17.1 |
| `interferometer/pixelization_numba_jax` | sma | local_cpu_fp64 | sparse (w-tilde) | 584.6 ms | v2026.8.17.1 |
| `interferometer/pixelization_numba_reference` | alma | local_cpu_fp64 | sparse (numba) | 7.31 s | v2026.8.17.1 |
| `interferometer/pixelization_numba_reference` | sma | local_cpu_fp64 | sparse (numba) | 480.5 ms | v2026.8.17.1 |
| `interferometer/preload` | alma | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 |
| `interferometer/preload` | alma_high | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 |
| `interferometer/preload` | sma | local_cpu_fp64 | sparse (numba) | — | v2026.8.17.1 |
| `point_source_image/image_plane` | simple | local_cpu_fp64 | dense (mapping) | 62.7 ms | v2026.8.17.1 |
| `point_source_image/image_plane` | simple | hpc_ral_cpu_fp64 | dense (mapping) | 24.7 ms | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | local_cpu_fp64 | dense (mapping) | 438 μs | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | hpc_ral_cpu_fp64 | dense (mapping) | 146 μs | v2026.8.17.1 |
| `point_source_source/source_plane` | simple | hpc_a100_fp64 | dense (mapping) | 222 μs | v2026.8.17.1 |
<!-- END auto-table:headline -->

The tables above are auto-generated by `scripts/misc/tooling/build_readme.py` from the artifacts under [`results/`](./results/README.md) — never edit them by hand; run `python scripts/misc/tooling/build_readme.py` after a profiling run and commit the result (CI checks idempotence via `--check`). Narrative context — per-cell "where to optimize next" recommendations and the mp-vs-fp64 verdicts — lives in [`scripts/misc/likelihood_runtime/OPTIMIZATION_NOTES.md`](./scripts/misc/likelihood_runtime/OPTIMIZATION_NOTES.md).

**PreOptimizationTimes** is the named baseline the upcoming optimization work is measured against: a frozen snapshot of the full campaign (laptop CPU, HPC CPU, HPC A100 × fp64/mp) under `results/baselines/PreOptimizationTimes/`, rendered as a baseline column in the dashboard once populated. The convention is defined in [`results/notes/design_lock_in.md`](./results/notes/design_lock_in.md).

(Historical multi-config sweeps up to 2026-07 were committed under [`autolens_workspace_developer/jax_profiling/results/jit/`](https://github.com/PyAutoLabs/autolens_workspace_developer/tree/main/jax_profiling/results/jit); sweeps now write in-repo to `results/runtime/` by default.)

## JAX gradients and compile time

Exploratory gradient work continues in
[`autolens_workspace_developer/jax_profiling/gradient/`](https://github.com/PyAutoLabs/autolens_workspace_developer/tree/main/jax_profiling/gradient);
sampler-level benchmarking belongs to
[`autolens_inference`](https://github.com/PyAutoLabs/autolens_inference).

**Compile time** is profiled separately from run time, because for JAX
likelihoods XLA compilation is a first-class cost in its own right —
`scripts/misc/jax_compile/` measures trace / compile / first-call / steady-state
separately per likelihood × transform. Standing conclusions:

- **Settings suffice** — the persistent compilation cache, `--xla_gpu_autotune_level=0`
  and `--xla_gpu_enable_triton_gemm=false` (all shipped as autonerves defaults) take the
  worst measured first fit from ~70 min to ~35 s; the Triton flag is there because at
  autotune level 0 XLA's default Triton tile makes dense fp64 GEMMs ~5× slower than
  cuBLAS ([`results/notes/xla_autotune_triton_gemm.md`](./results/notes/xla_autotune_triton_gemm.md)).
  Never restructure a likelihood or sampler for compile time
  ([`scripts/misc/jax_compile/README.md`](./scripts/misc/jax_compile/README.md)).
- **`af.MultiStartProdigy` compile is a non-problem** on MGE and every pixelized
  mesh (rectangular / KNN / Delaunay) — ≤ 75 s cold, ≤ 2 s warm on a 32-core
  node. The Delaunay family, however, **never hits the persistent cache** and
  pays full compile in every process, and rectangular's `batch_size=4` is
  load-bearing for memory (~9.2 GB per start).
  Findings: [`results/notes/multistart_prodigy_compile_census.md`](./results/notes/multistart_prodigy_compile_census.md).
- **Multi-band `FactorGraphModel` + MultiStartProdigy compile is fixed at the
  source** (PyAutoFit#1430): `batch_size` now sweeps vmapped chunks from a
  Python loop instead of an in-XLA `lax.map` scan, and the broad-start filter
  is jitted. Cold multi-band fit on the 1-core laptop: intractable → ~6.5 min
  (CPU and laptop GPU); warm ~2–3 min, bit-identical numerics. The scan
  explosion is CPU-backend-specific — the GPU pipeline compiles it fine.
  Findings: [`results/notes/multiband_pyloop_productized.md`](./results/notes/multiband_pyloop_productized.md).
- **Matrix-free pixelized likelihood (2026-09)** — PCG + SLQ reference vs the exact
  Cholesky path at n≈1500 and on the N_src 3000–12000 sweep: no-go on speed (~350× at
  1500, ~16× at 12000, no crossover) and on log-det accuracy (SLQ never reaches the
  0.5-nat bar); cond(F+λH) ≈ 4e10 is set by the linear-MGE columns. Dense and sparse
  single calls fit an A100 to n=12000; the NNLS row is 60–85 % of every call.
  Findings: [`results/notes/matrix_free_pixelized_2026_09.md`](./results/notes/matrix_free_pixelized_2026_09.md).
- **Fixed lens light, source-only inversion (2026-09)** — with the MGE converted to
  regular profiles at their solved intensities and subtracted ("S3"), cond(F+λH) drops
  4.1e10 → 1.2e7 / 2.1e6 and the library call falls 26–30 % with no solver change
  (rect 51.7 → 38.4 ms). A certified active-set positive solve then returns the exact
  constrained optimum (≤ 1.2e-10 nats) in 4.21 ms at pass 2 on Delaunay and 11.05 ms at
  pass 7 on rectangular, against a 25.8–28.3 ms S3 PDIP row; dropping positivity costs
  6.4 / 335 nats and stays blocked on an injection witness.
  Findings: [`results/notes/fixed_lens_light_source_only_2026_09.md`](./results/notes/fixed_lens_light_source_only_2026_09.md).
- **Fixed lens light, the library-path row (2026-09)** — the whole `FitImaging.figure_of_merit`
  jit with the solver actually inside it, on the A100. With S3 and a certified active-set solve
  injected at the library's own positive-solve entry point (a harness monkeypatch — no
  PyAutoArray change), the call falls 50.97 → 25.10 ms on rectangular (2.03×), 65.10 → 25.39 ms
  on Delaunay (2.56×) and 72.95 → 36.26 ms on DelaunayNN (2.01×), matching the library's own
  likelihood to 1e-15…1e-11. Phase 0's projection held. Dropping positivity is faster still
  (21.13 ms) and still costs +334.93 nats. Two traps: the library subsets to
  `solve_ids_to_keep` *before* calling its solver, and `lax.cond` under `vmap` runs both
  branches.
  Findings: [`results/notes/fixed_lens_light_library_path_2026_09.md`](./results/notes/fixed_lens_light_library_path_2026_09.md).
- **Fixed lens light on CPU and a consumer GPU (2026-09)** — the phase-1 rows again on a laptop:
  JAX-CPU at 1 and 8 threads, and an RTX 2060 (6 GB) in fp64 and mixed precision. The certified
  active set wins everywhere and the prize shrinks with the hardware — 2.03×/2.56×/2.01× on the
  A100, 1.28×/1.48×/1.46× on the RTX 2060, 1.83×/1.74×/1.35× on 8 CPU threads, 1.16–1.22× on one.
  The GeForce fp64 penalty never bit (mixed precision buys 5–8 % for ≤ 2.5e-3 nats and 16 % more
  VRAM); **memory is the consumer wall** — `@vmap 16` needs 11.88 GiB, batch 4 OOMs, batch 2 is
  slower than a single call, and the same shape OOM-killed a 16 GB host. In numpy the certified
  active set does *not* beat the library's own `fnnls` NNLS, and both are 1.9–3.6× slower at 8
  BLAS threads than at 1.
  Findings: [`results/notes/fixed_lens_light_hardware_2026_09.md`](./results/notes/fixed_lens_light_hardware_2026_09.md).
- **Fixed lens light on low-likelihood draws (2026-09)** — is the certified active set fast only
  because the model is good? A seeded 41-model draw set (four one-parameter walks bisected onto
  Δlog L ≈ −10 / −100 / −1000 / −1e4, plus 24 random draws from SLaM-like priors at 5σ), run on
  the A100 and on JAX-CPU. **Both phase-0 pass budgets break.** The pass count *grows* with model
  error on Delaunay (Spearman +0.698; pass 2 falls back on **67.5 %** of the set and 95.8 % of the
  random draws) and *falls* on rectangular (−0.535 — a worse model has a bigger active set but an
  easier one), where the spread alone puts the pass-7 budget at a **27.5 %** fallback rate. The
  smallest zero-fallback budgets are **7 (Delaunay)** and **11 (rectangular)**. The lever survives
  at about half its fiducial headline — 2.6×/5.4× median over PDIP, worst draw still faster than
  the best PDIP call — and PDIP's own cost barely moves with the model (15→17 / 17→21 iterations),
  so budget-plus-fallback stays the right design. Pass counts and PDIP iterations are **identical**
  on the A100 and the CPU, so the fallback rate is a property of the problem. Dropping positivity
  is now unambiguously out: the A2 error grows to +4.1e4 / +6.3e3 nats.
  Findings: [`results/notes/fixed_lens_light_low_likelihood_draws_2026_09.md`](./results/notes/fixed_lens_light_low_likelihood_draws_2026_09.md).
- **Fixed lens light, source-pixel scaling per hardware (2026-09)** — the same cell at
  N_src 500 / 1000 / 1500 / 2500 / 4000, on rect and Delaunay, over four hardware legs (A100
  fp64 `@vmap 16`; RTX 2060 6 GB in fp64 and mixed precision; JAX-CPU at 8 threads) — 40 legs,
  none OOMed, none timed out. **The certifying pass budget does not scale with N**: it wanders
  in 5–10 (rect) and 1–2 (Delaunay) with no trend, so phase 3's safe budgets **11 / 7** hold
  from 500 to 4000 pixels. The certified active set leads at every N — 1.44–1.62× over S3 PDIP
  on rect and 2.21–3.00× on Delaunay (A100), and on Delaunay the lever *grows* with N because
  PDIP's iteration count does. **Phase 1's batched row has an N ceiling**: `@vmap 16` amortises
  3.6× at N≈500, nothing at 2500, and is a **1.5× penalty at 4000** on an A100. Memory is not
  the wall — the 6 GB laptop card fits the single call at ~4000 pixels (3.13 GB fp64 / 3.63 GB
  mp) and the A100 uses 7.8 GB of 80 with 16 lanes. Affordable N is **4000 (A100) / 1500 (RTX
  2060) / 1000 (JAX-CPU)**, set by time. On the A100 every solver row fits α ≈ 1 while the
  dense `F+λH` build fits **α ≈ 1.69** and overtakes the certified solve above ~2500 pixels —
  the next lever there is the assembly, not the solver.
  Findings: [`results/notes/fixed_lens_light_source_pixel_scaling_2026_09.md`](./results/notes/fixed_lens_light_source_pixel_scaling_2026_09.md).
- **Fixed lens light on the numba CPU path (2026-09)** — the same lever on the *production*
  CPU path: the whole `AnalysisImaging.log_likelihood_function`, numba sparse operator, HST
  Delaunay N=1500, one thread. Fixing the lens light takes the call **932.4 → 459.2 ms
  (2.03×)** with no solver change (laptop 2.036×, second host). The S3 call then splits
  28 % positivity solve / 25 % `F + λH` / 21 % assembly / 18 % log-dets, and its single
  largest site is `inversion.regularization_matrix` (112.3 ms, 24 %) — bigger than
  `fnnls_cholesky` (62.1 ms). A factor-reuse NNLS (one Cholesky + 15 downdates) reaches
  412.6 ms (1.11×), but the library's own cross-evaluation memo — **on by default, so already
  in production** — reaches 404.6 ms (1.13×) by removing the wrapper's dense-sign seed
  (59.8 → 7.5 ms), so **no PyAutoArray solver change is proposed**. Remaining solver headroom
  is ~9 % of the call; the untouched levers are the regularization matrix and the two log-dets.
  Findings: [`results/notes/fixed_lens_light_numba_2026_09.md`](./results/notes/fixed_lens_light_numba_2026_09.md).
- **HST GPU non-solver residue, phase 1 (2026-09)** — the first *measurement* of where the
  **fused production program** spends its device time: one `jax.jit`, one process, one XLA
  timeline, every GPU kernel joined to the library source line that emitted it through the
  compiled HLO's stack frame index. Three A100 legs and two RTX 2060 legs, HST Delaunay
  N=1500 at the production pass budget 7; every table reconciles to its wall within 2.6 %
  with **zero unjoined kernel time**. **The campaign map's "~13.9 ms mesh / mapper / weights
  / imaging / blurring" bucket is refuted** — mesh, mapper and weights are **0.37 ms**
  (1.2 %). The production A100 call (**31.64 ms**, not the map's 25.39 ms, which was a
  pass-budget-2 number) is **32 % certified solve, 25 % device idle — 5.44 ms of it one gap
  at the qhull `pure_callback` — 22 % PSF convolution of the mapping-matrix cube, 13 % the
  `A.T A` GEMM**. The HLO census settles two drafts: **no `add` of `F + λH` survives at
  `abstract.py:371`** (XLA fuses it into a single shared `cublas-lt` producer, so
  `curvature_reg_matrix_rebuilt_every_access` has no JAX cost), and the two
  `operated_mapping_matrix_list` accesses **compile to one PSF convolution**. The border
  relocator is production's default for lensing and costs 0.100 ms. No PyAutoArray change.
  Findings: [`results/notes/hst_gpu_residue_phase1_2026_09.md`](./results/notes/hst_gpu_residue_phase1_2026_09.md).
- **HST GPU non-solver residue, phase 3 (2026-09)** — the PSF convolution of the mapping-matrix
  cube, phase 1's largest *computation* (7.11 ms, 22 %), measured as seven harness-injected candidates
  inside the same fused whole-call jit on the A100 (HST Delaunay N=1500, fp64, budget 7, array 350573).
  Each fp64 row is gated at `1e-9` against the unmodified library on the fiducial and eight seeded
  draws, and every trace reconciles within 0.5 % with zero unjoined time. **There is no fp64 lever**:
  the library's FFT of the padded `(180, 180, 1500)` cube is the fastest fp64 implementation measured.
  A source-first layout ties (31.79 vs 31.66 ms). A power-of-two frame is +4.52 ms at 2× peak memory.
  Real-space convolution is 1.45× (batched cuDNN) and 3.82× (the library's `use_fft=False`) slower.
  Only the fp32-cube and full-complex64 rows are faster (−2.14 / −3.98 ms, 7-13 %). They miss the pin
  by ~1e-3 nats, so they are **diagnostic, not levers**; accepting them is a human policy decision.
  No PyAutoArray change.
  Findings: [`results/notes/hst_gpu_residue_phase3_psf_2026_09.md`](./results/notes/hst_gpu_residue_phase3_psf_2026_09.md).
- **Point-source CPU speed-up, phase 1 (2026-09)** — the RAL CPU fp64 baseline for the
  PointSolver image-plane likelihood on the unoptimized library revisions (PyAutoArray
  `22e6d608`, PyAutoLens `2aaa1c1a`), job 350580 on an idle `ral` Xeon 8490H node at 8 CPUs.
  Simple fused solved 24.69 ms and plain control 24.72 ms. Cluster (13 components, two sources)
  fused plain 128.05 ms and solved 134.52 ms. Likelihoods and cluster positions are bit-identical
  to the laptop. The laptop rows are load-inflated (2.5–3.0× slower on the simple cell) and are
  not baselines. The reported September 4.9× / 2.4× gain from dropping the throwaway `jnp.unique`
  is still a hypothesis for phase 2.
  Findings: [`results/notes/point_source_cpu_campaign.md`](./results/notes/point_source_cpu_campaign.md).
- **Point-source CPU speed-up, phase 2 (2026-09)** — removing the throwaway JAX-path `jnp.unique`
  vertex dedup (static `size=3N` made it save zero deflections while sorting 3N fp64 rows every
  refinement step; PyAutoArray #568). Interleaved in-process A/B, RAL job 350582 (`ral`, EPYC 7763,
  8 CPUs): simple solved 24.00 → 5.38 ms (**4.47×**, 90 % CI 4.37–4.57), vmap-4 2.63×, two-source
  cluster 155 → 78 ms (**1.98×**); HLO sorts 15 → 7 / 25 → 12, compile 11–21 % faster. Log-likelihoods,
  positions and gradients bit-identical on every instance. GPU regression check (RAL job 350587,
  A100, 20 × 20 calls, branch PyAutoArray imported): no regression, nodedup 1.71–2.05× faster
  (simple solved 1.70 → 0.84 ms, cluster 4.21 → 2.46 ms), every gate bit-identical. Accepted; the post-fix call is
  deflection-dominated (70 % of FLOPs), so phase 3 is the static initial-lattice precompute.
  Findings: [`results/notes/point_source_cpu_campaign.md`](./results/notes/point_source_cpu_campaign.md).
- **Point-source CPU speed-up, phase 3 (2026-09)** — precompute the JAX PointSolver's static step-0
  lattice: deflect the 11 859 geometrically unique vertices instead of the 69 849 flat slots
  (46 516 vs 276 507 on the cluster; PyAutoArray #568, PyAutoLens). Four-route in-process A/B
  `static_lattice_ab.py`, RAL job 350636 (`ral`, pinned Xeon 8490H, 8 CPUs, folding off): simple solved
  3.54 → 1.76 ms (**2.01×**), vmap-4 1.43×, two-source cluster 47.4 → 9.1 ms (**5.21×**); FLOPs −52 % /
  −72 %, XLA temp 4.8 → 1.7 MB / 91.6 → 22.6 MB, compile +3.3–12.2 % (+16.5 % worst with constant folding on,
  which gives 2.1–4.5× on the scalar rows). A100 (job 350637): no regression, 1.01–1.07× faster. All 31 gates bit-identical.
  Accepted; a bit-exact source-on-vertex tie (control 3 images incl. a duplicate root vs lattice 2) PASSED by
  human decision 2026-09-24 and is pinned as a PyAutoLens test. Post-fix, the refinement-step deflections (≈ 60 % of simple FLOPs) lead phase 4.
  Findings: [`results/notes/point_source_cpu_campaign.md`](./results/notes/point_source_cpu_campaign.md).
- **Point-source source-plane, phase 2a (2026-09)** — RAL rows for the `source_plane` breakdown and an
  interleaved pytree-input A/B `pytree_input_ab.py` (#322): production `ModelInstance` pytree argument vs a
  flat physical vector (`instance_from_vector` inside the trace) vs flat leaves (`tree_unflatten` inside the
  trace). RAL job 356368 (`ral`, pinned Xeon 8490H, 8 CPUs, quiet): fused solved 0.1465 ms, plain 0.1450 ms
  (the loaded laptop's 0.44 ms was ~3× inflated); `value_and_grad` 2.26× forward. A/B, 20 × 20:
  pytree/flat_vector **1.36×** solved (90 % CI 1.34–1.39, 0.0385 ms saved) and **1.44×** plain (1.43–1.46,
  0.0452 ms); flat_leaves ≈ flat_vector, so the cost is the Python `ModelInstance` flatten. A100 (job 356369):
  single call 0.22–0.27 ms, launch-bound (scalar floor 0.128 ms), pytree/flat_vector 1.25–1.28×. All route
  agreement (rtol 1e-10) and non-zero-gradient asserts passed. Phase-2b rule (≥ 0.05 ms AND ≥ 15 % on RAL CPU):
  **no-go** (27–31 % but < 0.05 ms); the backward-pass lever (+0.185 ms) is promoted.
  Findings: [`results/notes/point_source_source_plane_campaign.md`](./results/notes/point_source_source_plane_campaign.md).

## How to read this repo

Performance profiling scripts write two timing artifact shapes under `results/`
(full reference: [`results/README.md`](./results/README.md)):

```
# Versioned summaries — standalone runs; history retained side-by-side
results/<section>/<subfolder>/<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].{json,png}

# Per-config sweeps — sweep.py + aggregate.py; latest sweep per cell
results/runtime/<class>/<model>[/<instrument>]/<config_name>[_sparse].{json,png,log} + comparison.{json,png}
```

The version string matches the PyAutoLens release that produced the numbers (e.g. `v2026.5.29.4`). The JSON carries structured timings; the PNG is the at-a-glance plot. Cross-release **trend** questions read the versioned summaries; cross-hardware **comparison** questions read `comparison.json`.

Numerical hazards use a third, semantic shape under `results/hazards/`: stable
finding IDs, typed measurements, source anchors, reproducer plots, and a
consumer-facing `hazards_index.json`. Those records are re-verified by behavior,
not versioned by filename.

## Section index

Scripts are laid out **dataset-first, task-second**: `scripts/<dataset>/<task>/<model>.py`
(`imaging` / `interferometer` / `point_source_image` / `point_source_source` / `multi_dataset` / `cluster`), mirroring the
`autolens_workspace*` repos. Each task's shared drivers, framework and narrative README (with the
auto-tables) live under `scripts/misc/<task>/`; dataset-agnostic tooling lives under `scripts/misc/`.

Beside those dataset-first families sits a second, **dataset-free** axis:
[`scripts/lens/`](./scripts/lens/README.md) profiles a single **library component** — one function,
one grid, one set of fiducial parameters — rather than a pipeline. A dataset is loaded only to build
a realistic grid; nothing about the data enters the measurement. It answers *"what does this piece of
the lensing calculation cost per call, and where inside it does the time go?"*, which is the evidence
library-level optimisation work needs and a pipeline breakdown cannot give. Today that is
[`lens/deflections/`](./scripts/lens/deflections/README.md) (deflection angles per mass profile);
`convergence/`, `potential/` and `shear/` follow the same shape.

| Task (`scripts/<dataset>/<task>/` + shared home) | Contents |
|--------|----------|
| `likelihood_runtime/` · [README](./scripts/misc/likelihood_runtime/README.md) | Full-pipeline JIT only, driven by `scripts/misc/likelihood_runtime/sweep.py` across CPU/GPU/A100 × fp64/mp. *How long will this likelihood take on this hardware?* |
| `likelihood_breakdown/` · [README](./scripts/misc/likelihood_breakdown/README.md) | Per-step JIT decomposition. Single config. *Where does time go inside the likelihood?* |
| `latent/` · [README](./scripts/misc/latent/README.md) | Latent-variable profiling. |
| `quick_update/` · [README](./scripts/misc/quick_update/README.md) | Fast incremental re-profiling helpers. |
| [`scripts/misc/jax_compile/`](./scripts/misc/jax_compile/README.md) | JAX/XLA **compile-time** profiling — trace / compile / first-call / steady split per likelihood × transform. *How long before this fit starts running?* |
| [`scripts/misc/vram/`](./scripts/misc/vram/README.md) | GPU memory profiling + the per-cell vmap batch-size table for the A100. |
| [`scripts/misc/delaunay_nn/`](./scripts/misc/delaunay_nn/README.md) | DelaunayNN full-mapper runtime and fixed-shape cap scaling. |
| [`scripts/misc/hazards/`](./scripts/misc/hazards/README.md) | Numerical-hazard profiling — saturations, non-finite gradients, backend divergence, and conditioning mechanisms. |
| [`scripts/misc/simulators/`](./scripts/misc/simulators/README.md) | Run-time tracking for the PyAutoLens simulators. |
| [`scripts/misc/pipeline_resume/`](./scripts/misc/pipeline_resume/README.md) | SLaM pipeline resume overhead — the wall time a re-run pays per completed stage. |
| [`scripts/lens/`](./scripts/lens/README.md) | **Library-component profiling** (dataset-free axis). [`deflections/`](./scripts/lens/deflections/README.md) — per-call deflection-angle cost per mass profile on the numpy CPU path, pinned on the values it computes. |
| [`instruments/`](./instruments/README.md) | Instrument presets (pixel scale, shape) that frame every result. |
| [`hpc/`](./hpc/README.md) | SLURM submit scripts for the RAL HPC (A100 rows of the sweep matrix). |
| [`results/`](./results/README.md) | JSON + PNG artifacts written by the above scripts; named baselines. |

## Roadmap

This repo is being built in phases (bootstrap history now archived in `PyAutoMind`).

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repo bootstrap | ✓ shipped |
| 1 | Mirror JIT likelihood profiling scripts + per-section READMEs | ✓ shipped |
| 2 | Mirror simulator profiling scripts + run-time tracking | ✓ shipped |
| 3 | Nautilus profiling, design for sampler expansion | ✓ shipped, since retired ([#245](https://github.com/PyAutoLabs/autolens_profiling/issues/245)) |
| 4 | Top-level + per-section README dashboard with instrument framing | ✓ shipped |
| 5 | GitHub Actions for lint + profile re-runs + README refresh | ✓ shipped (`lint.yml` per-PR; `profile.yml` manual/on-release) |
| 6 | Design lock-in + results/dashboard groundwork ([#52](https://github.com/PyAutoLabs/autolens_profiling/issues/52)) | in progress |
| 7 | **PreOptimizationTimes** baseline campaign (vram-first, then runtime + breakdown) | ✓ shipped (runtime [#56](https://github.com/PyAutoLabs/autolens_profiling/issues/56); breakdown + dashboard [#59](https://github.com/PyAutoLabs/autolens_profiling/issues/59); laptop-GPU legs extend in a later re-run) |

### Future enhancements (Phase 4 follow-ups)

Dashboards can grow in many directions. The list below captures candidate improvements that fit the "profiling and run-times" theme; none of them block the current dashboard from being useful.

- **Regression-watch indicator** — colour or arrow per cell showing whether the latest cost regressed (>5%) or improved versus the previous PyAutoLens release. Needs the second-latest version per axis kept alongside the latest. Trivial to add to `scripts/misc/tooling/build_readme.py`.
- **Per-axis version-history PNGs** — small inline plot of run-time vs PyAutoLens release version, generated from the JSON artifacts (reusing the `_developer/jax_profiling/results/jit/.../*_v<version>.png` generator). Embeds nicely above each section table.
- **Plotly-rendered interactive timeline** — hostable on GitHub Pages once the static dashboard stabilises; lets readers hover/filter across instrument × model × release.
- **Flamegraph captures** — alongside the headline timing numbers, store a flamegraph per instrument × model for the most recent release.
- **Hardware-tier columns** — extend `scripts/misc/tooling/build_readme.py` table renderers to show CPU / laptop GPU / HPC GPU as separate columns once result artifacts encode the hardware label (filename suffix or JSON `"hardware"` field).
- **Archive old versions** — once a script has >6 minor releases of artifacts, move the older ones to `results/archive/` so the latest views stay uncluttered.

## Related repos

- [`PyAutoLabs/PyAutoLens`](https://github.com/PyAutoLabs/PyAutoLens) — the library being profiled.
- [`PyAutoLabs/autolens_workspace`](https://github.com/PyAutoLabs/autolens_workspace) — user-facing science scripts and tutorials.
- [`PyAutoLabs/autolens_workspace_developer`](https://github.com/PyAutoLabs/autolens_workspace_developer) — the developer workspace this repo's scripts were migrated from; still hosts the pre-2026-07 sweep history and the gradient-profiling work.
- [`Jammy2211/autolens_colab_profiling`](https://github.com/Jammy2211/autolens_colab_profiling) — sibling repo, Colab-specific scope. Not yet migrated to PyAutoLabs.

## Package vs scripts

This repo is a **collection of standalone profiling scripts**, not an installable Python package. There is no `pyproject.toml`. Run scripts from the repo root.

Scripts follow the JIT conventions documented in `autolens_workspace_developer/CLAUDE.md`:

- Extract `.array` from autoarray types before crossing the `jax.jit` boundary (autoarray types are not JAX pytrees as inputs).
- Pass `xp=jnp` through PyAutoLens / PyAutoGalaxy / PyAutoArray functions to select the JAX backend.

## Community & support

- **Slack** — [PyAutoLens workspace](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg) for questions.
- **Issues** — file profiling bugs and feature requests on this repo's [issue tracker](https://github.com/PyAutoLabs/autolens_profiling/issues).

<sub><i><a href="https://open.spotify.com/track/7c584s9RZQzkJDoC08VDJB">i just know that it get better with time</a></i></sub>
