# Interferometer mesh likelihood breakdown — Delaunay-1500 and rectangular on the A100, sparse (W~) path, ranked levers (2026-09)

autolens_profiling issue [#320](https://github.com/PyAutoLabs/autolens_profiling/issues/320),
branch `feature/interferometer-mesh-breakdown-a100`, epic `interferometer-likelihood-campaign`
(2/3). This is the interferometer mirror of the imaging A100 pixelized baseline
([`a100_pixelized_baseline_2026_09.md`](./a100_pixelized_baseline_2026_09.md), #241: Delaunay
67.5 ms, rectangular 60 ms step sum). The MGE sibling is
[`interferometer_mge_breakdown_2026_09.md`](./interferometer_mge_breakdown_2026_09.md) (1/3).

**Headline.** A full alma (1M visibilities) sparse-operator likelihood costs **49.5 ms
(Delaunay-1500) / 44.5 ms (rectangular 39×39)** on the A100 in fp64. That is below the imaging
HST baseline (67.5 / 60 ms). Two rows set the cost. The PDIP positive solve is 47–49 % of the
call at alma. The W~ curvature matrix F is 36–42 % at alma and 91–92 % at jvla, and F is
FFT-bound on the mask extent, not on N_vis. The certified solver cuts the alma Delaunay call
by 33 % today (opt-in, scalar jit).

Cells: `scripts/interferometer/likelihood_breakdown/{delaunay,pixelization}.py`, harness
`scripts/misc/likelihood_breakdown/interferometer_pixelized.py`. SLURM:
`hpc/batch_gpu/submit_breakdown_interferometer_{delaunay,pixelization}_a100_{sma,alma,alma_high,jvla}_{fp64,mp}`
plus the two `..._alma_fp64_n_sweep` arrays. Every number below is read from the committed JSONs:

| Leg | JSON (under `results/breakdown/interferometer/`) |
|---|---|
| A100 alma (fp64, mp) | `{delaunay,pixelization}_hpc_a100_{fp64,mp}.json` |
| A100 alma ConstantSplit bridge | `delaunay_hpc_a100_fp64_constant_split.json` |
| A100 sma / alma_high / jvla | `{sma,alma_high,jvla}/{delaunay,pixelization}_hpc_a100_{fp64,mp}.json` |
| A100 alma N sweep | `n_sweep/delaunay_hpc_a100_fp64_n{1000,2500,4000}.json`, `n_sweep/pixelization_hpc_a100_fp64_n{1024,2500,4096}.json` |
| CPU sma / alma (laptop) | `{delaunay,pixelization}_breakdown_{sma,alma}_v2026.8.17.1.json` |

## Scope — read this before quoting a number

- **Model.** Isothermal + ExternalShear at the simulator truth. The priors are tight Gaussians,
  so the prior median is the truth and the ray-trace prefix is not constant-folded.
  **No lens light:** `simulators/interferometer.py` puts no lens emission in the visibilities.
- **Delaunay:** Hilbert-1500 image mesh, `AdaptSplit(inner=0.1, outer=10, signal_scale=0.1)`
  (the imaging default since #232), 1500 vertices, 1500 solved. The ConstantSplit(1.0) row
  bridges to the v2026.5 numbers. It gives 47.82 ms full / 49.66 ms steps against
  AdaptSplit's 49.51 ms, so the regularization choice does not move the call.
- **Rectangular:** `RectangularBilinearAdaptImage` 39×39 = 1521 pixels, `Constant(1.0)`. It is
  edge-zeroed, so 1369 of 1521 pixels are solved.
- **Path.** `TransformerNUFFT`, `apply_sparse_operator(method="nufft", batch_size=128)`,
  `InversionInterferometerSparse` (recorded as `configuration.inversion_class`). The steps are
  the sparse-path steps: triplets, `D = Lᵀ d~`, blocked-rfft2 F, H, solve, log-dets,
  fast chi-squared. There is no transformed-mapping-matrix row in the sparse arm. The dense
  (`InversionInterferometerMapping`) arm is a separate comparison block.
- **Mask.** 3.5″ radius at every instrument. The W~ operator lives on the mask's bounding-box
  extent (`Mask2D.extent_index_for_masked_pixel`, PyAutoArray `mask/mask_2d.py:746-776`). That
  extent is 70² / 140² / 280² / 700² at sma / alma / alma_high / jvla (pixel scale 0.1 / 0.05
  / 0.025 / 0.01″), and the FFT grid is (2y, 2x): 140² → 1400².
- **Mass fixed.** F changes every call because the mapper changes with the mass. Nothing here
  measures a fixed-mass preload; lever 3 bounds it from the F row.

## Provenance and gate

- A100 = RAL `euclid-ral-gpu-{1,2}`, NVIDIA A100 80 GB PCIe, jobs **356370-356387**, all
  `COMPLETED`. Every `.err` has 0 Tracebacks and 0 `truncated to dtype float32`, and the cell
  refuses to run with x64 off.
- Library revisions: the RAL mirror was refreshed with `HPCPullPyAuto` on 2026-09-26, and the
  mirror HEADs equal the local canonical mains. PyAutoNerves `2b3bc533`, PyAutoFit `cf83504e`,
  PyAutoArray `14d63360` (includes #575 MGE W~ route + #577 real scatter), PyAutoGalaxy
  `0e4b89cf`, PyAutoLens `4487eb47`; autolens_profiling `66e45e90` on RAL. Each JSON records
  them in its `source_revisions`.
- RAL venv vs library floors: nufftax 0.6.1 (floor ≥ 0.6.1, OK — the 0.4.0 trap from 1/3 does
  not apply), jax/jaxlib 0.10.2, jaxnnls 1.0.1, scipy 1.17.1. **Drift: anesthetic 2.8.14 <
  PyAutoFit floor ≥ 2.9.0.** anesthetic is not on the likelihood path, so no number here
  depends on it; it is already tracked by `ral_venv_dependency_floor_drift`. getdist /
  zeus-mcmc are absent (optional extras). The venv was not modified.
- ⚠ **JAX compilation cache NOT fresh on the A100 legs.** Each JSON has
  `autotune_cache_entries_at_start = 204` (the shared cache `/mnt/ral/jnightin/.cache/pyauto_jax`),
  because the submits did not set a per-job `JAX_COMPILATION_CACHE_DIR`. `XLA_FLAGS` carried
  `--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` (autonerves wrapper). At
  autotune level 0 a seeded cache cannot pick a different GEMM lowering than a fresh one, so the
  steady-state rows should stand. **The legs were not re-run with a fresh cache**, so this is an
  argument, not a measurement. Compile-time rows from these JSONs are not cold-compile numbers.
- CPU = shared laptop, one visible core, load avg ~5. Compare CPU rows only within one JSON.

**Witness — PASS.** `delaunay_hpc_a100_fp64.json` (alma) step sum **50.73 ms** vs full JIT
**49.51 ms** (`step_sum_over_full_jit` 1.025, within 10 %). Its step names are the sparse steps
(`Sparse triplets (extent grid)`, `Data vector D = Lᵀ d~`,
`Curvature matrix F = Aᵀ W~ A (12 blocks of 128)`, `Reconstruction (pdip)`, …), with no
transformed-mapping-matrix row.

Correctness gates:

- The standalone steps reproduce the library `FitInterferometer.figure_of_merit` to 0 (sma),
  1.5e-8 (alma), 7.6e-7 (alma_high) and 1.4e-5 nats (jvla, where |log L| ≈ 3e8).
- Sparse = dense (fp64). CPU sma: 0.0 nats on both meshes (F max rel diff 5e-14, D 3e-15).
  A100 sma chunked dense vs sparse: 0.0 on both meshes.
- CPU peak RSS on sma is 1.71 GB (Delaunay) / 1.65 GB (rect), against the prompt's < 4 GB
  target; the old cell was OOM-killed at 14.6 GB. CPU alma is 2.75 / 1.93 GB.
- HLO census of the fused pipeline: 3 fft ops and 4 (Delaunay) / 5 (rect) while loops. That is
  one F build plus its constant Khat transform, with no duplicated FFT work after XLA CSE.

> **Caveat — ~1e-6 to 1e-5 nat GPU spread on the Delaunay AdaptSplit system.** On the A100 at
> sma, library dense one-shot `FitInterferometer` vs library sparse differs by **5.2e-6 nats**
> (Delaunay; rect exactly 0.0). The A100 sma Delaunay figure of merit also sits **2.3e-5 nats**
> from the CPU value (−3162.627213 vs −3162.627236). Certified vs PDIP full pipeline differ
> 1.8e-6. Structural sparse = dense holds: CPU and the A100 chunked arm give 0.0, and rect is
> 0.0 everywhere. The spread is GPU PDIP / reduction round-off on the Delaunay AdaptSplit
> system. At < 1e-4 of the 0.5-nat bar it is **judged not a blocker; no follow-up filed.**

## A100 fp64 baseline — per-call ms, library path, PDIP

This table is the interferometer analogue of #241's 67.5 / 60 ms.

| Instrument | N_vis | Extent (FFT) | Mesh | full JIT | step sum | mapper | **F (W~)** | **solve** | log-dets | other | F share | solve share |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sma | 190 | 70² (140²) | Delaunay | 32.61 | 33.91 | 5.51 | 4.29 | 20.57 | 2.08 | 1.46 | 13 % | 63 % |
| sma | 190 | 70² (140²) | rect | 28.08 | 29.20 | 0.91 | 4.66 | 20.33 | 2.15 | 1.15 | 17 % | 72 % |
| **alma** | 1M | 140² (280²) | **Delaunay** | **49.51** | 50.73 | 5.78 | 17.80 | 23.46 | 2.12 | 1.57 | 36 % | 47 % |
| **alma** | 1M | 140² (280²) | **rect** | **44.46** | 45.46 | 0.95 | 18.71 | 21.99 | 2.25 | 1.54 | 42 % | 49 % |
| alma_high | 5M | 280² (560²) | Delaunay | 101.45 | 100.00 | 6.63 | 67.00 | 21.94 | 2.07 | 2.36 | 66 % | 22 % |
| alma_high | 5M | 280² (560²) | rect | 95.88 | 97.69 | 1.37 | 69.84 | 22.05 | 2.20 | 2.23 | 73 % | 23 % |
| jvla | 25M | 700² (1400²) | Delaunay | 563.31 | 551.62 | 11.86 | 510.45 | 18.90 | 2.28 | 8.13 | 91 % | 3 % |
| jvla | 25M | 700² (1400²) | rect | 564.36 | 556.96 | 3.77 | 517.41 | 23.46 | 2.16 | 10.16 | 92 % | 4 % |

"other" is ray-trace + L + H + triplets + D + chi-squared. Each is < 3.3 ms; D grows from 0.2 to
3.3 ms with the masked-pixel count. Shares are of the step sum. The one-off operator build
(`operator_build_s`) is 2.8 / 3.2 / 4.3 / 7.4 s (sma / alma / alma_high / jvla), amortised over
a search.

Against the imaging HST baseline: alma is 27 % (Delaunay) / 26 % (rect) cheaper than HST
imaging, and alma_high is 1.5× / 1.6× dearer. The solve row is about the same absolute size in
both (~20–23 ms interferometer, 37 ms imaging). Where interferometer differs is F: it sets the
cost from alma_high upward.

**F is set by the mask extent, not by N_vis.** F grows 4.3 → 17.8 → 67.0 → 510 ms as the
extent area M = y·x goes 4,900 → 19,600 → 78,400 → 490,000. That tracks M log M plus a memory
term at jvla (alma → alma_high: 4× M, 3.8× F; alma_high → jvla: 6.25× M, 7.6× F). N_vis grows
by 131,000× over the same ladder. Everything else in the call is flat across instruments.

### Dense (mapping) arm

| | sma Delaunay | sma rect |
|---|---:|---:|
| dense step sum (chunked T, 50 columns) | 67.13 ms | 63.00 ms |
| of which T (NUFFT of the mapping matrix) | 37.66 | 38.19 |
| dense F / D | 0.34 / 0.14 | 0.34 / 0.13 |
| library dense one-shot `FitInterferometer` (full) | 65.57 ms | 61.24 ms |
| sparse full JIT | **32.61** | **28.08** |

The sparse path is 2.0× / 2.2× faster than dense even at sma's 190 visibilities.
**alma dense: OOM on the A100.** The chunked transform requests 27.6 GB (Delaunay) / 25.4 GB
(rect), and T alone is 22.4 GiB complex128. alma_high and jvla were skipped (T 112 / 559 GiB),
and so was CPU alma (22 GiB). For pixelized interferometer fits above sma the sparse path is the
only path.

### vmap (sparse full pipeline, per-call amortised, PDIP)

| Instrument | Delaunay | rect | single-JIT (Del / rect) |
|---|---:|---:|---:|
| sma @ b64 | 15.3 ms | 10.1 ms | 32.61 / 28.08 |
| alma @ b64 | 29.4 ms | 24.1 ms | 49.51 / 44.46 |
| alma_high @ b16 | 86.6 ms | 83.4 ms | 101.45 / 95.88 |
| jvla @ b16 | **OOM** (requests 74 / 75 GB) | **OOM** | |
| jvla @ b4 | 563.3 ms | 558.6 ms | 563.31 / 564.36 |

Batching amortises the small, launch-bound rows. It buys nothing once F dominates: jvla b4
equals single-JIT, because F is memory/FFT-bound.

### CPU (laptop, one core, loaded) — reference only

- sma: Delaunay step sum 1.64–1.74 s vs full 1.55–2.50 s (noisy); rect 1.42 vs 1.71 s.
- alma: Delaunay 4.18 vs 4.25 s, rect 3.69 vs 4.22 s. F is 2.71 / 2.56 s (65–70 %) and PDIP
  1.08 / 0.74 s.
- The A100 is ~85–95× faster at alma. The numba CPU path and the CPU-vs-GPU verdict belong to
  task 3/3.

## N sweep at alma — A100 fp64, full JIT ms

**These rows are the GPU half of the epic decision matrix.** Task 3/3 owns the matrix itself
(`organs/PyAutoMind/draft/research/autolens_profiling/interferometer_mesh_breakdown_numba_cpu_decision_matrix.md`,
witness note `interferometer_likelihood_decision_matrix_2026_09.md`); this note does not build
it.

| Mesh | N | full PDIP | full certified | F | solve PDIP | solve certified (passes) | vmap b16 per call |
|---|---:|---:|---:|---:|---:|---:|---:|
| Delaunay | 1000 | 32.72 | 21.37 | 12.14 | 15.07 | 3.72 (2) | 21.9 |
| Delaunay | 1500 | 49.51 | 33.11 | 17.80 | 23.46 | 6.80 (3) | 29.4 (b64) |
| Delaunay | 2500 | 84.73 | 54.39 | 29.37 | 42.09 | 11.56 (3) | 76.4 |
| Delaunay | 4000 | 142.55 | 91.54 | 47.43 | 74.34 | 24.25 (4) | 179.2 |
| rect | 1024 (32²) | 30.01 | 24.19 | 12.83 | 14.82 | 8.88 (9) | 18.6 |
| rect | 1521 (39²) | 44.46 | 39.65 | 18.71 | 21.99 | 17.34 (12) | 24.1 (b64) |
| rect | 2500 (50²) | 71.24 | 66.22 | 31.07 | 35.37 | 30.51 (13) | 60.2 |
| rect | 4096 (64²) | 119.20 | 109.96 | 48.85 | 62.73 | 53.38 (12) | 145.3 |

- F scales roughly linearly in N (blocks = ceil(N/128), each one FFT pair on the fixed extent).
- PDIP scales ~N^1.2.
- vmap b16 stops amortising above N ≈ 2500. At 4000 it is a 1.26× (Delaunay) / 1.22× (rect)
  **penalty** over single-JIT, matching the imaging fixed-light N ceiling.
- Every N up to 4000 fits the A100 single-call. On time alone, alma Delaunay at N = 4000 is
  ~92 ms with certified, below the imaging HST 1500-pixel baseline.

## Mixed precision vs fp64 — log-evidence shift (in-run fp64 reference; bar 0.5 nats)

| Instrument | Delaunay Δ | rect Δ | time (fp64 → mp) |
|---|---:|---:|---|
| sma | 1.7e-6 | 1.1e-9 | no change |
| alma | 2.8e-5 | 1.2e-5 | 49.51 → 49.33; 44.46 → 44.02 ms |
| alma_high | 3.7e-4 | 8.1e-4 | 101.45 → 100.26; 95.88 → 94.29 ms |
| jvla | 1.3e-2 | 4.4e-3 | 563.3 → 542.7; 564.4 → 550.1 ms (~3 %, within noise) |

Every leg holds the bar, and mp buys nothing. `use_mixed_precision` reaches only the mapper /
mapping matrix; `InterferometerSparseOperator` hard-casts the W~ path to float64
(`jnp.float64` throughout `curvature_matrix_diag_from`, PyAutoArray
`inversion/inversion/interferometer/inversion_interferometer_util.py:1219-1238`). May's "mp
helped at jvla" does not survive on the sparse path.

## Ranked levers

| Rank | Lever | Step attacked | Structural bound | Measured / predicted gain | Follow-up |
|---|---|---|---|---|---|
| 1 | Certified positive solver as the default on the interferometer sparse path | solve (PDIP) | solve share: 63–72 % sma, 47–49 % alma, 22–23 % alma_high, 3–4 % jvla | **measured** (scalar jit): alma Delaunay 49.5 → 33.1 ms (**−33 %**), N=4000 142.6 → 91.5 ms (**−36 %**), alma rect 44.5 → 39.7 ms (−11 %), jvla ≤ −6 % | amendment to certified-solver C2 |
| 2 | F FFT size: pruned padded FFT + real-space extent | F (W~) | F share 36–42 % alma, 66–73 % alma_high, 91–92 % jvla; F ∝ M log M of the mask extent | **predicted**: pruning ≤ ~25 % of the FFT work in F (≤ ~15–20 % of the jvla call); extent: at jvla, a 2× coarser real-space grid is ~4× smaller M | new research prompt |
| 3 | Fixed-mapper `preloads.curvature_matrix` across calls (source-only / fixed-mass searches) | F (W~) | the whole F row: 36–42 % alma → 91–92 % jvla | **bounded, not measured**: e.g. jvla Delaunay 563 → ~53 ms if F is skipped | new research prompt |
| — | F block size / `fori_loop` | F | flat | none: flat within noise | not filed |
| — | fp32 FFT arm | F | — | **slower** on the A100; fails the bar at jvla | not filed |
| — | func-list off-diagonal assembly | F assembly | not exercised | none for mapper-only fits | not filed |
| — | mixed precision | mapper | — | none (above) | not filed |

### 1. Certified solver default for the interferometer sparse path

- **Step attacked:** the positive-only solve (step 8), PDIP today (`nnls.iterations` 14 at alma
  Delaunay).
- **Evidence it already works here.** The interferometer sparse path honours
  `Settings(positive_only_solver="certified")`. The inversion is mapper-only JAX, and
  `positive_only_solver_used == "certified"` was confirmed at trace time.
  - Standalone on the same (D, F + H), certified / PDIP is 0.14–0.33× on Delaunay (0–4 passes)
    and 0.27–0.87× on rect (3–13 passes). The edge-zeroed rect system needs more passes.
  - Certified vs PDIP figure of merit agrees to ≤ 1.2e-7 nats (0.0 on most legs).
- **Measured full pipeline** (`solver_ab.full_pipeline_certified`):

  | Instrument | Delaunay PDIP → certified | rect PDIP → certified |
  |---|---|---|
  | sma | 32.61 → 14.70 ms (−55 %) | 28.08 → 20.02 ms (−29 %) |
  | alma | 49.51 → 33.11 ms (**−33 %**) | 44.46 → 39.65 ms (−11 %) |
  | alma N=4000 / 4096 | 142.55 → 91.54 ms (**−36 %**) | 119.20 → 109.96 ms (−8 %) |
  | alma_high | 101.45 → 84.95 ms (−16 %) | 95.88 → 81.26 ms (−15 %) |
  | jvla | 563.31 → 526.85 ms (−6 %) | 564.36 → 547.50 ms (−3 %) |

- **Why it is not simply "flip the default".** These are scalar `jax.jit(fn)` rows. Production
  runs `jax.jit(jax.vmap(fn))`, where the certified solver's PDIP-fallback `lax.cond` becomes a
  `select` and every lane pays both solvers (certified-solver phase B). The scalar-default flip
  was retired on 2026-09-24 in favour of phase C2's uncertified-lane guard. The interferometer
  lever is therefore **an extra cell for C2**, not its own prompt: C2's witness table today is
  HST imaging only. Filed as an amendment note (see Follow-ups).
- **Predicted gain:** at alma, up to the scalar −33 % (Delaunay) / −11 % (rect). The vmap gain
  is whatever C2 measures. Negligible at jvla, where F is 91 % of the call.

### 2. F extent / FFT size — F is FFT-bound

- **Step attacked:** F = Aᵀ W~ A, `InterferometerSparseOperator.curvature_matrix_diag_from`
  (`inversion_interferometer_util.py:1171-1253`). It runs a `lax.fori_loop` over ceil(S/128)
  column blocks, each a scatter → `apply_operator` → gather + `segment_sum`. `apply_operator`
  (`:1099-1151`) zero-pads each (B, y, x) block to (B, 2y, 2x) and runs a full `rfft2` /
  `irfft2` pair on it (`:1145-1150`).
- **What bounds it (the block-size and kernel-split evidence, prompt items 1 and 5):**
  - The block-size sweep is flat within noise at every instrument. alma Delaunay
    B = 32/64/128/256/512 → 17.98 / 19.32 / 17.80 / 17.99 / 18.45 ms; jvla B = 64/128/256 →
    534 / 510 / 518 ms.
  - Kernel split (one block per part × n_blocks): the FFT apply is **71–85 %** of predicted F,
    gather + `segment_sum` 14–23 % and the scatter 1–8 %. At sma the FFT is 55 %, scatter
    17 %, gather 27 %. The unfused prediction exceeds the measured fused F by 5–40 %.
  - **Verdict: the A100 is FFT-bound** on the (B, 2y, 2x) transform pair, not launch- or
    scatter-bound. The cost is n_blocks × B × (2y·2x) log(2y·2x).
- **No slack in the extent itself.** The operator already runs on the mask's bounding box
  (`mask_2d.py:746-776`), not on `real_space_shape` (800² at alma+). The (2y, 2x) pad is the
  minimum for a linear (non-circular) correlation of pixel offsets. Two places are left:
  - **(a) Pruned transform (library).** Only the top-left y × x quadrant of each padded input
    is non-zero, and only the top-left y × x quadrant of the output is kept (`:1150`). A
    separable transform can skip half the rows in the first forward pass and half in the last
    inverse pass: rfft along x on y rows, pad y, fft along y, multiply, ifft along y and keep y
    rows, irfft along x. That is up to ~25 % of the FFT work. The bound is ≤ ~25 % of the FFT
    share of F (71–85 %), so ≤ ~15–20 % of F, which is ~90 ms of the 563 ms jvla call and
    ~3 ms at alma. Whether separate 1-D cuFFT calls beat one fused 2-D cuFFT is unmeasured; the
    prompt measures it before any library change.
  - **(b) Real-space pixel scale / mask radius (workspace / science choice).** M is set by the
    mask radius over the real-space pixel scale. At jvla 0.01″ gives 700². A 2× coarser grid
    (0.02″) gives 350², ~4× smaller M, and would put jvla F near the alma_high row (~67 ms,
    against 510 ms). Whether the reconstruction and evidence tolerate it is a science question
    this note does not answer. The mask-radius half (2.0 / 3.5 / 5.0″) is already in task
    3/3's decision-matrix witness on CPU; the prompt adds the GPU pixel-scale half.
- **Predicted gain:** (a) ≤ ~15–20 % of F at every instrument, if pruning wins on cuFFT at all.
  (b) Up to ~4× on F per 2× in pixel scale, bounded by the science tolerance.

### 3. Fixed-mapper `preloads.curvature_matrix` across calls

- **Step attacked:** F, skipped outright. `InversionInterferometerSparse.curvature_matrix`
  returns `preloads.curvature_matrix` directly when it is set (`sparse.py:162-163`); the dense
  mapping path has no such hook.
- **What exists.** The datacube case is already shipped. `AnalysisInterferometer(shared_preloads=True)`
  builds F once per evaluation on the lead factor and shares it with every channel via
  `shared_state_from` → `aa.PreloadsInterferometer` (PyAutoLens
  `interferometer/model/analysis.py:185-227`). That amortises F **across channels within one
  call**, not across calls.
- **What does not exist.** Reuse **across calls** when the mapper does not change: a
  source-only search with mass fixed, a fixed image mesh and only regularization parameters
  free (for example a SLaM-style source-pix stage that fixes the mass). There F and D are
  constant for the whole search, and only H, the solve and the log-dets change.
- **Structural bound:** the F row, i.e. 36–42 % at alma, 66–73 % alma_high, 91–92 % jvla. At
  jvla Delaunay that would take the call from 563 ms to ~53 ms (F 510 ms removed). This is a
  bound, not a measurement. Which production stages actually have a fixed mapper is
  unverified, and so is whether the image mesh / adapt image is constant across such a search.
  The prompt measures both first.

### Not levers (evaluated, not filed)

- **F block size / `fori_loop` (prompt item 1).** The block-size sweep is flat within noise,
  and the kernel split shows F FFT-bound (lever 2). The block width is not a lever.
- **fp32 FFT arm (prompt item 5).** This was a script copy of `curvature_matrix_diag_from` in
  float32, for measurement only. It is **slower on the A100 in every leg**: alma 34.3 vs 17.8 ms
  Delaunay and 97 vs 18.7 ms rect; jvla 3669 vs 510 ms and 7146 vs 517 ms. It also **fails the
  0.5-nat bar at jvla** (Δ 9.3 / 8.5 nats). alma_high (0.028–0.030), alma (1e-4) and sma (4e-8)
  hold. On CPU it is 1.33–1.79× faster with Δ 1e-4 nats (alma). The A100's fp64 FFT is not a
  bottleneck an fp32 cast fixes. The script's fp32 loop probably loses the library's fusion;
  that was not investigated. `use_mixed_precision` does not reach F anyway.
- **Func-list off-diagonal assembly (prompt item 3).** Not exercised: these inversions are
  mapper-only, and the single-mapper branch bypasses block assembly and mirroring
  (`sparse.py:165-168`). The HLO census shows one F build and no duplicate FFTs. Static read:
  `_curvature_matrix_func_list_and_mapper` (`sparse.py:317-419`) recomputes the mapper triplets
  per mapper in both the diagonal and the off-diagonal loops. Triplets cost 0.12–0.21 ms here,
  so this is cheap. No measured lever for this model; revisit with a mapper + MGE model.
- **Mixed precision.** No time change and it does not reach F (above).

## Follow-ups

Drafted as PyAutoMind prompts / notes (`Status: draft`, epic `interferometer-likelihood-campaign`)
and handed to the main session for filing. None is implemented here.

| Lever | Draft | Target |
|---|---|---|
| 1 — certified solver on the interferometer sparse path | amendment to `draft/feature/autofit/certified_solver_batched_guard_c2.md`: add alma Delaunay / rect sparse cells to C2's matched table | PyAutoFit / PyAutoArray (C2) |
| 2 — F FFT size (pruned transform; real-space pixel scale on GPU) | `interferometer_w_tilde_fft_size_levers` | autolens_profiling → PyAutoArray |
| 3 — fixed-mapper F preload across calls | `interferometer_fixed_mapper_curvature_preload` | autolens_profiling → PyAutoLens / PyAutoArray |

Not filed: block size, fp32 FFT arm, func-list assembly and mp (reasons above). The
~1e-6 to 1e-5 nat GPU Delaunay spread needs no follow-up (caveat above). The anesthetic floor
drift is already in `ral_venv_dependency_floor_drift`. A fresh-cache re-run of one leg is
optional; it was not done, and the rows stand on the autotune-level-0 argument.
