# DRAFT — phase-C data for the interferometer mesh A100 note (#320). Delete after folding in.

Branch `feature/interferometer-mesh-breakdown-a100`. Commits: 66e45e9 (cells + CPU sma), e07b962 (CPU alma),
e474025 (A100 results + READMEs). Every number below is read from the committed JSONs under
`results/breakdown/interferometer/` (alma in the dir itself, others in `sma/`, `alma_high/`, `jvla/`, `n_sweep/`).
Harness: `scripts/misc/likelihood_breakdown/interferometer_pixelized.py`.

## Provenance / caveats

- A100 = RAL `euclid-ral-gpu-{1,2}`, NVIDIA A100 80 GB PCIe, jobs 356370-356387, all COMPLETED; every
  `.err` has 0 Tracebacks and 0 `truncated to dtype float32`; the cell refuses to run with x64 off.
- Shared mirror refreshed via `HPCPullPyAuto` 2026-09-26; mirror HEADs == local canonical mains:
  Nerves 2b3bc533, Fit cf83504e, Array 14d63360 (#575 MGE W~ route + #577 real scatter), Galaxy 0e4b89cf,
  Lens 4487eb47. autolens_profiling 66e45e90 on RAL. Recorded in every JSON's `source_revisions`.
- RAL venv vs floors: nufftax 0.6.1 (floor >=0.6.1 OK), jax/jaxlib 0.10.2, jaxnnls 1.0.1, scipy 1.17.1,
  psutil 6.1.0, dynesty 2.1.5, tfp-nightly OK. **Drift: anesthetic 2.8.14 < PyAutoFit floor >=2.9.0**
  (not on the likelihood path). getdist / zeus-mcmc absent (optional extras). Venv not modified. Full freeze:
  RAL `/mnt/ral/jnightin/autolens_profiling_wt/interferometer-mesh-breakdown-a100-freeze.txt`.
- JAX compilation cache NOT fresh on the A100 legs (`autotune_cache_entries_at_start = 204`, cache
  `/mnt/ral/jnightin/.cache/pyauto_jax`); the submits did not set a per-job cache. XLA_FLAGS carried
  `--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` (autonerves wrapper).
- Model: Isothermal + ExternalShear at truth (tight priors), no lens light (datasets carry none).
  Delaunay: Hilbert-1500 + AdaptSplit(0.1, 10, 0.1); rectangular: RectangularBilinearAdaptImage 39x39 +
  Constant(1.0) (edge-zeroed: 1369 of 1521 pixels solved). TransformerNUFFT, sparse operator
  `method="nufft"`, batch_size 128.
- CPU = shared laptop, one visible core, load avg ~5: compare rows within one JSON only.

## Witness

- Prompt witness: `delaunay_hpc_a100_fp64.json` (alma) step sum **50.73 ms** vs full JIT **49.51 ms**
  (ratio 1.025, within 10 %); step names are the sparse steps, no transformed-mapping-matrix row. PASS.
- Step-reproduction fidelity (standalone steps vs library `FitInterferometer.figure_of_merit`): 0 (sma),
  1.5e-8 (alma), 7.6e-7 (alma_high), 1.4e-5 nats (jvla, |logL| = 3e8).
- Sparse vs dense (fp64): CPU sma 0.0 nats both meshes (F rel diff 5e-14, D 3e-15); A100 sma chunked-dense
  vs sparse 0.0 both meshes. **Finding:** A100 sma library dense one-shot FitInterferometer vs library sparse
  = 5.2e-6 nats (Delaunay; rect 0.0), and the A100 sma Delaunay figure of merit sits 2.3e-5 nats from the CPU
  value (-3162.627213 vs -3162.627236); certified vs PDIP full pipeline differ 1.8e-6. Structural
  sparse = dense holds; the ~1e-6-1e-5 spread is GPU PDIP / reduction round-off on the Delaunay AdaptSplit
  system (rect is 0.0). Worth a one-line check in phase C, not a blocker.
- CPU sma peak RSS: 1.71 GB Delaunay, 1.65 GB rect (target < 4 GB; old cell OOM at 14.6 GB). CPU alma
  2.75 / 1.93 GB.
- HLO census of the fused pipeline: 3 fft ops, 4 (Delaunay) / 5 (rect) while loops — one F build, no
  duplicated FFT work after XLA CSE.

## A100 fp64 baseline (ms per call; library path, PDIP)

| Instrument | N_vis | mesh | full JIT | step sum | mapper | F (W~) | solve | log-dets | other | F share | solve share |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sma | 190 | Delaunay | 32.61 | 33.91 | 5.51 | 4.29 | 20.57 | 2.08 | 1.46 | 13 % | 63 % |
| sma | 190 | rect | 28.08 | 29.20 | 0.91 | 4.66 | 20.33 | 2.15 | 1.15 | 17 % | 72 % |
| alma | 1M | Delaunay | 49.51 | 50.73 | 5.78 | 17.80 | 23.46 | 2.12 | 1.57 | 36 % | 47 % |
| alma | 1M | rect | 44.46 | 45.46 | 0.95 | 18.71 | 21.99 | 2.25 | 1.54 | 42 % | 49 % |
| alma_high | 5M | Delaunay | 101.45 | 100.00 | 6.63 | 67.00 | 21.94 | 2.07 | 2.36 | 66 % | 22 % |
| alma_high | 5M | rect | 95.88 | 97.69 | 1.37 | 69.84 | 22.05 | 2.20 | 2.23 | 73 % | 23 % |
| jvla | 25M | Delaunay | 563.31 | 551.62 | 11.86 | 510.45 | 18.90 | 2.28 | 8.13 | 91 % | 3 % |
| jvla | 25M | rect | 564.36 | 556.96 | 3.77 | 517.41 | 23.46 | 2.16 | 10.16 | 92 % | 4 % |

"other" = ray-trace + L + H + triplets + D + chi-squared (each < 3.3 ms, D grows 0.2 -> 3.3 ms with M_pix).
F cost is set by the mask extent M = y*x (4900 / 19600 / 78400 / 490000 at sma / alma / alma_high / jvla),
not N_vis: the FFT grid is (2y, 2x). ConstantSplit bridge (alma Delaunay): full 47.82 ms, steps 49.66 ms
(`delaunay_hpc_a100_fp64_constant_split.json`), vs AdaptSplit 49.51 ms.

Operator build (one-off, `operator_build_s`): 2.8 / 3.2 / 4.3 / 7.4 s (sma / alma / alma_high / jvla).

### Dense (mapping) arm

| | sma Delaunay | sma rect |
|---|---:|---:|
| dense step sum (chunked T, 50 cols) | 67.13 ms | 63.00 ms |
| of which T (NUFFT) | 37.66 | 38.19 |
| dense F / D | 0.34 / 0.14 | 0.34 / 0.13 |
| library dense one-shot FitInterferometer (full) | 65.57 ms | 61.24 ms |
| sparse full JIT | 32.61 | 28.08 |

alma dense arm: **OOM** on the A100 (requested 27.6 GB Delaunay / 25.4 GB rect in the chunked transform;
T alone is 22.4 GiB complex128); alma_high/jvla skipped (T 112 / 559 GiB). CPU alma skipped (22 GiB).

### vmap (sparse full pipeline, per-call amortised)

sma: b64 15.3 ms (Del) / 10.1 ms (rect); alma: b64 29.4 / 24.1 ms; alma_high: b16 86.6 / 83.4 ms;
jvla: b16 **OOM** (requests 74 / 75 GB), b4 563.3 / 558.6 ms (no amortisation: F is memory/FFT-bound).

### CPU (laptop, one core, loaded) for reference

sma: Delaunay step sum 1.64-1.74 s vs full 1.55-2.50 s (noisy); rect 1.42 s vs 1.71 s. alma: Delaunay
4.18 s vs 4.25 s, rect 3.69 vs 4.22 s; F 2.71 / 2.56 s (65-70 %), PDIP 1.08 / 0.74 s.

## N sweep at alma (A100 fp64, full JIT ms; PDIP vs certified full pipeline)

| mesh | N | full PDIP | full certified | F | solve PDIP | solve cert (passes) | vmap b16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Delaunay | 1000 | 32.72 | 21.37 | 12.14 | 15.07 | 3.72 (2) | 21.9 |
| Delaunay | 1500 | 49.51 | 33.11 | 17.80 | 23.46 | 6.80 (3) | 29.4 (b64) |
| Delaunay | 2500 | 84.73 | 54.39 | 29.37 | 42.09 | 11.56 (3) | 76.4 |
| Delaunay | 4000 | 142.55 | 91.54 | 47.43 | 74.34 | 24.25 (4) | 179.2 |
| rect | 1024 (32²) | 30.01 | 24.19 | 12.83 | 14.82 | 8.88 (9) | 18.6 |
| rect | 1521 (39²) | 44.46 | 39.65 | 18.71 | 21.99 | 17.34 (12) | 24.1 (b64) |
| rect | 2500 (50²) | 71.24 | 66.22 | 31.07 | 35.37 | 30.51 (13) | 60.2 |
| rect | 4096 (64²) | 119.20 | 109.96 | 48.85 | 62.73 | 53.38 (12) | 145.3 |

F scales ~linearly in N (blocks = ceil(N/128)); PDIP ~N^1.2; vmap stops amortising above N~2500.

## mp vs fp64 (log-evidence shift, in-run fp64 reference; bar 0.5 nats)

| inst | Delaunay Δ | rect Δ | time change |
|---|---:|---:|---|
| sma | 1.7e-6 | 1.1e-9 | none |
| alma | 2.8e-5 | 1.2e-5 | none (49.33 vs 49.51; 44.02 vs 44.46 ms) |
| alma_high | 3.7e-4 | 8.1e-4 | none (100.26 vs 101.45; 94.29 vs 95.88) |
| jvla | 1.3e-2 | 4.4e-3 | 542.7 vs 563.3; 550.1 vs 564.4 (~3 %, within noise) |

All hold the bar; mp buys nothing because `use_mixed_precision` reaches only the mapper / mapping matrix —
`InterferometerSparseOperator` hard-casts the W~ path (F) to float64.

## Solver share (lever 2)

The interferometer sparse path **honours** `Settings(positive_only_solver="certified")` (mapper-only JAX
inversion; `positive_only_solver_used == "certified"` confirmed at trace time). PDIP share of the full call:
63-72 % (sma), 47-49 % (alma), 22-23 % (alma_high), 3-4 % (jvla). Standalone certified / PDIP on the same
system: Delaunay 0.14-0.33x (passes 0-4), rect 0.27-0.87x (passes 3-13, the edge-zeroed rect system needs
more passes); certified-vs-PDIP figure of merit ≤ 1.2e-7 nats (0.0 mostly). Full pipeline with certified:
alma Delaunay 49.5 -> 33.1 ms (-33 %), N=4000 142.6 -> 91.5 ms (-36 %); alma rect 44.5 -> 39.7 ms (-11 %);
alma_high Delaunay 101.5 -> 85.0; jvla 563 -> 527 ms. sma Delaunay 32.6 -> 14.7 ms.

## Lever evidence (the five in the prompt)

1. **F block size / fori_loop.** Block-size sweep flat within noise at every instrument: alma Delaunay
   B=32/64/128/256/512 -> 17.98/19.32/17.80/17.99/18.45 ms; jvla 64/128/256 -> 534/510/518 ms. Kernel split
   (one block jitted per part x n_blocks): FFT apply 71-85 % of predicted F, gather+segment_sum 14-23 %,
   scatter 1-8 % (sma: FFT 55 %, scatter 17 %, gather 27 %). Predicted (unfused) exceeds measured fused F by
   5-40 %. Verdict: F is **FFT-bound** on the (B, 2y, 2x) rfft2/irfft2 pair, not launch- or scatter-bound; the
   block width is not a lever. The FFT size is set by the mask extent — a tighter extent / smaller real-space
   grid (e.g. real_space_shape or mask shape) is the lever with a bound: F ∝ M log M.
2. **PDIP vs certified.** See above: certified is the largest available cut at sma/alma (-33 % Delaunay
   full call at alma, -36 % at N=4000), small for rect, negligible at jvla. Default flip is the certified
   solver phase-C policy decision (opt-in today).
3. **Func-list off-diagonal assembly.** Not exercised: these inversions are mapper-only, and the library's
   single-mapper branch bypasses block assembly and mirroring (`sparse.py` `curvature_matrix`). HLO census:
   3 fft ops and 4-5 while loops in the fused pipeline, the
   same FFT count as one standalone F plus its constant Khat transform: one F build, no duplicate FFTs. Static read: in `_curvature_matrix_func_list_and_mapper` the mapper triplets are
   recomputed per mapper in both the diag and the off-diag loops (triplets are 0.12-0.21 ms here, so cheap).
   No measured lever for this model.
4. **preloads.curvature_matrix.** Honoured by the sparse path (`sparse.py:curvature_matrix` returns it
   directly). With mass free, F changes every call, so it applies only to fixed-mass work (datacube channels,
   fixed-lens source-only fits). Bound = the F row: 36-42 % at alma, 66-73 % alma_high, 91-92 % jvla.
5. **fp32 FFT arm** (script copy of `curvature_matrix_diag_from` in float32, measurement only): **slower**
   on the A100 in every leg (alma 34.3 vs 17.8 ms Delaunay, 97 vs 18.7 rect; jvla 3669 vs 510, 7146 vs 517 ms)
   and fails the 0.5-nat bar at jvla (Δ 9.3 / 8.5 nats; alma_high 0.028-0.030, alma 1e-4, sma 4e-8 hold).
   CPU: 1.33-1.79x faster, Δ 1e-4 nats (alma). Verdict: no GPU lever; the A100's fp64 FFT is not the
   bottleneck an fp32 cast fixes (and the script's fp32 loop likely loses the library's fusion — not
   investigated). `use_mixed_precision` does not reach F anyway.

## Resume (tomorrow)

- All 18 jobs 356370-356387 finished and their JSONs are committed in e474025 (the two rectangular jvla
  jobs 356384/356385 completed at 21:53/21:54 before the pull, so nothing is outstanding).
  If a re-pull is ever needed:
  `cd ~/Code/PyAutoLabs-wt/interferometer-mesh-breakdown-a100/autolens_profiling && rsync -a -e "ssh -o IdentitiesOnly=yes" --include='*/' --include='delaunay_hpc_a100*' --include='pixelization_hpc_a100*' --exclude='*' euclid_jump:/mnt/ral/jnightin/autolens_profiling_wt/interferometer-mesh-breakdown-a100/results/breakdown/interferometer/ results/breakdown/interferometer/`
  Logs: RAL `.../interferometer-mesh-breakdown-a100/hpc/batch_gpu/{output/output,error/error}.3563NN.{out,err}`.
- RAL worktree `/mnt/ral/jnightin/autolens_profiling_wt/interferometer-mesh-breakdown-a100` (branch at
  66e45e9, from the bundle `/mnt/ral/jnightin/autolens_profiling_wt/interferometer-mesh-breakdown-a100.bundle`).
- Remaining for phase C: write `results/notes/interferometer_mesh_a100_breakdown_2026_09.md` from this file
  (baseline table = interferometer analogue of #241's 67.5 / 60 ms: alma **49.5 / 44.5 ms**), rank levers
  (certified solver first; F extent/FFT-size second; preload for fixed-mass third; block size, fp32, func-list
  no lever), file follow-up prompts, decide whether the 5e-6-nat GPU sma dense-vs-sparse spread needs a
  check, optionally re-run one leg with a fresh JAX cache, delete this draft, ship via ship_workspace.
