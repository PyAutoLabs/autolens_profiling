# Interferometer MGE likelihood breakdown — JAX CPU and A100, ranked levers (2026-09)

autolens_profiling issue [#308](https://github.com/PyAutoLabs/autolens_profiling/issues/308),
branch `feature/interferometer-mge-breakdown`, epic `interferometer-likelihood-campaign` (1/3).
Cell: `scripts/interferometer/likelihood_breakdown/mge.py` (steps, `setup_split`, `jit_phases`,
`vmap_batch` — the imaging MGE schema) plus a measurement-only func-list W~ arm (`--w-tilde`).
PyAutoLens 2026.8.17.1, fp64 on every leg, nufftax 0.6.1 on the A100.

**Witness:** `results/breakdown/interferometer/mge_hpc_a100_fp64.json` (alma, A100 fp64,
non-null `steps`) and this note. Every number below is read from the committed JSONs:

| Leg | JSON |
|---|---|
| CPU sma NUFFT / DFT | `results/breakdown/interferometer/mge_breakdown_sma_v2026.8.17.1.json`, `mge_dft_breakdown_sma_v2026.8.17.1.json` |
| CPU alma / alma_high | `mge_breakdown_alma_v2026.8.17.1.json`, `mge_breakdown_alma_high_v2026.8.17.1.json` |
| A100 sma | `interferometer/sma/mge_hpc_a100_fp64.json` |
| A100 alma | `interferometer/mge_hpc_a100_fp64.json` |
| A100 alma_high (fp64, mp) | `interferometer/alma_high/mge_hpc_a100_{fp64,mp}.json` |
| A100 jvla (fp64, mp) | `interferometer/jvla/mge_hpc_a100_{fp64,mp}.json` |
| A100 probes (nufftax dispatch, step-3 scatter) | `interferometer/mge_a100_diagnostics_probes_2026_09.json` |

CPU = laptop (`DESKTOP-H143S82`), one thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`,
`--xla_cpu_multi_thread_eigen=false`, recorded as pre-existing in `thread_env`). A100 =
`euclid-ral-gpu-1`, NVIDIA A100 80 GB PCIe, RAL jobs 351078-351083.

## Scope — read this before quoting a number

- **Model.** Isothermal + ExternalShear at the simulator truth (tight Gaussian priors, so
  the prior median is the truth and the ray-trace prefix is not constant-folded), source =
  MGE-20 via `al.model_util.mge_model_from(total_gaussians=20)`, 20 linear Gaussians, PDIP
  NNLS (`preconditioning=raw`).
- **No lens light.** `simulators/interferometer.py` puts no lens emission in the
  visibilities, so a lens-light MGE would fit nothing; the only linear objects are the 20
  source Gaussians. Consequence: `log_det_terms` is structurally zero (no regularization on
  an MGE-only inversion) and is recorded, not timed.
- **Above sma the A100 steps are not the library path.** A single
  `jax.jit(FitInterferometer)` OOMs the A100 at alma and above (`library_path.status =
  oom`, see lever 2), so steps 3-8 there run on the cell's measurement-only
  **chunked-transform arm** (`transform = script_chunked`, 4 columns per batch, 1M-visibility
  chunks at alma_high/jvla), which agrees with the laptop fit to 1.5e-8 nats (alma) and
  7.6e-7 nats (alma_high). sma on the A100 and every CPU leg are the library path.
- **The W~ rows are measurement-only.** No MGE-only fit takes that path in the library today
  (lever 1).
- **nufftax 0.6.1 only.** The first RAL round (jobs 351055-351057) ran on the RAL venv's
  nufftax 0.4.0, below the PyAutoArray floor (`>=0.6.1,<0.7`). 0.4.0 auto-routes GPU type-1
  spreading at M >= 10,000 and type-2 interpolation at M >= 1,000,000 visibilities to
  float32 Pallas/Triton kernels even for x64 inputs: type-2 rel err 1.05e-4 and type-1
  6.85e-5 vs 2.9e-13 / 1.9e-12 on 0.6.1 (probe JSON, jobs 351062/351063). It put the W~
  figure of merit 2.7e-3 nats (alma) / 0.25 nats (alma_high) off the Fit. **Those JSONs are
  invalid and not committed**; the venv was upgraded to 0.6.1 on 2026-09-25 and every A100
  number here is from the 0.6.1 re-run.

## Headline — per-call seconds, dense chain vs W~ chain

`dense` = the full step-by-step likelihood (`total_step_by_step`); `dense 3-8` = the fused
dense chain from the mapping matrix (`steps_sub_rows`, the like-for-like comparator of the W~
chain); `W~ 3-8` = `w_tilde.chain_from_mapping_matrix_s`.

| Instrument | N_vis | Device | dense total | dense 3-8 | W~ 3-8 | W~ speed-up | W~ operator build (one-off) |
|---|---:|---|---:|---:|---:|---:|---:|
| sma | 190 | CPU | 84.0 ms | 79.1 ms | 6.32 ms | 12.5x | 0.96 s |
| sma (DFT) | 190 | CPU | 32.9 ms | 28.6 ms | 9.97 ms | 2.9x | 1.89 s |
| sma | 190 | A100 | 856.2 ms | 855.0 ms | 2.40 ms | 356x | 1.07 s |
| alma | 1M | CPU | 41.19 s | 36.58 s | 28.6 ms | 1280x | 18.54 s |
| alma | 1M | A100 | 943.2 ms¹ | 937.4 ms¹ | 1.90 ms | 494x | 3.00 s |
| alma_high | 5M | CPU | 212.19 s | 114.56 s | — ² | — | — |
| alma_high | 5M | A100 | 3.88 s¹ | 3.81 s¹ | 3.35 ms | 1137x | 3.70 s |
| jvla | 25M | A100 | 23.81 s¹ | 23.62 s¹ | 13.53 ms | 1746x | 8.24 s |

¹ chunked-transform arm (the library path OOMs). ² the CPU alma_high leg ran without the W~
arm. CPU library full-pipeline single-jit: sma 86.9 ms, DFT sma 61.6 ms, alma 41.08 s,
alma_high 144.57 s.

W~ correctness: F~ vs dense F max rel diff 2.6e-14 (CPU sma), 8.0e-14 (CPU alma), 2.7e-14
(A100 sma), 3.3e-13 (A100 alma), 2.8e-12 (A100 alma_high), 1.6e-11 (A100 jvla); W~ figure of
merit off the Fit by 1.4e-12 (sma), 2.0e-8 (CPU alma), 1.7e-8 (A100 alma), 7.1e-7 nats
(A100 alma_high). jvla has no Fit reference (library OOM, CPU leg not run).

Where the time goes (dense, A100): step 3 `transform_mapping_matrix` is 99.6 % at sma
(852.7 of 856.2 ms), 97.0 % at alma (914.8 of 943.2 ms), 96.6 % at alma_high, 97.3 % at jvla;
curvature F is the only other visible row (25.3 ms alma, 124.8 ms alma_high, 621.4 ms jvla).
On CPU step 3 is 97.6 % at sma and 98.8 % at alma (40.71 of 41.19 s).

## Ranked levers

### 1. W~ route for MGE-only fits — the lever that makes MGE usable at alma+

- **Step attacked:** 3 (transform) + 4 (D) + 5 (F), i.e. the whole O(N_vis) part.
- **Why it is off:** PyAutoArray `inversion/inversion/factory.py:202-208` sets
  `use_sparse_operator = False` when every linear object is an
  `AbstractLinearObjFuncList`, so an MGE-only fit takes the dense NUFFT path even after
  `apply_sparse_operator()`. The func-list W~ blocks already exist
  (`inversion_interferometer_util.py:1466` `operated_matrix_slim_from`, `:1628`
  `curvature_matrix_func_list_from`) and are used by mixed mapper + MGE inversions.
- **Structural bound:** F~ = Bᵀ W~ B costs O(n · M log M) on the padded real-space grid and
  never touches the visibilities; D~ comes from the cached dirty image; the chi-squared uses
  the `dᵀN⁻¹d - 2sᵀD + sᵀFs` identity with constants hoisted. N_vis-independent — it grows
  only with the mask (operator 70² sma, 140² alma, 280² alma_high, 700² jvla: 1.90 → 3.35 →
  13.53 ms on the A100). The one-off operator build is two eager type-1 NUFFTs (1.07-8.24 s
  A100, 18.5 s CPU alma).
- **Measured gain:** dense 3-8 → W~ 3-8 is 356x (sma), 494x (alma), 1137x (alma_high), 1746x
  (jvla) on the A100; 12.5x (sma) and 1280x (alma, 36.58 s → 28.6 ms) on CPU. With the W~
  route an alma MGE likelihood on the A100 is ~2.3 ms of steps (`w_tilde.total_step_by_step`
  2.32 ms) plus ray-trace + mapping matrix (0.9 ms) — and it runs at every instrument,
  including the ones the library path OOMs.
- **Caveat:** small data. At sma the W~ gain on CPU is 12.5x NUFFT / 2.9x DFT; the operator
  build (~1-2 s) is amortised over a search, not a single call.

### 2. Chunked `transform_mapping_matrix` — makes the dense path run on the A100 at all

- **Step attacked:** 3. `TransformerNUFFT.transform_mapping_matrix`
  (`operators/transformer.py:505`) is one `nufft2d2` over every column and every
  visibility and ignores `chunk_size` (recorded in each JSON's `chunk_size_note`).
- **Measured block:** `library_path.requested_bytes` = 65,906,273,157 B (65.9 GB, "61.38
  GiB") at alma, 322,433,932,328 B (322 GB) at alma_high, 1,605,286,976,552 B (1.61 TB) at
  jvla — against the 80 GB device. On CPU the same one-shot transform fits under XLA fusion
  (the cell docstring records compiled temp 2.1 GB alma / 3.1 GB alma_high; peak RSS 4.09 GB
  / 11.66 GB), so this is a GPU-only failure.
- **Structural bound:** peak memory O(column_batch × vis_chunk × kernel width) instead of
  O(n × N_vis × kernel width); time stays O(n × N_vis) — the chunked arm's step 3 is 0.915 s
  alma, 3.75 s alma_high, 23.17 s jvla.
- **Predicted gain:** OOM → runs (the numbers in the headline table); needed for any dense
  MGE fit above sma on GPU and for the mixed mapper + MGE path, which lever 1 does not cover.
  It does not beat lever 1 on speed at any instrument.

### 3. complex128 slim→native scatter in step 3 — a fixed ~0.85 s GPU cost

- **Step attacked:** 3 (GPU). `transform_mapping_matrix` casts the mapping matrix to
  complex128 and scatters it into a `(20, N_y, N_x)` complex128 image stack
  (`transformer.py:525-527`) before the NUFFT.
- **Measured (probe JSON, jobs 351070/351077/351084):** complex128 scatter of 20 columns
  3422 ms, one column 224 ms; float64 scatter then cast 0.57 ms (~6000x faster);
  `unique_indices`/`indices_are_sorted` do not help (3422 ms). A bare `nufft2d2` of 20
  columns on 800×800 is 6.0 ms. sma step 3 on the A100 is 852.7 ms vs 82.0 ms on the laptop
  CPU, the same on nufftax 0.4.0 and 0.6.1 (K=190 is below every Pallas threshold).
- **Structural bound:** N_vis-independent; sets a ~0.85-3.5 s floor on step 3 at every
  instrument depending on surrounding fusion (A100 sma 0.853 s and alma 0.915 s are nearly
  equal despite 5000x more visibilities).
- **Predicted gain:** at sma it is essentially the whole A100 likelihood — removing it takes
  sma from ~856 ms toward the ~4 ms of the other steps, and flips the sma GPU-vs-CPU verdict.
  At alma+ it is bounded by the scatter's share of step 3 on the chunked arm, which is not
  separately measured; the follow-up must measure it. Cheap, local, library-only fix
  (scatter the real mapping matrix, then cast). Irrelevant once lever 1 lands for MGE-only
  fits, still relevant for mixed mapper + MGE and pixelized dense paths that share the method.

### 4. GPU vs CPU verdict per instrument

| Instrument | CPU total | A100 total | A100 / CPU speed | Note |
|---|---:|---:|---:|---|
| sma | 84.0 ms | 856.2 ms | **0.098x (GPU ~10x slower)** | step 3 scatter (lever 3); CPU DFT is 32.9 ms |
| alma | 41.19 s | 943.2 ms¹ | **43.7x** | 43.6x vs the CPU single-jit 41.08 s |
| alma_high | 212.19 s | 3.88 s¹ | **54.7x** | 37.3x vs the CPU single-jit 144.57 s |
| jvla | — | 23.81 s¹ | — | CPU leg not run; A100 library path OOMs |
| W~ sma | 6.32 ms | 2.40 ms | 2.6x | |
| W~ alma | 28.6 ms | 1.90 ms | 15.1x | |

¹ chunked-transform arm. Verdict: on the dense path use the CPU at sma (DFT is fastest,
32.9 ms) and the GPU at alma+ only with a chunked transform; on the W~ route the GPU wins at
every instrument measured, and both devices are ms-scale. (The May runtime cell's
"A100 711 ms vs CPU 231 ms at sma" is consistent with the same step-3 floor; not re-measured here.)

### 5. Mixed precision — no gain

| Instrument | fp64 total | mp total | fp64 W~ 3-8 | mp W~ 3-8 |
|---|---:|---:|---:|---:|
| alma_high | 3.879 s | 3.929 s (+1.3 %) | 3.349 ms | 3.368 ms |
| jvla | 23.812 s | 23.715 s (-0.4 %) | 13.526 ms | 13.518 ms |

Step 3 (complex128 scatter + NUFFT) and F dominate and mp does not touch either. Drop the mp
arm for interferometer MGE; not filed.

### 6. Lower-ranked (not filed as their own prompts)

- **NUFFT `eps` / per-Gaussian vs batched NUFFT.** Not separately measured here; the
  library already batches all columns in one call, and the transform cost is scatter-bound on
  GPU (lever 3) and made moot by lever 1. Revisit only if a dense MGE path survives.
- **Shared eccentric-radius evaluation on the JAX path** (NumPy shares it,
  `linear/abstract.py:397-470`). The mapping-matrix step is 0.19-2.5 ms on the A100 and
  0.7 ms (sma) / 2.0 ms (alma) / 29.9 ms (alma_high) on CPU — below 1 % of the dense chain.
  It becomes a visible share only after lever 1 (alma A100: 0.38 ms next to a 1.9 ms W~
  chain). Fold into a lever-1 follow-up if it shows.
- **PDIP at n=20.** 9-17 iterations, always converged; 1.5-2.6 ms on the A100, 0.2-0.35 ms
  on CPU. The certified active-set solver is excluded by design for any inversion containing
  an `AbstractLinearObjFuncList` (`inversion/inversion/abstract.py:580-583`, guard at
  `:598-599`), and an MGE-only inversion has no `Mapper` either. After lever 1, PDIP is the
  largest W~ step on the A100 (1.46 of 2.32 ms at alma), so a small-n dense solve
  (e.g. an exact active-set / fnnls-style solve for n=20) is the natural phase after lever 1.

## Other findings

- **vmap batch 4 amortises the step-3 floor on the A100:** `setup_split_vmap` transform
  per call 0.225 s (sma, vs 0.853 s unbatched) and 0.344 s (alma, vs 0.915 s).
- **VRAM config.** `scripts/misc/vram/config.py` marked MGE at alma+ "inherently blocked"
  (62 GB gather buffer, measured before chunking existed). Re-tested: the block is the
  unchunked library transform on GPU, not inherent — the chunked transform and the W~ route
  both run at every instrument. The `None` rows stay until lever 1 or 2 lands; the comment
  now says so.
- **Workspace docs mismatch.** `autolens_workspace/scripts/interferometer/features/
  multi_gaussian_expansion/modeling.py:322-324` says that with `apply_sparse_operator()` MGE
  memory depends on the real-space mask alone; the factory takes the dense path for MGE-only
  fits, so memory scales with N_vis (filed as a docs follow-up, not edited here).

## Follow-ups

Drafted as PyAutoMind prompts (`Status: draft`, epic `interferometer-likelihood-campaign`);
not implemented here.

| Slug | Lever | Target |
|---|---|---|
| `interferometer_mge_w_tilde_route_mge_only` | 1 — W~ route for MGE-only fits | PyAutoArray |
| `interferometer_chunked_transform_mapping_matrix` | 2 — chunked `transform_mapping_matrix` | PyAutoArray |
| `interferometer_transform_mapping_matrix_real_scatter` | 3 — float64 scatter then cast | PyAutoArray |
| `ral_venv_dependency_floor_drift` | RAL venv below library floors (anesthetic, dynesty, psutil, tfp-nightly; nufftax found and fixed in phase B) | PyAutoHeart |
| `workspace_interferometer_mge_sparse_operator_memory_docs` | workspace docs mismatch (`modeling.py:322-324`) | autolens_workspace |

Not filed: levers 5 (mp) and 6 (eps / batched NUFFT, shared eccentric radius, PDIP at
n=20 — revisit PDIP after lever 1). The `scripts/misc/test/` failures seen on this branch
before merging `origin/main` (6 tests, PyAutoArray#572 `preconditioning`) are already fixed
on main (912 passed after the merge).

## Library W~ route (2026-09-26)

Lever 1, landed as PyAutoArray#575 (PyAutoArray#576, PyAutoGalaxy#629, PyAutoLens#750 — branch
`feature/interferometer-mge-w-tilde-route`, measured before merge). The inversion factory no longer
turns the sparse operator off for an all-func-list inversion. An MGE-only `FitInterferometer` on a
dataset with `apply_sparse_operator()` applied now takes `InversionInterferometerSparse`. Fits that
also have ordinary light profiles use the dirty image of the profile-subtracted visibilities
(`d~ - W~ i_p`).

**Cell:** the same `mge.py` with `--sparse` (PART C2). It runs the library path in the dense schema:
nested `params -> stage` prefixes, `jit_phases`, the jitted `FitInterferometer.log_likelihood`, and
the `AnalysisInterferometer` full pipeline on the sparse dataset. The dense and `--w-tilde` arms
still run in the same job, so each JSON holds before and after:
- the library rows are in `steps` / `library_sparse`;
- the dense rows are in `dense_steps`.

Libraries were imported from the branch. On RAL that means a private copy at
`/mnt/ral/jnightin/PyAuto_branch/interferometer-mge-w-tilde-route/`, whose `__file__` the submit
asserts; the shared mirror was not touched. SHAs are in each JSON's `source_revisions`. RAL used the
mirror's PyAutoFit `dd9fbe0a` and PyAutoNerves `1fa613aa`, with nufftax 0.6.1.

Two rounds were measured:
- **Round 1:** PyAutoArray `b5ef2e89`, PyAutoGalaxy `068daa9d`, PyAutoLens `acb136cd` (RAL jobs
  356359-356361).
- **Round 2:** the zero-image NUFFT fix below, PyAutoGalaxy `36d1b436`, PyAutoLens `ac333b17`
  (PyAutoArray unchanged). It re-ran CPU sma/alma and A100 alma/jvla (RAL jobs 356362, 356363); the
  committed JSONs for those legs are from round 2. The A100 alma_high JSON is still round 1.

| Leg | JSON |
|---|---|
| CPU sma / alma | `results/breakdown/interferometer/mge_breakdown_{sma,alma}_v2026.8.17.1_sparse.json` |
| A100 alma / alma_high / jvla | `interferometer/mge_hpc_a100_fp64_sparse.json` (356362), `interferometer/alma_high/mge_hpc_a100_fp64_sparse.json` (356360, round 1), `interferometer/jvla/mge_hpc_a100_fp64_sparse.json` (356363) |

### Before / after — per-call seconds, fp64

`library full` is `library_sparse.full_pipeline_single_jit`, i.e. `AnalysisInterferometer` on the
sparse dataset, which is what a sampler calls. `library log L` is the jitted
`FitInterferometer.log_likelihood`, which also forms the mapped visibilities. `dense` is the same
run's dense path: the library path where it runs, otherwise the chunked-transform arm (¹). `W~ arm` is
the #308 measurement-only chain from the mapping matrix.

| Instrument | Device | Inversion (library) | library full, round 1 | library full, round 2 | library log L, round 2 | dense | dense library path | W~ arm 3-8 |
|---|---|---|---:|---:|---:|---:|---|---:|
| sma | CPU | InversionInterferometerSparse | 11.1 ms | 7.77 ms | 16.1 ms | 82.1 ms (full pipeline) | runs | 7.06 ms |
| alma | CPU | InversionInterferometerSparse | 1.29 s | 34.7 ms | 2.86 s | 13.10 s (full pipeline) | runs | 34.7 ms |
| alma | A100 | InversionInterferometerSparse | 8.92 ms | 2.69 ms | 9.04 ms | 939.6 ms¹ | OOM (asks 61.4 GiB) | 1.93 ms |
| alma_high | A100 | InversionInterferometerSparse | 19.6 ms | — (not re-run) | 35.2 ms (round 1) | 3.94 s¹ | OOM (asks 300.3 GiB) | 3.36 ms |
| jvla | A100 | InversionInterferometerSparse | 92.2 ms | 16.7 ms | 93.1 ms | 24.20 s¹ | OOM (asks 1.46 TiB) | 13.6 ms |

¹ Dense step total on the chunked-transform arm, because the dense library path OOMs.

The `dense`, `library log L` and `W~ arm` columns are from the round-2 JSONs, except alma_high.

After round 2, on the A100 the library full pipeline is 349x (alma) and 1449x (jvla) faster than
the chunked dense arm, and within 1.2-1.4x of the measurement-only W~ chain. It runs where the dense
library path cannot run at all. On CPU it is 378x (alma) and 10.6x (sma) faster than the dense full
pipeline. The jitted `FitInterferometer.log_likelihood` stays O(N_vis), because it forms the mapped
visibilities by design; the sampler path (`figure_of_merit`) does not.

CPU caveat: the laptop was shared with other sessions (load average 1.3-6, recorded in
`host_load_avg_*`). The dense alma rows in this run (13.0 s full pipeline, 11.2 s transform) are
about 3.6x faster than #308's (41.1 s, 40.7 s) on the same host; round 2 reproduced 13.1 s.
Compare rows within one JSON, not across runs.

### Witness

- **Inversion class:** `InversionInterferometerSparse` on every leg. The cell raises otherwise.
Witness numbers are identical in rounds 1 and 2.

- **|Δ log L| ≤ 1e-6 nats** against the dense library reference, where it runs (CPU, fp64):
  - sma: log L 0.0, figure of merit 4.5e-13;
  - alma: log L 0.0, figure of merit 1.9e-9;
  - `library_sparse.witness_pass = true` on both.
- **A100**, where the dense reference OOMs, against the laptop fp64 pins:
  - alma: log L 0.0, figure of merit 1.9e-9;
  - alma_high: log L 0.0, figure of merit 5.2e-8.
- **jvla:** no dense reference or pin. The library route agrees with the #308 W~ arm to 3.6e-5 nats
  in log L (1.4e-5 in figure of merit), relative 1.2e-13 at log L = -3.0e8.
- **Curvature and data vector:** library F~ vs dense F max rel diff 8.0e-14 (CPU alma); D~ 1.8e-15.
- **No OOM:** the jitted library fit ran at alma, alma_high and jvla on the A100.
- **MGE + lens Sersic** (`lens_sersic_variant`, the dirty-image correction): sparse matches dense to
  0.0 nats in log L at sma and alma. The control with the correction switched off is 3.8e-3 nats off
  (sma) and 779 nats off (alma), so the check discriminates. Not run on the A100, where the dense
  reference OOMs.

### The zero-image NUFFT — found in round 1, fixed in round 2

In round 1 the library full pipeline was 4-7x slower than the #308 W~ arm chain (alma A100: 8.9 ms
against 1.9 ms; jvla: 92 ms against 13.5 ms). The step that grew was
`Fast chi-squared + figure of merit (FitInterferometer)`: 6.2 ms at alma, 15.3 ms at alma_high and
75.8 ms at jvla on the A100, and 2.37 s at alma on CPU.

That step was O(N_vis). `FitInterferometer.profile_visibilities` NUFFTed the standard-light image
whenever `tracer.has(LightProfile)`, and a linear MGE counts as a `LightProfile`. So every call did a
forward NUFFT of an all-zero image, and `fast_chi_squared`'s `dᵀN⁻¹d` term then ran over the
profile-subtracted visibilities. A laptop probe at alma (jit, 3 repeats) measured
`profile_visibilities` at 3.2 s, `inversion.fast_chi_squared` at 4.8 s and `reconstruction` at 43 ms.

**Fix** (folded into the library PRs: PyAutoGalaxy `36d1b436`, PyAutoLens `ac333b17`): when no
ordinary light profile is present, `profile_visibilities` returns zeros without a transform. This is
the same structural check `sparse_dirty_image_from` makes.

Before / after, figure-of-merit row (`setup_split`) and full pipeline:

| Instrument | Device | FoM row, round 1 | FoM row, round 2 | full pipeline, round 1 | full pipeline, round 2 |
|---|---|---:|---:|---:|---:|
| sma | CPU | 4.07 ms | 2.87 ms | 11.1 ms | 7.77 ms |
| alma | CPU | 2.37 s | ≈0 (−11.6 ms, differencing noise) | 1.29 s | 34.7 ms (37x) |
| alma | A100 | 6.23 ms | 0.13 ms | 8.92 ms | 2.69 ms (3.3x) |
| jvla | A100 | 75.8 ms | 0.32 ms | 92.2 ms | 16.7 ms (5.5x) |

`dᵀN⁻¹d` is still a per-call reduction over the visibilities, but it is cheap next to the NUFFT it
replaced. Hoisting it as a dataset constant is a possible further step, not needed now.

### vmap (A100, sparse full pipeline) and VRAM rows

| Instrument | batch 64 | batch 16 | `VMAP_BATCH_SPARSE` row |
|---|---|---|---|
| alma | 0.38 ms/call (round 2; 0.48 round 1) | — | 64 |
| alma_high | 1.54 ms/call (round 1) | — | 64 |
| jvla | OOM, asks 41.1 GiB in both rounds | 10.8 ms/call (round 2; 15.5 round 1) | 16 |

The jvla batch-64 request did not change with the fix, so the zero-image NUFFT was not what capped
it. The request is consistent with the W~ curvature's FFT convolution of the 20 mapping-matrix
columns on the padded operator grid: operator 700² at jvla, padded 1400² complex128, times 20
columns times 64 replicas is about 37 GiB. That attribution is inferred from sizes, not profiled.

These rows are now in `scripts/misc/vram/config.py` under `VMAP_BATCH_SPARSE`. The dense
`VMAP_BATCH` rows stay `None`, because the dense library path still OOMs.

### Not run

- **CPU alma_high.** The laptop kernel OOM-killed it at 10.8 GB anon RSS inside `apply_sparse_operator()`
  (the eager operator build, with a 1M-visibility NUFFT chunk from the instrument preset), after the dense
  arms had finished. The partial JSON is not committed. The A100 alma_high leg covers the witness.
- **CPU jvla** was not attempted.
- **A100 alma_high** was not re-run in round 2; its JSON is round 1.
