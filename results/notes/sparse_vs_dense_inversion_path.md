# Sparse (w-tilde) vs Dense inversion path — local CPU baselines

**Issue:** [autolens_profiling#44](https://github.com/PyAutoLabs/autolens_profiling/issues/44)
**Branch:** `feature/sparse-vs-dense-profile`
**Status:** Local CPU fp64 baselines complete. A100 submits drafted, not yet
dispatched.

## TL;DR

Sparse (w-tilde) inversion path wins on **all three** HST imaging cells when
measured at the production-relevant full-pipeline single-JIT level — between
**−25% and −48%** wall per likelihood evaluation versus dense, with
bit-identical log-evidence. The per-step breakdown numbers told a contrary
story for pixelization (sparse +14.6% slower) and have to be read carefully:
XLA fusion across steps in the full-pipeline JIT is invisible to the
per-step decomposition.

The recommendation is to **enable `apply_sparse_operator()` in
`autolens_profiling/searches/_setup.py:_build_imaging`** for the existing
A100 search runs (Nautilus + NSS × {mge, pix, delaunay}). Pending A100
confirmation that the local CPU pattern holds at production scale.

## Headline numbers (HST, CPU fp64)

### Full-pipeline single-JIT (the production-relevant cost)

| Cell | Dense per-eval | Sparse per-eval | Δ | log_L agreement |
|---|---:|---:|---:|---|
| MGE | 0.173 s | **0.090 s** | **−48.0%** | exact (27379.388906855238) |
| Pixelization | 9.04 s | **6.73 s** | **−25.5%** | bit-identical to 12 sig figs (28398.444158983…) |
| Delaunay | 7.23 s | **3.92 s** | **−45.8%** | bit-identical to 12 sig figs (29110.920857938…) |

Raw JSONs land under `results/likelihood/imaging/local_cpu_fp64{,_sparse}.json`
(pix, del) and `mge_likelihood_summary_hst_v<v>{,_sparse}.json` (mge).

### Per-step breakdown (informative but misleading)

The per-step breakdown JIT-compiles each pipeline step in isolation. XLA can't
fuse across step boundaries here, so the sum-of-bars overcounts the
production cost — exactly as the [breakdown
README](../../likelihood_breakdown/README.md) warned. Numbers:

| Cell | Dense total | Sparse total | Δ |
|---|---:|---:|---:|
| Pixelization (Rectangular, 1521 src) | 7.79 s | 8.94 s | **+14.6%** (sparse loses) |
| Delaunay (1500 src) | 11.29 s | 7.96 s | **−29.5%** (sparse wins) |
| MGE | 0.178 s | (factory short-circuits — N/A) | — |

The breakdown verdict for pixelization (sparse +14.6%) is **wrong as a guide
to production cost**. Full-pipeline JIT is the source of truth.

### Per-step breakdown — where does the cost live?

For both pix and delaunay on the dense path, ~97% of per-eval cost lives in
three steps that w-tilde targets:

| Step | Pix dense | Del dense |
|---|---:|---:|
| Curvature matrix (F) | 47.2% | 53.4% |
| Inversion setup (mapping matrix build) | 25.2% | 18.3% |
| Regularized reconstruction (Cholesky) | 24.6% | 26.3% |
| **Sum (matrix-construction + solve)** | **97.0%** | **98.0%** |
| Ray-trace, lens-light image, etc. | 3.0% | 2.0% |

For MGE the dominant cost is also in the mapping matrix:

| Step | MGE dense |
|---|---:|
| Blurred mapping matrix (PSF convolution) | 59.7% |
| Mapping matrix | 35.9% |
| Curvature matrix (F) | 2.2% |
| **Sum (matrix-construction)** | **95.6%** |
| Everything else | 4.4% |

## Surprises

1. **MGE shows a 48% sparse win** despite the inversion factory's
   `all-AbstractLinearObjFuncList` short-circuit. Pure-MGE-source models
   should land on `InversionImagingMapping` (dense) even with
   `sparse_operator` attached. The 48% gain is real and reproducible (CPU
   fp64), so either (a) the short-circuit isn't firing as the factory code
   suggests, or (b) attaching `sparse_operator` to the dataset changes
   downstream cost outside the inversion-class choice. **Worth a
   diagnostic look** — instrument `InversionImagingMapping.__init__` or
   `InversionImagingSparse.__init__` to log which class actually gets
   constructed on each `--sparse` run.
2. **Pixelization breakdown disagrees with runtime.** The breakdown says
   sparse +14.6% slower; the runtime says sparse −25.5% faster. Trust the
   runtime — XLA fusion across steps is the difference, exactly as the
   breakdown README documented as a caveat.
3. **A100 search runs were dense by design.** Earlier in this investigation
   I'd assumed `_setup.py:305` (the `apply_sparse_operator(use_jax=True,
   show_progress=False)` line) applied to imaging. It does not — that's
   the interferometer/datacube builder. `_build_imaging` in the same file
   has no `apply_sparse_operator` call. So all the NSS+Nautilus A100
   numbers from PyAutoFit#1303/#1305 work were dense-path numbers. The
   sparse path has *never* been exercised on imaging in this codebase
   before this investigation.

## Recommendation

1. **Enable sparse in `_build_imaging`.** Add
   `dataset = dataset.apply_sparse_operator()` after the `apply_over_sampling`
   chain in `autolens_profiling/searches/_setup.py:_build_imaging`. Gate
   behind a flag or a config knob for safety until A100 confirmation.
2. **Run the 6 A100 submits** at
   `hpc/batch_gpu/submit_runtime_imaging_{mge,pixelization,delaunay}_a100_hst_fp64{,_sparse}`.
   Each does a `--vmap-probe` phase (writes `vmap_probe_<cell>{,_sparse}.json`
   with `peak_memory_in_bytes` per batch + recommended A100 batch size) then
   a `--config-name hpc_a100_fp64` phase (full single-JIT + vmap timing,
   writes `hpc_a100_fp64{,_sparse}.json`). Compare against the existing
   A100 dense numbers committed to `results/searches/` from the
   NSS-vs-Nautilus session.
3. **Investigate the MGE 48% surprise** before declaring sparse-everywhere
   safe. If the factory short-circuit *is* firing as expected, the gain
   comes from somewhere outside the inversion path — possibly the way
   `sparse_operator` is consulted (or not) by downstream code paths.
4. **Defer matrix-free CG + SLQ** (the original plan-B). The sparse path
   already gives uniform wins on imaging, with direct Cholesky log-det
   preserved. Matrix-free CG would add SLQ noise for log-det and would
   only be worth pursuing if a future scenario (e.g. ultra-large source
   meshes on H100) revealed a regression.

## Artifacts in the branch

```
autolens_profiling/results/
├── breakdown/imaging/
│   ├── pixelization_breakdown_hst_v<v>.json         # dense
│   ├── pixelization_breakdown_hst_v<v>_sparse.json
│   ├── pixelization_breakdown_hst_v<v>.png + _sparse.png
│   ├── delaunay_breakdown_hst_v<v>.json             # dense
│   ├── delaunay_breakdown_hst_v<v>_sparse.json
│   ├── delaunay_breakdown_hst_v<v>.png + _sparse.png
│   └── mge_breakdown_hst_v<v>.json                  # dense, sparse N/A
├── runtime/imaging/stdout_captures/
│   ├── pixelization_local_cpu_fp64_{dense,sparse}.stdout
│   ├── delaunay_local_cpu_fp64_{dense,sparse}.stdout
│   └── mge_local_cpu_fp64_{dense,sparse}.stdout
└── likelihood/imaging/
    ├── local_cpu_fp64.json + local_cpu_fp64_sparse.json     # pix + del
    └── mge_likelihood_summary_hst_v<v>{,_sparse}.json
```

A100 outputs will land under `results/likelihood/imaging/hpc_a100_fp64{,_sparse}.json`
and (per probe) `results/likelihood/imaging/vmap_probe_<cell>{,_sparse}.json`.
