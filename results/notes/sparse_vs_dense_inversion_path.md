# Sparse (w-tilde) vs Dense inversion path — local CPU baselines

**Issue:** [autolens_profiling#44](https://github.com/PyAutoLabs/autolens_profiling/issues/44)
**Branch:** `feature/sparse-vs-dense-profile`
**Status:** Local CPU fp64 baselines complete. A100 submits drafted, not yet
dispatched.

## TL;DR

**Updated with A100 data — the CPU and A100 verdicts disagree.**

- **CPU**: sparse wins big on pix (−41%) and delaunay (−34%), loses on MGE (+51%).
- **A100**: sparse is **slightly slower** on pix (+10%) and delaunay (+7%) per-eval but uses **7–10× less VRAM per replica**. MGE is tied per-eval (both paths land on dense via factory short-circuit) but vmap batch=64 shows sparse 2× faster.

The A100 result reframes the recommendation: **enable sparse for memory, not speed**.
On HST × fp64, every dense-path inversion replica eats ~920 MB of VRAM, which is
exactly why the A100 NSS init OOM'd at `n_live=150 × 184 MB ≈ 28 GB` earlier in
this session (PyAutoFit#1303 / #1305). Sparse replicas eat ~95–130 MB — fits
the same vmap batches in ~10× less memory, no chunked-vmap workaround needed.

For inversion-heavy cells (`pix`, `delaunay`) the per-eval slowdown is small
(<10%) and the VRAM savings unlock larger batches / larger source meshes /
multi-cell vmap. For MGE the factory short-circuits so it doesn't matter
unless you're paying the harness-overhead cost (visible on CPU; invisible on
A100). Recommendation: **enable `apply_sparse_operator()` in
`autolens_profiling/searches/_setup.py:_build_imaging` when the model
contains a pixelization Mapper. Skip for pure-MGE-source.**

## NSS production validation — A100 fp64 (HST)

After landing the `_build_imaging` patch (sparse on by default for imaging)
and the `build_nss` patch (`chunk_size=None` — sampler's native un-chunked
`jax.vmap(num_delete)`), three NSS production searches went out as final
validation. The first sparse pix attempt at 3h wall (323029) timed out
mid-sample; the 12h-wall runs landed:

| Cell | Sampler | Path / chunk | Wall | per-eval | evals | log_Z |
|---|---|---|---:|---:|---:|---:|
| MGE | Nautilus | dense | 13.9 min | 12.1 ms | 63,800 | 31690.47 |
| MGE | NSS | dense, no chunk *(322590)* | 11.3 min | 1.6 ms | 394,321 | 31697.70 |
| **MGE** | **NSS** | **sparse, no chunk** *(323160)* | **11.0 min** | **1.60 ms** | 383,289 | 31786.30 |
| Pix | Nautilus | dense | **46.1 min** | **46.5 ms** | 58,464 | 29066.32 |
| Pix | NSS | dense, chunk=16 *(322619)* | 351 min | 75.9 ms | 277,050 | 29087.77 |
| **Pix** | **NSS** | **sparse, no chunk** *(323161)* | **320 min** | **71.95 ms** | 266,043 | 29142.5 (max L) |
| Delaunay | Nautilus | dense | **45.4 min** | **84.8 ms** | 31,536 | 30562.22 |
| Delaunay | NSS | dense, chunk=16 *(322620)* | 464 min | 135.8 ms | 204,874 | 30569.20 |
| Delaunay | NSS | sparse, no chunk *(323162)* | see HPC | see HPC | see HPC | see HPC |

### What the NSS sparse runs prove

1. **The OOM ceiling is gone.** `chunk_size=None` survived `algo.init` on
   both pix and delaunay at `n_live=150` (the configuration log line —
   which only prints after init — fired ~1 min into each run). Before this
   work, that init step OOM'd at 28 GB on the 80 GB device. Sparse drops
   per-replica VRAM from 931 MB to 95 MB and `n_live=150 × num_delete=50`
   un-chunked fits comfortably.
2. **The chunked-vmap workaround is no longer needed for these cells.**
   PyAutoFit#1303 and #1305 remain valid plumbing for other contexts
   (smaller-VRAM hardware, larger source meshes), but the autolens_profiling
   A100 search path doesn't need them now.
3. **Sparse + un-chunked beats chunked-dense by ~9% wall on pix** (320 min
   vs 351 min) — modest, but it's the chunking-serialisation penalty
   coming back.

### What sparse can't fix

**NSS is still ~7× slower than Nautilus on pixelization (320 min vs 46 min).**
Per-eval is 71.95 ms vs Nautilus's 46.5 ms (~1.55× slower per call) AND
NSS does 4.55× more evaluations (266,043 vs 58,464). The eval-count
multiplier is structural to slice nested sampling — it's not a memory or
vmap-shape issue, it's how NSS explores the prior. Sampler-side optimisation
can't close that gap without a different algorithm (e.g. NUTS).

### Net production guidance

- **MGE / parametric cells**: NSS wins by ~20% wall and 7.5× per-eval.
  Sparse is harmless here (factory short-circuits to dense anyway).
- **Pixelization / Delaunay / other inversion-heavy cells**: Nautilus
  remains the right tool. NSS even at its best vmap shape is ~7× slower
  due to the eval-count gap. The sparse-enable in `_build_imaging` is
  still the right change because future Nautilus runs benefit from the
  same 10× VRAM reduction and would similarly avoid OOMs at larger
  `n_live` / `n_batch` configurations.

## Headline numbers — A100 fp64 (HST)

Six A100 runtime jobs (323017–323022, mge/pix/del × dense/sparse), each
running `--vmap-probe` (writes `results/runtime/imaging/a100_probes/`)
then full timing (logs in `results/runtime/imaging/a100_logs/`). The
likelihood-summary JSONs collided due to a `--config-name hpc_a100_fp64`
filename bug; only pix_d's JSON was salvaged. Timing extracted from logs
into the table below.

| Cell | Phase | Dense | Sparse | Δ |
|---|---|---:|---:|---|
| **MGE** | single-JIT per-call | 5.91 ms | 5.90 ms | tied |
| MGE | vmap batch=64 per-call | 0.77 ms | **0.39 ms** | **sparse 2.0× faster** |
| MGE | per-replica VRAM | 16.4 MB | 16.4 MB | tied (factory short-circuits) |
| **Pixelization** | single-JIT per-call | 52.7 ms | 58.3 ms | sparse +10.5% slower |
| Pix | vmap batch=16 per-call | 32.9 ms | 35.9 ms | sparse +9.1% slower |
| Pix | per-replica VRAM | **931.0 MB** | **94.9 MB** | **sparse 9.81× LESS** |
| Pix | rec. batch (A100 80GB) | 62 | 64 | sparse fits +3% more batch |
| **Delaunay** | single-JIT per-call | 80.0 ms | 85.6 ms | sparse +7.0% slower |
| Delaunay | vmap batch=16 per-call | 106.8 ms | 104.4 ms | tied |
| Delaunay | per-replica VRAM | **921.9 MB** | **131.5 MB** | **sparse 7.01× LESS** |
| Delaunay | rec. batch (A100 80GB) | 62 | 64 | sparse fits +3% more batch |

log_L agreement to 6+ sig figs across dense/sparse on all three cells.

### Why A100 differs from CPU

On CPU the sparse w-tilde path's cache-friendly pixel-pair access pattern
out-paces the dense scatter+matmul by 30-40%. On A100 the dense scatter+matmul
gets to use all 6,912 CUDA cores at once, so the absolute per-eval cost falls
1000× from CPU's 7.9 s to A100's 53 ms — and within that compute-rich regime,
the dense path is marginally faster because each step has good GPU parallelism.
What A100 *can't* hide is the VRAM cost: dense materialises the full
(15,361 × 1500) × 8 byte = 184 MB mapping matrix per replica, which scales
linearly with `n_live` and would OOM the 80 GB device at `n_live ≈ 432`
without batched chunking. Sparse keeps the per-replica VRAM at ~95–131 MB,
fits much larger vmaps, and would have made the chunked-vmap workaround
in PyAutoFit#1303/#1305 unnecessary.

### Filename collision note

The submit scripts passed `--config-name hpc_a100_fp64` which makes every
cell's `likelihood_summary` JSON write to the same `hpc_a100_fp64.json`
filename (mod the `_sparse` suffix). The `vmap_probe_<cell>{,_sparse}.json`
files DID disambiguate by cell, so those all survived. Only pix_d's
final JSON was salvaged (by renaming on the HPC before del_d overwrote);
the rest of the dense JSONs were lost, and the final `_sparse.json` is
whatever the last sparse cell to finish wrote (mge_s). Future submits
should use `--config-name hpc_a100_fp64_<cell>` to avoid the collision
**OR** patch `_profile_cli.resolve_output_paths` to always include the
cell name in the basename. Doesn't affect the comparison numbers because
the logs preserved the full timing.

## Headline numbers (HST, CPU fp64)

### Full-pipeline single-JIT (the production-relevant cost)

Clean re-run, no `--config-name` (so each cell writes to its own filename):

| Cell | Dense per-eval | Sparse per-eval | Δ | log_L agreement |
|---|---:|---:|---:|---|
| MGE | 0.083 s | 0.125 s | **+50.9% (sparse loses)** | exact (27379.388906855238) |
| Pixelization | 7.90 s | **4.65 s** | **−41.1%** | bit-identical to 12 sig figs (28398.444158983…) |
| Delaunay | 5.53 s | **3.63 s** | **−34.4%** | bit-identical to 12 sig figs (29110.920857938…) |

Raw JSONs land under `results/likelihood/imaging/{pixelization,delaunay,mge}_likelihood_summary_hst_v<v>{,_sparse}.json`.

**Cross-run variance note:** earlier stdout captures (see
`results/runtime/imaging/stdout_captures/`) showed slightly different numbers
on the same hardware (pix dense 9.04 s, MGE sparse appearing 48% faster).
Those runs overlapped with the system doing other work; the clean re-run
above is the more reliable read. The pix/delaunay verdict (sparse wins big)
is robust across both runs; only MGE is noise-floor-sensitive.

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

1. **MGE loses on sparse (+51%).** The factory's
   `all-AbstractLinearObjFuncList` short-circuit fires — pure-MGE-source
   models land on `InversionImagingMapping` (dense) even with
   `sparse_operator` attached. The 51% slowdown comes from the
   `apply_sparse_operator()` call building the w-tilde kernel at dataset-
   construction time, which then gets carried through every likelihood
   eval as unused state. Confirmation: the JSON's `inversion_path` field
   says "sparse" (flag was set), but the factory's actual class choice
   was `InversionImagingMapping` and the per-eval cost reflects only
   the dense path plus harness overhead.
2. **Pixelization breakdown disagrees with runtime.** The breakdown says
   sparse +14.6% slower; the runtime says sparse −41.1% faster. Trust the
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
4. **First-run MGE measurement was noise.** An initial run showed sparse
   48% faster on MGE; the clean re-run showed sparse 51% slower. Both
   runs were on the same hardware; the first happened during higher
   system load (other Python procs in flight). Single-eval timings at
   ~100ms scale are noise-floor sensitive; the pix/delaunay numbers are
   robust (multi-second per-eval, ratio holds across both runs).

## Recommendation

1. **Conditionally enable sparse in `_build_imaging`.** The cleanest
   intervention is to attach `apply_sparse_operator()` only when the
   `model` argument indicates a pixelization Mapper is in play
   (pix / Delaunay source). For pure-MGE-source cells, leave the dataset
   alone — the sparse-kernel build is wasted work because the factory
   short-circuits to dense regardless. The simplest signal:
   inspect `model_type` in the caller, or split `_build_imaging` into
   `_build_imaging_pixelized` (which calls `apply_sparse_operator`) and
   `_build_imaging_parametric` (which doesn't).
2. **Run the 6 A100 submits** at
   `hpc/batch_gpu/submit_runtime_imaging_{mge,pixelization,delaunay}_a100_hst_fp64{,_sparse}`.
   Each does a `--vmap-probe` phase (writes `vmap_probe_<cell>{,_sparse}.json`
   with `peak_memory_in_bytes` per batch + recommended A100 batch size) then
   a `--config-name hpc_a100_fp64` phase (full single-JIT + vmap timing,
   writes `hpc_a100_fp64{,_sparse}.json`). Compare against the existing
   A100 dense numbers committed to `results/searches/` from the
   NSS-vs-Nautilus session.
3. **MGE sparse-loss is expected**, not a bug. Don't investigate further
   unless A100 numbers contradict this CPU finding.
4. **Defer matrix-free CG + SLQ** (the original plan-B). The sparse path
   gives clean wins on inversion-heavy imaging cells (pix / Delaunay)
   with direct Cholesky log-det preserved. Matrix-free CG would add SLQ
   noise for log-det and is only worth pursuing if a future scenario
   (e.g. ultra-large source meshes on H100) reveals sparse can't keep up
   either.

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
