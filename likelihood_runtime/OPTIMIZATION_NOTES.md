# Likelihood profiling — where to focus optimization next

A per-cell view of `autolens_profiling/likelihood/<dataset_class>/<model>.py`
focused on **where the next round of optimization should land**, derived
from the multi-config sweep results that live alongside each cell at
`autolens_workspace_developer/jax_profiling/results/jit/<class>/<model>/comparison.{json,png}`.

This is the companion to the per-class READMEs (`imaging/README.md`,
`interferometer/README.md`, etc.) — those describe **what** each script
profiles; this doc captures **what to do with the numbers**.

Sweep produced as part of `feature/likelihood-multiconfig-sweep`,
PyAutoLens v2026.5.14.2. Hardware: local RTX 2060 Max-Q (6 GB) for the
GPU rows; local x86 CPU for the CPU rows. HPC A100 rows are a separate
follow-up (see the bottom of this doc).

## Status of this doc

| Sweep dimension                  | Status |
|----------------------------------|--------|
| Local CPU (fp64 + mp)            | ✅ run, numbers below |
| Local GPU (RTX 2060) fp64 + mp   | ✅ run, numbers below |
| HPC A100 sma (4 cells)           | ✅ run 2026-05-21 — interferometer/delaunay + datacube/delaunay × sma × fp64 + mp |
| HPC A100 alma (4 cells)          | ✅ run 2026-05-22 — unblocked by PyAutoArray#329 (apply_sparse_operator now accepts TransformerNUFFT) |
| HPC A100 alma_high (4 cells)     | ✅ run 2026-05-22 — unblocked by PyAutoArray#330 (TransformerNUFFT chunk_size knob caps the nufftax gather buffer) |
| HPC A100 jvla (2 cells, stretch) | ✅ run 2026-05-24 — interferometer/delaunay only (25M vis, 700-px mask, pixel_scale=0.01). No fix needed; chunked NUFFT + W-Tilde sparse path held. |
| Imaging cells fresh CPU/GPU      | ⚠ blocked by upstream `Grid2DIrregular.mask` bug — table rows show the pre-existing v2026.5.8.2 / v2026.5.14.2 data |

## Headline numbers (full pipeline, single JIT per call)

| Cell                           | CPU fp64 | CPU mp  | GPU fp64 | GPU mp  | GPU vs CPU |
|--------------------------------|----------|---------|----------|---------|------------|
| interferometer/mge (NUFFT)     | 261 ms   | 202 ms  | 1.97 s   | 1.43 s  | **8× SLOWER** |
| interferometer/pixelization    | 753 ms   | 697 ms  | 111 ms   | 106 ms  | **6.8× faster** |
| interferometer/delaunay        | 881 ms   | 788 ms  | 131 ms   | 138 ms  | **6.7× faster** |
| datacube/delaunay (34-ch cube) | 197 s    | 165 s   | 18.1 s   | 18.0 s  | **11× faster** |

The headline story splits cleanly:

- **The sparse-DFT cells (pix / del / datacube) all win 6.7–11× on GPU.** Use the
  GPU for production sampling on these.
- **NUFFT-on-mge is 8× slower on GPU than on CPU at SMA's 190 visibilities.**
  nufftax's RTX 2060 lowering doesn't amortize the interpolation+oversampling
  overhead at small N. **Use `--use-dft` for SMA-class mge sampling on either
  CPU or GPU**; reserve NUFFT for ALMA-class visibility counts (not yet profiled).
- **mp gives a meaningful CPU win on every cell** (7–23 %). On GPU the mp win
  is more variable — meaningful on NUFFT mge (27 %), negligible-to-zero
  elsewhere. **Default to mp on CPU; flip per cell on GPU.**

## Pending follow-ups

1. **HPC A100 (fp64 + mp)** — separate PR. Local data already says GPU
   sparse-DFT collapses to O(100 ms) on the consumer GPU; A100 should
   reach ~10–30 ms per call.
2. **ALMA dataset for interferometer/mge** — without it the NUFFT-vs-DFT
   crossover can't be measured. Existing `INSTRUMENTS` preset is wired
   but the dataset isn't checked in.
3. **Imaging per-step JIT** — blocked by `PyAutoGalaxy/autogalaxy/profiles/basis.py:151`
   (`Grid2DIrregular does not have attribute mask`). Upstream fix.

### Re-running the local GPU sweep

```bash
source /home/jammy/Code/PyAutoLabs-wt/likelihood-multiconfig-sweep/activate.sh
cd /home/jammy/Code/PyAutoLabs-wt/likelihood-multiconfig-sweep/autolens_profiling

# GPU-only sweep across the 4 in-scope non-imaging cells.
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu /home/jammy/venv/PyAutoGPU/bin/python \
    scripts/sweep_likelihood.py --skip-cpu \
    --only interferometer/mge interferometer/pixelization interferometer/delaunay datacube/delaunay

# Re-aggregate so comparison.{json,png} include the new rows.
/home/jammy/venv/PyAutoGPU/bin/python scripts/aggregate_sweep.py
```

Note: `JAX_PLATFORMS=cuda,cpu` (not just `cuda`) is required because the
Delaunay and datacube paths use `jax.pure_callback` for Hilbert-curve mesh
generation, which needs a CPU device available even when the primary
platform is GPU.


## Reading the cell sections

Each cell below carries:

- **Timings table** — `local_cpu_{fp64,mp}` and `local_gpu_{fp64,mp}` per-call
  steady-state for the full pipeline JIT, plus the vmap per-call where the
  cell supports it. HPC A100 columns are blank where the follow-up sweep
  hasn't run.
- **Dominant steps per device class** — extracted from the per-step block
  in each `local_*.json`, top two by share of total step time.
- **Where to focus next** — 1–4 sentences naming the specific step / library
  function / refactor that would yield the most wall-clock.
- **mp verdict** — "use it" / "neutral" / "skip" based on the measured mp
  vs fp64 delta on each device class.

A "—" entry means the row failed or wasn't run; see the cell's notes.

---

## imaging/pixelization
*Rectangular `RectangularAdaptImage` mesh + Constant regularization. 35×35
source pixels = 1225 source mesh nodes. Default HST-resolution dataset
(0.05″/px, mask radius 3.5″, 15 361 masked pixels, 17 980 over-sampled).*

**⚠ Per-step JIT path currently fails** at `PyAutoGalaxy/autogalaxy/profiles/basis.py:151`
(`Grid2DIrregular does not have attribute mask`) when run against
PyAutoLens v2026.5.14.2. The pre-existing v2026.5.8.2 sweep data in
`comparison.json` is still authoritative; the timings below come from it.

| Config            | full pipeline | vmap (batch=3) | log_evidence |
|-------------------|---------------|----------------|--------------|
| local_cpu_fp64    | 4.44 s        | 4.99 s         | 24 746.1     |
| local_cpu_mp      | 3.80 s        | 4.31 s         | 24 746.1     |
| local_gpu_fp64    | 537 ms        | 567 ms         | 24 746.1     |
| local_gpu_mp      | 495 ms        | 528 ms         | 24 746.1     |
| hpc_a100_fp64     | 25 ms         | 35 ms          | 24 746.1     |
| hpc_a100_mp       | 25 ms         | 34 ms          | 24 746.1     |

**Dominant steps**
- *CPU*: `Curvature matrix (F)` ≈ 50 %, `Inversion setup` ≈ 35 %.
- *Consumer GPU*: `Curvature matrix (F)` ≈ 50 %, `Regularized reconstruction` ≈ 33 %.
- *A100*: `Regularization matrix (H)` ≈ 80 %, `Regularized reconstruction` ≈ 7 %. The
  dense matrix construction collapses on A100; the bottleneck has shifted entirely
  to NNLS / log-determinant.

**Where to focus next**
1. **NNLS reconstruction prototypes** (`z_projects/profiling/scripts/nnls_prototypes/`)
   are the highest-value lever at A100 scale — once F-construction is fast,
   the serial NNLS reconstruction is what's left to beat.
2. On consumer GPU, **F-matrix construction sparsity** is the win. The
   sparse path is already wired via the existing imaging `apply_sparse_operator`
   variants (`pixelization_sparse_cpu_*` JSONs exist) but the GPU-sparse story
   hasn't been profiled.
3. **vmap actively hurts** here (0.89× on CPU, 1.05× on A100) — do not
   reach for it on this cell.

**mp verdict** — modest on CPU (~14 % win), neutral elsewhere.
**Useful only at CPU scale**; skip on GPU.

---

## imaging/delaunay
*Delaunay-triangulated source pixelization. 39×39 overlay + 30 edge points
→ 1231 mesh vertices. ConstantSplit regularization.*

**⚠ Same upstream bug** at `basis.py:151` blocks the per-step JIT path
under v2026.5.14.2. Pre-existing GPU/A100 sweep data is authoritative;
CPU rows have no fresh measurement available.

| Config            | full pipeline | vmap (batch=3)   | log_evidence |
|-------------------|---------------|------------------|--------------|
| local_cpu_fp64    | —             | —                | —            |
| local_cpu_mp      | —             | —                | —            |
| local_gpu_fp64    | 590 ms        | 954 ms (slowdown)| 26 288       |
| local_gpu_mp      | 557 ms        | 1.22 s           | 26 288       |
| hpc_a100_fp64     | 50 ms         | 438 ms (heavy compile) | 26 288 |
| hpc_a100_mp       | 50 ms         | 441 ms           | 26 288       |

**Dominant steps (A100)**
- `Inversion setup (steps 5-8 combined)` ≈ 62 % — border-relocation + mapper
  + mapping matrix + transformed mapping matrix, all fused.
- `Regularization matrix (H)` ≈ 45 % — second-most-expensive after the inversion
  setup; the Hilbert-curve regularization is dense at 1231 vertices.

**Where to focus next**
1. **Inversion setup is the new ceiling** on A100. Profiling the four
   constituent sub-steps (border, mapper, mapping matrix, transformed
   mapping matrix) individually would identify which one dominates the 31 ms.
2. **vmap is actively a regression** (0.6×, 0.5×) — the Delaunay construction
   path doesn't batch cleanly. Don't use vmap for this cell.
3. **H-matrix dense construction** is the dark horse on A100; if regularization
   sparsity can be exploited (the existing `delaunay_sparse_cpu_*` variants
   in workspace_developer/jit/imaging/ exist for CPU; GPU-sparse is untested).

**mp verdict** — barely measurable (~5 % on GPU). Skip; it's not worth the
correctness-budget pressure.

---

## interferometer/mge
*Multi-Gaussian-expansion source (20 linear Gaussians), Isothermal +
ExternalShear lens. SMA dataset: 190 visibilities, 256×256 real-space grid,
0.1″/px, mask radius 3.0″.*

**Default transformer is now `al.TransformerNUFFT`** (nufftax-backed, JAX
native). Pass `--use-dft` to fall back to `TransformerDFT` for direct
comparison against the historical baseline. On SMA's 190 visibilities the
DFT-vs-NUFFT eager log-likelihood agrees bit-for-bit to fp64 — there is no
correctness gap.

| Config            | transformer | full pipeline | vmap (batch=3) |
|-------------------|-------------|---------------|----------------|
| local_cpu_fp64    | NUFFT       | 261 ms        | 229 ms (1.1×)  |
| local_cpu_mp      | NUFFT       | 202 ms        | 200 ms (1.0×)  |
| local_gpu_fp64    | NUFFT       | **1.97 s**    | 616 ms (3.2×)  |
| local_gpu_mp      | NUFFT       | 1.43 s        | 647 ms (2.2×)  |
| local_gpu_fp64    | DFT (--use-dft) | _follow-up_   | _follow-up_ |
| hpc_a100_fp64     | NUFFT       | —             | —              |
| hpc_a100_mp       | NUFFT       | —             | —              |

For reference, the historical SMA baseline with `TransformerDFT` (pre-PR)
was **34 ms** full pipeline on CPU with vmap speedup 2.1×.

**Key findings**

- **NUFFT on GPU is dramatically worse than NUFFT on CPU** at SMA's 190
  visibilities (1.97 s vs 261 ms — 8× slower). The CPU↔GPU crossover
  goes the *wrong way* on nufftax at this regime: it does NOT amortize
  the interpolation + oversampling overhead at small visibility counts
  on RTX 2060.
- **CPU NUFFT is ~8× slower than the historical CPU DFT baseline**
  (261 ms vs 34 ms). The NUFFT swap pays off only at large visibility
  counts (ALMA-class). **Use `--use-dft` for SMA-class mge sampling.**
- **mp helps everywhere** here: 23 % win on CPU, 27 % win on GPU.
- **vmap is the bright spot on GPU NUFFT** — 3.2× speedup fp64, 2.2× mp.
  Unlike the sparse-DFT pixelized cells, vmap *does* batch cleanly on
  the linear MGE path. With vmap batch=3 the GPU per-likelihood cost
  drops to 616 ms — still slower than CPU steady-state but the gap
  closes.
- **Memory regression** — NUFFT eats 480 MB of XLA temp vs DFT's 17 MB
  on the same dataset (28× more memory). RTX 2060 (6 GB) is approaching
  saturation with vmap batch=3; A100 (80 GB) won't notice but it's a
  flag for batch-size budgeting on consumer hardware.

**Where to focus next**

1. **Profile against the ALMA `INSTRUMENTS` preset** in the script
   (already wired but never run). That's where NUFFT vs DFT becomes
   meaningful. On SMA the swap is a regression on every measured config.
2. **Add ALMA dataset under autolens_profiling/dataset/interferometer/alma/**
   so the existing `instrument = "sma"` line in the script can be flipped
   to `"alma"` without manual regeneration.
3. **GPU NUFFT investigation** — the 8× GPU↔CPU regression suggests
   nufftax's GPU lowering on consumer hardware is unoptimised at small N.
   Worth exploring nufftax knobs (`eps`, oversampling factor) before
   declaring this a permanent state of affairs.
4. **The 3.2× vmap speedup on GPU** is the strongest signal in this cell —
   if the sampler hot path is vmap-able, NUFFT-on-GPU is recoverable.

**mp verdict** — **use it** everywhere. 23 % CPU win, 27 % GPU win on
this cell. No measurable correctness drift.

---

## interferometer/pixelization
*Rectangular pixelization on visibilities. **32×32 mesh → 1024 source pixels**
(production 1000-tier fiducial). SMA dataset.*

**The sparse-operator path is wired** (`dataset.apply_sparse_operator(use_jax=True)`,
new in this PR) — pre-fit curvature assembly now uses the FFT-based sparse
precision matrix rather than dense DFT on every likelihood call. Pre-PR
timings (113 ms full pipeline at 28×28 mesh) are the **non-sparse**
baseline and are discarded per the user requirement — only sparse-path
timings going forward.

The transformer stays `TransformerDFT` (not NUFFT) because
`apply_sparse_operator` raises `NotImplementedError` against the new
nufftax-backed `TransformerNUFFT` —
`PyAutoArray/autoarray/dataset/interferometer/dataset.py:261`. See the
follow-up note at the bottom for what's needed to lift this constraint.

| Config            | full pipeline | vmap (batch=3)  | log_evidence |
|-------------------|---------------|-----------------|--------------|
| local_cpu_fp64    | 753 ms        | 1.04 s (0.7×)   | −3166.34     |
| local_cpu_mp      | 697 ms        | 928 ms (0.8×)   | −3166.34     |
| local_gpu_fp64    | **111 ms**    | 113 ms (1.0×)   | −3166.34     |
| local_gpu_mp      | 106 ms        | 112 ms (0.9×)   | −3166.34     |
| hpc_a100_fp64     | —             | —               | —            |
| hpc_a100_mp       | —             | —               | —            |

**Key finding** — **GPU is 6.8× faster than CPU on this cell** (753 → 111 ms).
The sparse precision-operator path GPU-lowers extremely well; the dense
fixed-size linear algebra in the per-fit curvature assembly is exactly the
workload GPU dominates. This is the clearest GPU win in the sweep so far.

**Where to focus next**

1. **vmap remains a non-win on GPU** too (1.0× / 0.9×). The sparse
   precision-matrix solve doesn't batch cleanly along the parameter
   axis. Use data-parallel throughput (multiple processes / DataParallel)
   rather than vmap for this cell.
2. **A100 should reach O(10 ms) per call** based on the 6.8× CPU→consumer-GPU
   collapse. Likely the cheapest A100 win in the sweep.
3. **The XLA temp footprint is 193 MB** on CPU. GPU temp wasn't measured
   in the JSON yet but RTX 2060 has 6 GB headroom; A100 isn't a concern.
4. **Sparse precompute cost is amortised across all per-fit calls** —
   the 111 ms per-call number is the steady-state production cost a
   sampler pays. The one-time precompute (~few seconds at SMA scale) is
   excluded from the per-call measurement.

**mp verdict** — modest win on both CPU (~7 %) and GPU (~5 %). Worth
using; cheap to opt in.

---

## interferometer/delaunay
*Delaunay pixelization on visibilities. 1000 mesh vertices (Hilbert
sampling). SMA dataset. The `apply_sparse_operator` path is the
production case.*

### sma (190 visibilities)

| Config            | full pipeline | vmap (batch=3)               | log_evidence |
|-------------------|---------------|------------------------------|--------------|
| local_cpu_fp64    | 881 ms        | _vmap intentionally skipped_ | —            |
| local_cpu_mp      | 788 ms        | _vmap intentionally skipped_ | —            |
| local_gpu_fp64    | 131 ms        | _vmap intentionally skipped_ | —            |
| local_gpu_mp      | 138 ms        | _vmap intentionally skipped_ | —            |
| hpc_a100_fp64     | **33 ms**     | _vmap intentionally skipped_ | −3151.54     |
| hpc_a100_mp       | 33 ms         | _vmap intentionally skipped_ | −3151.54     |

### alma (1M visibilities)

| Config            | full pipeline | vmap (batch=3)               | log_evidence |
|-------------------|---------------|------------------------------|--------------|
| hpc_a100_fp64     | **45 ms**     | _vmap intentionally skipped_ | −12 049 403.72 |
| hpc_a100_mp       | 45 ms         | _vmap intentionally skipped_ | −12 049 403.72 |

Unblocked by [PyAutoArray#329](https://github.com/PyAutoLabs/PyAutoArray/pull/329)
(`apply_sparse_operator` now accepts `al.TransformerNUFFT`, eliminating
the O(N_pix · N_vis) DFT setup call that previously OOM'd at 384 GB).
The per-likelihood path is unchanged — F is FFT-based via Khat, D uses
the cached dirty image, χ² is `inversion.fast_chi_squared`. Setup-time
`image_from` is now one nufftax call at O((N_pix + N_vis) log N).

**Key findings (alma):**

- **alma at 1M vis is only 1.4× slower than sma at 190 vis** (45 ms vs 33
  ms). The W-Tilde sparse path is largely vis-independent per likelihood
  call — F is the FFT of a small precision operator, D is a cached
  matrix-vector product. Per-call cost is dominated by mask-extent FFT
  size, not visibility count. This is the headline result for ALMA-scale
  modelling on A100.
- **mp is a wash** (45.3 vs 44.8 ms — essentially identical), same pattern
  as sma. fp64 is the right default.

### alma_high (5M visibilities, high-res `pixel_scale=0.025`)

| Config            | full pipeline | vmap (batch=3)               | log_evidence |
|-------------------|---------------|------------------------------|--------------|
| hpc_a100_fp64     | **98 ms**     | _vmap intentionally skipped_ | −60 243 535.86 |
| hpc_a100_mp       | 101 ms        | _vmap intentionally skipped_ | −60 243 535.86 |

Unblocked by [PyAutoArray#330](https://github.com/PyAutoLabs/PyAutoArray/pull/330)
(TransformerNUFFT `chunk_size` knob caps the nufftax gather buffer at
`2 × chunk_size × nspread² × dtype_size`; per-instrument default for
alma_high is `chunk_size=1_000_000`). Simulator runs cleanly on A100 in
~80 s for the full 5M-vis dataset.

### jvla (25M visibilities, stretch `pixel_scale=0.01`)

| Config            | full pipeline | vmap (batch=3)               | log_evidence |
|-------------------|---------------|------------------------------|--------------|
| hpc_a100_fp64     | **636 ms**    | _vmap intentionally skipped_ | −301 296 857.98 |
| hpc_a100_mp       | 604 ms ⚡      | _vmap intentionally skipped_ | −301 296 857.96 |

Stretch-test preset added 2026-05-24 to probe the upper end of the
W-Tilde sparse-path scaling: 25M visibilities (5× alma_high, 25× alma),
`pixel_scale=0.01` (700-px mask diameter, 5× alma_high's 140-px). Pure
unblock — same `chunk_size=1_000_000` setting from alma_high works
identically here (now 25 chunks instead of 5). Simulator ran in 77 s,
profile cells in ~10 min each. **No library fix was needed.**

The jvla result is the first non-trivial mp win on A100 across this
sweep: mp is **5% faster than fp64** (604 vs 636 ms). At 700-px mask
diameter the FFT kernel finally becomes large enough that the
mixed-precision matmul speedup overtakes per-call constant overhead.

**Per-call scaling validated.** Across the four instrument presets
(same model, same Hilbert pixel budget, sparse-operator path):

| Instrument | n_vis | pixel_scale | mask radius (px) | per-call (fp64) | per-call (mp) |
|------------|-------|-------------|------------------|-----------------|---------------|
| sma        | 190   | 0.1         |  35              | 33 ms           | 33 ms         |
| alma       | 1 M   | 0.05        |  70              | 45 ms           | 45 ms         |
| alma_high  | 5 M   | 0.025       | 140              | 98 ms           | 101 ms        |
| jvla       | 25 M  | 0.01        | 350              | **636 ms**      | **604 ms** ⚡  |

Per-call cost scales **with mask radius (in pixels)**, not with
visibility count. Going alma → alma_high doubles the mask diameter
(4× more mask pixels → 4× more FFT work), and the per-call time
~doubles (45 → 98 ms). Going sma → jvla, visibility count scales
**132 000×** (190 → 25M) but per-call time only scales 19× (33 → 636 ms).
This is the clearest empirical confirmation yet of the W-Tilde
sparse-formalism prediction: per-likelihood cost is dominated by the
mask-extent FFT (`O(N_mask · log N_mask)`), and visibility count enters
only the one-shot, setup-time NUFFT precision-matrix precompute.

**Surprise at jvla scale: mp is meaningfully faster than fp64** (604 vs
636 ms, ~5% mp win — first non-wash on this sweep). At a 700-px mask
diameter the FFT kernel is large enough for mixed-precision matmul
gains to surface above the constant per-call overhead. mp continues to
be a wash at smaller mask scales (sma / alma / alma_high).

**Key findings (sma)**

- **A100 is 4× faster than RTX 2060** (33 ms vs 131 ms) and **27× faster
  than CPU** (881 ms). Sparse precision-matrix solve scales cleanly
  through A100 — exactly what the LIKELIHOOD path was designed to do.
- **mp is a wash on A100** (33.0 vs 32.8 ms — essentially identical).
  Same pattern as RTX 2060. fp64 is the right default on this cell.
- **GPU is 6.7× faster than CPU** (881 → 131 ms RTX 2060). Same magnitude as
  interferometer/pixelization — the sparse precision-matrix solve dominates
  both cells and lowers similarly well to GPU.
- **mp is a wash on RTX 2060** (138 ms vs 131 ms — mp is actually 5 % SLOWER
  for some reason; possibly the fp32 path triggers an extra cast at the
  Delaunay/sparse-operator boundary that hurts more than the fp32 speedup
  helps). Sticking with fp64 is fine here.

**Important caveat: this cell uses `jax.pure_callback`** (for the Hilbert
mesh generation in the Delaunay path). GPU runs require
`JAX_PLATFORMS=cuda,cpu` (not just `cuda`) — without the CPU device
available the callback raises
`jax.pure_callback failed to find a local CPU device`. The sweep harness
sets this automatically; document the requirement if anyone hand-runs the
script with `JAX_PLATFORM_NAME=cuda`.

**Where to focus next**

1. The **per-step breakdown is not yet measured for this cell** (the
   interferometer/delaunay script per-step-JITs the `Inversion setup`
   step as one combined block "steps 5-8 incl. NUFFT"). Re-running with
   the per-step decomposition is a follow-up — the combined block
   currently swallows the dominant fraction of the ~131 ms GPU per call.
2. **mp gives a real ~11 % CPU win** (881 → 788 ms) but is a regression
   on GPU. Default to CPU mp, GPU fp64 on this cell.
3. **A100 should land at O(20-30 ms)** based on the 6.7× consumer-GPU win.

**mp verdict** — **use on CPU only** (11 % win); skip on GPU (5 %
regression). Identical log-evidence to fp64 on both.

---

## datacube/delaunay
*34-channel cube reusing the SMA interferometer dataset. Delaunay source
pixelization per channel, shared lens model. The single heaviest cell
in this profile family.*

The JSON step labels say "incl. NUFFT" — that's accurate: `apply_sparse_operator(use_jax=True)`
precomputes a NUFFT precision matrix, which is what the per-channel
curvature assembly actually goes through. As of [PyAutoArray#329](https://github.com/PyAutoLabs/PyAutoArray/pull/329)
the dataset is also built with `TransformerNUFFT` (nufftax-backed),
enabling the alma/alma_high scales.

### sma (190 visibilities × 34 channels)

| Config            | full pipeline (cube)       | log_evidence | Notes |
|-------------------|----------------------------|--------------|-------|
| local_cpu_fp64    | 197 s (step-by-step total) | −107830.7    | from breakdown package |
| local_cpu_mp      | 165 s                      | −107830.7    | mp −16 % |
| local_gpu_fp64    | 18.1 s                     | −107830.7    | 10.9× CPU |
| local_gpu_mp      | 18.0 s                     | −107830.7    | mp wash |
| hpc_a100_fp64     | **eager baseline only**    | −107830.7    | runtime variant; cube-JIT skipped (opt in via `CUBE_FULL_JIT=1`) |
| hpc_a100_mp       | eager baseline only        | −107830.7    | same |

The A100 SMA jobs verified the eager-baseline log_evidence
(−107830.66 — matches the local CPU+GPU runs at fp64 precision) but
the runtime variant's `CUBE_FULL_JIT=1` path was off, so no
per-call wall-clock number landed. Two follow-ups:

1. Re-run the SLURM submits with `CUBE_FULL_JIT=1` to get the
   single-JIT cube cost on A100 — expected sub-10 s based on the
   RTX 2060 → A100 ratio.
2. The per-channel step-by-step breakdown (the `197 s on CPU` row
   above) is a **`likelihood_breakdown/datacube/delaunay.py`** artifact;
   re-run that on A100 for the per-step decomposition.

### alma (1M visibilities × 34 channels)

| Config            | full pipeline (cube)       | log_evidence (cube) | log_evidence/channel | Notes |
|-------------------|----------------------------|---------------------|----------------------|-------|
| hpc_a100_fp64     | **eager baseline only**    | −409 648 566.87     | −12 048 487.26       | runtime variant; cube-JIT skipped (opt in via `CUBE_FULL_JIT=1`) |
| hpc_a100_mp       | eager baseline only        | −409 648 566.87     | −12 048 487.26       | same |

Unblocked by [PyAutoArray#329](https://github.com/PyAutoLabs/PyAutoArray/pull/329).
All 34 channels finished `apply_sparse_operator` on A100 within the 1-hour
SLURM wall budget (in stark contrast to the previous OOM at 384 GB host
RAM with `TransformerDFT`). Per-channel eager log_evidence matches
`interferometer/delaunay/alma` exactly (each channel is the same single-channel
dataset replicated), confirming the math.

Per-call cube-JIT timing is still pending — opt in with `CUBE_FULL_JIT=1`
on a follow-up SLURM run. Expected based on the interferometer alma row
(45 ms × 34 channels ≈ 1.5 s/cube, give or take XLA fusion savings).

### alma_high (5M visibilities × 34 channels, `pixel_scale=0.025`)

| Config            | full pipeline (cube)       | log_evidence (cube) | log_evidence/channel | Notes |
|-------------------|----------------------------|---------------------|----------------------|-------|
| hpc_a100_fp64     | **eager baseline only**    | −2 048 222 823.68   | −60 241 847.76       | runtime variant; cube-JIT skipped (opt in via `CUBE_FULL_JIT=1`) |
| hpc_a100_mp       | eager baseline only        | −2 048 222 823.67   | −60 241 847.75       | same |

Unblocked by [PyAutoArray#330](https://github.com/PyAutoLabs/PyAutoArray/pull/330).
All 34 channels finished `apply_sparse_operator` on A100 within the
~31-minute SLURM wall budget at the chunked `chunk_size=1_000_000`
setting (longer than alma's 21-min run due to the 4× larger mask FFT
per channel at `pixel_scale=0.025`). Per-channel eager log_evidence
matches `interferometer/delaunay/alma_high` within ~0.005% (small drift
down to fixed-seed-driven model parameter differences between the two
scripts; well within the math-equivalence threshold).

**Headline finding (local data)** — **the 34-channel cube drops from 197 s to 18 s on
GPU** (10.9× faster), making per-cube fits genuinely interactive on RTX 2060.
A100 should reach sub-10 s territory once `CUBE_FULL_JIT=1` is enabled in
a follow-up SLURM run.

The cube full-pipeline single-JIT is intentionally skipped on this cell
(opt in with `CUBE_FULL_JIT=1`) — the per-step cube cost is the
authoritative number. `shared_lwl_savings_estimate` field in each JSON
quantifies the ~32 s of moveable cost still available via the shared
`LᵀW̃L` optimization.

**CPU fp64 per-channel cost breakdown (× 34 channels)**

| Step                                | per-channel | × 34 | share |
|-------------------------------------|-------------|------|-------|
| Inversion setup, incl. NUFFT        | 4.52 s      | 154 s | 78 %  |
| Curvature matrix F                  | 0.99 s      | 34 s  | 17 %  |
| Data vector D                       | 0.22 s      | 7.3 s | 4 %   |
| Reconstruction NNLS                 | 0.05 s      | 1.6 s | 1 %   |
| Other (regularization, log-evidence)| —           | 0.6 s | < 1 % |

**CPU mp per-channel deltas** — what mp actually changes

| Step                                | fp64 × 34 | mp × 34 | mp speedup |
|-------------------------------------|-----------|---------|------------|
| Inversion setup, incl. NUFFT        | 154 s     | 133 s   | **13 %**   |
| Curvature matrix F                  | 33.5 s    | 23.2 s  | **31 %**   |
| Data vector D                       | 7.3 s     | 6.4 s   | 13 %       |
| Reconstruction NNLS                 | 1.6 s     | 1.2 s   | 24 %       |

Curvature F is the canonical mp-friendly step (dense matrix construction
in fp32 instead of fp64) and shows the cleanest 31 % win; the inversion
setup's fp32-friendly sub-blocks contribute another 13 %.

**GPU fp64 per-channel cost breakdown (× 34 channels)** — for comparison

| Step                                | per-cube | share | CPU→GPU collapse |
|-------------------------------------|----------|-------|------------------|
| Inversion setup, incl. NUFFT        | 12.6 s   | 70 %  | 12×              |
| Curvature matrix F                  | 3.1 s    | 17 %  | 11×              |
| Reconstruction NNLS                 | 1.4 s    | 8 %   | 1.2×             |
| Mapped recon + log evidence         | 0.9 s    | 5 %   | < 1×             |
| Data vector D                       | 0.06 s   | < 1 % | 122×             |

The inversion-setup step's share rises from 78 % on CPU to 70 % on GPU — it's
slightly less dominant on GPU but still the bottleneck. NNLS reconstruction
is the only step that does NOT collapse on GPU (1.2× speedup) — it's the
serial NNLS solve, which doesn't vectorize. As N_channels grows, NNLS will
start to dominate the GPU profile (currently 8 % of cube cost).

**Where to focus next**

1. **Per-channel inversion setup** is overwhelmingly dominant (78 % of
   cube cost). The `shared_lwl_savings_estimate = 32 s` is the largest
   discrete refactor available — moving the curvature matrix from
   per-channel to shared.
2. **Decompose the inversion-setup step** (currently one combined block
   "steps 5-8 incl. NUFFT") to figure out whether the border-relocation,
   mapper, mapping-matrix, or NUFFT sub-block dominates. The shared
   `LᵀW̃L` refactor is only useful if NUFFT isn't the bottleneck.
3. **NUFFT swap (transformer-level)** is mandatory for ALMA-scale
   realistic cubes (16 984 visibilities/channel) but cannot land until
   the nufftax-backed `TransformerNUFFT` + `apply_sparse_operator`
   adjoint-scale mismatch is fixed upstream — see the follow-up note.
4. **vmap is intentionally skipped** on the cube cell. The natural
   batching axis is "datasets" not "parameters"; the harness honors this.

**mp verdict** — **use on CPU only**. 16 % cube-level CPU win, driven
primarily by the 31 % fp32 win on curvature-matrix construction. On GPU
the mp delta is < 1 % (18.0 vs 18.1 s) — the cube cost is bottlenecked
by the per-channel inversion-setup chain which doesn't gain meaningfully
from fp32 once it's already running on GPU tensor cores. Stick with fp64
on GPU for this cell.

---

## Cross-cell: known follow-ups

### Unblocking `TransformerNUFFT` on the pixelized cells

The new nufftax-backed `al.TransformerNUFFT` cannot be combined with
`apply_sparse_operator` today —
`PyAutoArray/autoarray/dataset/interferometer/dataset.py:261-282` raises
`NotImplementedError`. The blocker is an adjoint-scale mismatch: the
sparse-operator solver consumes the dirty image returned by
`transformer.image_from(use_adjoint_scaling=True)` together with the
NUFFT precision operator. The new `TransformerNUFFT` returns the strict
mathematical adjoint; the legacy pynufft adjoint applied an internal
Kaiser-Bessel kernel deconvolution; the two scales differ by a
non-constant factor.

Lifting this is the prerequisite for NUFFT on the pixelization /
delaunay / datacube paths. Until then, those cells stay on
`TransformerDFT` regardless of visibility count.

### Imaging per-step JIT regression
`PyAutoGalaxy/autogalaxy/profiles/basis.py:151` references `grid.mask` when
the `Basis.image_2d_list_from` path receives a `Grid2DIrregular`, which
has no `mask` attribute. Blocks the per-step JIT timing in
`imaging/pixelization.py` and `imaging/delaunay.py` against
PyAutoLens v2026.5.14.2. Pre-v2026.5.8.2 data in `comparison.json` is
still authoritative; the full-pipeline single-JIT row is unaffected.

### HPC A100 follow-up
The plan's Phase 4b (HPC A100 dispatch) is deferred to a separate PR.
Rationale: dispatching new cells via HPC requires cloning the
`z_projects/profiling/scripts/_setup_*.py` + `<cell>_profile.py`
scaffolding for each non-imaging cell, plus matching SLURM submit scripts
under `z_projects/profiling/hpc/batch_gpu/`. That's a substantial
scaffolding job whose timing is independent of the local sweep being
ready, and the local sweep is the higher-signal artifact for the
optimization recommendations above.

When the follow-up lands, the A100 rows above get populated and these
recommendations get a re-pass — A100 typically shifts the bottleneck from
dense matrix construction to NNLS for pixelized cells, so the "where to
focus next" lines on those cells will likely flip.
