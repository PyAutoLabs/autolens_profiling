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
| Local GPU (RTX 2060) fp64 + mp   | ⏸ pending user-launched run (see command below) |
| HPC A100 fp64 + mp               | ⏸ separate follow-up PR |
| Imaging cells fresh CPU/GPU      | ⚠ blocked by upstream `Grid2DIrregular.mask` bug — table rows show the pre-existing v2026.5.8.2 / v2026.5.14.2 data |

### Running the local GPU sweep yourself

When you're away from the laptop (GPU runs will pin the GPU and slow it
down for desktop use):

```bash
source /home/jammy/Code/PyAutoLabs-wt/likelihood-multiconfig-sweep/activate.sh
cd /home/jammy/Code/PyAutoLabs-wt/likelihood-multiconfig-sweep/autolens_profiling

# 1. GPU-only sweep across all 6 in-scope cells (datacube is the heaviest;
#    estimate ~30 min total at RTX 2060 fp64+mp throughput).
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda /home/jammy/venv/PyAutoGPU/bin/python \
    scripts/sweep_likelihood.py --skip-cpu

# 2. Re-aggregate so comparison.{json,png} include the GPU rows.
/home/jammy/venv/PyAutoGPU/bin/python scripts/aggregate_sweep.py
```

After both finish, the GPU rows will appear in the tables below the
next time someone re-renders the doc (the tables are hand-maintained;
the live source of truth is the `comparison.json` per cell).


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
| local_gpu_fp64    | NUFFT       | _GPU sweep_   | _GPU sweep_    |
| local_gpu_mp      | NUFFT       | _GPU sweep_   | _GPU sweep_    |
| local_gpu_fp64    | DFT (--use-dft) | _follow-up_   | _follow-up_ |
| hpc_a100_fp64     | NUFFT       | —             | —              |
| hpc_a100_mp       | NUFFT       | —             | —              |

For reference, the historical SMA baseline with `TransformerDFT` (pre-PR)
was **34 ms** full pipeline on CPU with vmap speedup 2.1×.

**Key findings so far**

- At SMA's 190 visibilities, NUFFT on CPU is **~8× slower than the
  historical DFT baseline** (261 ms vs 34 ms). NUFFT's interpolation
  kernels + oversampling add fixed cost that doesn't amortize at small
  visibility counts. **The NUFFT swap pays off only at large visibility
  counts (ALMA-class)** — keep the SMA default on `--use-dft` for any
  production sampling on SMA-shaped data.
- **mp gives a real ~23 % CPU win** (261 → 202 ms). The FFT side benefits
  visibly from fp32 here. Use mp on CPU NUFFT runs.
- **vmap is neutral** on CPU NUFFT (1.0–1.1×). The historical DFT path
  showed 2× — the loss is intrinsic to nufftax's CPU lowering.
- **Memory regression** — NUFFT eats 480 MB of XLA temp vs DFT's 17 MB
  on the same dataset (28× more memory). RTX 2060 (6 GB) is approaching
  saturation with vmap batch=3; A100 (80 GB) won't notice but it's a
  flag for batch-size budgeting on consumer hardware.

**Where to focus next**

1. **Profile against the ALMA `INSTRUMENTS` preset** in the script
   (already wired but never run). That's where NUFFT vs DFT becomes
   meaningful. On SMA the swap is a regression.
2. **Add ALMA dataset under autolens_profiling/dataset/interferometer/alma/**
   so the existing `instrument = "sma"` line in the script can be flipped
   to `"alma"` without manual regeneration.
3. **GPU NUFFT investigation** — the validation single-run earlier showed
   1.4 s on GPU. Combined with the 8× CPU regression, this suggests
   nufftax's GPU lowering on consumer hardware (RTX 2060) is currently
   unoptimised. May be a knob (interpolation `eps`, oversampling factor)
   worth exploring.

**mp verdict** — **use it** on CPU runs. 23 % CPU win, no measurable
correctness drift (log-likelihood agrees with fp64 baseline at fp64
precision on this regime). GPU and A100 to be measured.

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
| local_gpu_fp64    | _GPU sweep_   | _GPU sweep_     | _GPU sweep_  |
| local_gpu_mp      | _GPU sweep_   | _GPU sweep_     | _GPU sweep_  |
| hpc_a100_fp64     | —             | —               | —            |
| hpc_a100_mp       | —             | —               | —            |

**Where to focus next**

1. **vmap is actively a regression** on this cell (0.7× CPU). The sparse
   precision-operator solve's matrix dimensions don't batch cleanly along
   the parameter axis. Do not use vmap here; rely on data-parallel
   throughput instead.
2. **The XLA temp footprint is 193 MB** on CPU. On GPU expect that to
   dominate the 6 GB RTX 2060 budget with any meaningful batching;
   monitor `nvidia_smi` field in the GPU JSONs.
3. **Sparse precompute cost is amortised across all per-fit calls** — the
   753 ms per-call number is the steady-state production cost a sampler
   pays. The one-time precompute (~few seconds at SMA scale) is excluded
   from the per-call measurement.

**mp verdict** — modest CPU win (~7 %). Worth using; cheap to opt in.
GPU mp to be measured.

---

## interferometer/delaunay
*Delaunay pixelization on visibilities. 1000 mesh vertices (Hilbert
sampling). SMA dataset. The `apply_sparse_operator` path is the
production case.*

| Config            | full pipeline | vmap (batch=3)   | log_evidence |
|-------------------|---------------|------------------|--------------|
| local_cpu_fp64    | 881 ms        | _vmap intentionally skipped_ | —    |
| local_cpu_mp      | 788 ms        | _vmap intentionally skipped_ | —    |
| local_gpu_fp64    | _GPU sweep_   | _GPU sweep_      | _GPU sweep_  |
| local_gpu_mp      | _GPU sweep_   | _GPU sweep_      | _GPU sweep_  |
| hpc_a100_fp64     | —             | —                | —            |
| hpc_a100_mp       | —             | —                | —            |

vmap is intentionally skipped on this cell — opt in with `DELAUNAY_VMAP=1`
per the script's design. Delaunay mesh construction doesn't batch
cleanly along the parameter axis.

**Where to focus next**

1. The **per-step breakdown is not yet measured for this cell** (the
   interferometer/delaunay script per-step-JITs the `Inversion setup`
   step as one combined block "steps 5-8 incl. NUFFT"). Re-running with
   the per-step decomposition is a follow-up — the combined block
   currently swallows ~300 ms of the ~881 ms per call.
2. **mp gives a real ~11 % CPU win** (881 → 788 ms). Same pattern as the
   other sparse cells; use mp.
3. **GPU/A100 numbers** are critical here — the sparse precision-matrix
   solve is the kind of dense-linear-algebra workload A100 demolishes,
   so we expect this cell to be near A100-optimal once the HPC sweep
   lands. The local RTX 2060 numbers will tell us how much of that
   benefit is reachable on consumer hardware.

**mp verdict** — **use it**. 11 % CPU win, identical log-evidence to fp64.

---

## datacube/delaunay
*34-channel cube reusing the SMA interferometer dataset. Delaunay source
pixelization per channel, shared lens model. The single heaviest cell
in this profile family.*

The JSON step labels say "incl. NUFFT" — that's accurate: while the
dataset is built with `TransformerDFT`, `apply_sparse_operator(use_jax=True)`
precomputes a NUFFT precision matrix internally, which is what the
per-channel curvature assembly actually goes through.

| Config            | step-by-step total (cube) | mp vs fp64 |
|-------------------|---------------------------|------------|
| local_cpu_fp64    | 197 s                     | —          |
| local_cpu_mp      | 165 s                     | **−16 %**  |
| local_gpu_fp64    | _GPU sweep_               | _GPU sweep_ |
| local_gpu_mp      | _GPU sweep_               | _GPU sweep_ |
| hpc_a100_fp64     | —                          | —           |
| hpc_a100_mp       | —                          | —           |

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

**mp verdict** — **use it**. 16 % cube-level CPU win, driven primarily
by the 31 % fp32 win on curvature-matrix construction. GPU mp to be
measured but expectation is similar or larger (GPU fp32 throughput is
typically 2× fp64).

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
