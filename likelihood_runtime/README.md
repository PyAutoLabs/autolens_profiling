# likelihood_runtime

End-to-end full-pipeline timing of the PyAutoLens likelihood function across hardware tiers and precisions. The headline question is:

> *"How long will this likelihood take per call on this hardware?"*

Run scripts in this package — or, more commonly, the [`sweep.py`](sweep.py) driver — when you need to predict production sampler cost, compare CPU/GPU/A100 throughput on the same dataset, or measure the impact of mixed precision on a specific cell. The output of a multi-config sweep is a single `comparison.json` + `comparison.png` per cell with a row per hardware/precision config and the production cost on that row.

For *where the time goes inside the likelihood*, use the sibling package [`likelihood_breakdown/`](../likelihood_breakdown/). The two packages are deliberately disjoint so neither has to pay the other's cost.

The empirical findings from previous sweeps — per-cell timings, mp verdicts, the GPU-NUFFT regression, the upstream blockers — live in [`OPTIMIZATION_NOTES.md`](OPTIMIZATION_NOTES.md) in this directory.

## Methodology

Each script measures **one** quantity per run: the steady-state cost of the entire likelihood as a single JIT-compiled JAX program. There is no per-step decomposition. The measurement is:

1. **Eager numpy baseline** — `FitImaging` / `FitInterferometer` with `xp=np`, used as the correctness reference.
2. **Full-pipeline JIT** — `jax.jit(analysis.log_likelihood_function)(instance)` on a pytree-registered `ModelInstance`. Records lower / compile / first-call / steady-state × 10. The `steady × 10 / 10` average is the headline `full_pipeline_per_call` number.
3. **Vmap batched evaluation** — `jax.jit(jax.vmap(full_pipeline_from_params))(batched_params)` with `batch_size=3`. Records the same four phases. The reported `vmap_per_call` is `batch_time / batch_size`; speedup-vs-single-JIT is `single_jit / vmap_per_call`. Some cells skip vmap by design (datacube — wrong batching axis; delaunay — opt-in via `DELAUNAY_VMAP=1` because compilation can take 20+ minutes).
4. **Correctness assertions** — eager ≡ JIT and JIT ≡ vmap log-likelihoods agree at `rtol=1e-4`. Mp paths shift the inversion compute dtype to fp32, which loosens the rtol to `1e-3` for the JIT/vmap checks.
5. **Static memory analysis** — XLA's compiled-program memory footprint (output + temp) is recorded in `memory_mb`.

## The 6-config matrix

The sweep harness drives every in-scope cell through this matrix:

| Config | Backend | Precision | Env / Flag |
|--------|---------|-----------|------------|
| `local_cpu_fp64` | CPU | fp64 | `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu` |
| `local_cpu_mp` | CPU | mixed (fp32 inversion) | same + `--use-mixed-precision` |
| `local_gpu_fp64` | RTX 2060 (consumer) | fp64 | `JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu` |
| `local_gpu_mp` | RTX 2060 | mixed | same + `--use-mixed-precision` |
| `hpc_a100_fp64` | A100 (80 GB) | fp64 | SLURM-dispatched via `z_projects/profiling/hpc/sync` |
| `hpc_a100_mp` | A100 | mixed | same + `--use-mixed-precision` |

The `cuda,cpu` listing on GPU configs is load-bearing: the Delaunay + datacube paths use `jax.pure_callback` for Hilbert-curve mesh generation, which needs a CPU device available even when the primary platform is CUDA. Without the trailing `cpu` the callback raises `pure_callback failed to find a local CPU device`.

## What mixed precision actually means

`--use-mixed-precision` threads `SettingsInversion(use_mixed_precision=True)` through `FitImaging` / `FitInterferometer` / `AnalysisInterferometer` and into the inversion's compute path. Under JAX (`xp=jnp`) this drops the inversion's working dtype to fp32; under numpy mp is a no-op. Paths that honour the flag:

- The PSF FFT convolution in `Convolver.convolved_image_from` (linear MGE bulge path).
- The cube allocation in `mapper_util.mapping_matrix_from`.
- The noise-weighted curvature accumulation in `inversion_util.curvature_matrix_via_mapping_matrix_from`.

Paths that intentionally stay fp64:

- The NNLS reconstruction (active-set / Cholesky / `cho_solve`) — sensitive to fp32 noise on ill-conditioned source meshes.
- The log-determinant of the curvature regularisation matrix used by `figure_of_merit` — condition numbers can exceed 1e6 on fine pixelisations.
- Light profile evaluation on the over-sampled grid; only the resulting mapping matrix is downcast.

Empirical effect: mp on CPU consistently helps 7 – 23 %. On GPU it's more variable — significant on FFT-heavy MGE (27 % on the local NUFFT mge run), modest-to-zero on pixelisation/delaunay, slightly *negative* on some Delaunay configs where the fp32/fp64 cast at the sparse-operator boundary costs more than it saves. Per-cell verdicts live in `OPTIMIZATION_NOTES.md`.

## Scripts

| Script | Dataset class | Source model |
|--------|--------------|--------------|
| `imaging/mge.py` | Imaging | MGE linear bulge |
| `imaging/pixelization.py` | Imaging | RectangularAdaptImage |
| `imaging/delaunay.py` | Imaging | DelaunayBrightnessImage |
| `interferometer/mge.py` | Interferometer | MGE linear bulge; `TransformerNUFFT` (nufftax) default, `--use-dft` opt-in |
| `interferometer/pixelization.py` | Interferometer | RectangularAdaptImage + `apply_sparse_operator(use_jax=True)` |
| `interferometer/delaunay.py` | Interferometer | DelaunayBrightnessImage + `apply_sparse_operator(use_jax=True)` |
| `datacube/delaunay.py` | Datacube (34-channel cube) | DelaunayBrightnessImage per channel, shared lens model |
| `point_source/image_plane.py` | Point source | Image-plane χ² via `PointSolver` |
| `point_source/source_plane.py` | Point source | Source-plane χ² (cheaper proxy) |

## Driving the matrix — `sweep.py` and `aggregate.py`

The harness drives each cell through every config as a subprocess; the aggregator consolidates the per-config JSONs.

```bash
# Run the full local matrix (CPU + GPU × fp64 + mp) on every in-scope cell
python likelihood_runtime/sweep.py

# Restrict to certain cells
python likelihood_runtime/sweep.py --only interferometer/mge interferometer/delaunay

# Skip a backend
python likelihood_runtime/sweep.py --skip-gpu       # CPU only
python likelihood_runtime/sweep.py --skip-cpu       # GPU only

# Skip the mixed-precision rows
python likelihood_runtime/sweep.py --skip-mp

# Dry-run to inspect the planned subprocess commands
python likelihood_runtime/sweep.py --dry-run
```

The sweep writes per-config artifacts at:

```
<output-root>/<class>/<model>/local_cpu_fp64.json
<output-root>/<class>/<model>/local_cpu_fp64.png
<output-root>/<class>/<model>/local_cpu_fp64.log   (captured stdout/stderr)
```

Default `<output-root>` is `autolens_workspace_developer/jax_profiling/results/jit/` — the canonical multi-config result store across PRs.

Then aggregate:

```bash
# Aggregate every cell that has at least one local_*.json
python likelihood_runtime/aggregate.py

# One cell only
python likelihood_runtime/aggregate.py --cell interferometer/mge
```

The aggregator writes `comparison.json` + `comparison.png` into the same per-cell directory. The PNG is a log-scale grouped bar chart with one bar per (step, config); the JSON is one entry per config containing the full payload from each per-config JSON.

## How to read the output

For each cell, the headline reading order is:

1. **`comparison.png`** — log-scale view of full_pipeline_per_call per config. The slope from CPU → consumer GPU → A100 tells you what hardware tier this cell needs.
2. **`comparison.json` → `configs.<name>.full_pipeline_per_call`** — the production cost a sampler will pay per likelihood evaluation on that hardware.
3. **`vmap.per_call` + `vmap.speedup_vs_single_jit`** — the cheapest throughput lever. For cells where vmap helps (typically MGE / non-iterative inversions), batching is the right knob; for cells where vmap ≤ 1× (sparse pixelisation, datacube), reach for data-parallel processes instead.
4. **`memory_mb.temp`** — XLA's compiled-program working memory. Compare against your hardware budget (RTX 2060 is 6 GB; A100 is 80 GB) before increasing vmap batch size.
5. **mp verdict** — `(fp64 - mp) / fp64`. A solid 10 %+ win, with the log-likelihood unchanged at `rtol=1e-3`, means "default to mp on this hardware tier".

For headline cross-cell insights and the running list of where each cell sits in terms of "where to optimize next", read [`OPTIMIZATION_NOTES.md`](OPTIMIZATION_NOTES.md).

## Auto-simulation

If `dataset/<class>/<instrument>/` is missing, the script shells out to `simulators/<dataset_type>.py --instrument <name>` and waits for the dataset to land before continuing. Datasets are seeded — re-running the simulator produces bit-identical files. The simulator INSTRUMENTS dict is the single source of truth that the runtime scripts import directly:

```python
from simulators.interferometer import INSTRUMENTS
```

Currently configured presets:

| Class | Presets |
|-------|---------|
| `imaging` | euclid, hst, jwst, ao |
| `interferometer` | sma (190 vis), alma (1 M vis), alma_high (10 M vis) |
| `point_source` | simple |

## When to choose runtime vs breakdown

| Question | Package |
|----------|---------|
| "How long will my A100 sampler run take per likelihood call?" | **runtime** |
| "Does mixed precision actually save time on this cell?" | **runtime** |
| "How does production cost change between consumer GPU and A100?" | **runtime** |
| "Where should I focus PyAutoLens optimisation work for this cell?" | breakdown |
| "Which step fuses cleanly under XLA and which doesn't?" | breakdown (compare against runtime's `full_pipeline_per_call`) |
