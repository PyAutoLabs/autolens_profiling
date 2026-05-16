# likelihood/datacube

JAX JIT profiling for the PyAutoLens **datacube** likelihood function — fitting an N-channel cube of interferometer observations (e.g. ALMA velocity / frequency channels) that share a single lens model, with each channel reconstructing its own Delaunay-pixelized source.

## Channel-invariant vs channel-variant split

The key new ingredient relative to the single-channel interferometer path is the **channel-invariant vs channel-variant** decomposition: most steps are computed once for the whole cube (shared lens model, shared mesh, shared mask), while only a few steps recur per channel.

| Step | Channel-invariant? | Computed |
|------|--------------------|----------|
| 1. Ray-trace data grid | yes | once for the cube |
| 2. Ray-trace mesh grid | yes | once for the cube |
| 3. Inversion setup (border + mapper + NUFFT) | NUFFT depends on `uv_wavelengths` | once per channel |
| 4. Data vector D | per channel | once per channel |
| 5. Curvature matrix F | per channel | once per channel |
| 6. Regularization matrix H | yes | once for the cube |
| 7. Reconstruction (NNLS) | per channel | once per channel |
| 8. Mapped recon + log-evidence | per channel | once per channel |

The cube total is:

```
cube_cost = sum(channel_invariant_costs) + N_channels * sum(channel_variant_costs)
```

That number quantifies how much a future "shared `Lᵀ W̃ L`" optimisation would save: moving the curvature matrix from per-channel to shared would subtract `(N − 1) * curvature_matrix_cost` from the cube total. The profiling script reports this number directly.

## Scripts

| Script | What it profiles |
|--------|------------------|
| [`delaunay.py`](./delaunay.py) | Step-by-step JIT profiling of an N-channel datacube with shared lens model and per-channel Delaunay source reconstruction. Mirrors the per-step structure of [`../interferometer/delaunay.py`](../interferometer/delaunay.py). |

## Default dataset

`dataset/interferometer/sma/` — the same SMA-like mock used by the single-channel interferometer scripts, **loaded N times as a 4-channel cube**. Every channel has identical visibilities, noise map, and uv_wavelengths — the point here is timing, not science. The N-channel cube log-evidence is exactly `N × single-channel log-evidence`, which makes the regression assertion trivial.

For a realistic per-channel-distinct cube, point the loader at the workspace simulator output at `autolens_workspace/dataset/interferometer/datacube/sim_simple/`. The JIT-cost taxonomy doesn't change — it's a function of which arrays are loop-variables in `FitInterferometer`, not the data values themselves.

## Headline run-times (populated by Phase 4)

| Script | Dataset | N channels | CPU | Laptop GPU | A100 |
|--------|---------|------------|-----|------------|------|
| `delaunay.py` | SMA × 4 | 4 | _populated_ | _populated_ | _populated_ |

Numbers are the **steady-state per-call cost** (single-JIT, post-warmup), in milliseconds. Phase 4's dashboard auto-fills this from the latest `*_summary_v<version>.json` artifacts under `results/likelihood/datacube/`.

## Output

The script writes:

```
results/likelihood/datacube/delaunay_likelihood_summary_<instrument>_v<al.__version__>.json
results/likelihood/datacube/delaunay_likelihood_summary_<instrument>_v<al.__version__>.png
```
