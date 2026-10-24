# scripts/misc

Shared tools that are not tied to one dataset or model. The profiling cells
themselves live under `scripts/<dataset>/<model>/`; the folders here hold the
helpers they share, the repo's own tooling, and a few standalone studies.

Each folder has its own README with the details. Results from a folder here go
to `results/misc/<folder>/` unless its README says otherwise.

| Folder | What it holds | Writes to |
|--------|---------------|-----------|
| [`likelihood_runtime/`](likelihood_runtime/README.md) | The runtime sweep and its aggregator | `results/runtime/`, `results/misc/likelihood_runtime/` |
| [`likelihood_breakdown/`](likelihood_breakdown/README.md) | Step-by-step timing helpers the breakdown cells share | `results/misc/likelihood_breakdown/` |
| [`hazards/`](hazards/README.md) | Checks for numerical problems in a likelihood | `results/misc/hazards/` |
| [`jax_compile/`](jax_compile/README.md) | JAX compile-time probes and pins | `scripts/misc/jax_compile/results/` |
| [`vram/`](vram/README.md) | GPU memory batch sizes for each cell | nothing (a config table) |
| [`wall/`](wall/README.md) | Step rates and the HPC `--time` check | nothing (a CI check) |
| [`simulators/`](simulators/README.md) | Timing of the mock-data simulators | `results/misc/simulators/` |
| [`latent/`](latent/README.md) | Timing of each latent variable | `results/misc/latent/` |
| [`pipeline_resume/`](pipeline_resume/README.md) | Cost of resuming a finished pipeline | `results/misc/pipeline_resume/` |
| [`delaunay_nn/`](delaunay_nn/README.md) | Natural-neighbour mesh timing | `results/misc/delaunay_nn/` |
| [`nnls_warm_start/`](nnls_warm_start/README.md) | NNLS warm-start experiment | `results/misc/nnls_warm_start/` |
| [`numba_interferometer/`](numba_interferometer/README.md) | The old numba interferometer likelihood, for comparison | see its README |
| [`quick_update/`](quick_update/README.md) | Notes for the fast re-profiling scripts | `results/misc/quick_update/` |
| [`cell_builders/`](cell_builders/README.md) | Model and dataset setup shared by cells | nothing |
| [`cells/`](cells/README.md) | Tests for cells that have no helper folder here | nothing |
| [`tooling/`](tooling/README.md) | Builders and checks for the catalogue, dashboard, READMEs and wiki | `dashboard/`, `catalogue/`, READMEs |

## Tests

Each folder keeps its tests in its own `test/` folder. Run them all from the
repo root:

```bash
pytest scripts/misc/*/test -q
```

or one folder, for example `pytest scripts/misc/hazards/test -q`.
