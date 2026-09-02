# autolens_profiling

Profiling and run-time tracking for [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) likelihood functions, simulators, and samplers across CPU, laptop GPU, and HPC GPU.

> **Rulings of record live in [PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex)**
> (`projects.yaml`, row `inference_programme`) — see [`CORTEX.md`](CORTEX.md).

## Vision

This repository is the single home for PyAutoLens performance measurement. It exists so that the run-times that matter for science — fitting a real lens, simulating an Euclid-resolution dataset, sampling a model with Nautilus — are visible, reproducible, and versioned across PyAutoLens releases.

**What is profiled:**

- **Likelihood functions** — imaging, interferometer, point-source, and datacube paths, across the MGE, pixelization, and Delaunay model compositions used in real science cases.
- **Mesh kernels** — full mapper-table profiling for Delaunay and Sibson natural-neighbour `DelaunayNN`, including static-cap scaling and split regularization.
- **Simulators** — run-time tracking for the imaging, interferometer, point-source, cluster, group, and multi-plane simulators.
- **Searches / samplers** — sampler-level profiling, starting with Nautilus. Other samplers (Dynesty, Emcee, BlackJAX, NumPyro, LBFGS, PocoMC) follow in later sweeps.
- **Numerical hazards** — saturations, non-finite gradients, backend divergence, and scale-dependent conditioning mechanisms that shape sampler behavior.

**Hardware tiers covered:**

- CPU (single-machine, numpy backend).
- Laptop GPU (consumer-class, JAX backend).
- HPC GPU (A100 and similar, JAX backend).

**Dataset framing:**

Results are framed by **astronomy instrument** (HST, Euclid, JWST, …) rather than by raw pixel counts. Pixel counts are recorded too, but the headline numbers a reader sees first are the ones that map onto a real observing programme.

## Latest run-times

<!-- BEGIN auto-table:headline -->

**Likelihood runtime** — full-pipeline per-call cost per cell × config:

| Cell | local_cpu_fp64 | local_cpu_mp | local_cpu_fp64_sparse | local_cpu_mp_sparse | PreOptimizationTimes |
|---|---|---|---|---|---|
| `datacube/delaunay/sma` | — | — | — | — | — |
| `imaging/delaunay/hst` | 16.73 s | 17.84 s | 4.14 s | 4.49 s | 4.49 s |
| `imaging/delaunay/jwst` | 48.81 s | 22.43 s | 10.42 s | 14.05 s | 14.05 s |
| `imaging/mge/ao` | 3.11 s | 5.71 s | — | — | 5.71 s |
| `imaging/mge/hst` | 117.7 ms | 164.3 ms | 256.1 ms | 140.5 ms | 140.5 ms |
| `imaging/mge/jwst` | 716.2 ms | 678.5 ms | 387.3 ms | 488.7 ms | 488.7 ms |
| `imaging/pixelization/hst` | 13.72 s | 14.78 s | 5.79 s | 5.25 s | 5.25 s |
| `imaging/pixelization/jwst` | 21.78 s | 43.58 s | 9.57 s | 9.42 s | 9.42 s |
| `interferometer/delaunay/alma` | **GPU-only** | 6.51 s | — | — | 6.51 s |
| `interferometer/delaunay/sma` | 2.58 s | 3.34 s | — | — | 3.34 s |
| `interferometer/mge/sma` | 230.7 ms | 231.5 ms | — | — | 231.5 ms |
| `interferometer/pixelization/sma` | 2.04 s | 2.39 s | — | — | 2.39 s |

**Likelihood breakdown** — latest per-step decompositions:

| Cell | Instrument | Platform | Inversion path | Step-sum total | PyAutoLens version |
|------|------------|----------|----------------|----------------|--------------------|
| `cluster/image_plane` | — | local_cpu_fp64 | dense (mapping) | 10.25 s | v2026.7.23.1 |
| `cluster/source_plane` | — | local_cpu_fp64 | dense (mapping) | 4.5 ms | v2026.7.23.1 |
| `datacube/delaunay` | alma_high | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/delaunay` | alma_high | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | hpc_a100_fp64 | dense (mapping) | — | v2026.7.6.649 |
| `datacube/inversion` | alma_high | hpc_a100_mp | dense (mapping) | — | v2026.7.6.649 |
| `imaging/delaunay` | hst | local_cpu_fp64 | dense (mapping) | 10.07 s | v2026.7.6.649 |
| `imaging/delaunay` | hst | local_cpu_fp64 | sparse (w-tilde) | 8.81 s | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_fp64 | dense (mapping) | 96.6 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_fp64 | sparse (w-tilde) | 98.0 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_mp | dense (mapping) | 96.8 ms | v2026.7.6.649 |
| `imaging/delaunay` | hst | hpc_a100_mp | sparse (w-tilde) | 95.5 ms | v2026.7.6.649 |
| `imaging/delaunay_numba` | euclid | local_cpu_fp64 | sparse (numba) | 1.19 s | v2026.8.17.1 |
| `imaging/delaunay_numba` | hst | local_cpu_fp64 | sparse (numba) | 482.4 ms | v2026.8.17.1 |
| `imaging/mge` | hst | local_cpu_fp64 | dense (mapping) | 179.5 ms | v2026.7.6.649 |
| `imaging/mge` | hst | hpc_a100_fp64 | dense (mapping) | 7.8 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | local_cpu_fp64 | dense (mapping) | 8.65 s | v2026.7.6.649 |
| `imaging/pixelization` | hst | local_cpu_fp64 | sparse (w-tilde) | 10.17 s | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_fp64 | dense (mapping) | 57.6 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_fp64 | sparse (w-tilde) | 57.9 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_mp | dense (mapping) | 56.4 ms | v2026.7.6.649 |
| `imaging/pixelization` | hst | hpc_a100_mp | sparse (w-tilde) | 55.5 ms | v2026.7.6.649 |
| `imaging/pixelization_numba` | euclid | local_cpu_fp64 | sparse (numba) | 124.3 ms | v2026.8.17.1 |
| `imaging/pixelization_numba` | hst | local_cpu_fp64 | sparse (numba) | 304.5 ms | v2026.8.17.1 |
<!-- END auto-table:headline -->

The tables above are auto-generated by `scripts/misc/tooling/build_readme.py` from the artifacts under [`results/`](./results/README.md) — never edit them by hand; run `python scripts/misc/tooling/build_readme.py` after a profiling run and commit the result (CI checks idempotence via `--check`). Narrative context — per-cell "where to optimize next" recommendations and the mp-vs-fp64 verdicts — lives in [`scripts/misc/likelihood_runtime/OPTIMIZATION_NOTES.md`](./scripts/misc/likelihood_runtime/OPTIMIZATION_NOTES.md).

**PreOptimizationTimes** is the named baseline the upcoming optimization work is measured against: a frozen snapshot of the full campaign (laptop CPU, HPC CPU, HPC A100 × fp64/mp) under `results/baselines/PreOptimizationTimes/`, rendered as a baseline column in the dashboard once populated. The convention is defined in [`results/notes/design_lock_in.md`](./results/notes/design_lock_in.md).

(Historical multi-config sweeps up to 2026-07 were committed under [`autolens_workspace_developer/jax_profiling/results/jit/`](https://github.com/PyAutoLabs/autolens_workspace_developer/tree/main/jax_profiling/results/jit); sweeps now write in-repo to `results/runtime/` by default.)

## JAX gradients and compile time

Gradient-based search profiling now lives here: the multi-start gradient
optimisers are profiled as first-class search cells under
`scripts/<dataset>/searches/multi_start_{adam,prodigy}/`. Exploratory gradient
work continues in
[`autolens_workspace_developer/jax_profiling/gradient/`](https://github.com/PyAutoLabs/autolens_workspace_developer/tree/main/jax_profiling/gradient).

**Compile time** is profiled separately from run time, because for JAX
likelihoods XLA compilation is a first-class cost in its own right —
`scripts/misc/jax_compile/` measures trace / compile / first-call / steady-state
separately per likelihood × transform. Standing conclusions:

- **Settings suffice** — the persistent compilation cache and `--xla_gpu_autotune_level=0`
  (both shipped as autonerves defaults) take the worst measured first fit from
  ~70 min to ~35 s. Never restructure a likelihood or sampler for compile time
  ([`scripts/misc/jax_compile/README.md`](./scripts/misc/jax_compile/README.md)).
- **`af.MultiStartProdigy` compile is a non-problem** on MGE and every pixelized
  mesh (rectangular / KNN / Delaunay) — ≤ 75 s cold, ≤ 2 s warm on a 32-core
  node. The Delaunay family, however, **never hits the persistent cache** and
  pays full compile in every process, and rectangular's `batch_size=4` is
  load-bearing for memory (~9.2 GB per start).
  Findings: [`results/notes/multistart_prodigy_compile_census.md`](./results/notes/multistart_prodigy_compile_census.md).
- **Multi-band `FactorGraphModel` + MultiStartProdigy compile is fixed at the
  source** (PyAutoFit#1430): `batch_size` now sweeps vmapped chunks from a
  Python loop instead of an in-XLA `lax.map` scan, and the broad-start filter
  is jitted. Cold multi-band fit on the 1-core laptop: intractable → ~6.5 min
  (CPU and laptop GPU); warm ~2–3 min, bit-identical numerics. The scan
  explosion is CPU-backend-specific — the GPU pipeline compiles it fine.
  Findings: [`results/notes/multiband_pyloop_productized.md`](./results/notes/multiband_pyloop_productized.md).

## How to read this repo

Performance profiling scripts write two timing artifact shapes under `results/`
(full reference: [`results/README.md`](./results/README.md)):

```
# Versioned summaries — standalone runs; history retained side-by-side
results/<section>/<subfolder>/<cell>_<purpose>_<instrument>_v<YYYY>.<M>.<D>.<PATCH>[_sparse].{json,png}

# Per-config sweeps — sweep.py + aggregate.py; latest sweep per cell
results/runtime/<class>/<model>[/<instrument>]/<config_name>[_sparse].{json,png,log} + comparison.{json,png}
```

The version string matches the PyAutoLens release that produced the numbers (e.g. `v2026.5.29.4`). The JSON carries structured timings; the PNG is the at-a-glance plot. Cross-release **trend** questions read the versioned summaries; cross-hardware **comparison** questions read `comparison.json`.

Numerical hazards use a third, semantic shape under `results/hazards/`: stable
finding IDs, typed measurements, source anchors, reproducer plots, and a
consumer-facing `hazards_index.json`. Those records are re-verified by behavior,
not versioned by filename.

## Section index

Scripts are laid out **dataset-first, task-second**: `scripts/<dataset>/<task>/<model>.py`
(`imaging` / `interferometer` / `point_source` / `multi_dataset` / `cluster`), mirroring the
`autolens_workspace*` repos. Each task's shared drivers, framework and narrative README (with the
auto-tables) live under `scripts/misc/<task>/`; dataset-agnostic tooling lives under `scripts/misc/`.

Beside those dataset-first families sits a second, **dataset-free** axis:
[`scripts/lens/`](./scripts/lens/README.md) profiles a single **library component** — one function,
one grid, one set of fiducial parameters — rather than a pipeline. A dataset is loaded only to build
a realistic grid; nothing about the data enters the measurement. It answers *"what does this piece of
the lensing calculation cost per call, and where inside it does the time go?"*, which is the evidence
library-level optimisation work needs and a pipeline breakdown cannot give. Today that is
[`lens/deflections/`](./scripts/lens/deflections/README.md) (deflection angles per mass profile);
`convergence/`, `potential/` and `shear/` follow the same shape.

| Task (`scripts/<dataset>/<task>/` + shared home) | Contents |
|--------|----------|
| `likelihood_runtime/` · [README](./scripts/misc/likelihood_runtime/README.md) | Full-pipeline JIT only, driven by `scripts/misc/likelihood_runtime/sweep.py` across CPU/GPU/A100 × fp64/mp. *How long will this likelihood take on this hardware?* |
| `likelihood_breakdown/` · [README](./scripts/misc/likelihood_breakdown/README.md) | Per-step JIT decomposition. Single config. *Where does time go inside the likelihood?* |
| `searches/<sampler>/` · [README](./scripts/misc/searches/README.md) | Sampler / search profiling, Nautilus first. |
| `latent/` · [README](./scripts/misc/latent/README.md) | Latent-variable profiling. |
| `quick_update/` · [README](./scripts/misc/quick_update/README.md) | Fast incremental re-profiling helpers. |
| [`scripts/misc/jax_compile/`](./scripts/misc/jax_compile/README.md) | JAX/XLA **compile-time** profiling — trace / compile / first-call / steady split per likelihood × transform. *How long before this fit starts running?* |
| [`scripts/misc/vram/`](./scripts/misc/vram/README.md) | GPU memory profiling + the per-cell vmap batch-size table for the A100. |
| [`scripts/misc/delaunay_nn/`](./scripts/misc/delaunay_nn/README.md) | DelaunayNN full-mapper runtime and fixed-shape cap scaling. |
| [`scripts/misc/hazards/`](./scripts/misc/hazards/README.md) | Numerical-hazard profiling — saturations, non-finite gradients, backend divergence, and conditioning mechanisms. |
| [`scripts/misc/simulators/`](./scripts/misc/simulators/README.md) | Run-time tracking for the PyAutoLens simulators. |
| [`scripts/misc/pipeline_resume/`](./scripts/misc/pipeline_resume/README.md) | SLaM pipeline resume overhead — the wall time a re-run pays per completed stage. |
| [`scripts/lens/`](./scripts/lens/README.md) | **Library-component profiling** (dataset-free axis). [`deflections/`](./scripts/lens/deflections/README.md) — per-call deflection-angle cost per mass profile on the numpy CPU path, pinned on the values it computes. |
| [`instruments/`](./instruments/README.md) | Instrument presets (pixel scale, shape) that frame every result. |
| [`hpc/`](./hpc/README.md) | SLURM submit scripts for the RAL HPC (A100 rows of the sweep matrix). |
| [`results/`](./results/README.md) | JSON + PNG artifacts written by the above scripts; named baselines. |

## Roadmap

This repo is being built in phases (bootstrap history now archived in `PyAutoMind`).

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repo bootstrap | ✓ shipped |
| 1 | Mirror JIT likelihood profiling scripts + per-section READMEs | ✓ shipped |
| 2 | Mirror simulator profiling scripts + run-time tracking | ✓ shipped |
| 3 | Nautilus profiling, design for sampler expansion | ✓ shipped |
| 4 | Top-level + per-section README dashboard with instrument framing | ✓ shipped |
| 5 | GitHub Actions for lint + profile re-runs + README refresh | ✓ shipped (`lint.yml` per-PR; `profile.yml` manual/on-release) |
| 6 | Design lock-in + results/dashboard groundwork ([#52](https://github.com/PyAutoLabs/autolens_profiling/issues/52)) | in progress |
| 7 | **PreOptimizationTimes** baseline campaign (vram-first, then runtime + breakdown) | ✓ shipped (runtime [#56](https://github.com/PyAutoLabs/autolens_profiling/issues/56); breakdown + dashboard [#59](https://github.com/PyAutoLabs/autolens_profiling/issues/59); laptop-GPU legs extend in a later re-run) |

### Future enhancements (Phase 4 follow-ups)

Dashboards can grow in many directions. The list below captures candidate improvements that fit the "profiling and run-times" theme; none of them block the current dashboard from being useful.

- **Regression-watch indicator** — colour or arrow per cell showing whether the latest cost regressed (>5%) or improved versus the previous PyAutoLens release. Needs the second-latest version per axis kept alongside the latest. Trivial to add to `scripts/misc/tooling/build_readme.py`.
- **Per-axis version-history PNGs** — small inline plot of run-time vs PyAutoLens release version, generated from the JSON artifacts (reusing the `_developer/jax_profiling/results/jit/.../*_v<version>.png` generator). Embeds nicely above each section table.
- **Plotly-rendered interactive timeline** — hostable on GitHub Pages once the static dashboard stabilises; lets readers hover/filter across instrument × model × release.
- **Flamegraph captures** — alongside the headline timing numbers, store a flamegraph per instrument × model for the most recent release.
- **Hardware-tier columns** — extend `scripts/misc/tooling/build_readme.py` table renderers to show CPU / laptop GPU / HPC GPU as separate columns once result artifacts encode the hardware label (filename suffix or JSON `"hardware"` field).
- **Archive old versions** — once a script has >6 minor releases of artifacts, move the older ones to `results/archive/` so the latest views stay uncluttered.

## Related repos

- [`PyAutoLabs/PyAutoLens`](https://github.com/PyAutoLabs/PyAutoLens) — the library being profiled.
- [`PyAutoLabs/autolens_workspace`](https://github.com/PyAutoLabs/autolens_workspace) — user-facing science scripts and tutorials.
- [`PyAutoLabs/autolens_workspace_developer`](https://github.com/PyAutoLabs/autolens_workspace_developer) — the developer workspace this repo's scripts were migrated from; still hosts the pre-2026-07 sweep history and the gradient-profiling work.
- [`Jammy2211/autolens_colab_profiling`](https://github.com/Jammy2211/autolens_colab_profiling) — sibling repo, Colab-specific scope. Not yet migrated to PyAutoLabs.

## Package vs scripts

This repo is a **collection of standalone profiling scripts**, not an installable Python package. There is no `pyproject.toml`. Run scripts from the repo root.

Scripts follow the JIT conventions documented in `autolens_workspace_developer/CLAUDE.md`:

- Extract `.array` from autoarray types before crossing the `jax.jit` boundary (autoarray types are not JAX pytrees as inputs).
- Pass `xp=jnp` through PyAutoLens / PyAutoGalaxy / PyAutoArray functions to select the JAX backend.

## Community & support

- **Slack** — [PyAutoLens workspace](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg) for questions.
- **Issues** — file profiling bugs and feature requests on this repo's [issue tracker](https://github.com/PyAutoLabs/autolens_profiling/issues).

<sub><i><a href="https://open.spotify.com/track/7c584s9RZQzkJDoC08VDJB">i just know that it get better with time</a></i></sub>
