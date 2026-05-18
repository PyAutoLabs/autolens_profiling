# autolens_profiling

Profiling and run-time tracking for [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) likelihood functions, simulators, and samplers across CPU, laptop GPU, and HPC GPU.

## Vision

This repository is the single home for PyAutoLens performance measurement. It exists so that the run-times that matter for science — fitting a real lens, simulating an Euclid-resolution dataset, sampling a model with Nautilus — are visible, reproducible, and versioned across PyAutoLens releases.

**What is profiled:**

- **Likelihood functions** — imaging, interferometer, point-source, and datacube paths, across the MGE, pixelization, and Delaunay model compositions used in real science cases.
- **Simulators** — run-time tracking for the imaging, interferometer, point-source, cluster, group, and multi-plane simulators.
- **Searches / samplers** — sampler-level profiling, starting with Nautilus. Other samplers (Dynesty, Emcee, BlackJAX, NumPyro, NSS, LBFGS, PocoMC) follow in later sweeps.

**Hardware tiers covered:**

- CPU (single-machine, numpy backend).
- Laptop GPU (consumer-class, JAX backend).
- HPC GPU (A100 and similar, JAX backend).

**Dataset framing:**

Results are framed by **astronomy instrument** (HST, Euclid, JWST, …) rather than by raw pixel counts. Pixel counts are recorded too, but the headline numbers a reader sees first are the ones that map onto a real observing programme.

## Latest run-times

The table below is auto-generated from the latest versioned artifacts under `results/`. Each row is the latest steady-state per-call cost for a likelihood path at a given instrument; numbers refresh whenever the producing scripts are rerun and committed. Hardware tier is **CPU only** today — laptop GPU and HPC GPU columns will land once `results/**` artifacts are tagged with a hardware label.

<!-- BEGIN auto-table:headline -->
| Section | Script | Instrument | Latest single-JIT per-call | PyAutoLens version |
|---------|--------|------------|----------------------------|--------------------|
| likelihood/datacube | `delaunay.py` | hannah | — | v2026.5.14.2 |
| likelihood/imaging | `delaunay.py` | hst | 833.4 ms | v2026.5.14.2 |
| likelihood/imaging | `mge.py` | hst | 41.6 ms | v2026.5.14.2 |
| likelihood/imaging | `pixelization.py` | hst | 782.3 ms | v2026.5.14.2 |
| likelihood/interferometer | `delaunay.py` | sma | 154.5 ms | v2026.5.14.2 |
| likelihood/interferometer | `mge.py` | sma | 33.6 ms | v2026.5.14.2 |
| likelihood/interferometer | `pixelization.py` | sma | 113.6 ms | v2026.5.14.2 |
| likelihood/point_source | `image_plane.py` | — | 22.5 ms | v2026.5.14.2 |
| likelihood/point_source | `source_plane.py` | — | 691 μs | v2026.5.14.2 |
<!-- END auto-table:headline -->

(Generator: `scripts/build_readme.py`. Run `python scripts/build_readme.py` after producing new artifacts to refresh; `--check` exits non-zero in CI if it would change anything.)

## JAX gradients — currently out of scope

Gradient profiling (`jax.grad` of the likelihood, autodiff-based optimisers) is **not yet** part of this repo. It is tracked in [`PyAutoLabs/autolens_workspace_developer/jax_profiling/gradient/`](https://github.com/PyAutoLabs/autolens_workspace_developer/tree/main/jax_profiling/gradient) and will fold into this repo in a future phase once the gradient story stabilises.

## How to read this repo

Each profiling script writes a **versioned artifact pair** under `results/`:

```
results/<section>/<subfolder>/<profile_name>_summary_v<YYYY>.<M>.<D>.<PATCH>.json
results/<section>/<subfolder>/<profile_name>_summary_v<YYYY>.<M>.<D>.<PATCH>.png
```

The version string matches the PyAutoLens release that produced the numbers (e.g. `v2026.5.1.4`). Older versions are retained alongside newer ones, so trends across releases stay inspectable. The JSON carries structured timings; the PNG is the at-a-glance plot.

Examples that already exist in the source-of-truth repo:

- `imaging_summary_v2026.5.1.4.json` / `.png`
- `point_source_summary_v2026.5.1.4.json`
- `delaunay_sparse_cpu_likelihood_summary_hst_v2026.5.1.4.json`

## Section index

| Folder | Contents |
|--------|----------|
| [`likelihood/`](./likelihood/README.md) | Likelihood JIT profiling — imaging, interferometer, point-source, datacube. |
| [`simulators/`](./simulators/README.md) | Run-time tracking for the PyAutoLens simulators. |
| [`searches/`](./searches/README.md) | Sampler / search profiling, Nautilus first. |
| [`results/`](./results/README.md) | Versioned JSON + PNG artifacts written by the above scripts. |

## Roadmap

This repo is being built in phases. Phase numbers correspond to internal sub-prompts under `PyAutoLabs/PyAutoPrompt/z_features/autolens_profiling.md`.

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repo bootstrap | ✓ shipped |
| 1 | Mirror JIT likelihood profiling scripts + per-section READMEs | ✓ shipped |
| 2 | Mirror simulator profiling scripts + run-time tracking | ✓ shipped |
| 3 | Nautilus profiling, design for sampler expansion | ✓ shipped |
| 4 | Top-level + per-section README dashboard with instrument framing | ✓ shipped |
| 5 | GitHub Actions for lint + profile re-runs + README refresh | queued |

### Future enhancements (Phase 4 follow-ups)

Dashboards can grow in many directions. The list below captures candidate improvements that fit the "profiling and run-times" theme; none of them block the current dashboard from being useful.

- **Regression-watch indicator** — colour or arrow per cell showing whether the latest cost regressed (>5%) or improved versus the previous PyAutoLens release. Needs the second-latest version per axis kept alongside the latest. Trivial to add to `scripts/build_readme.py`.
- **Per-axis version-history PNGs** — small inline plot of run-time vs PyAutoLens release version, generated from the JSON artifacts (reusing the `_developer/jax_profiling/results/jit/.../*_v<version>.png` generator). Embeds nicely above each section table.
- **Plotly-rendered interactive timeline** — hostable on GitHub Pages once the static dashboard stabilises; lets readers hover/filter across instrument × model × release.
- **Flamegraph captures** — alongside the headline timing numbers, store a flamegraph per instrument × model for the most recent release.
- **Hardware-tier columns** — extend `scripts/build_readme.py` table renderers to show CPU / laptop GPU / HPC GPU as separate columns once result artifacts encode the hardware label (filename suffix or JSON `"hardware"` field).
- **Archive old versions** — once a script has >6 minor releases of artifacts, move the older ones to `results/archive/` so the latest views stay uncluttered.

## Related repos

- [`PyAutoLabs/PyAutoLens`](https://github.com/PyAutoLabs/PyAutoLens) — the library being profiled.
- [`PyAutoLabs/autolens_workspace`](https://github.com/PyAutoLabs/autolens_workspace) — user-facing science scripts and tutorials.
- [`PyAutoLabs/autolens_workspace_developer`](https://github.com/PyAutoLabs/autolens_workspace_developer) — the developer workspace; **source of truth during the migration**. Each phase mirrors the relevant subdirectories from here into this repo.
- [`Jammy2211/autolens_colab_profiling`](https://github.com/Jammy2211/autolens_colab_profiling) — sibling repo, Colab-specific scope. Not yet migrated to PyAutoLabs.

## Package vs scripts

This repo is a **collection of standalone profiling scripts**, not an installable Python package. There is no `pyproject.toml`. Run scripts from the repo root.

Scripts follow the JIT conventions documented in `autolens_workspace_developer/CLAUDE.md`:

- Extract `.array` from autoarray types before crossing the `jax.jit` boundary (autoarray types are not JAX pytrees as inputs).
- Pass `xp=jnp` through PyAutoLens / PyAutoGalaxy / PyAutoArray functions to select the JAX backend.

## Community & support

- **Slack** — [PyAutoLens workspace](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg) for questions.
- **Issues** — file profiling bugs and feature requests on this repo's [issue tracker](https://github.com/PyAutoLabs/autolens_profiling/issues).
