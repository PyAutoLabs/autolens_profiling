# autolens_profiling — Agent Instructions

This repo is the single home for **PyAutoLens performance measurement**: it benchmarks likelihood
runtime, per-step breakdown, VRAM usage, simulators, and samplers/searches across CPU, laptop GPU,
and HPC GPU (A100), framed by astronomy instrument (HST, Euclid, JWST, …). It is a collection of
standalone profiling scripts, **not** an installable package — there is no `pyproject.toml`. These
are the canonical, agent-agnostic instructions for this repo. The `README.md` is the human-facing
overview (vision, latest run-times, roadmap); this file is the operational guide.

## Repository Structure

```
likelihood/             Per-instrument likelihood profile scripts (imaging, interferometer,
                        datacube, point_source) — import the shared _profile_cli helper
likelihood_runtime/     Full-pipeline JIT runtime, driven by sweep.py across CPU/GPU/A100 × fp64/mp
likelihood_breakdown/   Per-step JIT decomposition of a single likelihood config
vram/                   GPU memory-usage profiling
instruments/            Instrument definitions (pixel scale, shape) used to frame results
searches/               Sampler / search profiling (Nautilus first)
simulators/             Run-time tracking for the PyAutoLens simulators
latent/                 Latent-variable profiling
quick_update/           Fast incremental re-profiling helpers
hpc/                    SLURM submit scripts for the RAL HPC
results/                Versioned JSON + PNG artifacts (`*_v<YYYY>.<M>.<D>.<PATCH>.{json,png}`)
config/ dataset/ output/   Config, input data, runtime output
```

## Running Profiles

Run a script from the repo root. Each profiling script writes a versioned `summary` JSON + PNG pair
under `results/` whose version string matches the PyAutoLens release that produced the numbers, so
trends stay inspectable across releases. A script auto-simulates its dataset if missing.

```bash
python3 likelihood/imaging/mge.py --config-name hst --use-mixed-precision
```

`_profile_cli.py` is the **shared helper module** imported by the likelihood scripts (not a runnable
command): it defines the common sweep flags (`--config-name`, `--output-dir`,
`--use-mixed-precision`), the device-info capture, the output-path resolver, and the
auto-simulate-if-missing hook, so per-script boilerplate stays minimal.

The PyAuto* libraries are **not pip-installed** here — they are resolved from sibling source
checkouts via `PYTHONPATH`. On the HPC, `source activate.sh` activates the shared venv (third-party
deps only) and points `PYTHONPATH` at the canonical `PyAutoConf`/`PyAutoFit`/`PyAutoArray`/
`PyAutoGalaxy`/`PyAutoLens` checkouts; `HPCPullPyAuto` is then the whole library-update story.

JAX convention (mirrors `autolens_workspace_developer`): pass `xp=jnp` through PyAuto* functions to
select the JAX backend, and extract `.array` from autoarray types before crossing the `jax.jit`
boundary **as inputs**. See the PyAutoArray deep dive
`../PyAutoArray/docs/agents/jax_and_decorators.md` for the full boundary story.

## Testing

The PR gate is `lint.yml` on Python 3.12 (every PR + push to `main`). Its headline lint is **ruff**,
not black:

```bash
ruff check .
ruff format --check .
```

The same job also runs `scripts/build_readme.py --check` (dashboard idempotence), a `lychee`
markdown link-rot check over the `README.md` files, and a per-section **smoke** that imports one
script from each area under `AUTOLENS_PROFILING_SMOKE=1` (catches import-graph breakage without
running a full profile). None of these produce result artifacts.

`profile.yml` runs the actual profile sweeps + dashboard refresh, but it is **manual / on-release
only** (`workflow_dispatch` + release tag) — it is **not** a per-PR gate (profiling burns CI minutes
and is noisy; releases are the natural cadence).

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python3 likelihood/imaging/mge.py
```

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- `../PyAutoLens` — the library being profiled (plus `../PyAutoGalaxy`, `../PyAutoArray`,
  `../PyAutoFit`, `../PyAutoConf` on `PYTHONPATH`).
- `../autolens_workspace` — user-facing science scripts and tutorials.
- `../PyAutoBuild` — build/CI tooling.

## Task Workflows

When adding or updating a profile script, keep `ruff check .` and `ruff format --check .` clean
(the PR gate), write the versioned `results/` artifact pair, and do not commit machine-specific
absolute paths. Flag any change that affects the source libraries or `autolens_workspace` in your PR.

## Clean state

Never rewrite history on a repo with a remote (no `git init` over a tracked tree, no force-push to
`main`, no rebasing pushed shared branches). To reset a dirty tree the only correct sequence is:

```bash
git fetch origin
git reset --hard origin/main
git clean -fd
```
