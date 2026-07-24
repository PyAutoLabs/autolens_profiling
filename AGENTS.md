# autolens_profiling — Agent Instructions

This repo is the single home for **PyAutoLens performance measurement**: it benchmarks likelihood
runtime, per-step breakdown, VRAM usage, simulators, and samplers/searches across CPU, laptop GPU,
and HPC GPU (A100), framed by astronomy instrument (HST, Euclid, JWST, …). It is a collection of
standalone profiling scripts, **not** an installable package — there is no `pyproject.toml`. These
are the canonical, agent-agnostic instructions for this repo. The `README.md` is the human-facing
overview (vision, latest run-times, roadmap); this file is the operational guide.

## Repository Structure

Scripts are laid out **dataset-first, task-second** (`scripts/<dataset>/<task>/<model>.py`),
mirroring the `autolens_workspace*` taxonomy:

```
scripts/
  <dataset>/            imaging/ interferometer/ point_source/ multi/ cluster/ — one folder per
                        PyAutoLens dataset family. Group-scale cells live under cluster/; the
                        interferometer datacube cells nest under interferometer/<task>/datacube/.
    likelihood_runtime/   Full-pipeline JIT runtime per cell (<model>.py; driven by the sweep driver)
    likelihood_breakdown/ Per-step JIT decomposition of a single likelihood config
    searches/<sampler>/   Sampler / search profiling (Nautilus first)
    latent/               Latent-variable profiling
    quick_update/         Fast incremental re-profiling helpers (unversioned scratch tier)
  misc/                 Dataset-agnostic material + each task's shared drivers / framework / README:
                        misc/likelihood_runtime/ (sweep.py + aggregate.py + README dashboard),
                        misc/searches/ (framework _*.py + sweep/aggregate), misc/vram/ (A100 vmap
                        batch-size table), misc/simulators/, misc/latent/, misc/jax_compile/,
                        misc/pipeline_resume/, misc/test/, misc/tooling/ (build_readme.py +
                        build_baseline.py)
_profile_cli.py         Shared CLI/JSON/auto-simulate helper imported by every per-cell script
_adapt_image_util.py    Shared adapt-image helper
instruments/            Instrument definitions (pixel scale, shape) used to frame results
hpc/                    SLURM submit scripts for the RAL HPC (A100 rows of the sweep matrix)
results/                JSON + PNG artifacts: versioned summaries, sweep comparisons, named
                        baselines (results/README.md defines the shapes; sweeps default here)
config/ dataset/ output/   Config, input data, runtime output
```

**Import model.** Leaves sit several levels below the repo root, so each finds the root by walking
up to the directory containing `ruff.toml` (a depth-proof sentinel) and puts both the **repo root**
and **`scripts/misc/`** on `sys.path`. That keeps the shared libraries importable by their
top-level names with no per-file path math: `_profile_cli` / `_adapt_image_util` / `instruments`
(repo root) and `vram` / `simulators` / `searches` (under `scripts/misc/`).

## Running Profiles

Run a script from the repo root. Each profiling script writes a versioned `summary` JSON + PNG pair
under `results/` whose version string matches the PyAutoLens release that produced the numbers, so
trends stay inspectable across releases. A script auto-simulates its dataset if missing.

```bash
python3 scripts/imaging/likelihood_runtime/mge.py --config-name hst --use-mixed-precision
```

`_profile_cli.py` is the **shared helper module** imported by the likelihood scripts (not a runnable
command): it defines the common sweep flags (`--config-name`, `--output-dir`,
`--use-mixed-precision`), the device-info capture, the output-path resolver, and the
auto-simulate-if-missing hook, so per-script boilerplate stays minimal.

The PyAuto* libraries are **not pip-installed** here — they are resolved from sibling source
checkouts via `PYTHONPATH`. On the HPC, `source activate.sh` activates the shared venv (third-party
deps only) and points `PYTHONPATH` at the canonical `PyAutoNerves`/`PyAutoFit`/`PyAutoArray`/
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

The same job also runs `scripts/misc/tooling/build_readme.py --check` (dashboard idempotence), a `lychee`
markdown link-rot check over the `README.md` files, and a per-section **smoke** that imports one
script from each area under `AUTOLENS_PROFILING_SMOKE=1` (catches import-graph breakage without
running a full profile). None of these produce result artifacts.

`profile.yml` runs the actual profile sweeps + dashboard refresh, but it is **manual / on-release
only** (`workflow_dispatch` + release tag) — it is **not** a per-PR gate (profiling burns CI minutes
and is noisy; releases are the natural cadence).

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python3 scripts/imaging/likelihood_runtime/mge.py
```

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- `../PyAutoLens` — the library being profiled (plus `../PyAutoGalaxy`, `../PyAutoArray`,
  `../PyAutoFit`, `../PyAutoNerves` on `PYTHONPATH`).
- `../autolens_workspace` — user-facing science scripts and tutorials.
- `../PyAutoHands` — build/CI tooling.

## Task Workflows

When adding or updating a profile script, keep `ruff check .` and `ruff format --check .` clean
(the PR gate), write the versioned `results/` artifact pair, and do not commit machine-specific
absolute paths. Flag any change that affects the source libraries or `autolens_workspace` in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
