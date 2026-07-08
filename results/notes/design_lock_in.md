# Design lock-in — pre-PreOptimizationTimes review (2026-07-08)

One final holistic review of the repo before the **PreOptimizationTimes**
baseline campaign — the last full profiling sweep before the optimization
push. This note records what is locked in, what changed, and why, so future
extensions (more datasets, instruments, packages) build on a settled core.
Tracked by [autolens_profiling#52](https://github.com/PyAutoLabs/autolens_profiling/issues/52);
parent intent: PyAutoMind `maintenance/autolens_profiling/polish.md`.

## What is locked in (unchanged — the design is right)

- **The package split.** `likelihood_runtime/` (how long per call?) vs
  `likelihood_breakdown/` (where does the time go?) vs `vram/` (does it fit,
  and at what vmap batch size?) are deliberately disjoint questions with
  deliberately disjoint packages. `latent/`, `searches/`, `simulators/`,
  `quick_update/` follow the same one-question-per-package rule. New profiling
  concerns get a **new package**, not a flag on an existing one.
- **The cell grid.** Work is addressed as `<dataset_class>/<model>` cells
  (e.g. `imaging/mge`, `datacube/delaunay`), optionally deepened by
  `/<instrument>`. Sweep drivers, aggregators, output dirs and HPC submit
  scripts all speak this grid. New datasets/models slot in as new cells.
- **`_profile_cli.py` as the single CLI surface.** Every per-cell script takes
  `--config-name / --output-dir / --use-mixed-precision / --instrument /
  --vmap-probe / --sparse` through the shared helper. New flags go here, once,
  never per-script.
- **Instrument framing** via `instruments/` presets; headline numbers are
  named for observing programmes (HST, ALMA, JWST…), not pixel counts.
- **Config axis names**: `local_cpu_fp64 | local_cpu_mp | local_gpu_fp64 |
  local_gpu_mp | hpc_a100_fp64 | hpc_a100_mp` (the 6-config matrix), with
  `_sparse` as a filename suffix, not a seventh config.
- **Correctness gates inside every timing script** (eager ≡ JIT ≡ vmap at
  documented rtol) stay mandatory for new cells.

## What changed in this review

1. **Results live wholly in this repo.** `sweep.py` / `aggregate.py`
   previously defaulted `--output-root` to
   `../autolens_workspace_developer/jax_profiling/results/jit` — a migration
   leftover. The default is now **`results/runtime/`** in-repo. The
   workspace_developer tree remains readable history; nothing new is written
   there. (`--output-root` still overrides for scratch runs.)
   The same fix propagated to every default that still pointed at a legacy
   home: the per-cell `likelihood_runtime` scripts (standalone/HPC runs
   defaulted to the retired `results/likelihood/<class>/`; now
   `results/runtime/<class>/<model>/`, so SLURM jobs land exactly where a
   local sweep writes) and `latent/sweep.py` / `latent/aggregate.py`
   (workspace_developer → `results/latent/`). Because a cell dir can now hold
   both sweep configs and versioned standalone summaries, `aggregate.py`
   filters to config-shaped stems when building `comparison.json`.
2. **No machine-specific defaults.** `sweep.py`'s hard-coded
   `/home/jammy/venv/PyAutoGPU/bin/python` default became `sys.executable`
   (violated the repo's own "no machine-specific absolute paths" rule).
3. **`imaging/mge` joined the sweep `CELLS` grid** so the runtime campaign
   covers the same imaging cells as the breakdown package; `sweep.py` also
   gained `--sparse` passthrough so the imaging sparse-vs-mapping comparison
   runs through the same driver as everything else.
4. **The README dashboard was revived and retargeted.**
   `scripts/build_readme.py` still pointed at the retired `likelihood/`
   package (deleted in the runtime/breakdown split) and scanned 0 artifacts.
   It now scans the real `results/` sections (`breakdown`, `runtime`,
   `simulators`, `searches`), understands both artifact shapes (see below),
   and renders auto-tables into the top-level README + per-package READMEs.
   CI's `--check` idempotence gate is unchanged.
5. **Stale docs fixed**: retired-`likelihood/` references in `AGENTS.md`,
   `README.md`, `results/README.md` and `_profile_cli.py`; the
   `z_projects/profiling/hpc/sync` references (HPC dispatch lives in-repo
   under `hpc/`); the roadmap's retired `PyAutoPrompt` registry pointer.
   `quick_update/`, `hpc/` and `scripts/` gained READMEs.

## The two artifact shapes (both canonical)

| Shape | Written by | Pattern | Versioning |
|-------|-----------|---------|------------|
| **Versioned summary** | per-cell scripts run standalone | `<cell>_<purpose>_<instrument>_v<PyAutoLens-version>[_sparse].{json,png}` | in the filename; history retained side-by-side |
| **Per-config sweep** | `sweep.py` → `aggregate.py` | `<class>/<model>[/<instrument>]/<config_name>[_sparse].{json,png,log}` + `comparison.{json,png}` | in each JSON's metadata; dirs hold the *latest* sweep |

Rule of thumb: **cross-release trend** questions read versioned summaries;
**cross-hardware comparison** questions read `comparison.json`.

## The PreOptimizationTimes convention (for phases 2–4)

- A **baseline** is a named, frozen snapshot of campaign results:
  `results/baselines/<BaselineName>/` containing (a) the `comparison.json`
  per swept cell, (b) the versioned summary JSONs the campaign produced, and
  (c) a rendered **`<BaselineName>.md`** — every headline number in one
  browsable page (cells × configs × instruments).
- `PreOptimizationTimes` is the first such baseline: laptop CPU, HPC CPU and
  HPC A100 (+ mp where supported); vmap-only runtime numbers using the
  `vram/` batch-size table; laptop GPU appended later by hand.
- The top-level README dashboard grows a baseline column once
  `results/baselines/PreOptimizationTimes/` exists — the "compare against
  this" anchor for all optimization work that follows.
- Baselines are **append-only**: never edited after the campaign closes; a
  post-optimization campaign snapshots a new name next to it.

## Deliberately out of scope here

Profiling runs (phases 2–4); searches profiling; `point_source` cells (in the
grid, excluded from this campaign); laptop-GPU rows (human-run follow-up);
gradient profiling (still in `autolens_workspace_developer`, folds in later);
the future PyAutoBrain profiling agent (separate `feature/pyautobrain/` task).
