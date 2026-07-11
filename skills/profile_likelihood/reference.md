# profile_likelihood — reference detail

Factored out of `SKILL.md` to keep the primary skill concise. The skill body is
authoritative for the flow; this file holds the verbose command blocks, the
gotchas, and the run precedent.

## Prerequisites

1. **Hardware** — a CUDA GPU is needed for `local_gpu_*`. CPU-only machine: run a
   4-config sweep without GPU rows (the aggregator tolerates missing configs).
2. **HPC access** — the A100 sweep needs `z_projects/profiling/hpc/sync.conf`
   (HPC_HOST, HPC_BASE, PROJECT_NAME). `hpc/sync check` verifies SSH + remote dir
   + sbatch. If unavailable, the local sweep alone is still valuable.
3. **Venvs** — local GPU: `/home/jammy/venv/PyAutoGPU/bin/activate` (Py3.10 +
   JAX-CUDA12), sourced **before** the worktree `activate.sh`; HPC:
   `/mnt/ral/jnightin/PyAutoNSS/PyAutoNSS/bin/activate` (auto-sourced by
   `z_projects/profiling/activate.sh`).
4. **Canonical reference** —
   `autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py`
   must exist; the profiling script is an argparse-driven simplification of it.

## Step detail

### Scaffold the per-likelihood profiling script (first time only)

Template: `z_projects/profiling/scripts/mge_profile.py` (imports
`_setup.build_dataset / build_model / build_analysis`). For a new likelihood:
add `_setup_<likelihood_type>.py` (mirror `_setup.py`, adjust `build_model` to
the right pixelization/profile/regularization from the canonical reference) and,
if the step structure differs, a `<likelihood_type>_profile.py` mirroring
`mge_profile.py` (same JSON schema so the aggregator is unchanged).

### SLURM submit scripts (first time only)

Clone `z_projects/profiling/hpc/batch_gpu/submit_mge_profile_{fp64,mp}` →
`submit_<likelihood_type>_profile_{fp64,mp}`, updating `#SBATCH -J`, the python
invocation (`--config-name hpc_a100_<fp64|mp>` + script path), and
`--output-dir $PROJECT_PATH/output/<dataset_type>/<likelihood_type>`. `chmod +x`
both.

### Plan + worktree

Run `$plan-branches` (`/plan_branches` in Claude). Affected:
`z_projects/profiling` (no remote, local-only — commit to local main) and
`autolens_workspace_developer` (artifacts, has remote — feature branch + PR).
Branch `feature/<likelihood_type>-profiling-a100`.

```bash
source admin_jammy/software/worktree.sh
worktree_create <likelihood_type>-profiling-a100 autolens_workspace_developer
```

### Local sweep — 4 configs

```bash
source /home/jammy/venv/PyAutoGPU/bin/activate
source /home/jammy/Code/PyAutoLabs-wt/<likelihood_type>-profiling-a100/activate.sh
WORKTREE_OUTPUT="$PYAUTO_ROOT/autolens_workspace_developer/jax_profiling/results/jit/<dataset_type>/<likelihood_type>"

python z_projects/profiling/scripts/<likelihood_type>_profile.py --config-name local_gpu_fp64 --output-dir "$WORKTREE_OUTPUT"
python z_projects/profiling/scripts/<likelihood_type>_profile.py --use-mixed-precision --config-name local_gpu_mp --output-dir "$WORKTREE_OUTPUT"
JAX_PLATFORM_NAME=cpu python z_projects/profiling/scripts/<likelihood_type>_profile.py --config-name local_cpu_fp64 --output-dir "$WORKTREE_OUTPUT"
JAX_PLATFORM_NAME=cpu python z_projects/profiling/scripts/<likelihood_type>_profile.py --use-mixed-precision --config-name local_cpu_mp --output-dir "$WORKTREE_OUTPUT"
```

The worktree `activate.sh` exports `PYAUTO_ROOT` so canonical writes land on the
feature branch. Spot-check each JSON's `device.backend`. (Optional pre-fix
ingestion of `/tmp` artifacts: `<likelihood_type>_aggregate.py --ingest-pre-fix /tmp`.)

### HPC sweep — A100 fp64 + mp

```bash
cd z_projects/profiling
hpc/sync push
hpc/sync submit gpu submit_<likelihood_type>_profile_fp64
hpc/sync submit gpu submit_<likelihood_type>_profile_mp
hpc/sync jobs                          # ~5 min each; /loop a wakeup to detach
hpc/sync pull                          # when both jobs are gone
python z_projects/profiling/scripts/<likelihood_type>_aggregate.py --consolidate-from z_projects/profiling/output/<dataset_type>/<likelihood_type>
```

### Aggregate

```bash
python z_projects/profiling/scripts/<likelihood_type>_aggregate.py
```

Writes `comparison.{json,png}`. Sanity: A100 O(1)–O(10) ms; consumer GPU
O(10)–O(100) ms; CPU O(100)–O(1000) ms — re-run any wildly-off config.

### Commit + PR

`z_projects/profiling` (local main): stage only the new
`scripts/<likelihood_type>_*.py`, `scripts/_setup_<likelihood_type>.py` (if
scaffolded) and the SLURM scripts; commit. Worktree
`autolens_workspace_developer`: stage only the new
`jax_profiling/results/jit/<dataset_type>/<likelihood_type>/` subdir; commit,
push, `gh pr create`. PR body: headline timings table, key findings (fp64 vs mp;
A100 vs RTX 2060), caveats (jax_enable_x64; cache state), test-plan checklist.

### Post-merge cleanup

`worktree_remove <likelihood_type>-profiling-a100`; on canonical
`autolens_workspace_developer`, `fetch` + `checkout main` + `pull --ff-only` +
delete the feature branch; move the `active.md` entry to `complete.md` and
`prompt_sync_push`.

## Gotchas

- **`JAX_PLATFORMS=cpu` is broken on JAX 0.4.38** (pre-existing CUDA arrays can't
  move). Use `JAX_PLATFORM_NAME=cpu`.
- **GPU venv is PyAutoGPU, not PyAuto** — activate it before the worktree
  `activate.sh`; verify `python -c "import jax; print(jax.default_backend())"` →
  `gpu`.
- **`PYAUTO_ROOT`** (set by worktree `activate.sh`) routes canonical writes to the
  feature branch — without it they hit canonical main.
- **A100 `jax_enable_x64`** may be off on the HPC venv → JIT log-likelihood
  truncates to fp32; the eager numpy reference is the trustworthy fp64 value.
- **Single-machine cross-session deltas are unreliable** (JAX cache + thermal
  state). Cross-platform comparisons are robust.
- **HPC dataset can be stale** — `hpc/sync push` skips existing files; use
  `hpc/sync push-data-init` to force re-upload.
- **Pre-existing dirty files** — stage only the new `<likelihood_type>/` subdir;
  never `git add -A`/`prompt_sync_push` blindly.

## Run precedent

- `mge-profiling-a100` (autolens_workspace_developer #56, merged 2026-05-09) —
  first run; 8-step MGE; A100 fp64 = 5.7 ms; vmap ~2×; F + Mapping matrix dominate.
- `pixelization-profiling-a100` (#57) — RectangularAdaptDensity; 11-step;
  A100 fp64 = 9.7 ms; vmap does not help (serial NNLS); bottleneck shifts from F
  (consumer GPU) to NNLS (A100).
- Canonical implementations: `z_projects/profiling/scripts/mge_profile.py` /
  `pixelization_profile.py` + aggregators (https://github.com/PyAutoLabs/z_projects,
  no remote).
