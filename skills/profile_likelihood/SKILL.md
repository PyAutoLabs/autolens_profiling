---
name: profile-likelihood
description: Run a multi-config JAX likelihood profiling sweep (CPU/GPU × fp64/mp, optional A100 on RAL HPC) for a given likelihood function. Produces structured JSON+PNG outputs with comparison tables under autolens_workspace_developer/jax_profiling/results/jit/.
---

# Profile Likelihood

Profile a JAX likelihood function across multiple device + precision
configurations and consolidate results into the canonical tracking directory
`autolens_workspace_developer/jax_profiling/results/jit/<dataset_type>/<likelihood_type>/`.
A **science-profiling** skill homed in `autolens_profiling` (PyAutoLens
performance measurement) — not an organism dev-workflow skill. It reuses the
tooling at `z_projects/profiling/scripts/` and mirrors the `mge-profiling-a100`
workflow. Verbose command blocks, gotchas and run precedent are in
[`reference.md`](reference.md).

## Usage

```
$profile-likelihood <dataset_type> <likelihood_type>
```

In Claude, invoke the same skill as `/profile_likelihood`.

`<dataset_type>` ∈ `imaging` / `interferometer` / `point_source`.
`<likelihood_type>` is a canonical reference script present at
`autolens_workspace_developer/jax_profiling/jit/<dataset_type>/<likelihood_type>.py`
(mge, delaunay, rectangular, delaunay_mge, rectangular_mge, …). Examples:
`$profile-likelihood imaging mge`,
`$profile-likelihood interferometer rectangular_mge`.

## What this produces

`results/jit/<dataset_type>/<likelihood_type>/` gains
`local_cpu_{fp64,mp}`, `local_gpu_{fp64,mp}`, `hpc_a100_{fp64,mp}`, and
`comparison` — each a `.json` + `.png` pair. Each `<config>.json` carries
per-step JIT timings (step count is likelihood-specific), full-pipeline single-JIT
time, vmap batch=3 time, eager/JIT/vmap log-likelihoods, device info, and static
memory analysis. `comparison` is the aggregated cross-config view.

## Steps

Verify prerequisites first ([`reference.md`](reference.md) → "Prerequisites":
GPU/HPC availability, the PyAutoGPU venv ordering, and the canonical reference).

0. **Read prior-art context (not optional).** Read `comparison.json` for any
   already-profiled likelihood in the same `<dataset_type>` family; extract
   per-config `full_pipeline_per_call` / `vmap_per_call` and the dominant steps.
   Use it to predict where the new run should land and to frame findings; if this
   is the family's first likelihood, say so and skip. Summarise to the user in
   4–6 lines before scaffolding.
1. **Identify + check tooling** — confirm the canonical reference and whether
   `z_projects/profiling/scripts/<likelihood_type>_profile.py` already exists
   (skip scaffolding if so).
2. **Scaffold** the per-likelihood profile script + `_setup` module (first time
   only) — [`reference.md`](reference.md) → "Scaffold".
3. **SLURM submit scripts** (first time only) — [`reference.md`](reference.md) →
   "SLURM submit scripts".
4. **Plan + worktree** — run `$start-dev` (`/start_dev` in Claude); worktree
   `autolens_workspace_developer` on `feature/<likelihood_type>-profiling-a100`.
5. **Local sweep** — 4 configs (GPU/CPU × fp64/mp); spot-check `device.backend`.
6. **HPC sweep** — A100 fp64 + mp via `hpc/sync`, then consolidate.
7. **Aggregate** — write `comparison.{json,png}`; sanity-check the magnitudes.
8. **Commit + PR** — stage only the new profiling scripts and the new results
   subdir; open the PR with the timings table + key findings.
9. **Post-merge cleanup** — remove the worktree, return canonical to `main`, move
   the task to `complete.md`.

Command blocks for steps 2–9 are in [`reference.md`](reference.md) → "Step
detail".

## Notes

- This skill profiles performance; it does not gate releases (that is Heart) or
  change library behaviour.
- Keep `ruff check .` / `ruff format --check .` clean if you touch
  `autolens_profiling` itself; never commit machine-specific absolute paths.
- Gotchas (`JAX_PLATFORM_NAME=cpu`, the PyAutoGPU venv, `PYAUTO_ROOT`) and run
  precedent are in [`reference.md`](reference.md).
