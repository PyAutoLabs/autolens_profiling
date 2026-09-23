# Point-source CPU study, 2026-09-17 — REPORTED evidence

This folder preserves the 2026-09-17 read-only study of the PointSolver
image-plane chi-squared on CPU, recovered from the session scratch directory it
was written in so that it is not lost. It is **REPORTED evidence, not a current
baseline.** Nothing here was produced by a cell in this repository.

Campaign: autolens_profiling issue #297
(<https://github.com/PyAutoLabs/autolens_profiling/issues/297>). The campaign
note that interprets these files is
[`results/notes/point_source_cpu_campaign.md`](../point_source_cpu_campaign.md);
its phase-1 section holds the RAL baseline these ratios must be reproduced against.

## How these numbers were made

- **Host:** WSL2 laptop, 8 logical cores (the same machine as the
  `*_local_cpu_fp64` rows), BLAS family pinned to 1 thread, JAX CPU backend.
- **Stack:** jax 0.10.2, autolens 2026.8.17.1.
- **Harness:** a scratch harness that **monkeypatched library internals** (for
  example, dropping the `jnp.unique` de-duplication inside the PointSolver).
  The patched code is **not shipped** in PyAutoArray / PyAutoLens; the rows
  describe what a change *would* buy, not what the library does.
- **Ratios, not milliseconds.** Every A/B was measured as within-run,
  interleaved ratios on one host at one load. Absolute numbers are not
  comparable across runs, across hosts, or with the committed breakdown rows.

## What is here

| file | contents |
|---|---|
| `pointsolver_cpu_research_note.md` | the study's write-up, verbatim except that absolute local paths were replaced with `<workspace>` / `<scratch>` |
| `characterise.json` | per-step triangle counts and finite-point census of the solve |
| `bench_*.json` | the individual benches the note's tables cite (ABBA ratios, breakdown, cluster solve, no-dedup full likelihood, `n_steps` × grid, NumPy path, step 0) |

The note's section on reproducing refers to scratch scripts (`common.py`,
`harness.py`, `bench_*.py`) and HLO dumps. Those were **not** copied: the
scripts patch library internals and the dumps are large compiler output, so
neither belongs in the repository. The note's claims should be re-measured by
the campaign's own cells before anything is quoted as a baseline.
