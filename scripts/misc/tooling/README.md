# scripts

Repo tooling (not profiling scripts).

- **`build_readme.py`** — renders the auto-generated dashboard tables in the
  top-level and per-package READMEs from the artifacts under `results/`.
  Run `python scripts/build_readme.py` after a profiling run and commit the
  result; CI's `lint.yml` runs `--check` to enforce idempotence.
- **`list_timing_assertions.py`**: read-only AST lister of every timing assertion
  and timing-gate constant (cutoffs, budgets, repeat and bootstrap settings).
  `--check` fails when a key it finds is missing from the timing-noise
  inventory [`results/notes/timing_noise_audit_2026_10.md`](../../../results/notes/timing_noise_audit_2026_10.md)
  (autolens_profiling#362). It runs in `lint.yml`, so a new timing gate has to be
  inventoried, with its noise model and verdict, in the PR that adds it.
