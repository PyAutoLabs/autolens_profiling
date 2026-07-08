# scripts

Repo tooling (not profiling scripts).

- **`build_readme.py`** — renders the auto-generated dashboard tables in the
  top-level and per-package READMEs from the artifacts under `results/`.
  Run `python scripts/build_readme.py` after a profiling run and commit the
  result; CI's `lint.yml` runs `--check` to enforce idempotence.
