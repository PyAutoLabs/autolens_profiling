# scripts

Repo tooling (not profiling scripts).

- **`build_readme.py`** — renders the auto-generated dashboard tables in the
  top-level and per-package READMEs from the artifacts under `results/`.
  Run `python scripts/build_readme.py` after a profiling run and commit the
  result; CI's `lint.yml` runs `--check` to enforce idempotence.
- **`list_timing_assertions.py`**: read-only AST lister of every timing assertion
  and timing-gate constant (cutoffs, budgets, repeat and bootstrap settings).
  `--check` fails when a key it finds is missing from the timing-noise
  inventory [`wiki/campaigns/misc/timing_noise_audit_2026_10.md`](../../../wiki/campaigns/misc/timing_noise_audit_2026_10.md)
  (autolens_profiling#362). It runs in `lint.yml`, so a new timing gate has to be
  inventoried, with its noise model and verdict, in the PR that adds it.
- **`build_migration_map.py`** / **`apply_migration_map.py`** — the day-zero move of
  `results/` into the mirrored tree (autolens_profiling#442). The generator writes
  `catalogue/migration_day_zero.json` (old -> new path + sha256) from the pre-move tree;
  once applied, its `--check` reports the map frozen. The applier `git mv`s every entry
  (`--apply`, sha-checked, re-runnable) and repoints references from the map alone
  (`--rewrite-refs`, idempotent; never edits `results/`, `archive/` or generated output).
- **`results_layout.py`** — readers' view of the moved tree: a moved file keeps its
  pre-move identity (`iter_legacy`, `legacy_glob`, `current`) until the catalogue builder
  reads the mirrored tree natively; `destination` places a new result with the map's
  classification (used by `_profile_cli.resolve_output_paths`).
