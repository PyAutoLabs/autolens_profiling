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
  pre-move identity (`iter_legacy`, `legacy_glob`, `current`); only `build_dashboard.py` and
  `build_readme.py` still read through it. `destination` places a new result with the map's
  classification (used by `_profile_cli.resolve_output_paths`).
- **`build_catalogue.py`** / **`catalogue_lifecycle.py`** — the `profiling-summary@3`
  lifecycle catalogue (`dashboard/catalogue.json`) and the companion v2 setup catalogue
  (`dashboard/catalogue/setups.json` + shards), read natively from the mirrored tree; CI runs
  `--check --validate-with ../PyAutoPulse`. A result row with no recorded timestamp takes its
  `measured_at` from the UTC date its file was first committed (an upper bound), read from
  `catalogue/measured_at_first_commit.json`. The builder never calls git (CI clones are
  shallow), so after committing new undated rows run, in a checkout with full history,
  `python scripts/misc/tooling/build_catalogue.py --refresh-commit-dates` and commit the file;
  `--check` fails with that instruction when an undated file has no date.
