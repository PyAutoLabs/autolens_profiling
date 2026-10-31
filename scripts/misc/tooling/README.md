# tooling

Repo tooling (not profiling scripts). Its tests, and the repo-wide layout tests
(results mirror, write destinations, script routes), are in `test/`:
`pytest scripts/misc/tooling/test -q`.

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
- **`build_series_id_map.py`** — `catalogue/series_id_map.json`, generated from the day-zero
  map: every pre-move dashboard series id (legacy file grammar) -> the v3 id it now publishes
  under, with the releases it held; the pre-move points outside the mirrored tree it attaches;
  and the pre-move runtime cell -> current cell map `build_readme.py` matches archived
  baselines with. `--check` reports it stale; `test_build_dashboard.py` checks history stays
  continuous.
- **`build_catalogue.py`** / **`catalogue_lifecycle.py`** — the `profiling-summary@3`
  lifecycle catalogue (`dashboard/catalogue.json`) and the companion v2 setup catalogue
  (`dashboard/catalogue/setups.json` + shards), read natively from the mirrored tree; CI runs
  `--check --validate-with ../PyAutoPulse`, which also fails when
  `catalogue/pulse_designations.json` drifts from PyAutoPulse's designation records and resolves
  every `decision_id` (re-sync with `--sync-designations ../PyAutoPulse`, then rebuild). A result row with no recorded timestamp takes its
  `measured_at` from the UTC date its file was first committed (an upper bound), read from
  `catalogue/measured_at_first_commit.json`. The builder never calls git (CI clones are
  shallow), so after committing new undated rows run, in a checkout with full history,
  `python scripts/misc/tooling/build_catalogue.py --refresh-commit-dates` and commit the file;
  `--check` fails with that instruction when an undated file has no date.
- **`coverage_matrix.py`** — the lifecycle coverage matrix (`dashboard/coverage.{json,md}`):
  per likelihood x class x measurement, how many of the catalogue's planned cells are current,
  stale (release pin behind the catalogue pin), superseded or missing (a `size_fallback`-only
  cell counts as missing), plus a parity witness per planned setup cell. It cross-checks each
  planned cell's class against `catalogue/pulse_designations.json` (the PyAutoPulse designation
  mirror) and `catalogue/designation_holds.json`. CI runs `--check`; re-run it after
  `build_catalogue.py`.
- **`parity_witness.py`** — the `parity.json` convention and its validator (`--check` over
  `results/**/parity.json`); see [results/README.md](../../../results/README.md#parity-witness).
