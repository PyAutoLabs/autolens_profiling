# archive

Dead experiments and frozen evidence moved out of `results/` in the day-zero move
(autolens_profiling#442): dated-suffix snapshots, `_lever` and `_probe` experiments, the
pre-2026-09 series, host-less versioned summaries, `results/baselines/PreOptimizationTimes/`
and the frozen evidence packs (`clipper_campaign/`, `point_source_cpu_2026_09_17_reported/`).

Layout: `archive/YYYY/MM/<slug>/<path under results/ or inside the pack>`. `YYYY/MM` is the
file's own date (file name or JSON), else the latest date in its slug, else the date of a
ledger that cites it; `archive/undated/<slug>/` when no date is derivable. Every file's old
path, new path and sha256 is in `catalogue/migration_day_zero.json`; nothing here changed
bytes.

Archived rows are history: they are never a current measurement. The catalogue builder still
reads them (`catalogue/registry.json` source `archive/**/*.json`) so its output is unchanged
by the move; the `profiling-summary@3` builder counts them as `archive_record_count`
(#442 PR 3). Never write new results here.
