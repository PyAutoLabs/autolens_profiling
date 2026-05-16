# `.github/workflows/`

CI for `autolens_profiling`. Two workflows, deliberately split.

## `lint.yml` — PR + push-to-main gate

Runs on every PR and every push to `main`. CPU-only, target wall time under 5 minutes.

What it checks:

| Step | Purpose |
|------|---------|
| `ruff check .` | Pyflakes + pycodestyle + isort + pyupgrade + flake8-bugbear (see `ruff.toml`) |
| `ruff format --check .` | Formatting parity with sister PyAutoLabs repos (black-compatible defaults) |
| `python scripts/build_readme.py --check` | Dashboard idempotence — the auto-generated tables in every section README must match what `build_readme.py` would generate from the current `results/` artifacts. Catches the "forgot to rerun the dashboard generator after dropping a new result" class of bug |
| `lychee` | Markdown link-rot across every `README.md` |
| Smoke — one script per section | Runs `likelihood/imaging/mge.py`, `simulators/imaging.py`, and `searches/nautilus/simple.py` with `AUTOLENS_PROFILING_SMOKE=1`. Every profile script reads that env var at module top and exits 0 after the import + setup section. Catches import-graph breakage (broken `sys.path` injection, missing dependency, renamed module) without running the full profile |

The smoke step does **not** produce real result artifacts — every script short-circuits before the JIT compile / sampling / FITS writes. If you need full smoke output, run `profile.yml` manually instead.

## `profile.yml` — manual + on-release profile re-run

Triggered by:

- `workflow_dispatch` (manual via the GitHub UI). Optional `sections` input lets you scope a run to one of `likelihood`, `simulators`, `searches`, or any comma-separated combination. Leave blank to run everything.
- `release: published` — when a new GitHub release is published.

What it does:

1. Runs every script under `likelihood/`, `simulators/`, and `searches/nautilus/`, producing JSON+PNG artifacts under `results/`. `continue-on-error: true` per section so a single regression doesn't block the dashboard refresh for the remaining 16+ scripts; failures emit a `::warning::` annotation and the matching dashboard cell will show `ERR`.
2. Skips `simulators/point_source.py` in the simulator loop because its default `dataset_name="simple"` overwrites the Phase 1 likelihood input JSONs (see `simulators/README.md`). Run that one manually with a non-conflicting `dataset_name` when needed.
3. Runs `python scripts/build_readme.py` to refresh every auto-generated table from the latest artifacts.
4. Commits the diff back to `main` as `github-actions[bot]` with `[skip ci]` in the subject (prevents the lint workflow from re-triggering on the auto-generated commit).

Hardware: GitHub-hosted `ubuntu-latest` (CPU). Expect Nautilus's `n_live=200` runs to take 30–60 minutes each on CPU; total job time can approach the 4-hour `timeout-minutes` budget on a full run. Self-hosted GPU runners can be added later as a separate job that appends `*_gpu*.json` artifacts to `results/` without restructuring this workflow — the dashboard's hardware-tier column extension (top-level README "Future enhancements") is the matching reader-side change.

## How to trigger `profile.yml` manually

From the GitHub UI:

1. Repo → **Actions** → **profile** workflow.
2. Click **Run workflow**.
3. (Optional) Enter a sections filter, e.g. `likelihood,simulators`, or leave blank for all.
4. Click **Run workflow**.

Or via `gh`:

```bash
gh workflow run profile --repo PyAutoLabs/autolens_profiling -F sections=likelihood
gh workflow run profile --repo PyAutoLabs/autolens_profiling  # all sections
```

## Design decisions (captured for future maintainers)

- **CPU-only runners** to start. GitHub-hosted is free and the dashboard's CPU column is the most-requested baseline. Future GPU laptop / A100 columns require either self-hosted runners or external upload of `*_<hardware>_<version>.json` files; both are additive on this workflow's shape.
- **No matrix across Python versions / OSes**. Lens-modelling profiling is Linux-only in practice, and a single Python version (3.12) keeps the matrix lean.
- **No coverage reporting**. This repo has no unit tests by design — it's a scripts collection, and the smoke step + dashboard idempotence check are the practical equivalents.
- **Why `[skip ci]` rather than path filters**: the lint workflow's smoke step does run scripts, and a commit that touches only `results/` and `README.md` files could in principle re-trigger the lint workflow (which is cheap, but pointless). Subject-line `[skip ci]` is the simplest fix and is honoured by GitHub Actions natively.
- **`continue-on-error: true`** on each profile section, rather than fail-fast: a regression in one script shouldn't block the dashboard refresh for the other 16. The `::warning::` annotation surfaces the failure in the run UI and the dashboard cell will show `ERR` until the next successful refresh.
