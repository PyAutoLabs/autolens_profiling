# dashboard/ — generated, never hand-edited

Every file here is rendered by `scripts/misc/tooling/build_dashboard.py` from the result JSONs
under `results/{runtime,breakdown,simulators,lens}` (autolens_profiling#345), plus the
[declared catalogue sources](../catalogue/registry.json) for the companion v2 export. `lint.yml` runs
`build_dashboard.py --check`, so a PR that adds rows without re-rendering fails; `profile.yml`
re-renders on release. `pages_dashboard.yml` publishes the folder as-is at
<https://pyautolabs.github.io/autolens_profiling/>.

| File | Reader | Contract |
|---|---|---|
| `index.html` | a human | the setup browser (shared theme, inline code, catalogue and lazy shards) |
| `series.json` | `index.html` | the trend data, `schema_version` 1 — this project's own shape |
| `state.json` | this project's own Pages badge | `PyAutoBrain/board/state_schema.json` v1; project label `autolens_profiling` |
| `catalogue.json` + `catalogue/shards/*.json` | setup browser / assistant foundation | companion **v2**, [catalogue documentation](../catalogue/README.md) |
| `summary.json` | the **PyAutoPulse** organ | **`profiling-summary` v1** — below |

The Brain board and cockpit read the **PyAutoPulse** organ feed (`PyAutoPulse/state.json`);
this project's feed retains its drift items, triage prompts and Pages link.

## `summary.json` — the `profiling-summary` v1 read contract

The project → organ interface of the layered design
(`PyAutoBrain/docs/research/profiling_inference_organs.md`, Brain #444): this project produces
measurements and owns their meaning; the PyAutoPulse organ registers this file's path, reads it at
one resolved commit, validates the **exchange** contract and displays. It never recomputes a ratio,
moves a pin, merges unmatched timings or issues a verdict. Everything in the file is derived from
`series.json`'s scan; nothing is measured or judged here that is not already on the page.

### Envelope

| Field | Meaning |
|---|---|
| `schema`, `version` | `"profiling-summary"`, `1`. A breaking change bumps the integer; the organ refuses versions it does not support |
| `project`, `scope` | `"autolens_profiling"`, `"release-runtime"` — per-call run time per PyAutoLens release across the four result sections |
| `generated_at` | render time, UTC `Z`; the same stamp as `series.json`'s `generated` |
| `evidence_updated_at` | newest **release date** encoded in a measured library version (`2026.9.27.1` → `2026-09-27T00:00:00Z`), or `null` with `evidence_updated_at_reason`. Result rows do not record measurement wall-clock |
| `valid_until` | always `null`: this project declares no freshness policy. The organ shows the age and "freshness policy unspecified"; it must not invent a TTL |
| `producer_revision` | `git rev-parse HEAD` of this checkout at render — the producer **code** revision, never the hash of the commit that first contains the file (which cannot know itself). `--check` reuses the committed value so only data can make the folder stale. `null` when not rendered inside a git checkout |
| `comparison_policy` | the drift rule the comparisons carry: `id` `runtime-drift-2x-1ms`, `ratio` 2.0, `floor_s` 0.001, the reference host and load-average cap from `hpc/release_sweep.conf` |
| `coverage.expected` | the sections scanned; `cells` is `null` with a reason (there is no declared cell matrix — the scan is the inventory) |
| `coverage.observed` | counts of series, points (= records), releases and cells |
| `coverage.excluded` | the **refused** rows (load average above the cap): `id`, `reason`, `evidence`. Never in `records` |
| `records` | one per plotted point — below |
| `comparisons` | one per series — below |
| `limitations` | explicit sentences the organ must show beside the data |

### Records

`id` is `<series key>@<library version>` (`runtime:imaging/delaunay/hst:hpc_a100_fp64@2026.9.27.1`),
unique within the file. Each record has:

- `axis: "runtime"`, `unit: "s"` — a memory value or a compile time can never enter this feed;
- `measurement`: `single_jit_s` (the per-call headline, `build_readme.py`'s ladder) and
  `vmap_per_call_s` (`null` where the row has no vmap block);
- `identity`: `section`, `cell`, `config`, `sparse`, `tier` / `device` / `precision` parsed from
  the sweep config label (`null` + `reason` outside the grammar), `backend` as the row recorded it,
  `library: "PyAutoLens"`, `library_version`, `release_date`;
- `provenance`: `host`, `job`, `loadavg`, `has_provenance`, `qualified`, `reason` — exactly what the
  page plots hollow or solid, and why;
- `evidence`: `path` (repo-relative, resolve it against the commit the organ captured) and
  `fragment` (the config key inside a `comparison.json`, else `null`).

### Comparisons

The producer's own drift badge for each series, so the organ **displays** drift candidates rather
than computing them: `comparison_key` (= series key), `policy`, `axis`, `metric`, `baseline` and
`candidate` (the two newest releases), `ratio`, `status` ∈ `drifted | improved | flat |
insufficient`, `qualified` (both endpoints qualified under the release-sweep pin) and `reasons`
(one release only; no headline; an unqualified endpoint and why). A `drifted` row with
`qualified: false` is a contextual flag, not evidence of regression — the distinction the organ
must keep visible.

### What the producer refuses to publish

`validate_summary()` runs before every write and under `--check`: unknown schema/version, a
missing required field, a duplicate record id or comparison key, a non-finite number (`NaN` /
`inf` are refused, never written), a malformed or non-UTC timestamp, `valid_until` before
`generated_at`, an absolute or `..` evidence path, a status or policy outside the contract. A valid
file with no points is a valid **empty** feed: it says "no measurements", not "all measurements
passed".

Adding a field is additive and allowed under version 1; changing the meaning of one is a
version bump, reader first (the organ), producer second (this script).

## Setup browser

`index.html` now opens with dataset/model disclosures, then instrument and exact
configuration selectors. Runtime is visible first; breakdown, compilation, memory,
configuration and source evidence expand on demand. Bars are linear within a setup
and unit. Missing coverage is explicit, and archive evidence remains unreviewed.
The v1 JSON files above retain their historical meaning for existing consumers;
temporal charts are no longer the default human view.

`scripts/misc/tooling/setup_page.py` renders the page with the real
`PyAutoBrain/board/_theme.py` (set `PYAUTO_BRAIN` or use the supported sibling/grouped
checkout). The lint and profile workflows check out that dependency. Source UI assets
live in `catalogue/browser.js` and `catalogue/browser.css`, inlined by generation.
The page binds the exact catalogue bytes by SHA-256 and validates each selected
shard against its manifest; a mixed publication shows a reload/error message.
Source links use the catalogue producer revision. URL hashes retain selections for
sharing and browser history; the original evidence is also available without JS.
Serve `dashboard/` over HTTPS or localhost for Web Crypto and fetch support.

Real Chromium checks run in CI via `scripts/misc/test/browser_setup_page.cjs` with
Playwright 1.63.0 on `NODE_PATH`: values, navigation, history/deep links, keyboard,
320/390/768/1280px, dark mode, corrupt/retry, late responses, missing catalogue and
no-JS fallback. The test starts and stops its own local HTTP server.
