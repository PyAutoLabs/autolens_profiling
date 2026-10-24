# Setup evidence catalogue

One exporter pass over the mirrored results tree writes two documents:

- [`dashboard/catalogue.json`](../dashboard/catalogue.json) is the **`profiling-summary@3`
  lifecycle catalogue** Pulse reads. Each record names its likelihood, instrument, R1
  `setup_id`, `host_class`, `model_variant`, measurement, `class` and `status`.
- [`dashboard/catalogue/setups.json`](../dashboard/catalogue/setups.json) and its shards are the
  **companion v2 setup catalogue**, organised by dataset → model → instrument → exact
  configuration. It powers this repo's setup browser and setup wiki and keeps every row,
  including those the lifecycle catalogue does not render.

Neither is a new benchmark or an accepted baseline. The exporter does not run scientific
scripts.

## Lifecycle labels (phase 0, autolens_profiling#442)

The rules live in [`catalogue_lifecycle.py`](../scripts/misc/tooling/catalogue_lifecycle.py):

- **Current**: the canonical cell file (not `history/`, `scans/` or `experiments/`) on a
  reference host class (`hpc_ral_cpu`, `hpc_a100`) for a planned baseline cell in
  [registry.json](registry.json) or a declared reference. Exactly one per key. `local_*` rows
  are never current (R7).
- **Size**: the standard source-pixel count is Delaunay 1250 and rectangular 1225 (35×35).
  With no standard-size row, the nearest-size row is current and carries a `size_fallback`
  limitation.
- **Superseded**: every other row on a key with a current row; a `local_*` row points at the
  reference-host row of the same setup. Rows whose key has no current row are counted in
  `coverage.excluded` and stay in the companion catalogue. `archive/` rows are only counted in
  `archive_record_count`.
- **Class**: the provisional §3.3 designations. A provisional Optimal row is `class: secondary`
  with `provisional_optimal: true` in `limitations` until a Pulse designation decision exists
  (R3); mixed precision is `experimental`. `planned_cells` carry the provisional class.
  [provisional_designations.json](provisional_designations.json) transcribes the §3.3 table
  row by row with its machine reading; `catalogue_lifecycle.py` and `coverage_matrix.py` both read
  it, so it is the only copy of the table.
- **Parity**: a cell's `parity.json` witness becomes a `measurement: parity` record; see
  [results/README.md](../results/README.md#parity-witness).
- **Coverage**: `coverage_matrix.py` renders [`dashboard/coverage.md`](../dashboard/coverage.md)
  (current / stale / superseded / missing per likelihood x class x measurement; a cell
  covered only by a `size_fallback` row is missing). Re-run it after `build_catalogue.py`.
- **measured_at**: a row with no recorded timestamp is dated by the UTC day its file was first
  committed to git (an upper bound on the measurement date), from
  [measured_at_first_commit.json](measured_at_first_commit.json), and says so in
  `limitations`. Refresh that file with `build_catalogue.py --refresh-commit-dates` (needs full
  git history); `--check` fails when an undated file is missing from it.

## Regenerate and validate

From the repository root (Python standard library only):

```bash
python scripts/misc/tooling/build_catalogue.py
python scripts/misc/tooling/build_catalogue.py --check
python scripts/misc/tooling/build_catalogue.py --check --validate-with ../PyAutoPulse
python scripts/misc/tooling/coverage_matrix.py        # then --check
```

`build_dashboard.py` also builds/checks the catalogue and the page bound to its hash.
After changing catalogue inputs, run that combined builder before publication. The separate command reuses the
committed v1 render timestamp and producer revision for reproducibility. These identify
publication, **not measurement freshness**. The optional Pulse checkout runs its independently
maintained validators: v3 on `catalogue.json`, v2 on the companion index and every shard; CI
requires this check.

## Registry and evidence policy

[registry.json](registry.json) declares:

- `families`, `model_patterns`, `model_aliases`: navigation vocabulary. Aliases group legacy
  names such as `pixelization` under `rectangular`; they never equate scientific configurations.
- `cells`, `devices`, `slots`: the intended baseline matrix, independent of files found.
- `sources`: explicit source globs and adapters. `overrides` are ordered prefix rules for
  streaming, hazards and inventory-only material. Overlapping source globs are rejected.
- `references`: exact source path and row JSON pointer plus a reason for choosing it. These
  select supported metrics in that row as **unreviewed candidates**, never accepted results.
  A missing/unsupported reference fails generation; no latest/fastest fallback exists.
- `hazard_bindings`, `recommendations`: explicit applicability and support.
  No prose headline or static batch-size estimate becomes assistant advice automatically.
- `hazard_discovery`: explicit editorial categories for unbound findings. A finding
  names model families or the shared component/method collection, with a rationale.
  These are discovery links, never exact setup/version applicability. Unlisted
  findings remain uncategorized; filename spelling does not determine relevance.

The exporter reads all declared JSON files and inventories all scientific script entry points,
including shared measurement tools. Unrecognized experiment formats remain `indexed_only`
with a reason, source path and SHA-256. This is honest coverage of the archive, not a claim
that every historical schema is normalized. Adding an adapter needs a representative fixture
and an explicit interpretation of units, scope, failure status and source pointers.

## Transport and lookup

`dashboard/catalogue/setups.json` is an independently valid `profiling-summary` v2 document with
all setups, selected reference records, baseline plans, source inventory and extensions.
`evidence_shards` maps each measured setup ID to a relative `catalogue/shards/<id>.json`
path (relative to `dashboard/`), SHA-256 of its exact UTF-8 bytes and record count. Additive `axes`, `devices`
and `precisions` lists summarize the actual shard records for selector labels; unknowns
remain explicit and no hardware is inferred from filenames. Resolve paths relative to the
dashboard page URL (or `dashboard/` in git). Each shard is itself a valid v2 document containing
that setup and all its measurements. Load only the selected setup's shard. Advice lives in
the root index; shards do not duplicate it. Root records are a subset of shard records;
deduplicate by record ID when combining them. `archive_record_count` counts unique records
across shards; `coverage.observed.records` counts records physically present in the document.

Evidence paths are **repository-root-relative**, unlike shard paths. Resolve them against
the same repository commit that supplied the index, never an independently fetched `main`.
Their `fragment` is an RFC 6901 JSON pointer into the original artifact, not a GitHub heading.
Every emitted metric is checked against that original value. Source inventory checksums
allow verification of the whole artifact. Generated obsolete shards are removed on rebuild;
only filenames matching the generated hash grammar are eligible for cleanup.

## Scientific identity and limits

Each legacy setup ID includes its source path, row pointer and recorded configuration.
This intentionally keeps rows separate when incomplete metadata prevents proving equivalence.
A shared instrument/model label alone does **not** permit comparison or combining runtime,
compile, memory and breakdown values into one synthetic run. Future baseline producers can
supply complete identities; this adapter never guesses them from today's defaults.

Recorded settings include source pixels, PSF shape, mask/image sizes, regularization,
transformer, solver, preloads, batch size and experiment arms where available. Missing values
are `null` with reasons. Local filesystem paths, commands and cache directories are redacted.
Measured software versions/revisions are preserved; evidence with none stays inventory-only.
Unknown precision, backend, host, measurement wall-clock, warmup and synchronization stay
unknown. Filename/config hints only use the existing explicit hardware/precision grammar.
All imported measurements are `unreviewed` and unqualified, even if legacy provenance exists.
No acceptance, temporal comparisons, drift score or current-performance assertion is produced.

Axes stay distinct:

| Evidence | Export meaning |
|---|---|
| Full likelihood timings | `runtime`, with statistic and batching distinction |
| Component/step totals and compile-probe steady calls | `breakdown`; never a full-likelihood headline |
| Trace, lower, compile, first-call and setup wall times | `compile`, distinct metric names; never silently summed |
| Observed Linux peak RSS | `memory`, `host_peak_rss`, MiB; **not GPU VRAM** |
| XLA static memory analysis | `static_memory_estimates` extension; not measured peak VRAM or validated advice |
| Failed streaming attempts | failed/unusable selection, no successful timing or memory record |

`planned_cells` explicitly reports every future baseline slot as `not_measured`, with no
record. It is an extension because v2 selections require measured software identity, while
the baseline stack has not been chosen. `coverage.expected.cells` therefore counts v2
selections only; `planned_slots` counts these future slots. No dummy version is invented.
Missing baseline slots are not evidence of failure, and archive rows do not fill them by age.

## Browsing measurements

The model/instrument overview lists recorded runs by runtime, breakdown,
compilation and memory before selecting an individual evidence run. Its optional
`axis_devices` shard metadata maps each axis to devices actually recorded for that
axis, keeping unknown device identity explicit. This is availability, not a
comparison or a merged scientific setup.

Initial selection prefers runtime evidence (an explicit reference candidate within
that axis first), then other measured axes. Ties use labels and stable IDs, never
speed or recency. Named axis buttons and the device selector narrow the run list;
explicit setup deep links are preserved. A populated panel opens even for
breakdown-only runs. Empty panels distinguish absence in the selected run from
absence throughout the model/instrument archive and link to other recorded runs.

Unbound findings carry optional `discovery` metadata (`models`, `shared`, `reason`).
The imaging rectangular fixture has model-specific discovery links; component,
matrix and shared-method studies have their own clearly qualified collection.
MGE mass-profile deflection findings are shared component evidence, not automatically
hazards of the imaging MGE light-model likelihood. No exact bindings are added
without evidence of matching configurations and measured versions.

`unbound_findings` retains deduplicated hazards with exact source pointers and an explicit
unknown-applicability reason. Versioned setup bindings can expose them as v2 `hazards`.
Recommendations must name exact setup IDs, library versions, constraints, limitations,
supporting record IDs, evidence and validation. Accepted recommendations require accepted
supporting records; the legacy exporter produces none. An assistant must therefore report
unknown coverage and limitations, not present candidates as validated settings or estimates.

The project browser consumes this transport; the Pulse front-page migration follows. The later baseline campaign defines and
measures trustworthy current setups before any records are promoted for recommendations.

## Scientific script routes

[script_routes.json](script_routes.json) maps legacy entry points to canonical
`scripts/<dataset>/<model>/<measurement>.py` paths. Datacube is a top-level
dataset family; `imaging/pixelized/` holds shared CLI mesh experiments and
latent cells live under `sersic/`. Shared tooling stays in `scripts/misc/` and
library component measurements stay in `scripts/lens/`. Compatibility wrappers were
retired after the caller audit in [migration.md](migration.md). Legacy paths remain
lookup aliases, not executable files; canonical targets must exist. Historical
result paths and cell IDs remain unchanged.

## Wiki and historical recommendations

The [setup wiki](../wiki/setups/index.md) is generated from this export and verified
shards, with compact dataset/model indexes and exact configuration pages. Run
`python scripts/misc/tooling/build_setup_wiki.py [--check]` after standalone catalogue
regeneration; the combined dashboard builder handles both. Campaign links are exact
setup-ID bindings in [wiki_bindings.json](wiki_bindings.json), never guessed from labels.

Five historical interferometer decision-matrix rules are transcribed as **unreviewed**
observations, narrowed to supporting SDP.81 configurations (the mask-extent observation
also cites recorded alma r3.5 analogues). Each carries exact supporting record IDs,
measured software and hardware by record, explicit bounds and historical acceptance
caveats. They do not grant baseline acceptance or extrapolate a NUFFT choice to
unmeasured setups. The journal's broader draft rules remain historical prose;
its solver/revision mismatch waiver is recorded, not promoted to catalogue acceptance.
