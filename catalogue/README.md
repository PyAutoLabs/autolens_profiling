# Setup evidence catalogue

The companion [v2 catalogue](../dashboard/catalogue.json) organizes committed evidence by
**dataset → model → instrument → exact configuration**. It powers the project setup browser and provides the data foundation for the assistant;
it is not a new benchmark or an accepted baseline. The existing v1 JSON feed remains
available while Pulse migrates to the setup browser. The catalogue exporter does not run scientific scripts.

## Regenerate and validate

From the repository root (Python standard library only):

```bash
python scripts/misc/tooling/build_catalogue.py
python scripts/misc/tooling/build_catalogue.py --check
python scripts/misc/tooling/build_catalogue.py --check --validate-with ../PyAutoPulse
```

`build_dashboard.py` also builds/checks the catalogue and the page bound to its hash.
After changing catalogue inputs, run that combined builder before publication. The separate command reuses the
committed v1 render timestamp and producer revision for reproducibility. These identify
publication, **not measurement freshness**. The optional Pulse checkout runs the independently
maintained v2 validator against the index and every shard; CI requires this check.

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
- `hazard_bindings`, `recommendations`: explicit applicability and support, empty initially.
  No prose headline or static batch-size estimate becomes assistant advice automatically.

The exporter reads all declared JSON files and inventories all scientific script entry points,
including shared measurement tools. Unrecognized experiment formats remain `indexed_only`
with a reason, source path and SHA-256. This is honest coverage of the archive, not a claim
that every historical schema is normalized. Adding an adapter needs a representative fixture
and an explicit interpretation of units, scope, failure status and source pointers.

## Transport and lookup

`dashboard/catalogue.json` is an independently valid `profiling-summary` v2 document with
all setups, selected reference records, baseline plans, source inventory and extensions.
`evidence_shards` maps each measured setup ID to a relative `catalogue/shards/<id>.json`
path, SHA-256 of its exact UTF-8 bytes and record count. Additive `axes`, `devices`
and `precisions` lists summarize the actual shard records for selector labels; unknowns
remain explicit and no hardware is inferred from filenames. Resolve paths relative to the
catalogue URL (or `dashboard/` in git). Each shard is itself a valid v2 document containing
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
library component measurements stay in `scripts/lens/`. Legacy wrappers remain
until an explicitly approved removal after Brain and assistant migration;
there is no automatic expiry. Historical result paths remain unchanged.
