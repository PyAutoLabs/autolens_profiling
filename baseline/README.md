# Setup baseline readiness

This is a **draft campaign specification**, not a baseline. Phase 6 prepares
collection and review; it authorizes no jobs and accepts no measurements.
Pulse owns the pending [campaign task](https://github.com/PyAutoLabs/PyAutoPulse/issues/17).
The project owns [campaign.json](campaign.json), its dry-run tooling and the
[campaign journal](../wiki/campaigns/setup_baseline.md).

## Inspect without running profiles

From the repository root, with standard Python only:

```bash
python scripts/misc/tooling/baseline_readiness.py enumerate
python scripts/misc/tooling/baseline_readiness.py validate
python scripts/misc/tooling/baseline_readiness.py report --receipt /path/to/receipt.json
```

All commands print JSON to stdout, read only explicitly named inputs and import
no scientific library. They have no execution, submission, write or promotion
option. Redirect stdout to a scratch file if desired; do not overwrite results.
`enumerate` exits 0 for a structurally valid draft; `validate` and `report` exit
1 when blocked, and malformed input exits 2. A valid empty receipt reports every
applicable cell missing. No archived files are discovered or converted.

## Proposed matrix and methods

The specification pins the catalogue and route registry by SHA256. The 29
catalogue cells expand to CPU, GPU and A100 × fp64/mixed × six measurements:
**1,044 slots**. CPU device-allocator memory is explicitly not applicable (58
slots); host RSS remains applicable. Two compile rows distinguish a cold and a
warm persistent cache. The initial enumeration has 620 unverified slots and
366 unsupported slots. The existing runtime sweep exposes only 22 of the 29
cells; a script path alone does not prove that its instrument or protocol is
supported. `runtime_sweep_dispatch` identifies that narrower existing dispatch.

- Runtime: synchronized single-call timings, batch size 1, three warmups, ten
  repetitions, raw seconds retained; median is a reporting convention.
- Breakdown: synchronized named component samples, ten repetitions, explicit
  exclusive/cumulative accounting. Components never become full-call runtime.
- Compile: `jit` only for this proposal, a fresh process per repetition, no
  within-process warmup. Cold uses a fresh empty isolated persistent cache;
  warm uses a separately seeded matching cache with hit evidence. Record the
  tracing/lowering/compilation boundaries. The current compile probe's cell
  builder is **unavailable**; neither archived coverage nor this plan repairs it.
- Host memory: OS peak RSS over an isolated full workload, including setup and
  compile, converted to MiB with the OS unit recorded; starting RSS separate.
- Device memory: peak allocated bytes from a verified allocator high-water
  counter reset after warmup and read after synchronization. A reservation,
  static memory estimate or one instantaneous snapshot is not a peak. A backend
  without that facility stays unsupported until a suitable method is reviewed.

The proposed reference policy uses one thread, fixed affinity, no competing GPU
processes and background host load per allocated CPU ≤0.1. Hardware identities,
RAM/VRAM, driver and environment fingerprints remain unresolved. The background
load method must separate campaign work from other load and be recorded in the
frozen environment documentation; the screen only checks submitted observations
against the bound. **All production CPU arrays use `ral`, never `gpu`**; GPU
reference arrays use `gpu` and must actually allocate GPU TRES. This is a future
collection constraint, not a submit command.

## Freeze before collection

The shipped draft intentionally fails readiness. A later approved task must:

1. Choose immutable 40-character project and library Git revisions, capture an
   environment lock SHA256 (Python, JAX/jaxlib, dependency versions, backend and
   build flags), and verify clean source trees on the actual host. Record the
   reference host, processor/accelerator, RAM/VRAM, driver, affinity and threads.
2. Fill **every** setup's full `configuration` and `input_sha256` map. Required
   groups are geometry, model, solver, parameters and seed. Include mask and
   pixel/grid/PSF shapes, over-sampling, visibility/channel counts, mesh shape,
   sparse/dense mode, regularization and actual parameter vector as applicable.
   Instrument names do not supply these values. Explicitly explain inapplicable
   fields. Hash actual input bytes; a random seed is not a file hash. Never infer
   unknown fields from old results or claim an unmeasured setup is equivalent.
3. Choose an independent correctness reference (SHA256) and absolute **and**
   relative error tolerances per setup. The proposed screen requires both bounds
   to pass; document the relative-error normalization and check the likelihood,
   reconstruction and any gradient required by that setup. A self-comparison is
   not an independent witness. Thresholds require scientific review.
4. Review proposed methods, host-load sampling, warmups/repetitions, cache
   isolation/seed/hit witnesses and 1,800-second CPU run cap. Instrument and
   validate producer support before collecting. Missing routes, the compile
   builder and peak-memory instrumentation are separate implementation work.
5. Freeze the specification with `status: frozen`, the collection authorization
   reference and UTC time window. Preserve the exact canonical spec SHA256 from
   `enumerate`; any change invalidates earlier receipts. Resolve all nulls and
   re-run validation. Frozen means a declared contract, not authorized by this
   CLI: obtain the separate human compute decision before any collection.

The project revision can refer to the reviewed producer commit preceding the
separate frozen manifest commit; the spec SHA256 binds that later manifest.
The inventory hashes pin coverage; catalogue edits require explicit re-review.
A partial report may screen a subset but cannot hide the remaining missing cells.

## Evidence receipt and acceptance report

The future producer must write fresh observation JSON under
`results/campaigns/<campaign_id>/`. This phase does not add or execute a producer.
An explicit receipt has the following shape (the empty version is runnable now):

```json
{"schema": "baseline-evidence-receipt", "version": 1, "records": []}
```

Each record locator has `cell_id` from enumeration, `artifact` relative to that
campaign directory, and the artifact's byte `sha256`. Absolute paths, traversal,
symlink escape, duplicate cells, unknown cells and changed hashes are rejected.
Files copied from archives are still archives; renaming them or inventing a
campaign label does not create fresh evidence.

Each artifact must carry `schema: baseline-observation`, integer `version: 1`,
`campaign_id`, `spec_sha256`, `cell_id`, `origin: fresh-campaign`, UTC
`observed_at`, and `context`. The latter contains exactly the spec's software,
setup, hardware, device, precision, protocol and measurement objects (see
`context()` in the [screening tool](../scripts/misc/tooling/baseline_readiness.py)).
Observed context must reflect the actual run, never blindly copy planned values.
The timestamp must fall inside the frozen window and not in the future.

Measured observations have:

- `outcome: measured`, boolean `fresh_process` and `synchronized`, and integer
  `warmup_calls` (zero for compile, at least the configured count otherwise).
- `resources.load_per_allocated_cpu`: one finite nonnegative observation per
  repetition. GPU rows also need `other_gpu_processes`, all integer zero.
- `correctness.reference_sha256`, `max_absolute_error`, `max_relative_error`.
- `samples`: a mapping from the measurement's metric to raw finite nonnegative
  samples, exactly the configured repetition count. Breakdown instead names
  each component and declares `component_accounting: exclusive|cumulative`.
  Compile also declares `cache_state_witness: cold|warm`.

A fresh CPU runtime timeout can instead use `outcome: cpu_timeout`,
`elapsed_seconds` at least the frozen limit, **one** load observation and no
`samples`. It becomes only `cpu_exclusion_for_human_review`. It supplies no
likelihood timing, does not establish GPU usability, and does not exclude
breakdown/compile/memory by inference. Existing `.unusable.json` markers remain
historical context and cannot fill this new campaign.

The report checks declarations and artifact integrity, not physical execution.
A matching receipt can only become `candidate_for_human_review`; even an entirely
filled report has **`accepted: false`**. Producer honesty, full configuration
adequacy, actual hardware/cache/synchronization and reference independence still
need raw logs and human verification. A renamed archive or falsified timestamp
cannot be ruled out cryptographically by this tool. Unsupported cells remain
missing; fix their producer in a separate task before re-freezing.

Later collection and later scientific acceptance are separate decisions. A human
reviews missing/excluded cells, correctness witnesses and provenance before any
baseline promotion task is authorized. Do **not** run legacy `build_baseline.py`
for this process: it snapshots existing result trees and is not an acceptance
gate. Existing result bytes, pins and temporal metadata remain unchanged; no
temporal charts are enabled by this work.
