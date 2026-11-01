# Setup baseline readiness

**Status:** complete
**Question:** Which fresh, fully specified setup measurements can support a later baseline decision?
**Pre-registered rule:** Draft only; resolve and freeze the [campaign specification](../../../baseline/superseded/campaign_draft_v1.json), then obtain separate compute authorization. Structural screening never accepts a baseline.
**Verdict:** Closed 2026-11-01 without collection (ruling R10, autolens_profiling#503): superseded by the day-zero baseline, which answers the draft's six unknowns ([closure manifest](../../../baseline/campaign.json)).
**Headline:** Day zero `2026.10.25.1`: 480 stamped records over 116 cells, 39 designations, 232/232 planned cells current, 0 optimal cells not measured (the draft proposed 1,044 slots and ran none).
**Library PRs:** None; no library API changes.
**Profiling PRs:** [issue #384](https://github.com/PyAutoLabs/autolens_profiling/issues/384) (draft); closure #503.
**Ledger:** [closure manifest](../../../baseline/campaign.json); [day-zero rule pages](../../day_zero.md); draft [readiness contract](../../../baseline/superseded/README_draft_v1.md).
**Mind contract:** epic `likelihood-lifecycle` phase 3 (R10); `complete/2026/11/likelihood-lifecycle-p3-setup-baseline-close.md` (was `profiling-baseline-readiness`).
**Next:** none (closed).

## Why this campaign

The setup browser retains historical evidence with its qualification and unknowns.
Those archives are not a fresh baseline. This campaign defines the missing
configuration and measurement contract before a later collection proposal.

## Phases

| Phase | Question | Rule | Result | Jobs |
|---|---|---|---|---|
| Readiness | Is the collection contract explicit? | Enumerate every declared cell and keep unknowns blocked | Draft specification and read-only report | None |
| Collection | Can the frozen matrix be measured reproducibly? | Separate compute authorization after capabilities and choices resolve | Unstarted | None |
| Acceptance | Are the fresh observations scientifically suitable? | Human reviews provenance, witnesses and exclusions | Unstarted | None |

## Caveats

The draft has unresolved revisions, environments, hardware, full setup settings,
input hashes and correctness references/tolerances. The compile cell builder is
unavailable. Peak-memory instrumentation and method compliance are unverified.
A script path is not executable campaign support. CPU production arrays use
`ral`; CPU timeout exclusions require fresh campaign evidence. Archived results
cannot fill missing cells. Reports screen submitted declarations and hashes;
they do not prove physical execution or grant scientific acceptance.

No results, baseline pins or time-series metadata change. Temporal charts stay
deferred. The [readiness guide](../../../baseline/superseded/README_draft_v1.md) defined the proposed
matrix, protocols, receipt format and later decision boundaries.

## Journal

### 2026-10-06 — Phase 6 readiness implementation

Recorded a proposed matrix and explicit unknowns, plus dry-run enumeration and
acceptance-report generation without execution or promotion. Pulse owns the
pending campaign intent; this project owns the specification and evidence rules.

### 2026-11-01 — Closed: superseded by day zero (R10)

The likelihood lifecycle epic replaced this protocol rather than resuming it (ruling R10). The
day-zero baseline (tag `2026.10.25.1`, autolens_profiling#500, frozen as PyAutoPulse
`snapshots/day_zero/lens.json` in PyAutoPulse#85) answers the six unknowns: the released tag
pins the revisions; reference hosts `hpc_ral_cpu` and `hpc_a100` only; each cell's script, dataset,
parity witness and fit figure are its settings and correctness evidence; compile is not a class
axis; mixed precision is experimental. Coverage: 232/232 planned cells current, no optimal cell
not measured. `baseline/campaign.json` is now the closure manifest; the 1,044-slot draft and its
readiness contract moved to `baseline/superseded/`; `baseline_readiness.py` was retired
(autolens_profiling#503). Runtime estimates for phase 5E: `catalogue/runtime_estimates.yaml`.
