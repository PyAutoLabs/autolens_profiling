# Setup baseline readiness

**Status:** draft
**Question:** Which fresh, fully specified setup measurements can support a later baseline decision?
**Pre-registered rule:** Draft only; resolve and freeze the [campaign specification](../../baseline/campaign.json), then obtain separate compute authorization. Structural screening never accepts a baseline.
**Verdict:** None; no measurements collected or scientifically accepted.
**Headline:** 1,044 proposed slots; 58 CPU device-memory slots not applicable; no measured headline.
**Library PRs:** None; no library API changes.
**Profiling PRs:** Implementation tracked in [issue #384](https://github.com/PyAutoLabs/autolens_profiling/issues/384).
**Ledger:** No measurement ledger yet; [readiness contract](../../baseline/README.md) only.
**Mind contract:** `profiling-baseline-readiness`; parent `draft/feature/pyautopulse/profiling_setup_browser.md`, Phase 6.
**Next:** Resolve setup/hardware/revision decisions and capability gaps through the [Pulse campaign task](https://github.com/PyAutoLabs/PyAutoPulse/issues/17); collection requires a later human authorization.

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
deferred. The [readiness guide](../../baseline/README.md) defines the proposed
matrix, protocols, receipt format and later decision boundaries.

## Journal

### 2026-10-06 — Phase 6 readiness implementation

Recorded a proposed matrix and explicit unknowns, plus dry-run enumeration and
acceptance-report generation without execution or promotion. Pulse owns the
pending campaign intent; this project owns the specification and evidence rules.
