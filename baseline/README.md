# Setup baseline (closed)

The setup-baseline campaign is **closed** (2026-11-01, autolens_profiling#503; likelihood
lifecycle ruling R10). It was never frozen, collected or accepted. The day-zero baseline replaces
it:

- [campaign.json](campaign.json) is the closure manifest: what superseded the campaign (tag
  `2026.10.25.1`, [day_zero.json](../catalogue/day_zero.json), the PyAutoPulse
  `snapshots/day_zero/lens.json` snapshot), the counts, and how day zero answers each of the
  draft's six unknowns.
- The day-zero rule per likelihood: [wiki/day_zero.md](../wiki/day_zero.md).
- Runtime estimates for the library (phase 5E):
  [runtime_estimates.yaml](../catalogue/runtime_estimates.yaml).

The 1,044-slot draft specification and its readiness contract are kept for reference in
[superseded/](superseded/) (`campaign_draft_v1.json`, `README_draft_v1.md`). Its screening tool,
`scripts/misc/tooling/baseline_readiness.py`, was retired with the closure; git history keeps it.
