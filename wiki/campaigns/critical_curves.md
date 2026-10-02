# Critical curves — geometry, dispatch and cost

**Status:** open
**Question:** Which critical-curve engine/settings preserve cluster field coverage and multi-plane geometry without avoidable compilation cost?
**Pre-registered rule:** Match every reference component within two fine-grid pixels (0.1 arcsec control, 0.25 arcsec cluster); analytic control and coarse/fine convergence; errors/timeouts never pass.
**Verdict:** NO-GO for blindly routing cluster plots to default zero-contour; retain marching squares pending grid-cap, seed-coverage and path-completion contracts.
**Headline:** CPU fp64 pinned witness: fixed ±3 arcsec auto seeds miss both cluster tangential curves; explicit far-source curve has a 20.65 arcsec closing chord.
**Library PRs:** None; no production change in phase 3a.
**Profiling PRs:** Phase-3a shipping resumed; task: autolens_workspace_test#337.
**Ledger:** [critical_curves_dispatch.md](../../results/notes/critical_curves_dispatch.md)
**Mind contract:** epic `cluster-strong-lensing`; `active/critical_curves_dispatch_audit.md`
**Next:** Ship/review phase 3a; then issue one bounded grid-cap fix. Full workspace smoke now passes 33/33.

## Why this campaign

The Source & Cluster arc needs trustworthy critical-curve overlays and boundaries
before its magnification-map work. The historical phase-3 prompt mixed engine
dispatch, JIT, documentation and cluster numerical coverage. This campaign gives
the experiments a cumulative home, as requested by the human on 2026-10-02.
It belongs under `scripts/lens/`, because the measured object is a LensCalc
operation on a grid, not a dataset likelihood. Stable numerical contracts belong
in workspace/library CI; this wiki accumulates research and robust-settings advice.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Source & Cluster 3a | 2026-10-02 | Is existing dispatch safe at cluster scale? | Full component match and two-pixel distance bound | Seed and path-completion failures; grid-cap field distortion; outer-JIT unsupported | local CPU, no HPC | pending |

## What shipped and where it is

No production fix has shipped from this campaign. Earlier zero-contour work is
an input, not evidence that cluster plots are safe. Keep PR/merge/release records
here as the subsequent bounded phases ship.

## Open / parked / drafts

- First bounded candidate: preserve field extent when the evaluation-grid cap
  activates, with a numerical CI invariant.
- Then define cluster seed coverage and tracing completion before engine dispatch.
- Reconcile the unissued `cluster_curves_engine_dispatch.md` Mind candidate; its
  direct selector-only approach is insufficient for these numerical witnesses.
- Broader cluster morphologies, multiple mass redshifts, GPU and gradients are
  outside phase 3a; no production completeness claim is made.

## Caveats

Synthetic cored host/member and two source planes are not a realistic cluster
validation corpus. Explicit seeds come from the marching reference. A caustic
can meet a distance tolerance despite an invalid/incomplete parent curve.
Runtime version metadata is old; use the JSON's Git/source hashes to identify
what actually ran. No work on the dropped Cortex phase 11 is implied.

## Journal

### 2026-10-02 — Birth from Source & Cluster phase 3a

The human assigned research and accumulating wiki knowledge to autolens_profiling,
with short examples in autolens_workspace_test for CI. One task (#337) covers
both repositories. The initial 14-worker matrix separates automatic/explicit
seeds, cold/warm costs, disabled-JIT timeout and outer-JIT failure. Final results
and validation are recorded in the linked ledger, with an immutable JSON/PNG pair.


The pinned artifacts validate against analytic controls and source hashes. Four
workers timed out at 120 seconds; retained partial curves support failure
witnesses but are not successful experiments. Both explicit cluster radial legs
remain unmeasured in this run. See the ledger for collection provenance, CI
validation and the shipping gate. No successor issue has been queued.

Shipping checkpoint: Heart RED (three library checkouts behind origin) stopped
PR creation. Full smoke retry passed five scripts before stopping; remaining
scripts are unrun. Both research and CI worktrees are preserved for resumption.

Resumption: all 33 workspace smoke scripts passed; the human acknowledged the
remaining unrelated manifest YELLOW and authorized `/prm and continue`. The
previous incomplete-smoke checkpoint is superseded; historical evidence remains
unchanged.
