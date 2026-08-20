# Inference programme — decision log

Append-only gate/decision log for the PyAutoLens inference programme
(`PROGRAMME.md`). Each entry: date, what was decided, the evidence it rests
on, and the record it points to. Newest entries at the bottom. Never edit or
delete an entry — supersede it with a new one.

---

## 2026-08-17 — Programme plan approved (with notes)

**Decision:** The full phased R&D programme (state of play, phases 0–13,
gates A–F, benchmark/result schema v2, method-card knowledge structure,
source-change list, risk register, critical path, hypothetical decision tree)
is **approved** as the plan of record. Canonical copy committed as
`PROGRAMME.md` in this directory, per the programme's own knowledge rules
(nothing lives only in chat history or an artifact).

**Evidence:** 2026-08-17 planning pass — 6 parallel reconnaissance sweeps
over autolens_profiling, PyAutoFit@d6fef747, PyAutoArray@394514c0,
PyAutoLens, PyAutoMind/Brain/Memory, and the external blackjax/literature
ecosystem. No runs, no PRs, no source edits were performed during planning.

**Human review notes recorded at approval:**

1. **PositionsLH accumulation defect (§2.1)** — "worrying ... so defo check"
   → verified same day and fixed, **PyAutoLens#699 / PR#700**. See the CP-1
   entry below.
2. **BlackJAX NS (§2.2)** — "on mge blackjax NS was fastest or comparable so
   worth a look, and it may scale better, so worth rerunning." Phase 2 stands
   as planned (re-tune against mainline blackjax ≥1.6.2; logZ-bias
   inner-steps scan is the sharpest pre-registered test; Gate A judged per
   model family).
3. through 6. — state-of-play corrections, phase structure, schema, and
   critical path approved as written (no changes requested).

**Gate states after this entry:** Gates A–F all open. Phase 0 items (b)–(c)
and (e) outstanding; (a) complete (below); (d) complete (this commit).

---

## 2026-08-17 — CP-1 complete: PositionsLH accumulation fix shipped

**Decision:** Critical-path item CP-1 (Phase 0(a)) is **complete**.

- **Defect:** `AnalysisLens.log_likelihood_penalty_from`
  (`autolens/analysis/analysis/lens.py:165-181`) returned 2× the LAST
  penalty and discarded earlier entries — the loop overwrote the accumulator
  and then added it to itself. Verified by direct read and by the halved
  test pins.
- **Fix:** true sum over `positions_likelihood_list`; 3 test pins corrected
  (imaging single −44097289521.73 → −22048644768.18 = exactly half; imaging
  double-plane → −44140499627.75 = the true sum, distinct from the old
  2×-last value; interferometer −44097289569.23 → −22048644815.85);
  regression test added (analysis penalty == sum of per-object penalties).
- **Suite:** 538 passed, 0 failed.
- **Science impact:** inside-threshold behaviour unchanged;
  outside-threshold fence slope halves to the documented 1e8/arcsec;
  multi-plane penalty stacking corrected.

**Record:** **PyAutoLens#699** (issue) / **PR#700** (fix + regression test).

**Consequence:** The positions arc (Phases 4, 5, 12) is unblocked on the
defect side. Every positions-on target from here on bakes the fixed penalty
in; the Phase-1 reference posteriors must be built against the fixed
likelihood. Any historical run that had a positions penalty ACTIVE at its
recorded likelihood is not comparable to post-fix rows (different
`target_id`) — but none of the searches-framework benchmarks used positions,
so the existing truth bars are unaffected.

---

## 2026-08-18 — Scope change: af.NSS re-mainlined ahead of Gate A (human-directed)

**Decision:** The human directed early re-mainlining of `af.NSS` into
PyAutoFit as a proper first-class search, ahead of Gate A. Rationale: the
search's removal (PyAutoFit#1356/#1357) was driven solely by unshippable
git-fork dependency pins, and nested sampling has since merged into
**mainline blackjax 1.6** — the removal reason no longer exists. **Gate A is
unchanged in meaning:** it still decides whether NSS becomes the principal
GPU nested-sampling *baseline*; this decision only restores the search's
availability.

**Record:** PyAutoFit#1491 (issue) / **PR#1492** (implementation, pending
human merge). Port pre-validated on CPU in the cloud session before any
source change (see `phase_00_unblocking/RESULTS.md` for the environment
half): the `autofit_workspace_developer/searches/nss/` stash restored with
the mainline port — `blackjax.ns.utils` imports, `SliceInfo.num_steps` →
`num_expansions`, chunked builder recomposed from mainline public helpers
(no `update_strategy=` kwarg in mainline). 17/17 unit tests pass; 2D
analytic toy through `search.fit` gives logZ −4.558 ± 0.078 vs analytic
−4.605; the chunked GPU-memory path is bit-identical to unchunked at fixed
seed. Optional-extra blackjax floor bumped ≥1.2.0 → ≥1.6.2 (the §7 floor
item, also pulled forward).

**Consequences:** Phase 2 no longer needs a profiling-local NSS runner
wired by hand — it can drive `af.NSS` directly once PR#1492 merges (the
plan's "profiling-local runner first" hedge is obsolete). Follow-ups:
workspace tutorial restoration (stash checklist item 5) and the Phase 2
re-benchmarking campaign itself (GPU, laptop/RAL).

---

## 2026-08-19 — Phase 0 gate satisfied: (a) + (b) both landed; Phases 1–2 unblocked

**Decision:** Phase 0's gate condition — "nothing else proceeds until (a)
and (b) land" — is **satisfied**. Phase 0(b) is complete in all three
environments; the programme's downstream phases (Phase 1 targets registry,
Phase 2 NSS re-tuning) are unblocked.

**Evidence:**

- PyAutoFit **PR#1492 merged 2026-08-18** — `af.NSS` is a first-class search
  on main (blackjax optional-extra floor ≥1.6.2).
- **Local venv**: blackjax 1.5 → 1.6.2; `nss_smoke.py` (extended with an
  `af.NSS` end-to-end analytic-evidence check, the surface Phase 2 drives)
  PASS on all three checks; `af.NSS` unit suite 17/17 against the local
  checkout.
- **RAL stack**: was still the obsolete 2026-01 fork build
  (0.1.0b1.dev86+g795058671, §2.2's stale pin) — upgraded to 1.6.2 with a
  pre-upgrade freeze snapshot; library mains re-synced; the same smoke PASS
  on all three checks (login node, CPU). Seeded results identical to local.
- Record: `phase_00_unblocking/RESULTS.md` (2026-08-19 sections), issue #146.

**Still open in Phase 0:** (c) is **partial** — the stranded *local*
fork-NSS A100 artifacts (the §1.2 delaunay / pixelization / mge-2nd-run
rows) are now committed, but the RAL NFS harvest (SMC warm arms job 331058,
NaN-counter split arms 335003-5, Nautilus pixgrad baseline logs) remains.
(c) was never part of the gate condition and does not block Phases 1–2.

**Next per the critical path (§9):** the laptop-GPU MGE half of CP-2 (first
Phase 2 work item), then CP-3 (Prodigy MGE reliability scan ± PositionsLH —
needs the PyAutoFit per-lane-best change first) and CP-4 (slogdet A/B, can
start any time).

---
