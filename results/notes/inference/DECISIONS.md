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
