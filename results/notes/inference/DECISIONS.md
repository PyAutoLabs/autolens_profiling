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

1. **PositionsLH accumulation defect (§2.1)** — verified real; fix shipped as
   **PyAutoLens#699 / PR#700**. See the CP-1 entry below.
2. **BlackJAX NS (§2.2)** — "was fastest or comparable on MGE and may scale
   better — worth rerunning." Phase 2 stands as planned (re-tune against
   mainline blackjax ≥1.6.2; logZ-bias inner-steps scan is the sharpest
   pre-registered test; Gate A judged per model family).
3. through 6. — approved as planned (no changes requested to the remaining
   corrections, rules, phases, gates, or the critical path).

**Gate states after this entry:** Gates A–F all open. Phase 0 items (b)–(c)
and (e) outstanding; (a) complete (below); (d) complete (this commit).

---

## 2026-08-17 — CP-1 complete: PositionsLH accumulation fix shipped

**Decision:** Critical-path item CP-1 (Phase 0(a)) is **complete**. The
suspected accumulation defect in `AnalysisLens.log_likelihood_penalty_from`
(`autolens/analysis/analysis/lens.py:165-181` — loop overwrote the
accumulator and then added it to itself, returning 2× the last entry's
penalty and discarding the rest) was verified as a real bug and fixed.

**Record:** **PyAutoLens#699** (issue) / **PR#700** (fix + regression test).

**Consequence:** The positions arc (Phases 4, 5, 12) is unblocked on the
defect side. Per the plan, every positions-on target from here on bakes the
fixed penalty in; the Phase-1 reference posteriors must be built against the
fixed likelihood. Any pre-fix positions-on numbers encountered in old
artifacts are against an undocumented target and must not be compared with
post-fix rows (different `target_id`).

---
