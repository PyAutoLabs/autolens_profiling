# Certified positive solver

**Status:** parked
**Question:** Can a certified active-set positive solver replace the default positive-only solve, including under `jit(vmap)`?
**Pre-registered rule:** phase C1: the phase-B 1e-9 own-composition per-lane gate
**Verdict:** phase A shipped opt-in; phase B passed its gate on seeded draws; phase C1 gate FAILED (not loosened); scalar-default flip retired 2026-09-24 as superseded by phase C
**Headline:** C1 uncertified-lane rate 1.9–4.9 % overall on real Nautilus batches (A100 replay job 350768)
**Library PRs:** PyAutoArray#567 (release not verified)
**Profiling PRs:** #299, #302, #309
**Ledger:** [certified_solver_policy_phase_b_2026_09.md](../../results/notes/certified_solver_policy_phase_b_2026_09.md), [certified_solver_phase_c1_lane_rate_2026_09.md](../../results/notes/certified_solver_phase_c1_lane_rate_2026_09.md)
**Mind contract:** epic `certified-positive-solver`; `complete/2026/09/certified-positive-solver.md`, `certified-solver-phase-b.md`, `certified-solver-phase-c1-lane-rate.md`, `certified-solver-scalar-default-flip.md`
**Next:** C2 draft `draft/feature/autofit/certified_solver_batched_guard_c2.md`

Backfill pending (phase 2 of epic profiling-research-wiki).

- [certified_solver_policy_phase_b_2026_09.md](../../results/notes/certified_solver_policy_phase_b_2026_09.md)
- [certified_solver_phase_c1_lane_rate_2026_09.md](../../results/notes/certified_solver_phase_c1_lane_rate_2026_09.md)

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.
