# Certified positive solver

**Status:** parked
**Question:** Can a certified active-set positive solver replace the default positive-only PDIP solve in production, including under the `jax.jit(jax.vmap(fn))` composition that Nautilus runs?
**Pre-registered rule:** per phase (see the table); the common numerical gate for B and C1 is 1e-9 relative per lane against the library-PDIP reference of the *same* composition; C1 also fixed the C2 build rule (projected guarded cost at B=20 ≥ 15 % below the best zero-code option, rate near zero).
**Verdict:** phase A shipped opt-in (default stays `pdip`); phase B passed 20/20 on seeded draws and the policy was adopted 2026-09-24 (vmap keeps PDIP until a guard exists); scalar-default flip retired 2026-09-24, never issued; phase C1 gate FAILED as pre-registered and was not loosened, but the human (2026-09-25) let C2 proceed on 3/4 build-rule cells under a new near-peak gate (Δ = 100 nats, 0.1 nat pin), rectangular pix1 staying on library PDIP.
**Headline:** C1 uncertified-lane rate 1.9–4.9 % overall (0 in the late half) on real Nautilus batches at n_batch=20, A100 80GB fp64, replay job 350768 (captures 350659, 350766).
**Library PRs:** PyAutoArray#567 (released 2026.9.26.1).
**Profiling PRs:** #299, #302, #309.
**Ledger:** [certified_solver_policy_phase_b_2026_09.md](../../results/notes/certified_solver_policy_phase_b_2026_09.md), [certified_solver_phase_c1_lane_rate_2026_09.md](../../results/notes/certified_solver_phase_c1_lane_rate_2026_09.md)
**Mind contract:** epic `certified-positive-solver` (not in epics.md); `complete/2026/09/certified-positive-solver.md` (A), `certified-solver-phase-b.md`, `certified-solver-scalar-default-flip.md` (retired), `certified-solver-phase-c1-lane-rate.md`.
**Next:** C2 guard via `draft/feature/autofit/certified_solver_batched_guard_c2.md` (its `Blocked-by:` release of PyAutoArray#567 is now satisfied by 2026.9.26.1).

## Why this campaign

The fixed-lens-light work ([Fixed lens light](fixed_lens_light.md), [HST GPU residue](hst_gpu_residue.md))
showed that on a source-only system a certified active-set solve returns the exact constrained
optimum in 4.21 ms (Delaunay, pass 2) / 11.05 ms (rectangular, pass 7) against a 25.8–28.3 ms S3 PDIP
row (A100, autolens_profiling#248). The human asked on 2026-09-15 to move the harness solver
into the library, with structure-aware dispatch (mapper-only vs MGE-inclusive) and a CPU check.
Phase A (PyAutoArray#566) scoped that to an opt-in JAX solver and reserved the default decision for
phase B, because the `lax.cond` PDIP fallback becomes a select under `vmap` and both solvers run.

Phase B (autolens_profiling#300) pre-registered its gate at `24e734c` before submission: per lane,
1e-9 relative against the library-PDIP reference of the same composition, cross-composition recorded
only, a `none`-fallback row eligible only if every lane certifies. Phase C1 (autolens_profiling#304)
reused that gate unchanged on lanes a real `af.Nautilus` run proposed, and wrote the C2 build rule
into the sliced prompt before the captures ran.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| A library opt-in | 2026-09-15 → 09-23 | Ship the certified active-set solver in PyAutoArray with backend/composition dispatch | selected routes preserve reconstruction/evidence and avoid material regression; unproven cases keep the baseline (prompt Acceptance) | shipped opt-in behind `positive_only_solver`, JAX mapper-only; MGE-inclusive and NumPy keep their solvers; PDIP agreement 8e-12, gradient FD 3e-9; RTX 734.9 → 566.1 ms | laptop (RTX) | PyAutoArray#567, #299 |
| B policy on seeded draws | 2026-09-24 | Which solver/fallback under `jit(vmap)` vs `jit(fn)` at B = 4/8/16, Delaunay + rectangular? | 1e-9 rel per lane vs same-composition library PDIP (`24e734c`) | PASS 20/20, worst 5.21e-10, 0 uncertified; certified+none B16 1.50x (Delaunay) / 1.25x (rect) over PDIP vmap; certified+PDIP slower than PDIP at every B; scalar certified 1.7x / 1.4x; policy adopted | 350588 | #302 |
| Scalar-default flip | 2026-09-24 | Flip the scalar `jit(fn)` default to certified + PDIP fallback | witness ≤ 1e-9 vs pre-flip PDIP (never run) | retired, not implemented (three reasons, journal) | none | none |
| C1 lane rate | 2026-09-24 → 09-25 | Uncertified-lane rate and matched timing on real Nautilus batches, B = 16/20/50/100 | phase-B gate unchanged (`be01a52`); build C2 only if guarded cost at B=20 ≥ 15 % below best zero-code option with rate near zero | gate FAILED (24/24 timed tasks, 6 B=100 OOM); rate 1.9–4.9 %, 0 late half; B=50 `jit(vmap)` wrong on 44–50/50 lanes; build rule 3/4 cells (rect pix1 11 %) | 350659, 350766 (captures), 350768 | #309 |
| C2 guard | draft (filed 2026-09-24) | Does a host-side re-run of uncertified lanes give certified+none cost with PDIP semantics? | adopted 2026-09-25: lanes within Δ = 100 nats of the batch max pinned at 0.1 nats vs scalar PDIP, plus gated cross-composition and capture checks | not started | none | none |

C1 matched timing at the production batch B=20 (ms per lane, A100): certified+none `jit(vmap)` is
1.41x over library PDIP vmap on Delaunay pix1 (30.5 vs 43.0), 1.16x rect pix1, 1.73x Delaunay pix2,
1.47x rect pix2; guarded projections 31.4 / 27.3 / 15.3 / 12.4 ms against PDIP 43.0 / 30.6 / 22.8 /
17.0 (C1 ledger, Decisions).

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#567 | certified active-set solver (`jax_active_set.py`), opt-in via `positive_only_solver` / `certified_fallback` / `certified_pass_budget` 16 / `certified_tau_rel`; `positive_only_solver_used` | `11b93476` | 2026.9.26.1 |

The packaged default is still `pdip` everywhere. No config flip shipped (scalar flip retired; the
vmap change waits on C2).

## Open / parked / drafts

- `draft/feature/autofit/certified_solver_batched_guard_c2.md` — C2 guard (PyAutoFit `Fitness._vmap` re-run, PyAutoArray/PyAutoGalaxy flag surface); carries C1 open questions 4 (PDIP `converged`/`iterations` per lane) and 5 (NaN on catastrophic lanes), plus an interferometer amendment.
- `draft/bug/autoarray/batched_jit_vmap_b50_wrong_log_likelihood_a100.md` — B=50 `jit(vmap)` returns wrong log likelihoods (C1 Diagnosis 3).
- `draft/bug/autolens_workspace_test/multi_dataset_jax_likelihood_delaunay_wrong_likelihood.md` — pre-existing failure found in phase A.
- `draft/research/autolens_profiling/post_certified_solver_likelihood_breakdown.md` — the follow-on breakdown ([page](post_certified_breakdown.md)).
- Phase A's optional `custom_jvp` reuse of the final Cholesky factor: not filed as a draft.

## Caveats

- **The 1e-9 gate is tighter than PDIP delivers.** Both phases gate at 1e-9 relative, but library PDIP (tolerance `min(n·eps, 1e-2)` ≈ 3.3e-13, `max_iter` 50) disagrees with *itself* between compilations by 1e-9 to 1e-5 relative (up to 1.8 nats) on catastrophic lanes (C1 Diagnosis 1); the mechanism is a hypothesis, no per-lane PDIP convergence was recorded. Delaunay also has a ~2e-10 run-to-run floor and a ~2.5e-9 cross-composition residual (phase B). Other gates in this wiki are in nats (C2: 0.1 nat near-peak; [Matrix-free pixelized](matrix_free_pixelized.md): 0.5 nats), so numerical gates are not consistent across campaigns.
- **The own-composition gate is blind to a composition fault.** At B=50 the vmap arm and its vmap reference share the error (~5200–6000 nats median); only the ungated cross-composition column (9.12 / 8.59 relative) shows it. B=50 vmap timings and uncertified counts are unusable.
- **Phase B numbers are seeded draws; C1 numbers are real Nautilus batches.** Phase B's lanes are one seed-0 family near the fiducial with regularization fixed; C1 replays captured proposals with regularization free. Every C1 row is slower than its phase-B counterpart (Delaunay B16 PDIP vmap 47.0 vs 38.9 ms/lane); do not mix them.
- **Lane-rate spread.** The 1.9–4.9 % overall rate averages a prior phase at 11.7–20.6 % with a late half at 0, over captures capped at 40000 likelihoods / 3600 s; "near zero" is arguable (ledger). At prior-phase rates the guard is slower than PDIP on rectangular. Every rate task hit pass 16. C1 scalar certified arms vary between tasks more than in phase B (Delaunay pix1 38.9 vs 44.3 ms on overlapping lanes).
- **The scalar path reaches almost no production run.** The human: nobody runs AutoLens without JAX vmap mode, which is why the phase-B scalar gain (1.7x / 1.4x) was never shipped as a default.
- **Provenance gaps.** The C1 `.npz` captures and all `.out`/`.err` logs stay on RAL, not committed; the Delaunay captures (350659) were taken on PyAutoArray `7fa8d271`, the replay ran on `3de624b5`. Sidecars: [phase B](../../results/notes/certified_solver_policy_phase_b_job350588.json), [C1](../../results/notes/certified_solver_phase_c1_job350768.json).
- **B=100 is a memory limit**, not a result: one 73–74 GiB allocation on an 80 GB A100.

## Journal

### 2026-09-24 — scalar-default flip retired (never issued)

The prompt split out of phase B asked to make certified + PDIP fallback the default for scalar JAX
likelihoods (`jax.jit(fn)`, `use_jax_vmap=False`) while `jit(vmap)` kept PDIP, on phase B's 1.7x
(Delaunay) / 1.4x (rectangular) scalar gain (job 350588). At `/start_dev` planning the human retired
it as superseded by phase C, for three reasons in `complete/2026/09/certified-solver-scalar-default-flip.md`:
(1) no production run takes the scalar path ("I dont think anyone is running AutoLens without JAX
vmap mode on"); (2) batching already beats scalar, and the winning row is certified+none *under*
`jit(vmap)` (25.9 ms/lane Delaunay B16); (3) telling scalar from batched calls inside the library has
no gradient-safe public API (`custom_vmap` breaks `jax.grad` in JAX 0.10.2, `BatchTracer` detection
needs `jax._src`, PyAutoFit vmaps in four places). No PR, no job. Scalar users can opt in via config.

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records phases A, B, the retired scalar flip, C1 and the C2 draft, with the pre-registered
rules from #300, #304 and the phase-A prompt. Resolved: PyAutoArray#567 is released in 2026.9.26.1
(stub said "not verified"), so C2's release block is lifted; the C2 gate values (Δ = 100 nats, 0.1
nat pin) and the rect-pix1-stays-on-PDIP decision come from the C2 draft and the C1 Mind record,
since the C1 ledger's Decisions section still words them as open. Open: C2 itself, the B=50 bug.
