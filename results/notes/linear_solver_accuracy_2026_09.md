# Linear-solver accuracy study — ledger (phase 1, 2026-09)

Campaign page: `wiki/campaigns/linear_solver_accuracy.md`. Package: `scripts/lens/solver/`.

## Pre-registered decision rule

Pre-registered decision rule — linear-solver accuracy study, phase 1 (written 2026-09-30, after the
8-system `slam_fixture_571` smoke run and BEFORE the deciding run on `slam48_hst` + `slam_spread_hst` +
`euclid_vis_lp`; committed at autolens_profiling commit `327f571`).

A candidate is **admissible** iff, on every system of every corpus group (CPU fp64, released library
2026.8.17.1 source checkouts as recorded in the summary provenance):

1. **Converged flag.** `converged == 1` on 100 % of systems, all 48 `slam48_hst` points included (the
   PyAutoArray#571 constraint). A candidate whose iterate is accurate but whose flag says unconverged is
   inadmissible: the flag is what production trusts (the fixture smoke run showed jaxnnls's own tolerance
   on the raw system is accurate to 2e-11 yet hits the 50-iteration cap on 8/8 systems — recorded as a
   caveat, and as evidence for a solution-based criterion in phase 2, not as a pass).
2. **Solution accuracy vs fnnls.** worst `amp_rel_max_sig` (columns with x_ref > 1e-6 · max x_ref) ≤ 1e-3
   AND worst |`flux_rel_source`| ≤ 1e-4 on every system. `amp_rel_max` over all x_ref > 0 columns is
   recorded but not gated: on these near-singular systems (cond(Q) ~ 1e11) it reaches 1e7 for iterates whose
   source flux agrees to 1e-5, because it is dominated by reference columns at the 1e-9 level (fixture smoke
   run, 2026-09-30).
3. **Euclid witness proxy.** On `euclid_vis_lp`, |`flux_rel_source`| ≤ 1e-4 — the eager NumPy path the
   euclid test compares against solves with fnnls, so this is the in-harness form of the rel-1e-3 latent
   check with 10x headroom.
4. **Optimality.** scaled KKT residual ≤ 10x `pdip_jacobi`'s on the systems where jacobi converges.

Among admissible candidates choose the lowest **median iteration count**; CPU wall is the tie-break; ties
go to the smallest library change (a constant change beats a new code path). If no candidate is
admissible the verdict is "no drop-in candidate; phase 2 needs a solution-based stop", and the study
names which metric the released stop is blind to.

A human may re-base this rule; the campaign page records the re-basing and its reason.
