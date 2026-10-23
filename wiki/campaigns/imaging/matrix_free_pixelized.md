# Matrix-free pixelized likelihood

**Status:** no-go
**Question:** Does a matrix-free pixelized likelihood (preconditioned CG on `(F + λH) x = D` through the sparse operator, SLQ log-dets) overtake the exact Cholesky path at any source-pixel count up to ~12000, with log-evidence noise a sampler can tolerate?
**Pre-registered rule:** autolens_profiling#247 (written before the runs): go bar |Δ log-evidence| ≤ 0.5 nats at the chosen (probes, Lanczos steps); a PyAutoArray phase proceeds only if the sweep shows a size where matrix-free is the only path that fits or is faster than sparse, with noise under that bar and positivity handled.
**Verdict:** no-go on both axes: no crossover against dense or sparse at any measured N, and SLQ never reaches 0.5 nats at the fiducial N on any mesh; the PyAutoArray phase was not filed.
**Headline:** matrix-free 1435.3 ms vs exact 4.02 ms per call (357x), rectangular n ≈ 1500, A100 80GB fp64 (`euclid-ral-gpu-2`) job 342664; still 15–17x at n = 12000 (jobs 342673, 342674).
**Library PRs:** none
**Profiling PRs:** #249 (issue #247)
**Ledger:** [matrix_free_pixelized_2026_09.md](../../results/notes/matrix_free_pixelized_2026_09.md)
**Mind contract:** no epic; `complete/2026/09/matrix-free-pixelized-likelihood.md` (parent `complete/2026/09/a100-pixelized-baseline.md`).
**Next:** none

## Why this campaign

The matrix-free line (CG for the solve, stochastic Lanczos quadrature for the log-dets, never forming
`F + λH`) was deferred in June 2026 as plan-B because the sparse w-tilde path kept an exact log-det. The
2026-09-10 [A100 pixelized baseline](a100_pixelized_baseline.md) re-opened it, and the 2026-09-11
reconstruction split reframed it: the ~37 ms solve row is 21–22 PDIP iterations, while the exact
unconstrained solve is only 1.52–1.58 ms and the two log-det Choleskys ~2.4 ms. The prompt therefore
said up front it was not a speed play at n ≈ 1500; the deliverable was the reference implementation
plus the N_src sweep that finds where (if anywhere) matrix-free wins or dense stops fitting.

The go/no-go rule was fixed in autolens_profiling#247 before any A100 leg ran: the 0.5-nat
log-evidence bar (scoping decision 3) and the decision gate (prompt item 4). Positivity was scoped
as unconstrained PCG on every leg plus a matrix-free PDIP comparator at the fiducial tier.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 kernels + tests | 2026-09-11/12 | Do the matvec, PCG, SLQ and matrix-free PDIP kernels reproduce dense references? | plan tolerances in #247 (PCG vs solve rtol 1e-8, matrix-free PDIP vs `solve_nnls` < 1e-6, SLQ within 3σ) | kernels + CPU tests committed (`522328c`); pass counts not recorded | none (CPU unit tests) | #249 |
| 2 fiducial n ≈ 1500 | 2026-09-12 | Cost and accuracy of each matrix-free piece against the exact comparators, three meshes | 0.5-nat go bar; speed no-go expected | preconditioned PCG 1153–2386 ms (5.3k–11.5k iterations) vs 1.55–1.59 ms Cholesky solve; matrix-free PDIP 223x / 413x / 454x exact `nnls_pdip` (Delaunay not converged at `max_iter` 50); best SLQ −4.98 / −1.09 / +0.55 nats, none within 0.5 | 342664, 342665, 342666 | #249 |
| 3 N_src sweep | 2026-09-12 | Crossover N vs dense and sparse; where dense stops fitting | #247 decision gate | no crossover at 3000 / 5000 / 12000 (rect, Delaunay); every dense and sparse single-call leg fits to n = 12000; dense vmap-16 NNLS row OOM at n ≥ 5000 | 342669–342694 | #249 |

Matrix-free per call (best PCG + SLQ m = 320) against the like-for-like exact path (Cholesky solve
plus two log-det Choleskys), from the ledger's crossover table:

| Mesh | n ≈ 1500 | 3000 | 5000 | 12000 |
|---|---:|---:|---:|---:|
| rectangular | 1435.3 / 4.02 ms (357x) | 1587.2 / 7.51 (211x) | 1732.9 / 17.79 (97x) | 2042.7 / 132.38 (15.4x) |
| delaunay | 1936.7 / 3.88 (499x) | 2164.8 / 7.22 (300x) | 2054.7 / 16.93 (121x) | 2178.2 / 130.63 (16.7x) |
| delaunay_nn | 1698.2 / 3.87 (439x) | not measured | not measured | not measured |

The ratio narrows with N because the exact path grows while matrix-free stays ~1.4–2.2 s. The cause is
`cond(F+λH)` ≈ 4.1e10 on every leg and every N, set by the 60 unregularised linear-MGE columns, so
PCG needs thousands of matvecs even though each is 0.036–0.052 ms per column at ×32.

## What shipped and where it is

Nothing shipped to a library: the PyAutoArray phase was conditional on a go. The reference
implementation (`scripts/misc/likelihood_breakdown/matrix_free_steps.py`, the `matrix_free.py` cell,
`--source-pixels N`, the sweep aggregator) is in this repo via #249.

## Open / parked / drafts

- None for this campaign. The lever it pointed at (positivity and MGE conditioning) was taken by
  [Fixed lens light](fixed_lens_light.md) (autolens_profiling#248) and then [Certified positive solver](certified_positive_solver.md).
- `draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md` — w-tilde weight-map bug found alongside, filed separately.
- A DCT preconditioner for rectangular λH is named as a possible research prompt; not filed, and it cannot reach the MGE conditioning floor.

## Caveats

- **No crossover at any measured N, and nothing extrapolated.** The largest N is 12000 (rectangular, Delaunay). There is no matrix-free leg at n = 8000, no DelaunayNN matrix-free sweep leg at all (fiducial only), and matrix-free PDIP was disabled on every sweep leg. #247 planned 45 legs; 29 ran (3 fiducial + 26 reduced sweep).
- **The crossover comparator leaves positivity out.** It is unconstrained PCG + SLQ vs Cholesky solve + two log-dets; the NNLS/PDIP row (60–85 % of per-call cost at every N) is not in either side. The unconstrained-PCG log-evidence difference (14 / −68 / −71 nats) is a different minimiser, not solver error; do not quote it as accuracy.
- **The accuracy no-go is independent of speed.** The closest SLQ settings cost 160–720 ms (2.4x to ~12x the whole exact likelihood); the error is dominated by `log det(F+λH)`. The ledger's lead says the 720 ms setting is "~14x" the whole exact likelihood and its table text says "~12x"; 12x matches 720.459 / 59.889 ms (rect step-by-step total), the lead's denominator is not stated.
- **Dense did not stop fitting.** The prompt expected dense F to exceed an A100 by n = 5000; every dense and sparse single-call leg fit to 12000. Only the optional vmap-16 batched-NNLS comparator OOM'd (up to 93.73 GiB, dense 342688–342694).
- **Fiducial rows compare across jobs.** Blurred-mapping, F and total columns come from the independent `_recon_split` legs 342643–342645 ([A100 pixelized baseline](a100_pixelized_baseline.md)); the matrix-free cell and those rows overlap and must never be summed. There is no sparse exact comparator at n ≈ 1500.
- **Aggregator vs note.** `results/sweep/size_sweep_tables.md` uses the Jacobi PCG row (rect fiducial 1462.3 ms), the note the fastest PCG row (1435.3 ms); no verdict changes. Regularization differs by mesh (`Constant(1.0)` rect, `adapt_split` Delaunay), so compare log-det terms within a mesh only.
- All 29 legs ran on `euclid-ral-gpu-2` with a fresh cache; pins PASSED on the fiducial legs and were skipped by design on non-fiducial N.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the three phases, the crossover table per mesh and N, and the conditioning cause.
Resolved: the pre-registered rule was written in #247 (0.5-nat go bar and the decision gate), so the
stub's "not recorded" is replaced; the "~350x" headline is the rectangular figure (357x), Delaunay is
499x and DelaunayNN 439x at n ≈ 1500. Nothing open: the lever moved to the fixed-lens-light and
certified-solver campaigns.
