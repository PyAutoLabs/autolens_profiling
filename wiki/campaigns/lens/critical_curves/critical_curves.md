# Critical curves — geometry, dispatch and cost

**Status:** open
**Question:** Which critical-curve engine/settings preserve cluster field coverage and multi-plane geometry without avoidable compilation cost?
**Pre-registered rule:** Match every reference component within two fine-grid pixels (0.1 arcsec control, 0.25 arcsec cluster); analytic control and coarse/fine convergence; errors/timeouts never pass.
**Verdict:** Keep marching squares as the modeling-run default at every scale (#521). Zero-contour is correct where its seeds reach the curve, but it recompiles on every figure. Cached, it would only pay off for interferometer and group, and only after the seed, duplicate-path and fallback fixes. Cluster stays NO-GO.
**Headline:** 17-leg phase-4 posterior corpus (RAL CPU fp64, 211 draws). Production zero-contour costs 6–22 s per figure against marching squares' 0.1–3.7 s. The automatic ±3 arcsec seeds miss group and cluster curves, and the coarse fallback detector catches all 47 zero-contour failures.
**Library PRs:** [Galaxy #646](https://github.com/PyAutoLabs/PyAutoGalaxy/pull/646), phase 3b cap fix; awaiting merge/release.
**Profiling PRs:** [#364](https://github.com/PyAutoLabs/autolens_profiling/pull/364) merged (phase 3a); phase-3b ledger update pending; [#521](https://github.com/PyAutoLabs/autolens_profiling/issues/521) modeling-run viz feasibility (PR pending).
**Ledger:** [critical_curves_dispatch.md](critical_curves_dispatch.md)
**Mind contract:** epic `cluster-strong-lensing`; phase 3a complete; `active/evaluation_grid_cap_preserves_field.md` (3b)
**Next:** Ship the bounded phase-3b cap fix. Then, if the human wants zero-contour for interferometer/group figures, do the gated pytree compile-cache refactor together with field-wide seeds, duplicate-path removal and the coarse fallback. The alternative lever is a smaller marching-squares grid around the lens.

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
| Source & Cluster 3a | 2026-10-02 | Is existing dispatch safe at cluster scale? | Full component match and two-pixel distance bound | Seed and path-completion failures; grid-cap field distortion; outer-JIT unsupported | local CPU, no HPC | [profiling #364](https://github.com/PyAutoLabs/autolens_profiling/pull/364), [CI #341](https://github.com/PyAutoLabs/autolens_workspace_test/pull/341), merged |
| Source & Cluster 3b | 2026-10-02 | Does the cap preserve the effective physical field? | Bound both axes, no cropping, less than one pixel total padding | 60 arcsec field retained at 0.06 arcsec/pixel; 1315 library tests pass | local grid recorder, no Hessian/HPC | [Galaxy #646](https://github.com/PyAutoLabs/PyAutoGalaxy/pull/646), open |
| Modeling-run viz feasibility | 2026-11-06 to 2026-11-09 | Is zero-contour correct and cheaper for modeling-run figures, per scale? | Same two-pixel rule against a finer marching-squares reference on phase-4 posterior draws; zero-contour paths closed; errors never pass | Correct where seeded; recompiles every figure; seeds miss group/cluster curves; duplicate paths on interferometer; cached cost only wins for interferometer/group | RAL 414456 (17 legs, `ral`, CPU fp64) | [#521](https://github.com/PyAutoLabs/autolens_profiling/issues/521), PR pending |

## What shipped and where it is

Phase 3a research and CI are merged. No production fix has yet shipped from this campaign. Earlier zero-contour work is
an input, not evidence that cluster plots are safe. Keep PR/merge/release records
here as the subsequent bounded phases ship.

## Open / parked / drafts

- Zero-contour for modeling-run figures (#521) needs four things before any default change: a compile cache that survives new posterior samples (`draft/refactor/autogalaxy/zero_contour_pytree_compile_cache.md`), seeds covering the whole plotting field instead of ±3 arcsec, removal of curves traced twice, and the coarse marching-squares fallback.
- Production cluster figures (`point_sources.curves_for_planes`, 400² over ±50 arcsec) miss the small member-galaxy tangential curves on the nearer source plane in every draw (11 of 22 plane-draws fail against a 1600² reference).
- Interferometer and group figures spend 2–4 s on marching squares because the plotting field is the full real-space or image grid (512² at 0.05 arcsec for SMA). Restricting the grid to the region around the lens may be the cheaper fix.
- Docs drift: PyAutoGalaxy `plot_utils` docstrings call `zero_contour` the default; the config default is `marching_squares`.
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

### 2026-11-09 — Modeling-run visualization feasibility (#521)

The human asked (2026-11-06) whether zero-contour could draw critical curves in modeling-run figures. We tested it on posterior draws from the phase-4 autolens_inference runs, because a running fit's figures show models away from the truth. The harness is `scripts/lens/critical_curves/modeling_viz_feasibility.py` (galaxy) and `modeling_viz_feasibility_scales.py` (group, cluster, corpus summary). It times the exact production helpers and grades both engines against a finer marching-squares reference.

Corpus: 17 legs, one per scale, likelihood family and variant (imaging/hst 6, interferometer/sma 3, group/euclid 6, cluster/simple 2). Each leg is the maximum-likelihood sample plus 10 weighted draws. It ran as RAL array 414456 (`ral`, exclusive, 8 CPU, fp64) on a private library checkout at the 2026.11.6.1 mains. The summary is `results/lens/critical_curves/viz_feasibility/summary_ral_cpu_fp64_2026-11-09.json`.

| Family | Zero-contour tangential (graded draws) | Marching squares s/figure | Zero-contour s/figure (production) | Zero-contour warm, cached |
|---|---|---|---|---|
| imaging/hst (Delaunay, Delaunay-NN, MGE) | 66/66 pass | 0.26–0.30 | 6.2–14.4 | 0.30–0.58 |
| interferometer/sma | 6 pass; 10 duplicate paths; 2 mismatch (marching squares fails the same 2) | 2.1–2.9 | 7.4–14.4 | 0.18–0.37 |
| group/euclid MGE | 22/22 pass | 3.0 | 21.2 | 0.84 |
| group/euclid Delaunay, rectangular | 9/44 pass; 35 no curve found | 3.3–3.7 | 9.1–9.2 | not a tracing cost (nothing traced) |
| cluster/simple (2 source planes) | 0/44: no tangential curve found on either plane | 0.11–0.13 | 16–22 | 0.27–0.38 |

What we learned:

- **Recompilation is the cost.** Every production call builds a fresh `LensCalc` whose closure captures the tracer, so JAX compiles again for each figure. A repeat on the same tracer is just as slow. Cached, the warm call is 0.2–0.8 s.
- **Caching would only help where marching squares is slow.** For HST imaging, marching squares is already 0.3 s, so cached zero-contour gains nothing. Interferometer (about 8–12×) and group (about 4×) would gain. Cluster would not.
- **Seeds are the correctness problem.** The automatic seed grid spans ±3 arcsec. Group critical curves (radius about 4.8 arcsec) and cluster curves lie outside it, so zero-contour often returns no curve at all. Seeding from the reference recovers most cluster curves, but the large far-plane curve is longer than 500 steps of 0.05 arcsec can trace (phase 3a's closing chord again).
- **Interferometer duplicates.** The same curve is traced 2–4 times from different seeds. It looks fine on a plot but breaks curve counts and the one-to-one match.
- **A cheap fallback works.** Comparing the zero-contour curve count with a coarse 0.2 arcsec marching-squares count, plus a closure check, flagged all 47 zero-contour failures, with 2 false alarms in 150 draws.

Caveats: the laptop pilots (2026-11-06) ran marching squares about 4× slower than RAL, so use the RAL numbers. Radial curves are small and resolution-limited; both engines disagree with the reference there, so they are reported but not decisive. Interferometer draws whose reference curve touches the ±4 arcsec reference field are excluded (15 of 33). The RAL venv's numpy (2.2.6) and scikit-image (0.24) are older than the laptop's; jax and jax_zero_contour match. No default or library code changed.

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


### 2026-10-02 — Phase 3a merged; first cap fix validated

The two phase-3a PRs merged after all CI legs passed; issue #337 is closed.
Human approved one successor, Galaxy #645, with separate-scope concurrency.
The same cap probe on its fixed branch retains the 60 arcsec field: 1000×1000
at 0.06 arcsec, pixel centres ±29.97. Library validation passes 1315 tests;
the ledger records before/after geometry and the measured source hash.
This fixes the dimensional cap formula and both-axis limit, preserving existing
Zoom2D centre/mask support and conservative subpixel rounding. It does not
establish seed completeness or repair truncated zero-contour paths. Original
phase-3a evidence remains immutable. Library and linked workspace shipping pending.
