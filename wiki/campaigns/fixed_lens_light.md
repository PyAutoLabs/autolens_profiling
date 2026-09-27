# Fixed lens light (JAX, source-only)

**Status:** complete
**Question:** With lens light fixed, what does the whole source-only likelihood cost on each hardware type (A100, CPU, consumer GPU) on HST and Euclid data?
**Pre-registered rule:** no programme go/no-go threshold recorded; per-phase correctness gates written in the issues before the legs (certified Δlog-evidence vs PDIP < 1e-6 nats, #248; route d == route b at rtol 1e-8, #251; pins re-derived per precision, #253); phases 3–5 pre-specified the metric (fallback rate at fixed budgets, ms-vs-N, the HST + Euclid grid) but no threshold.
**Verdict:** programme verdict (phase 5): fix the lens light, solve only the source, certified active set at budget 7 (Delaunay) with a PDIP fallback, fp64, never drop positivity. Corrected 2026-09-18 by the HST GPU residue campaign: 25.39 ms was pass budget 2 (budget 7 = 31.64 ms), and the ~13.9 ms mesh/mapper bucket measured 0.37 ms.
**Headline:** A100 Delaunay whole library call 65.10 → 25.39 ms (2.56×) at pass budget 2, RAL euclid-ral-gpu-2 job 342909; the budget-7 call is 31.64 ms (job 343350_0, [HST GPU residue](hst_gpu_residue.md)).
**Library PRs:** none (the certified solver was a harness monkeypatch in every phase; no PyAutoArray change).
**Profiling PRs:** #250, #252, #254, #256, #258, #260 (issues #248, #251, #253, #255, #257, #259)
**Ledger:** [fixed_lens_light_verdict_2026_09.md](../../results/notes/fixed_lens_light_verdict_2026_09.md), [fixed_lens_light_hardware_2026_09.md](../../results/notes/fixed_lens_light_hardware_2026_09.md), [fixed_lens_light_library_path_2026_09.md](../../results/notes/fixed_lens_light_library_path_2026_09.md), [fixed_lens_light_low_likelihood_draws_2026_09.md](../../results/notes/fixed_lens_light_low_likelihood_draws_2026_09.md), [fixed_lens_light_source_only_2026_09.md](../../results/notes/fixed_lens_light_source_only_2026_09.md), [fixed_lens_light_source_pixel_scaling_2026_09.md](../../results/notes/fixed_lens_light_source_pixel_scaling_2026_09.md); correction in [profiling_campaign_status_2026_09.md](../../results/notes/profiling_campaign_status_2026_09.md)
**Mind contract:** epic `fixed-lens-light-profiling`; `complete/archive/epics/fixed_lens_light_profiling_epic.md`; phase records `complete/2026/09/fixed-lens-light-source-only.md`, `fixed-light-library-path.md`, `fixed-light-hardware.md`, `fixed-light-draws.md`, `fixed-light-scaling.md`, `fixed-light-verdict.md`
**Next:** none (successors: [HST GPU residue](hst_gpu_residue.md), [Fixed light numba CPU](fixed_light_numba_cpu.md), [Certified positive solver](certified_positive_solver.md))

## Why this campaign

The matrix-free sweep ([Matrix-free pixelized](matrix_free_pixelized.md), #247) ended on
`cond(F+λH) ≈ 4e10`, set by the 60 unregularised linear-MGE columns. A 2026-09-11 CPU probe
showed that freezing the lens light after SLaM `light[1]` (MGE converted to regular profiles at
the solved intensities and subtracted, system "S3") removes that block, and the human asked
"what happens if we fix the mge light profiles". Phase 0 (#248) measured it on the A100 and
found a *certified active-set* positive solve viable on the source-only system.

On 2026-09-13 the human queued five further tasks in one brief (verbatim in the epic ledger):
profile the positive-negative solver for completeness, repeat on CPU and the laptop GPU, check
the certified solve is not fast only because the model is good, scale in source pixels, and
assess the whole likelihood on HST and Euclid at 500 / 1250 / 2500 pixels. A same-day
clarification dropped JWST and the sparse operator and required the MGE to be subtracted, not
solved, on the positive-negative route. The phases were worked strictly one at a time, each
grid chosen from the previous answer, as a stack of six PRs merged in base order on 2026-09-14.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 0 kernel measurement | 2026-09-12 → 09-13 (merged 09-14) | A100 cost of the S3 source-only inversion, with and without positivity | pins PASS; S3 mapper-block log-dets == S0; certified Δlog-evidence vs PDIP < 1e-6 nats (#248) | S3 PDIP 25.8–28.3 ms; certified 4.21 ms Delaunay (pass 2) / 11.05 ms rect (pass 7), same solution to ≤ 1.2e-10 nats; library call 26–30 % faster from fixing the light alone; A2 +6.4 / +334.9 nats | 342802–342806 | #250 |
| 1 library path | 2026-09-13 (merged 09-14) | Whole `log_likelihood_function` with the certified and positive-negative solvers inside it | equivalence pins d == b and e == b at rtol 1e-8 (#251) | a → d 50.97 → 25.10 (rect, 2.03×), 65.10 → 25.39 (Delaunay, 2.56×), 72.95 → 36.26 ms (DelaunayNN, 2.01×); positive-negative still +334.93 nats; `lax.cond` under `vmap` runs both branches | 342908–342910 | #252 |
| 2 CPU + RTX 2060 | 2026-09-13 (merged 09-14) | Same rows on CPU (1 / 8 threads) and a 6 GB consumer GPU, fp64 and mixed precision | fp64 legs assert pins; mixed-precision legs record only, `tau_rel` re-derived (#253) | a → d 1.28–1.48× RTX 2060, 1.35–1.83× 8 threads, 1.16–1.22× 1 thread; `@vmap 16` needs 11.88 GiB and OOMs; mp buys 5–8 % for ≤ 2.5e-3 nats; numpy certified does not beat `fnnls` | laptop | #254 |
| 3 low-likelihood draws | 2026-09-13 (merged 09-14) | Is the certified solve fast only because the model is good? | not recorded (metric pre-specified: fallback rate at budgets 2 / 4 / 6 / 7, #255) | phase-0 budgets fail: pass 2 falls back on 67.5 % (Delaunay), pass 7 on 27.5 % (rect); zero-fallback budgets 7 / 11; median 5.4× / 2.6× over PDIP; pass counts identical A100 vs CPU | 343011, 343012; laptop | #256 |
| 4 source-pixel scaling | 2026-09-13 (merged 09-14) | How do the solvers scale over N = 500–4000 per hardware? | not recorded (a leg that OOMs or times out is a result, #257) | certifying budget has no trend in N (rect 5–10, Delaunay 1–2); certified leads at every N (1.44–1.62× rect, 2.21–3.00× Delaunay, A100); `@vmap 16` amortisation 3.64× → 0.68× at N≈4000; affordable N 4000 / 1500 / 1000 | 343023, 343024 (343013 / 343018 failed at checkout); laptop | #258 |
| 5 HST + Euclid verdict | 2026-09-13 (merged 09-14) | Whole likelihood on HST and Euclid at 500 / 1250 / 2500; production config per hardware | not recorded (budgets fixed at 11 / 7 before the legs, #259) | certified 1.20–1.59× over today on HST, 1.43–2.28× on Euclid; dropping positivity +109.1 nats on Euclid at N=2500 vs +6.4 on HST; rect certifies at exactly 11 on Euclid N=2500; 12 A100 legs not run | laptop (A100 not submitted) | #260 |

## What shipped and where it is

No library PR shipped from this campaign: every certified number comes from a harness monkeypatch
of `inversion_util.reconstruction_positive_only_from`. The library implementation was formalised
later as the [Certified positive solver](certified_positive_solver.md) campaign.

## Open / parked / drafts

- `draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md` — the sparse-path weight-map bug that kept the sparse operator out of scope.
- Unformalised `ideas.md` bullets `[from: fixed-lens-light-profiling phase 5]`: the matched-injection witness before any unconstrained (A2) use; re-derive the rectangular pass budget on Euclid over a graded draw set; run the 12 A100 phase-5 legs (`hpc/batch_gpu/submit_fixed_light_verdict.sh --node euclid-ral-gpu-2`).
- Formalised follow-ups: the library certified solver (`draft/feature/autoarray/implement_and_optimize_certified_positive_solver.md`, now `complete/2026/09/certified-positive-solver.md`) and `draft/research/autolens_profiling/post_certified_solver_likelihood_breakdown.md` ([page](post_certified_breakdown.md)).
- Two fresh programmes filed 2026-09-14: the numba CPU programme ([page](fixed_light_numba_cpu.md)) and the non-solver residue campaign ([page](hst_gpu_residue.md)).

## Caveats

- **25.39 ms is pass budget 2** (phase 1, job 342909). Phase 3 showed budget 2 is unsafe; the budget-7 A100 Delaunay single call is **31.64 ms** (job 343350_0, command buffers on), per the 2026-09-18 correction in both the library-path and verdict ledgers.
- **The ~13.9 ms "mesh / mapper / weights" residue is attribution arithmetic across two cells** (phase-0 kernel rows subtracted from phase-1 library calls). A single-process trace measured that bucket at **0.37 ms**; it was PSF convolution, the `A.T A` GEMM and device idle at the qhull callback.
- **The phase-5 A100 column was never measured.** The RAL jump host refused publickey; every A100 number in the verdict is carried from phases 1 and 4, and the ≈ 57 ms N=2500 figure is arithmetic, not a measurement. No Euclid A100 row exists in any phase.
- **Kernel rows and library rows are different cells.** Phases 0, 3 and 4 time kernels in their own process (phases 0 and 4 also carry one S3 library-call row); phases 1, 2 and 5 time the whole library call. Never sum one into the other.
- **Laptop rows (phases 2–5) ran on a WSL2 i9-10885H + RTX 2060 Max-Q**, quiet, one leg at a time; they are not RAL numbers. The CPU rect n=484 PDIP row (162.3 ms) is dispatch-bound and its 5.44× is an artefact.
- **Mixed-precision "certification" is looser, not faster**: `tau_rel` re-derived to 1.4775e-5, so rect certifies at 4 instead of 7. A production budget must never be read off an mp leg.
- **The solver-only `@vmap` scaling curve (phase 4) does not set whole-likelihood batching policy**; see the correction and the HST GPU residue phase-2 experiment.
- **The Euclid budgets come from one model each**; phase 3's draw set ran on HST only. Euclid legs use this cell's presets (60 × 1 MGE, `[4, 2, 2]`), not the Euclid pipeline's, so ratios transfer and absolute ms do not.
- **Rectangular edge zeroing** makes phase 3's A2 figure (+119.7 nats, vs full-system PDIP) and phase 0's (+334.93, vs the edge-zeroed library) differ by exactly the 215.18-nat edge-zeroing penalty; neither is solver error.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records all six phases (0–5) with dates, jobs, PRs and the per-phase gates read from
issues #248–#259, plus the 2026-09-18 correction (budget 2 vs 7, the refuted 13.9 ms bucket).
Resolved: `Library PRs` is none, not "not recorded" (no PyAutoArray change in any phase, per the
Mind records); the budget-7 figure is job 343350_0. Open: the 12 A100 phase-5 legs, the Euclid
rectangular budget and the matched-injection witness remain `ideas.md` bullets.
