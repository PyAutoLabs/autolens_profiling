# Fixed lens light — is the certified active set fast only because the model is good? (2026-09-13)

autolens_profiling issue [#255](https://github.com/PyAutoLabs/autolens_profiling/issues/255),
branch `feature/fixed-light-draws` (stacked on phase 2's `feature/fixed-light-hardware`),
harvest commits `3fcf17a` (A100) and `e06c005` (CPU). Phase 3 of the
`fixed-lens-light-profiling` epic. Four legs — rectangular 1521 and Delaunay 1500, each on
the A100 (`euclid-ral-gpu-2`, jobs 343011 / 343012, fp64) and on JAX-CPU at 8 threads —
each running the **same seeded 41-model draw set**: four one-parameter walks bisected onto
Δlog L ≈ −10 / −100 / −1000 / −1e4, 24 random draws from SLaM-like priors at 5σ, and the
fiducial.

**The answer is yes on Delaunay and no on rectangular, and neither mesh's phase-0 budget is
safe.** On **Delaunay** the pass count is exactly what phase 0 feared it was: it *grows*
with model error (Spearman ρ = **+0.698**), from 2 at the fiducial to a median of 3 and a
maximum of 7, and the pass-2 budget phase 0 certified at **falls back on 67.5 % of the draw
set**. On **rectangular** the pass count *falls* with model error (ρ = **−0.535**) — a bad
model has a bigger active set but an easier one, because pass 0 already finds most of it —
yet the spread alone breaks the budget: passes run 4 to 11 around a median of 7, and the
pass-7 budget falls back on **27.5 %**. The smallest budget with **zero** fallback over
these 40 draws is **7 on Delaunay and 11 on rectangular**, in both cases well above the
fiducial's own. The scheme stays worth having: it is 2.8–9.9× faster than PDIP per call on
the A100 across the whole draw set (median 5.4× Delaunay, 2.6× rectangular), it is still
exact on Delaunay to 3.7e-10 nats on **every** draw, and its worst draw (16.6 ms
rectangular, 10.7 ms Delaunay) is still faster than the best PDIP call (25.7 / 26.5 ms).
What has to change is the budget, and the fact that the budget must be chosen for the
*population* a search visits, not for the answer it ends at.

## Scope — read this before quoting a number

- **What was measured.** Per draw, per mesh, fp64, dense: the certified active set's pass
  count to certification (`active_set_certified`, numpy, `max_passes = 16`) and its
  per-call milliseconds at that budget (`active_set_masked_jax`, the jit-able twin phases
  0–2 timed); the library PDIP's iteration count and ms
  (`reconstruction_steps.nnls_pdip`, the *driver*, so the count survives); the seed set
  `|{x_unc < 0}|` and the final fixed set `|Z|`; the unconstrained solve's Δlog-evidence
  and negative-pixel count; and the draw's Δlog L. Derived: the fallback rate at fixed pass
  budgets, and Spearman ρ of each quantity against `|Δlog L|`.
- **These are KERNELS.** Every millisecond here is measured in the cell's own process on
  the S3 system rebuilt at that draw's mass — the same terms as phase 0. **Never sum one
  into a `figure_of_merit`.** Phase 1 measured the library-path row; this phase did not
  re-measure it per draw, and the per-call library cost at a bad model is not in this note.
- **What Δlog L means.** `S0 figure_of_merit(draw) − S0 figure_of_merit(fiducial)` — the
  whole library likelihood of the system the cells fit today, at the draw's mass. S0, not
  S3, because S0 is what a search scores; phases 0–1 pinned the two equal to 1e-14
  relative. The draw's S3 system is then rebuilt exactly as the phase-0 probe rebuilds its
  ±1 % draws: S0 fit at the draw's mass → solved MGE intensities → regular profiles →
  subtracted from the dataset → source-only system.
- **The fiducial is not "the truth", it is the prior median.** It is the model SLaM
  `light[1]` leaves behind, which is what makes it the right reference for this question —
  but the walk offsets below are offsets from *it*, not from the simulator's input.
- **What was NOT measured.** No library-path row per draw. No `@vmap` row: the pass budget
  differs per draw, so a batched budget is a different measurement (phase 1's route `d0`).
  No RTX 2060 leg — phase 2 established the consumer ordering and the pass count is
  device-independent (below), so a third device would add milliseconds, not an answer. No
  sparse operator, no JWST. **No PyAutoArray change** — the certified scheme is still a
  harness kernel.
- **Regularization** is `Constant(1.0)` on rectangular and `adapt_split` (0.1 / 10.0 / 0.1)
  on Delaunay, each cell's shipped default. Compare evidence within a mesh, never across.

## The draw set

Seeded and **identical on every mesh and device**. Built by
`scripts/misc/likelihood_breakdown/fixed_light_draws_steps.py`.

**Walks.** `einstein_radius`, `ell_comps_0`, `centre_x` and the mass **slope**, each scaled
by bisection onto the four targets. The bisection seeds itself from a one-evaluation
quadratic model (`Δlog L ∝ −δ²`) and then brackets and bisects through a per-parameter
cache, capped at 12 new S0 evaluations per target. That is not a micro-optimisation: every
evaluation is a full eager S0 fit (~4.5 s on the laptop, and CPU-bound on the A100 node
too), and the seed is why the whole construction cost 42 S0 evaluations on rectangular and
54 on Delaunay rather than the ~190 a naive ladder would have spent. The achieved Δlog L is
recorded and quoted; the target is only where the bisection aimed.

**The slope walk promotes the mass.** The fiducial mass is `al.mp.Isothermal`, which has no
slope, so the slope rows use `al.mp.PowerLaw` at `slope = 2.0` — the Isothermal special
case. That promotion is **pinned, not assumed**: the promoted fiducial's S0 figure of merit
is compared against the Isothermal's before a single slope row is measured (PASS at 1.8e-6
rectangular, 3.3e-5 Delaunay — the closed-form vs non-closed-form deflection code path,
≈ 0.2–1 nat of 2.9e4), and the slope rows are withdrawn if it drifts past 1e-4. The residual
is not carried into the rows either: each mass family's Δlog L is referenced to **its own**
fiducial, so the code-path offset cancels exactly.

**Random draws.** 24 seeded Gaussian draws (`default_rng(0)`) at 5× the cell's own prior σ
on `einstein_radius` (0.25), both `ell_comps` (0.05) and both `centre` components (0.025).
Isothermal, deliberately: the random family is the early-search sample, and promoting it
would make it a different model family from the walks. It lands **much** further out than
the walks — Δlog L from −3.2e3 to −8.7e4 (rectangular, median −2.3e4) and −7.5e3 to −9.8e4
(Delaunay, median −4.1e4) — which is the point: a real search's first evaluations are not
one-parameter walks from the answer.

**Two targets were not reached and say so.** `walk_slope` at −1e4 on rectangular landed at
−7559 (the walk hit its `MAX_DELTA` slope ceiling of +0.9), and `walk_slope` at −10 on
Delaunay landed at −18.9 after spending all 12 evaluations. Both are reported at the Δlog L
they reached, never relabelled as the target. The Delaunay miss is informative: the Delaunay
`figure_of_merit` is **not smoothly monotone** in a small `centre_x` or `slope` offset,
because moving the mass moves the Hilbert mesh's vertices discretely — the bisection log
shows δ = 0.00428 → −41.9 and δ = 0.00642 → −43.3. The bisection tolerates it (it is a
bracket-and-bisect, not a Newton step) but a walk into that region costs evaluations.

## Provenance and gate

| Leg | Mesh | N_src | Device | Exit | Elapsed | `cache_fresh` | autotune entries | Pins |
|---|---|---:|---|---|---:|---|---:|---|
| 343011 | rectangular | 1521 | euclid-ral-gpu-2, A100 80GB | 0:0 | 00:07:49 | true | 0 | **all PASS** |
| 343012 | delaunay | 1500 | euclid-ral-gpu-2, A100 80GB | 0:0 | 00:08:51 | true | 0 | **all PASS** |
| local | rectangular | 1521 | JAX-CPU, i9-10885H, NPROC 8 / BLAS 8 | 0 | 00:14:54 | true | 0 | RECORDED |
| local | delaunay | 1500 | JAX-CPU, i9-10885H, NPROC 8 / BLAS 8 | 0 | 00:15:46 | true | 0 | RECORDED |

**Counts: legs = 4; result JSONs written = 4; off-node = none; `cache_fresh: false` = none;
non-zero exit = none; pin FAILED = none.** Both A100 legs report `autolens 2026.8.17.1`,
`autotune_cache_entries_at_start: 0` and the same `AP_ROOT` (`a5a9b67`).

The fiducial pins are the ones that make this phase's numbers comparable to phase 0's, and
they are **asserted** on the A100 (where phase 0 measured them) and **recorded** on the CPU:

| Pin | rectangular | delaunay |
|---|---|---|
| fiducial certifies at phase 0's budget | **PASS** 7 vs 7 | **PASS** 2 vs 2 |
| fiducial S3 PDIP iterations | **PASS** 15 vs 15 | **PASS** 17 vs 17 |
| S0 `log det(F+λH)` / `log det(H)` | **PASS** 3888.258090 / 1692.786817 | **PASS** 8360.401763 / 7756.614959 |
| Isothermal == PowerLaw(2.0) at the fiducial | **PASS** rel 1.8e-6 | **PASS** rel 3.3e-5 |

Two more agreements worth recording because they were not asserted, they were observed:

- The fiducial's certified kernel reproduces phase 0 to the third digit — **11.083 ms** here
  against 11.049 ms then (rectangular, pass 7), **4.249 ms** against 4.213 ms (Delaunay,
  pass 2) — and its PDIP row **25.745 / 28.034 ms** against 25.796 / 28.347 ms. Different
  cell, different process, same numbers.
- The fiducial's A2 (unconstrained) error is **+119.7 nats** on rectangular here, against
  phase 0's +334.93. The difference is **exactly** phase 0's recorded edge-zeroing penalty
  of 215.18 nats: this cell scores A2 against the *full-system* PDIP, phase 0 scored it
  against the *library's edge-zeroed* reconstruction. 334.93 − 215.18 = 119.75. The two
  notes are consistent and neither number is a solver error.

### The draw set is a property of the problem, not of the device

For **both** meshes, all 41 draws, the A100 and CPU legs agree on:

| Quantity | agreement |
|---|---|
| certified pass count to certification | **identical**, all 41 draws, both meshes |
| PDIP iteration count | **identical**, all 41 draws, both meshes |
| seed set `\|{x_unc < 0}\|` and final `\|Z\|` | **identical** |
| Δlog L per draw | rel ≤ **4.2e-6** (rect), **6.9e-7** (Delaunay) |

So every pass-count, fallback-rate and correlation number in this note is one number per
mesh, not one per hardware; only the milliseconds are per-device. That is the validation the
CPU leg was run for, and it also means phase 2's CPU threading finding does not touch these
counts: `NPROC=8` sizes XLA's intra-op pool and governs the JAX millisecond rows, while the
certified pass count is thread-independent by construction (`active_set_certified` is a
numpy reference whose branch decisions are exact comparisons).

## The headline table — the fallback rate

Fraction of the 40 off-fiducial draws whose certification needs **more** passes than the
budget. Identical on both devices (above), so one table per mesh.

| Pass budget | rectangular | delaunay | note |
|---:|---:|---:|---|
| 1 | 100.0 % | 90.0 % | |
| **2** | **100.0 %** | **67.5 %** | Delaunay's phase-0 certifying budget |
| 3 | 100.0 % | 47.5 % | |
| **4** | 97.5 % | **30.0 %** | |
| 5 | 80.0 % | 7.5 % | |
| **6** | **52.5 %** | **2.5 %** | |
| **7** | **27.5 %** | **0.0 %** | rectangular's phase-0 certifying budget / **Delaunay safe** |
| 8 | 10.0 % | 0.0 % | |
| 9 | 5.0 % | 0.0 % | |
| 10 | 5.0 % | 0.0 % | |
| **11** | **0.0 %** | 0.0 % | **rectangular safe** |

Split by family (n = 16 walks, n = 24 random):

| Budget | rect walks | rect random | delaunay walks | delaunay random |
|---:|---:|---:|---:|---:|
| 2 | 100.0 % | 100.0 % | 25.0 % | 95.8 % |
| 4 | 100.0 % | 95.8 % | 6.2 % | 45.8 % |
| 6 | 81.2 % | 33.3 % | 0.0 % | 4.2 % |
| 7 | 25.0 % | 29.2 % | 0.0 % | 0.0 % |

**No draw failed to certify**: `passes_to_certification` is non-null on all 40 draws on both
meshes, so every rate above is a genuine "needs more passes", never a "never certified".
The scan ceiling was 16 and the worst draw used 11.

Read the Delaunay column first, because it is the one production wanted. Phase 0's
**pass-2** budget — the one that bought the 6.7× reduction of the reconstruction row —
fails to certify **two draws in three** across this set, and **23 of 24** random draws. The
walks are much kinder to it (25 %) precisely because they stay close to the fiducial in
every direction but one. A budget chosen on walks would have been badly wrong.

## Pass count versus Δlog L

Spearman ρ against `|Δlog L|`, over the 40 off-fiducial draws:

| ρ vs \|Δlog L\| | rectangular | delaunay |
|---|---:|---:|
| **certified passes** | **−0.535** | **+0.698** |
| PDIP iterations | +0.760 | +0.340 |
| seed set `\|{x_unc < 0}\|` | +0.918 | +0.807 |
| \|A2 Δlog-evidence\| | +0.977 | +0.895 |

### Delaunay — the pass count grows, as phase 0 suspected

| Δlog L target | passes (4 walks) | PDIP it | seed set | A2 Δlog-ev (nats) |
|---|---|---|---|---|
| fiducial | 2 | 17 | 4 | +6.40 |
| −10 | 2, 1, 2, 2 | 17–18 | 4–5 | +4.7 … +6.4 |
| −100 | 2, 2, 2, 2 | 18 | 4–7 | +3.9 … +7.0 |
| −1000 | 2, 1, 1, 1 | 16–17 | 3–10 | +6.4 … +15.0 |
| −1e4 | 3, 5, 4, 3 | 18–21 | 61–85 | +174 … +581 |
| random (n=24) | q25 3, **median 4**, q75 5, max **7** | 17–21 | 43–406 | up to +6339 |

Nothing moves until Δlog L ≈ −10³. At −1e4 and beyond the picture changes completely: the
unconstrained solve starts producing tens, then hundreds of negative pixels (seed set 4 →
406), and the scheme needs 3–7 passes instead of 2. The random draws, which sit at −7.5e3 to
−9.8e4, are entirely in that regime — hence their 95.8 % fallback rate at budget 2.

### Rectangular — the pass count *falls*, and that is not good news

| Δlog L target | passes (4 walks) | PDIP it | seed set | A2 Δlog-ev (nats) |
|---|---|---|---|---|
| fiducial | 7 | 15 | 79 | +119.7 |
| −10 | 8, 7, 7, 7 | 15 | 69–80 | +114.6 … +122.0 |
| −100 | 7, 7, 8, 8 | 15 | 74–78 | +114.5 … +121.9 |
| −1000 | 7, 7, 6, 8 | 15 | 112–137 | +133.5 … +219.9 |
| −1e4 | 6, 7, 6, 7 | 15–17 | 186–386 | +1749 … +3563 |
| random (n=24) | q25 5, **median 6**, q75 8, max **11** | 15–17 | 181–525 | up to +4.07e4 |

The negative correlation is real and has a mechanism. Rectangular's cost is its **dual**
tail: 152 pixels are edge-zeroed from pass 0 and the KKT test then has to *release* the
wrongly-fixed ones one or two at a time (phase 0's 20 → 7 → 4 → 2 → 2 → 2 → 0 trace). At a
bad model the unconstrained solve is negative over a much larger set (79 → 525 pixels), so
pass 0's seed is closer to the true active set and there is less to release: the tail
shortens. **A worse model has a bigger active set but an easier one.**

That is not a licence to keep budget 7. The median is still 7 and the *spread* is 4 to 11,
so 27.5 % of draws exceed it, and the ones that do are not the worst models — they are the
middle ones. Six of the 24 random draws sit in the −10³ decade and need 6 to 11 passes
(median **8**); the 18 in the −10⁴ decade need 4 to 9 (median **6**). **The hardest models
for the rectangular scheme cluster in the moderately-wrong decade, which is where a search
spends its middle phase** — not at its worst first guesses.

## PDIP iterations versus Δlog L — the fallback is cheap and stable

| | rectangular | delaunay |
|---|---|---|
| PDIP iterations, fiducial | 15 | 17 |
| PDIP iterations, draw set (min / median / max) | 15 / 16 / **17** | 16 / 18 / **21** |
| PDIP ms, A100 (min / median / max) | 25.7 / 27.3 / 29.0 | 26.5 / 29.7 / 34.4 |
| PDIP ms, CPU-8 (min / median / max) | 623 / 756 / 991 | 667 / 792 / 971 |

PDIP barely notices the model: +2 iterations on rectangular and +4 on Delaunay over four
decades of Δlog L, an 8–21 % cost range. **This is the finding that makes the
budget-plus-fallback design viable in spite of the fallback rates above**: the fallback
path's cost is nearly model-independent, so the worst case is predictable. Phase 1 measured
that worst case on the library path (route `e`, 41.45 / 51.83 / 58.32 ms) and found it still
faster than the library today; nothing here moves that.

## Certified milliseconds across the draw set

| | rectangular A100 | delaunay A100 | rectangular CPU-8 | delaunay CPU-8 |
|---|---:|---:|---:|---:|
| certified, fiducial | 11.083 | 4.249 | 345.3 | 103.7 |
| certified, draw set min / median / max | 7.10 / 10.98 / **16.60** | 2.84 / 5.49 / **10.74** | 216 / 377 / **631** | 67 / 192 / **384** |
| PDIP / certified, min / median / max | 1.65 / **2.64** / 3.85 | 2.76 / **5.44** / 9.86 | 1.30 / **1.99** / 3.92 | 1.96 / **4.13** / 10.86 |

The scheme never loses on the A100: its **worst** draw (16.60 ms rectangular, 10.74 ms
Delaunay) is still faster than the **best** PDIP call on the same mesh (25.68 / 26.47 ms).
On the CPU that holds comfortably on Delaunay (384 vs 667 ms) and only just fails on
rectangular — the worst certified draw, 631 ms, is 1 % *slower* than the fastest PDIP call,
623 ms, the single crossing anywhere in the four legs. The median speed-up over the draw
set — 2.6× rectangular, 5.4× Delaunay on the A100 — is close to the fiducial-only ratio
phases 0–1 reported, because the certified row's growth at a bad model is partly offset by
PDIP's own.

A note for the CPU column: these are **JAX-CPU** rows on both sides, so phase 2's
"the numpy certified active set does not beat the library's own `fnnls`" finding is not
contradicted — it is a different comparison (numpy certified vs numpy fnnls). What this
table says is that *within the JAX path*, the masked certified scheme beats the JAX PDIP
driver on the CPU by 1.3–3.9× (rect) and 2.0–10.9× (Delaunay) at every model in the set.

## Exactness, and the growing cost of positivity

**Delaunay: the certified scheme is exact at every model.** `|Δlog-evidence| vs the PDIP
solution ≤ 3.7e-10 nats` on all 41 draws. Whatever the pass count does, the answer it
returns at certification is the constrained optimum.

**Rectangular: the delta is the edge-zeroing penalty, not solver error.** It is −215.2 nats
at the fiducial (phase 0's recorded 215.18) and reaches −553.8 at the worst draw, because
the certified scheme solves the library's *edge-zeroed* problem while this cell's PDIP
reference solves the full system. Worth its own line, though: **the cost of the library's
edge-zeroing convention more than doubles at a bad model.** Phase 4 or 5 should decide
whether holding 152 pixels at zero is still the right default when a search spends its time
far from the answer.

**A2 — dropping positivity — gets much worse with model error, as phase 0 predicted.**
ρ(|A2 error|, |Δlog L|) = **+0.977** (rect) / **+0.895** (Delaunay), and the range is brutal:

| | fiducial | worst draw |
|---|---:|---:|
| rectangular A2 Δlog-ev (nats) | +119.7 | **+4.07e4** |
| rectangular negative pixels | 136 | 571 |
| delaunay A2 Δlog-ev (nats) | +6.40 | **+6339** |
| delaunay negative pixels | 4 | 406 |

Phase 0's verdict on A2 was "not free, and blocked on the injection witness". This phase
hardens it: **the positive-negative route's error is a function of model quality, and at the
models a search actually visits it is 3–4 orders of magnitude worse than at the answer.**
Delaunay's "+6.4 nats" — the number that made A2 look almost tolerable on that mesh — is a
fiducial-only number. There is now no case for A2 anywhere.

## The mechanism, in one line

The seed set `|{x_unc < 0}|` tracks model error almost perfectly (ρ = +0.918 / +0.807),
growing 79 → 525 on rectangular and 4 → 406 on Delaunay. **Everything else follows from
that.** On Delaunay a fiducial with 4 negative pixels needs one correction; a draw with 400
needs several, so the pass count grows. On rectangular the free set is already dominated by
the 152-pixel edge-zero seed, so a larger negative set makes pass 0 a *better* guess and the
dual tail shortens. Same mechanism, opposite sign, because the two meshes start from
different pass-0 sets.

## Verdict

1. **The phase-0 pass budgets are fiducial artefacts and must not ship.** Delaunay's pass 2
   falls back on 67.5 % of this draw set (95.8 % of the random draws); rectangular's pass 7
   on 27.5 %. The question this phase was filed to ask is answered: **yes, the 4.2 ms figure
   was partly a consequence of the model being good.**
2. **The safe fixed budgets over this set are 7 (Delaunay) and 11 (rectangular)** — zero
   fallback on 40 draws. At those budgets the certified row costs roughly 10.7 ms (Delaunay)
   and ~16.6 ms (rectangular) on the A100 against 26.5–29.0 ms of PDIP: the lever survives,
   at about half its fiducial-only headline. **Phase 4 and phase 5 should sweep budgets 7
   and 11, not 2 and 7.**
3. **Budget-plus-fallback remains the right design, and is now measured.** PDIP's cost is
   nearly model-independent (15 → 17 and 17 → 21 iterations over four decades), so the
   fallback is a bounded, predictable worst case rather than a tail risk. A production
   policy of "budget 4 with PDIP fallback" on Delaunay would pay the fallback 30 % of the
   time and still beat today's library; "budget 7" pays it 0 % here. The choice is an
   expected-cost calculation the numbers above now support.
4. **The rectangular scheme's hard case is the middle of a search, not its start.** Its
   random draws in the −10³ decade need a median of 8 passes against 6 in the −10⁴ decade,
   and both of the 11-pass draws are in the −10³ decade. Any rectangular tuning should be
   validated there rather than at the extremes.
5. **Positivity is now unambiguous.** A2's error grows with model error to +4.1e4 nats
   (rect) and +6.3e3 (Delaunay). Phase 0's "no case for dropping positivity on the Delaunay
   family" now holds on every mesh at every model. The injection witness is no longer the
   only blocker — the measurement itself is disqualifying.
6. **The draw set is device-independent and reusable.** Pass counts, PDIP iterations and
   seed sets are identical on the A100 and the CPU for all 41 draws. Later phases can build
   the same set from the same seed and compare rows directly.

## Next

Phase 4 of the epic — **source-pixel scaling of the new approach across hardware**
(`PyAutoMind/draft/research/autolens_profiling/fixed_light_source_pixel_scaling.md`) — was
gated on this phase's answer, and now has it: **sweep the certified scheme at budgets 7
(Delaunay) and 11 (rectangular), with the PDIP fallback, not at the fiducial budgets.**
Whether the safe budget itself scales with N_src is an open question this phase cannot
answer — it measured one mesh size per mesh.

Two follow-ups this phase raised that belong to phases 4/5 rather than here:

- **The edge-zeroing penalty grows with model error** (215 → 554 nats on rectangular).
  Whether `solve_ids_to_keep` should still hold 152 pixels at zero during a search is a
  separate design question from the solver.
- **The Delaunay `figure_of_merit` is not smoothly monotone** in a small `centre_x` or
  `slope` offset, because the Hilbert mesh's vertices move discretely with the mass. That is
  a gradient-path concern (it is the same surface a JAX gradient sees), not a solver one,
  and it is worth its own prompt if the JAX gradient work touches Delaunay.

## Artifacts

```
results/breakdown/imaging/fixed_light_draws_{rectangular,delaunay}_hpc_a100_fp64_fixed_light_draws.{json,png}
results/breakdown/imaging/fixed_light_draws_{rectangular,delaunay}_local_cpu_fp64_fixed_light_draws.{json,png}
```

Code:

```
scripts/misc/likelihood_breakdown/fixed_light_draws_steps.py     # the draw set + the derived tables
scripts/imaging/likelihood_breakdown/fixed_light_draws.py        # the cell
scripts/misc/test/test_fixed_light_draws_steps.py                # 30 numpy tests
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_draws_{pixelization,delaunay}_a100_hst_fp64
hpc/batch_gpu/submit_fixed_light_draws.sh
```

JSON keys specific to this cell: `draws` (one row per model — `kind`, `offsets`,
`target_d_log_l`, `d_log_l`, `passes_to_certification` (**null** = never certified),
`certified_ms`, `pdip_iterations`, `pdip_ms`, `seed_set_size`, `final_fixed_set_size`,
`n_primal_violations_per_pass`, `n_dual_violations_per_pass`,
`d_log_evidence_unconstrained_vs_pdip`, `unconstrained_n_negative`,
`d_log_evidence_certified_vs_pdip`), `derived` (`all_off_fiducial` / `walks` / `random`,
each with `fallback`, the four Spearman ρ and quantile summaries), `slope_promotion`,
`fiducial_pins`, `s0_evaluations_total`. The `hpc_a100_fp64_fixed_light_draws` and
`local_cpu_fp64_fixed_light_draws` config labels are README-table-invisible by design
(`CONFIG_TAGGED_RE`), as in phases 0–2; this note and the README prose row are the index
into these legs.

## The phases

| # | Phase | Note |
|---|---|---|
| 0 | Kernel measurement, A100 | [`fixed_lens_light_source_only_2026_09.md`](./fixed_lens_light_source_only_2026_09.md) |
| 1 | Library-path row, A100 | [`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md) |
| 2 | CPU + consumer GPU | [`fixed_lens_light_hardware_2026_09.md`](./fixed_lens_light_hardware_2026_09.md) |
| **3** | **Low-likelihood draws** | **this note** |
| 4 | Source-pixel scaling | filed |
| 5 | HST + Euclid verdict | filed |
