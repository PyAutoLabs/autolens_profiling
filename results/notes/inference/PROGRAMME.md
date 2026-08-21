# PyAutoLens Inference Programme

A phased R&D plan for determining the fastest inference strategies that remain
scientifically correct and reliably find the right solution as PyAutoLens models
grow from ~15 to 100+ dimensions.

**Status: ACTIVE.** Planned 2026-08-17 (planning pass — no runs, no PRs, no
source edits performed); **human-approved 2026-08-17 with notes** (see
`DECISIONS.md`). Evidence base: 6 parallel reconnaissance sweeps over
autolens_profiling, PyAutoFit@d6fef747, PyAutoArray@394514c0, PyAutoLens,
PyAutoMind/Brain/Memory, and the external blackjax/literature ecosystem.

This file is the canonical, maintained copy of the plan (the 2026-08-17
artifact adapted to markdown). It is updated as phases execute; gate verdicts
are recorded append-only in `DECISIONS.md`; external references live in
`LITERATURE.md`; per-phase write-ups land in `phase_<NN>_<slug>/RESULTS.md`.

Evidence tags used throughout: **[VERIFIED]** checked against current
source/artifacts during the planning pass · **[HYPOTHESIS]** untested
expectation · **[CONTRADICTED]** existing evidence pushed back on the brief ·
**[CAUTION]** design constraint.

## Phase & gate state (maintained)

| Item | State | Record |
|---|---|---|
| Phase 0(a) PositionsLH defect fix | **complete** | CP-1: verified + fix shipped, PyAutoLens#699 / PR#700 (DECISIONS.md 2026-08-17) |
| Phase 0(b) blackjax ≥1.6.2 upgrade | **complete** — cloud CPU (PR#143), local venv + RAL stack (2026-08-19, #146) | Cloud: 1.6.2 installs clean next to autofit, `blackjax.nss` 2D smoke PASS, `af.BlackJAXNUTS` unmodified. Local + RAL: 1.5 / 2026-01 fork build → 1.6.2, full smoke PASS in both (now incl. `af.NSS` end-to-end — PR#1492 merged 2026-08-18); RAL library mains re-synced — `phase_00_unblocking/RESULTS.md` |
| Phase 0(c) RAL artifact harvest | **partial** — stranded local fork-NSS A100 artifacts (mge 2nd run, delaunay, pixelization — the §1.2 canonical rows) committed 2026-08-19; RAL NFS harvest (job 331058 SMC arms, 335003-5 NaN-counter arms, Nautilus pixgrad logs) remains | `phase_00_unblocking/RESULTS.md` |
| Phase 0(d) commit plan + ledger structure | **complete** | PR#136 + PR#137 (2026-08-18) |
| Phase 14 default CPU mesh decision | **adjudicated + shipped 2026-08-21** — Bilinear (rank-CDF) default, RTU (kernel-CDF) advanced; both meshes explicit in the cells via `--rect-mesh` (PR #155); versioned Bilinear-vs-RTU measurement outstanding | autolens_profiling#153, PyAutoArray#461/#462 |
| Phase 0(e) searches README dashboard loop | **complete** | PR#139 (2026-08-18): nested-layout scanner, 34 rows render, truth-bar rows verified |
| Phases 1–13 | not started | — |
| Gates A–F | open | — |

---

## 1 · State of play

### 1.1 The Phase-0 checklist, answered

| Question from the brief | Answer | Evidence |
|---|---|---|
| MGE MultiStartProdigy ever profiled with PositionsLH? | **[VERIFIED] NO** | No positions plumbing at all in `scripts/misc/searches/_setup.py`; MGE analyses never pass `positions_likelihood_list`. |
| Mesh MultiStartProdigy profiled with PositionsLH? | **[VERIFIED] NO** | Pixelized analyses explicitly *disable* the positions guard (`_setup.py:1138-1152`, `raise_inversion_positions_likelihood_exception=False`) "for pure profiling". |
| Clipping / step-size variants with PositionsLH? | **[VERIFIED] NO** | `clipper_campaign.py` uses the same positions-free analyses. |
| NUTS ever initialized from previous NS/MAP results? | **[VERIFIED] NO** (in-library) | `af.BlackJAXNUTS` exists but is single-chain, diagonal-mass, no covariance/start-vector injection. A workspace-side warm-start cache (`_warm_start.py`: MLE + Laplace covariance) was built for the SMC prototype and is reusable. |
| Any SMC implementation in PyAutoFit? | **[VERIFIED] NO** | Zero hits for smc/tempered in `autofit/`. A working *prototype* (warm-started gradient SMC with a valid evidence bridge) is parked on wsdev branch `feature/blackjax-smc-gradient-kernel` (issue wsdev#113, open). |
| BlackJAX NSS integration exists / in development? | **[VERIFIED] — REMOVED, NOW MAINLINE UPSTREAM** | `af.NSS` was shipped, benchmarked, and deliberately removed 2026-07-11 (PyAutoFit#1357); implementation stashed at `autofit_workspace_developer/searches/nss/` with pinned fork SHAs. Since then, nested sampling **merged into mainline blackjax** (PR #947, 2026-06-29; released in blackjax 1.6, current 1.6.2). The fork pins are obsolete; re-integration should target `blackjax.nss` mainline. |

### 1.2 What is already established (do not re-run)

The single benchmark cell with the deepest history is `imaging/mge/hst` (15
free parameters; MGE amplitudes linear-solved). Canonical numbers, all A100
fp64 unless noted:

| Cell | Method (config) | max logL | logZ | Wall | Likelihood evals | Verdict |
|---|---|---:|---:|---:|---:|---|
| `imaging/mge/hst` | Nautilus (n_live 200) | 31786.782 | 31690.5 | 831 s | 63,800 | The truth bar. Truth basin is the dominant mode. |
| | NSS fork (n_live 200, 5 inner steps, num_delete 50) | 31786.3–31786.5 | 31697.7 / 31700.4 | 657–679 s | ~390,000 | Matches; mildly faster; logZ sits 7–10 nats *high* (see §2.2). |
| | MultiStartProdigy 16×3000, fp64 laptop GPU, seed 0 | 31787.929 | — | — | 48,000 lane-steps | Beats the bar by 1.15 nats. |
| | same, seed 1 | −139,485.8 | — | — | 48,000 | **171,272 nats away** — degenerate θ_E=0 "no lens" basin on the U(0,8) lower wall. Clipping and per-parameter step scaling both tested; neither rescues it. |
| | Multi-start Adam 128×, fp32 A100 (warm) | 31787.9 | — | 50 s | — | p_hit ≈ 0.18/start, stable; ~10× faster than Nautilus (523 s same node). |
| `imaging/delaunay/hst` | Nautilus (n_live 150) | 30623.5 | 30562.2 | 2,723 s | 31,536 | NSS matched the answer but was **11× slower** (inner slice evals dominate). |
| | NSS fork (same config) | 30622.2 | 30567.8 | 29,770 s | 206,448 | (same row-pair verdict) |
| `imaging/pixelization/hst` | Nautilus / NSS fork | 29143.3 / 29142.5 | 29066.3 / 29078.9 | 2,768 / 19,190 s | 58k / 266k | NSS **7× slower**; logZ +12.6 nats. |

Other settled findings the plan builds on rather than re-derives:

- **[VERIFIED] Basin selection, not speed, is the discriminator for gradient
  MAP.** Every cold single-start optimizer lands in a wrong basin; only
  population methods (multi-start, SVGD) recover truth. Line-search /
  quasi-Newton methods (L-BFGS, BFGS, NCG, LM/GN) categorically fail on the
  NNLS-kinked objective; fixed-step first-order methods are robust.
  Adam→L-BFGS polish is actively harmful. (wsdev #95/#97 + phase-3 findings
  docs.)
- **[VERIFIED] Lane deaths are prior-support events, not likelihood NaNs**
  (MGE); the clipper eliminates deaths at zero accuracy cost but does not
  change which basin wins. Step-scaling was falsified with three of four
  pre-registered conditions firing. (autolens_profiling #128/#131, PR#133.)
- **[VERIFIED] Pixelized gradients work** in the production shape (adaptive
  meshes at os_pix=4; kernel-CDF bandwidth transform; Delaunay via
  stop-gradient connectivity tables; new autodiff Sibson DelaunayNN merged
  2026-08-09). The regularization log-det NaN is localized: absolute 1e-8 lift
  below the eigenvalue noise floor; slogdet opt-in shipped for both log-det
  terms (PyAutoArray#391). ConstantZeroth confirmed dead code on main.
- **[VERIFIED] Compile time is solved as a settings problem** (persistent
  cache + autotune-off: worst cold fit ~70 min → ~35 s). Never restructure a
  likelihood or sampler for compile time. Delaunay-family cache misses remain
  a known exception.
- **[VERIFIED] Warm-started gradient SMC samples correctly** once (a) whitened
  by the full Laplace Cholesky, (b) step-scaled to the posterior width, (c)
  evidence kept via a geometric bridge from a *normalized* Gaussian reference.
  Cold-starting posterior samplers on this likelihood is meaningless (~190,000
  nats from the solution). Parked, resumable; RAL job 331058 (MALA-tuned/HMC
  arms) is unharvested.
- **[VERIFIED] MultiStartGradient auto-convergence already exists** (plateau
  of global-best FOM: window 50, rtol 1e-4, atol 1e-3, min 100 steps;
  `n_steps` is a ceiling; `stop_reason` persisted) — but it is *disabled when
  `resurrect=True`* (the pixelized regime) and cannot distinguish a
  wrong-basin plateau from convergence.
- **[VERIFIED] Cluster-scale point-source objectives are not
  gradient-friendly** — Prodigy uncompetitive on every cluster cell; Nautilus
  is the cluster-scale answer today (point-source defaults campaign,
  PyAutoLens#678).
- **[VERIFIED] Hazard ledger** holds the NNLS active-set-kink finding (toy
  scale only: 31-point θ_E grid, ~10 source pixels, 5 transitions, relative
  likelihood steps 0.006–0.014) and *two* curvature-floor records — including
  a real-path one (floor ≤ 4.5e-5 of touched diagonal scale; scale-aware
  counterfactual max output error 3.4e-5). Phase 10 is smaller than the brief
  assumes.

### 1.3 Canonical reference map (stale numbers resolved)

| Loose reference | Canonical | State |
|---|---|---|
| "#104" (reg-NaN localization) | **autolens_workspace_developer#104** (not autolens_profiling#104, which is a jax_compile PR) | closed |
| "#117 campaign" in _setup docstrings | autolens_workspace_developer#117 (pix Prodigy campaign) | closed |
| Clipper arc | autolens_profiling issues #128, #129, #131 (all **open**); PRs #130/#132/#133 (#133 = step-scaling falsification + θ_E=0 diagnosis) | mixed |
| NaN counters / fitness guard / slogdet / scaler | PyAutoFit#1473, PyAutoFit#1391-92, PyAutoArray#391-92, PyAutoFit#1483/1485 | all closed/merged |
| SMC prototype | wsdev#113 + branch `feature/blackjax-smc-gradient-kernel` @ 6867762 | open, parked |
| Open sampler-benchmark issues in autolens_profiling | #69 (optimizer settings tuning — its "pixelized out of scope" note is stale), #82 (group MGE benchmark), #103 (warm-compile baselines) | open |

## 2 · Where the evidence corrects the brief

Six places where reconnaissance changed the programme as written (adversarial
review, as requested by the brief):

### 2.1 [CONTRADICTED] A PositionsLH accumulation defect must be fixed before any Phase 4/5 result is interpretable

`AnalysisLens.log_likelihood_penalty_from`
(`autolens/analysis/analysis/lens.py:165-181`) appeared to overwrite the
accumulator inside its loop and then add it to itself: for N penalty objects it
returned **2× the last entry's penalty**, discarding the rest; for the common
single-entry case, 2× the documented penalty. Every PositionsLH experiment
would otherwise benchmark an undocumented target. Verify → fix →
regression-test was Phase 0 work, and because SLaM pipelines pass exactly one
penalty, the fix plausibly shifts published-pipeline likelihood values — the
fix itself needed a target-change classification (§4, Phase 0) and its own
tiny before/after record.

> **Resolution (2026-08-17): verified real and fixed.** Shipped as
> PyAutoLens#699 / PR#700 — CP-1 complete. See `DECISIONS.md`.

### 2.2 [CONTRADICTED] BlackJAX NSS is not currently the scalable candidate on pixelized cells — and the fork pins are obsolete

NSS has *already* been benchmarked here: ~1.2× faster than Nautilus on MGE, but
**7–11× slower on delaunay/pixelization** (206k–266k evals vs 31k–58k). Two
confounds make that non-final: (i) it ran `num_mcmc_steps=5` inner steps in
15–20D, where the now-merged upstream guidance is ≥ max(5, 2·d) —
under-mixing that also predicts the observed **+7–13 nat logZ bias** in every
recorded NSS row; (ii) it was the 2026-01-era fork. Nested sampling is now in
**mainline blackjax 1.6.2** (native-space `logprior_fn` API, `num_delete` as
the GPU axis, dlogz termination, ensemble logZ error bars). Phase 2 is
therefore a *re-tuning of a known quantity against a moved ecosystem*, with
the logZ-bias hypothesis as its sharpest pre-registered test — not a first
integration. Also: the local/RAL environments run blackjax 1.5, which is
incompatible with the installed `nss` wrapper (verified import failure) —
environment upgrade is Phase 0 work.

> **Review note (2026-08-17):** BlackJAX NS was fastest or comparable on MGE
> and may scale better — worth rerunning. Phase 2 stands as planned.

### 2.3 [HYPOTHESIS, REFRAMED] Phase 3's question is a per-start hit probability, not a seed lottery

The 16-lane seed-0/seed-1 split reads naturally as binomial luck: multi-start
Adam measured p_hit ≈ 0.18/start on this cell, at which P(all 16 miss) ≈ 4% —
roughly one seed in 25. GIGA-Lens, solving similar smooth models, runs **300
starts** and reports 1–5% per-start hit rates. So Phase 3 should *measure
p_hit for Prodigy directly* (per-lane basin classification across n_starts ×
seeds) and derive reliability as 1−(1−p)^n, rather than accumulating more
whole-run anecdotes. That also converts the brief's key question into
arithmetic: the n_starts where reliability crosses (say) 99% has a computable
cost, comparable head-to-head with Nautilus/NSS eval budgets. One blocker:
PyAutoFit currently preserves each lane's *final* position but not its *best*
— a small source change Phase 3 needs first.

### 2.4 [SUPPORTED, WITH A TWIST] PositionsLH should fence the θ_E=0 basin — but its gradient shape needs characterizing first

The penalty is `1e8 × (max_separation − threshold)` outside the threshold,
exactly 0 inside, via `lax.cond`; pure deflection-map tracing, no PointSolver;
explicitly JAX-differentiable. The θ_E=0 basin un-lenses the positions, giving
a large max-separation — so the penalty converts that basin from "local
optimum" to "steep downhill slope away from it," which is precisely the
missing global structure. Three structural cautions the brief should absorb:
the penalty has a **C⁰ hinge** at the threshold, an **argmax-switching kink**
inside `max()`, and — by design — **zero gradient inside the threshold**: it
fences, it never guides. And a 1e8 slope against a ~3×10⁴-scale logL may fling
Prodigy lanes (its step adapts to gradient scale). Phase 4 therefore starts
with a transect/eager-gradient characterization and a penalty-factor
sensitivity arm, not with full campaigns. Note the workspace frames positions
as an early-stage aid with thresholds loose enough to be inactive at the
solution, yet SLaM keeps the penalty active (auto-threshold, factor 3, min
0.2) through the final MASS stage — the double-counting question is genuinely
open and empirically decidable (Phase 4).

### 2.5 [VERIFIED] Half of Phase 7-8 groundwork already exists

The "mathematically valid bridge" the brief asks SMC to plan is *already
derived and prototyped* (temper geometrically from a normalized Gaussian
reference g: `logprior := log g`, `loglik := log π + log L − log g`; evidence
preserved) — Phase 7 resumes the parked wave, it does not redesign. Likewise
8A's slogdet opt-in is shipped and 8C's ConstantZeroth bugs are already filed
with a draft. And one connection the brief misses: the regularization
coefficient priors are `LogUniform(1e-6, 1e6)`, so **nested samplers already
sample in log-space** via the unit cube — the λ⁴ pathology is specific to
physical-space gradient steppers. 8B's "log-space coordinate" is exactly "step
in the prior's own CDF coordinate," which generalizes: a per-parameter
bijector for the multi-start searches subsumes both 8B and the previously
rejected unit-cube-stepping idea (rejected in an arc whose diagnosis was later
falsified — worth one deliberate revisit).

### 2.6 [CAUTION] Two hardware facts constrain the campaign design

(i) On plain Delaunay, `batch_size` changes *which optimum is found* (batch 4
lands 5,622 nats below batch 2; DelaunayNN is insensitive) — so laptop-GPU
mesh results carry a correctness confound, not just a wall-time one: mesh
*conclusions* need A100 or fixed-batch discipline. (ii) Vmapped many-chain
NUTS runs in lockstep and pays the slowest chain's tree depth every step
(documented 43× pathologies) — the GPU-native posterior samplers are the
fixed-work ones (ChEES/MEADS-adapted HMC, MCLMC/MAMS), which is why Phase 6
carries a ChEES arm rather than assuming vmapped NUTS.

## 3 · Programme-wide experimental rules

- **Target ≠ algorithm.** Every run records a `target_id` — a hash over model
  composition, priors, likelihood settings (log_det_method, positivity,
  floors, precision), PositionsLH config (on/off, positions set, threshold,
  factor), dataset, and mask. Two runs are comparable only when target_ids
  match; any cross-target claim must state that the target difference *is*
  the experiment (this is how the PositionsLH and slogdet phases stay honest).
- **Reliability is P(correct | fixed budget), measured over ≥5 seeds** at
  every decision point; single-seed results are anecdotes. Success = max logL
  within a stated tolerance of the target's truth bar *and* parameter
  recovery within stated tolerances of the simulator truth.
- **Per-step histories, not lifetime totals**, for every counter added
  (alive, clips, NaNs, acceptance, temperature schedule) — totals cannot
  answer "when."
- **Per-lane preservation:** every lane/chain/particle-set final *and best*
  state survives into artifacts. No winner-only records.
- **Cold vs warm split:** compile time recorded separately (cache state
  named); never folded into algorithmic wall time. Compile timings only from
  idle/dedicated nodes.
- **Hardware tiers never mix in one comparison row.** Laptop GPU decides what
  deserves A100 time; RAL CPU carries MGE-scale volume; A100 carries mesh
  work and confirmations. Every RAL sbatch exports `JAX_ENABLE_X64=True`
  explicitly and the artifact records precision + device.
- **Smoothing taxonomy enforced at record level:** every
  likelihood/parameterization change carries `target_class: 1|2|3`
  (target-preserving / equivalent-within-tolerance / target-changing) with
  the proof or measurement attached. Category 3 is never presented as a
  numerical fix (Gate F).
- **Tuning hierarchy:** cheap config scan → operating region → repeated-seed
  reliability → A100 confirmation. Every knob experiment answers a named
  question and lands in the method's configuration record.
- **Termination is a benchmark metric:** record `stop_reason`,
  steps-after-best (waste), and — for plateau detectors — outcome
  classification against wrong-basin plateaus, not just successful runs.
- **Delegation:** per the workspace contract, planning/judgment stays in the
  main session; mechanical execution phases delegate to Sonnet-class
  subagents; independent adversarial review (falsify-the-interpretation
  passes) precedes every gate decision that commits A100 time or a source
  change.

## 4 · Phase-by-phase plan

Expense scale: **S** < 1 laptop/CPU-day · **M** = several laptop + RAL-CPU
days · **L** = one or more dedicated A100 sessions · **XL** = multi-week A100
campaign. Every phase writes its results under
`results/notes/inference/phase_<NN>_<slug>/RESULTS.md` plus schema-conformant
JSON under `results/searches/`.

### Phase 0 — Reconnaissance remainder & unblocking

- **Question:** Is the bench trustworthy? (The archaeology itself is done —
  §1 is the deliverable.)
- **Work items:** (a) **Verify + fix the PositionsLH accumulation defect**
  (§2.1) with a regression test and a before/after likelihood record on one
  SLaM-style fit — classify the fix's target impact explicitly.
  *(Done: PyAutoLens#699 / PR#700.)* (b) **Upgrade blackjax → ≥1.6.2** in the
  local venvs and the RAL stack; smoke `blackjax.nss` import + a 2D toy;
  confirm `af.BlackJAXNUTS` against the 1.6 API. (c) **Harvest stranded RAL
  artifacts**: SMC warm arms (job 331058), NaN-counter split arms (335003-5),
  any Nautilus pixgrad baseline log — commit summaries in-repo so results
  stop living only on NFS. (d) Commit this plan + create the programme ledger
  structure (§6) + targets registry skeleton. *(Done: this commit covers the
  plan + ledger; targets registry skeleton outstanding, moves to Phase 1.)*
  (e) Close the loop on the searches README dashboard ("No data yet" despite
  existing artifacts).
- **Hardware / cost:** Laptop + RAL login. **S**.
- **Gate:** Everything downstream depends on (a) and (b); nothing else
  proceeds until both land.

### Phase 1 — Standard benchmark matrix & targets registry

- **Question:** Can every future result be expressed as (target,
  algorithm-config, seed, hardware) → metrics, stable across years?
- **Design:** Formalize what §1 shows already half-exists. Targets v1 (all
  HST fixtures, existing `_setup.py` builders): `mge` (15D), `delaunay`
  (ConstantSplit), `knn` (free AdaptSplit — the λ⁴ stressor),
  `delaunay_matern`, `delaunay_nn` (new Sibson autodiff), and a new
  `slam_source_pix` target mirroring the production SLaM SOURCE-PIX shape
  (RectangularAdaptImage + reg.Adapt + AdaptImages + border relocator +
  positions). Each target × {positions off, positions on(threshold set)} ×
  precision. Extend the truth-anchor mechanism (currently point_source/cluster
  only) to imaging cells; store a long-run Nautilus reference posterior + logZ
  per target as a named baseline (`results/baselines/InferenceRefs_v1/`).
- **Pass/fail metrics:** Correctness: Δ(max logL) vs truth bar ≤ 2 nats and
  parameter recovery within per-target tolerances. Reliability: success
  fraction over ≥5 seeds. Posterior agreement: per-parameter mean shift ≤
  0.2σ_ref and σ ratio ∈ [0.8, 1.25] vs reference. Evidence: |ΔlogZ| within
  combined error estimates. Performance: wall (per tier), likelihood +
  gradient evals, ESS/s where applicable, peak VRAM, compile split.
- **Cheap first:** MGE target end-to-end through the schema before any mesh
  target is registered.
- **Hardware / cost:** Laptop + RAL CPU; one A100 session to refresh
  reference posteriors. **M**.
- **Depends:** Phase 0(a) — references bake the fixed penalty in.
- **Outputs:** `_targets.py` registry + schema v2 (§5) + reference baselines
  + per-target tolerance docs.

### Phase 2 — Global MGE: Nautilus vs mainline BlackJAX NSS

- **Question:** Is GPU-native nested sampling ready to replace/parallel
  Nautilus as the global baseline, at matched correctness?
- **Hypotheses:** **[H2.1]** Mainline NSS with `num_inner_steps ≥ 2d` removes
  the +7–13 nat logZ bias (under-mixing per upstream docstrings). **[H2.2]**
  With `num_delete` tuned (k/m ≈ 0.1–0.5), NSS beats Nautilus wall-time on
  MGE on A100 while matching posterior + logZ. **[Standing contrary
  evidence]** fork NSS was 7–11× slower on pixelized cells — Gate A must be
  judged per model family, not on MGE alone.
- **Arms:** Nautilus (n_live 200 SLaM-mirror + one n_live-scan row); NSS scan
  over n_live {200, 500, 1000} × num_delete {0.1m, 0.25m, 0.5m} × inner steps
  {5 (replicates history), 2d, 3d} × dlogz {−3, −10}; then ≥5-seed
  reliability at the chosen operating point; integration wired via
  `model.vector_from_unit_vector`/log-prior in a profiling-local runner
  first, `af.NSS` re-mainlining only after Gate A.
- **Metrics:** Full §5 schema; specifically logZ ensemble error (native NSS
  feature), evals, evals-per-replacement, GPU utilization/VRAM vs num_delete,
  posterior agreement, termination behaviour at both dlogz values.
- **Cheap first / expensive second:** Laptop-GPU + RAL-CPU scans → single
  A100 confirmation of the operating point + one pixelized-cell probe
  (decides whether Phase 5 carries NSS as a serious arm or a reference-only
  arm).
- **Hardware / cost:** **M** + 1 A100 session.
- **Gate:** **GATE A.** NSS matches Nautilus posterior+evidence within
  Phase-1 tolerances AND wins wall-time on GPU → NSS becomes the principal
  nested-sampling baseline for GPU campaigns (Nautilus remains the CPU
  reference). If the pixelized deficit survives tuning, NSS is scoped to
  parametric models and Nautilus keeps mesh duty.
- **Depends:** Phase 0(b), Phase 1.

### Phase 3 — Final MultiStartProdigy investigation (MGE, from broad priors, no positions)

- **Question:** What is Prodigy's per-start basin-hit probability p, at what
  n_starts does 1−(1−p)^n make it reliable, and is it then still cheaper than
  nested sampling?
- **Hypotheses:** **[H3.1]** The limitation is basin discovery (θ_E=0
  attractor), not bound handling — carried forward from #133, treated as the
  null. **[H3.2]** p_hit(Prodigy) is O(0.1–0.2) like Adam's, making 16 lanes
  a ~2–5% failure lottery and 64–128 lanes reliable. If p_hit is much lower,
  Prodigy is done as a global searcher regardless of n.
- **Pre-req (source):** PyAutoFit: preserve per-lane *best* (position, FOM,
  step index) alongside final — small change, ships behind Gate review;
  without it per-lane basin classification is unreliable.
- **Arms:** Best-supported config only (clip=prior_box for hygiene, no
  momentum reset, no scaler, auto-convergence ON with stop_reason
  accounting): n_starts {16, 64, 256} × ≥5 seeds, budget-matched tables vs
  Nautilus/NSS eval counts. One explicitly-labelled *diagnostic* arm: θ_E
  prior U(0.2, 8) (target-changing, mechanism probe only — does removing the
  wall's intersection with the degenerate basin recover seed 1?). No further
  scaling/clipping variants without evidence.
- **Metrics:** Per-lane basin classification, p̂_hit with binomial CI,
  reliability curve, convergence-detector confusion matrix (stopped-correct /
  stopped-wrong-basin / ceiling), full lane counters + alive curves, wall +
  evals at each n.
- **Cheap/expensive:** All of this is RAL-CPU + laptop-GPU scale (MGE); one
  A100 row only to time the 256-start config honestly.
- **Hardware / cost:** **M**.
- **Gate:** **GATE B (part 1).** If no n_starts ≤ 256 gives ≥99% reliability
  at cost below the nested-sampling budget, classify MultiStartProdigy as a
  LOCAL/INITIALIZED optimizer for parametric models and stop global-search
  investment (Phase 4 then only measures whether PositionsLH changes that
  classification).
- **Depends:** Phase 1; the per-lane-best source change.

### Phase 4 — PositionsLH on MGE (all three engines, identical treatment)

- **Question:** Does the position penalty eliminate the catastrophic basin
  for gradient search, what does it cost the nested samplers, and does it
  belong in final inference at all?
- **Hypotheses:** **[H4.1]** The θ_E=0 basin acquires a monotone 1e8-scale
  penalty slope → Prodigy p_hit rises substantially. **[H4.2]** The
  hinge/argmax kinks and zero-gradient interior create *new* failure modes
  (lane fling, oscillation at the fence) — measured, not assumed. **[H4.3]**
  At the converged posterior the penalty is exactly zero for essentially all
  accepted samples (loose-threshold design) → no double-counting in practice;
  if a non-trivial posterior fraction has an active penalty, positions are
  informative in final inference and the double-count concern is real.
- **Stage 1 (characterize, S):** Eager value_and_grad transects through the
  threshold and across the θ_E=0 basin; hazard-ledger record for the
  hinge/plateau/cliff; penalty-factor sensitivity (1e5 / 1e8) and threshold
  sensitivity (0.3 fixed vs SLaM-style factor-3 auto). Positions from the
  simulator's `positions.json` (truth positions — an idealization to note in
  the record; real-data positions are hand-drawn).
- **Stage 2 (campaign, M):** Matched trio at identical target: Nautilus /
  NSS(Phase-2 operating point) / Prodigy(best config, n_starts from Phase 3)
  × ≥5 seeds × {positions on, off}. Measure: basin-failure elimination, p_hit
  shift, NS wall/eval delta, posterior + evidence consistency on/off (the
  direct double-count measurement: compare truth-basin posteriors), fraction
  of posterior samples with active penalty.
- **Hardware / cost:** RAL CPU + laptop GPU; A100 only for the confirmation
  rows. **M**.
- **Gate:** **GATE B (part 2).** PositionsLH + Prodigy reliable across seeds
  AND substantially cheaper than NS → adopt as constraint-guided MAP engine
  (search stages). Still basin-sensitive → stop trying to make Prodigy a
  global solver; it becomes initialized-only. Separately: if H4.3's
  zero-active-penalty holds, recommend positions-on for search with a
  documented no-double-count argument; if not, recommend positions for early
  stages only and record the science rationale.
- **Depends:** Phases 0(a), 2, 3.

### Phase 5 — Pixelized / mesh global searches with PositionsLH

- **Question:** With wrong-mass basins fenced by positions, can Prodigy
  reproduce nested-sampling answers on mesh targets at a fraction of the cost
  — and how much residual mesh nonsmoothness remains as the limiting factor?
- **Hypotheses:** **[H5.1]** Positions (already "essentially required" for
  pixelized NS per project record — demagnified-source local maxima) raise
  Prodigy's mesh p_hit as in Phase 4. **[H5.2]** Plain Delaunay remains
  unreliable regardless (flip discontinuities + sqrt(dual_area) grad-NaN +
  batch sensitivity); DelaunayNN (continuous Sibson) and the kernel-CDF
  rectangular meshes are the gradient-viable ones. This is a mesh-family
  ranking experiment as much as a sampler experiment.
- **Arms:** Targets: `delaunay`, `delaunay_nn`, `knn`, `slam_source_pix` —
  all positions-on (matched thresholds). Engines: Nautilus / NSS (scope per
  Gate A) / Prodigy (resurrect=on ⇒ ceiling-budget; batch discipline: batch 2
  for plain Delaunay, per the steering hazard; full NaN/clip/alive
  accounting; per-lane best preserved). ≥3 seeds minimum (mesh cost), 5 where
  affordable.
- **Metrics:** Reliability, Δ to per-target truth bars, lane-level NaN
  attribution split by axis (mesh sqrt vs reg λ⁴), wall + evals + VRAM,
  batch-size sensitivity check on any surprising Delaunay result.
- **Cheap/expensive:** Laptop only for mechanics/smoke (6 GB VRAM caps batch;
  batch confound documented); all conclusions from A100 rows.
- **Hardware / cost:** **L** (primary A100 phase).
- **Gate:** Extends Gate B to mesh targets, per mesh family. Also feeds Gate
  E: if Prodigy/NUTS failures localize at mesh/NNLS/reg nonsmoothness sites,
  Phases 8–9 get their production-scale justification here.
- **Depends:** Phases 2–4.

### Phase 6 — Initialized posterior sampling (NUTS + GPU-native alternatives)

- **Question:** Once the right basin is known, is gradient MCMC the natural
  posterior engine — and which variant on which hardware?
- **Hypotheses:** **[H6.1]** NUTS warm-started at the MAP with a
  dense/low-rank mass matrix from the previous fit's covariance achieves
  ESS/s ≫ nested sampling on MGE (the 269× prior/posterior anisotropy and
  |r|=0.95 correlations are measured — diagonal mass will not suffice).
  **[H6.2]** Vmapped multi-chain NUTS underperforms its theoretical
  parallelism (lockstep tree depth); ChEES-adapted HMC or MAMS matches/beats
  it at chain counts ≥16.
- **Pre-req (source):** PyAutoFit: multi-chain BlackJAXNUTS (vmapped),
  inverse-mass-matrix injection (diagonal/dense/low-rank), start-point
  injection from a previous `Result` *without touching priors* (wire
  `InitializerParamStartPoints` + a covariance carrier; promote the wsdev
  `_warm_start` cache pattern toward a PyAutoFit warm-start abstraction).
  Scientific priors stay untouched — start points and metric only.
- **Arms:** Init sources: Nautilus posterior (mean/cov), Prodigy MAP +
  Laplace. Samplers: NUTS (1 chain; 16 vmapped chains), ChEES-HMC (16+
  chains), MAMS (adjusted MCLMC) as a third arm only if the first two leave a
  gap. Targets: MGE, then `delaunay_nn` + `slam_source_pix`
  (reverse-mode-only NNLS is fine — MCMC needs gradients, not JVPs).
  Positions per Phase 4 verdict.
- **Metrics:** Warmup cost, divergences (count + location), acceptance,
  ESS_bulk/ESS_tail (rank-normalized, blackjax 1.6 diagnostics), split-R̂ <
  1.01, ESS per gradient eval, ESS/s (per tier), posterior agreement vs
  nested reference, VRAM, compile split, tree-depth distribution (lockstep
  loss measurement).
- **Cheap/expensive:** MGE on laptop GPU → A100 mesh confirmation.
- **Hardware / cost:** **M** → **L**.
- **Gate:** **GATE C.** Initialized gradient MCMC with excellent ESS/s and no
  material divergences on the truth basin → default initialized posterior
  engine (record which variant per hardware). Divergences clustering at
  NNLS/reg sites → Phase 9 gains its inference-failure evidence (Gate E
  input).
- **Depends:** Phases 2–5 (initializers + basin knowledge).

### Phase 7 — SMC and other high-dimensional candidates

- **Question:** Does SMC solve a problem the Phase-6 engine cannot — mode
  multiplicity under model transitions, or informed-but-not-trusted
  initialization?
- **Approach:** **Resume, don't redesign.** Harvest job 331058; port the
  parked prototype onto `blackjax.adaptive_tempered_smc` (ESS-threshold
  tempering, MALA/HMC rejuvenation, inner-kernel tuning). The informed-start
  bridge already exists (normalized-Gaussian geometric path with exact
  evidence bookkeeping) and matches the literature's density-tempering
  construction; formalize it as the candidate PyAutoFit abstraction *only
  after* it earns its keep. Test on a transition that genuinely risks new
  modes (Phase 12's SOURCE→MASS handoff, or a multipole-degenerate target
  from Phase 13) — a bridge from an informed start cannot resurrect a mode
  the start missed; that failure mode gets an explicit test (deliberately
  mode-dropped initialization).
- **Sampler-zoo guard:** Pathfinder (initializer only, Pareto-k̂ diagnostic)
  and MAMS enter only against a named failure from Phases 6/13. JAXNS
  gradient-guided NS stays a literature comparator (upstream marks it
  experimental).
- **Metrics:** ESS trace across tempering, acceptance per λ, rejuvenation
  adequacy, mode-recovery under seeded bimodality, logZ vs nested reference,
  wall/evals, particle-count scaling.
- **Hardware / cost:** **M**; A100 rows only after Gate D.
- **Gate:** **GATE D.** Promote SMC only if it demonstrably solves
  multimodality / risky transitions / higher-D exploration that NUTS-class
  engines fail — otherwise it stays a documented research result.
- **Depends:** Phase 6; Phase 0(c).

### Phase 8 — Likelihood smoothness: regularization

- **Question:** Can the same AdaptSplit physical likelihood be made
  numerically smooth/robust for differentiation without moving the target?
- **8A — slogdet (S):** Re-run the free-AdaptSplit stressors (knn target
  truth-bar region; replay the recorded rejected draws) under
  `log_det_method="slogdet"`. Pre-registered: NaN disappearance, value
  equality on PD points (slogdet ≡ cholesky there; deltas only where cholesky
  failed — quantify as category 2), gradient finiteness, runtime,
  Prodigy/NUTS delta. If clean → recommend slogdet as the gradient-work
  default profile (opt-in remains; PyAutoArray default untouched).
  *[HYPOTHESIS: expected to pass — cheapest high-value test in the
  programme.]*
- **8B — log-coordinate (M):** Category-1 reparameterization: gradient
  searches step in log λ (the prior's own CDF coordinate — the physical
  AdaptSplit likelihood is evaluated unchanged; MAP objective needs no
  Jacobian, samplers in transformed coordinates carry the standard Jacobian
  and are proven equivalent). Implementation lever: generalize the Scaler
  slot into a per-parameter bijector (subsumes the falsified-diagnosis-era
  unit-cube-stepping question — one deliberate revisit, pre-registered).
  Measure: NaN wall position in steps, free-AdaptSplit convergence
  (historical: 2,200 steps vs 98 fixed), λ-trajectory behaviour. Removing
  AdaptSplit's squaring is out of scope except as an explicitly
  TARGET-CHANGING mechanism probe, if ever.
- **8C — ConstantZeroth (S/M):** Fix the two filed bugs (dead code: eye(P)
  shape + missing arg) as an ALTERNATIVE scheme, labelled target-changing
  relative to AdaptSplit. Verify the λ_z²I null-mode-lift hypothesis against
  the measured spectrum (eig_min pinned at 1e-8). Value: a well-conditioned
  user alternative — never a silent AdaptSplit replacement.
- **8+ — analytic log-det (S probe):** Fixed-topology rectangular meshes
  admit exact `Σ log(λ²μᵢ + ε)` (eigenvalues constant across the fit) —
  category 1, zero science change, kills the Cholesky entirely on those
  meshes. Not implemented anywhere today; one-day feasibility probe, source
  change if it wins.
- **8D — A100 comparative (L):** Justified variants only, on knn +
  slam_source_pix: likelihood equivalence, NaN/grad-NaN, convergence,
  Prodigy/NUTS/nested behaviour, runtime, posterior agreement.
- **Gates:** **GATES E & F.** Smoothing investment only where Phase 5/6 tied
  a production-scale nonsmoothness to an actual inference failure; no
  category-3 change is ever recommended as a numerical fix.
- **Depends:** Phases 5–6 for the failure evidence; 8A can start any time
  after Phase 0.

### Phase 9 — Likelihood smoothness: NNLS positivity

- **Question:** Do active-set kinks exist at production scale, do they
  measurably harm Prodigy/NUTS, and if so what smooth formulation preserves
  the science?
- **Stage 1 — confirm at scale (M):** The only existing evidence is a
  ~10-source-pixel toy (5 transitions across θ_E ∈ [0.1, 1.6], relative steps
  0.006–0.014). Instrument the production-scale solve (1500-pixel meshes):
  support-set size/changes along physically relevant transects (numpy fnnls
  exposes the active set exactly; JAX PDIP via reconstruction sign pattern),
  correlated with eager-AD gradient jumps and with Phase 5/6 diagnostics
  localized at the same parameter values (Prodigy deflections, NUTS
  divergence positions). Measure: transition density per unit θ, gradient
  jump magnitude vs typical gradient scale.
- **Stage 2 — smoothing research (conditional, M→L):** Only past Gate E.
  Natural first candidate: the JAX forward already runs a primal-dual
  interior-point solver — a **finite-μ barrier forward** (stop the barrier at
  μ>0 instead of driving to the exact vertex solution) is an in-family
  Moreau-Yosida-style smoothing with literature backing (proximal/ns-HMC).
  Alternatives: smooth positivity reparameterization (softplus amplitudes).
  Each classified: finite-μ and reparameterizations are **category 3** (they
  change the positivity model) unless proven within tolerance (category 2) —
  compare source morphology, likelihood, lens posterior, and derived
  quantities before any recommendation. Note the literature has no
  lensing-specific precedent here (verified gap) — findings are
  publishable-grade documentation for the ledger.
- **Hardware / cost:** Stage 1: laptop/RAL + one A100 day. Stage 2: **L** if
  gated in.
- **Gate:** Gate E (production-scale + correlated failure) before Stage 2;
  Gate F on any adoption.
- **Depends:** Phases 5–6 diagnostics.

### Phase 10 — Curvature-diagonal floor (small evidence run)

- **Scope check:** Smaller than briefed: a *real-path* hazard record already
  bounds the floor at ≤4.5e-5 of touched diagonal scale with a stable
  scale-aware counterfactual (output error ≤3.4e-5); a doc alignment shipped
  2026-08-13. What is missing is only the posterior-level statement.
- **Design:** One representative HST pixelized fit: current 1e-3 absolute
  floor vs scale-aware counterfactual → likelihood delta, reconstruction
  delta, short-Nautilus posterior delta, gradient smoothness, runtime.
  Expected outcome: "no default change justified" — recorded either way;
  PyAutoArray change only if this run surprises.
- **Hardware / cost:** **S** (one A100 or laptop day). Independent —
  schedulable any time after Phase 1.

### Phase 11 — Freeze the "good likelihood" baseline

- **Deliverable:** A canonical recommended likelihood configuration for the
  scaling campaign: what changed (e.g. slogdet-on profile, positions policy,
  any adopted smoothings) / what did not (AdaptSplit values, defaults) /
  which changes are category 1 vs 2 (with proofs/measurements attached) /
  remaining known kinks with evidence they do or don't matter. Re-baseline
  the Phase-1 reference posteriors if anything moved. Targets registry v2
  tagged; Phases 12–13 run only against frozen targets.
- **Hardware / cost:** **S** + reference re-runs if needed.

### Phase 12 — SLaM pipeline experiment

- **Question:** What is the fastest reliable engine per SLaM stage, using
  real stage structure and real chaining?
- **Design:** Actual SLaM (SOURCE LP → SOURCE PIX 1/2 → LIGHT → MASS TOTAL,
  current n_live 200/150/75/150/150 as the baseline pipeline) on the
  benchmark dataset. Candidate assignments follow gates: SOURCE PIX / LIGHT →
  initialized Prodigy where Gate B passed; MASS → Gate-C posterior engine;
  positions per Phase 4. For each transition record: parameters carried/new,
  initializer construction (start points + metric, priors untouched), whether
  new modes can appear (SMC bridge candidate per Gate D), whether a global
  stage is still required. Compare end-to-end wall + per-stage reliability vs
  the all-Nautilus baseline. Chaining mechanics note: current prior-passing
  (`model_centred_*`) *replaces* priors — the initialized engines must use
  the start-point/metric route to honor the no-narrowing rule; where SLaM's
  own prior passing is the production behaviour, record that distinction
  explicitly rather than silently mixing the two.
- **Hardware / cost:** **L**.
- **Depends:** Phases 4–7, 11.
- **Outputs:** Stage-by-stage recommendation table + transition ledger — the
  backbone of the eventual user decision tree.

### Phase 13 — Mass-model dimensional scaling

- **Question:** Where is the crossover at which nested sampling stops being
  the best practical choice, and which initialized gradient method takes over
  without losing reliability?
- **Design:** Ladder from PowerLaw+shear upward: +m=1, +m=3, +m=4 multipoles
  (each +2 free via prior-pairing to the base profile; m=1 is undocumented in
  the workspace — validate it first, including the dipole-vs-centroid
  degeneracy question), on MGE and mesh source configs; record N at each
  rung. Engines: gate survivors (Nautilus / NSS / initialized Prodigy /
  Gate-C sampler / SMC where Gate D). ≥5 seeds at each decision rung. JAX
  PowerLaw is a 20-term series vs numpy hyp2f1 — backend parity is a
  hazard-ledger row before the ladder starts. Measure scaling vs N:
  likelihood + gradient cost, wall, memory, effective samples, reliability,
  multimodality incidence. No scaling-law fits from ≤3 points — report the
  measured curve.
- **Hardware / cost:** **L → XL**.
- **Depends:** Phases 11–12.
- **Outputs:** The crossover measurement + the evidence base for Follow-up
  1's N>30 programme (multi-band offsets, group-scale — deliberately
  undesigned now, per the brief).

### Future follow-ups (deliberately undesigned)

1. N>30 / N~100+ (multi-band offsets, group-scale through SLaM) — design from
   Phase 13 evidence.
2. `autogalaxy_profiling` — analogous infrastructure; no SLaM transplant;
   staged morphology problems instead.
3. Graphical / hierarchical / EP scaling — from
   `autolens_workspace/scripts/guides/modeling/advanced/{graphical,hierarchical}.py`.

### Phase 14 — Default CPU mesh decision (new-user hazard) [ADJUDICATED + SHIPPED 2026-08-21]

- **Decision (2026-08-21, human-adjudicated):** option (2), in the
  "resurrect the empirical rank-CDF transform" form. The rectangular
  adaptive mesh family is split (PyAutoArray#461/#462, merged):
  `RectangularBilinearAdaptDensity` / `RectangularBilinearAdaptImage`
  (empirical rank-CDF — sort + cumsum, O(N log N), no hyperparameters)
  are the workspace default **everywhere, including interferometer**
  (autolens_workspace#495, autogalaxy_workspace#221, merged; no normal
  workspace uses RTU); `RectangularRTUAdaptDensity` /
  `RectangularRTUAdaptImage` (kernel-CDF, Enzi et al. arXiv:2606.30620)
  are the documented advanced option — required for gradient-based (JAX)
  samplers at os_pix=1 and on the interferometer sparse path, recommended
  on GPU. The interpolated-kernel-CDF forward (K=8192 -> dlnL <= +4e-3,
  18-55x on the step) was deliberately NOT used for the default (still a
  bandwidth hyperparameter); it remains #153's lever for speeding up RTU
  itself, if still wanted now that the CPU default is rank-CDF.
- **Both meshes are explicit in this repo's cells** (PR #155, merged):
  `--rect-mesh {bilinear,rtu}` on the rectangular likelihood_runtime /
  likelihood_breakdown / parallel_scaling cells via
  `_profile_cli.rect_mesh_classes`, with `rect_mesh` embedded in result
  JSONs and `_rtu`-suffixed result files; `--rect-mesh rtu` reproduces the
  pre-split kernel-CDF behaviour, so pre-split recorded numbers stay
  comparable. The sampler benchmark surfaces (misc/searches `_setup.py`,
  nautilus / multi_start_prodigy cells) stay pinned to RTU — gradient
  searches need the kernel-CDF mesh and the recorded truth bars remain
  valid.
- **Outstanding:** the versioned Bilinear-vs-RTU CPU measurement in the
  `pixelization_numba` cells (one run each of `--rect-mesh bilinear` /
  `--rect-mesh rtu`), recording the erf-sum elimination (kernel-CDF was
  55% of a euclid eval, 89% at hst, post-#458).
- **Original question / options / evidence:** autolens_profiling#153 (the
  intake, with all measurements and the 2026-08-21 adjudication comment);
  options were (1) Delaunay first-class, (2) simpler/faster Rectangular
  default, (3) backend-dependent defaults.
- **Depends:** resolved before Phase 11's frozen recommended configuration,
  as required.

## 5 · Benchmark & result schema (v2)

Extends the existing `results/searches/` JSON (which already carries
config/device/version/performance blocks) rather than replacing it. Additions
in v2:

```json
{
  "schema_version": 2,
  "target": {
    "target_id": "sha256:…12",
    "cell": "imaging/mge/hst",
    "model_dim": 15,
    "priors_ref": "targets.py@<sha>",
    "likelihood": {"log_det_method": "cholesky", "positive_only": true,
                    "curvature_floor": 1e-3, "precision": "fp64"},
    "positions": {"enabled": true, "source": "simulator_truth",
                   "threshold": 0.3, "penalty_factor": 1e8},
    "target_class_vs_v1": null
  },
  "algorithm": {
    "name": "nss", "config_id": "nlive500_del50_inner30",
    "settings": {"...": "every knob"}, "seed": 3,
    "initialization": {"kind": "broad|start_points|posterior",
                        "from_result": "<target_id/run>|null"}
  },
  "diagnostics": {
    "stop_reason": "converged", "steps_after_best": 412,
    "counters": {"value_nan": 0, "grad_nan": 0, "clipped": 0,
                  "resurrections": 0, "constrained": 0},
    "histories": "search_internal ref",
    "per_lane": [{"best_fom": 0, "best_x": [], "final_x": [],
                   "basin": "truth|other|dead"}],
    "mcmc": {"divergences": 0, "ess_bulk": 0, "ess_tail": 0, "rhat_max": 0,
              "tree_depth_hist": []},
    "ns": {"logZ": 0, "logZ_err_ensemble": 0, "n_like": 0}
  },
  "verdict": {"success": true, "delta_max_ll_vs_truth": 0.9,
               "posterior_agreement": {}, "notes": "…"},
  "hardware": {"tier": "hpc_a100_fp64", "peak_vram_gb": 0,
                "compile_s": 0, "cache_state": "warm", "shas": {}}
}
```

Field notes (from the approved plan): `target_id` hashes everything in the
`target` block; `priors_ref` is exact prior-spec provenance;
`target_class_vs_v1` is 1|2|3 when a target descends from another;
`diagnostics` blocks are per-method and nullable; `histories` references the
alive/fom/acceptance/λ curves in `search_internal`.

Plus two registries: `scripts/misc/searches/_targets.py` (builds targets from
named specs, computes target_id, owns tolerances) and per-method
**configuration records** (§6 method cards). Reference posteriors live as
named baselines under `results/baselines/InferenceRefs_v1/`, tagged with the
target_id they certify.

## 6 · Knowledge structure → future user guides

```
autolens_profiling/results/notes/inference/
  PROGRAMME.md            # this plan, maintained; phases + gate states
  DECISIONS.md            # append-only gate log: date, evidence refs, verdict
  LITERATURE.md           # arXiv/URL anchors per method + the lesson each earned
  methods/<method>.md     # living method cards (template below)
  phase_<NN>_<slug>/RESULTS.md  # per-phase write-ups (clipper_campaign style)
```

Method-card template — every field in the brief's two lists, grouped so the
eventual `autolens_workspace/scripts/guides/searches/` decision tree can be
*generated* from cards + DECISIONS.md without re-deriving reasoning:

```
# <Method> — evidence card
IDENTITY     global-or-local · gradient-required · JAX/GPU support · handles-multimodality
EVIDENCE     dimensional regimes tested · datasets/models · initialization modes tested
STRENGTHS / WEAKNESSES        (each line cites a result JSON or phase RESULTS.md)
CONFIGURATION  settings tested · recommended-by-regime · auto-adapted · knobs users
               should not touch · sensitivity notes
TERMINATION  rule · statistical meaning · multimodal reliability · required diagnostics
             · evidence termination is safe · waste (steps-after-best)
HAZARDS      prior-bound behaviour · smoothness sensitivity · failure modes ·
             diagnostics users should watch
PERFORMANCE  per-tier numbers (never cross-tier) · compile split · memory
RECOMMENDED  SLaM phases · CONFIDENCE level (anecdote / seeded / gated)
REFERENCES   literature + internal
```

Hazard findings continue in the existing `results/hazards/` ledger (stable
IDs, behaviour-verified). Nothing user-facing is written during the programme;
the guides are a post-Gate distillation task.

## 7 · Likely source-library changes (separated from profiling)

| Repo | Change | Trigger / gate | Class |
|---|---|---|---|
| PyAutoLens | PositionsLH accumulation fix (+ regression test) | Phase 0 — unconditional | Bug fix; target impact measured — **shipped, PyAutoLens#699/PR#700** |
| PyAutoFit | blackjax floor ≥1.6.2; `af.NSS` re-mainlined on `blackjax.nss` | ~~Gate A~~ **shipped early, human-directed 2026-08-18** (PyAutoFit#1492; DECISIONS.md) — Gate A still decides baseline adoption | Integration |
| PyAutoFit | Per-lane *best* position/FOM preservation in MultiStartGradient | Phase 3 pre-req | Diagnostics |
| PyAutoFit | Multi-chain BlackJAXNUTS + mass-matrix/start-point injection; warm-start abstraction (Result → start points + metric, priors untouched); ChEES/MAMS searches if Gate C selects them | Phase 6 / Gate C | Feature |
| PyAutoFit | Per-parameter bijector slot for multi-start searches (log-coordinate stepping) | 8B evidence | Category 1 |
| PyAutoFit | SMC search + informed-start bridge abstraction | Gate D only | Feature |
| PyAutoFit | Search seed reproducibility completion (existing draft) | Phase 1 | Hygiene |
| PyAutoArray | ConstantZeroth repair (alternative scheme) | 8C | Cat 3 alternative, opt-in |
| PyAutoArray | Analytic fixed-topology log-det (rectangular meshes) | 8+ probe wins | Category 1 |
| PyAutoArray | sqrt(dual_area) gradient guard (Delaunay) | Phase 5 attribution | Category 1/2 — must not change values |
| PyAutoArray | Finite-μ / smoothed-positivity forward option (opt-in) | Gates E+F only | Category 3 — research-gated |
| PyAutoArray | Curvature-floor scale-aware default | Phase 10 surprise only | Category 2 at best |

All follow the standard workflow: PyAutoMind prompt → start_dev → plan
approval → ship_library; hypothesis/reproducer/benchmark/invariance-check/
decision stay in autolens_profiling regardless of where code lands.

## 8 · Risk register

| Risk | Where it bites | Mitigation baked into the plan |
|---|---|---|
| Scientific-target drift (comparisons across unequal targets) | All phases | target_id hashing; cross-target claims must name the difference as the experiment; Phase 11 freeze before scaling. |
| PositionsLH 2× / last-entry-only defect | Phases 4, 5, 12 | Phase 0(a) verify+fix before any positions run; before/after record. **Fixed: PyAutoLens#699/PR#700.** |
| Positions double-counting (positions from the fitted image) | Phases 4, 12; science outputs | H4.3 direct measurement (active-penalty posterior fraction; on/off posterior comparison); recommendation records the science rationale either way. |
| Penalty-induced gradient pathologies (hinge, argmax kinks, 1e8 cliff, zero-gradient interior) | Phases 4–5 | Stage-1 transect characterization + factor/threshold sensitivity arms before campaigns; hazard-ledger record. |
| Seed dependence / lucky-seed conclusions | All reliability claims | ≥5 seeds at decision points; p_hit framing with CIs; PyAutoFit seed plumbing (shipped) + reproducibility draft. |
| Local maxima / mode loss in "informed" methods (warm NUTS, bridged SMC, warm NS) | Phases 6, 7, 12 | Nested-sampling cross-checks at transitions; deliberate mode-dropped-start test in Phase 7; R̂ multi-start chains from dispersed inits. |
| NSS logZ bias from under-mixed inner kernel | Phase 2; any NS evidence claim | Pre-registered inner-steps scan; ensemble logZ errors; Nautilus cross-reference. |
| NaNs: value vs gradient axes conflated | Phases 3, 5, 8, 9 | Disjoint shipped counters + per-axis attribution (mesh sqrt vs reg λ⁴); alive *curves*, never integrals. |
| Hard bounds / clipping semantics | Phases 3–5 | Clipper on as hygiene (evidence: zero accuracy cost); pinned-coordinate reporting; no further wall-handling arms without a named mechanism. |
| Plateau detector false convergence (wrong basin reads as converged) | Phases 3–6, 12 | Convergence confusion matrix is a first-class Phase-3 metric; detector never trusted on resurrecting populations (already disabled there). |
| Active-set kinks: toy-scale evidence over-generalized | Phase 9 | Production-scale confirmation gate (Gate E) before any smoothing work. |
| Regularization conditioning (λ⁴ wall, 1e-8 lift, CPU-vs-GPU PD coin-flip) | Phases 5, 8 | slogdet profile for gradient work (8A); log-coordinate (8B); default value path untouched. |
| SMC particle impoverishment / silent λ=1 garbage | Phase 7 | Judge by acceptance traces + ESS schedule, never "Converged: yes"; independent-run logZ variance. |
| NUTS divergences / lockstep waste on GPU | Phase 6 | Divergence localization recorded; tree-depth histograms; ChEES/MAMS arms. |
| GPU memory (58 GiB jvp fusions; mesh VRAM) | Phases 5–6 | batch_size sized from single-eval memory profile; plain-Delaunay batch fixed at 2 (basin-steering hazard); prealloc capped on laptop. |
| Compile artifacts mistaken for algorithmic results | All JAX phases | Persistent cache + autotune-off defaults; cold/warm split in schema; compile timing only on idle nodes; Delaunay cache-miss noted per-cell. |
| Stale benchmark / cross-hardware rows | Longitudinal claims | Tier recorded per row; cross-tier tables structurally disallowed in the dashboard; version strings per PyAutoLens release. |
| Environment artifacts (RAL fp32 default; local jax_grad false failures; PYTHONPATH) | All remote runs | Mandatory `JAX_ENABLE_X64` in sbatch templates + post-launch grep; known-good control script before believing local failures. |
| Knowledge stranded on RAL / in branches | Programme memory | Phase 0(c) harvest; parked-branch findings docs mirrored into the inference ledger; no results live only on NFS. |

## 9 · Minimal critical path

Maximum information before large A100 commitment — strictly ordered:

1. **CP-1 · PositionsLH defect verify + fix** (hours, laptop). Unblocks the
   entire positions arc; every later positions result depends on it.
   **✅ COMPLETE — PyAutoLens#699 / PR#700 (see DECISIONS.md; existing truth
   bars unaffected — no searches-framework benchmark used positions).**
2. **CP-2 · blackjax 1.6.2 upgrade + mainline NSS smoke on MGE** (~1 day,
   laptop GPU). Establishes whether Phase 2 is a tuning exercise or an
   integration project.
   **✅ Environment half COMPLETE — all three environments (cloud, local
   venv, RAL) on 1.6.2 with the full smoke (incl. `af.NSS` end-to-end)
   passing; `af.NSS` re-mainlined (PyAutoFit PR#1492, merged 2026-08-18).
   Remaining: the laptop-GPU MGE smoke half — the first Phase 2 work item.**
3. **CP-3 · Prodigy MGE reliability scan ± PositionsLH** (RAL CPU + laptop
   GPU; the single highest-information experiment). n_starts {16, 64, 256} ×
   5 seeds × positions {off, on} → measures p_hit, tests H3.1/H3.2/H4.1
   simultaneously, and effectively decides Gate B's shape before any A100
   time. If positions-on at 64 starts is not reliable on *MGE*, the mesh
   campaign (Phase 5) shrinks to nested-sampling + mesh-family ranking only.
4. **CP-4 · slogdet A/B on the AdaptSplit NaN wall** (hours). Expected pass;
   converts Phase 8A from plan to record and sets the gradient-work
   likelihood profile early.
5. **CP-5 · NSS inner-steps/logZ-bias scan + one pixelized probe**
   (laptop/RAL + 1 short A100 slot). Decides Gate A and whether NSS's 7–11×
   pixelized deficit is configuration or structure.

First full A100 block only after CP-1..5: Phase 5 mesh campaign + Phase 6
warm-start MGE confirmation, with arms already pruned by the cheap results.

## 10 · Expected final decision tree — HYPOTHETICAL

Written now so the programme has a falsifiable shape; every branch is
provisional until the phase evidence exists. Confidence tags: [E] existing
evidence · [H] hypothesis.

```
~10–20D parametric (MGE), broad priors
├─ CPU only        → Nautilus (n_live per SLaM table)               [E]
├─ GPU available   → NSS (tuned inner steps) or Nautilus-vmap       [H: Gate A]
└─ positions known → + PositionsLH during search                    [H: Gate B]
                     Prodigy(n≥64)+positions as fast MAP *if* Gate B passes;
                     otherwise nested sampling stays the global engine.

Pixelized source, any stage
├─ positions REQUIRED (demagnified-source maxima)                   [E]
├─ global search   → nested sampling (Nautilus; NSS if Gate A-mesh) [E]
├─ gradient work   → DelaunayNN / kernel-CDF meshes only; plain
│                    Delaunay is a gradient hazard (flips, sqrt-NaN,
│                    batch steering)                                 [E]
└─ likelihood profile → slogdet-on for differentiation              [H: 8A]

Good previous fit exists (any model, right basin trusted)
├─ posterior needed → warm NUTS (dense/low-rank metric)             [H: Gate C]
│                     └─ many-chain GPU → ChEES/MAMS variant        [H]
├─ point estimate  → initialized Prodigy (auto-converge, ceiling)   [E-partial]
└─ basin NOT trusted / new params may add modes
                   → bridged SMC from previous posterior            [H: Gate D]
                     (falls back to fresh nested sampling)          [E]

Dimensionality growing (multipoles, multi-band, groups)
├─ N ≲ 20–30       → nested sampling remains competitive            [E]
├─ crossover point → measured in Phase 13, not assumed              [—]
└─ N ≳ 50 w/ good initializer → warm gradient sampler family
                     (NUTS/MAMS/SMC-rejuvenated)                    [H]

Diagnostics gate every arrow: NaN/clip counters clean, stop_reason
"converged" with post-hoc check, R̂<1.01 + ESS floors + no divergence
clusters, logZ cross-checked when evidence matters.
```

---

Prepared as a planning pass 2026-08-17; approved same day (see
`DECISIONS.md`). Sources: autolens_profiling searches framework + results
ledger + hazards index; PyAutoFit@d6fef747; PyAutoArray@394514c0;
PyAutoLens/PyAutoGalaxy/autolens_workspace source; PyAutoMind ledger +
PyAutoBrain samplers faculty + PyAutoMemory methods wiki; blackjax 1.6
release line + PR #947; external literature per `LITERATURE.md`.
