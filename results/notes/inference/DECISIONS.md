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

## 2026-08-18 — Scope change: af.NSS re-mainlined ahead of Gate A (human-directed)

**Decision:** The human directed early re-mainlining of `af.NSS` into
PyAutoFit as a proper first-class search, ahead of Gate A. Rationale: the
search's removal (PyAutoFit#1356/#1357) was driven solely by unshippable
git-fork dependency pins, and nested sampling has since merged into
**mainline blackjax 1.6** — the removal reason no longer exists. **Gate A is
unchanged in meaning:** it still decides whether NSS becomes the principal
GPU nested-sampling *baseline*; this decision only restores the search's
availability.

**Record:** PyAutoFit#1491 (issue) / **PR#1492** (implementation, pending
human merge). Port pre-validated on CPU in the cloud session before any
source change (see `phase_00_unblocking/RESULTS.md` for the environment
half): the `autofit_workspace_developer/searches/nss/` stash restored with
the mainline port — `blackjax.ns.utils` imports, `SliceInfo.num_steps` →
`num_expansions`, chunked builder recomposed from mainline public helpers
(no `update_strategy=` kwarg in mainline). 17/17 unit tests pass; 2D
analytic toy through `search.fit` gives logZ −4.558 ± 0.078 vs analytic
−4.605; the chunked GPU-memory path is bit-identical to unchunked at fixed
seed. Optional-extra blackjax floor bumped ≥1.2.0 → ≥1.6.2 (the §7 floor
item, also pulled forward).

**Consequences:** Phase 2 no longer needs a profiling-local NSS runner
wired by hand — it can drive `af.NSS` directly once PR#1492 merges (the
plan's "profiling-local runner first" hedge is obsolete). Follow-ups:
workspace tutorial restoration (stash checklist item 5) and the Phase 2
re-benchmarking campaign itself (GPU, laptop/RAL).

---

## 2026-08-19 — Phase 0 gate satisfied: (a) + (b) both landed; Phases 1–2 unblocked

**Decision:** Phase 0's gate condition — "nothing else proceeds until (a)
and (b) land" — is **satisfied**. Phase 0(b) is complete in all three
environments; the programme's downstream phases (Phase 1 targets registry,
Phase 2 NSS re-tuning) are unblocked.

**Evidence:**

- PyAutoFit **PR#1492 merged 2026-08-18** — `af.NSS` is a first-class search
  on main (blackjax optional-extra floor ≥1.6.2).
- **Local venv**: blackjax 1.5 → 1.6.2; `nss_smoke.py` (extended with an
  `af.NSS` end-to-end analytic-evidence check, the surface Phase 2 drives)
  PASS on all three checks; `af.NSS` unit suite 17/17 against the local
  checkout.
- **RAL stack**: was still the obsolete 2026-01 fork build
  (0.1.0b1.dev86+g795058671, §2.2's stale pin) — upgraded to 1.6.2 with a
  pre-upgrade freeze snapshot; library mains re-synced; the same smoke PASS
  on all three checks (login node, CPU). Seeded results identical to local.
- Record: `phase_00_unblocking/RESULTS.md` (2026-08-19 sections), issue #146.

**Still open in Phase 0:** (c) is **partial** — the stranded *local*
fork-NSS A100 artifacts (the §1.2 delaunay / pixelization / mge-2nd-run
rows) are now committed, but the RAL NFS harvest (SMC warm arms job 331058,
NaN-counter split arms 335003-5, Nautilus pixgrad baseline logs) remains.
(c) was never part of the gate condition and does not block Phases 1–2.

**Next per the critical path (§9):** the laptop-GPU MGE half of CP-2 (first
Phase 2 work item), then CP-3 (Prodigy MGE reliability scan ± PositionsLH —
needs the PyAutoFit per-lane-best change first) and CP-4 (slogdet A/B, can
start any time).

---

## 2026-08-20 — Mind prompt absorbed: ell_comps trapping folds into Phase 3/CP-3

**Decision (human-directed, start_dev routing):** the Mind research prompt
`draft/research/autolens_profiling/ell_comps_trapping_unmasked.md` (filed
2026-08-16 as follow-up (2) of the mge-lane-death task, #128) is absorbed
into the programme rather than run as a standalone campaign. Its questions
1–3 (size under clipping; does trapping cost the answer; corner-vs-annulus
localisation) become Phase 3 metrics (**[H3.3]**) carried on every CP-3 arm
and repeated across the Phase 4 positions on/off axis per engine; its
question 4 (fix space) stays gated behind that evidence, preserving the
prompt's own ordering.

**Rationale:** CP-3 runs exactly the arms the prompt needs (clip=prior_box,
n_starts scan, ≥5 seeds, per-lane-best pre-req) at better statistics than a
standalone 2-seed rerun would buy. The clipper phase-2 campaign already
gives a preliminary clipped-run reading — 6/16 and 4/16 lanes end pinned
(fp64, 2 seeds) while seed 0 still beat the Nautilus bar — but that is
positions-off, 2-seed, and dominated by the θ_E=0 basin, so it counts as
evidence the trapping is live and large, not as the measurement. The human
directive on absorption: prior results had no PositionsLH and known
local-maxima issues, so trapping must be **re-measured under current
setups** (positions on/off, per engine/config), never closed on
pre-positions evidence.

**Prompt disposition:** retired to Mind `complete/archive/shelved/` with an
absorption pointer to this entry.

---

## 2026-08-21 — Phase 14 adjudicated + shipped: Bilinear (rank-CDF) default, RTU (kernel-CDF) advanced; both explicit in the cells (human-directed)

**Decision:** the Phase 14 default-CPU-mesh question is resolved as option (2)
in the "resurrect the empirical rank-CDF transform" form, and the whole slice
shipped same-day (human-adjudicated and human-merged).

- PyAutoArray#462 (merged): rectangular adaptive meshes split into
  `RectangularBilinearAdaptDensity` / `RectangularBilinearAdaptImage`
  (empirical rank-CDF — sort + cumsum, O(N log N), no hyperparameters;
  recovered from the pre-#402 implementation at `22b28463^`) and
  `RectangularRTUAdaptDensity` / `RectangularRTUAdaptImage` (pure renames of
  the kernel-CDF classes — likelihood values unchanged).
- Workspace defaults (autolens_workspace#495, autogalaxy_workspace#221,
  merged): Bilinear everywhere, **including interferometer** — per the
  follow-up human directive, no normal-workspace example uses RTU; RTU is
  documentation-only (required for gradient-based JAX fitting at os_pix=1
  and on the interferometer sparse path where Bilinear's likelihood is
  exactly piecewise-constant; recommended on GPU; Enzi et al.
  arXiv:2606.30620 cited).
- Downstream configs/docs: PyAutoGalaxy#579, PyAutoLens#707 (merged).
  Likelihood pins: autolens_workspace_test#259 (merged) — Bilinear pins
  regenerated under JAX x64, RTU pin scripts kept as pure renames.
- This repo (PR #155, merged): both meshes explicit via
  `--rect-mesh {bilinear,rtu}` + `_profile_cli.rect_mesh_classes`;
  `rect_mesh` recorded in result JSONs; `_rtu` filename suffix keeps the
  families' results disjoint; sampler benchmark surfaces stay pinned to RTU
  so the recorded truth bars remain valid.

**Why:** the kernel-CDF transform is 55% (euclid) to 89% (hst) of the numba
CPU likelihood even after the #458 windowed kernel — an O(M_sub x N_data)
erf sum. The rank CDF eliminates it at the cost of gradient smoothness
(certified July audit: piecewise-constant at os_pix=1; FD-validated at
os_pix=4). The interpolated-kernel-CDF forward was rejected for the default
(still a bandwidth hyperparameter) and remains #153's lever for RTU itself.

**Still open (Phase 14 tail):** the versioned Bilinear-vs-RTU CPU
measurement in the `pixelization_numba` cells (`--rect-mesh bilinear` /
`--rect-mesh rtu`, one run each). Tracked on #153.

**Records:** PROGRAMME.md §Phase 14 + status table (this commit);
adjudication comment on #153; Mind task rectangular-bilinear-rtu-mesh-split
(PyAutoArray#461).

---

## 2026-08-23 — GATE B part 1 CALLED (human-ratified): MultiStartProdigy(n=256) is a reliable global MAP searcher on MGE, positions-off

**Decision (human, 2026-08-23, on the post-adversarial-review narrowed
reading; PR #157 merged same day):** Gate B part 1's failure condition
("no n_starts ≤ 256 gives ≥99% reliability at cost below the
nested-sampling budget") is **NOT met**. MultiStartProdigy — n_starts=256,
clip=prior_box, no scaler, auto-convergence on, positions OFF — is
ratified as a global MAP searcher for the MGE-class parametric cell.

**Evidence (CP-3 wave 1 + adversarial review, both in
`phase_03_prodigy_reliability/`):** p̂_hit = 0.048 CP95 [0.037, 0.061]
(lower bound; budget/detector-conditional; n256 tier = ~1,280 distinct
draws). Demonstrated reliability ≥99.0% at the joint-95% worst case over
p-uncertainty and the largest within-run lane correlation the data cannot
exclude (ρ ≤ 0.0057); ≥99.99% under independence. 5/5 seeds succeeded at
n=256 in 172–225 s vs the recorded Nautilus A100 sampler wall of 772.7 s
(3.4–4.5×; second recorded baseline 523 s unreconciled — re-baseline
running). Zero parameter-recovery impostors across all 80 hit lanes
(θ_E spread 0.014%).

**Scope limits carried with the call (RESULTS.md caveats (a)–(h)):**
demonstrated at n=256 ONLY (no smaller n reaches 99% at 95% confidence;
n128 fresh-seed tier queued); single cell (imaging/mge/hst), single tier
(A100 fp64), positions-off; MAP-only — nested sampling retains posterior
+ evidence duty; p̂ conditional on the stop rule; compile not split;
`target_id`/Phase-1 tolerance infrastructure still missing;
`likelihood_evals` field wrong for MultiStart artifacts.

**Gate B part 2 (PositionsLH) remains open** — Phase 4 plumbing not
built. The θ_E~U(0.2,8) diagnostic was uninformative for H3.1 (original
tripling claim withdrawn on adversarial review — see
`ADVERSARIAL_REVIEW.md`).

**Records:** phase_03 RESULTS.md + ADVERSARIAL_REVIEW.md (PR #157);
PROGRAMME.md phase/gate table (this commit); overnight strengthening
jobs 339065-339073 (n128 fresh seeds, n256 extra seeds, Nautilus
re-baselines) submitted before the call and noted here for the record.

---

## 2026-08-24 — Phase 2 scan COMPLETE: H2.1 closed, NSS operating point recorded, GATE A CALLED (human): Nautilus stays the nested baseline

**Record (not a gate call):** Phase 2 wave 2 harvested (RAL 338870-338873,
339067[0-3], 339068, 339069, 339070/71). Mainline `blackjax.nss` (af.NSS)
reproduces the Nautilus answer on both `imaging/mge/hst` and
`imaging/delaunay/hst`; the fork-era +7–13-nat logZ bias is entirely
inner-kernel under-mixing (inner=30: 5/5 seeds within +1.0 ± 0.4 nats of
Nautilus 31690.50). Operating point n_live 200 / num_delete 100 / inner 30
/ dlogz −3 (−10 when the MAP matters). Cost at that point: **5.0× the
Nautilus sampler wall on MGE (3,528 vs 707 s), 18.4× on Delaunay (34,726 vs
1,891 s)**, measured same-night against Nautilus re-baselines on the
current stack. Nautilus truth bars reaffirmed to 2 dp; the 523 s Nautilus
wall is retired as a reference (707–773 s reproduces).

**GATE A CALLED (human, 2026-08-24):** Nautilus remains the nested
baseline on every model family; af.NSS stays mainlined as a correct,
tuned alternative, not default. Phase 5's NSS arm is dropped. Sample
economy sealed it beyond wall: Kish ESS 4,121 vs 1,315 per run on MGE
(14× wall for equal ESS; ~15 vs ~940 likelihood evals per effective
sample, reject-inclusive) — `phase_02_nss_mainline/RESULTS.md` "Sample
economy". Only re-opening condition (unmeasured): a GPU-only deployment
where Nautilus's host-side proposal is the bottleneck; W6 (n_batch scan)
queued to bound it. The pixelization-cell Nautilus re-baseline (339795)
is still queued and does not affect the call.

**Same-day human calls, recorded here to keep DECISIONS.md linear:**
- **Gate B pt 1 caveat (a) "n=256 only" STANDS.** n128 fresh-seed 5/5
  (phase_03 addendum) is consistent but cannot demonstrate ≥99 % at 95 %
  confidence; the ~1.3 GPU-h for a 30-seed n128 tier is not worth the
  ~90 s/fit it would save. Revisit only if Prodigy enters a loop where
  seconds compound.
- **Gate C criterion reworded (PROGRAMME §4 Phase 6):** initialized
  gradient MCMC is judged on *batched-pipeline* value (vmap across
  datasets, posterior without an evidence run, ESS per gradient eval
  acceptable), not on beating Nautilus's single-fit ESS/s — Nautilus's
  ~15 evals/ESS on MGE is not a bar warm NUTS is expected to clear.

