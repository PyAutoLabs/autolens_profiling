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

---

## 2026-08-24 — CP-4 / Phase 8A: slogdet FAILS the pre-registered A/B — human call: ADOPT as the profiling-repo testing default, keep library-optional, chase the residual NaNs

**Record (Gates E/F remain open):** slogdet vs cholesky A/B run on RAL
(A100 338808, CPU 338807). The pre-registered stressor (knn + free
AdaptSplit) never walls — VOID on both tiers, as the driver's #117 note
predicted. On the real wall (Delaunay + free AdaptSplit) slogdet rescues
64–73 % of Cholesky NaNs with zero regressions but leaves 20–32 draws NaN
under both arms, leaves all λ-transect gradients non-finite, disagrees
with Cholesky by up to 9,619 nats (A100) in the marginal band, and costs
3.7× on CPU. Three of four criteria fail.

**Human call (2026-08-24), overriding the record's "not recommended":**
zero regressions + 64–73 % rescue is worth having now. (1) `slogdet`
becomes the **default `log_det_method` for gradient-work cells in this
repo's searches framework** (`_setup.py`; GPU tiers — CPU keeps cholesky
for the 3.7× cost) — W8. (2) The PyAutoArray default stays cholesky
(opt-in), **to be revisited once (3) lands — reminder owed to the human at
the end of the queue: "make slogdet standard".** (3) The NaN-under-both
draws (32 A100 / 20 CPU) and the all-transect non-finite gradients are a
named investigation, W7 — are they genuinely singular systems (λ⁴, #104)
or a formula-independent overflow that a different scaling removes? Phase
8B stays the live reparameterization candidate alongside. Record:
`phase_08_regularization/RESULTS.md`.


---

## 2026-08-24 — W4 / Phase 1: targets registry, schema v2, `slam_source_pix(_nn)`, reference baselines (autolens_profiling#161)

**Record (Phase 1 IN PROGRESS, not a gate call):** landed the Phase 1
targets registry infrastructure PROGRAMME.md §"Phase 1" and §5 specify —
`scripts/misc/searches/_targets.py` (`Target`/`Tolerances` dataclasses,
`TARGETS` — 32 entries: {mge, delaunay, delaunay_nn, knn, delaunay_matern,
pixelization, slam_source_pix, slam_source_pix_nn} x {positions off/on} x
{fp64/mp} — and `target_id`/`target_block` canonical-identity hashing),
schema-v2 additions to every `results/searches/` JSON
(`schema_version`/`target`/`algorithm`/`hardware`, added beside every v1
key, never replacing one — `build_readme.py`'s dashboard renders both
unchanged), the imaging truth-anchor extension (previously point_source /
cluster only), and the MultiStart `likelihood_evals` correction (was
`total_samples`, a small posterior-storage count; now
`total_steps * n_starts`, the actual reject-inclusive evaluation count) +
Kish ESS.

**Human mesh decisions (this commit implements, does not re-litigate):**
- `slam_source_pix` = `al.mesh.RectangularRTUAdaptImage` (best gradient
  behaviour of the rectangular family) + free-coefficient `al.reg.Adapt`
  (inner/outer coefficient + signal_scale). **Deliberately differs** from
  the workspace SLaM `source_pix[1]` fiducial
  (`autolens_workspace/scripts/multi_galaxy/slam.py:653`, which pairs
  `al.mesh.RectangularBilinearAdaptImage` — not RTU — with the same
  `al.reg.Adapt`, and a 28x28 mesh vs this repo's 39x39
  `_PIXELIZATION_MESH_SHAPE` fiducial). The RTU/Bilinear choice mirrors the
  Phase 14 default-CPU-mesh adjudication (2026-08-21): RTU has the better
  measured gradient surface, which is what this profiling repo's targets
  exist to stress; the workspace's own default optimizes for a different
  axis (new-user CPU speed) and is not overridden by this decision.
- `delaunay_nn` is registered as a REAL target (scientifically the premier
  model — Sibson/natural-neighbour interpolation vs the Delaunay mesh's C0
  barycentric one), not a diagnostic cell: `al.mesh.DelaunayNN` (a
  `Delaunay` subclass, identical `(pixels, zeroed_pixels, areas_factor)`
  constructor) + the SAME `ConstantSplit` regularization `delaunay` uses,
  so the two targets are a pure mesh-family A/B. `slam_source_pix_nn` pairs
  DelaunayNN with the same free `reg.Adapt` as `slam_source_pix`, isolating
  the mesh choice the same way.

**Verification finding (not a target-definition bug):** at broad, untuned
prior draws (CPU, `use_jax=False`, 8 draws in the unit-cube's [0.2, 0.8]
band), `delaunay` resamples 0/8 times; `delaunay_nn` resamples 5/8
(3 finite, 3 NaN, 2 `FitException`); `slam_source_pix_nn` resamples 7/8
(1 finite, 2 NaN, 5 `FitException`). Both DelaunayNN-based targets are
registered as specified — an elevated resample rate at broad priors is
itself a legitimate Phase-1 finding about the mesh, not something this
registry silently works around by swapping mesh/regularization. Recorded
on the affected `Target.notes` and worth a named follow-up once Phase 5+
starts running real searches against these targets.

**Reference baselines:** adopted
`results/searches/nautilus/imaging/{mge,delaunay}/hst/hpc_hpc_a100_fp64.json`
(v2026.8.17.1, same-stack re-baselines already used as the programme's
Nautilus truth bars — DECISIONS.md 2026-08-24 Gate A entry) as
`certified_by: "retro"` `InferenceRefs_v1` baselines, each tagged with the
`git sha` current at adoption (`b9c47062f2a46a211ca0df92cbce7e9edd2a3c4c`).
The `pixelization` target's existing row
(`.../pixelization/hst/hpc_a100_fp64.json`, v2026.5.21.1) is explicitly
**NOT adopted** — it predates the version-gap refresh the other two rows
already got. 11 further reference rows (fresh Nautilus fp64 runs, seed 0,
`n_live >= 2x` fiducial via the new `SEARCHES_NAUTILUS_N_LIVE` override) are
queued in `results/baselines/InferenceRefs_v1/SUBMIT_LIST.md`, with a
prepared SLURM array
(`hpc/batch_gpu/submit_search_nautilus_inference_refs_v1_array.sh`) that has
**NOT been submitted** — writing the array is this commit's job; running an
~11-task multi-hour A100 campaign is a separate decision.

**Open questions carried forward:** (1) the DelaunayNN resample-rate finding
above — worth a dedicated investigation once the reference-row campaign
gives a real sample size to characterise it against. (2) Whether
`slam_source_pix`'s deliberate RTU/Bilinear deviation from the workspace
SLaM default should ever be reconciled (a future workspace-docs decision,
out of scope here). (3) The 11-row SUBMIT_LIST campaign itself — queued,
not run.

**Records:** `scripts/misc/searches/_targets.py`,
`results/baselines/InferenceRefs_v1/`, PROGRAMME.md Phase 1 section + W4
table row (this commit).

---

---

## 2026-08-24 — W7 (CP-4 follow-up, #164): per-draw attribution + CP-4 re-scored on the clean subset — verdict unchanged

**Record:** `scripts/misc/searches/slogdet_nan_attribution.py` replayed
individual stored `delaunay_adapt_split` draws (both tiers) non-jitted,
drilling into `inversion.*` matrices and small `jax.grad` closures to
assign each sampled draw one mechanism label (170 draws classified across
both tiers: 8 full-gradient, 87 A100 + 83 RAL CPU matrix-only). Two driver
artefacts identified and fixed at harvest (`slogdet_ab.py`): (1) six
`descent`-source draws carried a NEGATIVE physical regularization
coefficient and ten more were dead (entirely-NaN) checkpointed lanes — the
prior "Ops notes" misread this as a log axis; both are now dropped at
harvest. (2) all 128 `lambda_transect` draws sat at the anchor where
`ell_comps`/shear are exactly (0, 0), an undefined-gradient point in
`autogalaxy/convert.py`'s sqrt-magnitude conversion unrelated to the
regularization wall; the anchor is now jittered 1e-3 in unit-cube space off
any exact zero. A tier-attack on draw 96 (coefficients [3.5e5, 536]: finite
under both arms on A100 with a 9,619-nat slogdet/cholesky delta, NaN under
both arms on RAL CPU for the IDENTICAL input vector) traced the
tier-dependence to `cond(curvature_reg_matrix_reduced) = 4.5e18` — far past
float64's ~1e16 precision floor — where the log-det terms remain
individually computable (and agree with the original A100 value to 5
significant figures) while the reconstruction linear solve on the SAME
matrix is itself NaN; different BLAS/LAPACK backends (cuSOLVER vs OpenBLAS)
diverge on whether that solve returns a number. Cross-tier value agreement
on mutually-finite shared draws: slogdet max|Δ| 6.72 nats, cholesky 1.30
nats — both far above the ~1e-4 clean-PD floor, confirming disagreement
concentrates in the marginal band.

**CP-4 re-scored on the clean subset (excluding dead-lane,
invalid-coefficient and anchor-singularity draws):** A100 n=416 → 272 clean
(144 excluded: 10 dead-lane, 6 invalid-coefficient, 128
anchor-singularity); RAL CPU n=384 → 256 clean (128 excluded, all
anchor-singularity — this tier's harvest never ran the descent arm, so it
has no dead-lane/invalid-coefficient rows). On the clean subset, all three
per-draw criteria STILL fail on both tiers — criterion 1 (zero slogdet
NaNs): 22/272 A100, 20/256 CPU still NaN; criterion 2 (value equality):
48/218 A100 and 40/207 CPU mutually-finite comparisons exceed tolerance,
max|Δ| unchanged (9,619 nats A100, 1.62 nats CPU); criterion 3 (finite
gradients): 22/272 A100, 23/256 CPU still non-finite. Criterion 4 (runtime)
is population-level and unchanged (1.03× A100, 3.74× CPU). **Verdict: FAIL
on both tiers, unchanged from the original Phase 8A run.** Matrix-only
population sampling (170 draws) confirms the residual failures are not
driver artefacts: once the two fixed bugs are excluded, the sampled
`nan_both` draws are 53% (A100) / 80% (RAL CPU) `genuinely_singular`
(cond ≥ 1e16 or LAPACK-Cholesky failure) — the genuinely-singular λ⁴
population (`prior`-source draws, regularization coefficient ~4e5-9e5)
that CP-4 was measuring in the first place. A `marginal_tier_flippable`
band (`cond` in `[1e12, 1e16)`) is also a real, sizeable population
(18/87 A100, 18/83 RAL CPU sampled draws, every class except `truth_bar`)
— the conditioning range where a hardware/BLAS change can flip a draw's
NaN verdict even though the matrix is nominally still invertible.

**Human call (2026-08-24), confirming the W8 adoption stands unchanged:**
the re-score does not overturn anything — slogdet remains the GPU
gradient-work default in this repo (W8, already shipped), the library
default stays cholesky (opt-in), and the reminder to revisit the
PyAutoArray default is still owed once W9 lands. W9 (#166) is unblocked
with three concrete inputs from this investigation: (i) quantify whether
slogdet ever returns a wrong-but-finite number where cholesky legitimately
NaNs, using the `cond >= 1e16` threshold and the max|Δ| vs an
eigvalsh-reference logdet (established here: up to 9,619 nats,
tier-dependent, concentrated in the `[1e12, 1e16)` marginal band); (ii) the
library-default recommendation is slogdet on GPU / cholesky on CPU, with
the 3.7× CPU cost restated as the reason CPU keeps the historical default;
(iii) two guards independent of log-det method, both now shipped in
`slogdet_ab.py`: reject non-finite or out-of-prior-bound lane vectors at
descent-harvest time, and never anchor a transect/probe at an exact-zero
ell_comps/shear component. Record: `phase_08_regularization/RESULTS.md`
"W7 addendum" + "CP-4 re-scored" sections; attribution artifacts under
`slogdet_ab/attribution/`.

---

## 2026-08-27 — W6 (#163): `n_batch` scan recorded — 1.78× is per-eval, not wall; the recommended arm's logZ is a ~9σ outlier

**Record (not a gate call):** RAL 339842/339843, 8/8 arms, harvested
2026-08-25; MGE `n_live=200`, Delaunay `n_live=150`, fp64 A100, **one seed
per arm**. Recomputed from the raw JSONs 2026-08-27, the MGE scan from
`n_batch` 64 → 1000 recovers **1.775× per likelihood eval** (10.56 →
5.95 ms), **1.463× on sampler wall** (670 → 458 s) and **1.594× on
ESS/min**; likelihood evals rise 21 % (63,424 → 77,000) and Kish ESS per
eval falls ~10 %. The committed "1.78× free" sentence carried a per-eval
figure into a wall-shaped claim; corrected in `methods/nautilus.md` this
commit. Delaunay saturates by `n_batch=64` at 1.26×, unchanged.

**logZ is not flat at the arm the scan recommends.** logZ spans 0.12 nats
across the whole scan, but the `n_batch=1000` arm sits **−0.10 nat** from
the `n_batch=64` arm — ~9σ of the 0.011-nat five-seed logZ standard
deviation measured on the same cell in Phase 4 Stage 2. With one seed per
arm the scan cannot separate an `n_batch` bias from a seed draw, so no
`n_batch` above the measured baseline is adopted as a default on this
evidence; the finding is "the knob is real on MGE, its evidence cost is
unmeasured", not "free".

**Records:** `methods/nautilus.md` "n_batch SCAN"; PROGRAMME.md §9b W6
row; `results/searches/nautilus/imaging/{mge,delaunay}/hst/hpc_hpc_a100_fp64_nbatch*.json`.

---

## 2026-08-27 — W2 / Phase 4 Stage 2 (#160, PR#174) harvested: corrected reading of both halves

**Record (not a gate call — the Gate B pt 2 call is the separate entry
below):** RAL 340114 (Nautilus, 10 arms) / 340115 (Prodigy n256, 5 arms),
15/15 COMPLETED, harvested 2026-08-26 and merged as PR#174. Independently
recomputed from raw JSON 2026-08-27; two claims in the merged write-up are
corrected here rather than edited away:

1. **Nautilus + PositionsLH is not a strict no-op.** logZ and the recovered
   mode are unchanged to 0.02 nats across 5 seeds — that half holds — but
   max log-likelihood is **lower with positions on in 5/5 seeds** (mean
   −0.126 nats, paired t = −3.45, p ≈ 0.026): the penalty demonstrably
   fires at the maximum-likelihood point. Wall is **−3…+7 %**, not the
   committed "±3 %". No penalty diagnostic exists in the nested JSONs, so
   the penalty's size at the reported maximum cannot be read off the
   artifact.
2. **Prodigy positions-on is 2/5, not 1/5**, scored under Phase 3's own
   coded rule (≥1 lane ≥ 31784.782): seed 1 has a hit lane at 31785.464.
   The 1/5 headline used a stricter, undeclared 0.04-nat band around the
   positions-off plateau. **If that band is kept it must be declared as the
   rule of record**; the direction of the finding stands under either bar.
   The eval-counter caveat (positions-off arms schema v1 at 257 evals,
   positions-on schema v2 at 32k–248k) stands unchanged.

**Records:** `phase_04_positions/RESULTS.md` "Stage 2" (wording corrected
this commit); PROGRAMME.md §9b W2 row + phase/gate table.

---

## 2026-08-27 — W4 / Phase 1 (#161): InferenceRefs_v1 at 9/13 certified; the `slam_source_pix_nn` pilot thrashes

**Record (Phase 1 still IN PROGRESS):** RAL **341879** tasks 7/8 landed the
two rows lost to the missing `_N_LIVE` presets — `imaging/knn` (logZ
30010.170, maxL 30077.028, Kish ESS 4,848.8, 3,343.6 s,
`sha256:84c0d88d3032`) and `imaging/delaunay_matern` (30614.972 /
30676.594 / 5,728.2 / 3,351.7 s, `sha256:3f17a37225f9`), both Nautilus
`n_live=300`, fp64, A100, version stamp 2026.8.17.1. The registry stands at
**9 certified rows of 13 targets**.

**"Certified" means the three provenance checks only** — fresh version
stamp, wall consistent with the cell's recorded cost, no resume marker.
There is no certification function and **no coded tolerance**; `INDEX.json`
still lists only the 2 retro-certified rows; and the mge row ran
`n_live=400`, not the campaign's 300, sitting +265 nats from its truth bar
against a 2-nat registry tolerance.

**Pilot answer: it thrashes.** `slam_source_pix_nn` (free `AdaptSplit` on
DelaunayNN, positions-off) went up alone as **RAL 341908**; it compiled in
20 s and then made **zero Nautilus calls in 6 h** before hitting its
wall-clock limit. Per the 2026-08-24 W4 entry that is a Phase-1 finding to
record, not something to engineer around — the DelaunayNN resample
behaviour costs Nautilus as well as gradient search. The positions-on row
is not submitted.

**Flag on the knn reference row:** its maxL (30077.028) sits **480 nats
below** a same-`target_id` Phase 8B Prodigy `log_reg` arm (30557.03). Until
that is explained the knn reference is not a valid bar for that cell — do
not score anything against it.

**Records:** `targets/REFS_V1_HARVEST.md`; PROGRAMME.md §9b W4 row + phase/
gate table.

---

## 2026-08-27 — W5 / Phase 8B (#162): 340576 lost, the rerun crashed at results-write, six arms recovered offline — NO verdict yet

**Record (Gates E/F remain open; no 8B verdict is called and none may be
scored from today's data):**

**Dispatch history.** The prepared 39-task array went up as **340576** and
lost **35 of 39 arms**, dispatched at ~12 % of the wall budget a
3000-step pixelized arm needs. Reruns **341845** (15 tasks) and **341860**
(14 tasks) followed; 341860 lost 13 of its 14 tasks to
`PREFLIGHT: giving up after 12 requeues` — PR#181's MIG guard fires
correctly but its requeue cap is too low. The live reruns are **341874**
(knn, 13 tasks) and **341875** (delaunay, 20 tasks). Across the campaign
45 of 62 tasks never produced a step: 31 starved on the MIG-mode A100, 14
earlier ones died with `CUDA_ERROR_NO_DEVICE`.

**Crash root cause.** Six of the seven arms that reached their write step
died with `ModelParameterException: ell_comps must satisfy e0²+e1² < 1`
(magnitudes 1.03–1.414). The `ell_comps` prior is an independent
per-component box, so 21.5 % of it is non-physical; `validate_ell_comps`
returns silently on JAX tracers, so the jitted likelihood is finite and
differentiable in the corner and lanes settle there. On completion
`Result.instance` materialises through `SamplesSummary`, which inherits the
raising policy — the recovery added by PyAutoFit#1486 covers `Samples`, not
the path that runs. Filed as **PyAutoFit#1535** (2026-08-27); the same `updater._save_samples`
early return is the suspect named by the older, still-open PyAutoFit#1487
(weight-threshold prune never runs) — the #1535 PR fixes both.
Downstream, `updater._save_samples` swallows the exception and skips
`samples.csv`, and autolens `save_results` catches only `AttributeError`,
so the process dies before `.completed`. Bijector and log-det method are
innocent: `none` and `logit`, `cholesky` and `slogdet` all appear among the
crashes.

**Recovery (zero GPU time).** `search_internal.dill` survives because the
crash pre-empts its deletion; `MultiStartGradient.samples_via_internal_from`
rebuilds full `Samples` offline. All six arms were rebuilt through the
driver's own `collect_metrics` / `per_lane_block` / `_build_summary`,
marked `recovered_offline: true`, and verified: every −½·`best_fom` matches
that arm's final `prodigy step 3000/3000` log line to 4 d.p., and the knn
arm's `target_id` is byte-identical to its successful sibling's. A bare
rerun would short-circuit to 0 steps and crash identically, so the pending
arms are left to run — they will crash the same way and leave a
recoverable dill.

**Falsification state — partial, no verdict.** **F5 is clean**: the step-0
global-best fom is bit-identical across bijectors on the MGE control
(423546.4213174847 / 380535.00536054926), so the bijector provably leaves
the physical objective alone. **F4 is amended** from byte-identity to
"`best_fom` and max log-likelihood equivalent within fp64 on the winning
lane" — byte-identity is unachievable for a reparameterised 3000-step
optimizer, and F5 already carries the objective-inertness proof; F4 is
therefore informational, not a trip. **F1** becomes scorable only with the
recovered delaunay rows. **F3** (knn, n=1) is not falsified. **F2's
reference deviation** — the driver uses the max `none`-arm log-posterior
per (cell, log_det_method) group instead of a fixed-regularization control
— **still needs a human ruling** before any verdict; it is recorded here
as owed, per this file's discipline.

**Scorer diagnostic readout after the fix (13 rows, no verdict artifact written):** `score_rows` **HALTs at F5** on `delaunay_adapt_split[slogdet]` seed 0 — step-0 global-best fom 357347.020 (`none`) vs 357343.242 (`log_reg`), rel 1.06e-5 — so F5 is clean on MGE but NOT on the slogdet delaunay cell, and the halt is correct behaviour. Scored individually with F5 bypassed: F1[cholesky] falsified, F1[slogdet] UNSCORABLE, F2 UNSCORABLE (no matched seed), F3 falsified, F4 falsified (seed 0 agrees to 9.8e-15, seed 1 disagrees at 1.7e-2). This machine reading disagrees with the hand reading above (F3 "not falsified", F1 pending) and is recorded, not adjudicated: the verdict stage runs only once all 341875 arms land and the F2 reference ruling is in.

**Scorer defect (fixed in this PR's sibling half).** `score_f1` returned a
spurious PASS and `score_f2` a spurious FAIL on missing data (`bool(None)`
defaults); both now return UNSCORABLE. Run with today's data before the
fix, `bijector_ab.py --stage verdict` would have emitted
"falsified_criteria_count=2 → close, no rescoping to logit" — an
artefact-driven false close.

**Erratum to the 2026-08-24 W7 entry above ("CP-4 re-scored on the clean
subset"):** the RAL CPU marginal-band max|Δ| of **1.62 nats is a change
from the original 2.27**, measured on a different (clean) subset — only the
A100 figure (9,619) is genuinely unchanged.

**Records:** `phase_08_regularization/RESULTS.md` "8B"; recovered artifacts
under `results/searches/multi_start_prodigy/imaging/{knn,delaunay_adapt_split}/hst/phase8b/`;
`scripts/misc/searches/bijector_ab.py`.

---

## 2026-08-27 — GATE B part 2 CALLED (human-approved 2026-08-27): PositionsLH is safe for gradient MAP at factor ≤ 1e5 on MGE; factor 1e8 is rejected

**Decision (human, 2026-08-27):** **PositionsLH is not intrinsically
hostile to gradient MAP search on MGE; the pre-registered factor 1e8 was
mis-scaled for a fixed-step searcher.** At factor 1e5, Prodigy(n=256,
prior_box, autoconv) is **5/5 with positions on**, at parity with
positions-off in likelihood, parameters, steps and wall. **Gate B part 1
extends to positions-on at factor ≤ 1e5; factor 1e8 is rejected for
gradient search.**

**Evidence (RAL 341892, 10 arms, harvested 2026-08-27 —
`phase_04_positions/RESULTS.md` "Stage 3"):** the ledger's leading
hypothesis (0.3″ tighter than achievable position precision) does not
survive. The resolved `auto` threshold is 0.200000 — a *tighter* arm — and
it is the arm that fails hardest (0/4 hits, +1 invalid resume), while
loosening the factor at the *same* 0.3″ threshold restores 5/5. Hits
(±2 nats of 31786.78): positions-off 5/5, t0.3·f1e5 5/5, t0.3·f1e8 2/5,
tauto0.2·f1e8 0/4. The f1e5 winner reproduces the positions-off answer to
3 d.p. in likelihood and parameters (r_E 1.5997, shear ≈ (0.0485, 0.0496)),
so the penalty is ≈0 at the recovered model. The damage is *transit*
damage: a 1e8 slope against a 3×10⁴-scale log-likelihood inflates Prodigy's
step scale (median 0.21–0.22 vs 0.14–0.16 at f1e5), throws lanes into the
non-physical prior corner (29 % of best points at |e| ≥ 1 vs 17 % off) and
pins them.

**Six caveats that ride with the call:**

1. **Idealised positions.** The positions are the simulator's own truth,
   and for `auto` the threshold-resolution tracer is the truth tracer —
   not positions solved from a completed search's max-likelihood model as
   the SLaM chained-fit convention would.
2. **One cell, five seeds.** `imaging/mge/hst`, A100 fp64 only. Wilson-95
   lower bound on run success at 5/5 is 0.57 — **this does not re-establish
   the ≥99 % reliability Gate B pt 1 demonstrated at n=256 positions-off**.
3. **1e5 is shown safe, not calibrated.** Nothing was run between 1e5 and
   1e8, and SLaM's own `factor=3` convention is untested here. The call
   licenses "≤ 1e5", it does not locate the boundary.
4. **Nautilus is unaffected either way** — see the W2 entry above: logZ and
   mode unchanged to 0.02 nats, a small but systematic maxL penalty, no
   reliability consequence. The call is about gradient search only.
5. **No `penalty_at_best` field.** Schema v2 records no penalty readout at
   the recovered model, so "the penalty is ≈0 at the winner" is inferred
   from parameter/likelihood parity, not measured. Adding the field is
   owed.
6. **Provenance defect, fixed in this PR.** All three positions-on arms
   hashed to the same `target_id` (`sha256:bf3d096fda76`) and every JSON
   recorded threshold 0.3 / factor 1e8 regardless of what ran —
   `_targets.py`'s positions block was built from module defaults. A
   target_class-3 change was invisible to the target hash, which is exactly
   the §3 comparability guarantee Phase 4 rests on. Fixed here, and affected
   rows re-derived with `restamp_target_block.py` — the three arms now hash
   distinctly (`bf3d096fda76` / `cd522872a7ed` / `6b93f0e52ecd`). Rows whose
   recorded `positions` block was itself defaulted cannot be re-derived and
   must be read against the `config_name` suffix.

**Records:** `phase_04_positions/RESULTS.md` "Stage 3 — threshold vs
stiffness diagnostic (job 341892)"; PROGRAMME.md phase/gate table + §9b;
`results/searches/multi_start_prodigy_autoconv/imaging/mge/hst/hpc_hpc_a100_fp64_n256_seed{0-4}_pos_{t0.3_f1e5,tauto0.2_f1e8}.json`.

---

## 2026-08-28 — W5 / Phase 8B: F2 reference ruling, F5 demoted, preliminary verdict on 24/39 arms

**Ruling (architect, 2026-08-28, recorded BEFORE any scoring was run —
autolens_profiling#185).** The 2026-08-27 entry above left three things owed:
the F2 reference deviation needed a ruling, F5 was HALTing the verdict stage,
and F4's fp-equivalence limb was of uncertain status. All three are settled
here, and the scorer (`scripts/misc/searches/bijector_ab.py`) is amended to
match. The arm table, the readouts and the pre-registered "any two criteria
falsified → 8B falsified" threshold are **untouched**.

**1. The F2 reference is the group-wide physically-valid maximum.** The
reference log-posterior for a `(cell, log_det_method)` group is the maximum
`lane_best_log_posterior` over **ALL** arms in that group — every bijector,
`none` / `log_reg` / `logit` alike — restricted to physically valid rows. A row
is excluded if it is **void** (`diagnostics.valid = false`, total wall < 2 min,
or no `schema_version`) or if its **best point has an ell_comps magnitude
≥ 1** — i.e. it sits outside the unit disk, in the 21.5 % of the independent
per-component `ell_comps` prior box that is non-physical. The magnitude is read
from `recovered_offline_verification.best_point_ell_comps_magnitude` where the
row carries it, and otherwise computed from the winning lane's
`lane_best_params` through `diagnostics.ell_comps_pairs`; every row records
which field was used (`ell_comps_source`). The two agree exactly on all nine
recovered rows, so the computed path is not a second, looser measurement.
The tolerance is unchanged: `REFERENCE_TOLERANCE_NATS = 10`.

*Rationale.* The old reference — max over the `none` arm only — was defined by
the control arm's own stalling: it made "steps to reach where `none` got to"
the target, which is circular when the whole question is whether `none` stalls.
It is now also contaminated: `delaunay_adapt_split · slogdet · log_reg · seed 1`
reports a `lane_best_log_posterior` of **2.1e53** at a best point pinned to the
(±1, ±1) box corner (|e| = 1.41421 on both `ell_comps` pairs). A reference is a
physical target, so a non-physical point cannot set one — under any
max-over-arms rule the corner row would otherwise define every group's bar.

**2. F2 "never reached" has explicit semantics.** At a matched seed, within the
3000-step budget: `none` never comes within tolerance while `log_reg` does →
that seed's reduction ratio is **+inf** and counts as ≥ 2×; `log_reg` never
reaches while `none` does → ratio **0**, counting against; **both** never reach
→ the seed is **unscorable** and drops out. The median is taken over the
scorable seeds, as before. This replaces a silent conflation in which "never
reached" produced no ratio at all, so a group in which `none` never converged
and `log_reg` always did scored as UNSCORABLE rather than as the strongest
possible pass.

**3. F5 is demoted from HALT to a reported diagnostic.** As written, F5
compares the step-0 global-best (min-over-lanes) figure of merit between two
**separate GPU runs**. That is a measurement of floating-point reproducibility
across two processes, not of the objective: the MGE control, whose `log_reg`
map is provably EMPTY (an identity reparameterization), still diverges by
**1.7e-2** relative by step 3000, and the `delaunay_adapt_split · slogdet`
seed-0 pair differs by 1.06e-5 at step 0 — neither can be a bijector effect.
F5 is still computed and reported with the same 1e-9 number, now labelled an
**"fp-reproducibility diagnostic"**, and it no longer halts the verdict. The
sound version of F5 — evaluate the same physical point under both
parameterizations **in one process** and assert bit-equality — is an in-process
PyAutoFit unit test and is filed separately; it is not something a two-run A/B
can measure.

**4. F4's fp-equivalence limb is informational only.** Per the 2026-08-27
amendment, the MGE control's winning-lane `best_fom` / `max_log_likelihood`
comparison is **reported and never scored** — it is the same cross-run
fp-reproducibility quantity F5 measures, and it fails for the same reason.
F4 now trips on one thing only: the `knn` `logit` arm reproducing the
**pinned-lane-to-boundary pathology** (a lane parked on the logit box bound at
completion). The old per-lane byte-identity check remains as a second
informational field.

**5. Stack-version split across the campaign, deliberately not treated as an
A/B split.** The 24 arms landed and scored here ran on **2026.8.17.1 with
pre-#1536 PyAutoFit**. The 15 resubmitted arms (RAL job **341978**, array
indices 0, 2, 3, 14, 17, 19, 21, 24, 25, 26, 27, 30, 32, 33, 34) run on the
post-pull stack (**PyAutoFit f466dce1a, PyAutoGalaxy 0fbe863d, PyAutoLens
b23ee53e9**). The **likelihood code is unchanged between the two**: #1536,
#713, #1538 and #589 touch results-writing and an **opt-in** clipper only, and
no 8B arm opts in. The two halves are therefore pooled rather than analysed as
separate populations — **but it is flagged**, because the claim rests on a
reading of four PRs' diffs and not on a measurement, and the final verdict
should re-check it if any pooled criterion sits near its threshold.

**6. This verdict is PRELIMINARY (24 of 39 arms).** It is emitted now because
the ruling above is what was blocking it and because 24 arms is enough to see
the shape of every criterion; it is **not** the campaign's answer. The verdict
artifact carries `preliminary: true` and `n_rows_expected: 39`, and the final
verdict is re-run when 341978 lands.

**7. No 8B arm used `ClipperPriorBoxJoint`.** With a bijector set, the joint
disk clipper is **refused at construction** (`PyAutoFit
multi_start_gradient/search.py:368-383`) — the joint constraint is expressed in
the untransformed coordinate and the search has no way to apply it through an
arbitrary bijector. Every 8B arm therefore ran `SEARCHES_CLIPPER=prior_box`,
the per-component box that is faithful to a *wrong* box, which is exactly the
mechanism that puts best points outside the unit disk and forced ruling 1.
Filed as a PyAutoFit follow-up: the bijector and the joint clipper need to
compose, or the refusal needs to be a documented, surfaced incompatibility
rather than a silent constraint on the experiment design.

**Scored readout under the ruling (24 rows, `bijector_ab.py --stage verdict`,
artifact `phase_08_regularization/bijector_ab/verdict_<hardware>.json`):**

**VERDICT: FALSIFIED — 3 of 4 criteria fired (F1, F3, F4)**, against the
pre-registered "any two → 8B falsified; record and close, no rescoping to
logit" threshold. **PRELIMINARY (24 of 39 arms.)**

| criterion | state | numbers |
|---|---|---|
| **F1** NaN-wall position (delaunay) | **FALSIFIED** (cholesky tier; slogdet tier UNSCORABLE) | cholesky: median first-value-NaN step **0.0 under both** arms, and value-NaN lane-steps **rise** 18,143 (`none`, 2 rows) → 139,205 (`log_reg`, 5 rows). slogdet: no `none` arm recorded a single value-NaN, so neither limb is measurable. |
| **F2** steps-to-reference (knn) | **NOT falsified** | reference **30,559.28**; 1 matched seed (3). `none` never comes within 10 nats — it tops out 1,645 nats short at 28,914.21 — while `log_reg` is inside the band by step **2,882**. Ratio **+inf** ≥ 2×. |
| **F3** time at λ > 1e4 | **FALSIFIED** (delaunay cholesky) | cholesky `none` **0.0000** vs `log_reg` **0.00076**. The other two groups go the other way: slogdet 0.0469 → 0.0368, knn 0.0625 → 0.0520. |
| **F4** MGE control + logit pathology | **FALSIFIED** (logit limb) | `knn·logit·seed1` ends with a lane holding **7 parameters pinned to the box bound**. Informational: MGE seed 0 agrees (`best_fom` rel 0.0, maxL rel 9.8e-15), seed 1 disagrees at rel **1.73e-2**; byte-identity fails on both. |
| **F5** fp-reproducibility diagnostic | **1 pair above 1e-9**, does not halt | `delaunay·slogdet·seed0` step-0 fom 357,347.020 (`log_reg`) vs 357,343.242 (`none`), rel **1.06e-5**. Every other matched pair, MGE and knn included, agrees within 1e-9. |

**Resolved references and what ruling 1 removed.** delaunay·cholesky
**30,609.94** (`log_reg` s1, 2/7 rows kept) · delaunay·slogdet **30,286.10**
(`log_reg` s3, 2/7) · knn **30,559.28** (`log_reg` s3, 4/6) · mge **31,787.84**
(`log_reg` s0, 4/4). **Twelve of the 24 rows — exactly half — are excluded, all
twelve for the same reason: the best point is outside the unit disk**
(magnitudes 1.032 – 1.41421). **Zero rows are void**: all 24 have
`diagnostics.valid = true`, ran the full 3000 steps and took 1.7–4.0 h. On
`delaunay_adapt_split` the non-physical rate is **10 of 14 (71 %)** and it is
indifferent to bijector and log-det method alike. The 2.1e53 row is one of the
twelve, as intended.

**How much weight this verdict carries.** It is the pre-registered rule's
answer on today's data and is reported as such, but each fired criterion is
thin in a way 15 more arms can move, and this is recorded so the final verdict
is not read as a mere confirmation:

- **F1's fired limb sums raw counts over unbalanced arms** — 2 `none` rows vs
  5 `log_reg` rows (9,072 vs 27,841 per arm). The limb survives normalisation
  (a 3× rise, not a 50 % fall), but the scorer does not normalise and the
  pre-registration's wording fixes that; filed as a follow-up rather than
  changed mid-campaign.
- **F3 fires on a knife-edge**: `none` is *exactly* 0.0000 on the cholesky
  tier and the criterion is `>=`, so any non-zero `log_reg` value trips it.
  Both groups with real high-λ occupancy show `log_reg` spending **less** time
  there.
- **F4 fires on a necessary-not-sufficient proxy**: `n_pinned_final` counts all
  pinned parameters, not just the traced regularization ones, on a single
  `logit` seed.
- **F2, the criterion that did NOT fire, also rests on one matched seed**, and
  its reference is set by the `log_reg` arm at that same seed — so the `+inf`
  reads "`none` never reached what `log_reg` reached" (a real 1,645-nat gap)
  rather than the "2× fewer steps to a common target" the criterion was drafted
  to measure.

**Owed, and not paid here:** the final verdict when 341978 lands; the
in-process PyAutoFit unit test that is the sound F5; the PyAutoFit follow-up on
bijector × `ClipperPriorBoxJoint`; and an F1 limb that normalises per arm.

**Records:** `phase_08_regularization/RESULTS.md` "8B — PRELIMINARY verdict on
24/39 arms"; `phase_08_regularization/bijector_ab/verdict_<hardware>.json`
(`<hardware>` names the machine that ran the SCORER, not the arms — every arm
ran on the RAL A100); the 24 results JSON + 12 PNG under
`results/searches/multi_start_prodigy/imaging/{delaunay_adapt_split,knn,mge}/hst/phase8b/`;
`scripts/misc/searches/bijector_ab.py` ("Scorer amendments 2026-08-28") and
`scripts/misc/test/test_searches_bijector.py`. Issue autolens_profiling#185.

---

## 2026-08-29 — W5 / Phase 8B: FINAL verdict on 39/39 arms — **FALSIFIED, 3 of 4 criteria (F1, F3, F4)**

**This supersedes nothing and changes no rule.** RAL job **341978** landed its
15 arms (array indices 0, 2, 3, 14, 17, 19, 21, 24, 25, 26, 27, 30, 32, 33, 34;
`COMPLETED 0:0`, walls **1:46:42 – 4:02:41**), the campaign is at **39 of 39**
arms, and `bijector_ab.py --stage score` was re-run on the full set. The
2026-08-28 ruling (F2 reference, F5 demotion, F4's informational limb) is
applied unchanged; the pre-registered "any two criteria falsified → 8B
falsified, record and close, **no rescoping to logit**" threshold is applied
unchanged. The verdict artifact now carries `n_rows: 39`,
`n_rows_expected: 39`, **`preliminary: false`**.

**VERDICT: FALSIFIED — 3 of 4 criteria fired (F1, F3, F4). FINAL.**

| criterion | 24 rows (preliminary) | 39 rows (FINAL) |
|---|---|---|
| **F1** NaN-wall position | FALSIFIED on the **cholesky** tier only; slogdet **UNSCORABLE** | **FALSIFIED on BOTH tiers.** cholesky: median first-value-NaN step **0.0 under both** arms, value-NaN lane-steps 80,956 (`none`, 5 rows) vs **139,205** (`log_reg`, 5 rows). slogdet: now measurable — `none` 1 seed of 5 NaNs (median first step **938**, total **2,062**), `log_reg` 5 of 5 (median first step **144**, total **38,367**) — `log_reg` walls **earlier** and **18.6× harder** |
| **F2** steps-to-reference (knn) | NOT falsified; 1 matched seed | **NOT falsified; still 1 scorable seed.** 5 seeds now match, 4 are unscorable (neither arm reaches the reference). Reference **30,559.28**, unchanged, still set by `knn·slogdet·log_reg·seed3`; that seed's `none` never arrives, `log_reg` arrives at step **2,882** → ratio `+inf` ≥ 2× |
| **F3** time at λ > 1e4 | FALSIFIED on a knife-edge (`none` **exactly** 0.0000) | **FALSIFIED, knife-edge gone.** delaunay·cholesky `none` **0.000329** vs `log_reg` **0.000762** — both non-zero, and `log_reg` is 2.3× higher. The other two groups still go the other way: delaunay·slogdet 0.0375 → 0.0255, knn 0.0375 → 0.0326 |
| **F4** MGE control + logit pathology | FALSIFIED on **one** logit seed (7 pinned) | **FALSIFIED on all five.** `knn·logit` max pinned-at-completion parameters per seed: **9 / 7 / 9 / 10 / 7** (seeds 0–4). Informational MGE limb unchanged: seed 0 agrees (`best_fom` rel **0.0**, maxL rel **9.8e-15**), seed 1 disagrees at rel **1.73e-2**; byte-identity fails on both |
| **F5** fp-reproducibility *(diagnostic, never scored)* | 1 pair above 1e-9 | **2 pairs**, max rel **1.06e-05** (`delaunay·slogdet·seed0`; the second is `delaunay·cholesky·seed0` at 2.43e-06). Does not halt |

**What the 15 arms actually changed, criterion by criterion.**

1. **F1 gained a whole tier.** At 24 rows the slogdet limb was unscorable
   because no `none` arm had recorded a single value-NaN. `slogdet·none·seed4`
   supplies one (2,062 lane-steps, first at step 938), and with it the tier is
   measurable and fires on **both** limbs at once: `log_reg` NaNs **earlier**
   (median step 144 vs 938 — the only place in the campaign where the "not
   earlier" limb is actually contradicted) and **18.6×** more often. The
   preliminary verdict's F1 rested on one tier; the final one rests on two.
2. **F1's "unbalanced raw counts" caveat is gone.** The cholesky limb now sums
   **5 `none` rows against 5 `log_reg` rows**, not 2 against 5, so the raw
   comparison the scorer performs is the normalised one: **16,191 vs 27,841
   value-NaN lane-steps per arm**. The owed per-arm-normalised limb is
   therefore no longer load-bearing on this evidence.
3. **F1's ratio deflates 7.7× → 1.7×, and the criterion still fires.** The
   three new `none` cholesky rows are NaN-heavy, and one of them —
   `delaunay·cholesky·none·seed0`, **29,938 lane-steps, 37 % of the whole
   `none` total** — is the single largest contributor on either arm. It is
   also the arm with `max_log_likelihood = −75,839.87` and a best point at the
   box corner (|e| = 1.41421), i.e. a failed fit. Counterfactual, since a
   reader will ask: 24 rows **7.67×** (18,143 vs 139,205) · 39 rows with seed 0
   **1.72×** · 39 rows without it **2.73×**. The criterion is
   `log_reg ≥ 50 % of none`, and 139,205 exceeds `none`'s total outright in
   every one of the three, so **F1 fires with or without the row.** Recorded
   because the *effect size* moved by 4.5× even though the verdict did not.
4. **F3 stopped being a knife-edge.** The preliminary reading flagged that
   `none` was *exactly* 0.0000 on the cholesky tier and the criterion is `≥`,
   so any non-zero `log_reg` tripped it. With the three new `none` rows the
   comparison is 0.000329 vs 0.000762 — a real 2.3× ratio between two small
   numbers. The caveat that both tiers with substantial high-λ occupancy go the
   *other* way (`log_reg` spends **less** time at λ > 1e4) still stands.
5. **F4 stopped resting on one seed.** All five `knn·logit` seeds now finish
   with a lane pinned to the box bound (9/7/9/10/7 parameters). The proxy is
   still necessary-not-sufficient — `n_pinned_final` counts every pinned
   parameter, not only the traced regularization ones — but it is no longer a
   single observation.
6. **The `logit` arm collapses to a no-lens solution.** Four of the five
   `logit` seeds have a winning lane whose best point has
   `einstein_radius` ≈ **0** (0.0000, 0.0008, 0.0018, 0.0017; seed 4 is the
   exception at 6.99), and **all five** are non-physical in `ell_comps`
   (|e| 1.2366 – 1.41407). A degenerate no-lens fit at the corner of the
   parameter box is what the `logit` reparameterization produced on this cell,
   and it is the concrete content behind F4's pinning statistic.
7. **F2, the criterion that did not fire, did not improve.** Five knn seeds now
   match instead of one, but four are **unscorable** — neither arm reaches the
   reference. The `+inf` still comes from seed 3 alone and still reads
   "`none` never reached what `log_reg` reached", not the "2× fewer steps to a
   common target" F2 was drafted to measure. Fifteen more arms did not buy this
   criterion a second data point.

**Non-physical best points: 23 of 39 (59 %), and it is the campaign's largest
single finding.** Per group: delaunay·cholesky **6/10 (60 %)**,
delaunay·slogdet **7/10 (70 %)**, knn **10/15 (67 %)**, mge **0/4**. The
excluded magnitudes cluster hard at **1.41421** — the (±1, ±1) corner of the
per-component `ell_comps` box, the farthest point from the unit disk that the
box admits. Zero rows are void: all 39 have `diagnostics.valid = true` and ran
the full 3000 steps. The rate is indifferent to bijector and to log-det method,
and it is zero on the parametric control — this is a property of the **pixelized
cells' box-clipped `ell_comps` geometry**, not of the thing 8B was testing
(W10; PyAutoMind `feature/autogalaxy/ell_comps_joint_disk_constraint.md`).

**Stack provenance — what the artifacts do and do not support.** All 39 results
JSONs record `version: 2026.8.17.1` and **nothing else about the stack**: no
library git SHAs are captured in the schema. The 2026-08-28 entry's ruling 5
(the two halves ran different PyAutoFit commits, pooled because the likelihood
code is unchanged) therefore **cannot be re-verified from the harvested
artifacts** — it rests, as it did then, on a reading of four PR diffs. What can
be said from the artifacts: every row is the same declared version, and every
arm ran `SEARCHES_CLIPPER=prior_box`, so PyAutoFit **#1540** (merged 2026-08-28
16:35 UTC, before 341978 started at 18:11) is behaviour-neutral for this
campaign whichever side of it an arm ran on. A future campaign should record
the library SHAs in the results JSON so this is a measurement.

**Owed items, settled and outstanding.**

- **PAID — the sound, in-process F5.** PyAutoFit **#1540** ships
  `test_bijector.py::test__round_tripping_a_per_path_map_is_bit_exact_where_it_is_identity`,
  which evaluates a `BijectorPerPath` map in **one** process and asserts
  bit-equality on the identity coordinates (`rel=1e-12` on the `log` one). That
  is the test the 2026-08-28 entry said a two-run A/B could never be, and it is
  why F5 stays a reported diagnostic here rather than evidence about the
  bijector.
- **PAID — bijector × `ClipperPriorBoxJoint`.** #1540 replaces the blanket
  construction-time refusal with a per-pair resolution: a ball pair under one
  common positive linear scale composes (radius `R → R/s`), anything genuinely
  non-linear raises, and the check moved to model resolution inside `_fit`.
  **Restating the follow-up it creates:** a rerun of this experiment would run
  `BijectorPerPath` applying `logit` **everywhere except the `ell_comps`
  pair** — `logit` on the ball pair still raises, by design — so that the joint
  disk clipper can hold lanes off the 1.41421 corner while the
  regularization coefficients still step in a transformed coordinate. That is a
  **new experiment**, not a rescoping of 8B: the pre-registration forbids
  rescoping to logit, and this verdict is final on the arms as designed.
- **OUTSTANDING — a per-arm-normalised F1 limb** in the scorer. No longer
  load-bearing (point 2 above), still the correct wording, still not written.
- **OUTSTANDING — library SHAs in the results JSON schema**, per the provenance
  paragraph above.

**Records:** `phase_08_regularization/RESULTS.md` "8B — FINAL verdict on 39/39
arms" and its per-arm appendix; `phase_08_regularization/bijector_ab/verdict_<hardware>.json`
(`preliminary: false`, `n_rows: 39`, every exclusion reason); the 39 results
JSON under `results/searches/multi_start_prodigy/imaging/{delaunay_adapt_split,knn,mge}/hst/phase8b/`;
PROGRAMME.md phase table + §9b W5. Issues autolens_profiling#162 / #185 (both
closed) and #194.

---

## 2026-08-29 — W6 tail: MGE `n_batch` wall optimum is **2000**; the delaunay leg was lost to a cuFFT OOM that reported success

**Record, not a gate call.** The 2026-08-27 W6 entry closed on "the scan has
not plateaued at `n_batch=1000` and there is an optimum past it that this scan
does not bracket". Two tail jobs were dispatched to bracket it. One landed.

**MGE/hst (RAL 341987, both arms `COMPLETED`, 8:48 and 9:12).** Appending the
two new arms to the scan (`n_live=200`, fp64, one seed per arm, viz on):

| n_batch | ms/eval | sampler wall | total wall | evals | Kish ESS | ESS/min | logZ |
|--------:|--------:|-------------:|-----------:|------:|---------:|--------:|-----:|
| 64   | 10.56 | 670 s | 738 s | 63,424 | 4,304 | 386 | 31690.4548 |
| 1000 |  5.95 | 458 s | 526 s | 77,000 | 4,694 | 615 | 31690.3580 |
| **2000** | **4.19** | **444 s** | **519 s** | 106,000 | 5,378 | 727 | 31690.2393 |
| 4000 |  3.03 | 473 s | 540 s | 156,000 | 6,058 | 769 | 31690.2162 |

**The wall optimum is `n_batch=2000`, and the scan now brackets it.** Sampler
wall turns over between 2000 and 4000 (444 → 473 s) and so does total wall
(519 → 540 s). **ms/eval has still not plateaued** — it keeps falling to 3.03
ms at 4000 — which is exactly the trap the 2026-08-27 entry named: the per-eval
number improves monotonically while the thing anyone cares about stops
improving, because the eval count runs away (77,000 → 106,000 → **156,000**) as
larger batches overshoot the shrinking live set. Read the wall column.

**ESS/min disagrees with the wall, and that is not a contradiction.** ESS/min
is still rising at 4000 (727 → 769) because Kish ESS rises faster than wall
does (5,378 → 6,058). If the deliverable is a posterior sample, 4000 is
marginally better; if it is a completed fit, 2000 is better. Neither margin is
large and both rest on **one seed per arm**, so neither is adopted as a default
here.

**The logZ drift got worse, monotonically, and it is the reason nothing is
adopted.** logZ falls without interruption across the whole upper scan:
31690.4772 (nb256) → 31690.4738 (512) → 31690.3580 (1000) → **31690.2393**
(2000) → **31690.2162** (4000). Against nb64 that is **−0.2155 nat** at 2000
and **−0.2386 nat** at 4000 — roughly **20σ and 22σ** of the 0.011-nat
five-seed logZ standard deviation measured on this same cell in Phase 4 Stage
2, and 2–2.4× the −0.10 nat deviation that already blocked adopting nb1000.
A monotone drift over five arms is harder to read as a seed draw than the
single point was, but with one seed per arm the scan still **cannot** separate
an `n_batch` bias from seed noise. **No `n_batch` above the measured baseline
is adopted as a default.** The next step, if anyone wants this knob, is
`n_batch ∈ {64, 2000} × 5 seeds` on MGE — not more points along the scan.

**`n_batch=4000` is at the VRAM edge.** `error.341987_1.err` carries **10**
BFC-allocator retry lines ("ran out of memory trying to allocate … The caller
indicates that this is not a failure"); `error.341987_0.err` (nb2000) has
**none**. The 4000 arm completed, but it is buying its ESS/min inside the
allocator's fallback path on an 80 GB A100.

**Delaunay/hst (RAL 341988, `n_batch` 512 and 1000): NO DATA, and the job said
it was fine.** Task 0 died initializing a **cuFFT batched plan that wanted
25.31 GiB** of scratch (`Failed to initialize batched cufft plan with
customized allocator` → `INTERNAL: RET_CHECK failure (fft_thunk.cc:200)` →
`jax.errors.JaxRuntimeError`); task 1 died on a straight
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 100.97GiB`. Both
were recorded by SLURM as **`COMPLETED 0:0`**, in **1:02** and **1:07**.

This is **not** a resubmit. A pixelized cell at `n_batch` 512 batches the
convolution FFT 512-wide; the scratch requirement is linear in the batch and
100.97 GiB does not fit an 80 GB card at any allocator setting. It needs a
**chunked / smaller-batch redesign** of the batched evaluation before the
question can be asked at all. Its cost/benefit is also poor: delaunay
**saturated at `n_batch=64`** (1.26×, then flat) in the original scan, because
a pixelized cell's cost is the per-eval inversion and not batch occupancy. The
W6 reading is unchanged — **raise `n_batch` on parametric cells, leave it at
the default on pixelized ones** — and the delaunay tail stays **unrun**.

**The failure channel is closed in this commit.** Every `hpc/batch_*` submit
ended `python3 … ; echo "Finished." ; date`, so the script's exit status was
`date`'s and was always 0 — a Python crash was indistinguishable from a clean
run in `sacct`, and 341988's traceback existed only in `.err` beside a results
JSON that never appeared. The repo-root `activate.sh`, which every submit
already sources, now arms `set -eE` plus an `ERR` trap **when `$SLURM_JOB_ID`
is set** (interactive login shells are untouched, and `pipefail` is
deliberately not set — `_gpu_preflight.sh` tolerates a failing `nvidia-smi` by
design). A crashed job now exits with Python's status and SLURM records
`FAILED`. Verified against a submit-shaped script: the guard propagates exit 7,
`echo "Finished."` never runs, `|| echo` tolerated failures (`run_probe`,
`run_cell`) still pass, and the preflight's `nvidia-smi | head | tr` pipeline
still survives a failing `nvidia-smi`. Written up in `hpc/README.md`
"A crashed job FAILS".

**Records:** `methods/nautilus.md` "n_batch SCAN"; PROGRAMME.md §9b W6;
`results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64_nbatch{2000,4000}.json`;
`hpc/README.md` "A crashed job FAILS"; `activate.sh`. Issue #163 (closed),
#194. The tail submits themselves are PR#193, unmerged and untouched by this
work.

---

## 2026-08-29 — Phase 6 first NUTS probe (#187): the rate is measured, **H6.1 is unsupported with a MAP-sourced diagonal metric**, and the A/B is unscorable

**Record, not a gate call — GATE C is not called and is not close.** RAL job
**341981** ran the first `af.BlackJAXNUTS` arm this repo has ever produced, on
`searches/nuts/imaging/mge × hst × fp64`: 4 vmapped chains, 200 warmup + 200
samples, `target_accept=0.8`, `max_num_doublings=10`,
`inverse_mass_matrix=diagonal`, seed 0 fixed across both arms, visualization
off. The two arms differ **only** in initialization.

**Arm 0 (COLD, `InitializerBall` at the prior median): TIMEOUT.** Elapsed
**45:26** against a 45:00 limit, still inside window adaptation — the last line
it logged was `BlackJAXNUTS: window adaptation (200 steps x 4 chains …)` at
13:29:08, and it produced no result artifact. A cold start on this cell costs
**more than 2,726 s** and remains **unmeasured**.

**Arm 1 (WARM, 4 start points from a completed MultiStartProdigy MGE MAP fit,
jittered 0.05σ): COMPLETED in 1368.78 s.**

| quantity | value |
|---|---|
| wall | **1368.78 s** (22:49), visualization disabled |
| likelihood + gradient evals | **35,523** |
| **per eval** | **38.53 ms** |
| per draw (400 draws × 4 chains, lockstep) | **3.42 s**, i.e. 88.8 evals per draw |
| max log likelihood | **31,786.03** |
| split-R̂ max | **3.89** |
| divergences | **400 of 800** sampling draws |
| ESS min / bulk / tail | **2.01** / 4.33 / 4.04 |
| mean acceptance | 0.672 |
| ESS per gradient eval | 1 / **17,672** |
| ESS/min | **0.088** |
| log evidence | `nan` (NUTS computes none) |

**1. The warm start works: it reaches the right basin.** maxL **31,786.03**
sits within ~1 nat of the Nautilus optimum on the same cell (31,786.9 –
31,787.1) and 264.3 nats above the truth likelihood. `InitializerParamStartPoints`
placed the chains where it was asked to; nothing about the warm-start
machinery (PyAutoFit #1522) failed.

**2. The posterior it produced is invalid, and not marginally.** Split-R̂
**3.89** against Gate C's < 1.01; **half** the sampling draws divergent against
"no material divergences"; ESS per gradient eval **1/17,672** against "not
worse than ~1/30". These are not three near-misses, they are three failures by
2–3 orders of magnitude. Nothing about H6.1 — "NUTS warm-started at the MAP
achieves ESS/s ≫ nested sampling on MGE" — is supported by this run. For scale
on the same cell and hardware: Nautilus at `n_batch=2000` delivers **727
ESS/min**; this arm delivered **0.088**, a factor of ~8,000 the wrong way.

**3. The diagnosis is the metric, and H6.1 predicted it.** H6.1's own wording
says "dense/low-rank mass matrix from the previous fit's covariance … the 269×
prior/posterior anisotropy and |r| = 0.95 correlations are measured — **diagonal
mass will not suffice**". This arm ran `SEARCHES_NUTS_MASS=diag`, deliberately:
the warm source is a **MAP optimizer** whose stored samples are best-points and
not a posterior, so its covariance is MLE-only and PyAutoFit refuses to seed a
dense metric from it. So the probe tested the one configuration H6.1 says
should fail, because it is the only one a MAP warm-start can express. **H6.1 is
untested, not refuted** — what is measured is that a MAP-sourced *diagonal*
metric does not mix on this geometry, which is the anisotropy argument
reproduced from the inside.

**4. The warm-vs-cold A/B is unscorable.** The cold arm produced nothing, so
there is no comparison. It also cannot be recovered from this job's budget.

**5. The rate is now measured, which is what the probe was for.**
`wall/rates.py` gains
`("imaging", "mge", "hst", "a100", "fp64", 4, None) = 3.42` **seconds per
draw**, and `submit_search_nuts_imaging_mge_a100_hst_fp64_probe`'s WALL-BASIS
row moves from `source: unmeasured / probe-first: yes` to **`source: rates`**.
Three things about that row are load-bearing:
   - **The unit is per DRAW, not per optimizer step.** Every other row in
     `rates.py` is seconds per optimizer step = one likelihood evaluation. A
     NUTS draw is a whole leapfrog trajectory (up to `2**max_num_doublings`
     evaluations) and cost **88.8 evals** here. The 38.53 ms/eval figure is the
     one comparable with the optimizer rows (MGE unbatched n16 is 0.05 s/step);
     the per-draw figure is the one a submit can multiply by a step count it
     knows in advance, because the eval count depends on the tree depth NUTS
     chooses at runtime. Both are recorded; they must not be swapped.
   - **The rate is WARM-only.** Sizing a cold NUTS job from it is precisely the
     cross-configuration substitution `rates.py` exists to refuse.
   - **The submit's `--time` is raised 0:45:00 → 2:00:00** with
     `headroom: 5.0` declared against the 1.5 floor, so the budget covers the
     cold arm's demonstrated >2,726 s overrun rather than the warm arm alone.
     Leaving 45 minutes beside a `rates` row measured from the arm that *fit*
     it would have re-armed the trap that killed arm 0.

**6. What the next Phase 6 probe should be.** A **dense** metric seeded from a
**Nautilus** warm start — a real posterior, whose covariance PyAutoFit will
accept — on the same cell, with **≥ 2 h** budgeted for a cold control arm.
That is the configuration H6.1 actually names, and it is the first one capable
of testing it. A ≥5-seed reliability run comes after Phase 6 has a verdict, per
PROGRAMME.md §3. Filed as a follow-up, **not** started here.

**Records:** PROGRAMME.md Phase 6 + §9b W11;
`results/searches/nuts/imaging/mge/hst/hpc_hpc_a100_fp64_c4_w200_s200_warm.json`;
`scripts/misc/wall/rates.py` (`nuts_mge_hst_a100_fp64_c4_warm`);
`hpc/batch_gpu/submit_search_nuts_imaging_mge_a100_hst_fp64_probe`. Issue #187
(closed), #194.

---

## 2026-08-29 — 341908_5 diagnosed: it was a **likelihood-overflow flood**, not thrashing. The 2026-08-27 W4 pilot answer is superseded, and a **library/target stack boundary** is declared for every adapt-split row

**Supersedes** the 2026-08-27 W4 entry ("Pilot answer: it thrashes") and the
`targets/REFS_V1_HARVEST.md` text it pointed at. That entry is not deleted —
this one corrects it. Issue: autolens_profiling#196.

**Decision (five parts):**

1. **The 341908_5 record is corrected.** `slam_source_pix_nn` did not make zero
   Nautilus calls and did not thrash on resamples. It was killed by an
   overflow flood.
2. **The free-adapt-split targets move to `al.reg.AdaptSplitPower` under a
   capped `LogUniform(1e-6, 1e4)` coefficient prior** — `knn`,
   `slam_source_pix_nn` and the `delaunay_adapt_split` diagnostic cell, all
   built through one helper (`searches/_setup._free_adapt_split`) so the
   documented knn ≡ delaunay_adapt_split parameter identity holds by
   construction.
3. **A stack boundary is declared.** No adapt-split coefficient measured after
   2026-08-29 is comparable with one measured before it. Same number, different
   meaning: λ² here, λ⁴ there (`c_new = c_old ** 2` maps between them).
4. **The `knn` reference row is withdrawn as a bar** and queued to re-run. Its
   480-nat deficit is the same pathology at lower amplitude.
5. **Recorded rows are NOT re-stamped.** They ran the legacy target and their
   `target_id`s correctly identify it.

**Evidence — 341908_5's `checkpoint.hdf5`, the only artifact the run left:**

| | recorded | previously believed |
|---|---|---|
| likelihood calls | **90,000** | 0 |
| bounds | 29 | — |
| `explored` | **FALSE** | — |
| max log L | **30,701.3** | none |
| per-eval | 0.239 s | — |
| MaxRSS | 3.66 GB | — |
| exit | TIMEOUT (6 h wall) | TIMEOUT |
| NaN draws | **0** | "thrashing on resamples" |
| −inf draws | **0** | — |
| max finite `log_l` | **3e+303** (shells 14/23/24/26/28) | — |
| max `shell_log_l` | **1e56**, at `shell_n_eff` ≈ 1 | — |

**Mechanism.** `al.reg.AdaptSplit` squares its coefficient twice, so under the
legacy `LogUniform(1e-6, 1e6)` prior the regularization term reaches 1e24. The
curvature-plus-regularization matrix goes non-positive-definite from c ≈ 1e4,
and the fp64 Cholesky there returns **finite garbage** rather than NaN.
PyAutoFit's `Fitness` passed the finite value through — it screened NaN and
±inf and nothing else — Nautilus accepted 3e+303 as its best point,
`shell_log_l` reached 1e56 with an effective sample count of one, and `f_live`
could never fall below its termination threshold. The run could only end at the
wall clock. **A NaN would have been rejected by every search in the stack; a
finite 3e+303 was accepted.** That asymmetry is the whole failure.

The `.out` froze at `Calls | 0` for a separate and compounding reason: a SLURM
`.out` is a file, so Python block-buffers stdout at 8 KiB and the wall-clock
kill discarded the buffer. Two independent defects made a healthy-but-doomed
6 h run indistinguishable from a job that never started, and the run was
ledgered wrongly for two days on the strength of it.

**Why maxL 30,701.3 matters more than any of the above.** It is ABOVE the
certified `delaunay_nn` reference (30,650.77). The cell is not unaffordable
under nested sampling — it was finding good fits the entire time and could not
be told it had converged. The 2026-08-24 W4 entry's framing (DelaunayNN's
resample behaviour is a finding, not something to engineer around) still
stands; it simply was not what happened here. Zero NaN draws means the resample
hazard never fired.

**What was changed, on both sides:**

| | change | where |
|---|---|---|
| library | `al.reg.AdaptSplitPower` — squares the coefficient once; `power` is a `Constant` (never sampled), so the model dimension is unchanged and `power=2.0` reproduces the legacy numerics exactly | PyAutoArray, merged 2026-08-29 |
| library | `Fitness` rejects implausibly large finite log-likelihoods — `general.test.log_likelihood_ceiling`, default 1e20 | PyAutoFit, merged 2026-08-29 |
| target | coefficient priors capped at `LogUniform(1e-6, 1e4)`, below the measured non-PD onset | `searches/_setup._free_adapt_split` |
| ops | `export PYTHONUNBUFFERED=1` in the `$SLURM_JOB_ID` block — covers all 85+ submits at once | `activate.sh` |

The class change alone would not have been enough: it still admits c² = 1e12 at
the top of the legacy prior. The cap is what keeps the sampler off the non-PD
region; the `Fitness` ceiling is the backstop, not the fix.

**The stack boundary, precisely.** The `target_id`s of every registry target
that uses free adapt-split regularization have changed, because the priors are
hashed:

| target | before (legacy λ⁴, cap 1e6) | after (λ², cap 1e4) |
|---|---|---|
| `knn_fp64` | `sha256:84c0d88d3032` | `sha256:ccafb8b191bc` |
| `knn_mp` | `sha256:5b3f2dd1a8f9` | `sha256:a027990c04bf` |
| `knn_pos_fp64` | `sha256:f2bebfcc525f` | `sha256:d06e54bad6c0` |
| `knn_pos_mp` | `sha256:f946c4cae821` | `sha256:58fa92c1cec3` |
| `slam_source_pix_nn_fp64` | `sha256:1721493bba6b` | `sha256:ad291b57fc62` |
| `slam_source_pix_nn_mp` | `sha256:d302d3c1b597` | `sha256:dc6f6afac7e4` |
| `slam_source_pix_nn_pos_fp64` | `sha256:6befb71d64ee` | `sha256:8021b4b697ff` |
| `slam_source_pix_nn_pos_mp` | `sha256:16f97c3a75b4` | `sha256:fda3bd6f4be0` |

These are **new targets, not re-runs of the old ones**, and the id change is
the mechanism that says so. Every recorded row keeps its old id: 17 `knn` rows
(the 341879_7 reference plus 16 Phase 8B Prodigy arms) genuinely ran the legacy
target, and re-stamping them would erase exactly the boundary this entry draws.
`restamp_target_block.py` refuses them of its own accord — its reproduction
control fires, reporting that the environment computes the same id both the old
way and the corrected way while the row carries a third — which is the correct
behaviour: the hashing function did not change, the target did.

Cells NOT affected: `delaunay_nn` and `slam_source_pix` (ConstantSplit /
rectangular `Adapt` respectively), `delaunay_matern`, `mge`, `delaunay`,
`pixelization`. Their rows and ids are untouched.

**InferenceRefs_v1 moves from "9 certified of 13" to "8 that stand of 13".**
The knn row is withdrawn as a bar (not deleted — it is a valid measurement of
the legacy target), and three tasks go back up after `HPCPullPyAuto`:
`--array=5,6,7` on `submit_search_nautilus_inference_refs_v1_array.sh`.

**Two collateral findings, both fixed here:**

- **`build_nautilus`, `build_nss`, `build_nuts` and `build_smc` did not tag the
  `log_det_method` arm into their `unique_tag`.** #175 fixed this for the
  MultiStart path only, after RAL 340576 produced 10 output directories for 20
  arms. `Nautilus.__identifier_fields__` contains no log-det field either, so
  the `slam_source_pix_nn` slogdet A/B this wave adds would have hit the same
  defect on the nested path — two arms, one run, two results JSONs with
  different basenames and identical contents. `searches/_samplers.log_det_arm_tag`
  is now shared by all five builders. Tagged on the env override only, so every
  existing output path stays byte-identical.
- **The `slam_source_pix_nn` docstring's "RISK ON THIS ARM" note and the `knn`
  docstring's #117 lambda^4 lesson both describe the legacy class** and are
  marked as superseded in place, so the next reader is not budgeting a
  resurrection plateau for a surface that no longer exists.

**What is NOT decided here.** Whether the corrected target converges — that is
what refs 5/6/7 measure. Whether `slogdet` changes anything a nested sampler
can see on the corrected target — that is what the new 2-arm
`submit_search_nautilus_slogdet_ab_imaging_slam_source_pix_nn_a100_hst_fp64`
measures, and "no difference, because the cap already removed the failure" is a
real finding rather than a failed experiment. The factor-2 `Adapt` scatter
asymmetry (`inner=outer=1` is 2x `Constant(1)`) is documented and filed, not
fixed. Whether the `*Power` classes become the library default is a separate,
deferred, breaking-release decision.

**Records:** `targets/REFS_V1_HARVEST.md` (the corrected pilot section);
PROGRAMME.md §Phase-1 row + §9b W4 row; `searches/_setup._free_adapt_split`.

---

## 2026-08-29 — Phase 7 opened: `af.SMC` cell added, first A100 probe prepared (3 arms)

**Decision:** The searches framework gets a **Sequential Monte Carlo cell**
(`smc`), wrapping the `af.SMC` search merged into PyAutoFit on 2026-08-29
(blackjax adaptive tempered SMC, MALA or HMC inner kernel). One cell today,
`scripts/imaging/searches/smc/mge.py`, on the same `imaging/mge/hst` target
every other sampler in the framework fits. Phase 7 moves to **in flight**.
GATE C is not called and nothing is adopted.

**Why SMC and why now.** Every gate in this programme is scored against a
`log_evidence`, and every one of those numbers has come from a nested sampler.
Nautilus is the bar; `af.NSS` is another nested sampler and so is not an
independent check on it; `BlackJAXNUTS` and the MultiStart optimizers produce
no evidence at all. SMC's tempering bridge produces one **from the gradient
side, by a different route**, which makes a cold SMC run the first available
cross-check on the quantity Gates A and C are written in terms of. The
Phase-6 NUTS probe (341981) also left the gradient-MCMC question open in an
unsatisfying way — right basin, invalid posterior (split-R̂ 3.89, 400/800
divergences, ESS/min 0.088 against Nautilus's 727) — so "a gradient sampler
that gets there" is not yet demonstrated on this cell by anything.

**The probe** (`submit_search_smc_imaging_mge_a100_hst_fp64_probe`, 3 tasks,
`--time=2:00:00`, `source: unmeasured  probe-first: yes`):

| task | arm | what it answers |
|---|---|---|
| 0 | `mala_warm` | the cheap kernel's rate and acceptance from a known-good centre |
| 1 | `hmc_warm` | whether 8x the gradient work per rejuvenation step buys longer moves on a 269x-anisotropic posterior |
| 2 | `mala_cold` | the **evidence bridge**: cold SMC's `log_evidence` against Nautilus's 31690.42 nats on this cell |

256 particles, 5 rejuvenation steps, `target_ess=0.5`, `max_smc_steps=60`,
seed 0 fixed across all three, positions off.

**Warm start: `prior_scaled`, not `result`, and the artifact says so.**
`af.SMC.is_warm_start` is literally `inverse_mass_matrix is not None`, and
warmth changes three things at once — the Gaussian reference centre, the
particle draw, and the evidence Jacobian. There is no "warm centre, cold
reference" configuration. Passing the warm source itself would whiten by its
`samples.covariance_matrix`, and PyAutoFit **refuses** that here, correctly: the
only warm source on RAL is `multi_start_prodigy/imaging/mge/hst/n16_s3000_seed0`,
a 16-lane MAP run, and `imaging/mge` has 15 free parameters — 16 samples against
a `2 * n_dim = 30` floor, so its covariance is the identity fallback. The
library's own named alternative ("pass an explicit `inverse_mass_matrix` array
instead") is what the warm arms use: reference centred on the MAP, width
`0.1 x prior width` per parameter, recorded as `whitening_kind` in every row.
**It carries no parameter correlations, so it cannot test H6.1's anisotropy** —
that needs a Nautilus-sourced warm start and is a follow-up, exactly as it is
for the NUTS `result` arm.

**Two accounting decisions, both to keep an SMC row honest beside a nested one:**

- **Evals are DERIVED, not counted.** `af.SMC` records no evaluation counter,
  and `samples.total_samples` is `num_particles` — 256 stored particles for a
  run that made six figures of evaluations, the error class of issue #177.
  `_samplers.smc_likelihood_evals` reconstructs
  `num_particles * (1 + n_smc_steps * num_mcmc_steps * per_step)` from the
  recorded schedule (`per_step` = 1 for MALA, `num_integration_steps` for HMC).
  Exact for the kernel work, and it does **not** include the evaluations
  blackjax spends solving for the next λ — so an SMC `ms/eval` is slightly
  optimistic and must never be presented as a measured counter the way the NUTS
  `n_logl_evals` row is.
- **ESS is the Kish ESS, and here that is right.** SMC particles carry genuine
  normalised importance weights, so the generic path is correct. The NUTS
  substitution (rank-normalised `ess_min`) exists only because NUTS weights are
  all 1.0; it is deliberately not applied to SMC.

**`Converged` is not evidence of anything.** `af.SMC` sets it when λ reaches
1.0 and nothing else, and an adaptive schedule will walk a collapsed cloud all
the way there. Every SMC row carries a diagnostics block with the λ schedule,
the per-step acceptance trace and the per-step ESS, and marks itself
`valid: false` — shouted, never raised — on a partial tempering path or a final
ESS below 10% of the particle count. The local test-mode smoke exercised that
path: an 8-particle / 3-temperature run stopped at λ = 1.9e-4 and correctly
reported itself uninterpretable rather than offering its `log_evidence`.

**Records:** `scripts/misc/searches/README.md` "Sequential Monte Carlo (`smc`)";
PROGRAMME.md §Phase 7.
