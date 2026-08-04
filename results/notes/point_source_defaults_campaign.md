# Point-source defaults campaign — A100 truth-anchored evidence (#678 phase B)

**Issue:** [PyAutoLens#678](https://github.com/PyAutoLabs/PyAutoLens/issues/678)
**Branch:** `feature/point-source-defaults-campaign`
**Status:** COMPLETE — 23/25 cells produced a result JSON (A100 fp64, RAL);
the remaining 2 are the `simple_extra` exp-3 arms, which did not finish and
**whose non-completion is itself the finding** (see 3). Final jobs landed
2026-08-02: `multi_start_prodigy/cluster/image_plane_solved` (331887,
COMPLETED 00:14:16) following the two gradient bug fixes it exposed
(PyAutoFit#1441, PyAutoLens#685), plus clean re-runs of the two free `simple`
Nautilus cells (331888/331889) that had been silent resumes.
**Instrument:** `scripts/{point_source,cluster}/searches/**` via
`scripts/misc/searches/_runner.py`; every JSON carries
`results.truth_log_likelihood` (likelihood of the simulator-truth model under
the cell's own fit class) and `results.delta_max_ll_vs_truth`
(`max_log_likelihood − truth_log_likelihood`).

## TL;DR

**The evidence supports all four target defaults of the campaign prompt:**
tensor source-plane weighting, solved centres on both chi-squared flavours,
solved-centre demonstrated defaults in the workspace, and **all-to-all
image-plane pairing** — the missing-image discriminator is decisive
(`PairRepeatSolved` mis-ranks truth by ~1.8×10⁵ log-likelihood when one image
is absent; `PairAllSolved` recovers truth cleanly). Two caveats for the docs:
gradient searches are not yet competitive on **any cluster-scale** objective —
source-plane *or* image-plane-solved (use Nautilus at cluster scale) — and the
free-centre image-plane plateau **survives** the phase-A logsumexp fix — free
centres genuinely need sampler muscle, which is the argument for solved as the
demonstrated default.

## Reading the numbers

`delta = max_ll − truth_ll`. Small positive delta (≲ +10) = the search found
the truth basin (slightly above truth by fitting noise). Large positive delta
= the *model ranking* is broken (a wrong model beats truth — a likelihood
defect, the thing a default must never have). Negative delta = the *search*
failed to reach the truth basin (an optimizer defect, not a likelihood one).

## Findings

### 1. Source-plane weighting: tensor ranks truth first at both tiers (exp-1)

| Tier | free scalar | solved scalar | tensor | verdict |
|---|---|---|---|---|
| galaxy `simple` truth_ll | **−33788.4** | +0.60 | +12.75 | scalar mis-maps the μ=367 image's radial noise — the bias showcase |
| cluster `simple` truth_ll | +40.1 | +36.2 | **+61.3** | tensor gives truth its highest likelihood |

Nautilus recovers each objective's basin (deltas +2.2 to +8.5). The
2026-07-31 isolation result stands: solved+scalar still prefers wrong models;
**the tensor weighting is the ranking fix, the solved centre is the
orthogonal dimensionality win** (−2 params/source, compounding at cluster
scale).

**Caveat (docs prose):** MultiStartProdigy fails the cluster source-plane
basins (tensor delta −11062, solved −14.8) while completing in ~4 min. The
solved *image-plane* cell was the remaining gradient-search hope at cluster
scale; with 331887 landed it fails too (delta **−1723.6**, see 6). At cluster
scale, every objective tested is for Nautilus.

### 2. Image-plane centres: the free plateau survives logsumexp — solved confirmed (exp-2)

Post-#679 (PairAll max-shifted logsumexp, finite at ≳38σ), free-centre
Prodigy still plateaus below truth: delta **−75.7** (was −54.9 pre-fix at 256
starts). The plateau was therefore *not* (only) the `−inf` cliff. Solved
Prodigy converges: delta +1.95 in 118 s. Nautilus handles both (free **+2.38**,
solved +2.85). **Solved centres as the demonstrated image-plane default; free
documented as needing sampler muscle.**

The free number is from the clean re-run (331888, 141 s sampler wall, 14464
posterior samples), which replaced the silent-resume JSON that had reported
+2.27 off 0.8 s of sampling. The conclusion is unchanged — the re-run moved
the delta by +0.10, well inside run-to-run scatter — but the evidence now
rests on a search that actually ran. Its source-plane twin (331889) likewise
reproduced the free-scalar mis-ranking at +33474.5 (was +33476.5), against an
identical truth anchor of −33788.368.

### 3. Pairing discriminator: missing-image arm is decisive for all-to-all (exp-3)

On `simple_missing` (one true image removed from the dataset):

| fit class | truth_ll | delta |
|---|---|---|
| `PairAllSolved` (`image_plane_solved`) | +20.8 | **+1.3** — truth recovered |
| `PairRepeatSolved` (`image_plane_repeat_solved`) | **−183389** | +183402 — catastrophic mis-ranking |

Repeat pairing has no way to leave a model-predicted image unmatched, so a
missing observed image forces a huge spurious residual at truth; the
all-to-all Occam mixture absorbs it. On clean `simple` data the two are
statistically equivalent (deltas +2.85 vs +2.88, walls 147 s vs 163 s) — the
robustness case, not performance, decides the default.

The `simple_extra` arms (one spurious observed position) **did not finish, and
that is the result.** Resubmitted at 8h walls, both timed out again — 331885
(`PairAllSolved`) at 08:00:25 and 331886 (`PairRepeatSolved`) at 08:00:01. The
331885 detail: ~10 h of sampling including the resume, `f_live` = 0.92,
N_eff = 12, logZ = −14238. A spurious position imposes an honest ~−1.4×10⁴
likelihood floor on *every* model, so Nautilus never compresses the live set
and cannot converge. Neither was resubmitted again.

The DNF is **symmetric** — both pairings fail — so exp-3's extra-image arm
does *not* discriminate between the two candidate defaults, and the default
does not rest on it. The discriminating evidence is the missing-image arm
above, which was already in hand. What the extra arm does establish is a
practical warning for the docs: contamination destroys convergence outright
rather than quietly biasing the fit. **Phase C's default swap to
`FitPositionsImagePairAllSolved` therefore stands on the missing-image arm**
(shipped as PyAutoLens#686; verdict posted to
[#678](https://github.com/PyAutoLabs/PyAutoLens/issues/678#issuecomment-5177795748)).

### 4. Posterior width: solved is not overconfident here (exp-4)

The plug-in-profile worry (solved fits carry no marginalization term over the
source centre) does not materialize on `simple`: Nautilus `einstein_radius`
std is **0.0376 solved vs 0.0293 free** — solved error bars are ~28% wider,
not narrower. (The free figure is from the clean 331888 re-run; the
silent-resume JSON had given 0.0273, i.e. ~40% — the margin narrowed, the
direction did not.) The docs' free-centre section needs no overconfidence warning
for this regime; the guide should still note the caveat is theoretical and
untested near caustics.

### 5. Near-caustic stress: tensor holds at 0.95× the tangential caustic (exp-5)

At `near_caustic` (source at 0.95× caustic): tensor delta +1.99, solved
source-plane +1.29, solved image-plane +0.70 — all recover truth; no tensor
breakdown observed at this proximity. The domain-of-validity prose can state
the tensor is safe to at least 0.95× on this system; probing 0.99×+ is left
to the pairing guide if needed.

### 6. Cluster gradient cells: two library defects found and fixed

The `multi_start_prodigy/cluster/image_plane_solved` cell crashed
(`UnexpectedTracerError`) and, once that was fixed, produced all-NaN
gradients. Root causes and fixes (both merged 2026-08-01):

- **PyAutoFit#1441** — attributes derived inside `__init__` from prior
  parameters (`NFWMCRLudlowSph.scale_radius` from free `mass_at_200`) were
  classified into pytree aux data; under a trace they are tracers, and the
  PointSolver `custom_jvp` rule's inner `jax.jvp` received them stale.
  Flatten now promotes JAX-valued attributes to child leaves. Regression:
  `autofit_workspace_test#81`.
- **PyAutoLens#685** — padded solver rows were zeroed to (0,0) = the cluster
  profile centres, where the NFW deflection Jacobian is NaN; reverse mode
  row-sums cotangents, so one padded row poisoned every parameter. Padded
  rows are now anchored at a real solved image and the solve inputs
  sanitized on non-finite rows only.

After both fixes: 8/8 random cluster draws give finite value and gradient
with FOMs byte-identical to the broken build (forward path untouched). The
Nautilus twin (`nautilus/cluster/image_plane_solved`, delta +16.9, 12.6 min)
was unaffected throughout — the defects were invisible to every
non-gradient path, which is why they survived until the first cluster
MCR-halo gradient search.

**End-to-end validation (job 331887, 2026-08-02).** The re-run of the cell
that exposed both defects now completes on the A100 — 832 s wall, finite
`max_log_likelihood`, no crash and no NaN — confirming the merged fixes hold
through a full search and not merely on the 8-draw gradient probe. Its truth
anchor (+14.658) is identical to the Nautilus twin's, which cross-validates
the run.

**But the search still misses the basin: delta −1723.6** (max_ll −1708.9 vs
truth +14.7), against the Nautilus twin's +16.9 on the same objective. By this
document's reading key a negative delta is an *optimizer* defect, not a
likelihood one, so this does not touch the choice of default — the ranking is
sound, MultiStartProdigy just cannot find it. It does close the question
§1's caveat left open: the solved image-plane cell was the last candidate for
a cluster-scale gradient-search recommendation, and it fails alongside the
source-plane cells (−14.8, −11062). **Cluster-scale searches are Nautilus's,
across every objective tested.** Making the gradients finite was a
correctness fix; it did not buy convergence.

Two harness characteristics to read past when comparing the table above:
`likelihood_evals` is 65 for *every* MultiStartProdigy cluster cell (it counts
vmapped batch calls, not per-start steps), and `model_summary.best_fit` is
`AttributeError('ModelInstance' object has no attribute 'galaxies')` on *every*
cluster cell including the Nautilus ones. Both are pre-existing reporting
quirks of the cluster harness, not symptoms of this run.

## Gotchas recorded

- **sacct `COMPLETED` is unreliable for these cells** — job 331843 crashed
  in 1:49 yet reports COMPLETED. Result-JSON existence is the ground truth.
- The two free `simple` Nautilus cells (`image_plane`, `source_plane`) were
  silent PyAutoFit resumes (sampler wall <1 s; identifiers ignore data) of
  same-day post-logsumexp output. **Now re-run clean** (331888/331889,
  2026-08-02, ~141–162 s sampler wall, ~14.4k posterior samples each); both
  reproduced their prior conclusions within scatter, so the resumes had not
  misled — but the shipped numbers are the re-run ones. The general trap
  stands: a PyAutoFit identifier ignores the dataset, so a same-path re-run
  against *changed data* returns instantly with the old posterior. Sub-second
  sampler wall is the tell.
- Dataset noise conventions are deliberate: the tracked galaxy `simple`
  dataset predates a noise-config change (0.05 in file vs 0.005 in config;
  kept for baseline comparability — anchors evaluate against the dataset's
  own noise), and cluster 0.005″ noise models 5 mas centroiding.
