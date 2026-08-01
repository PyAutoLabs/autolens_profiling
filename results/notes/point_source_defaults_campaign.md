# Point-source defaults campaign — A100 truth-anchored evidence (#678 phase B)

**Issue:** [PyAutoLens#678](https://github.com/PyAutoLabs/PyAutoLens/issues/678)
**Branch:** `feature/point-source-defaults-campaign`
**Status:** 22/25 cells complete (A100 fp64, RAL). In flight: the two
`simple_extra` exp-3 arms (8h-wall resubmits, jobs 331885/331886) and the
`multi_start_prodigy/cluster/image_plane_solved` rerun (job 331887) following
the two gradient bug fixes it exposed (PyAutoFit#1441, PyAutoLens#685).
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
gradient searches are not yet competitive on **cluster source-plane**
objectives (use Nautilus there), and the free-centre image-plane plateau
**survives** the phase-A logsumexp fix — free centres genuinely need sampler
muscle, which is the argument for solved as the demonstrated default.

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
basins (tensor delta −11062, solved −14.8) while completing in ~4 min — at
cluster scale the source-plane objectives are for Nautilus; the
gradient-search story at cluster scale is the solved image-plane cell (see 6).

### 2. Image-plane centres: the free plateau survives logsumexp — solved confirmed (exp-2)

Post-#679 (PairAll max-shifted logsumexp, finite at ≳38σ), free-centre
Prodigy still plateaus below truth: delta **−75.7** (was −54.9 pre-fix at 256
starts). The plateau was therefore *not* (only) the `−inf` cliff. Solved
Prodigy converges: delta +1.95 in 118 s. Nautilus handles both (free +2.27,
solved +2.85). **Solved centres as the demonstrated image-plane default; free
documented as needing sampler muscle.**

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

The `simple_extra` arms (one spurious observed position) both hit the 2h wall
— itself a robustness observation (the solved searches grind on
spurious-position data) — and are rerunning at 8h walls. **Phase C's default
swap to `FitPositionsImagePairAllSolved` is gated only on this arm.**

### 4. Posterior width: solved is not overconfident here (exp-4)

The plug-in-profile worry (solved fits carry no marginalization term over the
source centre) does not materialize on `simple`: Nautilus `einstein_radius`
std is **0.0376 solved vs 0.0273 free** — solved error bars are ~40% wider,
not narrower. The docs' free-centre section needs no overconfidence warning
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

## Gotchas recorded

- **sacct `COMPLETED` is unreliable for these cells** — job 331843 crashed
  in 1:49 yet reports COMPLETED. Result-JSON existence is the ground truth.
- The two free `simple` Nautilus cells (`image_plane`, `source_plane`) are
  silent PyAutoFit resumes (sampler wall <1 s; identifiers ignore data) of
  same-day post-logsumexp output — acceptable, wipe-and-rerun optional.
- Dataset noise conventions are deliberate: the tracked galaxy `simple`
  dataset predates a noise-config change (0.05 in file vs 0.005 in config;
  kept for baseline comparability — anchors evaluate against the dataset's
  own noise), and cluster 0.005″ noise models 5 mas centroiding.
