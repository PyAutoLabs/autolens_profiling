# Phase 4 Stage 1 — PositionsLH characterisation (imaging/mge/hst): results

Issue #159. Eager `jax.value_and_grad` transects through `al.PositionsLH`'s
threshold and across the `theta_E` basin, for the `imaging/mge/hst` cell,
per `../PROGRAMME.md` §"Phase 4 — PositionsLH on MGE", Stage 1. Full run:
CPU, fp64 (`JAX_ENABLE_X64=True`), `positions_transects.py` (no `--quick`),
`n=601` for Transects A/C, `1e-4`-spacing fine windows for Transect B,
v2026.8.17.1. A `--quick` run (`n=41`, sparser fine grid) reproduces the
same crossing locations and qualitative behaviour and is the fast
iteration path.

## Classification rule

Three characteristic non-smooth features of `penalty =
factor * max(max_sep - threshold, 0)`, each verified against the REAL
`autolens.analysis.positions.PositionsLH.log_likelihood_penalty_from` /
`analysis.log_likelihood_function`, not a reimplementation:

- **C0 hinge** — penalty value continuous, gradient discontinuous at
  `max_sep == threshold`.
- **Interior plateau** — exact-zero value AND gradient throughout
  `max_sep < threshold`.
- **Argmax-switch kink** — `max_sep = max_i(furthest_distance(i))` over the
  observed positions is itself a max of several smooth functions, so which
  pair of positions realises it can switch discontinuously.

## Setup

- Cell: `imaging/mge/hst`, built via `_setup.build_for_cell(use_jax=True)`,
  positions off at construction; `analysis.positions_likelihood_list` swapped
  per arm.
- Truth positions: `dataset/imaging/hst/positions.json` — 4 quad-image
  positions, **derived once** from the committed `tracer.json` via the
  simulator's own `PointSolver.for_grid(pixel_scale_precision=0.001,
  magnification_threshold=0.1)` recipe (no `positions.json` shipped with this
  dataset previously) and committed alongside this change. Index order:
  `0=(0.271,1.673) 1=(-1.499,0.195) 2=(1.499,-0.195) 3=(-0.271,-1.673)`.
- Anchor: a 15-dim free-parameter vector with the lens mass
  (`centre=(0,0)`, `ell_comps=(0.05263,~0)`, `einstein_radius=1.6`) and
  shear (`gamma_1=gamma_2=0.05`) set to the simulator's truth, the two MGE
  bulges' `centre`/`ell_comps` also truth-mapped (their per-Gaussian
  intensities are linear, solved by the inversion — not part of this
  vector), matched via `unique_prior_paths` pattern lookup rather than a
  hardcoded index table.
- Arms: `{threshold: 0.3 fixed, auto} x {factor: 1e5, 1e8}` = 4 "on" arms,
  plus "off". `auto` = `max(3.0 * max_sep(truth positions, truth tracer),
  0.2)` (SLaM's own `result.positions_threshold_from(factor=3.0,
  minimum_threshold=0.2)` convention). Measured: `max_sep(truth) =
  7.746e-04`, so **`auto` collapses to the `0.2` floor** — truth positions
  trace to ~zero separation through the truth tracer by construction, so
  this collapse is expected and not itself a finding; see Idealisation below.

## Findings

### 1. Threshold hinge — confirmed, and scales exactly linearly with `factor`

Each arm's `max_sep(theta_E)` crosses its threshold at two `theta_E`
(symmetric-ish about the truth `1.6`):

| Arm | Crossings (theta_E) |
|---|---|
| `t0.3_f1e5` / `t0.3_f1e8` | 1.45074, 1.74982 |
| `tauto_f1e5` / `tauto_f1e8` | 1.50059, 1.69997 |

(Threshold alone sets the crossing location — `f1e5`/`f1e8` land at the
identical `theta_E`, as expected.) At each crossing, a tight
one-sided-derivative probe (adjacent fine-grid points, `1e-4` apart) gives:

| Arm | `d(penalty)/d(theta_E)` jump | value-continuity gap (same 2-step window) |
|---|---:|---:|
| `t0.3_f1e5` | 2.00618e+05 | ~20.06 |
| `t0.3_f1e8` | 2.00618e+08 | ~2.006e4 |
| `tauto_f1e5` | 2.00618e+05 | ~20.06 |
| `tauto_f1e8` | 2.00618e+08 | ~2.006e4 |

The gradient jump scales **exactly 1000x** between `f1e5` and `f1e8` (matches
`jump = factor * |d(max_sep)/d(theta_E)|`, local slope ≈2.006 independent of
factor) — internally consistent, not a numerical artefact. The
"value-continuity gap" (penalty value at the two points straddling the
crossing) is ~`factor * slope * grid_step` — i.e. it scales with the probe's
own step size, not a fixed discontinuity: this IS the C0-continuity evidence,
not evidence against it. A worked check away from any crossing (`theta_E =
1.4`, `max_sep = 0.40180`) confirms the exact formula end to end: e.g.
`t0.3_f1e8`: `penalty = 1e8 * (0.40180 - 0.3) = 1.01799e7`, matching the
recorded `10179897.32` to 6 sig. figs, and `logl = logl_off - penalty` exactly.

**Interpretation for [H4.2]**: a gradient optimiser crossing this boundary
sees a genuine step-function force (0 -> `~2e5`-`2e8` in `d(penalty)/d(theta_E)`
within one `1e-4` step) switching on, not a smoothly increasing potential —
consistent with the "lane fling" concern the hazard record documents.

### 2. Interior plateau — confirmed exact-zero on every arm

A window inset from each arm's own crossings (never the interior-vs-exterior
mistake of reusing a fixed `[1.5, 1.7]` for every arm — the tighter `auto`
threshold's own crossings sit at `1.50059`/`1.69997`, right at that window's
edge) shows, for all 4 "on" arms: `penalty_max_abs = 0.0`,
`dpenalty_max_abs = 0.0` **exactly** (bit-identical zero, not "small") across
21 points. A gradient-search lane starting and staying inside the fence gets
literally zero navigational signal from the positions term.

### 3. Argmax-switch kink — present, but does not (here) coincide with the active penalty region

2 distinct argmax-switch locations on the coarse `[0,3]` grid (a 3rd/4th
duplicate report from Transect B's fine re-scan of the same physical
locations):

| Location (theta_E) | Pair before -> after | `max_sep` there | Over any threshold? |
|---|---|---:|---|
| 1.595 -> 1.600 | `[1,2]` -> `[0,3]` | 7.75e-4 (the global minimum, at truth) | No |
| 1.665 -> 1.670 | `[0,3]` -> `[1,2]` | 0.130 -> 0.140 | No (below 0.2 and 0.3) |

The first switch sits exactly at the truth `theta_E=1.6`, where `max_sep` is
at its global minimum (~0) — a near-tie between the two diagonal position
pairs of a near-symmetric quad, not a meaningful kink (penalty is 0
regardless of which pair "wins" there). The second is a genuine kink in the
raw `max_sep(theta_E)` geometry, but at `theta_E~1.665-1.67` it is still
**below every Stage-1 threshold** (0.2, 0.3), so it produces no kink in the
*penalty* a search would actually see for these arms. **This is
dataset/threshold-specific, not a general guarantee** — a tighter threshold
or a different mass model could put an argmax-switch inside the active
region, which is exactly what the `likelihood.positions-penalty.argmax-switch`
hazard record exists to keep checking for (its own synthetic sweep found the
same "0 of N switches over threshold" result independently).

### 4. Gradient-norm ratio (Transect C, `t0.3_f1e8` arm, `ell_comps_0` / `gamma_1` sweeps at fixed `theta_E=1.6`)

| Sweep | x | `|grad_logl|` (15-dim) | `|grad_penalty|` | ratio |
|---|---:|---:|---:|---:|
| `mass_ell_comps_0` | -0.30 | 2.279e8 | 2.278e8 | 0.9993 |
| `mass_ell_comps_0` | -0.15 | 2.216e8 | 2.212e8 | 0.9978 |
| `mass_ell_comps_0` | 0.00 | 1.151e18 | 0 | 0 (interior) |
| `mass_ell_comps_0` | +0.15 | 6.591e5 | 0 | 0 (interior) |
| `mass_ell_comps_0` | +0.30 | 2.403e8 | 2.400e8 | 0.9988 |
| `shear_gamma_1` | -0.20 | 3.390e8 | 3.390e8 | 0.99996 |
| `shear_gamma_1` | -0.10 | 3.394e8 | 3.390e8 | 0.9988 |
| `shear_gamma_1` | 0.00 | 1.499e6 | 0 | 0 (interior) |
| `shear_gamma_1` | +0.10 | 2.161e6 | 0 | 0 (interior) |
| `shear_gamma_1` | +0.20 | 3.392e8 | 3.390e8 | 0.9992 |

Away from the interior, the penalty gradient's norm is **comparable to** (not
overwhelmingly larger than) the full 15-dim base-likelihood gradient's norm —
ratio 0.999-1.0, not >>1 — for these two off-`theta_E` directions at
`factor=1e8`. **Caveat**: the interior `|grad_logl|` at `ell_comps_0=0.0`
(1.15e18) is anomalously large relative to its neighbours (6.59e5 at
`+0.15`) — plausibly a near-degenerate inversion configuration at that exact
anchor point (a `conditioning_floor`-style effect, unrelated to positions);
it does not affect the ratio there (`0/huge = 0`, correctly "no penalty
signal") but the base-likelihood gradient's own scale should not be read as
representative of a typical point. Measured for one representative arm
(`t0.3_f1e8`) only — a full 15-dim `jax.grad` of the real MGE+inversion
likelihood is its own ~tens-of-seconds XLA compile per `(idx, arm)`; the
threshold/factor sensitivity this would otherwise re-demonstrate is already
Transects A/B's whole point.

## Caveats (read before citing any number above)

- **Truth-positions idealisation.** Positions come from the simulator's own
  `PointSolver` solve of the truth tracer, and (for `auto`) the
  threshold-resolution tracer is also the truth tracer — not positions/a
  tracer re-solved from a completed search's maximum-likelihood model (the
  SLaM chained-fit workflow this mirrors). This is why `auto` collapses to
  its `0.2` floor here: real-data / real-search positions would not
  generally do so.
- **Single cell.** `imaging/mge/hst` only — Stage 1 does not touch
  pixelization/Delaunay, interferometer, point_source, cluster, or datacube
  (all raise `NotImplementedError` for `SEARCHES_POSITIONS=on`, by design —
  see `_setup.py`).
- **One anchor.** All transects vary one (or, for the gradient-norm ratio,
  effectively one) free parameter at a time from a single truth-mapped
  anchor. Behaviour along other directions, or from a different (e.g.
  random-start) anchor, is not measured here.
- **Compute-cost lesson, not a science finding**: a single unchunked
  `jax.vmap` over the full Transect-A grid (601 points) OOMs (~30 GB
  requested) — the real MGE+inversion likelihood is not free to batch
  arbitrarily wide. `positions_transects.py` chunks every vmap'd evaluation
  at batch=64 (matching `vram.vmap_batch_for("imaging","mge","hst")`, this
  repo's own A100-probed cap for this exact cell) via a `chunked_call`
  helper, fixed-padding every chunk so each `(idx, arm)` pair also only
  needs ONE XLA compile regardless of how many different grid shapes are
  thrown at it across Transects A/B/C.
- **This is Stage 1 (characterise), not Stage 2 (campaign).** [H4.1] (does
  the penalty raise Prodigy's `theta_E=0`-basin `p_hit`?) and [H4.3]
  (is the penalty exactly zero at the converged posterior in practice, so
  there's no double-count?) are NOT addressed here — they need an actual
  multi-seed search campaign (Stage 2 per `../PROGRAMME.md`).

## Artifacts

`transects/transect_{a,b,c}.json` (full numeric arrays behind every table
above) + matching `.png` plots. Reproduce with:

```bash
JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py --quick   # ~1-2 min, sanity check
JAX_ENABLE_X64=True python scripts/misc/searches/positions_transects.py           # full A/B, ~20 min CPU
```

## Next

- Stage 2 (campaign): matched Nautilus / NSS / Prodigy trio x >=5 seeds x
  {positions on, off} at `imaging/mge/hst`, per `../PROGRAMME.md`'s Phase 4
  Stage 2 spec — this Stage-1 record is its prerequisite, not a substitute.
- Extend `_positions_likelihood_list_for` to pixelization/Delaunay /
  interferometer once Stage 2 needs them (currently `NotImplementedError`
  by design).
- The `ell_comps_0=0.0` anomalous base-likelihood gradient norm (§4 caveat)
  is unexplained; worth a standalone look if Stage 2's gradient search
  starts landing near that configuration.

---

# Phase 4 Stage 2 — matched trio +/- PositionsLH (imaging/mge/hst): results

RAL jobs 340114 (Nautilus, 10 arms) and 340115 (Prodigy n256, 5 arms),
harvested 2026-08-26. All 15 COMPLETED, version stamp 2026.8.17.1, walls
plausible, no resumed fits. The NSS arm was dropped per Gate A.

Positions arms use `threshold_mode=fixed`, `threshold_value=0.3`,
`factor=1e8`, `source=simulator_truth_positions` — **IDEALISED**: the
positions are the simulator's own truth, not solved from a completed
search's max-likelihood model as the SLaM chained-fit convention would.

## Nautilus — PositionsLH is a no-op

| seed | logZ (off) | logZ (on) | maxL (off) | maxL (on) | wall off | wall on |
|-----:|-----------:|----------:|-----------:|----------:|---------:|--------:|
| 0 | 31690.50 | 31690.50 | 31786.69 | 31786.46 | 774 s | 764 s |
| 1 | 31690.50 | 31690.48 | 31786.77 | 31786.76 | 734 s | 735 s |
| 2 | 31690.50 | 31690.49 | 31786.57 | 31786.49 | 742 s | 720 s |
| 3 | 31690.50 | 31690.50 | 31786.61 | 31786.47 | 716 s | 769 s |
| 4 | 31690.48 | 31690.49 | 31786.89 | 31786.73 | 758 s | 810 s |

logZ agrees to 0.02 nats and the recovered mode is unchanged; Kish ESS
4,209-4,315 (off) vs 4,262-4,623 (on). Nautilus already recovers the
dominant mode 5/5 without positions, so the penalty has no failure mode left
to prevent. **H4.1 as applied to Nautilus on MGE: positions do not change
the evidence, the mode, or reliability.**

**CORRECTION (2026-08-27, re-read from the raw JSONs — DECISIONS.md
2026-08-27 W2):** "no-op" overstates it, in two places. (1) Wall is
**-3.0 to +7.3%**, not "within +/-3%". (2) **maxL is LOWER with positions
in 5/5 seeds** (mean -0.126 nats, paired t = -3.45, p ~ 0.026): the penalty
demonstrably fires at the maximum-likelihood point, even though it leaves
logZ and the mode alone. No penalty diagnostic exists in the nested JSONs,
so its size at the reported maximum cannot be read off the artifact. The
heading above is kept for reference stability; read it as "inert on logZ and
the mode", not "inert".

## Prodigy n=256 — PositionsLH degrades it, 5/5 to 1/5

| seed | maxL (positions off) | maxL (positions on) | recovered model (on) |
|-----:|---------------------:|--------------------:|---------------------|
| 0 | 31787.906 | 31764.30 | r_E 1.5995 |
| 1 | 31787.913 | 31785.55 | r_E 1.5997 |
| 2 | 31787.904 | **31787.38** | r_E 1.5997 |
| 3 | 31787.906 | 31702.48 | r_E 1.5997, shear 0.0537 (truth ~0.0485/0.0495) |
| 4 | 31787.919 | **16727.52** | r_E 1.6023, centre (-0.028, -0.017) |

Positions-off is 5/5 within a 0.04-nat band. Positions-on is one clean hit
(seed 2), two near-misses, one -85 nat arm and one catastrophic arm.

**CORRECTION (2026-08-27 — DECISIONS.md 2026-08-27 W2):** scored under
**Phase 3's own coded rule** (run SUCCESS = >=1 lane at >= 31784.782, the
truth bar minus the Phase-1 2-nat tolerance) this arm is **2/5, not 1/5** —
seed 1 has a hit lane at 31785.464. The 1/5 headline used the stricter,
undeclared 0.04-nat band around the positions-off plateau. If that band is
the rule this campaign wants, it must be declared as such and applied to
every arm; the direction of the finding stands under either bar. All rates
below and in Stage 3 are quoted under the coded rule.

This is **not** purely a scoring artifact of the penalty riding on the
reported likelihood. Seeds 0/1/3 land at r_E ~ 1.5997 — the right basin — and
still fall short of the positions-off band, and seed 4 converges to a
genuinely displaced centre rather than to truth-plus-penalty.

LEADING HYPOTHESIS (untested): with idealised truth positions the hinge
should be inert at the true model, so the fact that it is not implies the
0.3" threshold is tighter than the PointSolver's achievable image-position
precision for the recovered model. The hinge then stays live in the
neighbourhood of truth, and a fixed-step gradient searcher — which cannot
line-search its way around a steep discontinuity — is exactly the method that
breaks on it. Nautilus, sampling by rejection, never feels it.

A ~1 GPU-h diagnostic separates "PositionsLH is hostile to gradient search"
from "this hinge is mis-scaled": re-run the Prodigy arms at
`threshold_mode=auto` (t=0.2) and at `factor=1e5`. Filed as PyAutoMind
`draft/bug/autolens_profiling/` follow-up work; NOT run here.

## CAVEAT — the two Prodigy arms are not on the same eval counter

The positions-OFF n256 arms are schema v1 (no `schema_version` key,
`likelihood_evals=257`); the positions-ON arms are schema v2
(reject-inclusive, 32,000-247,808 evals). `max_log_likelihood` is comparable
between them and every claim above rests only on that. `likelihood_evals`,
`time_per_eval_ms` and anything derived from them are **not** comparable, and
no wall or throughput claim is made for the Prodigy pair. The positions-off
baseline comes from the earlier CP-3 wave; nothing in either artifact flags
the mismatch. Filed as a guard task in PyAutoMind.

## Artifacts (Stage 2)

- `results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64_seed{0-4}.json`
  and `..._seed{0-4}_pos_t0.3_f1e8.json`
- `results/searches/multi_start_prodigy_autoconv/imaging/mge/hst/
  hpc_hpc_a100_fp64_n256_seed{0-4}_pos_t0.3_f1e8.json`

## Next (Stage 2)

- ~~The threshold/factor diagnostic above, before any Gate B pt 2 reading.~~
  **Run as RAL 341892 and harvested 2026-08-27 — Stage 3 below.** The
  LEADING HYPOTHESIS above is **falsified**: the tighter (0.2") threshold
  fails hardest, and loosening the *factor* at the same 0.3" threshold
  restores 5/5. The mechanism is penalty stiffness, not threshold tightness.
- ~~Gate B pt 2 cannot be called on this evidence.~~ **Gate B pt 2 CALLED
  2026-08-27** on Stage 2 + Stage 3 together (DECISIONS.md).

---

# Phase 4 Stage 3 — threshold vs stiffness diagnostic (job 341892)

RAL array job **341892** (10 tasks: 2 arms x 5 seeds), run 2026-08-27,
harvested 2026-08-27. Prodigy `n_starts=256`, `n_steps=3000`,
`clipper=prior_box`, no scaler, auto-convergence on, A100 fp64, version
stamp 2026.8.17.1. Arms: **A** `threshold_mode=auto` (+ factor 1e8) and
**B** `threshold=0.3` fixed with **factor 1e5**. Both are read against the
Stage 2 `t0.3_f1e8` arm and the CP-3 positions-off arm at the same seeds.

**Task 0 (arm A, seed 0) is INVALID and excluded.** Its log prints
`Resuming MultiStartGradient search (previous samples found).`, it finished
in 68 s, and its artifact records `total_steps: null` with zero per-lane
entries. It never prints `Fit Already Completed`, so the single-string
resume check the ledger used would have passed it — see the provenance rule
added to `../PROGRAMME.md` §3. Arm A is therefore **n = 4 fresh seeds**.

## The hypothesis is falsified — "auto" is the TIGHTER arm

Stage 2's leading hypothesis was that 0.3" is tighter than the achievable
image-position precision, leaving the hinge live near truth. The log
resolves arm A's threshold as `threshold=0.200000  factor=1e+08`
(`output.341892_1.out`) — as Stage 1 predicted, `auto` collapses to its 0.2
floor on truth positions. Arm A is **tighter** than the Stage 2 arm and
fails **harder**; arm B keeps the same 0.3" threshold, drops only the
factor, and recovers the positions-off answer 5/5. The controlling variable
is **penalty stiffness, not threshold tightness**.

| Arm | Hits (±2 nat of 31786.78) | max logL range | Steps to stop | Wall s | Constrained lane-step rate | Median step scale |
|---|---|---|---|---|---|---|
| positions off (CP-3) | **5/5** | 31787.906–.919 | 175–183 | 172–225 | 15–18 % | — |
| `t0.3 f1e5` (new) | **5/5** | 31787.907–.913 | 173–180 | 163–297 | 38–43 % | 0.14–0.16 |
| `t0.3 f1e8` (Stage 2) | **2/5** (1/5 on the undeclared 0.04-nat band) | 16727 … 31787.38 | 125–968 | 173–553 | 44–53 % | — |
| `tauto0.2 f1e8` (new) | **0/4** (+1 invalid resume) | 27913 … 31779.30 | 114–848 | 148–336 | 41–56 % | 0.21–0.22 |

Hits are scored under Phase 3's coded rule (>=1 lane at >= 31784.782).
Constrained lane-step rate = `n_constrained_lane_steps / (steps x 256)`;
median step scale is Prodigy's own `d` median on the final step line of each
task's log (the artifact does not serialise `opt_state` — see the known gap
in `../phase_08_regularization/RESULTS.md`).

Per-seed detail for the two new arms (from the artifacts):

| arm | seed | max logL | steps | sampler wall | stop_reason | constrained | median d |
|---|---:|---:|---:|---:|---|---:|---:|
| `t0.3 f1e5` | 0 | 31787.913 | 177 | 165 s | converged | 39.3 % | 0.140 |
| | 1 | 31787.907 | 177 | 164 s | converged | 41.7 % | 0.149 |
| | 2 | 31787.912 | 174 | 163 s | converged | 38.4 % | 0.152 |
| | 3 | 31787.911 | 173 | 163 s | converged | 37.6 % | 0.160 |
| | 4 | 31787.912 | 180 | 297 s | converged | 42.7 % | 0.152 |
| `tauto0.2 f1e8` | 0 | *(invalid resume, 68 s)* | — | — | — | — | — |
| | 1 | 31736.949 | 809 | 323 s | converged | 55.3 % | 0.217 |
| | 2 | 31779.299 | 848 | 336 s | converged | 47.5 % | 0.215 |
| | 3 | 27913.511 | 114 | 148 s | converged | 40.8 % | 0.212 |
| | 4 | 31660.334 | 612 | 316 s | converged | 55.9 % | 0.217 |

## The f1e5 winner is the positions-off answer

Every arm-B seed recovers `r_E = 1.5997`, `centre = (0.000, 0.000)`,
`shear ~ (0.0485, 0.0496)` — the positions-off result to 3 d.p. in both
likelihood and parameters, at 173–180 steps against positions-off's 175–183
and at the same wall. **The penalty is therefore ~0 at the recovered
model**: the winning model sits inside the fence, exactly as Stage 1's
interior-plateau finding predicts. (Inferred from parity, not measured —
schema v2 has no `penalty_at_best` field; adding one is owed.)

## Mechanism: transit damage, not a moved optimum

A factor of 1e8 against a 3x10^4-scale log-likelihood does not move the
optimum — it wrecks the route to it. The damage is visible in three
coupled readouts, all of which scale with the *factor*, not the threshold:

1. **Step scale inflates.** Prodigy's median `d` runs 0.21–0.22 on the
   1e8 arm against 0.14–0.16 on the 1e5 arm — a learning-rate-free stepper
   reads the hinge's slope as curvature and takes larger steps.
2. **Lanes are thrown into the non-physical prior corner and pinned.**
   Fraction of lane *best* points with |`ell_comps`| >= 1 (any of the three
   pairs): positions-off **216/1280 = 16.9 %**, `t0.3 f1e5` 280/1280 =
   21.9 %, `t0.3 f1e8` **368/1280 = 28.7 %**, `tauto0.2 f1e8` 315/1024 =
   30.8 %. At the *final* point: **25.9 %** off vs **52.7 %** on (both 1e8
   and 1e5 arms) — positions-on roughly doubles the share of lanes that end
   outside the ellipticity unit disk. The constrained-lane-step rate moves
   the same way (15–18 % -> 38–43 % -> 44–53 % -> 41–56 %).
3. **`alive` barely moves.** 241–251 of 256 lanes are alive at the final
   step on both 1e8 and 1e5 arms — the lanes are not dying, they are being
   parked in a corner the box prior admits and the clipper faithfully holds
   them in.

This is the concrete channel for the Stage 2 degradation, and it is the
same population as the Phase 8B crash (`ell_comps` magnitudes 1.03–1.414 at
the best point). See `../phase_03_prodigy_reliability/RESULTS.md`
"Non-physical ellipticity lanes" and PROGRAMME §9b W10.

## First demonstration of the converged-on-a-wrong-plateau limitation

Phase 3 recorded that the auto-convergence detector cannot distinguish a
wrong-basin plateau from convergence; every run there stopped `converged`
but also *was* in the right basin often enough that the failure was
inferred, not shown. Two runs here show it outright: `tauto0.2 f1e8` seed 3
stops `converged` at step **114/3000** with max logL **27913.51** (−3,873
nats), and Stage 2's `t0.3 f1e8` seed 4 stops `converged` at step
**125/3000** at **16727.52** (−15,059 nats). Both are stationary on a
plateau the penalty built, and both are reported as successful
terminations. `stop_reason` is not evidence of a correct answer;
reliability still has to come from `n_starts` (PROGRAMME §3, "Termination
is a benchmark metric").

## Verdict — Gate B part 2 CALLED (human, 2026-08-27)

**PositionsLH is not intrinsically hostile to gradient MAP search on MGE;
the pre-registered factor 1e8 was mis-scaled for a fixed-step searcher. At
factor 1e5, Prodigy(n=256, prior_box, autoconv) is 5/5 with positions on,
at parity with positions-off in likelihood, parameters, steps and wall.
Gate B part 1 extends to positions-on at factor <= 1e5; 1e8 is rejected for
gradient search.**

Six caveats ride with the call (full text in `../DECISIONS.md` 2026-08-27):

1. **Idealised positions** — simulator truth, and for `auto` the
   threshold-resolution tracer is the truth tracer, not a completed
   search's max-likelihood model (the SLaM convention).
2. **One cell, five seeds** — `imaging/mge/hst`, A100 fp64. Wilson-95 lower
   bound at 5/5 is **0.57**: this does **not** re-establish the >=99 %
   reliability Gate B pt 1 demonstrated positions-off at n=256.
3. **1e5 is shown safe, not calibrated** — nothing ran between 1e5 and 1e8,
   and SLaM's own `factor=3` convention is untested here.
4. **Nautilus is unaffected either way** (Stage 2 + its 2026-08-27
   correction): no evidence or mode change, a small systematic maxL
   penalty, no reliability consequence.
5. **No `penalty_at_best` field** in schema v2 — the "penalty ~0 at the
   winner" claim is inferred from parity, not measured.
6. **Provenance defect, fixed in this PR** — see below.

## Defect found and fixed: the positions block did not discriminate targets

All three positions-on arms hashed to the **same** `target_id`
(`sha256:bf3d096fda76`), and every positions-on JSON recorded
`threshold 0.3 / factor 1e8` in its `positions` block regardless of what
actually ran: `_targets.py`'s `_positions_block` built the block from module
defaults instead of the resolved `SEARCHES_POSITIONS_*` setup. A
target_class-3 change was therefore invisible to the target hash — exactly
the comparability guarantee `../PROGRAMME.md` §3 says the PositionsLH phase
depends on. **Fixed in this PR**: `_positions_block` now takes the resolved
positions block, and affected rows were **re-derived** (not hand-edited) with
`scripts/misc/searches/restamp_target_block.py`, so the three Phase-4 Prodigy
arms now hash distinctly — `sha256:bf3d096fda76` (t0.3 f1e8),
`sha256:cd522872a7ed` (t0.3 f1e5), `sha256:6b93f0e52ecd` (tauto0.2 f1e8).
A row whose *recorded* `positions` block was itself written from the
defaulted path cannot be re-derived from its own artifact and must be read
against the `config_name` suffix rather than the hash.

## Artifacts (Stage 3)

- `results/searches/multi_start_prodigy_autoconv/imaging/mge/hst/hpc_hpc_a100_fp64_n256_seed{0-4}_pos_t0.3_f1e5.json`
- `results/searches/multi_start_prodigy_autoconv/imaging/mge/hst/hpc_hpc_a100_fp64_n256_seed{1-4}_pos_tauto0.2_f1e8.json`
  (seed 0 harvested but marked INVALID — silent resume)
- Job logs `output.341892_{0-9}.out` (resolved threshold line, per-step
  `alive` / `constrained` / `d` readouts).
