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

logZ agrees to 0.02 nats, maxL to well inside the seed spread, wall within
+/-3%, Kish ESS 4,209-4,315 (off) vs 4,262-4,623 (on). Nautilus already
recovers the dominant mode 5/5 without positions, so the penalty has no
failure mode left to prevent. **H4.1 as applied to Nautilus on MGE: positions
neither help nor hurt.**

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

- The threshold/factor diagnostic above, before any Gate B pt 2 reading.
- Gate B pt 2 cannot be called on this evidence: the Nautilus half says
  positions are inert, the Prodigy half says they are harmful, and the
  Prodigy half is confounded by a threshold that may simply be mis-set.
