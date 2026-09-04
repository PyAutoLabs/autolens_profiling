# gradient-slam-baseline — the ledger

**Born 2026-09-04** from the retired `jax-inference-profiling` programme
(`results/notes/inference/PROGRAMME.md`, frozen the same day). Cortex epic
`gradient-slam-baseline`; the Cortex half of the state is
`PyAutoCortex/phases/inference_programme/` (`Epic: gradient-slam-baseline`), the
rulings of record are `PyAutoCortex/rulings/`. This file is the human-first
commentary: what the question is, what we already know, what the target is, and
what each phase must show.

## The question

**Dropped into a SLaM `mass[1]` search, would a gradient search beat Nautilus?**
The old programme asked a much bigger question — every sampler, every mesh, every
smoothness lever — and lost the thread across 19 phases and 13 rulings. This one
asks a single question at the shape a real pipeline uses. A SLaM `mass[1]` search
inherits its lens light and its source from the stages before it and varies the
mass model alone; so the baseline target here fixes the lens light to the
simulator truth, fixes the mesh and regularization at values a certified
`source_pix` run actually chose, and leaves **only mass + shear free** — 7
parameters. If a gradient search cannot win there, it does not win in a pipeline;
if it does, the win is one a SLaM stage could actually collect. Every phase is
one job, one witness, one ruling, in the loop with the human.

## Inherited evidence (all citable)

Everything below survived the retirement and can be cited without a new run. The
walls are `sampler_wall_s` from each row's own artifact.

| What | Result | Where it stands |
|---|---|---|
| **MGE Nautilus** — `mge_pos_fp64`, `pos_tauto0.2_f1e8`, n_live 400, A100 fp64 | max logL **31,786.797**, **96,704** evals, **939.1 s** sampler wall (9.71 ms/eval) | certified baseline, R-20260901-01 — `results/baselines/InferenceRefs_v1/mge_pos_fp64/` |
| **MGE Prodigy** — `MultiStartProdigy` autoconv n256, `pos_t0.3_f1e5`, seeds 0–4 | **5/5 hits**, **163–297 s** — 3.1–5.8× under the Nautilus wall | legacy but accepted and citable under the MGE reuse rule, R-20260902-09 / R-20260902-10 (`output/legacy/`) |
| **`slam_source_pix_pos`** — all-free mesh + reg + MGE light, Nautilus | max logL **31,547.240**, **93,700** evals, **3,982.8 s** | certified baseline, R-20260902-01 — but see "what this gap is not" below |
| **`pixelization_pos`** — fixed reg coefficient, Nautilus | max logL **30,941.226**, **48,592** evals, **2,617.9 s** | certified baseline, R-20260904-01 |
| **A100 fp64 forward cost** | **6.09 ms** (MGE) vs **52.7 ms** (39×39 pixelization) — the mesh likelihood is ~8.7× the MGE one | likelihood-runtime rows, `results/` |
| **vmap gain** | pixelization only **1.6×**, MGE **15.8×** — batching buys much less on a mesh | likelihood-runtime rows, `results/` |
| **CPU `value_and_grad` anomaly** | `value_and_grad` ≈ **17× the forward eval** on the rectangular kernel-CDF cell — the only rectangular-mesh gradient cost datum that exists, and it is a **CPU** one, **never re-measured on an A100** | autolens_workspace_developer#117, `searches_minimal/pix_prodigy_findings.md` |
| **Positions factor is engine-split** | On MGE, positions at factor **1e8** give Prodigy **0/4**; at **1e5** they give **5/5**. Nautilus is unaffected either way (Δ logZ ≤ 0.022 nats) | Gate B pt 2, R-20260902-10 |

**What the `slam_source_pix_pos` − `pixelization_pos` gap is not.** 31,547 vs
30,941 is **606 nats for 4 extra free parameters** — the `slam_source_pix*`
targets free the lens light, the mesh weights *and* every regularization
parameter at once, which no SLaM stage does. It is not a pipeline result and is
not read as one here. That withdrawal is why this epic exists.

## The `mass_pix` target

One cell: `imaging/mass_pix/hst`. Free parameters: **Isothermal (5) + ExternalShear
(2) = 7**. Everything else is an instance.

**Lens light — fixed to truth.** The `Sersic` instance in
`dataset/imaging/hst/tracer.json` (galaxy 0, `bulge`): `intensity` 2.0,
`effective_radius` 0.6, `sersic_index` 3.0, `ell_comps` (0.0526, 0.0), `centre`
(0.0, 0.0). Loaded from the file, never retyped — the file is the provenance.

**Source — fixed at the certified `slam_source_pix_pos` maximum likelihood.**
Provenance: the certified `slam_source_pix_pos_fp64` reference (R-20260902-01,
RAL job **342091** task **t4**, run identifier
**`4323a2ffcb3e50a71f229e46032d9e95`**):

| Component | Value |
|---|---|
| mesh | `al.mesh.RectangularRTUAdaptImage`, `shape=(39, 39)` |
| `weight_power` | `0.001` |
| `weight_floor` | `0.248` |
| regularization | `al.reg.Adapt` |
| `inner_coefficient` | `0.140` |
| `outer_coefficient` | `226.169` |
| `signal_scale` | `0.004` |

**Positions.** On, threshold `auto` resolving to **0.2**, as every certified mesh
reference. The **factor is engine-split**: Nautilus reference arms at **1e8** (the
`InferenceRefs_v1` convention, and Nautilus is measured inert to it), gradient
arms at **1e5** (Gate B pt 2 — 1e8 destroys a gradient search on MGE). One
Nautilus **1e5** arm is run alongside so the gradient result has a like-for-like
bar as well as the reference one.

## Inherited rules (binding, carried over)

1. **No `positions.info` ⇒ not citable.** A mesh / pixelization run dir without
   `positions.info` is unreliable and cannot be used or cited anywhere
   (R-20260901-03, generalised by R-20260902-01).
2. **WALL-BASIS is measured on the cell itself**, not transferred from another
   cell — every submit script carries the `WALL-BASIS` block and names the
   measurement it came from.
3. **Engine-split positions factor** — Nautilus 1e8, gradient 1e5 (R-20260902-10);
   whether MGE's 1e5 transfers to a mesh likelihood scale is *unmeasured*, and is
   the first thing phase 3 has to watch.
4. **`output/legacy_wrong/` is never cited** except as failure-mode
   documentation; `output/legacy/` is reusable MGE evidence under the MGE reuse
   rule.
5. **MGE reuse rule** — a quarantined MGE run may be cited and returned to the
   active tree by ruling, without a rerun (R-20260901-01).

## Phases

Cortex phases under `PyAutoCortex/phases/inference_programme/`, all with
`Epic: gradient-slam-baseline`. Each is one job and one ruling.

| Cortex phase | Slug | State | What it settles |
|---|---|---|---|
| **20** | `mass_pix_gradient_cost_probe` | `gated` on autolens_profiling#218 | Forward vs `value_and_grad` ms/eval on the `mass_pix` cell, strict FD on all 7 parameters, compile time. **Is the jvp/forward ratio sane (≲4×), or does the 17× CPU anomaly reproduce on an A100?** If it reproduces, a Mind bug comes before phase 22. |
| **21** | `nautilus_mass_pix_baseline` | `planned`, ready when 20 rules | Two Nautilus runs, n_live 300, refs settings: `pos_tauto0.2_f1e8` and `pos_tauto0.2_f1e5`. The bar the gradient search has to beat, plus the like-for-like 1e5 arm. |
| **22** | `prodigy_mass_pix` | `planned`, ready when 21 rules | `MultiStartProdigy` autoconv, n_starts 16, batch_size 4, n_steps ≤ 3000, `pos_tauto0.2_f1e5`, seeds 0–4. **The headline ruling**: p_hit and wall against phase 21. |
| **23** | `likelihood_term_levers` | `planned`, gated on 22's ruling | Arms **only if** 22 shows NaN lanes or misses: `SEARCHES_LOG_DET_METHOD=slogdet` vs `cholesky` (folds autolens_profiling#166), and relative jitter on `reg.Adapt` (today only the kernel schemes carry `jitter_relative`; `constant.py:53` / `adapt.py` still take the absolute 1e-8 lift — a PyAutoArray task if it triggers). |

**The development leg** — the `mass_pix` target, its drivers and the probe script
— is a Mind prompt, not a Cortex phase:
`PyAutoMind/draft/feature/autolens_profiling/gradient_slam_mass_pix_target.md`,
filed 2026-09-04 with its issue **autolens_profiling#218** opened at the same
moment as phase 20's gate ref (Cortex schema decision 55: a `Gates:` line holds
GitHub refs only). The prompt stays in `draft/` — an open issue there means
"there is a ref", not "the work is in flight" — and `/start_dev` reuses that
issue rather than opening a second.

## Work-up queue (not filed)

Filed **only when phase 22 rules**, and only if the headline answer earns them:

- **The reg-free variant** — the same target with the regularization coefficients
  free, to price what fixing them bought.
- **The MGE-at-ML lens-light variant** — lens light fixed at the certified
  `slam_source_pix_pos` MGE maximum-likelihood Basis instead of the Sersic truth,
  which is what a real SLaM `mass[1]` actually inherits.

Neither is a phase, an issue or a prompt today. They are written down so they are
not re-derived, and nothing more.
