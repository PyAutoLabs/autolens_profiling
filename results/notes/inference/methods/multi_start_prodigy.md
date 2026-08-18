# MultiStartProdigy — evidence card

Living method card (template: `../PROGRAMME.md` §6). Seeded 2026-08-18.
Phase 3 (per-start hit-probability measurement) and Phase 4 (PositionsLH) are
the pending experiments that decide this method's classification — Gate B.

## IDENTITY

Multi-start gradient MAP optimizer (`af.MultiStartProdigy`; Prodigy = a
learning-rate-free Adam-family stepper). Local per lane; global only through
start coverage. Gradient-required, JAX/GPU-native (vmapped lanes). Does not
handle multimodality beyond start coverage — no evidence, no posterior.

## EVIDENCE

- Dimensional regimes: 15D (mge), point-source/cluster cells (7–20D).
- Datasets/models: imaging/mge/hst (laptop RTX 2060 fp64, 2 seeds —
  `../../clipper_campaign/RESULTS.md`,
  `../../clipper_campaign/multi_start_prodigy_imaging_mge_hst.json`);
  point_source + cluster matrices, truth-anchored, A100 fp64
  (`results/searches/multi_start_prodigy/**`,
  `../../point_source_defaults_campaign.md`).
- Initialization modes tested: broad priors only. Warm-start / initialized
  mode is the Gate-B fallback classification, not yet plumbed.

## STRENGTHS

- When it hits the truth basin it beats the nested truth bar: mge seed 0
  reached 31,787.929 (+1.15 nats over Nautilus) in 16×3000 lane-steps
  (`../PROGRAMME.md` §1.2).
- Fixed-step first-order robustness on the NNLS-kinked objective, where every
  line-search/quasi-Newton method categorically fails (`../PROGRAMME.md`
  §1.2, wsdev #95/#97 findings).

## WEAKNESSES

- **Catastrophic basin failure is the discriminator**: mge seed 1 landed
  171,272 nats away in the degenerate θ_E=0 "no lens" basin on the U(0,8)
  wall — same config, same budget (`../PROGRAMME.md` §1.2; diagnosis PR#133).
  Reframed as per-start hit probability: plausibly binomial luck at 16 lanes
  (`../PROGRAMME.md` §2.3; H3.2).
- Not competitive on any cluster-scale objective — "use Nautilus at cluster
  scale" (`../../point_source_defaults_campaign.md`; two gradient bug fixes
  PyAutoFit#1441 / PyAutoLens#685 came out of those runs).
- Per-lane *best* positions are not preserved (only final) — blocks reliable
  basin classification; small PyAutoFit change is the Phase 3 pre-req
  (`../PROGRAMME.md` §2.3, §7).

## CONFIGURATION

- Best-supported config (Phase 3 baseline): clip=prior_box (hygiene — kills
  lane deaths at zero accuracy cost, does not change the winning basin),
  no momentum reset (dropped), no per-parameter scaler (falsified, 3 of 4
  pre-registered conditions fired), auto-convergence ON with stop_reason
  accounting (`../../clipper_campaign/RESULTS.md`; autolens_profiling
  #128/#131, PRs #130/#132/#133).
- Knobs users should not touch: per-parameter step scaling (falsified);
  further wall-handling variants without a named mechanism (`../PROGRAMME.md`
  §8 risk table).
- Sensitivity: n_starts is the reliability axis — Phase 3 scans {16, 64, 256}
  × ≥5 seeds.

## TERMINATION

- Rule: global-best-FOM plateau (window 50, rtol 1e-4, atol 1e-3, min 100
  steps); `n_steps` is a ceiling; `stop_reason` persisted.
- Limits: **disabled when `resurrect=True`** (the pixelized regime) and
  cannot distinguish a wrong-basin plateau from convergence — the Phase 3
  confusion matrix is the required diagnostic (`../PROGRAMME.md` §1.2, §3).
- Waste (steps-after-best): not yet measured — schema-v2 field.

## HAZARDS

- θ_E=0 degenerate basin on the prior wall (mge) — the standing failure
  mode; PositionsLH fencing is the Phase 4 hypothesis H4.1
  (`../PROGRAMME.md` §2.4).
- Lane deaths are prior-support events, not likelihood NaNs (mge); clipper
  is hygiene, not a fix (`../../clipper_campaign/RESULTS.md`).
- Plain-Delaunay batch_size steers which optimum is found (batch 2 vs 4 =
  5,622 nats) — mesh conclusions need A100 or fixed-batch discipline
  (`../PROGRAMME.md` §2.6).

## PERFORMANCE (never cross-tier)

- laptop RTX 2060 fp64: 16×3000 mge run = 48,000 lane-steps
  (`../PROGRAMME.md` §1.2); raw rows in
  `../../clipper_campaign/multi_start_prodigy_imaging_mge_hst.json`.
- hpc_a100_fp64 point-source cells: e.g. image_plane/simple 90.31 s / 65
  evals-per-lane-batch rows; 256-start arm 3,515 s
  (`results/searches/multi_start_prodigy/point_source/**`; dashboard table).

## RECOMMENDED

- Nothing unconditional yet. Candidate constraint-guided MAP engine for SLaM
  search stages **if** Gate B passes (Phase 3 reliability + Phase 4
  positions); otherwise classified LOCAL/INITIALIZED-only
  (`../PROGRAMME.md` Phases 3–4 gates).
- CONFIDENCE: **anecdote** on global mge reliability (2 seeds, split
  verdict); **seeded** on cluster-scale non-competitiveness (truth-anchored
  campaign across cells).

## REFERENCES

- Internal: `../PROGRAMME.md` §1.2, §2.3, §2.4, Phases 3–5;
  `../../clipper_campaign/RESULTS.md`;
  `../../point_source_defaults_campaign.md`;
  `../../multistart_prodigy_compile_census.md`.
- Literature: GIGA-Lens (300 starts, 1–5%/start hit rates) — the population
  precedent (`../LITERATURE.md`).
