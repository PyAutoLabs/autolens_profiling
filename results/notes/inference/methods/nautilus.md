# Nautilus — evidence card

Living method card (template: `../PROGRAMME.md` §6). Seeded 2026-08-18 from
the existing benchmark record; extend as phases execute. Every claim cites a
result JSON, a notes file, or a PROGRAMME section — no new claims here.

## IDENTITY

Global nested sampler (importance NS with neural bounds — Lange 2023,
`../LITERATURE.md`). Gradient-free. CPU-native (`number_of_cores`); runs on
GPU nodes but the sampler itself is not GPU-parallel — the likelihood is.
Handles multimodality (live-point population + bounds). PyAutoFit first-class
search (`af.Nautilus`); the organism's **CPU reference engine** and the
current truth-bar source.

## EVIDENCE

- Dimensional regimes: ~7–20D parametric (mge 15D, point-source/cluster
  cells), pixelized meshes with hyper-dims (delaunay/pixelization cells).
- Datasets/models: imaging {mge, delaunay, pixelization} × HST
  (`results/searches/nautilus/imaging/*/hst/hpc_a100_fp64.json`);
  point_source and cluster matrices, truth-anchored
  (`results/searches/nautilus/{point_source,cluster}/**`,
  `../../point_source_defaults_campaign.md`).
- Initialization modes tested: broad priors only (no warm-start mode exists).

## STRENGTHS

- Sets the truth bar on every imaging cell: mge/hst max logL 31,786.8
  (logZ 31,690.5), delaunay/hst 30,623.5, pixelization/hst 29,143.3
  (`results/searches/nautilus/imaging/*/hst/hpc_a100_fp64.json`;
  `../PROGRAMME.md` §1.2).
- Recovers each objective's basin at cluster/point-source scale (deltas
  +2.2 to +8.5 vs truth anchors) where gradient searches are not competitive
  (`../../point_source_defaults_campaign.md` — "use Nautilus at cluster
  scale").
- Structurally immune to prior-wall lane deaths (samples the unit cube;
  `../../clipper_campaign/RESULTS.md`).
- Evidence (logZ) comes with the run — the reference for NSS logZ-bias
  cross-checks (`../PROGRAMME.md` §2.2, Phase 2).

## WEAKNESSES

- Wall-time scales with likelihood evals × CPU-serial sampling: 831 s (mge)
  → 2,723–2,768 s (mesh cells) on A100 fp64 nodes; ~10× slower than
  multi-start Adam on the same node where the MAP alone suffices
  (`../PROGRAMME.md` §1.2).
- No warm-start / informed-init path — every SLaM stage pays the full global
  cost (`../PROGRAMME.md` Phases 6, 12 motivation).

## CONFIGURATION

- Benchmarked: n_live 200 (mge), 150 (delaunay/pixelization) — the
  SLaM-mirroring values (`scripts/misc/searches/_samplers.py`); point-source
  cells per campaign configs.
- Recommended-by-regime: SLaM n_live table (200/150/75/150/150 per stage,
  `../PROGRAMME.md` Phase 12). Knob scan (one n_live row) is a Phase 2 arm.
- Sensitivity notes: none recorded yet beyond the n_live tiers — Phase 1
  reference runs will pin per-target tolerances.

## TERMINATION

- Rule: live-set evidence criterion (importance NS); no `stop_reason`
  telemetry recorded in current artifacts.
- Multimodal reliability: dominant-mode behaviour verified on mge (truth
  basin is the dominant mode — `../PROGRAMME.md` §1.2); plateau/phase
  transition hazard is a literature caution (Fowlie et al.,
  `../LITERATURE.md`), untested here.

## HAZARDS

- None recorded in `results/hazards/` against Nautilus itself. Cross-tier
  comparison of its rows is the standing bench hazard (tiers never mix —
  `../PROGRAMME.md` §3).

## PERFORMANCE (never cross-tier)

- hpc_a100_fp64 (CPU-hosted sampler, GPU likelihood): mge/hst **707 s
  sampler / 775 s total, 10.56 ms per eval at `n_batch=64`** (63,424 evals)
  — the 2026-08-24 re-baseline on the current stack (RAL 339070) plus the
  W6 scan's baseline arm; the historical 831 s / 63,800 evals / 12.1 ms row
  pre-dates that re-baseline and is retired as a reference.
  delaunay/hst 2,723 s / 31,536 evals / 84.8 ms; pixelization/hst 2,768 s /
  58,464 evals / 46.5 ms
  (`results/searches/nautilus/imaging/*/hst/hpc_a100_fp64.json`).
- Full matrix: the searches dashboard
  (`scripts/misc/searches/README.md`, auto-table `searches`).

## n_batch SCAN (W6, issue #163 — A100, 2026-08-25)

Question: is the JAX-Nautilus per-eval overhead a config knob, or structural?
Answer: **partly a knob on MGE, essentially structural on Delaunay.**

MGE/hst, `n_live=200`, fp64, one seed per arm
(`results/searches/nautilus/imaging/mge/hst/hpc_hpc_a100_fp64_nbatch*.json`):

| n_batch | ms/eval | sampler wall | evals | Kish ESS | logZ |
|--------:|--------:|-------------:|------:|---------:|-----:|
| 64      | 10.56   | 670 s        | 63,424 | 4,304   | 31690.45 |
| 128     |  9.52   | 617 s        | 64,768 | 4,248   | 31690.45 |
| 256     |  8.15   | 540 s        | 66,304 | 4,156   | 31690.48 |
| 512     |  8.37   | 565 s        | 67,584 | 4,575   | 31690.47 |
| 1000    |  **5.95** | **458 s**  | 77,000 | 4,694   | 31690.36 |

**State the basis (corrected 2026-08-27 — `../DECISIONS.md` 2026-08-27 W6).**
64 -> 1000 recovers **1.775x per likelihood eval** (10.56 -> 5.95 ms) but only
**1.463x on sampler wall** (670 -> 458 s) and **1.594x on ESS/min**: evals rise
21% (63,424 -> 77,000) as larger batches overshoot the shrinking live set, and
Kish ESS per eval falls ~10%. The earlier "1.78x free" sentence carried the
per-eval figure into a wall-shaped claim.

**The evidence is not flat at the recommended arm.** logZ spans 0.12 nats over
the whole scan, but the n_batch=1000 arm sits **-0.10 nat** from the n_batch=64
arm — about **9 sigma** of the 0.011-nat five-seed logZ standard deviation
measured on this cell in Phase 4 Stage 2. max logL 31786.73-31787.04 and
best-fit r_E 1.5996-1.5998 across the scan. **One seed per arm**: the scan
cannot separate an n_batch bias from a seed draw, so no n_batch above the
baseline is adopted as a default on this evidence. The scan has **not**
plateaued at n_batch=1000 and there is an optimum past it that this scan does
not bracket.

Delaunay/hst, `n_live=150`
(`.../imaging/delaunay/hst/hpc_hpc_a100_fp64_nbatch*.json`):

| n_batch | ms/eval | sampler wall | evals | Kish ESS | logZ |
|--------:|--------:|-------------:|------:|---------:|-----:|
| 16      | 66.73   | 2,031 s      | 30,432 | 2,464   | 30562.10 |
| 64      | 52.95   | 1,698 s      | 32,064 | 2,335   | 30562.17 |
| 256     | 50.80   | 1,704 s      | 33,536 | 2,387   | 30562.17 |

Saturates by n_batch=64 (1.26x, then flat). The pixelized cell's cost is
dominated by the per-eval inversion, not by batch occupancy, so batching
cannot buy back what MGE's cheaper likelihood gives up to launch overhead.

READING: raise `n_batch` on parametric cells; leave it at the default on
pixelized ones. This does **not** close the ~4x JAX-vs-NumPy per-eval gap —
it recovers under half of it on the cell where it is recoverable at all.

CONFIDENCE: **single-seed per arm.** logZ agreement across five independent
MGE arms is itself weak evidence of seed-stability, but no arm was repeated;
treat the wall numbers as one draw each.

## RECOMMENDED

- All SLaM stages today (baseline pipeline); the global engine wherever no
  trusted previous fit exists; the cluster-scale answer
  (`../../point_source_defaults_campaign.md`).
- CONFIDENCE: **seeded** on imaging cells (multi-cell, truth-anchored,
  single-seed per cell); **seeded** at point-source/cluster scale
  (truth-anchored campaign); reliability-over-seeds quantification is
  Phase 1/2 work.

## REFERENCES

- Internal: `../PROGRAMME.md` §1.2, Phases 1–2, 12;
  `../../point_source_defaults_campaign.md`; `../../clipper_campaign/RESULTS.md`.
- Literature: Lange 2023 (nautilus); Ashton 2022 / Buchner 2023 (NS reviews);
  Fowlie et al. (plateau termination) — `../LITERATURE.md`.
