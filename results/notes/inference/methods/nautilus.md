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

- hpc_a100_fp64 (CPU-hosted sampler, GPU likelihood): mge/hst 831 s /
  63,800 evals / 12.1 ms per eval; delaunay/hst 2,723 s / 31,536 evals /
  84.8 ms; pixelization/hst 2,768 s / 58,464 evals / 46.5 ms
  (`results/searches/nautilus/imaging/*/hst/hpc_a100_fp64.json`).
- Full matrix: the searches dashboard
  (`scripts/misc/searches/README.md`, auto-table `searches`).

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
