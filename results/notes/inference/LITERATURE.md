# Inference programme — external literature ledger

The external references gathered during the 2026-08-17 planning pass, one
lesson line each — what each source actually earned in this programme's
reasoning, not a bibliography. Extend as phases execute; every method card
(`methods/<method>.md`) cites into this file.

## Nested sampling

- **blackjax nested sampling mainline merge** — blackjax PR #947 (merged
  2026-06-29; released in blackjax 1.6, current 1.6.2). Lesson: the 2026-01
  fork pins in the stashed `af.NSS` are obsolete — re-integration targets
  `blackjax.nss` mainline (native-space `logprior_fn` API, `num_delete` as
  the GPU axis, dlogz termination, ensemble logZ error bars), and local/RAL
  environments must upgrade from 1.5 first (Phase 0(b)).
- **Yallup et al., arXiv:2601.23252 (NSS paper)** — GPU-native nested slice
  sampling. Lesson: upstream guidance sets inner steps ≥ max(5, 2·d); our
  historical fork runs used 5 in 15–20D, which predicts the observed +7–13
  nat logZ bias — the sharpest pre-registered test in Phase 2 (H2.1).

## Gradient lens modelling (prior art)

- **Gu et al., arXiv:2202.07663 (GIGA-Lens)** — Lesson: on similar smooth
  parametric lens models, reliability comes from population scale — 300
  starts at 1–5% per-start hit rates — supporting the Phase-3 reframe from
  "seed lottery" to measured p_hit with reliability 1−(1−p)^n.
- **Galan et al., arXiv:2207.05763 (Herculens)** — Lesson: differentiable
  lens modelling with gradient-based inference is established prior art; the
  open gap this programme fills is reliability accounting (basin selection,
  seeds, termination) rather than feasibility.

## MCMC / posterior sampling

- **Vehtari et al., arXiv:1903.08008 (rank-normalized R̂ / ESS)** — Lesson:
  Phase-6 acceptance criteria use rank-normalized split-R̂ < 1.01 with
  ESS_bulk/ESS_tail floors (blackjax 1.6 ships these diagnostics); plain R̂
  on raw chains is not sufficient evidence of convergence.
- **Robnik et al., arXiv:2503.01707 (MAMS / adjusted MCLMC)** — Lesson:
  fixed-work-per-step GPU samplers (ChEES/MEADS-adapted HMC, MCLMC/MAMS)
  avoid vmapped NUTS's lockstep tree-depth waste (documented 43×
  pathologies) — hence Phase 6 carries ChEES/MAMS arms rather than assuming
  many-chain NUTS parallelism.
- **Zhang et al., arXiv:2108.03782 (Pathfinder)** — Lesson: quasi-Newton
  variational paths are an *initializer*, not a posterior engine — admitted
  in Phase 7 only as an initializer with the Pareto-k̂ diagnostic, and only
  against a named failure from Phases 6/13.

## SMC / informed-start bridges

- **Del Moral et al. 2006 (SMC samplers) + Duan & Fulop 2023
  (density-tempered / informed-start SMC)** — Lesson: the parked prototype's
  normalized-Gaussian geometric bridge (`logprior := log g`,
  `loglik := log π + log L − log g`; evidence preserved) matches the
  literature's density-tempering construction — Phase 7 resumes it rather
  than redesigning. Caveat carried into the plan: a bridge from an informed
  start cannot resurrect a mode the start missed (explicit mode-dropped-start
  test).

## Non-smooth targets / proximal smoothing

- **Proximal-smoothing line, arXiv:1401.3988 and arXiv:2510.22252** —
  Lesson: Moreau–Yosida-style smoothing of non-smooth targets (proximal /
  ns-HMC) is the literature backing for Phase 9's finite-μ barrier-forward
  candidate on the NNLS positivity kinks; no lensing-specific precedent
  exists (verified gap), so Phase-9 findings are publishable-grade ledger
  documentation. Any such smoothing is category 3 unless proven within
  tolerance (Gate F).
