# Inference programme — literature & ecosystem ledger

One entry per source: reference → the lesson it earned in this programme.
Add entries as phases consume new sources; cite these from method cards
(`methods/<method>.md`).

## Nested sampling

- Yallup, Kroupa & Handley, *Nested Slice Sampling* (arXiv:2601.23252, TMLR
  2026) — the blackjax NS algorithm. num_delete is the GPU axis (k/m ≈ 0.1
  conservative, ≤ 0.5 ceiling); inner steps ≥ max(5, 2·d) or logZ biases
  UPWARD (our fork rows used 5 in 15–20D and sat +7–13 nats high — Phase 2's
  pre-registered hypothesis); dlogz default −3, use −10 for phase-transition
  targets; near-flat runtime to 4000 live points on A100.
- blackjax PR #947 (merged 2026-06-29) → mainline release 1.6 (current 1.6.2):
  `blackjax.nss` / `blackjax.nsswig`, native-space `logprior_fn` (no unit-cube
  transform), `utils.finalise` + simulated volume sequences = free logZ error
  ensemble. The 2026-01 fork pins (handley-lab/blackjax@ef45acd2 +
  yallup/nss@69159b0f) are obsolete.
- Ashton et al. 2022 (arXiv:2205.15570); Buchner 2023 (arXiv:2101.09675) —
  region samplers degrade beyond d ≈ 20; slice-based NS is the high-d workhorse
  (PolyChord, arXiv:1506.00171).
- Lange 2023, *nautilus* (arXiv:2306.16923) — importance NS with neural bounds;
  our CPU reference engine.
- Fowlie et al. (arXiv:2010.13884) — plateau/phase-transition premature
  termination; why the dlogz −10 arm exists.

## MCMC / gradient samplers

- Hoffman & Gelman (arXiv:1111.4246); Betancourt (arXiv:1701.02434) — NUTS +
  divergences as geometry diagnostics.
- Vehtari et al. (arXiv:1903.08008) — rank-normalized split-R̂ < 1.01,
  ESS_bulk/ESS_tail > ~100/chain; the acceptance criteria for Phase 6
  (implemented in blackjax ≥1.6.1 diagnostics).
- blackjax many-chain docs + issue #251 — vmapped NUTS lockstep pays the
  slowest chain's tree depth (reported 43× waste); fixed-work kernels
  (ChEES/MEADS-adapted HMC, MCLMC) are the GPU-native choices and need ≥16
  chains for cross-chain adaptation.
- Robnik et al. — MCLMC (arXiv:2303.18221), adjusted/MAMS (arXiv:2503.01707):
  exact variant claims NUTS-beating high-d efficiency; tuning heuristics young;
  no multimodality story.
- Zhang et al., *Pathfinder* (arXiv:2108.03782) — initializer/preconditioner,
  Pareto-k̂ diagnostic; mode-local (pair with multi-start).

## SMC

- Del Moral, Doucet & Jasra 2006 — the framework permits an arbitrary initial
  distribution with importance corrections; the prior is a convention.
- Jasra et al. 2011; Zhou et al. (arXiv:1303.3123) — adaptive tempering via
  (C)ESS root-solving; target_ess ≈ 0.5.
- Beskos et al. (arXiv:1103.3965) — particle count must grow with dimension
  unless tempering steps shrink.
- Duan & Fulop 2023 (WIREs, doi:10.1002/wics.1598); Cai et al.
  (arXiv:2202.07070) — density-tempered / model-tempered SMC from an informed
  start with evidence preserved (normalized reference) — the construction our
  parked prototype already implements. A bridge cannot resurrect a mode the
  start missed (Phase 7's explicit test).
- Dau & Chopin (arXiv:2011.02328) — waste-free SMC; rejuvenation adequacy.

## Lens-modelling inference strategies

- Gu et al., *GIGA-Lens* (arXiv:2202.07663) — 300 multi-start Adam inits ×
  300 iters, 1–5% per-start global-basin hit rate; then full-rank Gaussian SVI
  at the MAP; then HMC with the SVI covariance as metric. The
  optimize-first/sample-second architecture our Phases 3→6 test; their
  "local minima have vanishing posterior mass" footnote is for 22-param smooth
  models and must NOT be assumed for pixelized likelihoods.
- Galan et al., *Herculens* (arXiv:2207.05763; wavelets arXiv:2210.09169) —
  gradient descent to a point estimate + Fisher/Hessian or HMC uncertainties,
  up to thousands of parameters. Both flagship JAX lensing codes converged on
  optimize-first.
- JAXNS (arXiv:2012.15286; phantom sampling arXiv:2312.11330; GGNS
  arXiv:2312.03911) — gradient-guided NS marked experimental upstream;
  literature comparator only.

## Nonsmooth likelihoods

- Chaari et al. (arXiv:1401.3988) ns-HMC via Moreau–Yosida smoothing; Shukla
  et al. (arXiv:2510.22252) proximal HMC with ergodicity guarantees; Pereyra
  (arXiv:1306.0187) proximal MCMC; Nishimura et al. (arXiv:1705.08510)
  discontinuous HMC — the principled foundations for Phase 9's finite-μ
  interior-point idea. No lensing-specific NNLS-kink literature exists
  (verified gap, 2026-08-17). Any adopted smoothing is category 3 unless
  proven within tolerance (Gate F).

## Optimizer termination

- GIGA-Lens: fixed 300-iter budget justified empirically by best-trajectory
  plateau. Pflug 1983 / Yaida (arXiv:1810.00004) stationarity tests; Pesme et
  al. (PMLR v119) show Pflug unreliable even on quadratics; confidence-sequence
  stopping (arXiv:2512.13123). All certify LOCAL stationarity only — no test
  detects a wrong basin; start coverage is the only control (Phase 3's p_hit
  framing).
