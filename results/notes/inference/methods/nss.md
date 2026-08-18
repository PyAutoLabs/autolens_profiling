# NSS (BlackJAX nested slice sampling) — evidence card

Living method card (template: `../PROGRAMME.md` §6). Seeded 2026-08-18 from
the fork-era benchmark record; **Phase 2 re-tunes this method against
mainline blackjax ≥1.6 — every number below is fork-era and pre-registered
for re-measurement.** Gate A pending.

## IDENTITY

Global nested sampler, slice-sampling inner kernel, GPU-native (vectorized
live-point updates; `num_delete` is the GPU axis). Gradient-free. Handles
multimodality (nested-sampling population). Integration status: `af.NSS` was
shipped, benchmarked, and deliberately removed 2026-07-11 (PyAutoFit#1357);
implementation stashed at `autofit_workspace_developer/searches/nss/` with
obsolete fork pins — nested sampling is now **mainline blackjax 1.6**
(PR #947; `../LITERATURE.md`). Re-integration targets `blackjax.nss` and is
gated on Gate A (`../PROGRAMME.md` Phase 2, §7).

## EVIDENCE

- Dimensional regimes: 15–20D (mge), pixelized mesh cells — all fork-era
  (2026-01 fork, 5 inner steps).
- Datasets/models: imaging mge/delaunay/pixelization × HST
  (`results/searches/nss/imaging/mge/hst/hpc_a100_fp64.json` is the surviving
  committed artifact; delaunay/pixelization numbers recorded in
  `../PROGRAMME.md` §1.2).
- Initialization modes tested: broad priors only.

## STRENGTHS

- Matched Nautilus's answer on every cell it ran (max logL within ~0.5–1.3
  nats of the truth bars — `../PROGRAMME.md` §1.2).
- Mildly faster than Nautilus on MGE: 657–679 s vs 831 s, same A100 fp64
  tier (`results/searches/nss/imaging/mge/hst/hpc_a100_fp64.json`). Human
  review note at plan approval: "on mge blackjax NS was fastest or comparable
  ... and it may scale better, so worth rerunning" (`../DECISIONS.md`
  2026-08-17).
- Native ensemble logZ error bars in mainline 1.6 (`utils.finalise` —
  `../LITERATURE.md`), which the fork rows lacked.

## WEAKNESSES

- **7–11× slower than Nautilus on pixelized cells** at fork settings:
  delaunay 29,770 s vs 2,723 s (206k vs 31.5k evals); pixelization 19,190 s
  vs 2,768 s (266k vs 58k) — inner slice evals dominate (`../PROGRAMME.md`
  §1.2, §2.2). Confounded by (i) under-mixed inner kernel, (ii) fork era —
  Gate A judges per model family.
- **+7–13 nat logZ bias** in every recorded row (mge logZ 31,697.7/31,700.4
  vs Nautilus 31,690.5) — predicted by running 5 inner steps in 15–20D where
  upstream guidance is ≥ max(5, 2·d); pre-registered as H2.1
  (`../PROGRAMME.md` Phase 2).
- ~390k likelihood evals on MGE vs Nautilus's 63.8k — wins wall only because
  evals vectorize on GPU.

## CONFIGURATION

- Benchmarked (fork): n_live 200, num_mcmc_steps 5, num_delete 50.
- Phase 2 scan (pre-registered): n_live {200, 500, 1000} × num_delete
  {0.1m, 0.25m, 0.5m} × inner steps {5, 2d, 3d} × dlogz {−3, −10}
  (`../PROGRAMME.md` Phase 2).
- Knobs users should not touch (upstream guidance, unverified here): inner
  steps below max(5, 2·d) — the logZ-bias mechanism (`../LITERATURE.md`,
  Yallup et al.).

## TERMINATION

- Rule: dlogz criterion (mainline); fork-era termination behaviour is part
  of what Phase 2 re-measures at both dlogz values.
- Evidence termination safety: unverified — the logZ bias makes fork-era
  evidence claims unsafe until H2.1 is resolved.

## HAZARDS

- Environment: local/RAL run blackjax 1.5, incompatible with the stashed
  wrapper (verified import failure) — Phase 0(b) upgrade blocks any run
  (`../PROGRAMME.md` §2.2).
- Obsolete fork pins (handley-lab/blackjax@ef45acd2 + yallup/nss@69159b0f)
  must not be reused (`../LITERATURE.md`).

## PERFORMANCE (never cross-tier; all fork-era, hpc_a100_fp64)

- mge/hst: 657–679 s / ~390k evals / 1.6 ms per eval
  (`results/searches/nss/imaging/mge/hst/hpc_a100_fp64.json`).
- delaunay/hst: 29,770 s / 206,448 evals; pixelization/hst: 19,190 s /
  266k evals (`../PROGRAMME.md` §1.2).

## RECOMMENDED

- Nothing yet — candidate GPU nested-sampling baseline, adoption is Gate A.
  If the pixelized deficit survives tuning, scoped to parametric models with
  Nautilus keeping mesh duty (`../PROGRAMME.md` Phase 2 gate).
- CONFIDENCE: **anecdote** (single-seed fork rows with a pre-registered bias
  hypothesis standing against them).

## REFERENCES

- Internal: `../PROGRAMME.md` §1.1, §1.2, §2.2, Phase 2; `../DECISIONS.md`
  (2026-08-17 review note 2).
- Literature: Yallup, Kroupa & Handley (arXiv:2601.23252); blackjax PR #947 /
  release 1.6 — `../LITERATURE.md`.
