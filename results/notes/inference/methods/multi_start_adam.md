# MultiStartAdam — evidence card

Living method card (template: `../PROGRAMME.md` §6). Seeded 2026-08-18.
The measured-p_hit precedent that shaped Phase 3's design; not itself the
Phase 3 subject (that is Prodigy).

## IDENTITY

Multi-start gradient MAP optimizer (`af.MultiStartAdam`, JAX/optax). Local
per lane; global only through start coverage. Gradient-required,
JAX/GPU-native (vmapped lanes); pure-NumPy configs raise. No evidence, no
posterior. Registered first-class only for the benchmark-proven cells where a
gradient MAP is meaningful (`scripts/misc/searches/README.md`, "MAP
optimizers alongside samplers").

## EVIDENCE

- Dimensional regimes: 15D (imaging/mge), group-scale mge.
- Datasets/models: imaging/mge/hst 128-start A100 fp32 warm runs
  (`../PROGRAMME.md` §1.2); group/mge/hst laptop-GPU row
  (`results/searches/multi_start_adam/group/mge/hst/local_local_gpu_fp64.json`);
  CPU/float32 clipper arm pair
  (`../../clipper_campaign/README.md`,
  `../../clipper_campaign/multi_start_adam_imaging_mge_hst.json`).
- Initialization modes tested: broad priors only.

## STRENGTHS

- **Measured per-start hit probability**: p_hit ≈ 0.18/start on imaging/mge,
  stable — the number that converts multi-start reliability into arithmetic,
  1−(1−p)^n (`../PROGRAMME.md` §1.2, §2.3).
- Speed at matched hardware: 128 starts, fp32 A100 warm ≈ 50 s to
  max logL 31,787.9 vs Nautilus 523 s on the same node — ~10× faster where
  the MAP alone suffices (`../PROGRAMME.md` §1.2).

## WEAKNESSES

- Same basin-selection ceiling as every cold gradient method: P(all 16 starts
  miss) ≈ 4% at p=0.18 — reliability is a start-count budget, never a
  guarantee (`../PROGRAMME.md` §2.3).
- Adam→L-BFGS polish is **actively harmful** on this objective
  (`../PROGRAMME.md` §1.2).
- The group/mge laptop row is far from any truth bar (max logL −231,891,
  33 evals recorded) — a mechanics smoke, not evidence
  (`results/searches/multi_start_adam/group/mge/hst/local_local_gpu_fp64.json`).

## CONFIGURATION

- Benchmarked: 16×150 (CPU clipper arms), 128-start A100 fp32 (warm).
- fp32 vs fp64: the 50 s headline is fp32; fp64 required for final-quality
  MAP values per the bench's precision rules (`../PROGRAMME.md` §3).
- Clipper: same hygiene verdict as Prodigy — deaths 2268 → 47 with best-fit
  logL identical to every printed digit; 11/16 surviving lanes end pinned to
  a bound (`../../clipper_campaign/README.md`).

## TERMINATION

- Fixed step budget in all recorded runs; the MultiStartGradient
  auto-convergence plateau detector applies to this family too, with the same
  wrong-basin-plateau blindness (`../PROGRAMME.md` §1.2).
- GIGA-Lens's fixed 300-iter budget is the literature precedent; all known
  stationarity tests certify local stationarity only (`../LITERATURE.md`,
  Optimizer termination).

## HAZARDS

- Prior-wall lane deaths / bound pinning (see clipper arc). Wrong-basin
  plateau reads as converged. Shared with Prodigy: mesh objectives are out of
  its registered use case entirely.

## PERFORMANCE (never cross-tier)

- hpc_a100 fp32 (warm): 128 starts ≈ 50 s, max logL 31,787.9
  (`../PROGRAMME.md` §1.2).
- cloud CPU float32 (clipper arms): 16×150 steps, both arms
  max logL −15,529.6 — budget too small to converge; recorded as a starting
  point, not a verdict (`../../clipper_campaign/README.md`).

## RECOMMENDED

- The fast-MAP precedent and the p_hit measurement instrument; day-to-day
  recommendation deferred to Gate B (Prodigy is the programme's multi-start
  candidate; Adam rows serve as its comparator).
- CONFIDENCE: **seeded** for p_hit on imaging/mge (many starts, stable
  estimate); **anecdote** everywhere else.

## REFERENCES

- Internal: `../PROGRAMME.md` §1.2, §2.3; `../../clipper_campaign/README.md`;
  `scripts/misc/searches/README.md` (registration scope).
- Literature: GIGA-Lens (arXiv:2202.07663) — 300 multi-start Adam inits,
  1–5%/start; optimizer-termination line — `../LITERATURE.md`.
