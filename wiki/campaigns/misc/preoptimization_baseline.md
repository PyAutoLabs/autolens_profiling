# PreOptimizationTimes baseline

**Status:** complete
**Question:** What did the likelihoods cost, and where did the time go, before the optimisation campaigns began (the frozen, append-only PreOptimizationTimes baseline)?
**Pre-registered rule:** not recorded as a go/no-go rule (a baseline); the working rules written before the runs were VRAM-first validation (phase 2), the CPU-usability rule (a run over 3600 s or a call over ~1 min is GPU-only, phase 3) and the GPU-first breakdown doctrine (human, 2026-07-10, phase 4).
**Verdict:** runtime baseline frozen as `results/baselines/PreOptimizationTimes/` (laptop CPU, 12 cells); breakdown laptop-CPU fallback tier and A100 imaging tier complete 2026-07-10; no library drift since the May v2026.5.29.4 rows (0.89–1.14×); the bottleneck moves on GPU (CPU F-matrix 42–48 %, A100 pixelization NNLS ~65 %). Delaunay A100 rows superseded 2026-09-05 by [DelaunayNN A100 series](delaunay_nn_a100_series.md).
**Headline:** HST `imaging/pixelization` dense 57.6 ms per call A100 fp64 (150× the laptop CPU's 8.65 s; Regularized reconstruction 65 %), RAL A100 imaging jobs 330062–330070 (per-cell job id not recorded).
**Library PRs:** none
**Profiling PRs:** #53 (phase 1), #55 (phase 2), #62 (phase 3), #63 (phase 4).
**Ledger:** [preopt_breakdown_baseline.md](../../results/notes/preopt_breakdown_baseline.md); frozen runtime snapshot [PreOptimizationTimes.md](../../results/baselines/PreOptimizationTimes/PreOptimizationTimes.md); phase notes [design_lock_in.md](../../results/notes/design_lock_in.md), [vram_validation_2026_07_08.md](../../results/notes/vram_validation_2026_07_08.md).
**Mind contract:** umbrella `complete/archive/shelved/autolens_profiling_polish.md` (was `maintenance/autolens_profiling/polish.md`, shelved 2026-09-04 as spent); phase records `complete/unknown/profiling-polish-design.md`, `complete/unknown/profiling-vram-validation.md`, `complete/2026/07/profiling-preopt-campaign.md`, `complete/2026/07/preopt-breakdown-dashboard.md`.
**Next:** none (the baseline is append-only; a later campaign snapshots a new name beside it).

## Why this campaign

The polish umbrella (filed 2026-05-18, phased 2026-07-08) asked for one last design review of
the repo and then a full profiling sweep named **PreOptimizationTimes**: the "compare against
this" anchor for every optimisation campaign that followed. Four phases ran back to back:
design lock-in (#52), VRAM-first CPU validation so memory bugs did not derail A100 time (#54),
the runtime sweep (#56), and the per-step breakdown plus README dashboard (#59).

Two human decisions shaped it. Phase 3 was frozen "done enough" at the human's direction after
the `--auto` run stalled with RAL down, so the runtime snapshot is laptop CPU only. In phase 4 the
human answered the platform question (2026-07-10, issue #59) with a GPU-first breakdown doctrine:
the canonical decomposition is A100 fp64 (+ mp where supported), laptop CPU fp64 is a fallback
tier, and the HPC-CPU breakdown leg was dropped.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 Design lock-in | 2026-07-08 | Lock the package split, cell grid, CLI surface and baseline convention | no profiling runs; small lock-in refactors only | locked: one question per package, `<dataset_class>/<model>` cells, append-only named baselines (layout later inverted to dataset-first, #84, 2026-07-24) | none | #53 |
| 2 VRAM-first validation | 2026-07-08 | Do all campaign cells run before A100 time is spent? | run VRAM first; fix only small in-repo defects, file the rest | 9/9 cells clean on laptop CPU (47–401 s) after three fixes: backend-aware vmap clamp, per-instrument pins, record-and-flag pins | laptop (A100 probes staged, RAL down) | #55 |
| 3 Runtime baseline | 2026-07-08 → 07-11 | Full-pipeline per-call cost per cell × config | CPU-usability rule (> 3600 s run or > ~1 min per call = GPU-only) | frozen: 12 cells, local CPU fp64/mp dense + sparse (e.g. HST pixelization 13.72 s fp64 dense, 5.79 s sparse); laptop-GPU and A100 legs deferred | laptop | #62 |
| 4a Breakdown, laptop CPU | 2026-07-10 | Where does the time go on CPU? | not recorded (methodology fixed during the run: quiet machine, policy `XLA_FLAGS` exported) | 5 HST imaging cells; F-matrix 42–48 % on every mesh cell, MGE convolution-bound; no drift vs May | laptop | #63 |
| 4b Breakdown, A100 imaging | 2026-07-10 | Where does the time go on the A100? | GPU-first doctrine (human, 2026-07-10) | mge 7.8 ms, pix ~58 ms (NNLS ~65 %), delaunay ~98 ms (inversion setup ~42 %); mp flat (≤ 2 %) | 330062–330070 | #63 |
| 4c Delaunay setup split | 2026-07-10 | Which piece of the Delaunay inversion setup dominates? | not recorded (check: prefix-sum vs combined block) | triangulation + interpolation 26.62 ms (~27 % of the call); prefix-sum 39.96 vs 41.14 ms | 330079 | #63 |
| 4d Breakdown, alma_high | 2026-07-10/11 | Interferometer and datacube at alma_high on the A100 | GPU-first doctrine; deferred from the laptop after the NUFFT build exceeded 2 h twice | datacube fp64 + mp landed; interferometer Delaunay OOM (61.44 GB one-shot column NUFFT), classified `gpu_unusable_breakdown` | within 330058–330072 (not separately recorded) | #63 |

## What shipped and where it is

No library PRs: every change was profiling-repo tooling, data and docs.

## Open / parked / drafts

- `complete/archive/shelved/nufft_mapping_matrix_column_chunking.md` — the column-chunked NUFFT follow-up for the alma_high interferometer breakdown, shelved 2026-08-11 as overtaken (the dense extraction path is off in the breakdown cell because production uses the sparse path).
- The deferred laptop-GPU and A100 runtime legs of phase 3 were never refreshed into this snapshot; the A100 pixelized rows now live in [A100 pixelized baseline](a100_pixelized_baseline.md).

## Caveats

- **Superseded rows.** The `imaging/delaunay` A100 rows were re-run and superseded 2026-09-05 by `delaunay_nn_breakdown.md` (job 342277); their "Regularization matrix (H)" row timed an eager 14.4 ms host-to-device copy, not a JIT step. The `mge` and `pixelization` rows and the CPU tier stand.
- **The runtime snapshot is loaded-laptop CPU.** Taken on the 2026-07-08/09 overnight matrix; issue #59 cross-posted to #56 that it may be inflated by the constant-folding `XLA_FLAGS` and ambient load. Breakdown step-sums vs runtime (pixelization 8.65 vs 13.7 s, Delaunay 10.07 vs 16.7 s) are qualitative only.
- **Retracted claim.** A "≥ 1.8× library slowdown since May" posted on #59 was withdrawn the same day: host contention and the `XLA_FLAGS` export, not code (the first pass came back 2.5–5.3× slow on every step uniformly).
- **Cross-version tiers.** The A100 jobs ran libraries at 2026.7.9.1, the laptop at 2026.7.6.649; per-JSON version fields keep them apart.
- **Ledger status line is stale.** It still says the alma_high A100 cells are "in flight"; issue #59 records the datacube cells landed and the interferometer cell OOM'd.
- **No per-cell job map** for 330062–330070 or the alma_high jobs is recorded in the ledger.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records all four polish phases (#52/#53, #54/#55, #56/#62, #59/#63) with their working
rules and results, the frozen runtime snapshot and the breakdown tiers. Resolved: profiling PRs and
the Mind contract from the phase records and the shelved umbrella; the alma_high outcome from issue
#59 (datacube landed, interferometer OOM, follow-up shelved). Open: per-cell A100 job ids are not
recorded, and the ledger's status line is stale.
