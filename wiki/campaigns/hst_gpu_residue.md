# HST GPU non-solver residue

**Status:** complete
**Question:** Around the certified solve, what else costs time in the HST A100 fp64 Delaunay likelihood (N=1500, budget 7), and is there an fp64 lever that keeps the answer?
**Pre-registered rule:** campaign-wide, every route or candidate returns the library's own log likelihood to ≤ 1e-9 relative on every fp64 leg; per phase: phase 1 a measured stage table + device-idle row reconciling to the wall within 5 %; phase 2 a three-way per-lane 1e-9 pin (vmap / scalar / vmapped library PDIP) before any batching policy; phase 3 a lever must save ≥ 1.5 ms whole-call at the pin (gate committed `e2b46f3`, 26 min before submit); phase 4 a lever must save ≥ 0.5 ms on both the `jit_profile` and interleaved bases within one task (committed `71e3f21`, amended `5f6f8c7` before submit).
**Verdict:** phase 1 refuted the attributed ~13.9 ms mesh/mapper bucket (0.37 ms measured) and re-based the call from 25.39 ms (budget 2) to 31.64 ms (budget 7); phase 2 inconclusive (pin failed on distinct lanes at B=8/16), no batching change authorised; phase 3 and phase 4 no fp64 lever. The fp64 levers in the campaign map are exhausted.
**Headline:** 31.64 ms, the budget-7 production Delaunay call (command buffers on), RAL A100 80GB PCIe, job 343350 (task 0).
**Library PRs:** none; no library source changed in any phase.
**Profiling PRs:** #270, #294, #296, #306 (issues #268, #273, #295, #303).
**Ledger:** [hst_gpu_residue_phase1_2026_09.md](../../results/notes/hst_gpu_residue_phase1_2026_09.md), [hst_gpu_residue_phase2_vmap_2026_09.md](../../results/notes/hst_gpu_residue_phase2_vmap_2026_09.md), [hst_gpu_residue_phase3_psf_2026_09.md](../../results/notes/hst_gpu_residue_phase3_psf_2026_09.md), [hst_gpu_residue_phase4_logdet_2026_09.md](../../results/notes/hst_gpu_residue_phase4_logdet_2026_09.md); summary in [profiling_campaign_status_2026_09.md](../../results/notes/profiling_campaign_status_2026_09.md) (GPU section).
**Mind contract:** epic `hst-gpu-non-solver-residue`; campaign map retired to `complete/2026/09/hst-gpu-non-solver-residue.md`; phase records `complete/2026/09/hst-gpu-residue-p{1,2,3,4}.md`; epic ledger `complete/archive/epics/hst-gpu-non-solver-residue.md`.
**Next:** none in this campaign; the carried items (fp32 policy, cond-free batched fallback, qhull callback / batching) are listed under Open.

## Why this campaign

The [fixed lens light](fixed_lens_light.md) epic ended with the certified solve no longer the
call: its map said ~21 of the 25.4 ms A100 Delaunay call was not the solver, and the largest term
("~13.9 ms mesh / mapper / weights / imaging / blurring") was attribution arithmetic across two
cells. The campaign map was filed on 2026-09-14 as that epic's named successor. Its rule was that
phase 1 must be a measurement (a one-process decomposition of the fused production jit) before any
lever was attacked, and that no optimisation may change the answer: fp64, budget 7, PDIP fallback,
positivity kept, ≤ 1e-9 pin. The human's request (`autolens_profiling#268`) was to optimise
"everything in the breakdown which isn't fnnls" on the A100 and RTX, with numba CPU left to #267.

The phases were issued one at a time. Phase 1 ranked three levers: the qhull `pure_callback`
host round-trip (5.44 ms idle), the PSF convolution of the mapping-matrix cube (7.12 ms), and the
second Cholesky for log det(F + λH) (0.89 ms). Phase 2 tested batching (production is
`jit(vmap)`, and the callback is sequential per lane). Phases 3 and 4 took the other two levers.
The map was retired complete on 2026-09-27 (`complete/2026/09/hst-gpu-non-solver-residue.md`).
The draft it lived in (`draft/research/autolens_profiling/hst_gpu_non_solver_residue_programme.md`,
cited by the phase records and issues) is not in the Mind drafts any more.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 fused-call trace | 2026-09-16 | Where does the fused production jit spend device time, measured in one process? | stage rows + measured idle sum to the traced wall within 5 %; route d == b ≤ 1e-9 every leg; HLO census of `F + λH` adds; no library edit (#268) | 31.64 ms at budget 7 (cmd buffers on): solve 10.22, idle 8.05 (5.44 = qhull gap), PSF cube 7.12, F GEMM 4.18, log-dets 0.89 + 0.89, mesh/mapper/weights 0.37; reconciliation −2.0 to −2.6 %, pins ≤ 3.2e-11; no `F + λH` add survives XLA; relocator on is production (0.10 ms) | 343350 (A100 ×3); RTX 2060 ×2 laptop | #270 |
| 2 batching (historical) | 2026-09-17 | vmap vs scalar jit for the retired `vmap(jit)` composition | as phase 2 below | superseded by PyAutoFit#1638's switch to `jit(vmap)`; historical evidence only, rows not relabelled | 343376 | #294 |
| 2 batching | 2026-09-19/20 | Is current `jit(vmap)` faster per lane than B scalar jits over distinct lanes? | three-way per-lane pin at 1e-9 (vmap, scalar, vmapped library PDIP); written policy with crossover (#273) | INCONCLUSIVE: fallback-on vmap slower per lane (scalar/vmap 0.329x / 0.512x / 0.697x at B=4/8/16); fallback-off B16 1.244x is diagnostic; pin FAILED at B=8 (7/8) and B=16 (13/16, 14/16); no policy change | 344635 | #294 |
| 3 PSF cube | 2026-09-23 | Is there a faster fp64 convolution of the `(180,180,1500)` mapping-matrix cube? | pin ≤ 1e-9 fiducial + 8 seeded draws; lever ≥ 1.5 ms whole-call at the pin (#295) | NO FP64 LEVER: control 31.66 ms; `layout_src_first` +0.13 (tie); `frame_pow2` +4.52, peak 2.38 GB; cuDNN batched 1.45x; real space 3.82x; fp32/complex64 rows −2.14 / −3.98 ms are DIAGNOSTIC | 350573 | #296 |
| 4 log-det reuse | 2026-09-24 | Can log det(F + λH) reuse the certified solve's Cholesky via a Schur complement? | pin ≤ 1e-9 vs the unmodified library route (fiducial + 8 draws), recon ±5 %, unjoined 0; lever ≥ 0.5 ms on both bases in one task (#303) | NO LEVER: 72/72 pins pass; interleaved saving Delaunay −0.27 / −0.31 / −0.45 ms (k32/k64/k256), rectangular +0.09 / +0.09 / −0.38; the ~1 ms `trsm` against the 1500×1500 factor exceeds the 0.90 ms dense Cholesky | 350651 | #306 |

## What shipped and where it is

No library PR shipped: every phase was a harness experiment, and the PyAutoArray prompts in
phases 3 and 4 were conditional on a lever and were correctly not filed. The profiling harness
(`xla_attribution.py`, the `fixed_light_trace.py` cell and its `--vmap-batch`, `--psf-candidate`
and `--logdet-candidate` modes) shipped in #270, #294, #296 and #306.

## Open / parked / drafts

- **qhull `pure_callback` idle (5.44 ms).** The phase-3 and phase-4 ledgers park it behind the
  batching-reproducibility study that phase 2 points at; the epic close routes it to "the qhull
  callback / batching work filed elsewhere" without a path. No Mind draft for the reproducibility
  study or the phase-2b batch-aware callback was found; the phase-2 record says either needs a new
  intake task.
- **Cond-free batched fallback:** `draft/feature/autofit/certified_solver_batched_guard_c2.md`
  (certified-solver C2). The phase-4 record cites it as
  `draft/feature/autofit/certified_solver_cond_free_batched_fallback.md`, which is not in the Mind
  drafts any more.
- **fp32 precision policy** (phase 3's diagnostic rows, 7–13 % of the call for ~1e-3 nats): a
  human decision. No Mind draft found. If taken up it must re-measure on a full inference.
- **Housekeeping follow-ups** in the phase-3 record: three stale census anchors in
  `xla_attribution.py` (re-anchoring was step 0 of #303's plan; phase 4's census is `ok`) and the
  stale `convolver.py:65-66` docstring (no draft found).
- Sibling draft [Post-certified breakdown](post_certified_breakdown.md): phase 1 superseded its
  GPU columns in part (per #268's scope).

## Caveats

- **Phase 2 is inconclusive, not a no-go.** The pre-declared three-way 1e-9 pin failed on distinct
  lanes at B=8 and B=16 (max vmap/scalar 2.8e-9); the production batched values all agree with
  their own vmapped library-PDIP reference (≤ 5.3e-10), and the scalar arm is the one outside. The
  threshold was not relaxed. Phase 3 placed the residual between compositions (`jit(vmap)` vs
  scalar jit), not in the solver. No batching or callback conclusion may be drawn from the grid.
- **Array 343376 is the retired `vmap(jit)` composition** and is historical only
  ([sidecar](../../results/notes/hst_gpu_residue_phase2_historical_job343376.json), status
  `historical-only`); array 344635 is the current-composition measurement. Tasks 0–2 of 344635
  exited non-zero after writing artifacts, and their logs lack the footer (ERR trap).
- **Phase-3 fp32-cube and complex64 rows are DIAGNOSTIC.** `mp_cube_c64` and `c64_full` miss the pin
  by ~1e-3 nats on the draws (max 1.39e-7 / 4.41e-7 relative) and may not be quoted as levers.
- **The certified solver in phases 1–3 was a harness injection**, not the shipped library
  implementation. Phase 4 ran the library certified solver (PyAutoArray#567, from the
  [certified positive solver](certified_positive_solver.md) campaign, released 2026.9.26.1) at the
  packaged budget 16 under scalar jit, so its 26.3–26.9 ms library route is not comparable with
  phases 1–3's ~31.6 ms.
- **Trace overhead.** A100 stage rows carry ~9 % profiler inflation (34.80 traced vs 31.89 ms
  untraced); scaling by 0.909 assumes uniform overhead and is not a measurement. Stage rows attribute
  the command-buffers-off program (+0.4 / +0.8 % on the Delaunay legs). DelaunayNN's idle row
  (+17.4 % command-buffer delta) is not a production timing. `mixed_fusion` is a resolution limit,
  and its prorated split is an estimate.
- **Two 31.6 ms bases.** 31.64 ms is the command-buffers-on traced executable; phase 1's route-d
  `jit_profile` was 31.49 ms. Phase 3's control reproduced these at 32.46 / 31.66 ms (within 2.6 %).
- **RTX 2060 rows are not quotable.** The same phase-1 leg read 1063 ms and 622 ms two hours apart;
  phase 4's RTX timings were contaminated by a concurrent job. They prove harness correctness only.
- **Phase 4 rows compare within a task only** (~3 % node-to-node band); the 0.5 ms threshold sits
  about 2× above the control rows' noise.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the four phases with their pre-registered rules (issues #268, #273, #295, #303),
job ids, results and PRs, and the carried items outside the map. Resolved: the headline 31.64 ms is
job 343350 (phase 1), not the four jobs the stub listed; 343376 is historical-only. Library PRs: none
shipped. Open: no Mind draft exists for the reproducibility study, the batch-aware callback or the
fp32 policy decision.
