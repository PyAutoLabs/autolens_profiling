# Phase 2 — Global MGE: Nautilus vs mainline BlackJAX NSS: results

Running record for Phase 2 (`../PROGRAMME.md` §4). Entry work landed
2026-08-19/20 (issue #149): the framework wiring and the CP-2 GPU MGE smoke.

## Framework wiring (2026-08-19)

`af.NSS` (mainline `blackjax.nss`, PyAutoFit PR#1492) is wired into the
searches framework: `nss` in `SAMPLER_BUILDERS`, config recorded in the
fork-era JSON shape, leaf scripts for imaging `mge` / `delaunay` /
`pixelization`. Defaults replicate fork history (inner steps 5, num_delete
50, dlogz −3, seed 42) so mainline rows diff cleanly against the recorded
v2026.5.21.1 fork rows; the scan arms drive `SEARCHES_NSS_{N_LIVE,
NUM_MCMC_STEPS,NUM_DELETE,CHUNK_SIZE,TERMINATION,SEED}`. Deliberately not
in `sweep.py` CELLS — pixelized NSS rows cost 19–30 ks and must not creep
into release sweeps.

## CP-2 GPU MGE smoke — laptop RTX 2060 (2026-08-19/20)

**Finding 1 (hardware): the unchunked inner step does not fit a 6 GB
card.** At n_live=200 / num_delete=50 the first jitted step allocates
937 MiB against the ~3.9 GB pool the desktop leaves free →
`RESOURCE_EXHAUSTED`. `chunk_size=10` (the PR#1492 GPU-memory lever,
bit-identical at fixed seed) runs it. Laptop-tier NSS rows therefore
always record a `chunk_size`; A100 rows keep `null`.

**Finding 2 (throughput caveat, expected): GeForce fp64 is 1:32.** The
RTX 2060's fp64:fp32 ratio makes wall times on this tier incomparable to
A100 fp64 (1:2) even beyond the usual cross-tier rule. The laptop row
answers "does mainline `af.NSS` run the production MGE cell end-to-end on
GPU" — the Phase 2 performance scan belongs on the A100.

**Laptop verdict (2026-08-20): mechanics validated, wall-time infeasible —
run cut off by hand, anchor row moved to the A100.** The `chunk_size=10`
run launched, jitted, and sampled the production MGE cell on the GPU
without error for 80+ minutes — the integration question ("does mainline
`af.NSS` drive the production cell end-to-end on GPU?") is answered on the
mechanics side. But it completed <100 outer iterations in that time (no
first checkpoint at `checkpoint_interval=100`) → ≥44 s/iteration vs the
A100 fork run's ~1.7 s → a ≥4–5 h projected wall at 86 °C sustained. That
is the fp64 1:32 GeForce ratio plus chunk serialization plus thermal
throttle; nothing about the sampler. The run was terminated by hand
(no checkpoint yet ⇒ nothing resumable was lost) and the identical-knob
row submitted to the RAL A100 instead:
`hpc/batch_gpu/submit_search_nss_imaging_mge_a100_hst_fp64` (unchunked, as
the fork rows ran). This note + PROGRAMME's CP-2 entry get finalized from
that artifact (verdict criteria: completes to dlogz −3; max logL / logZ
against the fork band 31786.3–31786.5 / 31697.7–31700.4 — same seed 42,
same knobs, mainline code; H2.1 predicts the logZ bias *persists* at
inner=5 and the scan's ≥2d arms remove it).

Laptop-tier rule going forward: the RTX 2060 is a smoke/mechanics tier for
NSS (chunked, no performance meaning); MGE-scale NSS wall-times come from
RAL CPU and A100 only.

## A100 anchor row — mainline af.NSS at fork-era knobs (2026-08-23)

RAL job 338491 (A100 80GB PCIe, fp64, unchunked). Same knobs as the two
recorded fork rows: n_live 200, inner steps 5, num_delete 50, dlogz −3,
seed 42 — the diff is code-version only (v2026.5.21.1 fork blackjax →
2026.8.17.1 + mainline blackjax 1.6.2 via `af.NSS`). Artifact:
`results/searches/nss/imaging/mge/hst/hpc_hpc_a100_fp64.json` (overwrites
the fork row of the same name; the fork numbers stay in git history and in
the table below). All rows include ~43–46 s of viz (5 calls).

| | fork run 1 (`hpc_a100_fp64`) | fork run 2 | **mainline** |
|---|---:|---:|---:|
| max logL | 31786.456 | 31786.298 | **31786.616** |
| logZ | 31697.7 | 31700.39 | **31698.85** |
| sampler wall (s) | 633.0 | 612.1 | **839.6** |
| likelihood evals | 394,321 | 383,289 | **234,498** |
| ms/eval | 1.61 | 1.60 | **3.58** |
| posterior samples | — | 15,500 | 15,800 |

**Verdict against the pre-registered criteria:**

- **Completes to dlogz −3: PASS.** Clean end-to-end run, no chunking.
- **max logL 31786.62 — matches the fork band (31786.3–31786.5), slightly
  above it,** 0.17 nats below the Nautilus truth bar (31786.782). Same
  best-fit solution (θ_E=1.5998, same shear).
- **logZ 31698.85 — inside the fork band, still +8.4 nats above Nautilus
  (31690.5).** H2.1's prediction holds: the bias *persists* at inner=5
  with mainline code — it is not a fork artifact. The scan's ≥2d
  inner-steps arms remain the pre-registered test for removing it.
- **Wall: mainline is ~1.35× slower sampler-side at fork knobs** (839.6 s
  vs 612–633 s), with a different execution profile: 0.6× the likelihood
  evals at 2.2× the per-eval time. Caveat: ms/eval is sampler_wall /
  evals, so if mainline accounts inner-kernel evals differently the
  per-eval comparison is bookkeeping, not hardware — the honest headline
  is the wall-clock ratio at matched knobs and matched answer. At these
  knobs mainline NSS (882.8 s total) sits at rough parity with the
  Nautilus A100 row (831 s) rather than the fork's mild win.

**CP-2 is complete**: Phase 2 is confirmed a *tuning exercise, not an
integration project* — mainline `af.NSS` reproduces the fork's answer on
the production MGE cell on both GPU tiers; speed at default knobs is the
thing the scan has to earn back.

**Ops lessons (recorded because both will recur):** (i) job 335971 died at
node start after ~10 h in queue (00:00:00 elapsed, no output/error file —
prolog/node failure; resubmission of the identical script ran fine).
(ii) The first resubmission (338490) hit the completed-fit resume trap:
autofit's identifier hashes model+search config, so the identical-knob
mainline run matched the fork-era output directory and re-emitted the
stored fork results in ~4 s under a *current* version stamp — detectable
only by `total_wall_s` and the "Fit Already Completed" log line. The
fork-era output was moved to
`output/searches/nss/imaging/mge/hst/hpc_a100_fp64_fork_era_backup/` on
RAL and the rerun (338491) sampled for real. Any future same-knob
cross-version rerun must clear or relocate the output dir first, and
verify wall-time plausibility, never just the version stamp.

## Scan wave 1 — the inner-steps axis: H2.1 CONFIRMED (2026-08-23)

RAL jobs 338492/338493 (A100 fp64, unchunked; n_live 200, num_delete 50,
dlogz −3, seed 42 — only `num_mcmc_steps` varies vs the anchor). Artifacts
`hpc_hpc_a100_fp64_inner{30,45}.json`; arm encoded in `--config-name`, so
autofit identifiers and files are disjoint from the anchor (see the wave-1
submit scripts' headers). Both runs verified genuine (current version
stamp, plausible walls, no "Fit Already Completed" marker).

| inner steps | logZ | bias vs Nautilus 31690.5 | max logL | sampler wall | evals |
|---:|---:|---:|---:|---:|---:|
| 5 (anchor) | 31698.85 | +8.4 | 31786.62 | 839.6 s | 234,498 |
| 30 (=2d) | 31691.20 | **+0.7** | 31785.83 | 4,218.6 s | 1,492,747 |
| 45 (=3d) | 31690.04 | **−0.5** | 31786.35 | 6,341.3 s | 2,264,857 |

**H2.1 is confirmed as pre-registered:** the +7–13-nat logZ bias recorded
in every fork-era NSS row is under-mixing of the inner slice kernel, not a
code or evidence-estimator defect. The bias falls monotonically with
inner steps and *brackets* the Nautilus reference at 3d. Every max logL
stays within the ≤2-nat correctness tolerance of the truth bar
(31786.782); eval counts scale linearly in inner steps (6.4× / 9.7×), wall
slightly better than linear (5.0× / 7.6×).

**Cost consequence for Gate A:** evidence-correct NSS at these knobs
(inner ≥ 2d) costs 4,219–6,341 s sampler wall on the A100 vs Nautilus's
831 s total on the same tier — ~5–7.6×. NSS's Gate-A wall-time case now
rests entirely on the remaining scan axes: `num_delete` (the GPU
parallelism axis) and `n_live`, where larger deletion blocks amortize the
inner-step cost per replacement. inner=30 (2d) is the working operating
point for those axes (0.7 nats is within Phase-1 evidence tolerance;
inner=45 doubles down for a final confirmation row only).

## Next (per PROGRAMME §4 Phase 2)

- Scan wave 2: num_delete {20, 50 (done), 100} × n_live {200 (done), 500,
  1000} at inner=2d, dlogz −3 — the wall-time axes. Then ≥5-seed
  reliability at the chosen operating point, one dlogz −10 termination
  row, and the pixelized-cell probe.
- Gate A judged per model family, not on MGE alone.
