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

## Scan wave 2 — the wall-time axes at inner=30, seeds, dlogz, pixelized probe (2026-08-24)

Overnight RAL A100 queue submitted 2026-08-23 ~22:30, harvested 2026-08-24
14:30 UTC. All jobs COMPLETED; every artifact carries the current version
stamp (2026.8.17.1), a plausible wall and no "Fit Already Completed" marker.
Nautilus re-baselines on the same stack ran alongside (see below) so every
ratio here is same-night, same-node.

### `num_delete` × `n_live` at inner=30, dlogz −3, seed 42

| arm (job) | n_live | num_delete | logZ | bias vs Nautilus 31690.50 | max logL | sampler wall | evals | ms/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inner30 (wave 1) | 200 | 50 | 31691.20 | +0.7 | 31785.83 | 4,219 s | 1,492,747 | 2.83 |
| nd20 (338871) | 200 | 20 | 31690.75 | +0.25 | 31786.04 | 6,138 s | 1,638,384 | 3.75 |
| **nd100 (338870)** | 200 | 100 | 31691.35 | +0.85 | 31786.29 | **3,528 s** | 1,236,644 | 2.85 |
| n500 (338872) | 500 | 125 | 31691.43 | +0.9 | 31786.62 | 9,966 s | 3,718,760 | 2.68 |
| n1000 (338873) | 1000 | 250 | 31690.21 | −0.3 | 31786.56 | 20,266 s | 7,519,030 | 2.70 |

- `num_delete` is the GPU-parallelism axis and it behaves as predicted:
  nd20 → nd100 cuts sampler wall 6,138 → 3,528 s (1.74×) at ~constant
  eval count and ms/eval, because the per-step vmap batch grows. Beyond
  nd100 = n_live/2 there is no further block to enlarge at n_live=200.
- `n_live` buys nothing on wall: evals and wall scale ~linearly (n500 2.8×,
  n1000 5.7× the nd100 wall). n1000 gives the closest logZ to Nautilus
  (−0.3) but at 29× the Nautilus sampler wall.
- **Operating point: n_live 200 / num_delete 100 / inner 30 / dlogz −3.**

### Seed reliability at the operating point (339067[0-3], seeds 43–46)

| seed | logZ | max logL | sampler wall |
|---:|---:|---:|---:|
| 42 | 31691.35 | 31786.29 | 3,528 s |
| 43 | 31690.98 | 31786.34 | 3,483 s |
| 44 | 31691.69 | 31786.29 | 3,449 s |
| 45 | 31691.53 | 31786.23 | 3,430 s |
| 46 | 31692.04 | 31786.25 | 3,456 s |

5/5 seeds land in the truth basin; logZ mean 31691.52, sample std 0.40,
range [31690.98, 31692.04]; max logL within 0.5 nats of the truth bar every
time; wall spread <3 %. The residual +1.0 ± 0.4 nat offset from Nautilus is
inside the seed scatter and inside the Phase-1 evidence tolerance — **H2.1
is closed: the fork-era +7–13-nat logZ bias was entirely inner-kernel
under-mixing. There is no evidence-estimator defect in mainline
`blackjax.nss`.**

### Termination row (339068, dlogz −10)

logZ 31691.81 / max logL **31787.33** (the highest NSS max logL on record,
0.55 above the Nautilus bar and matching the Prodigy +1.1 plateau to within
0.6) / 3,688 s (+4.5 % wall) / 15,400 posterior samples. Tighter
termination is nearly free and improves the MAP; it does not move logZ
beyond seed scatter. Adopt dlogz −10 for any row where the MAP matters.

### Pixelized probe — `imaging/delaunay/hst` mainline at fork knobs (339069)

| method | logZ | max logL | sampler wall | evals | ms/eval |
|---|---:|---:|---:|---:|---:|
| Nautilus fork-era row (2026.5.21.1) | 30562.22 | 30623.45 | 2,673 s | 31,536 | 84.8 |
| **Nautilus re-baseline (339071, 2026.8.17.1)** | 30562.24 | 30623.17 | **1,891 s** | 30,240 | 62.5 |
| NSS fork row (2026.5.21.1) | 30567.76 | 30622.15 | 29,721 s | 206,448 | 144.0 |
| **NSS mainline (339069, 2026.8.17.1)** | 30565.22 | 30624.13 | **34,726 s** | 150,991 | 230.0 |

Same answer (max logL 30624.13 — above both Nautilus rows; logZ +3.0 vs
Nautilus, consistent with the inner=5 under-mixing seen on MGE), same
pathology: mainline NSS is 1.17× the fork wall and **18.4× the Nautilus
re-baseline** on the pixelized cell (fork-era: 11×; the gap widened because
Nautilus got 1.4× faster on the current stack while NSS ms/eval rose 1.6×).
The per-eval cost (230 ms vs Nautilus 62 ms) says the deficit is
structural — the inner slice kernel's serial per-live-point evaluations
cannot be batched the way Nautilus's neural-network proposal is — not a
knob. No inner-steps scan was run on this cell (each row is a 10–12 h A100
slot); given the MGE result the bias would fall with inner steps and the
wall would grow further.

### Nautilus re-baselines on the current stack (339070, 339071, 339073)

| cell | logZ (fork-era → now) | max logL | sampler wall (fork-era → now) | n_batch |
|---|---|---:|---|---:|
| imaging/mge/hst | 31690.47 → 31690.50 | 31786.63 | 773 → **707 s** (total 831 → 775) | 100 → 64 |
| imaging/delaunay/hst | 30562.22 → 30562.24 | 30623.17 | 2,673 → **1,891 s** | 16 |
| imaging/pixelization/hst | 29066.32 → **29590.14** | 29670.38 (May: 29143.29) | 2,721 → 3,293 s (339795, 56 min) | 16 |

The mge and delaunay truth bars are reaffirmed to 2 dp on the current stack;
their fork-era Nautilus rows remain valid references. **The pixelization row
is NOT a reaffirmation: logZ moved +524 nats and max logL +527 because the
cell's mesh changed at the Phase-14 split (`_pixelization_model` now builds
`RectangularRTUAdaptImage`; the May row used `RectangularAdaptImage`). It is a
different target (different `target_id` under schema v2), so the May
pixelization row is retired as a reference and 29590.14 / 29670.38 is the
bar for the RTU cell going forward.** The MGE re-baseline settles the
"831/772.7 vs 523 s" wall discrepancy noted in phase_03: 707–773 s is the
reproducible fp64 A100 tier; 523 s is not reproduced on this stack and is
retired as a reference.

### Sample economy — ESS and reject-inclusive eval counts (2026-08-24)

Wall alone understates the gap. Kish ESS = (Σw)²/Σw² over each run's
stored `samples.csv` weights (autofit drops negligible-weight rows, so the
stored count is below the JSON `posterior_samples`). `likelihood_evals`
counts every likelihood call for both engines: for NSS that includes every
rejected slice proposal (~87 evals per dead point at inner=30, ~⅔ rejects);
for Nautilus every evaluated point is kept as a weighted sample and its
"rejects" are the ≈0-weight rows autofit drops.

| cell | engine / config | evals (incl. rejects) | stored (non-zero w) | ESS | ESS/n | sampler wall | ESS / min | evals per ESS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mge | **Nautilus** re-baseline | 62,208 | 10,586 | **4,121** | 0.39 | 707 s | **350** | **15** |
| mge | NSS fork knobs (inner 5, biased) | 234,498 | 3,125 | 1,611 | 0.52 | 840 s | 115 | 146 |
| mge | NSS op. point (inner30 nd100) | 1,236,644 | 2,725 | 1,315 | 0.48 | 3,528 s | 22 | 940 |
| mge | NSS nd100 seeds 43–46 | 1.24 M | 2,701–2,763 | 1,338–1,405 | 0.50 | 3,430–3,483 s | 23–25 | ~900 |
| mge | NSS nd100 dlogz −10 | 1,344,578 | 3,986 | 1,354 | 0.34 | 3,688 s | 22 | 993 |
| mge | NSS nd20 | 1,638,384 | 3,341 | 1,787 | 0.54 | 6,138 s | 17 | 917 |
| mge | NSS n500 | 3,718,760 | 7,820 | 4,236 | 0.54 | 9,966 s | 26 | 878 |
| mge | NSS n1000 | 7,519,030 | 15,558 | 8,397 | 0.54 | 20,266 s | 25 | 895 |
| delaunay | **Nautilus** re-baseline | 30,240 | 11,651 | **2,342** | 0.20 | 1,891 s | **74** | **13** |
| delaunay | NSS fork | 206,448 | 2,898 | 1,115 | 0.39 | 29,721 s | 2.3 | 185 |
| delaunay | NSS mainline | 150,991 | 3,163 | 1,271 | 0.40 | 34,726 s | 2.2 | 119 |
| pixelization | NSS fork | 266,043 | 2,431 | 1,114 | 0.46 | 19,142 s | 3.5 | 239 |

- NSS's ESS is pinned by `n_live` (ESS/n ≈ 0.50 on every inner/nd/dlogz
  arm; ESS scales linearly with n_live at linear wall). Matching Nautilus's
  4,100 ESS on MGE needs n_live ≈ 500 → 9,966 s = **14× the Nautilus wall
  for equal ESS**; per unit wall Nautilus is 16× (MGE) / 33× (Delaunay).
- Per likelihood call, Nautilus needs ~15 evals per effective sample on
  both cells; evidence-correct NSS needs ~940 (MGE, 60×) — the slice
  kernel walking a 269×-anisotropic, |r|=0.95 posterior one axis at a
  time. Nautilus's importance draws from a learned bound are near-i.i.d.
- Nautilus's evals are 4× dearer (11.4 vs 2.85 ms on MGE): the proposal
  step is host-side numpy/sklearn around n_batch=64 GPU batches. That
  overhead — not the eval economy — is the only thing a "JAX Nautilus"
  could improve (ceiling ~4× on MGE, ~1.3× on Delaunay); the cheap test is
  an `n_batch` scan (W6, queued).

### Gate A — CALLED 2026-08-24 (human): Nautilus stays the nested baseline

Evidence-correct mainline NSS matches Nautilus's answer everywhere and is
faster nowhere: 5.0× / 18.4× on wall, 14× for equal ESS, ~60× per
likelihood eval. **Nautilus remains the default nested sampler on every
model family; `af.NSS` stays mainlined as a correct, tuned alternative
(operating point n200 / nd100 / inner30 / dlogz −3, −10 when the MAP
matters).** Not measured, recorded as the only re-opening condition: a
GPU-only, likelihood-bound deployment where Nautilus's host-side proposal
is the bottleneck. Phase 5's NSS arm is dropped. Recorded in DECISIONS.md.

## Next

- Human Gate A call on the reading above (DECISIONS.md entry).
- Pixelization re-baseline 339795 pending (queue held by another user's
  8-GPU allocation at harvest time).
- If Gate A is to be re-opened on a non-wall criterion: one GPU-utilisation
  row per engine (nvidia-smi sampling) would decide whether Nautilus's wall
  is CPU-proposal-bound.
