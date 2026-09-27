# Numba interferometer revisit

**Status:** complete
**Question:** Is the numba CPU interferometer likelihood (removed in favour of JAX) worth reinstating in PyAutoArray?
**Pre-registered rule:** phase 2 kill gate, written before the bake-off: "if no numba variant beats `rfft2` by >1.3× at sma *or* alma, stop the numba lever work"; every kernel pinned to the recovered kernel on `F` at `rtol=1e-10` before its time counts. Phases 1 and 3: not recorded (parity pins only).
**Verdict:** kill gate passed. (d) switch `apply_operator` to `rfft2`/`irfft2` regardless; (a) reinstate a numba CPU curvature path on the new `direct_conv` kernel (not the recovered pair loop), gated on nnz per source column below ≈60–80; add a NumPy no-JAX application path. Phase 3: the preload is exactly a type-1 NUFFT, library follow-up justified (architect's call). Epic COMPLETE 2026-09-08.
**Headline:** `direct_conv` 5.93× faster than NumPy `rfft2` per sma Delaunay curvature build (0.2135 s vs 1.2660 s), laptop i9-10885H WSL2, single thread, no RAL job; quiet-machine re-run 5.49× — the host drifted 2.37–2.83× between runs while this ratio moved 8 %.
**Library PRs:** PyAutoArray#540, PyAutoArray#541, PyAutoArray#544, PyAutoArray#545 (released 2026.9.11.1).
**Profiling PRs:** #225 (phase 1, issue #223), #228 (phase 2, issue #226), #234 (phase 3, issue #229).
**Ledger:** [numba_interferometer_verdict.md](../../results/notes/numba_interferometer_verdict.md)
**Mind contract:** epic `numba-interferometer-revisit`; `complete/archive/epics/numba_interferometer_likelihood_revisit.md`; phase records `complete/2026/09/numba-interferometer-pack.md`, `numba-interferometer-kernel-levers.md`, `interferometer-preload-cpu.md`.
**Next:** none (successor: [Interferometer likelihood](interferometer_likelihood.md))

## Why this campaign

Filed 2026-09-07 by the human after the imaging numba work gave ~10× on CPU: the numba
interferometer likelihood had been removed 6–12 months earlier in favour of JAX, but many users
have no GPU. The request was to recover the old code (remembered as "secret"), build a breakdown
mirroring the imaging numba one, judge whether numba is worth reinstating, and treat the
curvature preload as its own line item (epic ledger, verbatim request). The archaeology found the
algebra identical to today's `InterferometerSparseOperator` (same preload, `F = Mᵀ W~ M`,
`D`, `fast_chi_squared`); only the application differs, and `W~` has no compact support for an
interferometer, so the imaging win's sparsity is not available.

A 2026-09-07 design review (synthetic, i9-10885H) predicted a direct extent-grid convolution
winning below ~35–45 nnz/col and a 1.6–2.2× `rfft2` gain; phase 2 was written to reproduce it
under a kill gate. PyAutoArray stayed read-only for all three phases; library changes went out as
follow-up prompts. Precursor for imaging: the PyAutoArray#513 research verdict
(`complete/2026/09/numba-vs-jax-sparse.md`, "same linear algebra, deliberately different
machines; keep both, port nothing", no code).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 Pack + breakdown | 2026-09-07 | Recover the numba w-tilde likelihood as a pack, bit-identical to the JAX sparse path | not recorded (parity pins: log evidence, `F`, `D`, ×1.01 control must fail) | log evidence bit-identical (`-3169.6493766794806`); sma 0.515 s Delaunay / 0.451 s rect; `F` row 45 % / 82 % of the call | laptop | #225 |
| 2 Kernel levers + verdict | 2026-09-07 | Can the `F` row be made fast enough to justify reinstating numba? | kill gate: a numba kernel > 1.3× `rfft2_numpy` at sma or alma; `rtol=1e-10` pin before timing | PASS: `direct_conv` 5.93× / 2.83× (sma D / rect), 1.54× / 0.99× (alma); in situ `F` row 7.0× / 4.1× / 2.0× / 1.10× vs JAX-CPU; crossover nnz/col ≈ 60 D / ≈ 77 rect; `rfft2` 1.27–1.61× over complex `fft2` | laptop | #228 |
| 3 Preload | 2026-09-08 | Is the preload exactly a type-1 NUFFT, and what does building it that way cost? | not recorded (array pin `10·eps·P[0,0]`; log-evidence pin `rtol=1e-6`) | exact: alma 2101.5 s → 7.278 s (289× wall, 111× CPU-s); log evidence bit-identical at alma; recovered numba builder fails the alma array pin (2.2e-11) | laptop | #234 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#540 | `apply_operator` via exact `rfft2`/`irfft2` (issue PyAutoArray#538, verdict (d)) | `7a4cb700` | 2026.9.11.1 |
| PyAutoArray#541 | preload built as a type-1 NUFFT, `eps=1e-12` default (issue PyAutoArray#539, phase 3) | `9bd76799` | 2026.9.11.1 |
| PyAutoArray#544 | NumPy/scipy application path, no JAX on `xp=np` (issue PyAutoArray#542) | `39d3024c` | 2026.9.11.1 |
| PyAutoArray#545 | numba `direct_conv` curvature path, gate `interferometer_numba_nnz_per_source_max` = 60 (issue PyAutoArray#543) | `35aa681f` | 2026.9.11.1 |

Records: `complete/2026/09/interferometer-apply-operator-rfft2.md`,
`interferometer-preload-nufft-type1.md`, `interferometer-sparse-operator-numpy-cpu-path.md`,
`interferometer-numba-cpu-direct-conv.md`. PyAutoArray#545 changed the default CPU route: an
`xp=np` single-mapper inversion below the gate now runs numba. The in-situ re-measurement of the
gate through the library dispatch continued as [Interferometer likelihood](interferometer_likelihood.md)
mesh CPU phases 1–2 (crossover ≈ 66 Delaunay / ≈ 72 rect, ledger section 8).

## Open / parked / drafts

- none for this epic. The A100 confirmation of the `rfft2` path (PyAutoArray#540) was waived at
  merge and is recorded as outstanding in `complete/2026/09/interferometer-apply-operator-rfft2.md`.
- Gate retune 60 → 70 (from the successor campaign): `draft/feature/autoarray/interferometer_numba_gate_retune_70.md`.

## Caveats

- **Every row is a laptop row.** i9-10885H under WSL2; no RAL job ran in this epic. The host
  throttled: a quiet re-run of the sma cell (`bakeoff_machine_drift_sma_v2026.8.17.1.json`) made
  all nine kernels 2.37–2.83× faster while the deciding `rfft2 / direct_conv` ratio moved 5.93× →
  5.49×. Quote ratios; absolute seconds compare only within a cell.
- **alma_high pair-loop cells are extrapolated**, not measured (`reference`, `hoisted`,
  `source_loop` rect recorded as `skipped: extrapolated`).
- **The JAX in-situ whole-likelihood column is not a production number**: eager
  `InversionInterferometerSparse` rebuilt `F` per uncached property (`steps_are_partial`); the
  jit-warm `F` row is the comparator. The same uncached-property duplication was fixed in the
  library later (PyAutoArray#582, successor campaign).
- **Thread scaling is a ceiling**: kernel threads are unavailable under a Nautilus pool; the
  single-thread number is the production number.
- **Phase 3 wall vs CPU-seconds**: the NUFFT builder ran on ~2.6 cores at alma; 289× is wall, 111×
  is CPU-s. `control_dgemm_s` drifted 0.2550 → 0.3162 between the brute-force and NUFFT runs.
- **The crossover moved in situ**: ≈ 60 / ≈ 77 here (prototype pack, rect S = 784) vs ≈ 66 / ≈ 72
  on the library dispatch (RAL EPYC 7702, rect 39×39); the gate constant is machine-dependent.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the three phases, the phase-2 kill gate and its pass, the four library PRs
(all released 2026.9.11.1), and the headline ratio with its host. Resolved: #226 is the phase-2
issue, not a PR (PR is #228, confirmed on GitHub); phase 3 (#229 / PR #234) belongs to this epic;
issue PyAutoArray#543 shipped as PR #545, #542 as PR #544. Open: no RAL row exists for this epic's
own measurements, and the `rfft2` A100 check is still outstanding.
