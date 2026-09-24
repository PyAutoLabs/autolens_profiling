# Certified positive solver, phase B — library solver policy under `jit(vmap)`

Issue [#300](https://github.com/PyAutoLabs/autolens_profiling/issues/300). Follows phase A
(PyAutoArray#566 / PR #567, the certified active-set solver shipped opt-in behind
`al.Settings(positive_only_solver="certified")`), the phase-2 harness vmap measurement
([hst_gpu_residue_phase2_vmap_2026_09.md](hst_gpu_residue_phase2_vmap_2026_09.md)) and the phase-3
PSF note ([hst_gpu_residue_phase3_psf_2026_09.md](hst_gpu_residue_phase3_psf_2026_09.md)), which
placed phase 2's ~2.5e-9 residual between the `jit(vmap)` and scalar-jit compositions rather than in
the solver.

## Verdict

**The pre-registered gate passes on all 20 tasks: 0 failed lanes, 0 uncertified lanes, every
fallback-none row policy-eligible.** Measured on the library mains with no monkeypatch:

- Under the production composition `jax.jit(jax.vmap(fn))`, **certified + PDIP fallback is slower
  than today's library PDIP at every B on both meshes** (B16: 45.6 vs 38.9 ms/lane Delaunay,
  38.6 vs 30.7 rectangular). The batched `lax.cond` is a select, so every lane pays both solvers.
  This row must not become the vmap default.
- **Certified + fallback none is the fastest batched row**: at B16 it is 1.50x (Delaunay) and 1.25x
  (rectangular) faster per lane than library PDIP `jit(vmap)`, and 1.97x / 1.65x faster than today's
  scalar PDIP `jit(fn)`. Against the scalar *certified* control it wins only at B16 (1.14x / 1.15x);
  at B=8 it ties and at B=4 it loses.
- In the scalar composition the fallback is free: the certified scalar arm is the same ~28–30 ms/lane
  with the fallback on or off (a real `lax.cond`), versus 51–53 ms (Delaunay) and 40 ms
  (rectangular) for scalar library PDIP.

Timing is never gated. The policy below was **ADOPTED by the human on 2026-09-24** ("This sounds good, follow the proposed plan."), as proposed, with no amendments.

## Experiment

- Job: RAL SLURM array `350588`, tasks 0–19, submitted 2026-09-24T08:42:21Z, all `COMPLETED 0:0`
  (1m20s–3m10s each) on `euclid-ral-gpu-1` / `euclid-ral-gpu-2`.
- Profiling source: `24e734c` (feature/certified-solver-phase-b); every JSON's
  `cell_source_sha256` (`9874c635…`) is the cell at that commit.
- Library mains: PyAutoNerves `1fa613aa`, PyAutoFit `a7368401`, PyAutoArray `11b93476` (contains
  #567), PyAutoGalaxy `70a61e26`, PyAutoLens `2aaa1c1a`. Unreleased.
- Device: NVIDIA A100 80GB PCIe; fp64; HST imaging (0.05"/px); fixed lens light at S3; dense
  inversion; border relocation on; N=1500 (Delaunay) and the cell's rectangular fiducial.
- Solver chosen through `al.Settings` (`--solver-source library`): `positive_only_solver`
  pdip | certified, `certified_fallback` pdip | none, `certified_pass_budget` = the resolved library
  default, 16. The cell asserted `positive_only_solver_used` and the traced solver kwarg equal the
  request on every task.
- Matched pair per task: the same B distinct lanes (seeded draws, seed 0) run as one
  `jax.jit(jax.vmap(fn))` call and as B sequential `jax.jit(fn)` calls, one process, one Settings.
  Tasks 18–19 are identical-lane controls, not results.
- Fresh compilation cache per task; `AUTOTUNE_ENTRIES count=0` in all 20 footers. No traceback in
  any `.out` or `.err`.
- Matched rows within one mesh x B triplet did not always land on the same node (e.g. Delaunay B16
  PDIP on gpu-2, certified+none on gpu-1). The scalar controls agree across nodes to ~3%
  (certified scalar 29.2–30.3 ms Delaunay, 27.9–28.3 ms rectangular), which bounds that effect well
  below the differences reported.

Provenance, per-task sacct rows, footers and artifact checksums:
[certified_solver_policy_phase_b_job350588.json](certified_solver_policy_phase_b_job350588.json).
The 20 JSON/PNG pairs are under `results/breakdown/imaging/` with suffix
`_jitvmap<B>_<lanes>_lib<solver>_fb<on|off>_b16_hpc_a100_fp64_fixed_light_trace`.

## Numerical gate (pre-registered at `24e734c`, before submission)

Gated at 1e-9 relative, per lane: the `jit(vmap)` arm vs the `jit(vmap)` library-PDIP reference, and
the scalar arm vs the scalar-jit library-PDIP reference. Recorded, not gated: the cross-composition
difference (relative and nats), the two references against each other, and `uncertified_lanes`.

| mesh | B | row | worst gated vmap rel | worst gated scalar rel | cross-comp rel | cross-comp nats | ref cross-comp rel | uncertified | passes | eligible |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| delaunay | 4 | library PDIP | 1.80e-10 | 5.85e-11 | 6.46e-10 | 1.31e-05 | 5.83e-10 | n/a | n/a | yes |
| delaunay | 4 | certified+PDIP | 3.23e-10 | 7.54e-11 | 6.55e-10 | 1.43e-05 | 6.77e-10 | 0 | 4-7 | yes |
| delaunay | 4 | certified+none | 3.40e-10 | 1.12e-10 | 6.85e-10 | 1.37e-05 | 5.38e-10 | 0 | 4-7 | yes |
| delaunay | 8 | library PDIP | 9.08e-11 | 2.69e-10 | 2.61e-09 | 1.68e-05 | 2.41e-09 | n/a | n/a | yes |
| delaunay | 8 | certified+PDIP | 1.71e-10 | 2.94e-10 | 2.02e-09 | 1.32e-05 | 2.48e-09 | 0 | 3-7 | yes |
| delaunay | 8 | certified+none | 2.79e-10 | 1.57e-10 | 2.47e-09 | 1.64e-05 | 2.38e-09 | 0 | 3-7 | yes |
| delaunay | 16 | library PDIP | 1.64e-10 | 2.56e-10 | 2.12e-09 | 1.73e-05 | 2.27e-09 | n/a | n/a | yes |
| delaunay | 16 | certified+PDIP | 3.28e-10 | 2.47e-10 | 2.22e-09 | 1.96e-05 | 2.16e-09 | 0 | 3-7 | yes |
| delaunay | 16 | certified+none | **5.21e-10** | 3.69e-10 | 2.11e-09 | 1.80e-05 | 2.48e-09 | 0 | 3-7 | yes |
| delaunay | 16 | CONTROL identical, cert+PDIP | 1.63e-10 | 1.13e-10 | 5.40e-10 | 1.58e-05 | 5.14e-10 | 0 | 2 | yes |
| rectangular | 4 | library PDIP | 0 | 0 | 2.49e-14 | 1.46e-11 | 2.49e-14 | n/a | n/a | yes |
| rectangular | 4 | certified+PDIP | 4.73e-13 | 4.73e-13 | 2.49e-14 | 1.46e-11 | 2.49e-14 | 0 | 6-11 | yes |
| rectangular | 4 | certified+none | 4.73e-13 | 4.73e-13 | 2.49e-14 | 1.46e-11 | 2.49e-14 | 0 | 6-11 | yes |
| rectangular | 8 | library PDIP | 0 | 0 | 1.25e-14 | 7.28e-12 | 1.25e-14 | n/a | n/a | yes |
| rectangular | 8 | certified+PDIP | 4.73e-13 | 4.73e-13 | 1.25e-14 | 7.28e-12 | 1.25e-14 | 0 | 6-11 | yes |
| rectangular | 8 | certified+none | 4.73e-13 | 4.73e-13 | 1.25e-14 | 7.28e-12 | 1.25e-14 | 0 | 6-11 | yes |
| rectangular | 16 | library PDIP | 0 | 0 | 7.46e-16 | 2.91e-11 | 7.46e-16 | n/a | n/a | yes |
| rectangular | 16 | certified+PDIP | 4.73e-13 | 4.73e-13 | 7.65e-16 | 2.91e-11 | 7.46e-16 | 0 | 6-11 | yes |
| rectangular | 16 | certified+none | 4.73e-13 | 4.73e-13 | 7.65e-16 | 2.91e-11 | 7.46e-16 | 0 | 6-11 | yes |
| rectangular | 16 | CONTROL identical, cert+PDIP | 4.70e-15 | 4.58e-15 | 0 | 0 | 1.27e-16 | 0 | 7 | yes |

Readings:

- The worst gated residual is 5.21e-10 (Delaunay B16 certified+none), under half the 1e-9 pin. On
  the rectangular mesh the certified solution sits 4.7e-13 from PDIP; the PDIP arms reproduce their
  references bit-for-bit.
- The Delaunay cross-composition residual, 2.0–2.6e-9 at B=8/16, is **equally present between the
  two library-PDIP references** (2.2–2.5e-9). It is a composition property of the Delaunay program,
  not the solver, which confirms phase 3's localisation. The rectangular program has no such residual
  (≤2.5e-14), so it is not the `vmap` transform itself.
- **Delaunay is not run-to-run deterministic at the ~2e-10 level.** The Delaunay library-PDIP arms
  differ from their separately compiled identical-program references by up to 1.8e-10 (rectangular:
  exactly 0), and in the identical-lane control the 16 lanes of one call disagree by 5.4e-6 nats
  (vmap) and 5.0e-6 nats (scalar, the same compiled program called 16 times on one input). The
  rectangular control lanes agree bitwise. This is below the gate and does not change any verdict,
  but it sets a floor on how tightly any Delaunay pin can be drawn; its source (atomic reductions,
  callback ordering) is not identified here.
- Certification: all 112 distinct-lane certified solves (and the 32 control lanes) certified within the packaged
  budget 16 (max passes 7 Delaunay, 11 rectangular, matching phase 3's safe budgets 7 / 11; the
  rectangular headroom at budget 16 is 5 passes). These lanes are one seeded draw family near the
  fiducial; they do not cover lanes far from the posterior mode.

## Matched timing (ms per completed likelihood, per lane)

Scalar certified is the mean of the fallback-on and fallback-off tasks' scalar arms (they agree to
within 1%, 30.30 vs 30.24 at Delaunay B4). "best vmap" is certified+none `jit(vmap)`.

| mesh | B | library PDIP `jit(vmap)` | certified+PDIP `jit(vmap)` | certified+none `jit(vmap)` | scalar PDIP `jit(fn)` x B | scalar certified `jit(fn)` x B | best vmap vs PDIP vmap | best vmap vs scalar PDIP | best vmap vs scalar certified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delaunay | 4 | 79.20 | 96.42 | 38.26 | 53.36 | 30.27 | 2.07x | 1.39x | 0.79x |
| delaunay | 8 | 51.82 | 62.28 | 29.73 | 51.53 | 29.72 | 1.74x | 1.73x | 1.00x |
| delaunay | 16 | 38.89 | 45.55 | 25.94 | 51.18 | 29.60 | 1.50x | 1.97x | 1.14x |
| rectangular | 4 | 59.56 | 81.60 | 39.26 | 39.77 | 27.97 | 1.52x | 1.01x | 0.71x |
| rectangular | 8 | 40.14 | 53.34 | 29.77 | 39.96 | 28.11 | 1.35x | 1.34x | 0.94x |
| rectangular | 16 | 30.68 | 38.56 | 24.52 | 40.43 | 28.23 | 1.25x | 1.65x | 1.15x |

Speedups greater than one favour the certified+none batch. Two further comparisons follow from the
table:

- Library PDIP `jit(vmap)` vs scalar PDIP (today's two defaults): batching loses at B=4 (0.67x),
  ties at B=8 and wins at B16 (1.32x Delaunay, 1.32x rectangular).
- certified+PDIP `jit(vmap)` vs library PDIP `jit(vmap)`: 0.82x / 0.83x / 0.85x (Delaunay B4/8/16)
  and 0.73x / 0.75x / 0.80x (rectangular) — the double-solver select cost, a structural loss at
  every B.

The phase-B library rows reproduce phase 2's harness rows at Delaunay B16 (certified+PDIP 45.6 vs
44.6; certified+none 25.9 vs 24.9; scalar certified 29.6–30.0 vs 31.1 at harness budget 7), so the
library port preserved the harness's cost profile.

Delaunay still pays one sequential qhull callback per lane (4.5–4.7 ms/lane host work in every
Delaunay task); the rectangular mesh took zero callbacks on every task, as asserted. vmap device-idle
per lane falls with B: 10.3 → 6.1 ms (Delaunay PDIP B4 → B16), 3.7 → 1.0 ms (rectangular).

## Policy — ADOPTED 2026-09-24

Proposed on #300 and adopted by the human on 2026-09-24 without amendment ("This sounds good,
follow the proposed plan."). The item 1 default change is a PyAutoArray config PR that waits for
the release that ships PyAutoArray#567. The item 4 follow-up is filed as its own PyAutoMind prompt.

1. **Scalar JAX (`jax.jit(fn)`, `use_jax_vmap=False`)**: make `positive_only_solver="certified"`
   with `certified_fallback="pdip"` the default. The fallback is a genuine `lax.cond` here and costs
   nothing on certified lanes; the gain is 1.7x (Delaunay) and 1.4x (rectangular) per likelihood
   with the full PDIP safety net. Budget 16 as packaged.
2. **Batched JAX (`jax.jit(jax.vmap(fn))`, `use_jax_vmap=True`)**: do **not** use certified+PDIP —
   it is slower than today's PDIP at every B. The two defensible choices are
   (a) keep library PDIP as the vmap default (no numerical risk, today's behaviour), or
   (b) certified with `certified_fallback="none"` (1.25–2.07x over PDIP vmap, gate-passing, 0
   uncertified lanes here) **only together with** a guard for the uncertified-lane case, because with
   `none` an uncertified lane silently returns the last active-set iterate. Adopted: (a) now, (b)
   once the guard in item 4 exists.
3. **NumPy backend**: unchanged. The library dispatches the certified solver on the JAX mapper path
   only; this grid measured no NumPy row.
4. **Follow-up prompt merited: a cond-free batched fallback.** The vmap row's only defect is the
   select. A fix outside the traced program — the vmapped likelihood also returns the per-lane
   `certified` flag, and PyAutoFit `Fitness._vmap` re-evaluates only the uncertified lanes through
   the scalar certified+PDIP program (a rare, host-side second pass) — would give the certified+none
   batch cost with PDIP semantics. That is a PyAutoFit Fitness batching change plus a small
   PyAutoArray/PyAutoGalaxy surface for the flag, and should be its own prompt. Before it is
   prioritised, a measurement of the uncertified-lane rate on real Nautilus batches (lanes spread
   across the prior, not near-fiducial draws) is needed: at a rate near zero (a) → (b) is cheap; at a
   high rate the second pass erodes the gain.
5. **Is batching worth it at all?** Against the scalar certified default proposed in item 1, the best
   batch wins only at B16 (1.14x / 1.15x) and loses at B=4. Any vmap policy should be stated together
   with the batch size Nautilus actually uses; small batches should stay scalar.
6. **Not changed by this grid**: the Delaunay run-to-run nondeterminism (~2e-10 rel,
   ~5e-6 nats) and the 2.5e-9 Delaunay cross-composition residual are properties of the Delaunay
   program under both solvers; neither blocks the policy, but any tighter Delaunay pin would first
   need them localised.
