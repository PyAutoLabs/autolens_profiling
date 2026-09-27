# Numpy deflections CPU speed-up

**Status:** complete
**Question:** How much faster can the plain-numpy (CPU) deflections of the mass profiles the numba likelihood evaluates go?
**Pre-registered rule:** epic goal set at birth (2026-09-02, from the baseline probe, before any library edit): every one of the nine numpy `deflections_yx_2d_from` ≥ 2× on the HST grid, the four `*Sph` ≥ 100×, gNFW/gNFWSph ≥ 5×; deflection pins rtol 1e-6 (re-pin only where the old routine is the less accurate one, against `mpmath`, decision 5); numba likelihood pins unchanged (hst rect 27661.910133664103, rtol 1e-6). Per-phase targets in the table.
**Verdict:** epic COMPLETE 2026-09-03. Met: IsothermalSph / PowerLawSph / NFWSph 142–710× on the direct `Grid2D` call, gNFWSph 58–67× after phase 2, PowerLaw 5.7×, Isothermal 2.1×, NFWSph 1.9×. Short: gNFW 2.4–3.1× against 5× and elliptical Gaussian 1.7–1.8× (phase 2 re-scoped both to measured ceilings; the Faddeeva call is irreducible), NFW 1.6× against 2× (HK24 polynomial arithmetic is the floor). Every pin held or was re-pinned with provenance under decision 5.
**Headline:** phase 3 NFW 2.97 → 1.83 ms (1.6×, hst `Grid2D`, the one closed-form profile under the 2× line), Claude Code web container (4 cores, `OMP_NUM_THREADS=1`); no RAL job (phases 1–2 ran on the laptop).
**Library PRs:** PyAutoArray#516, PyAutoGalaxy#595, PyAutoLens#718, PyAutoGalaxy#597, PyAutoArray#519, PyAutoGalaxy#599 (all released 2026.9.4.1).
**Profiling PRs:** #210, #212, #213.
**Ledger:** [numpy_deflections_cpu.md](../../results/notes/numpy_deflections_cpu.md) (parts "phase 1" to "After phase 3"; the same file also holds the JAX-path audit and the successor's memo sections).
**Mind contract:** epic `numpy-deflections-cpu`; `complete/archive/epics/numpy_deflections_cpu_speedup.md`; records `complete/2026/09/numpy-deflections-p{1,2,3}.md`
**Next:** none (successor: [Gaussian deflections precompute](gaussian_deflections_precompute.md))

## Why this campaign

After the numba CPU likelihood route reached 0.33 s per HST rectangular evaluation (21.3 s at the
start of that work), the human asked on 2026-09-02 for the plain-numpy (`xp=np`) deflections it
runs to be tuned for CPU: ten named profiles (GaussianSph turned out to be a light profile only, so
nine), no new public functions, one shared `xp` body unless a numpy/JAX split buys a clear win, and
the profiling coded out in a new `scripts/lens/` package (verbatim request in the epic ledger).
The birth probe (HST grid, 15,361 points) found the four `*Sph` profiles at 0.48–0.83 s/call against
~1 ms of physics (a decorator re-materialising `Grid2D.over_sampled` on every call) and gNFW at 0.202 s
in the MGE-30 Faddeeva kernel.

The Brain scored the work too-large and split it into three phases, one Mind prompt and one issue each
(PyAutoArray#514, PyAutoGalaxy#596, PyAutoGalaxy#598), library-first inside each phase. The human
approved decisions 1–5 on 2026-09-02, including `scripts/lens/` as a second top-level axis
(decision 4) and the `mpmath` re-pin policy (decision 5). Phase 1 carries the measurement package
every later phase reports through.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 measure + `*Sph` + double trace | 2026-09-02 | Land `scripts/lens/deflections/`, then two zero-numerics fixes: the `*Sph` over-sampled re-materialisation and the tracer's double trace | baseline committed before any library edit; `*Sph` ≥ 300×; tracer ≈ 2× on sub-size-1 grids; pins bit-for-bit at rtol 1e-6 | direct `Grid2D`: IsothermalSph 699 → 0.99 ms (710×), PowerLawSph 570×, NFWSph 142×, gNFWSph 4× (MGE remains); tracer 1.7–3.9× every profile; all pins held, no `--repin`; likelihood-side `Inversion build` moved 1–2 ms, inside noise (negative result) | laptop | PyAutoArray#516, PyAutoGalaxy#595, PyAutoLens#718, #210 |
| 2 MGE / Faddeeva | 2026-09-02 | `scipy.special.wofz` on numpy, an exact spherical MGE branch, a cached decomposition | gNFW ≥ 5×, gNFWSph ≥ 20×, Gaussian ≥ 1.5× on top of phase 1; re-pin only via `--repin --repin-reason` against `mpmath`; likelihood pins unchanged | gNFW 293 → 96 ms (2.4–3.1× over three passes, short), gNFWSph 301 → 4.5 ms (58–67×), Gaussian 1.7–1.8×, Gaussian(q=1) 6–9×; cache lever dropped (0.34 ms/call); `dark`/`stellar` re-pinned, `total` untouched | laptop | PyAutoGalaxy#597, #212 |
| 3 closed form + geometry | 2026-09-03 | PowerLaw series with a `factor`-driven term count, NFW/NFWSph masks, Isothermal hoists, rotation-matrix grid transform | epic 2× line for the closed-form profiles; no re-pin expected (a moved pin is a finding); series re-verified against `mpmath.hyp2f1`; likelihood pins unchanged | hst `Grid2D` PowerLaw 9.59 → 1.68 ms (5.7×), Isothermal 2.1×, NFWSph 1.9×, NFW 1.6×, Gaussian(q=1) 2.0×; euclid 4.3 / 1.7 / 1.6 / 1.6 / 2.0×; series 5.7e-11 vs `mpmath`; `stellar` re-pinned for one on-axis sample that became exactly 0.0 | web container (no RAL) | PyAutoArray#519, PyAutoGalaxy#599, #213 |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#516 | `to_grid` reads `_over_sampled` / `_over_sampler`; `Grid2D.over_sampled` short-circuits at sub-size 1 | `52b84f5e` | 2026.9.4.1 |
| PyAutoGalaxy#595 | `Galaxy.traced_grid_2d_from` skips the second deflection call at sub-size 1 | `e76c062e` | 2026.9.4.1 |
| PyAutoLens#718 | `Tracer.traced_grid_2d_list_from` skips the second ray-trace at sub-size 1 | `0fd9fd33` | 2026.9.4.1 |
| PyAutoGalaxy#597 | scipy `wofz` on numpy (`_wofz_rational` kept for JAX), `Gaussian.wofz` deduped, `_wofz_masked`, exact spherical MGE branch | `8d152b15` | 2026.9.4.1 |
| PyAutoArray#519 | rotation-matrix `transform_grid_2d_to_reference_frame` (7× cheaper); `VectorYX2D` reuses its `Grid2D` | `62feb7eb` | 2026.9.4.1 |
| PyAutoGalaxy#599 | PowerLaw omega series (Horner, `n_terms` from the tail bound), NFW subset `capital_F_from`, real NFWSph, Isothermal hoists | `e5fb32f7` | 2026.9.4.1 |

## Open / parked / drafts

- none inside the epic.
- The JAX-path follow-up the epic filed at phase 2 shipped separately as `complete/2026/09/jax-faddeeva-clamp-audit.md` (PyAutoGalaxy#600 → PyAutoGalaxy#603, merge `50599c2c`, released 2026.9.4.1; #215). Its sections live in this campaign's ledger; it has no wiki page of its own.
- `draft/bug/autogalaxy/mge_deflections_reverse_mode_nan_at_grid_centre.md` — filed by that audit.
- `PowerLawSph` returns 2 non-finite deflections at the exact centre (phase 1 trap, "library follow-up"): no Mind draft found.
- NFW beyond 1.6× would need a fused two-component expression; named in the ledger, not attempted, not filed.

## Caveats

- **Two hosts.** Phases 1–2 ran on the laptop (WSL2), phase 3 in a Claude Code web container (~10 % slower on PowerLaw, ~35 % on gNFW). Ratios are same-machine within each phase; do not chain them across phases or compare absolute ms between the two hosts.
- **Laptop variance ~±30 % between passes** (gNFWSph 0.88 / 1.04 / 1.20 s): non-MGE rows in the phase-2 table (PowerLaw 0.87×, hst NFW 1.58× on untouched code) are noise. The web container read ~±10 %.
- **The `*Sph` 570–710× is not a likelihood win.** `tracer_util` wraps each plane in `Grid2DIrregular`, so production ray-tracing never paid the ~500 ms; the likelihood-side phase-1 win is the double-trace removal, which scales with the profile (16–440 ms per evaluation for PowerLaw/gNFW models at tracer level).
- **Phase 1's own `*Sph` ≥ 300× line** was met by IsothermalSph and PowerLawSph only; NFWSph (142×) cleared the epic's ≥ 100× line, gNFWSph was left to phase 2.
- **The epic close-out says "only NFW missed its 2× line"**; against the epic's nine-profile goal, gNFW and elliptical Gaussian are also short (their phase-2 targets, 5× and 1.5×, were re-scoped to measured ceilings).
- **Re-pins went past decision 5's envelope.** Decision 5 allowed ~3e-6 moves from the Faddeeva swap (scipy shifts measured ≤ 3.97e-6); the spherical branch also removed the q = 0.9999 clamp bias, moving gNFWSph / Gaussian(q=1) pins by up to 7.23e-5 and two cross-axis samples to exactly 0 (`--repin-force`), each with a recorded mechanism. Phase 3 force-re-pinned one exact-zero `stellar` sample. The likelihood pin literal is written as 27661.910133664103 (epic gate) and 27661.910133665442 (phase 2–3 runs); both pass rtol 1e-6.
- **Breakdown artifacts were not overwritten** in any phase (load-contaminated re-runs); only pin verdicts come from those runs. `jax_compile` warm compile 321 → 268 ms under unequal load is inconclusive and not re-pinned.
- The ledger's library SHAs (`92bb9b2c`, `09785e32`, `755e43d1`, …) are pre-merge branch commits; the merge SHAs above come from the release sheet.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the epic goal as the pre-registered rule, all three phases with their per-phase targets,
and six library PRs with merge SHAs and releases (PyAutoGalaxy#597, PyAutoArray#519 and PyAutoGalaxy#599 were
missing; all six released 2026.9.4.1). Verdict written from the phase records; the NFW 1.6× headline is
located on the phase-3 web container, not RAL. Open: the JAX-path audit (#215, PyAutoGalaxy#603, released 2026.9.4.1) is folded
into this page as a follow-up rather than given its own page; no RAL row exists for any phase.
