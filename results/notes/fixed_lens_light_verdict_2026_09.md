# Fixed lens light — the programme verdict: the whole likelihood on HST and Euclid (2026-09)

autolens_profiling issue [#259](https://github.com/PyAutoLabs/autolens_profiling/issues/259),
branch `feature/fixed-light-verdict` (stacked on phase 4's `feature/fixed-light-scaling`,
itself on phases 3, 2, 1 and 0 — a stack of six). **Phase 5 of the
`fixed-lens-light-profiling` epic, and the last.** Phases 0–4 measured one axis at a time:
the kernels, the whole library call, the hardware, the model quality, the source-pixel
scaling. All of them on HST. This phase is the assessment the human actually asked for —
*for each hardware type, what does the whole likelihood cost on real survey data, and which
configuration should production use* — and it is the first time anything in this programme
has touched **Euclid**.

**Twenty-four legs**, one at a time on a quiet laptop (Intel i9-10885H under WSL2, 8 cores,
16 GB; NVIDIA GeForce RTX 2060 Max-Q, 6 GB, driver 580.97), each from a fresh JAX
compilation cache: the whole `AnalysisImaging.log_likelihood_function` under `jax.jit`, four
routes per leg, on **HST (0.05", 15 361 masked pixels)** and **Euclid (0.1", 3 841)** at
**500 / 1250 / 2500** source pixels. **The twelve A100 legs of the grid were not run** — the
RAL jump host refused publickey for the entire session — and that gap is stated everywhere
it matters rather than filled by extrapolation.

## The four results

1. **Euclid is where positivity earns its keep, and HST never showed it.** Dropping
   positivity costs **+3.5 → +24.3 → +109.1 nats** on Euclid as N goes 500 → 2500 (Delaunay)
   against a flat **+8.4 → +7.3 → +6.4** on HST, with **129** negative source pixels at
   N=2500 against HST's **6**. The reason is the constraint ratio, not the instrument: at
   N=2500 Euclid has **1.5 image pixels per source pixel** where HST has **6.1**. Every
   earlier phase measured the shortcut on HST and found single-digit nats on Delaunay; on
   the survey this programme exists to serve, the same shortcut is two orders of magnitude
   worse.
2. **The certified active set is the production solver on every hardware and both datasets —
   and its lead is *larger* on Euclid.** Against what the library runs today it is
   **1.20–1.59×** on HST and **1.43–2.28×** on Euclid, and on Euclid the lead *grows* with N
   (CPU 1.46× → 1.85× → 2.28×). It returns the library's own solution: every equivalence pin
   PASSED on every fp64 leg, Euclid included, at 1e-9 relative. **Its cost in nats is zero.**
3. **The pass budget is dataset-dependent, and Euclid eats the rectangular one exactly.**
   The rectangular mesh certifies at **6 / 9 / 11** passes on Euclid against **5 / 7 / 10** on
   HST — phase 3's safe budget of **11**, with nothing left, at the largest size measured.
   Delaunay keeps its margin on both (**Euclid 2 / 4 / 4**, HST 1 / 2 / 1, against a budget of
   7). Phase 4 concluded the budget does not scale with N; it does not, but it *does* move
   with the dataset, and **the rectangular budget of 11 must not be fixed on Euclid without
   a draw-set sweep of its own**.
4. **Nothing is memory-bound and nothing is precision-bound.** The worst peak on a 6 GB card
   is **2.70 GB** (HST, N=2500, fp64); Euclid at N=2500 is **1.09 GB**. Mixed precision buys
   **4–9 %** for ≤ 1.3e-4 nats and up to 11 % more memory, and on the Delaunay mesh its
   certifying budget matched fp64 at every configuration. **fp64 stays the path**, as phases
   2 and 4 said, and the reason is now that the saving is too small to be worth a withdrawn
   pin rather than that it costs accuracy.

## Scope — read this before quoting a number

- **What was measured.** Twenty-four legs of
  `scripts/imaging/likelihood_breakdown/fixed_light_library.py`, sequentially, each from a
  fresh JAX compilation cache, four routes each:

  | Routes | |
  |---|---|
  | **a** | S0 PDIP — the system **and** the solver the library runs today |
  | **b** | S3 PDIP — source-only, the reference every S3 route is scored against |
  | **c** | S3 positive-negative (`xp.linalg.solve`) — positivity dropped |
  | **d** | S3 certified active set at phase 3's **production** budget (11 rect / 7 Delaunay) with the PDIP fallback |

  - **18 Delaunay legs** — {HST, Euclid} × {500, 1250, 2500} × {RTX 2060 fp64, RTX 2060
    mixed precision, JAX-CPU at `NPROC=8` with BLAS pinned to 1};
  - **6 rectangular legs** — {HST, Euclid} × {500, 1250, 2500} on the RTX 2060 in fp64,
    added when the A100 legs could not be submitted, so the mesh axis would not go
    unmeasured on Euclid entirely.
- **The A100 legs were NOT run.** `ssh` to the RAL jump host (`finan.ncl.ac.uk`) returned
  `Permission denied (publickey)` on every attempt of this session, and the direct route to
  `euclid-saas.roe.ac.uk` times out. The four arrays are written, validated (`bash -n`,
  `check_submits.py`) and committed; the resume is one command on the RAL login node:
  `hpc/batch_gpu/submit_fixed_light_verdict.sh --node euclid-ral-gpu-2` (12 legs). **Every
  A100 number in this note is carried from phases 1 and 4 and labelled as such. This phase
  measured none of it, and nothing here is extrapolated to fill the gap.**
- **`--dataset` changes the preset and nothing else.** Pixel scale and dataset directory.
  The 3.5" circular mask, the `[4, 2, 2]` / `[0.3, 0.6]` light-profile over-sampling, the
  60 × 1 MGE, the mass and shear priors and the regularization are byte-identical across
  datasets. That is what makes the two columns one table rather than two experiments — and
  it also means **these are not the Euclid pipeline's production presets** (which run a
  20 × 2 MGE and `[4, 4, 2]` bins); they are this cell's configuration, on Euclid's data.
- **Euclid has no pins, and HST at these N has none either.** The phase-0 S0 log-det pins
  describe HST at the fiducial mesh in fp64. Every leg of this grid therefore **records**
  rather than asserts: a `reference_recorded` block in each JSON carries the log
  determinants, the evidences and the certifying budget as data — the first Euclid reference
  for this family, and what a later phase would pin against. Nothing was skipped: the
  equivalence pins are *identities*, not calibrations, and were asserted on every fp64 leg.
- **`--source-pixels` is a request.** The rectangular mesh builds `round(√N)²` —
  **484 / 1225 / 2500** — and the Delaunay/Hilbert mesh places exactly **500 / 1250 / 2500**
  vertices. Every table carries the built count.
- **What was NOT measured.** No A100 (above). No `delaunay_nn` (phases 0–3 show it tracks
  Delaunay). No JWST, no sparse operator — both out of scope for the whole epic by the
  human's own clarification, the sparse path additionally blocked on the PyAutoArray
  weight-map bug. No batched `@vmap` row: phase 4 measured its amortisation falling 3.6× →
  0.7× between N=500 and N=4000 on an A100 and answered the batching question there. No
  rectangular leg on the CPU or under mixed precision. **No library implementation** — the
  certified scheme is still a harness monkeypatch of
  `inversion_util.reconstruction_positive_only_from`, and every JSON says so under
  `solver_injection`.
- **No PyAutoArray change**, exactly as phases 0–4.
- **Compare within a mesh, never across.** Regularization is `Constant(1.0)` on rectangular
  and `adapt_split` on Delaunay, each cell's shipped default.

## Provenance and gate

| Legs | Device | Config | Precision | Pins | Exit | Wall |
|---|---|---|---|---|---|---|
| 6 Delaunay | RTX 2060 Max-Q 6 GB | `local_rtx2060_fp64_fixed_light_library` | fp64, single call | recorded | 0 ×6 | 39 s – 3 m 32 s |
| 6 Delaunay | RTX 2060 Max-Q 6 GB | `local_rtx2060_mp_fixed_light_library` | mixed, `--pins none` | recorded | 0 ×6 | 58 s – 3 m 30 s |
| 6 Delaunay | JAX-CPU i9-10885H, NPROC 8 / BLAS 1 | `local_cpu_fp64_fixed_light_library_t8` | fp64, single call | recorded | 0 ×6 | 52 s – 5 m 49 s |
| 6 rectangular | RTX 2060 Max-Q 6 GB | `local_rtx2060_fp64_fixed_light_library` | fp64, single call | recorded | 0 ×6 | 35 s – 3 m 22 s |
| **12 A100** | **euclid-ral-gpu-2** | `hpc_a100_fp64_fixed_light_library` | fp64 | — | **NOT SUBMITTED** | — |

**Counts: legs run = 24; result JSONs written = 24; non-zero exit = none; OOM = none;
wall-clock kill = none (RTX cap 45 min, CPU cap 30 min, worst leg 5 m 49 s); pin FAILED =
none.** Total wall clock 45 minutes.

| Gate | Result |
|---|---|
| route **d** (certified @ the production budget) == route **b** (library PDIP) | **PASS** on all 18 fp64 legs, max relative difference **5.6e-10** — Euclid included |
| route **b** (S3) == route **a** (S0) — fixing the light is a re-arrangement | **PASS** on all 18 fp64 legs, ≤ 5.7e-10 |
| route **c** == the unconstrained Cholesky of the same system | **PASS** on all 18 fp64 legs, ≤ 1.4e-12 |
| mapper-block log-dets S3 == S0 (rtol 1e-6) | **ASSERTED, PASS** on every fp64 leg, both datasets |
| the same pins on the 6 mixed-precision legs | **RECORDED**, not asserted (`tau_rel` re-derived to 1.4775e-5 on HST / 7.388e-6 on Euclid) |
| S0 log-det pins | **RECORDED, not asserted** — no pin exists for any configuration in this grid |
| cross-hardware agreement | the S3 log likelihood agrees to **≤ 3.0e-9** relative between the RTX 2060 and the CPU in fp64, and to ≤ 3.9e-8 including the mixed-precision legs, at every (dataset, N) |
| tie back to phase 4 (timing) | HST Delaunay N=2500, RTX fp64, route b: **2 241 ms here against 2 225 ms** in phase 4's independent leg (0.7 %) |
| tie back to phase 4 (evidence) | HST rectangular route c: **+318.00 / +347.11 nats** at N=484 / 2500 here, against phase 4's **+318.0 / +347.1** measured on the kernel path — the same problem reached two ways |

## The grid — ms per whole library likelihood call

### Delaunay (the production mesh), certified at pass budget 7

| Hardware | Dataset | N | **a** S0 today | **b** S3 PDIP | **c** S3 unconstrained | **d** S3 certified | a ÷ d | certifying budget |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A100 fp64 | HST | — | *not run this phase* | | | | | |
| RTX 2060 fp64 | HST | 500 | 171.3 | 138.4 | 113.0 | **123.4** | 1.39× | 1 |
| RTX 2060 fp64 | HST | 1250 | 631.2 | 555.4 | 407.9 | **450.4** | 1.40× | 2 |
| RTX 2060 fp64 | HST | 2500 | 2 289.0 | 2 241.0 | 1 449.4 | **1 654.5** | 1.38× | 1 |
| RTX 2060 fp64 | Euclid | 500 | 105.8 | 75.3 | 45.2 | **55.8** | 1.90× | 2 |
| RTX 2060 fp64 | Euclid | 1250 | 308.0 | 271.5 | 131.9 | **173.5** | 1.78× | 4 |
| RTX 2060 fp64 | Euclid | 2500 | 1 417.7 | 1 213.4 | 476.4 | **730.7** | 1.94× | 4 |
| RTX 2060 mp | HST | 500 | 155.2 | 128.4 | 101.8 | 111.8 | 1.39× | 1 |
| RTX 2060 mp | HST | 1250 | 597.3 | 511.3 | 374.3 | 425.4 | 1.40× | 2 |
| RTX 2060 mp | HST | 2500 | 2 208.4 | 2 216.9 | 1 380.3 | 1 584.2 | 1.39× | 1 |
| RTX 2060 mp | Euclid | 500 | 95.8 | 74.8 | 50.4 | 51.2 | 1.87× | 2 |
| RTX 2060 mp | Euclid | 1250 | 299.1 | 263.9 | 121.5 | 165.5 | 1.81× | 4 |
| RTX 2060 mp | Euclid | 2500 | 1 404.3 | 1 252.1 | 491.6 | 661.9 | 2.12× | 4 |
| JAX-CPU 8 thr | HST | 500 | 482.0 | 438.9 | 341.8 | **378.8** | 1.27× | 1 |
| JAX-CPU 8 thr | HST | 1250 | 1 940.8 | 1 688.0 | 1 302.1 | **1 420.5** | 1.37× | 2 |
| JAX-CPU 8 thr | HST | 2500 | 8 271.8 | 7 025.5 | 4 190.8 | **5 189.5** | 1.59× | 1 |
| JAX-CPU 8 thr | Euclid | 500 | 221.1 | 175.4 | 133.7 | **151.1** | 1.46× | 2 |
| JAX-CPU 8 thr | Euclid | 1250 | 1 107.7 | 976.7 | 440.0 | **599.5** | 1.85× | 4 |
| JAX-CPU 8 thr | Euclid | 2500 | 5 642.3 | 4 900.9 | 1 536.4 | **2 471.0** | 2.28× | 4 |

### Rectangular, certified at pass budget 11 (RTX 2060 fp64 only)

| Dataset | N built | **a** S0 today | **b** S3 PDIP | **c** S3 unconstrained | **d** S3 certified | a ÷ d | certifying budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| HST | 484 | 153.3 | 123.4 | 106.2 | **113.7** | 1.35× | 5 |
| HST | 1225 | 544.4 | 468.4 | 391.0 | **443.4** | 1.23× | 7 |
| HST | 2500 | 1 984.3 | 1 796.9 | 1 385.7 | **1 648.1** | 1.20× | 10 |
| Euclid | 484 | 83.1 | 60.8 | 50.0 | **52.5** | 1.58× | 6 |
| Euclid | 1225 | 241.7 | 199.6 | 122.8 | **169.4** | 1.43× | 9 |
| Euclid | 2500 | 1 049.5 | 837.4 | 455.5 | **717.0** | 1.46× | **11 — the safe budget exactly** |

### A100 — what is known, and from where

Nothing in this row was measured by phase 5. Carried, labelled:

| Source | Row | Value |
|---|---|---|
| phase 1 (#251), A100, HST, Delaunay n=1500 | a S0 today → d certified (at the *fiducial* budget 2) | 65.10 → 25.39 ms (2.56×) |
| phase 4 (#257), A100, HST, Delaunay | whole S3 call at N = 500 / 1500 / 2500 / 4000 | 15.7 / 49.4 / 93.4 / 165.2 ms |
| phase 4 (#257), A100, HST, Delaunay | certified **solve** at the safe budget 7 | 3.83 / 10.8 / 18.4 / 32.1 ms |
| *arithmetic on the two phase-4 rows*, **not a measurement** | whole call with the certified solve, N=2500 | *≈ 57 ms* |

The Euclid A100 column does not exist in any phase.

## What dropping positivity costs — in nats, which is the only unit that settles it

Route **c** against the library's own S3 solution, Delaunay, identical on every hardware:

| Dataset | image pixels | N | image px per source px | Δlog-evidence | negative pixels | negative flux |
|---|---:|---:|---:|---:|---:|---:|
| HST | 15 361 | 500 | 30.7 | **+8.35** | 3 | 0.0028 % |
| HST | 15 361 | 1250 | 12.3 | **+7.31** | 4 | 0.0014 % |
| HST | 15 361 | 2500 | 6.1 | **+6.35** | 6 | 0.0012 % |
| Euclid | 3 841 | 500 | 7.7 | **+3.51** | 16 | 0.0502 % |
| Euclid | 3 841 | 1250 | 3.1 | **+24.35** | 51 | 0.2196 % |
| Euclid | 3 841 | 2500 | **1.5** | **+109.13** | **129** | 0.4379 % |

On the rectangular mesh the same comparison runs +318.0 / +319.9 / +347.1 nats (HST) and
+105.9 / +117.1 / +140.7 (Euclid) — rectangular's 152-pixel edge-zero seed means positivity
was never optional there, on either dataset.

**A positive Δ is not an improvement.** The unconstrained solution is a different, infeasible
minimiser sitting above the constrained optimum, and turning positivity off silently turns
edge zeroing off with it (`Inversion.solve_ids_to_keep` returns `None`,
`abstract.py:540`). The reading is: *the faster route is wrong by this many nats*.

**The trend reverses between the datasets and that is the phase's real discovery.** On HST
the error *falls* as the mesh grows (+8.4 → +6.4) because 15 361 image pixels keep
over-constraining even 2 500 source pixels. On Euclid it *explodes* (+3.5 → +109.1) because
3 841 image pixels do not. Phase 4 found the error growing slowly with N on HST; on Euclid
the growth is a factor 31 over a factor 5 in N. **Any pipeline that reached for the
unconstrained solve because "Delaunay only costs six nats" was reading an HST number.**

## Memory — still not the wall

Peak device bytes, single call, `memory_stats()['peak_bytes_in_use']`:

| Dataset | N | RTX 2060 fp64 | RTX 2060 mixed precision |
|---|---:|---:|---:|
| HST | 500 | 0.54 GB | 0.54 GB |
| HST | 1250 | 1.42 GB | 1.49 GB |
| HST | 2500 | **2.70 GB** | **2.99 GB** |
| Euclid | 500 | 0.27 GB | 0.27 GB |
| Euclid | 1250 | 0.55 GB | 0.55 GB |
| Euclid | 2500 | 1.09 GB | 1.09 GB |

The JAX-CPU backend reports no peak. **The 6 GB card has 3.3 GB spare at the largest
configuration in this grid**, which reproduces phase 4's finding at a second dataset: the
consumer wall phase 2 hit belongs to the **batched** shape (`@vmap 16`, 11.88 GiB), not to
the problem. What ends every curve here is time.

## Mixed precision — a 4–9 % discount nobody should take

| Dataset | N | d fp64 | d mixed | saving | Δlog L | certifying budget fp64 / mp |
|---|---:|---:|---:|---:|---:|---:|
| HST | 500 | 123.4 | 111.8 | 9.4 % | −2.2e-5 | 1 / 1 |
| HST | 1250 | 450.4 | 425.4 | 5.6 % | +8.6e-6 | 2 / 2 |
| HST | 2500 | 1 654.5 | 1 584.2 | 4.2 % | +1.3e-4 | 1 / 1 |
| Euclid | 500 | 55.8 | 51.2 | 8.3 % | +6.3e-5 | 2 / 2 |
| Euclid | 1250 | 173.5 | 165.5 | 4.6 % | −2.0e-5 | 4 / 4 |
| Euclid | 2500 | 730.7 | 661.9 | 9.4 % | −1.3e-4 | 4 / 4 |

Phase 4 found mixed precision "certifying" earlier than fp64 on the rectangular mesh, at a
`tau_rel` re-derived to 1.4775e-5 — a looser tolerance, not a better solver. **On Delaunay
that gap does not appear at all**: the certifying budget is identical in both precisions at
every configuration here, on both datasets. The verdict is unchanged but its reason is
sharper: mixed precision is declined because 4–9 % is not worth withdrawing every
fp64-calibrated pin, not because it is inaccurate.

## The production configuration, per hardware

Read as: *the solver, the budget, the precision and the source-pixel count this hardware can
afford*, with the ms/call it costs and the nats it costs against **a** — the system and
solver the library runs today. In every case the certified route's evidence is **identical
to the library's own** (≤ 1.2e-9 relative, asserted on every fp64 leg), so **the nats cost
of the recommended configuration is zero** (5.6e-10 relative on a log evidence of order 1e4 is
~1e-5 nats); the nats column exists because the *alternative*
(dropping positivity) is not free.

### A100 (RAL `gpu-2`, fp64) — configuration stated, timings owed

- **Solver:** S3 certified active set, **pass budget 7** (Delaunay) with the PDIP fallback.
- **Precision:** fp64. **Batch:** single call above N ≈ 2000, `@vmap 16` below it (phase 4:
  the amortisation is 3.6× at N=500 and a 1.5× *penalty* at N=4000).
- **Affordable N:** phase 4 measured **4000 source pixels on HST** at 165 ms per whole S3
  call; this phase adds that **Euclid costs 2.1–2.6× less than HST at equal N** on every
  hardware measured, so Euclid at 2500 is comfortably inside it.
- **ms/call:** *not measured by this phase.* Phase 4's A100 rows and the arithmetic estimate
  above (≈ 57 ms at N=2500, HST) are what exist. **The twelve legs that would settle it are
  written and unrun; that is this phase's one open gap.**
- **Cost in nats vs a:** 0.

### RTX 2060 (6 GB consumer laptop GPU, fp64)

- **Solver:** S3 certified active set, **pass budget 7** (Delaunay) with the PDIP fallback.
- **Precision:** **fp64** — mixed precision buys 4–9 % and costs every calibrated pin.
- **Batch:** single call. `@vmap 16` does not fit this card at any N (phase 2).
- **Affordable N:** **HST 1500** (634 ms per call interpolated between two measured points;
  the call crosses 1 s at N ≈ 1900) and **Euclid 2500** (731 ms measured). Memory is not the
  constraint at either: 2.70 GB of 6 GB at the largest.
- **ms/call:** HST **123 / 450 / 1 655 ms** and Euclid **56 / 174 / 731 ms** at N = 500 /
  1250 / 2500 — against 171 / 631 / 2 289 and 106 / 308 / 1 418 for the library today.
- **Cost in nats vs a:** 0. The unconstrained alternative would save a further 1.12–1.53×
  and cost +6.4 nats (HST) to **+109 nats** (Euclid, N=2500).

### JAX-CPU (8 threads, `NPROC=8`, BLAS pinned to 1)

- **Solver:** S3 certified active set, **pass budget 7** (Delaunay) with the PDIP fallback.
  Phase 2's caveat stands and is unaffected by this phase: this is the **JAX** row. The
  *numpy* certified kernel does not beat the library's own `fnnls` on a CPU, so nothing here
  recommends a numpy active set.
- **Precision:** fp64 (no mixed-precision CPU leg was run).
- **Affordable N:** **HST 1000** (the call crosses 1 s at N ≈ 980) and **Euclid 1250**
  (599 ms measured; 1 s at N ≈ 1600).
- **ms/call:** HST **379 / 1 421 / 5 190 ms**, Euclid **151 / 600 / 2 471 ms**.
- **Cost in nats vs a:** 0.

### The mesh, and the one budget that is not safe

**Delaunay stays the production mesh** on both datasets — not because it is faster (at
matched N the two are within a few per cent on this card: HST 2500 1 655 vs 1 648 ms,
Euclid 2500 731 vs 717 ms, rectangular marginally ahead) but because of what it costs to run
*safely*. Its certified lever is larger (1.78–1.94× against rectangular's 1.43–1.58× on
Euclid, so it is the mesh the new solver actually helps), it certifies inside a budget of 7
rather than 11, and it keeps headroom on both datasets. **Rectangular on Euclid is the exception this phase found**: it certifies at
exactly 11 at N=2500 — phase 3's safe budget, with zero margin, on the first Euclid leg ever
run. A production run that fixes 11 there is one model away from paying the fallback. Either
run Delaunay, or re-derive the rectangular budget on Euclid over a graded draw set the way
phase 3 did on HST.

## What the programme did not measure

- **The A100 legs of this phase** (above). Twelve legs, one command, written and committed.
- **Any library implementation.** Every certified number in all six notes comes from a
  harness monkeypatch. There is no `Settings` option, no fallback policy, no batched
  cond-free path in PyAutoArray, and nothing in this programme has run inside a real search.
- **The sparse operator**, out of scope by the human's clarification and blocked on the
  PyAutoArray weight-map bug (`draft/bug/autoarray/sparse_inversion_ignores_profile_subtracted_image.md`).
- **JWST (0.03")**, dropped by the same clarification.
- **The Euclid pipeline's own presets.** This grid runs the cell's configuration on Euclid's
  data (60 × 1 MGE, `[4, 2, 2]` bins, `Constant`/`adapt_split`), not the pipeline's 20 × 2
  MGE with `[4, 4, 2]` bins. A production Euclid cost will differ, and the *ratios* in this
  note are what transfer, not the absolute milliseconds.
- **Model quality on Euclid.** Phase 3's graded draw set — which is what makes a budget
  "safe" — was run on HST only. The Euclid certifying budgets here come from one model each.
- **`delaunay_nn`**, the CPU rectangular column, and any mixed-precision rectangular leg.

## Verdict — the whole programme, in five lines

1. **Fix the lens light, solve only the source, and use the certified active set with a
   fixed budget and a PDIP fallback.** It is 1.2–2.3× faster than what the library runs
   today on every hardware and both datasets, and it returns the library's own answer to
   1e-9. Nothing in six phases argues against it.
2. **Budget 7 on Delaunay, and do not fix 11 on rectangular for Euclid** until a draw set
   says so. The budget is a property of the model (phase 3) and of the dataset (this phase),
   and not of N (phase 4).
3. **fp64 everywhere, single call above N ≈ 2000, `@vmap` only where it still amortises.**
4. **Never drop positivity.** The shortcut that looked like six nats on HST is **+109 nats
   on Euclid at 2500 source pixels**, and it is the survey case that matters.
5. **Affordable N: 4000 (A100, from phase 4) / 1500 HST and 2500 Euclid (RTX 2060) / 1000
   HST and 1250 Euclid (8 CPU threads).** Time is the constraint; memory never was.

## Next — filed as `ideas.md` bullets, not prompts

The verdict implies four follow-ups, all recorded in `PyAutoMind/ideas.md` under
`[from: fixed-lens-light-profiling phase 5]`:

- **Implement the certified solver in the library** as a `Settings` option with the PDIP
  fallback and a cond-free batched path — every number in this programme comes from a
  monkeypatch, and the fallback's `lax.cond` is the thing that breaks under `vmap`.
- **The matched-injection witness before any A2 use** — if anyone ever proposes the
  unconstrained solve, it must be measured on the matched dataset first; the HST number is
  not the Euclid number.
- **The Delaunay mapper/mesh cost is now the dominant term.** ≈ 21 ms of phase 1's 25 ms
  certified A100 call was not the solver, and phase 4 found the dense `F + λH` build
  overtaking the solve above N ≈ 2500. The next lever is the assembly, not positivity.
- **Re-derive the rectangular pass budget on Euclid** over a graded draw set, as phase 3 did
  on HST — this phase's 11 is a single-model measurement sitting exactly on the limit.

## Artifacts

```
results/breakdown/imaging/fixed_light_library_delaunay[_euclid]_n{500,1250,2500}_local_rtx2060_fp64_fixed_light_library.{json,png}
results/breakdown/imaging/fixed_light_library_delaunay[_euclid]_n{500,1250,2500}_local_rtx2060_mp_fixed_light_library.{json,png}
results/breakdown/imaging/fixed_light_library_delaunay[_euclid]_n{500,1250,2500}_local_cpu_fp64_fixed_light_library_t8.{json,png}
results/breakdown/imaging/fixed_light_library_rectangular[_euclid]_n{484,1225,2500}_local_rtx2060_fp64_fixed_light_library.{json,png}
```

Code:

```
scripts/imaging/likelihood_breakdown/fixed_light_library.py   # phase 1's cell + --dataset, --pass-budget auto, --routes, reference_recorded
scripts/misc/test/test_fixed_light_verdict.py                 # 30 tests (suite 398)
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_verdict_{pixelization,delaunay}_a100_{hst,euclid}_fp64
hpc/batch_gpu/submit_fixed_light_verdict.sh                   # 4 arrays x 3 arms = 12 A100 legs — WRITTEN, NOT RUN
```

JSON keys new in phase 5: `configuration.dataset`, `configuration.pass_budget_mode`,
`configuration.safe_budget`, `configuration.safe_budget_basis`,
`configuration.routes_selected`, and the `reference_recorded` block.

## The phases

| # | Phase | Note |
|---|---|---|
| 0 | Kernel measurement, A100 | [`fixed_lens_light_source_only_2026_09.md`](./fixed_lens_light_source_only_2026_09.md) |
| 1 | Library-path row, A100 | [`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md) |
| 2 | CPU + consumer GPU | [`fixed_lens_light_hardware_2026_09.md`](./fixed_lens_light_hardware_2026_09.md) |
| 3 | Low-likelihood draws | [`fixed_lens_light_low_likelihood_draws_2026_09.md`](./fixed_lens_light_low_likelihood_draws_2026_09.md) |
| 4 | Source-pixel scaling | [`fixed_lens_light_source_pixel_scaling_2026_09.md`](./fixed_lens_light_source_pixel_scaling_2026_09.md) |
| **5** | **HST + Euclid verdict** | **this note** |
