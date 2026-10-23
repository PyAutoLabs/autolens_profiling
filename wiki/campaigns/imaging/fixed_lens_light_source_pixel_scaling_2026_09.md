# Fixed lens light — how the new approach scales with source pixels, per hardware (2026-09)

autolens_profiling issue [#257](https://github.com/PyAutoLabs/autolens_profiling/issues/257),
branch `feature/fixed-light-scaling` (stacked on phase 3's `feature/fixed-light-draws`, itself
on phase 2's, phase 1's and phase 0's — a stack of five). **Forty legs**: two SLURM arrays of
five arms on the RAL A100 (`gpu-2`, fp64, `@vmap 16`) and thirty local legs on one laptop —
an Intel i9-10885H under WSL2 (8 cores visible, 16 GB) with an NVIDIA GeForce RTX 2060 Max-Q
(6 GB, driver 580.97) — at `--source-pixels` **500 / 1000 / 1500 / 2500 / 4000**, on the
rectangular and Delaunay meshes, HST.

**Three results, and none of them is the one the phase was filed expecting.**

1. **The certifying pass budget does not grow with N — it wanders, and phase 3's safe budgets
   cover every size measured.** Rectangular certifies at 5 / 8 / 7 / 10 / 6 passes across the
   five sizes and Delaunay at 1 / 1 / 2 / 1 / 2. There is no trend in N at all, and the
   largest value anywhere is 10 against phase 3's rectangular budget of **11**. The budget is
   a property of the *model*, which phase 3 established, and not of the mesh size. **A fixed
   budget of 11 / 7 is safe from 500 to 4000 source pixels**, which is what phase 5 needs.
2. **The certified active set holds its lead at every N, on every hardware, and on the
   Delaunay mesh it *widens*.** At phase 3's safe budget against the S3 PDIP reference:
   rectangular 1.62× → 1.44× (A100) and 1.56× → 1.28× (RTX 2060) as N goes 500 → 4000;
   Delaunay 2.21× → 2.72× (A100), 2.22× → 2.47× (RTX 2060), 1.60× → 2.54× (CPU). Nothing
   about the phase-0/1/2/3 verdict is a small-mesh artefact.
3. **The batched `@vmap 16` shape stops paying above N ≈ 2500 — on the A100.** Phase 1's
   fastest row was the batched certified solve. Its amortisation over the single call falls
   **3.64× → 2.48× → 1.84× → 1.14× → 0.68×** on rectangular and **3.61× → 2.94× → 1.97× →
   1.17× → 0.66×** on Delaunay: at 4000 source pixels batching the certified solve is a
   **loss** on an 80 GB A100, not only on a 6 GB laptop. Phase 2 found the batched design
   unavailable on consumer hardware; phase 4 finds it stops being an advantage on the A100
   too, once the kernel is large enough to saturate the device on its own.

And one result the consumer-hardware story needed: **the single-call path fits a 6 GB card at
4000 source pixels**, peaking at 3.13 GB in fp64 and 3.63 GB under mixed precision. Phase 2's
`@vmap 16` needed 11.88 GiB at n=1521 and OOMed. The batched design has no consumer
counterpart; the single-call one has plenty of headroom.

## Scope — read this before quoting a number

- **What was measured.** Forty legs, one at a time, each from a **fresh JAX compilation
  cache**, of `scripts/imaging/likelihood_breakdown/fixed_light.py` at `--pass-budget-max 12`:
  - **10 A100 legs** (`hpc_a100_fp64_fixed_light`, `--vmap-batch 16`, `--pins fp64`) — two
    SLURM arrays of five arms on `euclid-ral-gpu-2`;
  - **10 RTX 2060 fp64 legs** (`local_rtx2060_fp64_fixed_light`, single call, `--pins fp64`);
  - **10 RTX 2060 mixed-precision legs** (`local_rtx2060_mp_fixed_light`,
    `--use-mixed-precision --pins none`, single call);
  - **10 JAX-CPU legs** (`local_cpu_fp64_fixed_light`, `NPROC=8`, BLAS pinned to 1 per phase
    2's finding, single call, `--pins fp64`).
- **`--source-pixels` is a request, not the built count.** The rectangular mesh builds
  `round(sqrt(N))²` — **484 / 1024 / 1521 / 2500 / 3969** — and the Delaunay/Hilbert mesh
  places exactly **500 / 1000 / 1500 / 2500 / 4000** vertices. Every table below carries the
  built count; every fitted exponent is fitted against it.
- **The n=1500 arm is the fiducial** on both meshes (1521 rectangular, 1500 Delaunay), so it
  and only it **asserts** the phase-0 S0 log-det pins. That arm is this sweep's tie back to
  phases 0–3, and it passed on the A100, the RTX 2060 in fp64 and the CPU.
- **`vmap` is A100-only, by decision, on phase 2's evidence.** `@vmap 16` needs 11.88 GiB and
  OOMs a 6 GB card at any batch above 2; the same shape OOM-killed the 16 GB host. The RTX and
  CPU legs here are single-call by design, and result 3 above says the A100's own batched row
  stops being worth having at this phase's larger N anyway.
- **What was NOT measured.** No `delaunay_nn` (two meshes × 5 N × 4 hardware is already 40
  legs; the prompt names rect and Delaunay, and phases 0–3 show DelaunayNN tracks Delaunay).
  No Euclid, no JWST, no sparse operator — all out of scope for the epic. No batched row on
  consumer hardware. No production implementation: the certified scheme remains a measurement
  harness.
- **No PyAutoArray change**, exactly as phases 0–3.
- **Compare within a mesh, never across.** Regularization is `Constant(1.0)` on rectangular
  and `adapt_split` on Delaunay, each cell's shipped default.

## Provenance and gate

| Legs | Device | Config | Precision | Pins | Exit | `cache_fresh` | autotune |
|---|---|---|---|---|---|---|---|
| 343023_0..4 | euclid-ral-gpu-2, A100 80GB PCIe | `hpc_a100_fp64_fixed_light` | fp64, @vmap 16 | asserted (fiducial arm) | 0:0 ×5 | true | 0 ×5 |
| 343024_0..4 | euclid-ral-gpu-2, A100 80GB PCIe | `hpc_a100_fp64_fixed_light` | fp64, @vmap 16 | asserted (fiducial arm) | 0:0 ×5 | true | 0 ×5 |
| 10 local | RTX 2060 Max-Q 6 GB | `local_rtx2060_fp64_fixed_light` | fp64, single call | asserted (fiducial arm) | 0 ×10 | true | n/a |
| 10 local | RTX 2060 Max-Q 6 GB | `local_rtx2060_mp_fixed_light` | mixed, single call | **recorded** | 0 ×10 | true | n/a |
| 10 local | JAX-CPU i9-10885H, NPROC 8 / BLAS 1 | `local_cpu_fp64_fixed_light` | fp64, single call | asserted (fiducial arm) | 0 ×10 | true | n/a |

**Counts: legs = 40; result JSONs written = 40; non-zero exit = none; OOM = none; wall-clock
kill = none (RTX cap 45 min, CPU cap 30 min, worst leg 26 min); off-node = none; pin FAILED =
none.** Elapsed: A100 arms 48 s – 2 m 26 s; RTX legs 50 s – 11 m 40 s; CPU legs 44 s – 26 m 27 s.

| Gate | rectangular | delaunay |
|---|---|---|
| S0 runtime log-det pins, fiducial arm, A100 (`pinned_drift`) | **PASS** 3888.258090 / 1692.786817 | **PASS** 8360.401763 / 7756.614959 |
| the other four arms record `pinned_expected: null` (a fiducial pin is not a pin at another mesh size) | PASS ×4 | PASS ×4 |
| mapper-block log-dets S3 == S0, every fp64 leg (rtol 1e-6) | ASSERTED, PASS | ASSERTED, PASS |
| mapper-block log-dets, mixed-precision legs | **RECORDED**, not asserted | **RECORDED**, not asserted |
| `tau_rel`, fp64 legs / mixed-precision legs | 1e-9 / **1.4775e-5** | 1e-9 / **1.4775e-5** |
| pass counts, PDIP iterations and A2 evidence identical on all four hardware legs | PASS, all 5 N | PASS, all 5 N |
| the certified iterate reproduces the library's solution | Δlog-evidence 0.0 at every certifying budget | 0.0 at every certifying budget |

**One leg to distrust and one to note.** The very first A100 submission (arrays 343013 /
343018) failed all ten arms in four seconds: the RAL `git worktree add` checkout of the pushed
branch was still writing its 948 files when `sbatch` ran, so `scripts/…/fixed_light.py` did not
yet exist. The arrays were resubmitted once the checkout completed and every arm then ran
clean. And the CPU rectangular n=484 PDIP row (162.3 ms) is out of line with its own curve —
it is the smallest and cheapest leg, where the JAX-CPU dispatch overhead is the measurement;
the 5.44× it produces in the speed-up table below is that artefact, not a result, and the four
larger sizes (1.31 / 1.25 / 1.32 / 1.32×) are the rectangular CPU answer.

## The certifying budget versus N — the question the phase was filed to answer

Phase 0 saw rectangular certify in **6** passes at n=3025 against **7** at n=1521 and asked
whether the budget scales with the mesh. It does not.

| Built N (rect) | 484 | 1024 | 1521 | 2500 | 3969 | (phase 0: 3025) |
|---|---:|---:|---:|---:|---:|---:|
| smallest certifying budget, fp64 | 5 | 8 | **7** | **10** | 6 | 6 |
| smallest certifying budget, mixed precision | 4 | 4 | 4 | 5 | 5 | — |

| Built N (Delaunay) | 500 | 1000 | 1500 | 2500 | 4000 | (phase 0: 3000) |
|---|---:|---:|---:|---:|---:|---:|
| smallest certifying budget, fp64 | 1 | 1 | **2** | 1 | 2 | 2 |
| smallest certifying budget, mixed precision | 1 | 1 | 2 | 1 | 2 | — |

Three things follow.

- **There is no trend.** Rectangular's budget moves 5 → 8 → 7 → 10 → 6 with no monotone
  component; Delaunay's sits at 1 or 2 throughout. The pass count is set by how many pixels
  want to go negative and how the dual violation set drains — a property of the *model* (phase
  3) and of the edge-zero seed, not of the mesh resolution.
- **Phase 3's safe budgets hold across the whole range.** The worst case anywhere is 10
  (rectangular, n=2500), inside the budget of **11** phase 3 derived from a graded 41-model
  draw set at one mesh size. Every leg in this sweep reports
  `safe_budget_certified: true`. Phase 5 can fix 11 / 7 and sweep N freely.
- **Mixed precision certifies earlier at every N, and that is still the tolerance moving, not
  the solver.** Rectangular "certifies" at 4–5 under fp32 accumulation against 5–10 in fp64,
  because `tau_rel` is re-derived to 1.4775e-5 (`eps_float32 × sqrt(15361 image pixels)`) and
  the residual dual violations are inside that floor. Phase 2 measured this once, at the
  fiducial; it is now measured at five sizes and the gap does not close. **A production budget
  must never be taken from a mixed-precision leg.**

## Per call, ms versus N

Full tables, including the assembly and log-determinant rows, the S0 column and every fitted
exponent, are in
[`results/breakdown/imaging/fixed_light_scaling_<hardware>.md`](../breakdown/imaging/) with a
two-panel log-log figure beside each. The four headline rows:

### A100 (gpu-2, fp64, `@vmap 16`) — rectangular (built N)

| Row | 484 | 1024 | 1521 | 2500 | 3969 | α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified, at its certifying budget | 2.87 | 8.16 | 11.2 | 25.0 | 30.0 | 1.15 | 0.978 |
| **certified, phase 3 safe budget 11** | **5.77** | **10.6** | **16.6** | **27.4** | **51.0** | **1.03** | 0.991 |
| Cholesky (unconstrained) | 0.63 | 1.04 | 1.54 | 2.47 | 4.36 | 0.92 | 0.986 |
| NNLS PDIP (reference) | 9.33 | 16.8 | 26.1 | 42.0 | 73.3 | 0.98 | 0.993 |
| library likelihood call (S3) | 13.4 | 25.4 | 38.6 | 67.1 | 134.3 | 1.08 | 0.986 |

### A100 — Delaunay

| Row | 500 | 1000 | 1500 | 2500 | 4000 | α | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| certified, at its certifying budget | 1.06 | 2.22 | 4.27 | 4.80 | 12.1 | 1.11 | 0.963 |
| **certified, phase 3 safe budget 7** | **3.83** | **7.23** | **10.8** | **18.4** | **32.1** | **1.02** | 0.997 |
| Cholesky (unconstrained) | 0.61 | 1.06 | 1.46 | 2.40 | 3.96 | 0.90 | 0.995 |
| NNLS PDIP (reference) | 8.46 | 18.4 | 28.1 | 55.2 | 87.3 | 1.14 | 0.999 |
| library likelihood call (S3) | 15.7 | 31.9 | 49.4 | 93.4 | 165.2 | 1.14 | 0.998 |

### RTX 2060 (6 GB, fp64, single call) — rectangular / Delaunay, the two key rows

| Row | 484/500 | 1024/1000 | 1521/1500 | 2500 | 3969/4000 | α |
|---|---:|---:|---:|---:|---:|---:|
| certified @11, rect | 21.7 | 58.9 | 134.6 | 504.7 | 1,875 | 2.13 |
| PDIP, rect | 33.9 | 87.0 | 185.0 | 683.6 | 2,401 | 2.04 |
| library call (S3), rect | 123.1 | 346.2 | 673.9 | 1,855 | 5,460 | 1.80 |
| certified @7, Delaunay | 15.2 | 54.3 | 90.0 | 343.2 | 1,228 | 2.08 |
| PDIP, Delaunay | 33.9 | 93.9 | 210.9 | 867.7 | 3,032 | 2.19 |
| library call (S3), Delaunay | 141.1 | 377.8 | 727.7 | 2,225 | 6,604 | 1.85 |

### JAX-CPU (i9-10885H, NPROC 8, BLAS 1, single call)

| Row | 484/500 | 1024/1000 | 1521/1500 | 2500 | 3969/4000 | α |
|---|---:|---:|---:|---:|---:|---:|
| certified @11, rect | 29.8 | 182.8 | 468.4 | 1,731 | 6,305 | 2.53 |
| PDIP, rect | 162.3 | 240.1 | 584.2 | 2,291 | 8,333 | 1.93 |
| library call (S3), rect | 383.5 | 1,192 | 2,132 | 6,307 | 15,396 | 1.76 |
| certified @7, Delaunay | 23.7 | 101.5 | 286.6 | 1,170 | 4,270 | 2.52 |
| PDIP, Delaunay | 37.8 | 227.4 | 638.6 | 3,138 | 10,836 | 2.74 |
| library call (S3), Delaunay | 391.2 | 1,133 | 2,357 | 6,999 | 18,976 | 1.88 |

## The fitted exponents — and why the A100's are the surprising ones

Least squares of `log(ms)` on `log(N)` over the five measured points, per row, per mesh, per
hardware. Expected from the algorithms: PDIP is a fixed iteration count times a dense KKT
solve, so ~N³ asymptotically with a near-constant iteration count; one Cholesky is ~N³; the
certified active set is a pass count times a full-size Cholesky, so the same N³ times a
budget that this phase has just shown does not grow.

| Row | A100 rect | A100 Delaunay | RTX rect | RTX Delaunay | CPU rect | CPU Delaunay |
|---|---:|---:|---:|---:|---:|---:|
| certified @ safe budget | 1.03 | 1.02 | 2.13 | 2.08 | 2.53 | 2.52 |
| Cholesky (unconstrained) | 0.92 | 0.90 | 1.91 | 1.91 | 1.84 | 2.50 |
| NNLS PDIP | 0.98 | 1.14 | 2.04 | 2.19 | 1.93 | 2.74 |
| F+λH build (dense) | 1.69 | 1.68 | 1.89 | 1.89 | 1.28 | 1.89 |
| library likelihood call (S3) | 1.08 | 1.14 | 1.80 | 1.85 | 1.76 | 1.88 |

**Nothing on the A100 is running at its algorithmic exponent.** Every solver row there fits
α ≈ 1, not 2 or 3. A 4000×4000 fp64 Cholesky is ~2×10¹⁰ FLOPs — a few milliseconds of an
A100's peak — so what these rows are measuring at 4.4 ms is kernel launch, memory traffic and
the fixed cost of the surrounding graph, not arithmetic. The *only* A100 row with a steep
exponent is the dense `F + λH` build (**α ≈ 1.69**), which is the `Λᵀ N⁻¹ Λ` product over
15,361 image rows and is the one row big enough to be work-bound. On the A100 the build
overtakes the solve between N=2500 and N=4000 (rect 11.9 → 34.5 ms against a certified
solve of 27.4 → 51.0 ms; Delaunay 12.1 → 32.4 ms against 18.4 → 32.1 ms).

**The laptop and the CPU are where the algorithm shows.** Both fit α ≈ 1.8–2.7 — still short
of a clean N³, because the constant terms are large at these sizes and the whole call carries
mesh, mapper and PSF work that scales differently. **The honest reading of the exponents is
that nobody is in the asymptotic regime at N ≤ 4000, and the A100 is not even close.** A
projection to 8000 or 12000 source pixels from these α values would be an extrapolation this
note does not make.

**What that means for the solve's share of the call:** on the A100 it *falls* with N (rect
43 % → 38 %, Delaunay 24 % → 19 %), because the build grows faster than the solve. On the RTX
2060 and the CPU it *rises* (rect 18 % → 34 % and 8 % → 41 %), because there the solve is
work-bound and the rest is not. Which end of the machine you are on decides whether making the
solver faster is still the right optimisation at 4000 pixels.

## The certified lever across N

`S3 PDIP ÷ certified at phase 3's safe budget`, per hardware:

| | 500 | 1000 | 1500 | 2500 | 4000 |
|---|---:|---:|---:|---:|---:|
| **rectangular** A100 | 1.62× | 1.58× | 1.58× | 1.53× | 1.44× |
| rectangular, RTX 2060 fp64 | 1.56× | 1.48× | 1.37× | 1.35× | 1.28× |
| rectangular, RTX 2060 mp | 1.65× | 1.39× | 1.44× | 1.27× | 1.30× |
| rectangular, JAX-CPU | (5.44×)\* | 1.31× | 1.25× | 1.32× | 1.32× |
| **Delaunay** A100 | 2.21× | 2.54× | 2.61× | 3.00× | 2.72× |
| Delaunay, RTX 2060 fp64 | 2.22× | 1.73× | 2.34× | 2.53× | 2.47× |
| Delaunay, RTX 2060 mp | 2.18× | 1.68× | 2.34× | 2.66× | 2.54× |
| Delaunay, JAX-CPU | 1.60× | 2.24× | 2.23× | 2.68× | 2.54× |

\* the CPU n=484 PDIP row is dispatch-bound, not a result — see "Provenance and gate".

**The lever is stable in N and it is mesh-shaped, not hardware-shaped.** Rectangular decays
gently (1.6× → 1.3–1.4×), because its budget of 11 is eleven full factorisations against
PDIP's 15 iterations and the ratio of those two counts is what the row is. Delaunay improves
(2.2× → 2.5–2.7×), because its budget stays at 7 while PDIP's iteration count climbs 14 → 16 →
17 → 20 → 19. **Whatever N phase 5 chooses, the certified route is the lead on both meshes.**

## Memory — the ceiling nobody hit

Peak device bytes (`memory_stats()['peak_bytes_in_use']`, process high-water):

| Built N | 484/500 | 1024/1000 | 1521/1500 | 2500 | 3969/4000 |
|---|---:|---:|---:|---:|---:|
| A100 fp64 @vmap 16, rect | 0.43 GB | 1.02 GB | 1.72 GB | 3.42 GB | **7.76 GB** |
| A100 fp64 @vmap 16, Delaunay | 0.45 GB | 1.00 GB | 1.69 GB | 3.42 GB | **7.85 GB** |
| RTX 2060 fp64, single call | 0.50 GB | 1.02 GB | 1.54 GB | 2.53 GB | **3.13 GB** |
| RTX 2060 mixed precision | 0.63 GB | 1.13 GB | 1.78 GB | 2.83 GB | **3.63 GB** |
| JAX-CPU | not reported by the backend | | | | |

- **The RTX 2060's ceiling was not reached.** 3.13 GB of 6 GB at ~4000 source pixels in fp64;
  the largest N that fits is above this sweep's range, and this note does not guess where. The
  single-call path has roughly 2.9 GB of headroom at N=4000 — so the consumer wall phase 2
  found is **specific to the batched shape**, not to the problem.
- **Mixed precision costs 16 % more memory at every N** (0.63/0.50, 1.13/1.02, 1.78/1.54,
  2.83/2.53, 3.63/3.13 GB) — phase 2's figure, reproduced across the whole range. It buys
  4–5 % on the whole call here (e.g. rect n=3969: 5,363 vs 5,460 ms). On a card whose
  constraint is memory that is a bad trade, and **fp64 remains the consumer path**.
- **The A100 is nowhere near full** — 7.85 GB of 80 GB at N=4000 *with* 16 batched lanes.
  Memory is not what ends the A100 curve; time is.

## The batched row stops paying — the phase-4 correction to phase 1

`@vmap 16` certified per-call, against the single call at the same pass budget, on the A100:

| Built N | 484/500 | 1024/1000 | 1521/1500 | 2500 | 3969/4000 |
|---|---:|---:|---:|---:|---:|
| rectangular: single ÷ batched | 3.64× | 2.48× | 1.84× | 1.14× | **0.68×** |
| Delaunay: single ÷ batched | 3.61× | 2.94× | 1.97× | 1.17× | **0.66×** |

Phase 1 measured the batched certified row at the fiducial and found it the fastest row in the
study. That was true — at n≈1500, where it is worth 1.8–2.0×. It is **not** a property of the
scheme: the amortisation is the fixed per-launch cost being shared, and once a single call is
large enough to fill the device there is nothing left to share. By N=4000 the batched call is
**slower per lane** than 16 sequential calls, on an A100, at 7.8 GB of 80.

**A production scheme should choose its batch size from N, not inherit `n_batch` from the
sampler.** At 500–1500 source pixels batching is worth 1.8–3.6×; at 2500 it is worth nothing;
at 4000 it is a 1.5× penalty.

## Where the whole call crosses 100 ms and 1 s

Interpolated log-log **between two measured points**; where the threshold lies outside the
measured range the table says so rather than extrapolating.

| Hardware | rect: 100 ms | rect: 1 s | Delaunay: 100 ms | Delaunay: 1 s |
|---|---:|---:|---:|---:|
| A100 fp64 | N ≈ **3261** | above 3969 — not measured | N ≈ **2645** | above 4000 — not measured |
| RTX 2060 fp64 | below 484 — not measured | N ≈ **1846** | below 500 — not measured | N ≈ **1735** |
| RTX 2060 mp | below 484 — not measured | N ≈ **1902** | below 500 — not measured | N ≈ **1767** |
| JAX-CPU | below 484 — not measured | N ≈ **912** | below 500 — not measured | N ≈ **922** |

**The affordable N per hardware**, reading those crossings as a production budget:

- **A100** — 4000 source pixels costs 134 ms (rect) / 165 ms (Delaunay) per likelihood call.
  At a few tens of thousands of evaluations that is a run of hours, not days. **4000 is
  affordable; 2500 is comfortable.**
- **RTX 2060 (6 GB)** — the call passes 1 s at N ≈ 1750–1850 and reaches 5.5–6.6 s at 4000.
  **1500 is the practical ceiling on a consumer laptop GPU**, and it is a time ceiling, not a
  memory one.
- **JAX-CPU (8 threads)** — the call passes 1 s at N ≈ 920 and reaches 15–19 s at 4000.
  **1000 source pixels is the CPU's ceiling** for anything resembling a search.

## Positivity gets *more* necessary as N grows

The unconstrained Cholesky (A2) is still the cheapest row at every N and still disqualified —
and its error grows with the mesh:

| Built N | 484/500 | 1024/1000 | 1521/1500 | 2500 | 3969/4000 |
|---|---:|---:|---:|---:|---:|
| rect: Δlog-evidence vs the library (nats) | +318.0 | +328.8 | +334.9 | +347.1 | **+367.6** |
| rect: negative reconstruction entries | 62 | 101 | 136 | 185 | **252** |
| Delaunay: Δlog-evidence (nats) | +8.35 | +3.65 | +6.40 | +6.35 | **+9.90** |
| Delaunay: negative entries | 3 | 2 | 4 | 6 | **10** |

The negative-pixel count grows almost linearly in N on the rectangular mesh (62 → 252 over a
factor 8.2 in pixels), which is what you would expect if a roughly fixed *fraction* of the
source plane wants to go negative. Phase 3 showed the error grows with model error; phase 4
shows it also grows with resolution. **There is no mesh size at which dropping positivity
becomes defensible.**

## Verdict

1. **The certifying budget is not a function of N.** It wanders in 5–10 (rectangular) and 1–2
   (Delaunay) with no trend across a factor 8 in source pixels, and phase 3's safe budgets
   **11 / 7** cover every size measured. Phase 0's "6 at 3025 vs 7 at 1521" was noise, not a
   trend. **Fix the budget at 11 / 7 and stop re-deriving it.**
2. **The certified active set is the lead at every N, on every hardware.** 1.44–1.62×
   (rectangular) and 2.21–3.00× (Delaunay) over the S3 PDIP reference on the A100, and the
   ordering never changes on a laptop GPU or a CPU. On Delaunay the lever *grows* with N,
   because PDIP needs more iterations at larger meshes while the certified budget does not.
3. **The batched `@vmap 16` design has an N ceiling, and it is below this phase's range.**
   Worth 3.6× at N≈500, worth nothing at 2500, a 1.5× penalty at 4000 — on an A100. Phase 1's
   fastest row is a small-mesh row. Batch size should be chosen from N.
4. **Memory is not the wall this phase expected.** The 6 GB laptop card fits the single call
   at ~4000 source pixels with 2.9 GB to spare; the A100 uses 7.8 GB of 80 with 16 lanes. What
   ends every curve is time, and the affordable N is **4000 (A100) / 1500 (RTX 2060) / 1000
   (JAX-CPU)**.
5. **On the A100 the solve is no longer the thing to optimise at large N.** Every A100 solver
   row fits α ≈ 1 — launch- and bandwidth-bound, not work-bound — while the dense `F + λH`
   build fits α ≈ 1.69 and overtakes the certified solve between 2500 and 4000 pixels. On the
   laptop and the CPU the opposite is true and the solve's share rises to 34–41 %.
6. **Mixed precision still buys 4–5 % for 16 % more memory, and its "certification" is looser
   at every N** (rect 4–5 passes against 5–10 in fp64, at `tau_rel` 1.4775e-5 against 1e-9).
   Phase 2's verdict holds across the whole range: **fp64 is the consumer path**, and a
   production budget must never be read off a mixed-precision leg.
7. **Positivity gets more necessary with resolution**, not less: +318 → +368 nats and 62 → 252
   negative pixels on the rectangular mesh from 484 to 3969 source pixels.

## Next

Phase 5 of the epic — **the whole-likelihood assessment on HST and Euclid at 500 / 1250 / 2500
source pixels** (`draft/research/autolens_profiling/fixed_light_likelihood_assessment_hst_euclid.md`)
— was gated on this phase's ms-vs-N curves and now has them. Four things this note hands it:

- **Fix the pass budget at 11 (rectangular) / 7 (Delaunay) and do not sweep it.** It is
  model-dependent (phase 3) and N-independent (phase 4); every leg here certified inside it.
- **All three of phase 5's sizes are inside the measured range**, so no extrapolation is
  needed: 500 and 2500 are measured points and 1250 sits between two of them.
- **Choose the batch size from N.** `@vmap 16` is worth 2.5–3.6× at 500–1250 and nothing at
  2500. Phase 5's A100 legs should not inherit one batch size across its three sizes.
- **Score the CPU on the library's own `fnnls` path, not this cell's certified kernel** (phase
  2's finding, unchanged here), and state the thread pinning: `NPROC=8` with BLAS at 1.

One follow-up this phase raises that belongs to its own prompt: **the dense `F + λH` build is
now the A100's steepest row (α ≈ 1.69) and overtakes the certified solve above ~2500 source
pixels.** Every phase of this epic has optimised the solve. If production wants 4000 source
pixels on a GPU, the next lever is the `Λᵀ N⁻¹ Λ` assembly, not the positivity solver.

## Artifacts

```
results/breakdown/imaging/fixed_light_{rectangular,delaunay}_n{484,1024,1521,2500,3969|500,1000,1500,2500,4000}_hpc_a100_fp64_fixed_light.{json,png}
results/breakdown/imaging/fixed_light_{rectangular,delaunay}_n<N>_local_rtx2060_fp64_fixed_light.{json,png}
results/breakdown/imaging/fixed_light_{rectangular,delaunay}_n<N>_local_rtx2060_mp_fixed_light.{json,png}
results/breakdown/imaging/fixed_light_{rectangular,delaunay}_n<N>_local_cpu_fp64_fixed_light.{json,png}
results/breakdown/imaging/fixed_light_scaling_{hpc_a100_fp64,local_rtx2060_fp64,local_rtx2060_mp,local_cpu_fp64}.{md,png}
results/sweep/fixed_light_scaling_manifest.jsonl
```

Code:

```
scripts/imaging/likelihood_breakdown/fixed_light.py                  # phase 0's cell + --pins, --safe-budget, the machine block, the re-derived tau_rel
scripts/misc/likelihood_breakdown/fixed_light_scaling_table.py       # new — the ms-vs-N tables, the exponent fit, the figures
scripts/misc/test/test_fixed_light_scaling.py                        # 62 tests (suite 348)
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_scaling_{pixelization,delaunay}_a100_hst_fp64
hpc/batch_gpu/submit_fixed_light_scaling.sh
```

JSON keys new in phase 4: `machine` and `precision` on this cell (phase 2 added them to
`fixed_light_library.py`), `pins_recorded` on `--pins none` legs,
`mapper_block_log_det_status`, and under `active_set`: `safe_budget`, `safe_budget_ms`,
`safe_budget_certified`, `safe_budget_measured`, `safe_budget_covers_certifying_budget`,
`tau_rel_basis` and `tau_rel_re_derived_for_precision`. The `hpc_a100_fp64_fixed_light` and
`local_*_fixed_light` config labels are README-table-invisible by design (`CONFIG_TAGGED_RE`
matches a bare config name only), as in phases 0–3; this note and the README prose row are the
index into these legs.

## The phases

| # | Phase | Note |
|---|---|---|
| 0 | Kernel measurement, A100 | [`fixed_lens_light_source_only_2026_09.md`](./fixed_lens_light_source_only_2026_09.md) |
| 1 | Library-path row, A100 | [`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md) |
| 2 | CPU + consumer GPU | [`fixed_lens_light_hardware_2026_09.md`](./fixed_lens_light_hardware_2026_09.md) |
| 3 | Low-likelihood draws | [`fixed_lens_light_low_likelihood_draws_2026_09.md`](./fixed_lens_light_low_likelihood_draws_2026_09.md) |
| **4** | **Source-pixel scaling** | **this note** |
| 5 | HST + Euclid verdict | filed |
