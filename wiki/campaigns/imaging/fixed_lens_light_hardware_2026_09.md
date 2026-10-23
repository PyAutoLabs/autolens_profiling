# Fixed lens light — the same rows on CPU and on a consumer GPU (2026-09-13)

autolens_profiling issue [#253](https://github.com/PyAutoLabs/autolens_profiling/issues/253),
branch `feature/fixed-light-hardware` (stacked on phase 1's `feature/fixed-light-library-path`,
itself stacked on phase 0's). Nineteen local legs on one laptop — an Intel i9-10885H under
WSL2 (8 cores visible, 16 GB) with an NVIDIA GeForce RTX 2060 with Max-Q Design (6 GB,
driver 580.97). Same cell, same routes, same HST dataset and meshes as phase 1's A100 legs,
so the tables stack.

**The certified active set still wins on every hardware, and the margin collapses as the
hardware gets smaller.** On the A100 it halved the library likelihood call (2.03× / 2.56× /
2.01×). On the RTX 2060 it buys **1.28× / 1.48× / 1.46×**; on eight CPU threads **1.83× /
1.74× / 1.35×**; on one CPU thread **1.18× / 1.16× / 1.22×**. And two findings the epic did
not expect:

1. **The GeForce fp64 penalty is not what constrains this likelihood — memory is.** Mixed
   precision buys a flat **5–8 %** and costs at most **2.5e-3 nats**. Meanwhile the batched
   shape the A100 runs (`@vmap 16`) needs **11.88 GiB** and does not fit on a 6 GB card at
   any batch above 2 — and at batch 2, where it does fit, it is *slower per call* than the
   single call. The same batch OOM-killed the 16 GB host on the CPU leg.
2. **The certified active set is not the CPU production kernel.** In numpy, at one thread, it
   ties the library's own `fnnls` NNLS (274.7 vs 261.4 ms rectangular; 143.3 vs 133.3 ms
   Delaunay; 126.7 vs 130.0 ms DelaunayNN) and at eight threads it loses badly
   (1000.1 vs 512.5 ms rectangular). The A100 kernel result does not transfer to the CPU solve.

## Scope — read this before quoting a number

- **What was measured.** Nineteen legs, sequentially, on a quiet machine, one at a time, each
  from a **fresh JAX compilation cache**:
  - **six library legs on the CPU** (`JAX_PLATFORMS=cpu`) — three meshes × two thread
    settings, the same six routes as phase 1 (a S0 PDIP / b S3 PDIP / c S3 positive-negative /
    d S3 certified + PDIP fallback / d0 certified without the fallback / e budget 1 where the
    fallback always fires);
  - **six library legs on the RTX 2060** — three meshes × {fp64, mixed precision};
  - **one extra RTX 2060 leg at `@vmap 2`**, the largest batch that fits;
  - **six CPU *kernel* legs** — three meshes × two thread settings of the numpy/scipy solve
    rows, which are a different cell and must never be summed with the library rows.
- **The hardware.** `model name: Intel(R) Core(TM) i9-10885H CPU @ 2.40GHz`, `nproc --all` = 8,
  16 GB, kernel `5.10.16.3-microsoft-standard-WSL2`; `NVIDIA GeForce RTX 2060 with Max-Q
  Design, 6144 MiB, driver 580.97`. Every JSON carries this in a `machine` block.
- **`nproc` is not the core count on this host.** GNU `nproc` honours `OMP_NUM_THREADS`, which
  the session shell exports as 1, so it prints **1** on an 8-core machine. `nproc --all` is the
  answer, and both are recorded.
- **Two disjoint thread knobs, both set explicitly on every CPU leg.** `NPROC` sizes XLA's CPU
  intra-op pool and governs the **JAX** rows; `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `MKL_NUM_THREADS` pin OpenBLAS/MKL and govern the **numpy** rows, and do not touch JAX. The
  "1 thread" legs ran with all four at 1; the "8 threads" legs with all four at 8. Both
  families are in every CPU JSON, for both kinds of row.
- **What was NOT measured.** No batched (`@vmap 16`) row on any local hardware — it does not
  fit, and the failures are recorded below rather than quietly dropped. No low-likelihood
  draws (phase 3), no source-pixel sweep (phase 4), no Euclid, no sparse operator. No
  production implementation: routes d/d0/e remain a measurement harness.
- **No PyAutoArray change**, exactly as phase 1: the certified active set is injected by a
  scoped harness monkeypatch of `inversion_util.reconstruction_positive_only_from`, released
  afterwards, and every JSON says so under `solver_injection`.
- **Compare within a mesh, never across.** Regularization is `Constant(1.0)` on rectangular
  and `adapt_split` on the Delaunay family, each cell's shipped default.

## Pins — re-derived per precision, never inherited

A pin calibrated in fp64 is not a pin in fp32. The cell gained a `--pins {fp64,none}` switch
for exactly this, and `none` **withdraws the verdict, it does not skip the measurement**:
every comparison is still computed and written, with status `RECORDED` instead of PASS/FAIL.

| Leg | S0 log-det pins | mapper-block identity | equivalence pins | `tau_rel` |
|---|---|---|---|---|
| CPU fp64 (both thread settings) | asserted | asserted | asserted, rtol 1e-8 | 1e-9 |
| RTX 2060 fp64 | asserted | asserted | asserted, rtol 1e-8 | 1e-9 |
| RTX 2060 mixed precision | **recorded** | **recorded** | **recorded** | **1.4775e-5** |

`tau_rel` for the mixed-precision legs is re-derived, not guessed:
`eps_float32 (1.1921e-7) × sqrt(image pixels = 15361) = 1.4775e-5`. `use_mixed_precision`
makes the library accumulate `A.T A` in float32 before casting back to float64
(`inversion_util.py:127-136`) — the solve and both log determinants are still fp64 arithmetic,
on a matrix carrying float32 accumulation error. A dot product over 15361 image rows carries a
relative error of order `eps_f32 × sqrt(M)`, so the KKT gradient cannot be resolved below that
fraction of `max|q|`. Certifying at 1e-9 would be certifying against the matrix's own
round-off. The value used is in every JSON under `precision.tau_rel` with its derivation.

## Provenance and gate

| Leg | Mesh | Backend | Threads (NPROC / BLAS) | Precision | Pins | `cache_fresh` | Exit |
|---|---|---|---|---|---|---|---|
| `local_cpu_fp64_…_t1` × 3 | all three | cpu | 1 / 1 | fp64 | asserted | true | 0 |
| `local_cpu_fp64_…_tall` × 3 | all three | cpu | 8 / 8 | fp64 | asserted | true | 0 |
| `local_rtx2060_fp64_…` × 3 | all three | gpu (CudaDevice) | 8 / 1 | fp64 | asserted | true | 0 |
| `local_rtx2060_mp_…` × 3 | all three | gpu (CudaDevice) | 8 / 1 | mixed | recorded | true | 0 |
| `local_rtx2060_fp64_…_vmap2` | rectangular | gpu | 8 / 1 | fp64 | asserted | true | 0 |
| `local_cpu_fp64_fixed_light_kernels_{t1,tall}` × 3 | all three | numpy/scipy | 1 or 8 | fp64 | n/a | n/a | 0 |

**Counts: legs = 19; result JSONs written = 19; non-zero exit = 0; `cache_fresh: false` = none;
`pinned_drift` non-empty = none; pin FAILED = none; equality FAILED = none.**

| Gate | rect | delaunay | delaunay_nn |
|---|---|---|---|
| S0 runtime log-det pins, every fp64 leg (`pinned_drift`) | PASS (empty) | PASS (empty) | PASS (empty) |
| mapper-block log-dets S3 == S0, every fp64 leg | PASS | PASS | PASS |
| pin d == b, CPU t1 / CPU t8 / 2060 fp64 | 4.5e-15 / 4.5e-15 / 4.3e-15 | 6.1e-15 / 1.0e-15 / 4.3e-12 | 1.2e-15 / 1.7e-15 / 2.6e-11 |
| pin d0 == b, CPU t1 / CPU t8 / 2060 fp64 | 4.5e-15 / 4.5e-15 / 4.3e-15 | 6.1e-15 / 1.0e-15 / 1.1e-11 | 1.2e-15 / 1.7e-15 / 5.2e-11 |
| pin e == b (the fallback reproduces PDIP) | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 3.8e-11 | 0.0 / 0.0 / 5.5e-11 |
| pin b == a (S3 is a re-arrangement of S0) | 3.6e-13 / 3.6e-13 / 3.8e-15 | 2.5e-15 / 1.8e-15 / 4.8e-11 | 4.1e-15 / 5.5e-15 / 8.8e-12 |
| pin c == phase 0's A2 Cholesky | 9.0e-14 / 5.1e-14 / 9.2e-14 | 6.0e-13 / 6.5e-13 / 6.6e-13 | 2.7e-13 / 2.2e-13 / 3.9e-13 |
| route (c) entry point verified at run time | PASS | PASS | PASS |
| numpy NNLS == library reconstruction (kernel legs) | PASS 0.0 | PASS 0.0 | PASS 0.0 |
| numpy certified == numpy NNLS (kernel legs) | PASS 0.0 | PASS 8.7e-16 | PASS 1.6e-15 |

The mixed-precision legs' recorded comparisons are in the next section; nothing on those legs
was asserted.

## Headline — whole library likelihood call, per call in ms

Phase 1's A100 column is reproduced from
[`fixed_lens_light_library_path_2026_09.md`](./fixed_lens_light_library_path_2026_09.md) so the
five hardware settings stack into one table. **A100 = RAL `gpu-2`, NVIDIA A100 80GB PCIe.**

### rectangular (1521 source pixels; the solver sees 1369 after edge zeroing)

| Route | A100 fp64 | RTX 2060 fp64 | RTX 2060 mp | CPU 8 threads | CPU 1 thread |
|---|---:|---:|---:|---:|---:|
| **a** S0 PDIP — the library today | 50.97 | 761.6 | 720.2 | 3422.8 | 5529.2 |
| **b** S3 PDIP — source-only reference | 38.65 | 664.1 | 626.0 | 2425.9 | 4929.6 |
| **c** S3 positive-negative | 21.13 | 542.3 | 500.7 | 1727.8 | 4325.8 |
| **d** S3 certified active set + fallback | **25.10** | **595.4** | **555.6** | **1866.8** | **4703.0** |
| **d0** route d, fallback removed | 25.41 | 598.9 | 559.8 | 1917.8 | 4652.6 |
| **e** budget 1 — fallback FIRES | 41.45 | 717.7 | 671.6 | 2817.7 | 5391.3 |
| **a → d** | **2.03×** | **1.28×** | **1.30×** | **1.83×** | **1.18×** |

### delaunay (1500 source pixels, no edge zeroing)

| Route | A100 fp64 | RTX 2060 fp64 | RTX 2060 mp | CPU 8 threads | CPU 1 thread |
|---|---:|---:|---:|---:|---:|
| **a** S0 PDIP | 65.10 | 842.2 | 801.8 | 4923.9 | 5602.0 |
| **b** S3 PDIP | 49.67 | 723.8 | 681.4 | 3576.5 | 5133.8 |
| **c** S3 positive-negative | 27.16 | 554.9 | 511.4 | 2982.7 | 4545.9 |
| **d** S3 certified + fallback | **25.39** | **569.7** | **537.1** | **2834.5** | **4847.5** |
| **d0** fallback removed | 25.27 | 601.4 | 558.5 | 2301.5 | 5712.8 |
| **e** budget 1 — fallback FIRES | 51.83 | 767.7 | 743.2 | 2696.2 | 7761.3 |
| **a → d** | **2.56×** | **1.48×** | **1.49×** | **1.74×** | **1.16×** |

### delaunay_nn (1500 source pixels, no edge zeroing)

| Route | A100 fp64 | RTX 2060 fp64 | RTX 2060 mp | CPU 8 threads | CPU 1 thread |
|---|---:|---:|---:|---:|---:|
| **a** S0 PDIP | 72.95 | 867.8 | 820.0 | 3697.3 | 5869.9 |
| **b** S3 PDIP | 55.37 | 737.9 | 693.5 | 3321.7 | 5289.9 |
| **c** S3 positive-negative | 35.08 | 582.2 | 536.8 | 2995.2 | 4601.5 |
| **d** S3 certified + fallback | **36.26** | **594.8** | **546.9** | **2748.1** | **4819.0** |
| **d0** fallback removed | 33.27 | 638.2 | 555.7 | 2957.4 | 4792.5 |
| **e** budget 1 — fallback FIRES | 58.32 | 763.6 | 774.5 | 2985.6 | 5362.5 |
| **a → d** | **2.01×** | **1.46×** | **1.50×** | **1.35×** | **1.22×** |

**The hardware gap, on the row the library runs today (a).** The A100 is **14.9× / 12.9× /
11.9×** the RTX 2060, **67× / 76× / 51×** eight CPU threads, and **108× / 86× / 80×** one CPU
thread. On the certified row (d) the A100 lead *widens* to 23.7× / 22.4× / 16.4× over the
laptop GPU — the faster the solve gets, the more of the call is the part a consumer card is
worst at.

**The ordering is identical on every hardware.** a is always the slowest, c always the
fastest, d and d0 always within a few per cent of each other and of c, e always between d
and a. Nothing about the route ranking is an A100 artefact. What changes is the *size* of the
prize, and it shrinks monotonically as the hardware shrinks — because the residue phase 1
identified (mapper, mesh, imaging, blurring) is what a small device is worst at, and it is not
the solver.

**One CPU thread is the flattest case of all.** At `NPROC=1` the certified route buys
1.16–1.22×, against 1.35–1.83× at eight. A single-threaded CPU run is so dominated by
everything that is not the solve that halving the solve is nearly invisible.

## Mixed precision — what the lower precision buys and what it costs

Recorded on the `--pins none` legs, never asserted. The reference in each row is the fp64 leg
**on the same card**, not the A100.

| | rectangular | delaunay | delaunay_nn |
|---|---:|---:|---:|
| Speed-up, route a (S0 PDIP) | 1.057× | 1.050× | 1.058× |
| Speed-up, route b (S3 PDIP) | 1.061× | 1.062× | 1.064× |
| Speed-up, route d (certified) | 1.072× | 1.061× | 1.088× |
| Δ log-likelihood, route a (nats) | **−2.525e-03** | −7.394e-04 | −9.170e-04 |
| Δ log-likelihood, route b (nats) | **−4.419e-04** | −4.704e-05 | +1.443e-05 |
| Δ log-likelihood, route d (nats) | −4.445e-04 | −4.640e-05 | +1.419e-05 |
| Δ log-likelihood, route c (nats) | +1.321e-04 | −3.948e-05 | +2.395e-05 |
| Route (c) negative pixels, fp64 → mp | 136 → **136** | 4 → **4** | 4 → **4** |
| Smallest certifying budget, fp64 → mp | 7 → **4** | 2 → 2 | 2 → 2 |
| Peak device bytes, fp64 → mp | 1.63 → 1.89 GB | 1.61 → 1.88 GB | 1.61 → 1.88 GB |
| S0 log-det vs the fp64 pin (recorded) | 8.5e-11 / 2.2e-12 | 1.7e-11 / 4.5e-09 | 2.2e-11 / 2.0e-09 |

**The expected GeForce fp64 catastrophe did not happen.** A consumer card runs fp64 at 1/32
the fp32 rate, and the epic's premise was that this would force mixed precision on consumer
hardware. It does not: mixed precision buys a flat **5–8 %**, on every route and every mesh.
The reason is visible in the table above — this call is not FLOP-bound on the fp64 units. It is
bound by the mapper, the mesh, the PSF convolution and memory traffic, and
`use_mixed_precision` touches exactly one thing: the float32 accumulation of `A.T A`
(`inversion_util.py:127-136`). Everything else, including both log determinants and the solve,
is fp64 either way.

**What it costs is negligible, and it is not free.** The largest deviation measured anywhere is
**2.5e-3 nats** (route a, rectangular), five thousand times below the 0.5-nat bar the
matrix-free study used, and the negative-pixel counts on route (c) are identical to the last
pixel. **But mixed precision also costs 16 % more device memory** (1.61 → 1.88 GB peak), which
on a card whose real constraint is memory is a live cost, not a rounding error.

**Certification is looser, and that must not be read as faster convergence.** On rectangular
the smallest certifying budget falls 7 → 4 under mixed precision. That is the *tolerance*
moving, not the solver: at `tau_rel = 1.48e-5` the pass-4 iterate's residual dual violations
(phase 1 measured the dual trace 20 → 7 → 4 → 2 → 2 → 2 → 0) are already inside the fp32 noise
floor, so the scheme declares them gone. The iterate at pass 4 is **not** the iterate at pass 7;
phase 1 measured it at −4.4e-9 nats of the converged answer in fp64. A production scheme must
not take "certifies at 4 under mp" as licence to run a budget of 4: it certifies against a
matrix that is only known to 1e-5, and the honest statement is that **at fp32 accumulation the
certificate is weaker, not that the problem is easier**.

**Verdict: fp64 is the consumer-GPU path.** Mixed precision is a 5–8 % option with a real
memory cost and a weakened certificate, not the consumer route the phase premise assumed.

## Memory is the consumer-GPU wall — the batched shape does not fit

Phase 1 measured every A100 row at `@vmap 16`, the Nautilus `n_batch`, and found the batched
certified row (d0) the fastest of all: 21.35 / 21.43 / 23.64 ms. **None of that is available on
6 GB.** Measured, not estimated:

| Batch | Outcome on the RTX 2060 (6 GB), rectangular n=1521 |
|---|---|
| 16 | **OOM** — `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11.88GiB` |
| 4 | **OOM** — a further 2.97 GiB (fp64) / 3.65 GiB (mp) on top of a 1.63 GB single-call peak |
| 2 | **fits** — peak 3.21 GB, and **slower per call than the single call** |
| 1 | fits — peak 1.63 GB |

At batch 2 the per-call times are 848.9 (a) / 748.8 (b) / 551.1 (c) / 859.0 (d) / 610.6 (d0) /
769.7 (e) ms against 771.1 / 673.3 / 551.6 / 598.5 / 590.5 / 697.2 ms unbatched: batching is a
**loss** on every route but c. A 6 GB card has no batching headroom at n≈1500 and nothing to
gain from the headroom it has. The `@vmap 16` OOM is recorded in each GPU JSON's `vmap_error`;
the vmap-2 leg is `…_vmap2.json`.

**The host is no better.** The first CPU rectangular leg was launched with `--vmap-batch 16`
and the Linux OOM killer took the process (exit 137) partway through the batched block, on a
16 GB machine, *after* every unbatched row had been measured. The CPU legs in this note are
therefore unbatched by design, and that is a result, not a gap: **a 16 GB laptop cannot hold
the batched shape the A100 legs run either.**

This is the sharp consumer-hardware finding. Not "the GeForce is slow at fp64" — it is that a
scheme designed around `@vmap 16` is **not portable to consumer hardware at all**, at either
end of the machine.

## The CPU kernel rows — the certified active set is not the CPU production kernel

A CPU user does not run `jax_nnls`'s PDIP. `reconstruction_positive_only_from` branches on
`xp` and the numpy branch is `fnnls_cholesky` (Bro & de Jong 1997), seeded from the sign of the
dense unconstrained solve. So phase 2 measured the CPU *solve* in the tools a CPU uses. These
are **kernel rows in their own cell** — never sum them with the library rows above.

Median of 15 repeats, `time.perf_counter`, ms:

| Row | rect 1 thr | rect 8 thr | delaunay 1 thr | delaunay 8 thr | delaunay_nn 1 thr | delaunay_nn 8 thr |
|---|---:|---:|---:|---:|---:|---:|
| library numpy NNLS (`fnnls_cholesky`) | **261.4** | **512.5** | **133.3** | **300.2** | 130.0 | 269.8 |
| numpy certified active set | 274.7 | 1000.1 | 143.3 | 325.9 | **126.7** | **299.0** |
| scipy unconstrained (subset) | 23.2 | 17.6 | 41.2 | 24.2 | 34.4 | 17.8 |
| scipy unconstrained (full system) | 37.6 | 37.1 | 51.1 | 35.1 | 40.3 | 23.6 |
| certified: passes / factorisations | 7 / 8 | 7 / 8 | 2 / 3 | 2 / 3 | 2 / 3 | 2 / 3 |
| library NNLS: outer / inner iterations | 28 / 3 | 28 / 3 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| n seen by the solver | 1369 | 1369 | 1500 | 1500 | 1500 | 1500 |

Every positivity row reaches the library's own reconstruction exactly (rel diff 0.0 on
rectangular, ≤ 1.6e-15 on the Delaunay family), so these are timings of the same answer.

**Finding 1 — the A100 kernel result does not transfer.** On the A100 the certified active set
beat the library's PDIP by about 2.4× on rectangular (11.05 vs 25.8-28.3 ms) and about 6×
on Delaunay (4.21 ms) at the kernel level. In numpy it does
not beat `fnnls` at all: it is 5 % *slower* on rectangular, 7 % slower on Delaunay, and 3 %
faster on DelaunayNN — a wash at one thread, and a rout at eight. The reason is in the
iteration counts. `fnnls` reaches the Delaunay answer in **zero outer iterations** — the
dense-sign seed is already the passive set (1495 of 1500 pixels free, 4 negative) — so there is
nothing for a smarter scheme to save. On rectangular `fnnls` takes 28 outer iterations but each
one is a *Cholesky update*, not a refactorisation, while the certified scheme pays 8 full
restricted factorisations. The library's CPU kernel is already good.

**Finding 2 — BLAS threads make the iterative CPU kernels slower, not faster.** Both positivity
rows are **1.9× to 3.6× slower at eight threads than at one** (rect certified 274.7 → 1000.1 ms;
rect NNLS 261.4 → 512.5 ms), while the single unconstrained Cholesky on the same matrix gets
*faster* (23.2 → 17.6 ms). The pattern is consistent across all three meshes and is what you
would expect from OpenBLAS thread dispatch: a scheme that issues many small factorisations and
triangular solves on a 1369–1500 square matrix pays the fork/join on every one, and the work
per call is too small to amortise it. **A CPU production path should pin the BLAS threads to 1
for the positivity solve** — and note that this is the opposite of the JAX-CPU library rows,
where eight threads is 1.1–1.6× faster than one. The two knob families point in opposite
directions, which is exactly why both are recorded on every timing.

**Finding 3 — positivity is 3–11× the unconstrained solve on the CPU.** 261.4 vs 23.2 ms
(rect), 133.3 vs 41.2 ms (Delaunay), 130.0 vs 34.4 ms (DelaunayNN) at one thread. The
unconstrained row remains prohibited for the same reason it was on the A100: it costs
**+334.93 nats and 136 negative pixels** on rectangular (+86.21 nats and 79 negatives on the
edge-zeroed subset), **+6.40 nats / 4 pixels** on Delaunay and **+8.22 / 4** on DelaunayNN,
reproducing phase 0's A2 row and phase 1's route (c) to the last digit. A positive Δ is not an
improvement: it is a different, infeasible minimiser.

## Route (c) — positive-negative, never a bare millisecond

Unchanged from phase 1 to the last digit, on every hardware and both precisions:

| Quantity | rectangular | delaunay | delaunay_nn |
|---|---:|---:|---:|
| Δlog-evidence vs the library's S3 solution (nats) | **+334.926** | **+6.404** | **+8.219** |
| Negative reconstruction entries | 136 | 4 | 4 |
| Negative flux fraction | 0.1354 % | 0.0017 % | 0.0020 % |
| Identical under mixed precision? | yes (136) | yes (4) | yes (4) |
| `solve_ids_to_keep is None` (edge zeroing silently off) | true | true | true |

It is the fastest row on every hardware measured here, and it stays blocked on the
matched-injection witness phase 0 demanded. Cheapness on a laptop is not a new argument for it.

## Per-hardware verdict

1. **The certified active set is the right route everywhere — and it is an A100 optimisation.**
   2.03–2.56× there, 1.28–1.48× on a laptop GPU, 1.35–1.83× on eight CPU threads, 1.16–1.22× on
   one. The ordering never changes; the prize shrinks with the hardware, because what is left
   in the call after the solve is what small hardware is worst at. Phase 1 predicted this in
   one line ("a consumer GPU will make that worse, not better") and the measurement is that
   line's number.
2. **fp64 is the consumer-GPU path; mixed precision is a 5–8 % option, not a necessity.** The
   GeForce fp64 penalty does not bite because the likelihood is not FLOP-bound on the fp64
   units. The cost is ≤ 2.5e-3 nats, the benefit is 5–8 %, and the price is 16 % more device
   memory and a certificate that is weaker (rectangular "certifies" at pass 4 rather than 7
   only because the fp32-derived tolerance is 1.5e-5 rather than 1e-9).
3. **Memory, not precision, is the consumer-hardware wall, and it invalidates the batched
   design.** `@vmap 16` needs 11.88 GiB and OOMs a 6 GB card; batch 4 OOMs; batch 2 fits and is
   slower per call than batch 1; and the batched shape OOM-killed a 16 GB host on the CPU. The
   fastest A100 row in phase 1 — the batched certified row at 21 ms — has no consumer
   counterpart at all. **Any production scheme that assumes the batched path must have a
   single-call path that is not an afterthought.**
4. **On the CPU, keep the library's own NNLS.** The certified active set does not beat
   `fnnls_cholesky` in numpy, and both collapse under BLAS threading. The CPU recommendation is
   the library's existing kernel with `OMP_NUM_THREADS=1` for the solve — while the JAX-CPU
   library rows want `NPROC=8`. Two knobs, opposite directions, both recorded.
5. **Fixing the light still pays on its own, everywhere.** a → b, with no solver change and no
   library change: 1.32× on the A100, 1.15–1.18× on the RTX 2060, 1.11–1.41× on eight CPU
   threads, 1.09–1.12× on one.

## Next

Phase 3 of the PyAutoMind epic **`fixed-lens-light-profiling`**
(`active/fixed_light_certified_low_likelihood_draws.md`): is the certified active set fast only
because the model is good? Every leg in phases 0–2 is the same fiducial draw; phase 3 grades a
draw set by Δlog L and asks whether the certifying pass budget holds away from the optimum.
**DONE 2026-09-13** ([#255](https://github.com/PyAutoLabs/autolens_profiling/issues/255), note
[`fixed_lens_light_low_likelihood_draws_2026_09.md`](./fixed_lens_light_low_likelihood_draws_2026_09.md)):
**it does not.** Over a seeded 41-model draw set the pass count *grows* with model error on
Delaunay (Spearman +0.698) and *falls* on rectangular (−0.535), and both phase-0 budgets break —
Delaunay's pass 2 falls back on **67.5 %** of the set, rectangular's pass 7 on **27.5 %**. The
smallest zero-fallback budgets are **7 (Delaunay)** and **11 (rectangular)**; the scheme still
beats PDIP at every model (median 5.4× / 2.6× on the A100), and the pass counts are identical on
the A100 and the CPU. Phases 4 (source-pixel scaling across hardware) and 5 (the HST + Euclid
verdict) follow in order, and phase 4 should sweep budgets 7 and 11 rather than 2 and 7.

Three things this note hands phase 3 and beyond:

- **A hardware-dependent pass budget is a portability risk, not just a tuning knob.** Rectangular
  needs 7 passes in fp64 and "needs" 4 under fp32 accumulation. Phase 3's draw set should record
  the certifying budget per draw *per precision*, not just per draw.
- **The batched path is A100-only.** Phase 4's ms-vs-N curves should be measured single-call on
  consumer hardware, and the memory ceiling (not the time) is what will end the curve — on this
  card the single call already peaks at 1.63 GB at n=1521.
- **The CPU solve is already well served by the library.** Phase 5's CPU assessment should score
  the library's `fnnls` path, not the certified active set, and should state its thread pinning.

## Artifacts

```
results/breakdown/imaging/fixed_light_library_{rectangular,delaunay,delaunay_nn}_local_cpu_fp64_fixed_light_library_{t1,tall}.{json,png}
results/breakdown/imaging/fixed_light_library_{rectangular,delaunay,delaunay_nn}_local_rtx2060_{fp64,mp}_fixed_light_library.{json,png}
results/breakdown/imaging/fixed_light_library_rectangular_local_rtx2060_fp64_fixed_light_library_vmap2.{json,png}
results/breakdown/imaging/fixed_light_cpu_kernels_{rectangular,delaunay,delaunay_nn}_local_cpu_fp64_fixed_light_kernels_{t1,tall}.{json,png}
```

Code:

```
scripts/imaging/likelihood_breakdown/fixed_light_library.py       # phase 1's cell + --pins, the machine block, the re-derived tau_rel
scripts/imaging/likelihood_breakdown/fixed_light_cpu_kernels.py   # new cell — the CPU kernel rows
scripts/misc/likelihood_breakdown/fixed_light_cpu_kernels.py      # new module — the three CPU kernels and their gates
scripts/misc/likelihood_breakdown/library_solver_injection.py     # tau_rel is now a parameter, not a constant
scripts/misc/test/test_fixed_light_cpu_kernels.py                 # 22 tests
_profile_cli.py                                                   # machine_info_dict() — the host, beside device_info_dict()'s JAX device
```

JSON keys new in phase 2: `machine` (`cpu_model`, `nproc`, `nproc_all`, `ram_total`, `kernel`,
`gpu`, `jax_thread_env`, `blas_thread_env`), `precision` (`use_mixed_precision`, `pins_mode`,
`tau_rel`, `tau_rel_basis`, `float32_eps`), `pins_recorded` (on `--pins none` legs — the
withdrawn fp64 pins, computed and written), `vmap_error` (the recorded OOM), and, in the kernel
cell, `kernels` (`threads`, `rows`, `equalities`, `n_seen_by_solver`). The
`local_cpu_fp64_…` / `local_rtx2060_…` config labels are README-table-invisible by design
(`CONFIG_TAGGED_RE` matches a bare config name only), like phases 0 and 1; this note and the
README prose row are the index into these legs.
