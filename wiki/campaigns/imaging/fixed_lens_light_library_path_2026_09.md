# Fixed lens light — the library-path row (2026-09-13)

> **Status correction — 2026-09-18:** The 25.39 ms certified A100 row uses pass budget 2, not the later selected budget 7 (31.64 ms). The ~13.9 ms mesh/mapper attribution was refuted by the [single-process trace](hst_gpu_residue_phase1_2026_09.md). Use the [current campaign summary](profiling_campaign_status_2026_09.md) before quoting these historical numbers.

autolens_profiling issue [#251](https://github.com/PyAutoLabs/autolens_profiling/issues/251),
branch `feature/fixed-light-library-path` (stacked on phase 0's `feature/fixed-lens-light-source-only`),
harvest commit `f659dd5`. Three A100 legs (342908–342910) on `euclid-ral-gpu-2`, fp64, dense,
HST, meshes rectangular 1521 / Delaunay 1500 / DelaunayNN 1500. **Phase 0's projection held,
and the certified active set halves the library likelihood call.** Every row here is one whole
`AnalysisImaging.log_likelihood_function` under `jax.jit` — the row phase 0 could not measure.
With the MGE converted to regular light profiles at their solved intensities and subtracted
("S3"), and a certified active-set positive solve injected at the library's own solver entry
point, the call falls **50.97 → 25.10 ms** on rectangular (2.03×), **65.10 → 25.39 ms** on
Delaunay (2.56×) and **72.95 → 36.26 ms** on DelaunayNN (2.01×) against what the library runs
today, at a log-likelihood identical to the library's own PDIP answer to 1e-15…1e-11 relative.
The positive-negative route is faster still (21.13 / 27.16 / 35.08 ms) and still costs
**+334.93 nats on rectangular** — phase 0's prohibition stands, unchanged.

## Scope — read this before quoting a number

- **What was measured.** The whole library likelihood call, HST, mask 3.5″, MGE-60 lens light,
  dense inversion path, on `euclid-ral-gpu-2` (NVIDIA A100 80GB PCIe), fp64, single call
  (jit steady state) and `@vmap 16` per call. Five routes, all through the library:
  - **(a) S0 PDIP** — the current system (60 unregularised linear-MGE columns + source pixels)
    with the library's own positivity solver. What the library runs today.
  - **(b) S3 PDIP** — source-only system, library solver. The reference every S3 route is
    scored against.
  - **(c) S3 positive-negative** — `use_positive_only_solver=False`, which routes to
    `inversion_util.reconstruction_positive_negative_from` → `xp.linalg.solve`. Verified at
    run time (`route_verified`, one solver call counted), **and edge zeroing is off on this
    route**: `Inversion.solve_ids_to_keep` returns `None` the moment positivity is disabled
    (`abstract.py:540`), so (c) solves the *full* system while a/b/d/e solve the edge-zeroed one.
  - **(d) S3 certified active set + PDIP fallback** — at phase 0's per-mesh certifying budget
    (rect 7, Delaunay 2, DelaunayNN 2), `lax.cond(certified, certified_answer, library_PDIP)`.
  - **(d0)** route (d) with the fallback branch removed — the pure certified cost, and the only
    honest batched certified row (see the `vmap` finding below).
  - **(e)** route (d) at pass budget 1, where the scheme never certifies, so the fallback always
    fires: the worst-case per-call cost.
- **No PyAutoArray change.** The certified active set does not exist in the library. It is
  injected **in the harness**: a scoped monkeypatch of
  `autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from`, applied
  through a context manager for the duration of routes d/d0/e and released afterwards. Every
  other part of the call — mapper, `F + λH` assembly, both log-dets, the evidence — is the
  library's own path, and the JSON says so in `solver_injection`. Nothing in the library moved.
- **What was NOT measured.** No CPU or consumer-GPU leg (that is phase 2). No low-likelihood
  draws: every leg is the same good model (phase 3). No sparse-operator leg, no Euclid, no
  source-pixel sweep. No production implementation — routes d/d0/e are a measurement harness,
  not a proposal that has been reviewed.
- **Regularization** is `Constant(1.0)` on the rectangular cell and `adapt_split`
  (0.1 / 10.0 / 0.1) on the Delaunay family, each cell's shipped default. Compare within a
  mesh, never across.
- **Never sum these rows with the phase-0 kernel rows.** Phase 0's cell timed kernels in its
  own process; this cell times whole library calls. Where the two are combined below it is
  flagged as arithmetic across cells, not a measured decomposition.

## Provenance and gate

| Job | Mesh | N_src | Exit | Elapsed | Node | cache_fresh | Injected solver calls (d / d0 / e) |
|---|---|---:|---|---:|---|---|---|
| 342908 | rectangular | 1521 | 0:0 | 00:02:36 | euclid-ral-gpu-2 | true | 1 / 1 / 1 (JAX path) |
| 342909 | delaunay | 1500 | 0:0 | 00:02:48 | euclid-ral-gpu-2 | true | 1 / 1 / 1 (JAX path) |
| 342910 | delaunay_nn | 1500 | 0:0 | 00:03:32 | euclid-ral-gpu-2 | true | 1 / 1 / 1 (JAX path) |

**Counts: legs = 3; result JSONs written = 3; off-node = none; `cache_fresh: false` = none;
non-zero exit = none; pin FAILED = none.** Every leg reports
`autotune_cache_entries_at_start: 0`, `autolens 2026.8.17.1`, the standard cell XLA flags
(`constant_folding` disabled, `autotune_level=0`, Triton GEMM off), and a peak of 18.9–19.2 GB
across all six routes in one process.

| Gate | rect | delaunay | delaunay_nn |
|---|---|---|---|
| `hostname` = euclid-ral-gpu-2, A100 80GB | PASS | PASS | PASS |
| `device.cache_fresh` (autotune entries at start = 0) | PASS | PASS | PASS |
| S0 runtime log-det pins (`pinned_drift`) | PASS (empty) | PASS (empty) | PASS (empty) |
| S3 mapper-block log-dets == S0 (rel diff) | 0.0 | 0.0 | 0.0 |
| pin: route d == route b (rtol 1e-8) | PASS 4.6e-15 | PASS 9.4e-11 | PASS 3.4e-11 |
| pin: route d0 == route b (rtol 1e-8) | PASS 4.6e-15 | PASS 3.7e-12 | PASS 6.5e-11 |
| pin: route e == route b — the fallback reproduces PDIP (rtol 1e-8) | PASS 0.0 | PASS 4.6e-11 | PASS 2.5e-11 |
| pin: route b == route a — S3 is a re-arrangement of S0 (rtol 1e-6) | PASS 4.3e-15 | PASS 5.6e-12 | PASS 2.8e-11 |
| pin: route c reconstruction == phase 0's A2 Cholesky (rtol 1e-6) | PASS 7.6e-14 | PASS 1.0e-12 | PASS 7.0e-13 |
| route (c) entry point verified at run time | PASS (1 call) | PASS (1 call) | PASS (1 call) |

The mapper-block gate is again the one that makes S3 legitimate: converting the MGE to regular
profiles changes neither `log det(F+λH)` nor `log det(H)` on the mapper block (rel diff exactly
0.0), and route (b)'s likelihood equals route (a)'s to 1e-15…1e-11 relative.

## Headline — whole library likelihood call, per call in ms

| Route | rectangular (1521) | delaunay (1500) | delaunay_nn (1500) |
|---|---:|---:|---:|
| **a** S0 PDIP — the library today | 50.968 | 65.099 | 72.952 |
| **b** S3 PDIP — source-only reference | 38.652 | 49.666 | 55.370 |
| **c** S3 positive-negative (`xp.linalg.solve`) | **21.128** | **27.161** | **35.079** |
| **d** S3 certified active set + PDIP fallback | **25.104** | **25.386** | **36.257** |
| **d0** route d, fallback branch removed | 25.407 | 25.265 | 33.265 |
| **e** budget 1 — fallback FIRES (worst case) | 41.452 | 51.827 | 58.318 |
| — | | | |
| @vmap16 a | 32.128 | 42.700 | 47.566 |
| @vmap16 b | 28.802 | 35.284 | 36.573 |
| @vmap16 c | 23.057 | 25.913 | 28.076 |
| @vmap16 **d0** (the batched certified row) | **21.352** | **21.425** | **23.636** |
| @vmap16 d (cond → select: certified **plus** PDIP) | 33.848 | 37.539 | 38.916 |
| @vmap16 e | 30.290 | 37.136 | 38.451 |

Ratios against the row the library runs today (a): **a → b 1.32× / 1.31× / 1.32×** from fixing
the light alone, with no solver change; **a → d 2.03× / 2.56× / 2.01×** with the certified
solve; **a → c 2.41× / 2.40× / 2.08×** with positivity dropped. Batched, a → d0 is
**1.51× / 1.99× / 2.01×**.

**Phase 0's projection held.** It predicted ≈ 23.7 / 25.0 / 33.1 ms for the certified route by
subtracting kernel rows from the S3 library call and flagged the figure as arithmetic. The
measured library call is **25.10 / 25.39 / 36.26 ms** — within 6 % on rectangular and Delaunay,
9 % on DelaunayNN, and the ordering of every route is as projected. The projection was honest.

## Route (c) — positive-negative, never a bare millisecond

| Quantity | rectangular | delaunay | delaunay_nn |
|---|---:|---:|---:|
| Library call (ms) | 21.128 | 27.161 | 35.079 |
| Δlog-evidence vs the library's S3 solution (nats) | **+334.93** | **+6.404** | **+8.219** |
| Negative reconstruction entries | 136 | 4 | 4 |
| Negative flux fraction | 0.135 % | 0.0017 % | 0.0020 % |
| `solve_ids_to_keep is None` (edge zeroing silently off) | true | true | true |
| Max rel. diff vs phase 0's A2 Cholesky | 7.6e-14 | 1.0e-12 | 7.0e-13 |

The evidence figures reproduce phase 0's A2 row to the last digit, and the reconstruction
matches phase 0's unconstrained Cholesky to ≤ 1e-12 relative — the library reaching the same
infeasible minimiser through `xp.linalg.solve`. **A positive Δ here is not an improvement**: the
unconstrained solution scores above the constrained optimum because it is a different, and
infeasible, minimiser.

## Routes (d) / (e) — certification and the fallback

The QP the injected solver receives is the one the library hands its positive-only solver:
`F + λH` and `D` **already subset to `solve_ids_to_keep`** and Jacobi-scaled exactly as
`reconstruction_positive_only_from` does. `tau_rel = 1e-9`.

| | rectangular | delaunay | delaunay_nn |
|---|---:|---:|---:|
| n, full system | 1521 | 1500 | 1500 |
| n, edge-zeroed by the library | 152 | 0 | 0 |
| **n seen by the solver** | **1369** | 1500 | 1500 |
| Smallest certifying budget (measured here) | **7** | **2** | **2** |
| Phase 0's certifying budget | 7 | 2 | 2 |
| Route (d) budget / certifies | 7 / yes | 2 / yes | 2 / yes |
| Route (e) budget / certifies / fallback fires | 1 / no / **yes** | 1 / no / **yes** | 1 / no / **yes** |

Violation traces per budget, from the same legs (fixed / primal / dual per pass, and
Δlog-evidence vs the library reference at that budget):

| Budget | rect fixed | rect primal | rect dual | rect Δlog-ev | del Δlog-ev | del_nn Δlog-ev |
|---:|---|---|---|---:|---:|---:|
| 1 | 79 | 26 | 20 | +16.08 | +0.01472 | +0.02397 |
| 2 | …, 85 | …, 11 | …, 7 | +5.581 | **+1.2e-10 (cert.)** | **+1.1e-11 (cert.)** |
| 3 | …, 89 | …, 3 | …, 4 | +0.3449 | unchanged | unchanged |
| 4 | …, 88 | …, 0 | …, 2 | −4.4e-09 | — | — |
| 5 | …, 86 | …, 0 | …, 2 | −3.3e-10 | — | — |
| 6 | …, 84 | …, 0 | …, 2 | −3.3e-11 | — | — |
| **7** | …, 82 | …, 0 | …, **0** | **−3.6e-12 (cert.)** | — | — |
| 8 | …, 82 | …, 0 | …, 0 | −3.6e-12 | — | — |

The shape phase 0 found survives the move into the library path exactly: the **primal**
violations are gone by pass 4 and the evidence is already converged there, while the **dual**
trace (20 → 7 → 4 → 2 → 2 → 2 → 0) is what runs rectangular to pass 7; the Delaunay family has
no dual violation at any pass and settles after one correction. The per-pass fixed counts here
are on the 1369-variable subset, not phase 0's 1521.

Route (e) is the honest worst case: when the budget never certifies, the call costs
**41.45 / 51.83 / 58.32 ms** — i.e. the certified solve *plus* the full PDIP, which is still
**1.23× / 1.26× / 1.25× faster than route (a)** because S3 conditioning has already cut the
PDIP iteration count. A production scheme that mispredicts its budget on every call is still
not slower than the library today.

## What is left in the call

Attribution, not a measured decomposition — it subtracts phase-0 kernel rows (a different cell,
a different process) from this cell's library calls:

| | rectangular | delaunay | delaunay_nn |
|---|---:|---:|---:|
| Library call, certified route (d) | 25.104 | 25.386 | 36.257 |
| − phase-0 certified kernel row | 11.049 | 4.213 | 4.161 |
| **= everything that is not the solve** | **≈ 14.1** | **≈ 21.2** | **≈ 32.1** |
| of which (phase 0, S3): curvature+reg build | 5.038 | 4.918 | 4.840 |
| of which: `log det(F+λH)` | 1.241 | 1.171 | 1.199 |
| of which: `log det(H)` | 1.201 | 1.204 | 1.171 |
| **unattributed — mapper, mesh, imaging, blurring** | **≈ 6.6** | **≈ 13.9** | **≈ 24.9** |

Once the solve is 4–11 ms, **it is no longer the call**. On rectangular the residue is modest;
on the Delaunay family the unattributed remainder — mesh construction, the mapper and its
weights — is *larger than everything else put together*, and on DelaunayNN it is 69 % of the
call. That is where phase 2 onwards has to look: another factor of two on the solver buys
almost nothing on Delaunay.

## Two findings the code reading turned up, both recorded in the JSON

1. **The library subsets before it calls.** Under edge zeroing, `Inversion.reconstruction`
   restricts `F + λH` and `D` to `solve_ids_to_keep` *before* invoking the positive-only solver
   and scatters exact zeros back afterwards (`abstract.py:607-618`). The injected wrapper's
   initial fixed set is therefore `zeros(n_keep)` — on rectangular the solver sees n = 1369, not
   1521, because the 152 border pixels are not variables of that QP at all. This is
   mathematically the same problem phase 0 certified at full size with those indices permanently
   fixed, index for index, which is why phase 0's budgets transfer unchanged — and it is the
   reason the transfer is a *fact* here rather than an assumption.
2. **`lax.cond` becomes `select` under `vmap` and evaluates both branches.** A batched route-(d)
   row is therefore the certified solve **plus** the library PDIP: 33.85 ms against route d0's
   21.35 ms on rectangular, measured. **A production implementation must not put a `lax.cond`
   fallback inside a batched path** — it must pad-and-mask, or run a fixed budget with no
   fallback branch in the batched code and handle non-certification outside the batch. The
   single-call rows are unaffected (`cond` really does skip the untaken branch there).

## Verdict

1. **The certified active set halves the library likelihood call.** 50.97 → 25.10 ms
   (rect, 2.03×), 65.10 → 25.39 ms (Delaunay, 2.56×), 72.95 → 36.26 ms (DelaunayNN, 2.01×), at a
   likelihood identical to the library's own to 1e-15…1e-11 relative. This is the production row
   phase 0 could only project, and the projection held.
2. **Fixing the light pays 1.3× on its own**, with no solver change and no library change —
   a → b, on every mesh.
3. **The fallback is affordable.** Even when the budget never certifies (route e) the call is
   still faster than the library today. The risk of a wrong budget is bounded and small; the
   `cond`-under-`vmap` trap, not the fallback cost, is what constrains the implementation.
4. **The positive-negative route is fast and still unusable.** 21.13 ms on rectangular is the
   fastest row measured here, and it costs +334.93 nats and 136 negative pixels; on the Delaunay
   family +6.40 / +8.22 nats and 4 pixels. **It must not enter production before the
   matched-injection witness** phase 0 demanded — a simulated source with known structure, fitted
   with and without positivity, showing the negative pixels change neither the recovered source
   nor the inferred mass model. Until then this is a cost bound, not a candidate. Note also that
   it silently disables edge zeroing, so on rectangular it is not even solving the same system as
   the routes it is being compared with.
5. **The solve is no longer the bottleneck on Delaunay.** ≈ 21 ms of a 25 ms certified Delaunay
   call, and ≈ 32 ms of a 36 ms DelaunayNN call, is not the solver.

## Next

**Phase 2 is DONE** (autolens_profiling#253, note
[`fixed_lens_light_hardware_2026_09.md`](./fixed_lens_light_hardware_2026_09.md)): the same
cell on the CPU at two thread settings and on the laptop RTX 2060 in fp64 and mixed precision,
with the pins re-derived per precision. **Both warnings this note handed it were confirmed,
and one turned out sharper than expected.** The mapper/mesh residue does make a consumer device
worse: the a -> d prize falls from 2.03x / 2.56x / 2.01x here to 1.28x / 1.48x / 1.46x on the
RTX 2060 and 1.16-1.22x on a single CPU thread. And the `cond`/`vmap` constraint turned out to
be moot on consumer hardware, because **`@vmap 16` does not fit at all**: it needs 11.88 GiB on
a 6 GB card, batch 4 also OOMs, batch 2 fits and is *slower per call* than the single call --
and the same batched shape OOM-killed a 16 GB host on the CPU leg. The GeForce fp64 penalty, by
contrast, never bit: mixed precision buys a flat 5-8 % for <= 2.5e-3 nats and 16 % more device
memory, so fp64 stays the consumer-GPU path.

Phases 3-5 (low-likelihood draws, source-pixel scaling, the HST + Euclid verdict) follow in
order; each phase's grid is chosen from the previous phase's answer.

## Artifacts

```
results/breakdown/imaging/fixed_light_library_{rectangular,delaunay,delaunay_nn}_hpc_a100_fp64_fixed_light_library.{json,png}
```

Code:

```
scripts/imaging/likelihood_breakdown/fixed_light_library.py          # the cell
scripts/misc/likelihood_breakdown/library_solver_injection.py        # the scoped monkeypatch + the route-(c) probe
scripts/misc/likelihood_breakdown/active_set_steps.py                # phase 0's kernels, reused unchanged
scripts/misc/test/test_fixed_light_library.py                        # 35 tests
hpc/batch_gpu/submit_breakdown_imaging_fixed_light_library_{pixelization,delaunay,delaunay_nn}_a100_hst_fp64
hpc/batch_gpu/submit_fixed_light_library.sh
```

JSON keys specific to this cell: `routes` (`a_s0_pdip`, `b_s3_pdip`, `c_s3_positive_negative`,
`d_s3_certified_fallback`, `d0_s3_certified_no_fallback`,
`e_s3_certified_budget1_fallback_fires` — each with `ms`, `vmap_ms`, `log_likelihood` and, for
the injected routes, `pass_budget` / `fallback` / `injected_solver_calls_jax`), `certification`
(`n_seen_by_solver`, `smallest_certifying_budget`, `route_e_fallback_fires`, `budgets` — one
entry per pass budget with the per-pass fixed / primal / dual traces), `positive_negative`
(`route_verified`, `d_log_evidence_vs_library_s3`, `n_negative_entries`,
`negative_flux_fraction`, `edge_zeroing_disabled`), `equivalence_pins` (the five),
`solver_injection`, `mapper_block_log_dets`, `vmap`, `jit_phases`, `peak_bytes`,
`device.cache_fresh`, `pinned_expected` / `pinned_drift`. The
`hpc_a100_fp64_fixed_light_library` config label is README-table-invisible by design
(`CONFIG_TAGGED_RE`), like phase 0's; this note and the README prose row are the index into
these legs.
