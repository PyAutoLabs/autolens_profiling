# `jax_compile/` — JAX compile-time research

Research instrument + findings for
[autolens_profiling#71](https://github.com/PyAutoLabs/autolens_profiling/issues/71):
for complex likelihoods, XLA compile time is prohibitive — in the worst measured
case it *is* the wall time (A100 pixelized multi-start / Nautilus: ~1h32m, of
which 1h10m was a single `input_reduce_fusion` compile).

**Core question:** do we need jit boundaries inside the source code to break up
compilation, or do smaller changes / JAX settings (persistent compilation cache,
tiling choices, compiler flags) get us there?

**Verdict (2026-07-17): settings suffice — no source restructuring is
indicated.** See "Verdict" below for the recommendation and the follow-up filed.

## Instrument

`probe.py` — AOT-split timings (`trace_s` / `compile_s` / `first_s` /
`steady_s`) per likelihood × transform. See its module docstring for usage.
Records append under `results/<hardware>/<model_type>.json`.

The transform axis mirrors how samplers consume the likelihood — `jit` (Nautilus
row), `vag` (single-start optimizers), `vmap_vag` (MultiStartAdam unbatched),
`laxmap_vag` (MultiStartAdam with `batch_size=`, the production shape),
`pyloop_vag` (the batching boundary hoisted out of XLA into a Python loop).

**Measurement discipline:** compile happens on the *CPU*, so compile timings are
load-sensitive even for GPU jobs (XLA compiles on the host cores). Every number
below is from an otherwise-idle machine or a dedicated SLURM allocation; the
first measurements taken on a loaded machine were wrong by up to **7×** (851 s
vs 117 s for the same compile) and are retained in `results/` only with their
original tags for provenance.

## Pinned warm compile (auto-generated)

The regression surface: what a **warm** compile costs per cell/transform, so a
cache or autotune setting that stops applying shows up as a number moving rather
than as nobody noticing. Derived by `update_pins.py` from the corpus's warm rows
(most recent wins — a pin states what warm costs *now*).

Grouped by comparability key and never merged across it. A warm compile is only
comparable within `(hardware, jax_version, mixed_precision, cache_state)`; a
single ranked table would invite exactly the cross-key comparison the pins exist
to prevent. A `jax_version` bump recompiles once **by design**, so it is a new
key, not drift.

<!-- BEGIN auto-table:jax-compile-warm -->
**`local_cpu` · `DESKTOP-H143S82` · jax 0.10.2**

| Cell | Transform | Warm compile | Source |
|---|---|---|---|
| `datacube_img/mge/hst` | `vag` | 2.40 s | `mb_homo_warm` 2026-07-21T19:43:03 |
| `datacube_img_hetero/mge/hst` | `vag` | 6.96 s | `mb_hetero_warm` 2026-07-21T19:44:20 |
| `imaging/delaunay_matern/hst` | `jit` | 12.96 s | `prodigy-census-warm` 2026-07-28T17:44:17 |
| `imaging/delaunay_matern/hst` | `laxmap_vag` | 27.74 s | `prodigy-census-warm` 2026-07-28T18:38:08 |
| `imaging/delaunay_matern/hst` | `pyloop_vag` | 26.30 s | `prodigy-census-warm` 2026-07-28T18:10:11 |
| `imaging/delaunay_matern/hst` | `vag` | 27.79 s | `prodigy-census-warm` 2026-07-28T17:47:37 |
| `imaging/knn/hst` | `jit` | 263.0 ms | `prodigy-census-warm` 2026-07-28T16:28:21 |
| `imaging/knn/hst` | `laxmap_vag` | 1.79 s | `prodigy-census-warm-retry` 2026-07-28T18:58:25 |
| `imaging/knn/hst` | `pyloop_vag` | 1.73 s | `prodigy-census-warm` 2026-07-28T17:11:53 |
| `imaging/knn/hst` | `vag` | 1.32 s | `prodigy-census-warm` 2026-07-28T16:31:15 |
| `imaging/mge/hst` | `jit` | 312.0 ms | `prodigy-census-warm` 2026-07-28T16:01:24 |
| `imaging/mge/hst` | `laxmap_vag` | 2.32 s | `prodigy-census-warm` 2026-07-28T16:17:06 |
| `imaging/mge/hst` | `pyloop_vag` | 2.81 s | `prodigy-census-warm` 2026-07-28T16:13:44 |
| `imaging/mge/hst` | `vag` | 4.30 s | `prodigy-census-warm` 2026-07-28T16:04:53 |
| `imaging/mge/hst` | `vmap_vag` | 2.95 s | `prodigy-census-warm` 2026-07-28T16:09:52 |
| `imaging/pixelization/hst` | `jit` | 216.0 ms | `prodigy-census-warm` 2026-07-28T16:19:07 |
| `imaging/pixelization/hst` | `vag` | 1.04 s | `prodigy-census-warm` 2026-07-28T16:24:46 |

**`local_cpu` · `euclid-ral-compute-22` · jax 0.10.2**

| Cell | Transform | Warm compile | Source |
|---|---|---|---|
| `imaging/delaunay_matern/hst` | `laxmap_vag` | 21.16 s | `prodigy-census-ral32-warm` 2026-07-28T20:13:59 |
| `imaging/delaunay_matern/hst` | `vag` | 16.29 s | `prodigy-census-ral32-warm` 2026-07-28T20:04:53 |
| `imaging/knn/hst` | `laxmap_vag` | 1.37 s | `prodigy-census-ral32-warm` 2026-07-28T20:29:12 |
| `imaging/mge/hst` | `laxmap_vag` | 1.81 s | `prodigy-census-ral32-warm` 2026-07-28T20:30:57 |
| `imaging/pixelization/hst` | `laxmap_vag` | 1.16 s | `prodigy-census-ral32-warm` 2026-07-28T20:03:15 |
| `imaging/pixelization/hst` | `pyloop_vag` | 1.00 s | `prodigy-census-ral32-warm` 2026-07-28T19:21:06 |

**`local_gpu_NVIDIA_A100_80GB_PCIe` · `euclid-ral-gpu-2` · jax 0.10.2**

| Cell | Transform | Warm compile | Source |
|---|---|---|---|
| `imaging/pixelization/hst` | `jit` | 357.0 ms | `a100-census-warm` 2026-07-17T10:00:32 |
| `imaging/pixelization/hst` | `vag` | 1.82 s | `a100-census-warm` 2026-07-17T10:00:39 |
<!-- END auto-table:jax-compile-warm -->

## Record schema — `cache_state` and `host_state`

Each record carries `cache_state`, **derived from what the compile did** rather
than from its `tag`:

| value | meaning |
|---|---|
| `cold` | the compile wrote a new cache entry — a MISS |
| `warm` | the compile wrote nothing into a populated cache — a HIT |
| `none` | no `--cache-dir`, so the persistent cache was not in play |
| `unknown` | cache configured, empty, and nothing written |

It is derived **per transform**, not per run: each transform compiles its own
module, so one invocation can legitimately miss on one and hit on another —
something the old per-run tag could not express.

Why not parse the tag? It carries ~40 ad-hoc spellings across this corpus
(`census-warm`, `census-warm2`, `prodigy-census-warm-retry`, `mb_homo_cold`, …),
and the obvious shortcut is a trap: **`cache_dir` is non-empty on cold rows
too**, because the cold run is the one that populates the cache.

`host_state` records `cpu_count` and the 1-minute load average. XLA compiles on
the HOST cores, so this is load-bearing rather than bookkeeping — see the 7x
measurement-discipline warning above.

Records written before these fields were backfilled by
`backfill_cache_state.py`: exact where `cache_dir` was empty (`none`), inferred
only from an END-ANCHORED cold/warm tag, and left `unknown` otherwise — 33 cold,
34 warm, 19 none, 3 unknown. The three left `unknown` include
`mb_homo_cold_laxmap_gpu`, which *contains* "cold" mid-tag and is exactly the
false match an unanchored parse would have made.

## Established before this task (do not re-derive)

- ~~Autotuning ruled out (2026-07-15)~~ **downgraded to unproven 2026-07-17**:
  the flag never took effect — `autoconf/jax_wrapper.py` overwrote `XLA_FLAGS`
  (see Verdict item 3); "identical to the decimal" is exactly what clobbering
  produces. Re-test after PyAutoNerves#127 if autotune ever matters again.
- Fresh-closure-per-call JIT cache-busting is a known stack trap (cache the
  jitted closure on the instance).
- `analysis.print_vram_use()` triggers a full vmapped compile (not a cheap
  diagnostic on heavy cells).

## Findings

### 1. Differentiation is the compile multiplier; batching structure is free

MGE HST likelihood (15361 pixels, 8+ params), CPU, idle machine, fresh process
per row (`results/local_cpu/mge.json`, tags `idle-*`):

| transform | trace | XLA compile | steady eval |
|---|---|---|---|
| `jit` | 7.2 s | 10.9 s | 0.08 s |
| `grad` | 13.6 s | 163.9 s | 0.17 s |
| `vag` | 17.1 s | 117.0 s | 0.20 s |
| `vmap∘vag` (n=16) | 17.2 s | 124.2 s | 4.40 s |
| `lax.map∘vag` (bs=4) | 15.0 s | 116.8 s | 3.56 s |
| `pyloop_vag` (jit(vmap₄∘vag) ×4 from Python) | 14.7 s | 105.0 s | 3.01 s |

- `grad`/`value_and_grad` multiplies XLA compile **11–15×** over plain `jit`
  (163.9 s / 117.0 s vs 10.9 s); this is inherent to differentiating the whole
  graph, not a stack defect.
- Every batched-gradient structure compiles in the **same** ~105–125 s band:
  there is **no** `vmap` or `lax.map` compile penalty, and hoisting the batch
  boundary out of XLA (`pyloop_vag`) buys nothing. Earlier apparent penalties
  (388 s / 851 s) were host-load artifacts.
- Tracing (~15 s here, up to ~2 min for deep structures) is pure Python, is
  **not** cacheable, and recurs every process — it is the irreducible floor.

### 2. Compile cost is op-pattern-driven, not "model complexity"-driven

Pixelization (sparse-operator config), CPU, idle: `jit` compiles in **5.0 s**
and `vag` in **30.7 s** — several times *faster* than the parametric MGE model,
despite being the "heavy" likelihood at runtime. The pathological compiles live
in specific op patterns (the MGE positive-only linear-solve graph on CPU; the
kernel-CDF reduce fusion on GPU), so intuition from runtime cost does not
transfer to compile cost.

### 3. The A100 pathology: one ~7m30 fusion, once per shape — and lax.map is innocent

Controlled A/B on dedicated GPU nodes, fresh compilation caches (jobs
330536/330537, logs in `pixgrad_logs/` on RAL):

| shape | total | pathological `input_reduce_fusion` compile |
|---|---|---|
| plain `value_and_grad` (FD probe) | 475 s | **7m24s** |
| `lax.map(vag, batch_size=4)` (MultiStartAdam, full 300×16 fit) | 2081 s | **7m23s** |

Identical to the second. The historical ">30 min, repeatedly" observations were
(a) host-load contention on shared node CPUs and (b) the slow-compile alarm
banner re-firing during *one* long compile. The kernel-CDF pixelized
`value_and_grad` costs one ~7m30 fusion compile per (machine, jax version,
shape) — full stop.

### 4. The persistent compilation cache eliminates it (both scales)

`jax.config.update("jax_compilation_cache_dir", ...)` — cold/warm pairs:

| scale | cold | warm |
|---|---|---|
| local CPU, MGE `vag` | 117.0 s compile | **2.3 s** (51×; residue = trace ~14 s) |
| A100 pixelized Nautilus, end-to-end (jobs 330513 → 330534) | 5517.8 s wall (76 % = the fusion compile) | **937.1 s** (5.9×; compile gone, sampling underway at t≈2 min) |

The cache serves the AOT `.lower().compile()` path across processes, and the
1h10m worst-case fusion serializes into a **1.7 MB** entry. Science output is
unaffected (same basin, sampler stochasticity only). Cache keys include jax
version and shapes, so version bumps recompile once — acceptable.

### 5. Piecewise source jit-boundaries: not pursued, by evidence

The prototype was conditional on (3) showing the *monolith* caused the cold
cost. It does not: the cost is one specific fusion (not module size), batching
structure adds nothing, and the CPU-side "heavy" likelihood compiles fast.
Splitting `log_likelihood_function` into separately-jitted stages would add
host↔device boundary costs and per-stage dispatch to *every* eval, to attack a
one-time-per-machine cost the cache already removes. **Do not restructure.**

## Verdict

**Settings suffice.** Recommendation, in order:

1. **Enable the persistent compilation cache by default** across the stack
   (`jax_compilation_cache_dir` under the workspace `output/` or
   `~/.cache/pyauto_jax`, `jax_persistent_cache_min_compile_time_secs` ~1 s) —
   filed as the follow-up prompt
   `PyAutoMind draft/feature/autofit/enable_the_jax_persistent_compilation_cache_by.md`.
   This turns the worst measured case (70 min) into a once-per-machine cost.
2. **First-fit UX**: the remaining cold cost (~7m30 on GPU pixelized-gradient
   fits; ~2–4 min CPU MGE gradient fits) is honest and unavoidable without
   upstream XLA changes; surface it (log line "compiling — first run on this
   machine takes N min") rather than engineering around it.
3. **Upstream**: the 7m30 single-fusion compile is XLA-report material.
   CORRECTED 2026-07-17: `--xla_dump_to` is not inert — `autoconf/jax_wrapper.py`
   was *overwriting* `XLA_FLAGS` at import, silently discarding user/job flags
   (fixed in PyAutoNerves#127). Two consequences: (a) the HLO dump just needs a
   re-run once that fix lands (or `XLA_FLAGS` including the constant_folding
   disable so the wrapper leaves it alone); (b) the historical 2026-07-15
   "autotuning ruled out" A/B never actually flipped autotune — both runs were
   clobbered to identical flags — so that claim is **unproven** (the controlled
   A/B in finding 3 is unaffected: both sides equally clobbered). Cold-compile
   follow-up: `PyAutoMind draft/research/workspaces/investigate_ways_to_reduce_the_cold_jax.md`.
4. The companion feature prompt (cell-grid compile-time dashboard,
   `draft/feature/autolens_profiling/jax_compile_time_profiling.md`) can now
   reuse `probe.py` and should track *warm* compile times per cell so cache
   regressions are caught.

## Cold-compile findings (issue #74, 2026-07-17)

The persistent cache (above) solves repeat fits; these findings address the
**cold** cost it cannot remove.

### 6. The pathological cold compile IS GPU autotuning — one flag removes it

The first run in this stack's history where `--xla_gpu_autotune_level=0`
actually reached XLA (the pre-#128 wrapper clobbered every earlier attempt —
including the 2026-07-15 "ruled out" A/B and finding 3's controlled A/B, which
compared autotune-ON to autotune-ON):

| A100, fresh cache | autotune ON | autotune OFF |
|---|---|---|
| FD probe total (kernel-CDF pixelized vag) | 498 s | **29 s** (17×) |
| full 300×16 adam fit wall | 2081 s | **1253 s** (−40 %) |
| fixed-input logL | 25536.848940 | 25536.848940 (bit-identical) |

The `input_reduce_fusion` slow-compile alarm was autotune compiling candidate
kernel configs of that fusion as standalone modules (why it never appeared as
a dumpable module name). Autotune results cache like everything else, so with
the cache enabled this is first-fit UX; without the flag a new machine's first
pathological fit pays ~7m30.

### 7. Steady-state eval does not need autotune (measured cells)

A100 probe matrix, fresh caches per job (jobs 330601/330602):

| cell | steady ON | steady OFF | compile ON | compile OFF |
|---|---|---|---|---|
| mge / jit | 0.0042 s | 0.0042 s | 10.1 s | 9.2 s |
| mge / vag | 0.0096 s | 0.0098 s | 29.4 s | 29.3 s |
| pix / jit | 0.0571 s | 0.0574 s | 8.5 s | 5.9 s |
| pix / vag | 0.0871 s | 0.0910 s | 27.3 s | 19.7 s |

Worst case ~4 % on one cell — and the 4800-eval full fit ran *faster* end to
end with autotune off, so no real eval penalty is observed. (These standard
cells compile in seconds either way; the 7m30 autotune cost is specific to the
pathological kernel-CDF no-sparse-operator shape.)

### 8. The tracing floor is jax-internal — no PyAuto lever

cProfile attribution of a 40 s MGE `vag` trace (`trace_profile.py`): 58 % jax
internals, 34 % stdlib/numpy, 7 % autoarray, ~0 % autofit/autogalaxy/autolens.
Reducing it means emitting fewer ops (a jax-side concern), not optimizing
PyAuto Python. Documented and closed as a direction.

## Verdict 2 (cold compile)

**One more setting: default `--xla_gpu_autotune_level=0`.** Evidence: 17×
pathological cold-probe reduction, −40 % cold full fit, bit-identical fixed-
input likelihoods, steady-state eval parity across the measured matrix. With
the cache (#128) plus autotune-off, worst-case first-fit UX drops from ~70 min
to ~30 s. Recommended as an env-respecting wrapper default (same pattern as
the cache), so clusters can re-enable autotune where a tuned kernel matters.

Remaining leads deprioritized by these numbers: cache-entry proliferation and
pre-warming (cold is now ~seconds-to-a-minute); upstream XLA report (the "slow
fusion" is explained — autotune candidates on a 58 GiB fusion; the HLO dump
artifact from job 330596 exists if ever needed).

Not indicated: source restructuring, jit boundaries inside likelihoods,
replacing `lax.map` in MultiStartAdam, autotune flags.

## Final census — the defaults-live user experience (issue #77, 2026-07-17)

Measured through the merged wrapper defaults (#128 cache + #132 autotune-off),
no manual flags; cold = fresh cache dir, warm = same dir, fresh process.

**A100 (dedicated node)** — trace + XLA compile, seconds:

| cell | cold | warm | steady eval |
|---|---|---|---|
| mge / jit | 4.7 + 5.7 | 4.8 + 0.3 | 0.0042 s |
| mge / vag | 6.8 + 27.9 | 6.8 + 2.4 | 0.0098 s |
| pix / jit | 4.0 + 5.7 | 4.2 + 0.4 | 0.0574 s |
| pix / vag | 4.7 + 20.5 | 4.7 + 1.8 | 0.0911 s |

Worst cold cell ≈ **35 s** (was ~70 min in the worst pre-#128/#132 case);
warm ≈ **5–9 s**, almost entirely tracing.

**Local CPU (idle laptop; cross-day compile variance up to 2× — treat the A100
table as the reference):**

| cell | cold | warm | 
|---|---|---|
| mge / jit | 21.0 + 16.9 | 10.9 + 0.5 |
| mge / vag | 22.1 + 229.4 | 16.3 + 2.9 |
| mge / lax.map∘vag | 31.4 + 369.9 | 29.2 + 6.2 |
| pix / jit | 6.7 + 7.7 | 7.6 + 0.3 |
| pix / vag | 7.3 + 34.6 | 8.2 + 1.6 |

## Where any remaining speedup would come (final)

1. **`jax.export` — RULED OUT** (jax 0.10.2): deserialize is 4 ms (vs 16–22 s
   tracing) but the exported module recompiles **~156 s in every process** —
   its compile bypasses the persistent compilation cache (`export_probe.py`,
   two independent load processes). The standard warm path (trace + cached
   compile) wins by an order of magnitude. Re-evaluate only if a future jax
   caches exported-call compiles.
2. **XLA compile-parallelism flags — inconclusive, low ceiling.** CPU
   same-day: 137.7 s vs 229.4 s cold vag compile with
   `--xla_cpu_parallel_codegen_split_count=32`, but cross-day variance on the
   same cell (117 ↔ 229 s) swamps the signal. On the A100 the residual
   no-autotune compile is 20–28 s, so even a 2× flag win saves ~10 s once per
   machine — not worth productizing.
3. **The tracing floor (warm cost) is jax-internal** (finding 8: 58 % jax /
   34 % stdlib / 7 % autoarray). 4–7 s per transform on the A100 host, 11–29 s
   on the laptop, every process. Movable only by upstream jax tracing speed or
   by emitting fewer ops.

**Close-out:** with cache + autotune-off shipped, compile cost is seconds at
every point in the lifecycle. Any further reduction would come from upstream
JAX (tracing speed, compile speed), not from this stack — no further
engineering is warranted here. The compile-time arc (#71 → #74 → #77) is done.

## Multi-band `FactorGraphModel` `value_and_grad` — the heterogeneous-shape cliff

The #71 → #77 census measured **single-band** cells only. Real multi-wavelength
fits build an `af.FactorGraphModel` with one `AnalysisFactor` per band, and the
bands have **different pixel scales** (e.g. JWST F115W/F150W at 0.03 arcsec/px vs
F277W/F444W at 0.06 arcsec/px) — so the factors carry **different masked-pixel
counts**. This section bounds the cold `vag` compile of that graph.

Reproduce via the imaging-datacube cells added to `scripts/misc/searches/_setup.py`
(`datacube_img` = 4 identical `jwst` 0.03″ channels; `datacube_img_hetero` =
2×`jwst` 0.03″ + 2×`jwst_lw` 0.06″, two distinct shapes):

```bash
python scripts/misc/jax_compile/probe.py --dataset-class datacube_img        --model-type mge --transforms vag --cache-dir /tmp/c_homo
python scripts/misc/jax_compile/probe.py --dataset-class datacube_img_hetero --model-type mge --transforms vag --cache-dir /tmp/c_het
```

**Local CPU, MGE `vag`, 4-band factor graph (ndim 15):**

| arm | distinct shapes | cold compile | warm compile | trace | steady eval |
|---|---|---|---|---|---|
| `datacube_img` (homogeneous) | 1 | **120.0 s** | 2.4 s | ~70 s | 0.46 s |
| `datacube_img_hetero` (heterogeneous) | 2 | **704.4 s** | 7.0 s | ~62 s | 0.60 s |

Findings:

1. **Same-shape N-band compile ≈ single-band compile.** Four identical-shape
   factors cost 120 s — the single-band `mge / vag` figure (117 s). XLA fuses the
   identical factors into **one shared kernel**; the factor graph adds no compile
   cost when band shapes match.
2. **Heterogeneous shapes are a 5.9× cold-compile cliff, and superlinear in the
   number of distinct shapes.** Two distinct shapes cost ~6× (not 2×) a single
   fusion — XLA cannot share fused sub-graphs across differently-shaped factors,
   and welds them into one large `jit_call`. Trace and steady-state eval are
   unchanged (~62 s / 0.60 s), so this is a pure XLA fusion-*compilation* effect.
   **This alone is not the observed >1 h** — heterogeneity by itself tops out at
   ~12 min here; see finding 4 for what closes the gap.
3. **The persistent cache rescues both arms** (previously certified single-band
   only): warm compile is 2.4 s / 7.0 s — 50× and 101× — so the heterogeneous
   cliff is a **one-time, first-compile (cache-miss) cost per graph structure**;
   identical restarts warm from disk.
4. **The dominant driver of the real >1 h is the multi-start transform + core
   count, not heterogeneity.** Real fits use `MultiStartProdigy`, i.e.
   `lax.map(value_and_grad, batch_size)` over the starts — a `vmap` of the whole
   factor-graph fusion, `batch_size`-wide. On this **single-core** host
   (`nproc=1`; XLA compiles on host CPUs, so one core is near worst-case) compile
   scales steeply with start-width, on the *homogeneous* 4-band graph:

   | transform | start-width | cold compile |
   |---|---|---|
   | `vag` | 1 | 120 s |
   | `vmap_vag` | 2 | 209 s |
   | `laxmap_vag` (MultiStartProdigy default: `batch_size` 4 / `n_batch` 16) | 4 × scan | **did not finish in 55 min** |

   So the full production transform is intractable to compile cold on one core
   even *before* heterogeneity is added; heterogeneity's 5.9× then stacks on top.
   This reproduces the real >1 h and locates it in the transform × single-core
   compile, with heterogeneity as an additional multiplier. On a multi-core /
   A100 host the absolute numbers drop sharply (XLA parallelises compile across
   cores) — the CPU figures here are worst-case, not representative of HPC runs.
5. **The `lax.map` *scan*, not the multi-start batching, is the compile killer —
   and hoisting the loop out of XLA fixes it.** Holding vmap width fixed at 1 and
   varying only *how* the starts are iterated:

   | transform | multi-start loop | vmap width | cold compile |
   |---|---|---|---|
   | `pyloop_vag` | Python loop (batching hoisted out of XLA) | 1 | **166 s** |
   | `laxmap_vag` | in-XLA `lax.map` (scan) | 1 | **did not finish in >30 min** |

   Same graph, same vmap width — 166 s vs intractable. Compiling a `lax.map`
   scan whose body is a `value_and_grad` of the multi-band fusion is what
   explodes; iterating the starts in Python over small `vmap` chunks (the
   `pyloop` pattern) keeps cold compile at single-fit cost. **Clean re-confirm
   (fresh cache, 10 GB free): the `laxmap bs=1` compile was OOM-killed** (dmesg
   `Out of memory: Killed … anon-rss 6.0 GB`) — so the `lax.map` scan path is
   *memory-explosive to compile* here, not merely slow, whereas `pyloop`
   compiled at modest memory. This is the concrete `MultiStartProdigy` source
   lever — the "candidate `laxmap_vag` replacement" `probe.py` was built to test.
   (The scan's compile-memory blow-up may be host/jax-version specific; a
   multi-core / larger-RAM host might compile it slowly rather than OOM — but the
   `pyloop` path sidesteps it regardless.)

**Verdict / levers for N-band gradient fits:**

- **The cache already amortizes the whole cold cost** — transform, heterogeneity
  and all — for repeated fits of the same graph. Ensure `JAX_COMPILATION_CACHE_DIR`
  is set (shipped default) and the graph structure is stable across runs; this is
  the single biggest lever and it already ships.
- **Compile on a multi-core / GPU host.** The 1-core figures above are worst-case;
  XLA parallelises compilation across cores, so an HPC/A100 first-compile is far
  cheaper. The most certain speed-up for a user hitting the >1 h wall is to run
  the first (cache-populating) fit somewhere with more cores.
- **Python-loop multi-start batching in `MultiStartProdigy` — SHIPPED
  (PyAutoFit#1430, 2026-07-30).** Finding 5 productized: `batch_size` in
  `AbstractMultiStartGradient` now sweeps `jit(vmap(value_and_grad))` chunks
  from a Python loop (one chunk-shaped compile per search) instead of the
  in-XLA `lax.map` scan. The same change jits the broad-start filter's
  single-point objective, which was eager and cost a cache-immune ~13 min per
  multi-band process. Production-path result on this host: cold multi-band fit
  intractable → **395 s CPU / 392 s GPU**, warm **136 s / 199 s**, bit-identical
  numerics. New GPU probe rows show the scan explosion is **CPU-backend
  specific** (GPU compiles `laxmap_vag` in ~122 s at steady-eval parity with
  pyloop), so the Python loop is safe as the only implementation on both
  backends. Full write-up: `results/notes/multiband_pyloop_productized.md`.
- **Immediate user workaround — pad short-wavelength bands to a common grid** so
  all factors share one shape. That removes the heterogeneity multiplier
  (704 s → 120 s cold here) but *not* the transform cost, so it helps most when
  combined with the cache.
- **Open (sub-investigation B, secondary):** whether a **per-factor jit boundary**
  inside `FactorGraphModel.log_likelihood_function` additionally bounds the cold
  cost to N×single-band + a linear combine (attacking the heterogeneity multiplier
  at its source). Not yet measured.

Rows recorded in `results/local_cpu/mge.json` (tags `mb_{homo,hetero}_{cold,warm}`,
`mb_homo_vmap2_cold`, `mb_homo_pyloop_bs1_cold`) and, for the production-width
GPU-backend pair, `results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json`
(tags `mb_homo_cold_{pyloop,laxmap}_gpu`). A100 / multi-core rows are the natural follow-up — the
single-band A100 `vag` cold was ~28 s vs 229 s CPU, and the transform-width and
heterogeneity blow-ups above should both shrink dramatically with more compile
cores.

## MultiStartProdigy transform census — single-band, all endorsed model types (issue #93, 2026-07-28)

MultiStartProdigy is now the endorsed search for MGE and pixelized sources
(rectangular kernel-CDF / knn / delaunay, wsdev#117), but the #71→#77 census
measured single-start transforms only, and the mesh model types were never
compile-probed. This census measures the **production transform matrix** —
{`mge`, `pixelization`, `knn`, `delaunay_matern`} × {`jit`, `vag`, `vmap_vag`
(n=16), `pyloop_vag` (bs=4), `laxmap_vag` (bs=4)} × cold/warm — at the
production knobs (16 starts, `batch_size` 4). Tags
`prodigy-census-{cold,warm}` in `results/local_cpu/<model_type>.json`.

Host: the 1-core, **15 GB** WSL laptop — the worst-case compile tier (XLA
compiles on host cores), deliberately: if compile is fine here it is fine
everywhere. Memory limits below are host-tier facts, not library defects.

**Cold trace + XLA compile, seconds (warm compile in parentheses):**

| cell | jit | vag | vmap_vag (16) | pyloop_vag (4) | laxmap_vag (4) |
|---|---|---|---|---|---|
| mge | 11 + 14 (0.3) | 16 + 150 (4.3) | 31 + 179 (3.0) | 26 + 140 (2.8) | 26 + 120 (2.3) |
| pixelization | 5 + 4 (0.2) | 8 + 21 (1.0) | OOM-exec | OOM-exec | OOM-exec |
| knn | 5 + 4 (0.3) | 8 + 25 (1.3) | OOM-exec | 13 + 35 (1.7) | 19 + 55 (1.8†) |
| delaunay_matern | 39 + 11 (**13**) | 18 + 33 (**28**) | OOM-host | 21 + 33 (**26**) | 15 + 28 (**28**) |

† first warm attempt was host-OOM-killed mid-steady on the 15 GB host; the
retry served compile from cache in 1.8 s (tag `prodigy-census-warm-retry`) —
transient memory pressure, not structural.

### Findings

1. **Single-band MultiStartProdigy compile is NOT pathological — on any
   endorsed model type.** Worst cold cell on the worst-case host is MGE
   `vmap_vag` at ~210 s total; every mesh batched cell compiles in **35–55 s**
   cold and warms to seconds. The multi-band `lax.map` compile explosion
   (previous section: intractable / compile-OOM) **does not reproduce
   single-band**: `laxmap_vag` bs=4 compiles in 28–120 s across all four model
   types. The scan blow-up needs the multi-band factor-graph fusion as its
   body; a single-band likelihood body is benign. **Consequence: no PyAutoFit
   restructuring is indicated for single-band fits** — the settings-suffice
   verdict extends to the full production transform. The pyloop lever remains
   live *only* for the multi-band `FactorGraphModel` case.
2. **The delaunay family busts the persistent compilation cache.** Every
   `delaunay_matern` transform recompiles at cold cost in every process (warm
   compile ≈ cold: 13/28/26/28 s vs 11/33/33/28 s; n=8 process pairs), while
   the pure-JAX meshes (`knn`, `pixelization`, `mge`) warm to 0.2–4 s. Prime
   suspect: the qhull `pure_callback` in the Delaunay tables path — callback
   custom_calls embed a process-specific descriptor in the HLO, so the cache
   key never matches across processes. knn (no host callback) caching
   perfectly is the control. Cost: ~40–65 s of trace+compile per process,
   forever, on the mesh family where it can least be amortized. Follow-up
   filed: `PyAutoMind draft/research/autoarray/delaunay_callback_persistent_cache_miss.md`.
3. **Rect kernel-CDF batched gradients are memory-bound, not compile-bound:
   ~9.2 GB per start in the jvp** (sparse-operator config, fp64, 15361 px):
   width 4 → 37 GB (`RESOURCE_EXHAUSTED` on this host, fits RAL 128 GB CPU /
   A100 80 GB — exactly the campaign's working configs), width 16 → 147 GB
   (fits nowhere; `batch_size=4` is load-bearing, not a tuning nicety).
   Compile itself finished within each ~43 s wall before the exec-OOM. knn
   width 16 wants 47.5 GB; bs=4 knn/delaunay peak ~10–12 GB — borderline on a
   15 GB host, comfortable anywhere real.
4. **Steady-state per-step cost on 1 CPU core** (16-start batched eval): mge
   3–5 s, knn ~290–350 s, delaunay_matern ~180–205 s — mesh multi-start on a
   laptop CPU is hopeless for throughput regardless of compile; the compile
   verdict above is what matters because production mesh fits run on RAL/A100.

### RAL 32-core / 128 GB tier (job 331379, tags `prodigy-census-ral32-*`)

Cold trace + XLA compile, seconds (warm compile in parentheses):

| cell | vag | pyloop_vag (4) | laxmap_vag (4) | steady (16-start eval) |
|---|---|---|---|---|
| mge | — | — | 14 + 63 (1.8) | 0.4 s |
| pixelization | — | 9 + 16 (1.0) | 10 + 18 (1.2) | ~310 s |
| knn | — | — | 11 + 21 (1.4) | ~107 s |
| delaunay_matern | 10 + 16 (**16**) | — | 12 + 21 (**21**) | ~59 s |

- **The rect batched cells complete and compile fast on a real node** (job
  MaxRSS 39.4 GB — the predicted ~37 GB jvp): 16–18 s cold, ~1 s warm. Rect's
  cost is *throughput* (~310 s per 16-start step on 32 CPU cores — the
  campaign's known ~5.7 min/step), never compile.
- **The delaunay cache-miss reproduces on an independent host, to the
  decimal**: warm compile 16.3 s = cold 16.3 s (vag), 21.2 s = 21.2 s
  (laxmap), while knn/mge/rect warm to 1–2 s in the same job. Finding 2
  is cross-host confirmed.
- 32 compile cores buy 2–6× over the 1-core laptop (mge laxmap 63 s vs
  120 s; mesh cells 16–21 s vs 28–55 s), consistent with XLA's
  multi-core compile parallelism.

### A100 tier — attempted, not obtained (job 331380)

The GPU job ran while the node's A100s were saturated by an external multi-day
array: `cuInit(0)` returned `CUDA_ERROR_NO_DEVICE` and **JAX silently fell back
to CPU** (`An NVIDIA GPU may be present ... Falling back to cpu`). Its rows are
therefore 8-core CPU rows, not A100 rows, and were discarded rather than
committed.

**Trap:** a `--partition=gpu --gres=gpu:1` job that gets no usable device does
not fail — it warns and runs on CPU, producing plausible-looking numbers.
Verify the backend from the results themselves (the `local_gpu_*` vs
`local_cpu` output path) rather than trusting the partition; a "GPU" row slower
than a many-core CPU row (here: knn `laxmap_vag` 160 s vs the 32-core CPU's
107 s) is the tell.

A100 rows remain **confirmatory only** — #77 already put single-band A100
compiles at seconds-to-30 s, and the two CPU tiers agree the verdict is not
tier-sensitive. Re-run with `sbatch /mnt/ral/jnightin/pixgrad_logs/census_gpu.sbatch`
when a GPU node is genuinely free.

### Verdict (issue #93)

**Single-band MultiStartProdigy compile time is a non-problem on every
endorsed model type and every tier measured** — worst case ~3.5 min cold on a
1-core laptop, ≤ 75 s cold / ≤ 2 s warm on a 32-core node. The multi-band
`lax.map` compile explosion does not exist single-band, so **the pyloop
PyAutoFit change is not indicated** (phase B: evidence-based no-go); the
pyloop lever stays reserved for the multi-band `FactorGraphModel` case
documented in the previous section. The one real compile defect this census
found is the **delaunay-family persistent-cache miss** (finding 2) — filed as
`PyAutoMind draft/research/autoarray/delaunay_callback_persistent_cache_miss.md`.
`batch_size=4` is load-bearing for memory on all pix meshes (finding 3), and
mesh multi-start throughput (not compile) is the open cost axis, on the A100
follow-up list from wsdev#117.
