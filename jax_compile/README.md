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

## Established before this task (do not re-derive)

- ~~Autotuning ruled out (2026-07-15)~~ **downgraded to unproven 2026-07-17**:
  the flag never took effect — `autoconf/jax_wrapper.py` overwrote `XLA_FLAGS`
  (see Verdict item 3); "identical to the decimal" is exactly what clobbering
  produces. Re-test after PyAutoConf#127 if autotune ever matters again.
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
   (fixed in PyAutoConf#127). Two consequences: (a) the HLO dump just needs a
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

Not indicated: source restructuring, jit boundaries inside likelihoods,
replacing `lax.map` in MultiStartAdam, autotune flags.
