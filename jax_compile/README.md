# `jax_compile/` — JAX compile-time research

Research instrument + findings for
[autolens_profiling#71](https://github.com/PyAutoLabs/autolens_profiling/issues/71):
for complex likelihoods, XLA compile time is prohibitive — in the worst measured
case it *is* the wall time (A100 pixelized multi-start: ~35 min, almost entirely
compile; the same `input_reduce_fusion` recompiling >30 min).

**Core question:** do we need jit boundaries inside the source code to break up
compilation, or do smaller changes / JAX settings (persistent compilation cache,
tiling choices, compiler flags) get us there?

## Instrument

`probe.py` — AOT-split timings (`trace_s` / `compile_s` / `first_s` /
`steady_s`) per likelihood × transform. See its module docstring for usage.
Records append under `results/<hardware>/<model_type>.json`.

The transform axis mirrors how samplers consume the likelihood — `jit` (Nautilus
row), `vag` (single-start optimizers), `vmap_vag` (MultiStartAdam unbatched),
`laxmap_vag` (MultiStartAdam with `batch_size=`, the production shape).

## Established before this task (do not re-derive)

- **Autotuning ruled out** (2026-07-15): `--xla_gpu_autotune_level=0` vs on —
  2090 s vs 2100 s, identical results to the decimal.
- The same fusion compiled **7m36s** under plain `value_and_grad` vs **>30 min**
  under `lax.map` scan-of-vmap — sampler batching structure matters.
- Fresh-closure-per-call JIT cache-busting is a known stack trap.
- `analysis.print_vram_use()` triggers a full vmapped compile (not a cheap
  diagnostic on heavy cells).

## Findings

_(in progress — populated as measurements land)_

### 1. Where compile time goes (CPU baseline)

_(pending)_

### 2. Does `grad`/`value_and_grad` compound compile cost vs plain `jit`?

_(pending)_

### 3. Cheap levers: persistent cache, tiling, flags

_(pending)_

### 4. Piecewise-jit prototype (the core question)

_(pending)_

## Verdict

_(pending — settings suffice vs source restructuring needed; follow-up prompts
filed via intake)_
