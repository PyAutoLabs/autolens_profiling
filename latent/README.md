# latent

Per-latent runtime profiling for the PyAutoLens library latent-variable catalogue. The headline question is:

> *"How long does each latent cost per call, and is the closure cache for `effective_einstein_radius` actually helping?"*

Run scripts here — or, more commonly, the [`sweep.py`](sweep.py) driver — when you need to predict the cost of enabling a latent at scale (every-sample mode), compare CPU vs GPU throughput for the same latent, or measure first-call vs cached-call timing for the JAX-jit path through `LensCalc.einstein_radius_jit_from()`.

For the latent **values** (correctness, not timing) see `autolens_workspace_test/scripts/latent/latent_variables_smoke.py`. The two are deliberately disjoint — this package times; that one validates.

## Methodology

Each script measures **one** latent in isolation:

1. **Conf override.** Before constructing the Analysis, the script writes a temporary `config/latent.yaml` and calls `conf.instance.push(...)` to mark only the target latent as enabled. PyAutoFit's `compute_latent_samples` therefore dispatches just this one function, no contamination from the other four.
2. **Eager numpy baseline.** `AnalysisImaging(..., use_jax=False)` + a single call to `compute_latent_variables(parameters, model)`. This is the correctness reference and the worst-case (un-JIT'd) cost.
3. **Single-call JIT.** `jax.jit(lambda p: analysis_jax.compute_latent_variables(p, model))` — records `lower`, `compile`, `first-call`, and `steady-state × 10` (steady-state averaged). The steady-state number is what production code in N-draws mode actually pays per draw.
4. **Vmap batched.** `jax.jit(jax.vmap(...))` with batch=3 — records the per-call cost as `batch_time / batch_size`. Vmap is the honest measurement: per-sample JIT on a concrete `ModelInstance` can constant-fold parts of the computation and read 20-30× faster than reality (see memory `feedback_jax_pure_callback_const_fold`).
5. **Closure cache delta** (effective_einstein_radius only). The LensCalc `_zero_contour_cache` at `autogalaxy/operate/lens_calc.py:1580-1586` memoises the `(eigen_fn, ZeroSolver)` pair. We time first-call and second-call on the same fresh `LensCalc` to surface the cache hit. Expected: second call is ~20-50% faster on numpy, much more on JIT (one full recompile avoided).

All JAX timings use `block_until_ready()` to force synchronous measurement. Errors that arise from missing optional deps (`jax_zero_contour`, jax extras) are recorded in `jit_error` / `vmap_error` fields rather than failing the script — sweeps need to keep going past per-config failures.

## The 6-config matrix

Same matrix as `likelihood_runtime/`:

| Config | Backend | Precision | Env / Flag |
|--------|---------|-----------|------------|
| `local_cpu_fp64` | CPU | fp64 | `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu` |
| `local_cpu_mp` | CPU | mixed | same + `--use-mixed-precision` |
| `local_gpu_fp64` | RTX 2060 | fp64 | `JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu` |
| `local_gpu_mp` | RTX 2060 | mixed | same + `--use-mixed-precision` |
| `hpc_a100_fp64` | A100 (80 GB) | fp64 | SLURM-dispatched separately |
| `hpc_a100_mp` | A100 | mixed | same + `--use-mixed-precision` |

The `cuda,cpu` listing on GPU configs is load-bearing — the `effective_einstein_radius` path needs a CPU device available even when the primary platform is CUDA, because `ZeroSolver` uses `jax.lax.cond` / `jax.lax.while_loop` that occasionally fall back to host evaluation under specific solver states.

## What mixed precision means here

For the four flux latents (`total_lens_flux_mujy`, `total_lensed_source_flux_mujy`, `total_source_flux_mujy`, `magnification`), mixed precision affects the upstream `AnalysisImaging.fit_from(instance)` call — specifically the PSF convolution and the mapping-matrix accumulation if the lens / source uses linear light profiles. The latent itself is just a reduction (sum + magzero conversion), so its direct cost is unchanged; mp moves the needle by making the fit cheaper to build per sample.

For `effective_einstein_radius`, mixed precision is essentially a no-op — the `ZeroSolver` and the underlying deflection-field evaluation stay in fp64. The Einstein radius is sensitive enough that downcasting would compromise the zero-contour fidelity.

Expect: mp helps the flux latents in proportion to the underlying fit cost (~5-20%), and is neutral on `effective_einstein_radius`.

## Scripts

| Script | Latent | Cost class | Notes |
|--------|--------|-----------|-------|
| `imaging/total_lens_flux_mujy.py` | `total_lens_flux_mujy` | trivial | Sum over `fit.galaxy_image_dict[fit.galaxies[0]].array` + magzero conversion. ~µs scale once JIT'd. |
| `imaging/total_lensed_source_flux_mujy.py` | `total_lensed_source_flux_mujy` | trivial | Same shape as above, source index `[-1]`. |
| `imaging/total_source_flux_mujy.py` | `total_source_flux_mujy` | low | Evaluates `tracer_linear_light_profiles_to_light_profiles.galaxies[-1].image_2d_from(grid=...)` — heavier than the dict-lookup latents because it computes a fresh source-plane image. ~10x the dict-lookup variants. |
| `imaging/magnification.py` | `magnification` | low | Composes the lensed and intrinsic source fluxes; cost is dominated by the `total_source_flux_mujy` recompute. |
| `imaging/effective_einstein_radius.py` | `effective_einstein_radius` | high | The marquee — JIT path through `LensCalc.einstein_radius_jit_from` → `ZeroSolver.zero_contour_finder` → `jnp.roll` shoelace. First-call dominated by JAX trace + ZeroSolver compile. Closure cache hit on second call removes the `_make_eigen_fn` rebuild. |

## Driving the matrix — `sweep.py` and `aggregate.py`

```bash
# All 5 latents, local CPU + GPU x fp64 + mp (8 configs total per latent)
python latent/sweep.py

# Restrict to one latent during iteration
python latent/sweep.py --only imaging/effective_einstein_radius

# Skip a backend
python latent/sweep.py --skip-gpu       # CPU only
python latent/sweep.py --skip-cpu       # GPU only

# Aggregate per-config JSONs into a single comparison artefact
python latent/aggregate.py
```

Per-config JSONs land at `<output_root>/imaging/<latent_name>/<config_name>.json`. The aggregator produces `<output_root>/imaging/<latent_name>/comparison.json` + `.png` with one row per config and the production cost (steady-state JIT for N-draws mode; eager numpy for the every-sample fallback).

Default output root: `<wt_root>/autolens_workspace_developer/jax_profiling/results/latent/`. Mirrors the `likelihood_runtime/` precedent.

## GPU practicalities

If you're running locally on the RTX 2060 Max-Q (6 GB), set:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
renice -n 10 -p $$
```

The `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` cap stops JAX from preallocating most of the 6 GB VRAM (which makes the desktop unusable). The renice keeps the profiling job from stealing CPU from interactive work. The HPC A100 (80 GB) doesn't need either.

## How to read the output

Each per-config JSON looks like:

```json
{
  "latent_key": "effective_einstein_radius",
  "config_name": "local_cpu_fp64",
  "use_mixed_precision": false,
  "eager_value": 0.0,
  "eager_time_s": 0.0175,
  "closure_cache_first_call_s": 0.0170,
  "closure_cache_second_call_s": 0.0133,
  "jit_value": ...,
  "jit_lower_s": ...,
  "jit_compile_s": ...,
  "jit_first_call_s": ...,
  "jit_steady_state_s": ...,
  "jit_error": null,
  "vmap_per_call_s": ...,
  "vmap_value": ...,
  "vmap_error": null
}
```

Read in this order:

1. **`jit_steady_state_s`** — the per-call cost in production. This is the headline for N-draws-from-PDF mode (`compute_latent_variables` runs `latent_draw_via_pdf_size=100` times per fit). If it's larger than `eager_time_s`, JIT isn't helping for this latent (typical for the trivial flux latents, where eager numpy is already a few µs and JIT compile dominates).
2. **`vmap_per_call_s` vs `jit_steady_state_s`** — should be similar. If vmap is dramatically faster, the JIT path is hitting a constant-fold and the single-call number is overstated.
3. **`closure_cache_first_call_s` vs `_second_call_s`** (Einstein radius only) — the cache delta on numpy. A small delta (<10%) means the cache is being used but the per-call work dominates (i.e. cache hit doesn't save much). A large delta (>30%) means the cache is the right optimisation. Zero delta means the cache isn't being hit at all — investigate.
4. **`jit_error` / `vmap_error`** — non-null means the optional JAX extras (`jax_zero_contour` for Einstein radius, others as appropriate) aren't installed. Numpy fallback timings remain valid; install the extras to fill in the JIT/vmap columns.

The aggregator surfaces the production-cost column (steady-state JIT, or eager if JIT failed) as the headline, with first-call and compile times in adjacent columns for full provenance.

## When the cache helps / hurts

The `_zero_contour_cache` at `lens_calc.py:1580-1586` memoises by `(kind, pixel_scales, tol, max_newton)`. Two scenarios:

- **Cache helps**: every sample in a posterior draw uses the same solver settings (the default), so every call after the first reuses the same `(eigen_fn, ZeroSolver)` pair. First-call pays the `_make_eigen_fn` cost (which is the dominant cost on the JIT path); every subsequent call is pure compute. Expect 30-60% cache-hit speedup on JIT; 15-25% on numpy.
- **Cache hurts** (rare): if downstream code constructs a fresh `LensCalc` per call (instead of reusing one), the cache never hits. The `total_source_flux_mujy` and `effective_einstein_radius` latents both do `LensCalc.from_mass_obj(fit.tracer)` per call — but within a single `compute_latent_variables` invocation, the LensCalc is constructed once and the cache lives on it. Across calls (different posterior draws), JAX's JIT compile cache picks up the slack.

If you see the cache delta drop to zero across runs, suspect that the calling code is rebuilding LensCalcs between samples instead of reusing one. The current PyAutoLens dispatcher does it the right way — this is more relevant for hand-rolled custom Analysis subclasses (see memory `feedback_jax_closure_cache_busts` for the pattern).
