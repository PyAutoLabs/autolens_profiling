# MultiStartProdigy JAX compile times — MGE and all pixelized meshes

**Issue:** [autolens_profiling#93](https://github.com/PyAutoLabs/autolens_profiling/issues/93)
**Branch:** `feature/multistart-prodigy-compile`
**Status:** Local CPU (1-core) and RAL 32-core CPU tiers complete. A100 tier
outstanding (see "A100 tier" below).
**Instrument:** [`scripts/misc/jax_compile/probe.py`](../../scripts/misc/jax_compile/probe.py);
full tables and provenance in
[`scripts/misc/jax_compile/README.md`](../../scripts/misc/jax_compile/README.md).

## TL;DR

**`af.MultiStartProdigy` compile time is a non-problem on every endorsed model
type — MGE, rectangular kernel-CDF, KNN and Delaunay alike.** Nothing needed
speeding up. Worst case is ~3.5 min cold on a 1-core laptop (the deliberate
worst-case tier — XLA compiles on host cores); on a 32-core node every cell is
≤ 75 s cold and ≤ 2 s warm.

> **Correction — 2026-09-27:** "≤ 2 s warm" excludes the Delaunay family: finding 2 below and the `scripts/misc/jax_compile/README.md` table show `delaunay_matern` warm = cold (16.3 s `vag`, 21.2 s `laxmap_vag`, RAL job 331379) because the qhull `pure_callback` never hits the persistent cache (`PyAutoMind/draft/research/autoarray/delaunay_callback_persistent_cache_miss.md`).

The one real defect this census found is that **the Delaunay family never hits
the persistent compilation cache** — it pays full compile in every process,
forever, on the mesh family that can least afford it.

## Why this was measured

The compile-time arc ([#71](https://github.com/PyAutoLabs/autolens_profiling/issues/71)
→ [#77](https://github.com/PyAutoLabs/autolens_profiling/issues/77)) certified
"settings suffice" (persistent cache + autotune-off) for **single-start**
transforms only, and never probed the pixelized mesh model types. Meanwhile
`MultiStartProdigy` became the endorsed search for both MGE and pixelized
sources (autolens_workspace_developer#117), and a multi-band follow-up found
its production transform — `jax.lax.map(value_and_grad, batch_size=)` — could
be compile-*intractable* (OOM-killed) on a factor-graph likelihood. This census
asked whether that reproduces for ordinary single-band fits.

## Findings

### 1. Single-band multi-start compile is benign — the `lax.map` blow-up does not reproduce

{`mge`, `pixelization`, `knn`, `delaunay_matern`} × {`jit`, `vag`, `vmap_vag`
(n=16), `pyloop_vag` (bs=4), `laxmap_vag` (bs=4)}, cold and warm, at production
knobs (16 starts, `batch_size=4`):

| tier | worst cold (trace + XLA compile) | warm |
|---|---|---|
| 1-core, 15 GB laptop (worst case) | ~3.5 min (MGE `vmap_vag`; meshes 35–75 s) | 2–4 s |
| RAL 32-core, 128 GB node | ~75 s (MGE `laxmap_vag`; meshes 26–33 s) | 1–2 s |

The multi-band scan explosion needs the multi-band factor-graph fusion as its
scan body; a single-band likelihood body compiles in 28–120 s across all four
model types. **Consequence: no PyAutoFit change is indicated for single-band
fits** — the "settings suffice" verdict now extends to the full production
multi-start transform. The Python-loop (`pyloop`) batching lever stays reserved
for the multi-band `FactorGraphModel` case.

### 2. The Delaunay family busts the persistent compilation cache

Every `delaunay_matern` transform recompiles at cold cost in **every process**
(warm compile ≡ cold: 16–33 s), while `knn`, `pixelization` and `mge` warm to
0.2–4 s. Reproduced to the decimal on two independent hosts (laptop and a RAL
32-core node), n=8 process pairs.

KNN — pure JAX, no host callback — caching perfectly is the control, so the
prime suspect is the **qhull `pure_callback`** in the Delaunay tables path:
callback `custom_call`s embed a process-specific descriptor in the serialized
HLO, so the cache key never matches across processes.

Cost: ~40–65 s of trace + compile per process, indefinitely. Follow-up filed as
`PyAutoMind draft/research/autoarray/delaunay_callback_persistent_cache_miss.md`.

### 3. Rectangular batched gradients are memory-bound, not compile-bound

The rect kernel-CDF jvp costs **~9.2 GB per start** (fp64, 15361 image pixels,
sparse-operator config):

| start width | jvp allocation | outcome |
|---|---|---|
| 4 (`batch_size=4`, production) | ~37 GB | OOM on a 15 GB laptop; fits RAL CPU (job MaxRSS 39.4 GB) and A100 80 GB |
| 16 (unbatched `vmap`) | ~147 GB | fits nowhere |

**`batch_size=4` is load-bearing for memory, not a tuning nicety** — this is the
quantitative basis for the campaign's working configuration. Compile itself
completes in 16–21 s on a real node; rect's cost is throughput (~310 s per
16-start step on 32 CPU cores, matching the campaign's ~5.7 min/step), never
compile.

## A100 tier

Not yet measured. The submitted GPU job (331380) was allocated on the `gpu`
partition while the node's A100s were saturated by an external multi-day array;
`cuInit(0)` returned `CUDA_ERROR_NO_DEVICE` and **JAX silently fell back to
CPU**, so its rows are 8-core CPU rows, not A100 rows (they were discarded, not
committed).

**Trap for future GPU submissions:** a `--partition=gpu --gres=gpu:1` job that
gets no usable device does not fail — it logs a warning and runs on CPU at
full speed-looking numbers. Always verify the backend from the results
(`hardware` field / `results/local_gpu_*` path) rather than trusting the
partition, and treat a "GPU" row slower than a many-core CPU row as suspect.

The A100 rows are **confirmatory only** — #77 already put single-band A100
compiles at seconds-to-30 s, and both CPU tiers agree the verdict is not
tier-sensitive. Re-run when a GPU node is genuinely free:

```bash
sbatch /mnt/ral/jnightin/pixgrad_logs/census_gpu.sbatch
```

## Reproducing

From the `autolens_profiling` root, one cell at a time:

```bash
python3 scripts/misc/jax_compile/probe.py --model-type knn --transforms laxmap_vag \
  --n-batch 16 --batch-size 4 --cache-dir /tmp/jcc_knn --tag repro
```

Run cold (a fresh `--cache-dir`) then warm (the same dir, a fresh process).
**Compile happens on the host CPU, so timings are load-sensitive even for GPU
jobs** — use an idle machine or a dedicated allocation; historical
loaded-machine numbers were wrong by up to 7×.
