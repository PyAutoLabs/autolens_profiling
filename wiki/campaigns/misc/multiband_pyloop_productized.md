# Multi-band pyloop batching productized in MultiStartGradient (PyAutoFit#1430, 2026-07-30)

The `jax_compile/README.md` finding-5 lever — hoist the multi-start batching out
of XLA into a Python loop — is now the shipped implementation of
`AbstractMultiStartGradient` (`batch_size` set): the in-XLA
`jax.lax.map(value_and_grad, batch_size=)` scan was replaced by a Python sweep
over `jit(vmap(value_and_grad))` chunks (ragged final chunk padded, so exactly
one `(batch_size, ndim)`-shaped program is compiled per search). A second,
independent cost was found and fixed in the same change: the broad-start filter
evaluated an **eager** (un-jitted) single-point `value_and_grad` per draw —
cache-immune and ~13 min per multi-band process; it now reuses one jitted
single-point objective built in `_fit` (persistently cached like everything
else).

## Production-path benchmark

End-to-end `MultiStartProdigy._fit` on the homogeneous 4-band cell
(`datacube_img` = 4×`jwst` MGE `af.FactorGraphModel`, ndim 15; `n_starts=16`,
`batch_size=4`, `n_steps=2`; fresh vs same-dir persistent cache; 1-core WSL
laptop, RTX 2060 Max-Q via the PyAutoGPU venv, jax 0.10.2):

| arm | cold fit | warm fit |
|---|---|---|
| CPU, old `lax.map` path | intractable (chunk compile alone >55 min, OOM-killed — README finding 5) | — |
| CPU, pyloop only (eager broad-starts) | 1013.7 s | 838.1 s |
| CPU, pyloop + jitted broad-starts (shipped) | **395.5 s** | **136.2 s** |
| GPU, pyloop + jitted broad-starts (shipped) | **392.3 s** | **198.5 s** |

`best_fom` is **bit-identical** (5023604.7103) across every arm, backend and
cache state; the Gaussian-cell parity check (`batch_size` None vs 4 vs ragged 5)
is also bit-identical per start. Warm time is almost entirely re-tracing (host
CPU, 1 core) — the compile itself warms from the persistent cache.

## New census rows: the scan explosion is CPU-backend-specific

Probe rows at the production widths (`n_batch=16`, `batch_size=4`), GPU backend
(tags `mb_homo_cold_{pyloop,laxmap}_gpu` in
`results/local_gpu_NVIDIA_GeForce_RTX_2060_with_Max-Q_Design/mge.json`):

| transform | trace | cold compile | steady (16-start sweep) |
|---|---|---|---|
| `pyloop_vag` | 102.5 s | 117.9 s | 0.729 s |
| `laxmap_vag` | 146.0 s | 121.7 s | 0.740 s |

The multi-band `lax.map` compile explosion **does not reproduce on the GPU
backend** — the same scan that is intractable/OOM under the CPU backend
compiles in ~2 min with the CUDA pipeline on the same 1-core host. Steady-state
throughput is at parity (0.74 vs 0.73 s per sweep), so the Python loop's host
dispatch costs nothing measurable at these widths — the fallback consideration
in PyAutoFit#1430 (keep the scan behind an opt-in if GPU throughput regressed)
is **not needed**; pyloop is safe as the only implementation on both backends.

## Follow-ups (not measured here)

- A100 / multi-core rows for the multi-band matrix (`census_gpu.sbatch` re-run
  once the A100s free up) — confirmatory.
- `datacube_img_hetero` GPU rows (heterogeneity multiplier under the CUDA
  pipeline).
- The padding/shape-canonicalization user workaround and the per-factor jit
  boundary (README verdict bullets) remain open, secondary levers.
