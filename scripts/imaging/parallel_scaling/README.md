# Parallel scaling — numba CPU sparse-operator likelihood

Multiprocessing-scaling profiling for the **numba CPU** likelihood path
(`apply_sparse_operator_cpu()` + `AnalysisImaging(use_jax=False)`) — the
workspace `cpu_fast_modeling.py` production route for pixelized-source
modeling on CPU-abundant hardware. Companion cells:
`../likelihood_runtime/pixelization_numba.py` (per-eval cost) and
`../likelihood_breakdown/pixelization_numba.py` (per-step decomposition).

## Scripts

| Script | What it measures |
|--------|------------------|
| `pixelization_numba.py` | Likelihood throughput vs `--cores` under (a) serial in-process, (b) the Pool-object design `af.Nautilus(number_of_cores=P)` uses today, (c) an initializer-cached worker pool; plus the pickle payload each design pays. |

```bash
python scripts/imaging/parallel_scaling/pixelization_numba.py --instrument euclid
python scripts/imaging/parallel_scaling/pixelization_numba.py --cores 1,2,4,8,16,32 --n-points 100   # HPC CPU node
```

Results land in `results/parallel_scaling/imaging/` as versioned
`{pixelization,delaunay}_numba_scaling_<instrument>_v<version>.{json,png}`. The RAL
submit scripts are `hpc/batch_cpu/submit_parallel_scaling_pixelization_numba_euclid`
(Rectangular) and `hpc/batch_cpu/submit_parallel_scaling_delaunay_numba_euclid`
(campaign fiducial).

## How PyAutoFit's Nautilus parallelizes today (traced 2026-08-20)

- `number_of_cores > 1` → `fork_context().Pool(P)` passed to nautilus as a Pool
  **object**; nautilus's `Pool.map(fitness.call_wrap, points)` pickles the
  bound method — the whole `Fitness` (model + analysis + dataset + sparse
  operator) — once per chunk, ~4×P times per `n_batch=100` batch.
- nautilus's `pool=<int>` branch (not taken) would cache the likelihood once
  per worker via a Pool `initializer`; PyAutoFit's dormant `SneakierPool`
  implements the same idea. Variant (c) quantifies what switching would buy.
- `number_of_cores = 1` builds no pool (plain in-process map) — structurally a
  different path, so 1→2 cores compares "no IPC" against "fork + per-chunk
  pickle", not pure worker-count scaling.
- The same pool also trains nautilus's neural nets (`n_networks=4`), so
  end-to-end search scaling mixes in a non-likelihood parallel component this
  harness deliberately excludes.

> **Fiducial note:** the Findings section below is for the Rectangular
> (28×28 `RectangularBilinearAdaptDensity`) fiducial. The production campaign fiducial
> is Delaunay + Hilbert-1250 image-mesh (the `delaunay_numba` runtime/breakdown
> cells): run it with `--mesh delaunay` (artifacts land as
> `delaunay_numba_scaling_<instrument>`). First Delaunay pass (euclid, 4-core
> cloud container, 24 evals/map, 2026-08-20): serial 4.73 s/eval; speedup
> 1.8x at P=2, 3.4x at P=4 (84-86% efficiency); object-pool vs
> initializer-cached indistinguishable — the 4.7 s eval amortizes the 36.8 MB
> per-chunk fitness pickle, unlike the ~2 s Rectangular eval below; zero
> corrupted worker evals. The 1-32-core RAL sweep stays the authoritative
> core-count guidance.

## Findings — first pass, 2026-08-20 (v2026.8.17.1, local 8-core WSL box)

Fiducial: 28×28 `RectangularBilinearAdaptDensity` + `Constant` reg, MGE-60 linear lens
light, mask 3.5". Full JSONs under `results/{runtime,breakdown,parallel_scaling}/imaging/`.

**Per-evaluation cost** (`likelihood_runtime` cell):

| Instrument | masked px | per eval | operator setup (one-off) | operator memory |
|-----------|-----------|----------|--------------------------|-----------------|
| euclid (0.1") | 3,841 | **2.0 s** | 1.2 s | 35 MB |
| hst (0.05") | 15,361 | **21.6 s** | 6.0 s | 169 MB |

**Where the time goes** (`likelihood_breakdown` cell): the **mapper
sparse-triplet construction** (`Mapper.sparse_triplets_data` — building the
sparse [data → source-pixel] mapping weights) dominates and scales brutally
with resolution: 1.10 s = 49% of the euclid eval, **18.8 s = 88% of the hst
eval**. The numba linear algebra everyone would suspect is minor by
comparison (curvature F: 0.24 s euclid / 1.24 s hst; MGE operated mapping
matrices: 0.42 s / 0.87 s; BLAS solve: 0.27 s / 0.53 s). **Optimizing
`sparse_triplets_data` is the single biggest lever for this pipeline.**

**Multiprocessing scaling** (this harness, euclid, 50 evals/map):

| P | object pool (nautilus today) | initializer-cached |
|---|------------------------------|--------------------|
| 1 | 1.00× (100%) | 1.00× (100%) |
| 2 | 1.53× (77%) | 1.48× (74%) |
| 4 | 2.30× (57%) | 2.21× (55%) |
| 8 | 2.50× (31%) | 2.70× (34%) |

- The two variants are indistinguishable: the 36.8 MB per-chunk fitness pickle
  costs only ~15-30 ms against a 2 s evaluation, so **the object-pool design's
  re-pickling is NOT the bottleneck at these eval costs** (it would matter for
  sub-100 ms likelihoods). Switching PyAutoFit to nautilus's `pool=<int>` /
  `SneakierPool` design is not the win here.
- Efficiency decays with P (77% → 31%) with no IPC cost to blame —
  **memory-bandwidth contention** between workers (the triplet/curvature
  kernels stream large arrays) and this WSL box's topology are the suspects.
  The workspace docstring's "speed-up ≈ half the cores" held at P≤4 and
  plateaued at ~2.5-2.7× on 8 — re-measure on a real HPC CPU node (submit
  script provided) before quoting node-level guidance.
- Zero corrupted worker evaluations in this pass (~700 pool evals) — the
  intermittent hazard below did fire in an earlier 8-eval probe (2/8), so the
  counters stay.

## Known hazard — corrupted first/worker evaluations

Two related correctness hazards were found while building this harness
(tracked as a PyAutoArray bug prompt in PyAutoMind,
`draft/bug/autoarray/numba_first_call_garbage_psf_weighted_data.md`):

1. **Cold numba cache:** the first likelihood evaluation in a process returns
   NaN — `psf_weighted_data_from` returns uninitialized-memory-scale garbage
   (~1e299) on its first freshly-compiled call, with inputs identical to the
   correct second call. Warm caches are unaffected.
2. **Forked workers:** pool evaluations of *identical* parameter vectors
   intermittently return the `resample_figure_of_merit` sentinel (-1e99) —
   i.e. some worker-side likelihoods silently fail and would be discarded as
   resamples in a production `number_of_cores > 1` run.

The harness counts these per map (`corrupt_evals_first_map`,
`corrupt_evals_steady_maps` in the JSON) rather than asserting them away —
treat non-zero steady-state counts as a red flag for CPU-parallel campaigns
until the library bug is fixed.

## BLAS threading guidance

All numba kernels on this path are single-threaded (no `prange` anywhere in
the stack); only the reconstruction solve + Cholesky log-dets (~13% of a
euclid likelihood, see the breakdown cell) use BLAS/LAPACK threads. Amdahl
caps multi-threaded-BLAS gains at ~13%, so: **pin BLAS to 1 thread per worker
(`OMP_NUM_THREADS=1` etc.) and give every core to `number_of_cores`** —
which is also what nautilus's `sample_shell` enforces internally via
`threadpool_limits(limits=1)`.
