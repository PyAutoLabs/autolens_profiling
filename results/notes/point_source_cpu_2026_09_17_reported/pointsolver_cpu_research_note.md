# PointSolver image-plane chi-squared on CPU — research note

**Question (verbatim from the human):** *"I want us to speed up the point_source image plane
chi-squared, with this issue focusing on CPU. This will likely use the JAX implementation, for
other LH functions we had existing numba code to build on and there was lots of sparsity to
exploit. Im not sure doing a whole numba CPU implementation is worth it. However, its worth
some research, and asking if the JAX CPU implementation is sufficiently sub optimal that there
are obvious low hanging fruit improvements with a clever CPU sparse approach, noting that
cluster modeling will rely heavily on the PointSolver."*

**Answer in one line.** The JAX CPU path is sub-optimal, but not because of sparsity or padding:
**~84 % of the call is a `jnp.unique` sort that is provably a no-op on the JAX path.** Deleting it
gives **4.9×** on the simple point-source likelihood and **2.4×** on the real cluster solve with
**bit-identical** output. After that fix the solve sits close to its bare-deflection lower bound, so
**a numba CPU implementation is not worth it** — and the existing NumPy path is the evidence, not a
guess: it is the unpadded sparse implementation, it does 2.5× fewer deflection evaluations, and it
is **12× slower** than the fixed JAX path.

Read-only study. No file inside any repo under `<workspace>` was modified. All
scripts, HLO dumps and result JSONs live beside this note.

---

## 1. Environment and measurement caveats (read before quoting a number)

| item | value |
|---|---|
| machine | WSL2 laptop, 8 logical cores, affinity 0–7 |
| `NPROC` / `nproc --all` | `8` / `8` (XLA CPU intra-op pool correctly sized; `nproc` prints `1` only because `OMP_NUM_THREADS=1` is exported and coreutils honours it — JAX is unaffected) |
| pins in env | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (BLAS only; JAX CPU does not use them) |
| `XLA_FLAGS` | `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` (workspace-wide; **its effect on this cell was not measured** — see §8) |
| precision | `JAX_ENABLE_X64=True`, fp64 throughout |
| backend | `JAX_PLATFORMS=cpu`, `jax 0.10.2`, `autolens/autoarray 2026.8.17.1` |
| method | every case compiled, ≥3 warm calls, then **round-robin interleaved** (A B … B A) timing, ≥20–40 timed calls, **median** and **p10** reported |

**Timing variance is the dominant caveat.** Run-to-run medians for the *same* configuration ranged
25.9 → 42.3 ms across sessions, and individual calls spiked to 3–4× the median (`max` columns
below). Interleaving makes *within-run ratios* trustworthy; *absolute* numbers across runs are not.
Every speed-up quoted here is a within-run ratio.

**Not like-for-like with the committed baseline.** The committed
`results/runtime/point_source/image_plane_solved/image_plane_solved_summary_v2026.7.23.1.json`
records eager 273.5 ms / single-JIT 38.6 ms / vmap-3 28.1 ms on `al 2026.7.23.1`. That JSON carries
**no machine or device context field**, so the 38.6 ms cannot be compared like-for-like with the
25.9–42.3 ms measured here on a different laptop and a later library version. The structural
findings (FLOP counts, HLO sort shapes, triangle counts) are machine-independent and are the
primary evidence.

---

## 2. Algorithm characterisation — `simple` preset, JAX path

Configuration as `image_plane_solved.py` builds it:
`al.Grid2D.uniform(shape_native=(100,100), pixel_scales=0.2)` →
`PointSolver.for_grid(pixel_scale_precision=0.001, magnification_threshold=0.1)`.

| property | value |
|---|---|
| solver extent | y, x ∈ [−9.9, 9.9] (grid pixel *centres*, not edges) |
| initial triangle side `scale` | 0.2 (= `grid.pixel_scale`) |
| `n_steps` | **8** = `ceil(log2(scale / pixel_scale_precision))` = `ceil(log2(200))` |
| `MAX_CONTAINING_SIZE` | **15** (module constant in `autoarray/structures/triangles/array.py`; bound as a default arg at class-definition time, so it is *not* reachable through any `PointSolver` argument) |
| `neighbor_degree` | 1 |
| initial triangles (step 0) | **23 283** (2.3× the 10 000 grid pixels) |
| observed positions / solved images | 4 / 4 (of 15 padded rows) |

### What one refinement step does

`AbstractSolver.steps` (`PyAutoLens/autolens/point/solver/shape_solver.py`) per step:

1. `_plane_triangles` → `triangles.vertices` (a `jnp.unique` — see §3) → `tracer.deflections_between_planes_from` on those vertices → `with_vertices` builds an `ArrayTriangles`.
2. `containing_indices` → barycentric `Point.mask` over all triangles → `jnp.where(inside, size=15, fill_value=-1)`.
3. `for_indexes` on the *image-plane* `CoordinateArrayTriangles` (15 rows, `-1` → NaN).
4. `neighborhood()` → ×4 + `jnp.unique(size=60)` → 60 rows.
5. `up_sample()` → ×4 → **240** rows, side length halved. This is the next step's tiling.

So steps 1…7 all run at a **fixed padded 240 triangles / 720 vertex rows**, independent of how many
triangles actually contain the source.

### Deflection evaluations per solve (measured, `characterise.json`)

| step | triangles | vertex rows traced | finite | NaN-pad waste |
|---:|---:|---:|---:|---:|
| 0 | 23 283 | **69 849** | 28 665 | 59.0 % |
| 1–7 (each) | 240 | 720 | 104–152 | 79–86 % |
| **total / solve** | | **74 889** | 29 583 | **60.5 %** |

### The solve is invoked twice per likelihood — and it does not matter

`AbstractFitPositionsImagePair.model_data` is a plain `@property`, and
`FitPositionsImagePairAll` reads it twice (`all_permutations_log_likelihoods` line 118,
`chi_squared` line 175). The instrumented trace shows **16 step records = 2 × 8 steps**. But XLA
CSEs the two identical `custom_jvp` primals: `full_pipeline` = 15 582 802 FLOP vs `solve_once` =
15 529 212 FLOP (0.3 % apart) and the two time the same (§4). **Caching `model_data` is cosmetic,
not a lever** — it would only cut trace/compile time.

### What the fit costs on top of the solve

`full_pipeline − solve_once` = 53 590 FLOP = **0.34 %** of the call. `_filter_low_magnification`
runs a Hessian magnification on 15 points; `FitPositionsImagePairAllSolved` does a 4×15 log-sum-exp
pairing. The analytic β\* centre solve alone measured **0.61 ms** compiled. None of this is a lever.

---

## 3. The mechanism — why the "dedup" is a no-op under jit

`CoordinateArrayTriangles._vertices_and_indices`
(`PyAutoArray/autoarray/structures/triangles/coordinate_array.py`):

```python
flat_triangles = self.triangles.reshape(-1, 2)          # (3N, 2) = (69849, 2)
vertices, inverse_indices = jnp.unique(
    flat_triangles, axis=0, return_inverse=True,
    size=3 * self.coordinates.shape[0],                 # size = 3N — the SAME row count
    equal_nan=True, fill_value=jnp.nan,
)
```

`jnp.unique` needs a static output shape under jit, so it is given `size=3N`. The result therefore
has **exactly as many rows as the input it deduplicated** — 69 849 — of which 28 665 are the real
unique vertices and **41 184 are NaN fill**. `_plane_triangles` then ray-traces all 69 849 rows.

So on the JAX path the dedup:

* **saves zero deflection evaluations** (69 849 traced either way);
* **costs one lexicographic sort of 69 849 fp64 rows** (visible in the HLO as the single
  `f64[69849] sort`), plus a gather to rebuild the triangles;
* and its output — the `ArrayTriangles` — is used for **one thing only**: `containing_indices`.
  The kept set is then taken from the *image-plane* `CoordinateArrayTriangles` via `for_indexes`,
  so the deduplicated vertex table is thrown away every step.

**This is a JAX-path-only defect.** The NumPy sibling
(`coordinate_array_np.py::_vertices_and_indices`) calls `np.unique` with no `size=`, so it
genuinely shrinks 69 849 → **28 665** rows and really does save 59 % of the deflection work
(measured, §6). The static-shape requirement inverts the optimisation into a pure cost.

---

## 4. Measurement (a)+(c) — full pipeline, one solve, and the deflection lower bound

Interleaved, 30 rounds, one process (`bench_abba.json`):

| case | median ms | p10 ms | min ms | max ms | FLOP | HLO sorts |
|---|---:|---:|---:|---:|---:|---:|
| `full_pipeline` (log-likelihood) | **25.88** | 23.71 | 22.24 | 31.96 | 15 582 802 | 15 |
| `solve_once` (`model_data` only) | 27.17 | 24.14 | 22.88 | 35.74 | 15 529 212 | 15 |
| bare deflections, N = 69 849 | **2.71** | 2.30 | 2.09 | 6.09 | 2 916 501 | 0 |
| bare deflections, N = 28 665 | 1.31 | 1.13 | 1.08 | 1.96 | 1 351 509 | 0 |
| bare deflections, N = 720 | 0.47 | 0.40 | 0.31 | 1.06 | 289 599 | 0 |

**Overhead fraction.** One solve does 74 889 deflection evaluations = 2.71 + 7 × 0.47 ≈ **5.98 ms**
of deflection work inside a **25.9 ms** call → **(Z−Y)/Z = 77 % overhead** in the best-measured run,
and 86 % in the 42 ms runs. Physics is a minority of the call.

HLO op histogram for `solve_once` (28 562-line module, 641 fusions): 1 549 `broadcast`, 1 116
`compare`, 777 `select`, 507 `slice`, 66 `reduce`, 54 `gather`, 31 `scatter`, **15 `sort`**. No
`while`, no custom-calls, no dots — nothing structurally pathological beyond the sorts. Sort shapes:
one `f64[69849]` (step 0) and 14 small ones (7 × `f64[720]` from `remove_duplicates`, 7 × `f64[60]`
from `neighborhood`).

---

## 5. Measurement (b) — step-0 decomposition

Step 0 is the whole call (§7 proves the other 7 steps cost ~3 ms). Scratch re-implementations of
each piece, interleaved 30 rounds (`bench_step0.json`):

| piece | median ms | p10 ms | FLOP | sorts | share of `step0_current` |
|---|---:|---:|---:|---:|---:|
| `step0_current` (unique → trace → contain) | **43.39** | 32.03 | 14 454 467 | 1 × 69 849 | 100 % |
| `step0_unique_only` (the `jnp.unique` alone) | **35.61** | 27.77 | 8 299 384 | 1 × 69 849 | **82 %** |
| `step0_defl_only` (trace 69 849 rows) | 4.85 | 3.43 | 3 056 199 | 0 | 11 % |
| `step0_contain_only` (`Point.mask` + `where(size=15)`) | 1.27 | 1.00 | 2 074 458 | 0 | 3 % |
| `step0_nodedup` (trace flat → contain, no unique) | **7.68** | 5.33 | 5 410 049 | **0** | 18 % |
| `full_pipeline` (same process) | 42.29 | 33.71 | 15 582 802 | 15 | — |

`step0_current` and `step0_nodedup` return the **identical** kept-triangle index vector
`[-1 ×6, 10465, 11641, 12118, 12935, 13168, 13169, 13284, 13285, 13633]`.

**The dedup is 82 % of step 0 and ~84 % of the whole likelihood call, and buys nothing.**

---

## 6. The fix, proven end-to-end (`bench_nodedup_full.json`)

`AbstractSolver._plane_triangles` monkeypatched **in the scratch process only** to trace
`triangles.triangles.reshape(-1, 2)` directly and return a minimal object exposing
`containing_indices`. Full `AnalysisPoint.log_likelihood_function`, 40 interleaved rounds:

| case | median ms | p10 ms | min ms | max ms | FLOP | sorts | temp MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (library as-is) | 40.03 | 34.80 | 32.43 | 94.29 | 15 582 802 | 15 (69 849, 720×7, 60×7) | 5.20 |
| **no-dedup** | **8.14** | **6.42** | 6.17 | 18.23 | 6 495 109 | 7 (60 × 7) | 4.58 |
| **speed-up** | **4.92×** | **5.42×** | | | −58 % | | −12 % |

`log_likelihood` = `7.743201200876812` **bit-identical** in both — the same value the committed
regression pin `EXPECTED_LOG_LIKELIHOOD_IMAGE_PLANE_SOLVED` records
(`7.743201200876817`, within the script's own `rtol=1e-4`).

### Post-fix budget (8.14 ms)

| component | ms | share |
|---|---:|---:|
| step-0 deflections (69 849 pts) | ~2.7 | 33 % |
| step-0 containment + tiling | ~1.3 | 16 % |
| steps 1–7 (7 × ~0.45 ms, each a 720-pt deflection call) | ~3.2 | 39 % |
| β\* analytic centre + magnification filter + χ² | ~0.6 | 7 % |

The solve is now within **~1.4×** of its deflection lower bound.

---

## 7. Sensitivity (`bench_nsteps_grid.json`, 25 interleaved rounds, current library)

### n_steps (via `pixel_scale_precision = 0.2 / 2^k`)

| n_steps | precision | median ms | p10 ms | FLOP | log-likelihood |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 | 35.34 | 31.40 | 14 513 470 | 2.551 |
| 2 | 0.05 | 36.48 | 32.61 | 14 666 296 | 6.255 |
| 4 | 0.0125 | 39.36 | 33.17 | 14 971 798 | 7.804 |
| 6 | 0.003125 | 39.10 | 34.69 | 15 277 300 | 7.725 |
| 7 | 0.0015625 | 41.30 | 34.35 | 15 430 051 | 7.712 |
| 8 | 0.00078125 | 38.55 | 34.27 | 15 582 802 | 7.743 |
| 9 | 0.000390625 | 39.24 | 34.77 | 15 735 553 | 7.738 |

**All seven extra refinement steps together cost ≈3 ms (~8 %), i.e. ~0.45 ms each** (+152 751 FLOP
per step, exactly linear). Step 0 alone is ~92 % of the call. Cutting `n_steps` is *not* a lever —
and it degrades the likelihood badly below 4 steps.

### Initial grid extent (fixed `pixel_scales=0.2`, fixed `n_steps=8`)

| grid | extent (″) | triangles | median ms | p10 ms | FLOP | log-likelihood |
|---|---|---:|---:|---:|---:|---:|
| 30×30 | ±2.9 | 2 065 | **6.41** | 5.42 | 2 626 350 | 7.743201200876812 |
| 50×50 | ±4.9 | 5 841 | 11.89 | 10.75 | 4 917 337 | 7.743201200876812 |
| 100×100 | ±9.9 | 23 283 | 39.80 | 34.36 | 15 582 802 | 7.743201200876812 |
| 140×140 | ±13.9 | 45 477 | 72.25 | 65.91 | 29 336 582 | 7.743201200876812 |

Cost is **linear in the tiled area** (≈0.5–1.0 µs per flat-vertex row). The likelihood is
**bit-identical** across all four: the profiling cell tiles ±9.9″ to find images at ~1.6″, so
**92 % of the tiled area is wasted** and a 30×30 solver grid is a free **6.2×**.

### `MAX_CONTAINING_SIZE` — **NOT MEASURED**

The sweep script (`bench_mcs_vmap.py` / `bench_postfix.py`) was written but the run was cut by the
stall watchdog. Structurally: `MAX_CONTAINING_SIZE` fixes steps 1–7 at 15 → 60 → 240 triangles /
720 vertices, i.e. it scales the ~3.2 ms tail only, and cannot touch step 0. Expected gain from
halving it is therefore ≲1.5 ms post-fix, and it is a **correctness knob** (only 9 of 15 slots were
used here; a cluster with more images needs more). Low priority, and it is not reachable from
`PointSolver`'s API today.

---

## 8. The NumPy path — the free "sparse, unpadded" datapoint (`bench_numpy.json`)

Same solve, `xp=np`, dynamic shapes, no padding, real `np.unique`:

| step | triangles | vertices traced (all finite) |
|---:|---:|---:|
| 0 | 23 283 | 28 665 |
| 1 | 116 | 141 |
| 2–7 | 80 each | 104–152 |
| **total** | | **29 563** (vs 74 889 padded on the JAX path — **2.53× fewer**) |

| case | median ms | min | max |
|---|---:|---:|---:|
| `solver.solve(xp=np)` | **96.12** | 88.70 | 187.1 |
| full eager `log_likelihood` (2 solves + Python) | 233.26 | 186.2 | 326.0 |
| NumPy deflections on 28 665 pts | 2.15 | 2.13 | 2.36 |

**This is the decisive argument against a sparse CPU rewrite.** The unpadded implementation already
exists, already exploits the sparsity (2.5× fewer deflection evaluations), and is **96 ms** against
the JAX path's 40 ms as-shipped and **8.1 ms** once the redundant sort is removed — **12× slower**.
Its own deflection lower bound is 2.15 ms, so 98 % of *its* time is bookkeeping too. Sparsity is not
where the time is; a redundant sort was.

---

## 9. Cluster scale (`bench_cluster.json`)

Model rebuilt from `scripts/cluster/likelihood_breakdown/image_plane.py`: 2 main dPIE + 10 scaling
dPIE + NFW host at z = 0.5 (**13 mass components**), 2 point sources at z = 1.0 / 2.0,
**3 planes**, multi-plane, `plane_redshift` passed per source.
*Caveat:* the committed `dataset/cluster/simple/` ships no `scaling_galaxies.csv`, so the
scaling-tier centres/luminosities were reconstructed from the simulator source literals
(`scripts/misc/simulators/cluster.py` L160–175) rather than read from disk. Source centres are the
back-traced observed-position centroids, as the breakdown cell does it.

### (i) Cluster mass model at the *point-source* grid (100×100 @ 0.2″, 8 steps) — isolates lens complexity

| case | median ms | p10 ms | FLOP | sorts |
|---|---:|---:|---:|---:|
| current | 67.5 – 72.2 | 59.4 – 62.2 | 41 270 500 | 15 (69 849 …) |
| **no-dedup** | **31.7** | 28.1 | — | — |
| bare deflections, 69 849 pts, cluster tracer | 13.1 – 14.1 | 9.9 – 10.5 | 26 360 390 | 0 |

Positions **identical** between current and no-dedup. Speed-up **2.1×**. Deflection share rises from
~19 % (current) to **~41 %** (post-fix) purely because 13 mass profiles cost 4.8× one Isothermal.

### (ii) The real cluster breakdown configuration (200×200 @ 0.7″, precision 0.01″)

Solver extent ±69.65″, `scale` 0.7, **n_steps = 7**, **92 169 initial triangles → 276 507 flat
vertex rows**.

| case | median ms | p10 ms | FLOP | sorts | temp MB |
|---|---:|---:|---:|---:|---:|
| `solve z=1.0` current | 117.3 | 101.8 | 153 858 672 | 13 (276 507, 720×6, 60×6) | 81.3 |
| `solve z=1.0` **no-dedup** | **48.0** | 44.1 | 120 533 944 | 6 (60 × 6) | 80.4 |
| `solve z=2.0` current | 120.1 | 97.1 | 154 346 608 | 13 | 81.3 |
| `solve z=2.0` **no-dedup** | **49.6** | 46.3 | 121 583 528 | 6 | 80.4 |

Solved positions **identical** for both sources. **2.44× / 2.42×** per source; a 2-source cluster
likelihood goes **237 ms → 98 ms**. FLOPs fall only 22 % while time falls 59 % — `cost_analysis`
under-counts a comparison-bound lexicographic sort, which is exactly why FLOPs alone would have
missed this defect.

**Is the cluster case deflection-bound after the fix?** Linearly extrapolating the measured cluster
bare-deflection cost (13.1 ms / 69 849 pts = 0.187 µs/pt) to 276 507 points gives ≈51.8 ms, against
a measured no-dedup solve of 48.0 ms. So **post-fix the cluster solve is at (or just under) its
deflection lower bound** and the remaining lever at cluster scale is the mass-profile deflection
code itself — dPIE/NFW evaluation across 13 components — not solver bookkeeping. *This is an
extrapolation, not a measurement:* the 276 507-point bare deflection was not timed directly (larger
arrays amortise fixed overhead better, so the true bound is probably a little lower).

---

## 10. Ranked levers

| # | lever | measured gain | implementation cost | evidence |
|---|---|---|---|---|
| **1** | **Drop the throwaway `jnp.unique` vertex dedup on the JAX trace path** — trace `triangles.reshape(-1,2)` and reshape back to `(N,3,2)` instead of `vertices`+`indices` | **4.9× median / 5.4× p10** simple (40.0→8.1 ms); **2.4×** real cluster (117→48 ms/source); **2.1×** cluster@small grid | **Small and local.** `CoordinateArrayTriangles.with_vertices` / `AbstractSolver._plane_triangles` are the only consumers on the solve path; the deduped table is thrown away each step. Keep `ArrayTriangles` for `ShapeSolver`/plotting and add a no-dedup trace path. | §5, §6, §9; outputs bit-identical (LL `7.743201200876812`; cluster positions `array_equal`) |
| **2** | **Shrink the solver grid extent** (config/docs/defaults, not code) | **6.2×** (39.8→6.4 ms) for 100×100→30×30, bit-identical LL | Zero code. Needs guidance: extent must cover the image separation, not the data array. | §7 grid table |
| **3** | **Coarsen step 0 and add refinement steps** — cost is `area/scale²` at step 0 and only ~0.45 ms per extra step | doubling `scale` should quarter step 0 (≈4× on the dominant term) | Moderate; a *completeness* risk — coarse initial triangles can miss an image whose source-plane triangle image is small. **Not measured**: the grid sweep varied extent, not `scale`. | §7 both tables (23 283→2 065 triangles ⇒ 39.8→6.4 ms; +152 751 FLOP per extra step) |
| **4** | **Reduce padding in steps 1–7** (`MAX_CONTAINING_SIZE` 15→8, or drop the 240-triangle fixed fan-out) | ≲1.5 ms post-fix; 0 % of step 0 | Small, but changes how many images can be found — a correctness knob, not a free win. **Not measured.** | §2 (240/720 padded vs 104–152 finite), §7 n_steps table |
| **5** | Cache `model_data` so the solve is written once | **0 %** runtime (XLA already CSEs it); helps trace/compile only | Trivial | §2: `full_pipeline` 15 582 802 vs `solve_once` 15 529 212 FLOP, same time |
| **6** | Attack the mass-profile deflection code (dPIE/NFW) | the *only* remaining lever at cluster scale | Large | §9(ii): post-fix cluster solve 48.0 ms vs ≈51.8 ms extrapolated deflection bound |
| **7** | Warm-start from the previous sampler call's triangles | untested | **Recommend against.** Static shapes would need the previous solution as an input, breaking the pure-function contract `vmap` and the `implicit_diff` `custom_jvp` rely on, and making the likelihood path-dependent — a silent correctness hazard for a sampler. | §2, `implicit_diff.py` contract |
| **8** | XLA CPU pathology | none found beyond the sorts | — | §4 HLO histogram: no `while`, no custom-calls, 641 fusions. **Untested:** the workspace-wide `--xla_disable_hlo_passes=constant_folding` — the step-0 tiling is a large compile-time constant, so this flag is a plausible second-order cost and deserves one A/B run. |

---

## 11. Numba verdict — **not worth it**

1. **The premise does not hold here.** The other likelihood functions had numba kernels to build on
   *and* exploitable sparsity. The PointSolver has neither: there is no pre-existing numba solver,
   and the "sparse, unpadded" implementation that would be the target of a numba rewrite **already
   exists as the NumPy path** and is **12× slower** than the fixed JAX path (96 ms vs 8.1 ms) while
   doing 2.5× *fewer* deflection evaluations (§8). Padding is not the cost.
2. **The headroom is gone after a ~20-line fix.** Post-fix the simple solve is within ~1.4× of its
   deflection lower bound and the cluster solve is *at* it (§6, §9). A numba rewrite could only
   contest the 27 % (simple) → ~0 % (cluster) of the call that is still bookkeeping.
3. **The cost is high.** A numba/`pure_callback` refinement loop forfeits the `custom_jvp` implicit
   gradient (`implicit_diff.py`), `vmap` batching, and the GPU path that cluster modelling will want
   — for a fraction of a fraction.
4. **Consistent with workspace precedent.** `PyAutoMind/complete/2026/09/numba-interferometer-kernel-levers.md`
   ("the recovered kernel is not worth reinstating"; numba won 1 of 6 cells) and
   `fixed-light-numba-solver.md` (NNLS speed-up round CLOSED) reached the same shape of answer:
   numba wins only where a *specific* geometry-gated sparsity exists, which it does not here.

**Recommended sequence:** lever 1 (library fix, PyAutoArray + PyAutoLens, red-control the
bit-identical likelihood and the cluster positions) → lever 2 (workspace/doc guidance on solver
grid extent) → re-profile → then decide whether lever 3 or lever 6 is worth a phase.

---

## 12. What I could not measure (blocked / out of time)

* **vmap batch 1/4/16 on CPU** — script written (`bench_postfix.py` group C), run cut by the stall
  watchdog. The only vmap datapoint is the committed JSON's batch-3 (28.1 ms/call vs 38.6 ms
  single-JIT = 1.4×) on a different machine and library version.
* **`MAX_CONTAINING_SIZE` sweep** — same; see §7 for the structural bound.
* **Post-fix grid / n_steps sensitivity** — same; the pre-fix sweep is in §7.
* **`--xla_disable_hlo_passes=constant_folding` A/B** — never run.
* **Direct bare-deflection timing at 276 507 points** (cluster) — §9(ii) is a linear extrapolation.
* **Compile times.** `cluster_smallgrid_nodedup` took **42.9 s** to compile against 9.7 s for the
  current version — but at real cluster scale the no-dedup variants compiled **faster** (11.7 s /
  9.2 s vs 17.6 s / 20.2 s), and at simple scale `step0_nodedup` compiled in 0.42 s vs 0.74 s. The
  42.9 s is therefore almost certainly a **cold-cache artefact**: it was the first trace after
  `jax.clear_caches()` with the patch installed, so it re-traced the full model/Tracer pytree
  registration. No evidence the fix costs compile time; a dedicated A/B would settle it.

### Measurement trap worth recording

**`jax` caches traced jaxprs on function *identity*.** The first run of the no-dedup proof reported
"1.48× speed-up" with **byte-identical HLO** (15 sorts in both) — the monkeypatch had been installed
but `jax.jit(same_function_object).lower(...)` re-served the pre-patch trace. Any monkeypatch A/B of
library internals must use a **distinct function object** *and* call `jax.clear_caches()` before
each compile. Corrected numbers are the ones in §6; `harness.py` now clears caches per compile.

---

## 13. Reproducing

All under this directory; run with cwd = `<workspace>/autolens_profiling` and
`NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cpu`.

| script | produces | what it shows |
|---|---|---|
| `common.py` | — | replicates `image_plane_solved.py` PART A setup |
| `harness.py` | — | round-robin interleaved timing + FLOP/HLO structural readout |
| `characterise.py` | `characterise.json` | per-step triangle/vertex/NaN counts, n_steps, MCS, double solve |
| `bench_breakdown.py` | `bench_breakdown.json` | first-pass full/solve/deflection timings |
| `bench_abba.py` | `bench_abba.json`, `hlo_*.txt` | §4 table, HLO op histogram, sort shapes |
| `bench_step0.py` | `bench_step0.json` | §5 step-0 decomposition + identical-kept-set check |
| `bench_nsteps_grid.py` | `bench_nsteps_grid.json` | §7 n_steps and grid sweeps |
| `bench_numpy.py` | `bench_numpy.json` | §8 NumPy path |
| `bench_nodedup_full.py` | `bench_nodedup_full.json` | §6 end-to-end fix proof |
| `bench_cluster.py` | `bench_cluster.json` | §9 cluster scale |
| `bench_postfix.py`, `bench_mcs_vmap.py` | — | written, **not run** (§12) |
