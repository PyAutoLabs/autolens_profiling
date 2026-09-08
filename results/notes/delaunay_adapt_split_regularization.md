# Delaunay cells re-based on AdaptSplit regularization (A100 A/B, 2026-09-08)

`autolens_profiling` issue #232, branch `feature/delaunay-adapt-split-regularization`. Every
Delaunay-family profiling cell used to measure `ConstantSplit(coefficient=1.0)`, while
production (the Euclid pipeline, SLaM) pairs Delaunay meshes with `AdaptSplit`. The profiled
rows were therefore not the regularization production pays for. This note records the
same-node A100 A/B that re-bases the cells and prices the change.

## What changed

1. **`--regularization {adapt_split,constant_split}`** (shared flag, `_profile_cli.py`). The
   five Delaunay-family cells — `likelihood_breakdown/{delaunay,delaunay_nn}.py`,
   `likelihood_runtime/{delaunay,delaunay_nn,delaunay_numba}.py` — default to `adapt_split`;
   every other cell ignores the flag. `constant_split` reproduces the historical scheme
   exactly, so the pre-2026-09-08 rows stay reachable and comparable.
2. **Coefficients `AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1)`,
   not the constructor defaults.** `AdaptSplit()`'s defaults are `inner = outer = 1.0`, and
   `adapt_regularization_weights_from` returns `outer` everywhere when `inner == outer` — the
   scheme then degenerates to `ConstantSplit(1.0)`, the pins would not move, and the A/B would
   prove nothing. `0.1 / 10.0 / 0.1` are the production-shaped values already used in-repo by
   `likelihood_breakdown/delaunay_numba_nnls_iterations.py`.
3. **Adapt-image wiring.** The cells' `AdaptImages` carried only the image-plane mesh grid, but
   `AdaptSplit.regularization_weights_from` → `linear_obj.pixel_signals_from(signal_scale)`
   needs the mapper's `adapt_data`. Every `AdaptImages` construction in the five scripts (the
   instance-keyed dict, the name-keyed dict, and the JAX re-creation) now also carries
   `adapt_image` — the lensed-source image the cell already builds through
   `_adapt_image_util.adapt_image_for_dataset`. This is done **unconditionally**, on both legs:
   `ConstantSplit` ignores the entry, so the two legs' tables are identical apart from the
   scheme. Verified: the ConstantSplit pins reproduce through the new wiring (below).
4. **Provenance in every result JSON** — a top-level `regularization` key next to `sibson`:
   `{"scheme": "adapt_split", "inner_coefficient": 0.1, "outer_coefficient": 10.0,
   "signal_scale": 0.1}` or `{"scheme": "constant_split", "coefficient": 1.0}`. **A row with no
   `regularization` key was measured with `ConstantSplit(1.0)`.**
5. **Per-scheme pins.** Breakdown cells key `EXPECTED_LOG_EVIDENCE_HST` by scheme; runtime
   cells nest the scheme under the instrument. Existing ConstantSplit values kept verbatim.

**Row naming.** The canonical A100 row names (`delaunay{,_nn}_hpc_a100_fp64.{json,png}`) now
mean **AdaptSplit**; the same-node ConstantSplit control is written with `--config-name
hpc_a100_fp64_constant_split`. No historical JSON was deleted or overwritten: the three
previous canonical rows were `git mv`-ed to
`delaunay_hpc_a100_fp64_constant_split_2026_09_05.{json,png}`,
`delaunay_nn_hpc_a100_fp64_constant_split_2026_09_05.{json,png}` and
`runtime/.../delaunay_nn_hpc_a100_fp64_constant_split_2026_09_05.json` (the date they carry in
git, from #219's session) before the new rows landed. Those 2026-09-05 rows pre-date
PyAutoArray #537 and are **not** on this note's scale.

## Verdict

At the batched production point (`vmap` 16) **the AdaptSplit assembly itself is free**: the
DelaunayNN params→H prefix reads **7.250 ms (ConstantSplit) vs 7.277 ms (AdaptSplit) per call**,
a 0.4 % difference, against the 7.26 ms same-cell reference from the #537 assembly run. The
PyAutoArray #537 compaction therefore survives the regularization change intact.

What AdaptSplit does cost is **downstream, in the NNLS reconstruction**, not in `H`:
`Regularized reconstruction` rises 30.88 → 39.20 ms unbatched on DelaunayNN and 32.27 → 37.10 ms
on Delaunay, and the whole likelihood rises **40.92 → 45.76 ms per call at `vmap` 16 (+11.8 %)**
on DelaunayNN and **39.68 → 42.47 ms (+7.0 %)** on Delaunay. The mechanism is the solver, not
the matrix: `reconstruction_positive_only_from` on the JAX path is
`jax_nnls.solve_nnls_primal`, a primal-dual interior-point `lax.while_loop` that runs to a
convergence test capped at `max_iter = 50` — and under `vmap` "runs until the slowest lane
converges" (the module's own docstring). AdaptSplit's per-pixel weights span two orders of
magnitude (`inner 0.1` … `outer 10.0`) where ConstantSplit's are flat, so `F + H` is worse
conditioned and the loop takes more trips. This is a **finding about the production
configuration**, not a regression introduced here: production already pays it, and the profiling
rows now show it.

**Both witness halves are met** (detail in "Witness" below):

| Witness | Target | Measured | |
|---|---|---|---|
| Every cell's AdaptSplit pin passes on its A100 leg | pass at rtol 1e-4 / 1e-3 | 4 A100 legs, all `PASSED` / `pinned_drift: []`, max drift 3.3e-10 relative | met |
| Every cell's ConstantSplit pin passes on its control leg | unchanged values | 4 A100 legs, all `PASSED` / `pinned_drift: []`, max drift 2.6e-11 relative | met |
| DelaunayNN AdaptSplit params→H prefix @ `vmap` 16 within ~1 ms of the same-node control | ≤ ~1 ms | **+0.027 ms** (7.277 vs 7.250; 7.26 earlier reference) | met |
| Result JSONs carry the `regularization` key | present on all 8 | present on all 8 | met |

One honest caveat, stated plainly: **the unbatched rows carry a systematic offset that the
batched rows do not.** On both cells the AdaptSplit leg's *unbatched* `Inversion setup
(steps 5–8 combined)` reads ~+4.9 ms — a block that contains no regularization work at all —
while the same block at `vmap` 16 is flat to 0.1 % (15.197 vs 15.176 on DelaunayNN; 13.822 vs
13.834 on Delaunay). The AdaptSplit leg was the *first* job of each pair (see Provenance), so a
first-job-on-node warm-up effect cannot be excluded, and no reversed-order repeat was run. Read
the `vmap` 16 columns as the citable numbers; treat unbatched deltas below ~5 ms as noise on
this session.

## Provenance

| Job | Cell | Leg | Node | Start (2026-09-08 BST) | Elapsed | State |
|---|---|---|---|---|---:|---|
| 342340 | `likelihood_breakdown/imaging/delaunay_nn` | `adapt_split` | `euclid-ral-gpu-2` | 16:12:26 | 2:51 | COMPLETED |
| 342341 | `likelihood_breakdown/imaging/delaunay_nn` | `constant_split` | `euclid-ral-gpu-2` | 16:15:17 | 2:21 | COMPLETED |
| 342342 | `likelihood_breakdown/imaging/delaunay` | `adapt_split` | `euclid-ral-gpu-2` | 16:17:39 | 1:34 | COMPLETED |
| 342343 | `likelihood_breakdown/imaging/delaunay` | `constant_split` | `euclid-ral-gpu-2` | 16:19:13 | 1:33 | COMPLETED |
| 342344 | `likelihood_runtime/imaging/delaunay_nn` | `adapt_split` | `euclid-ral-gpu-2` | 16:20:46 | 1:19 | COMPLETED |
| 342345 | `likelihood_runtime/imaging/delaunay_nn` | `constant_split` | `euclid-ral-gpu-2` | 16:22:07 | 1:19 | COMPLETED |
| 342346 | `likelihood_runtime/imaging/delaunay` | `adapt_split` | `euclid-ral-gpu-2` | 16:23:27 | 1:05 | COMPLETED |
| 342347 | `likelihood_runtime/imaging/delaunay` | `constant_split` | `euclid-ral-gpu-2` | 16:24:32 | 1:03 | COMPLETED |

- **All eight jobs ran back-to-back on one node inside one 13-minute window** (16:12:26 →
  16:25:35), serially (the `gpu` partition gave one GPU at a time), so every A/B pair is a
  same-node, same-session comparison.
- Breakdown flags `--regularization <scheme> --split-setup --vmap-batch 16`; runtime cells full
  timing only (no `--vmap-probe`), so both legs fall back to the `VMAP_BATCH` table row 16 — the
  recorded `vmap.batch_size` is 16 on all four runtime legs. fp64 (`JAX_ENABLE_X64=True`), dense
  inversion path, `backend = gpu`, `device = cuda:0`, NVIDIA A100 80GB PCIe on every leg.
  `PyAutoLens 2026.8.17.1`.
- `PYAUTO_SIBSON_QUERY_CHUNK` deliberately unset on every leg; the library default
  `query_chunk = 4096` is what all legs ran, recorded under `sibson` in each DelaunayNN JSON
  (caps `max_cavity_triangles = 32`, `max_neighbors = 32`).
- Recorded `xla_flags` identical on all eight legs:
  `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`.
  The third flag is **new since the 2026-09-08 14:xx #537 session**, whose rows carry only the
  first two. The A/B here is internally clean; cross-note comparisons against
  `delaunay_nn_constant_split_assembly.md` carry that caveat — although in practice this note's
  ConstantSplit control reproduces that note's feature leg to 0.1–0.2 % (below).

### One library revision on both legs — the shared install, untouched

| Repo | Revision |
|---|---|
| PyAutoArray | `47a00e8c9d356b6ba708182309e20c20041f3e80` (merge of #537, contains `04152bb7`) |
| PyAutoNerves | `0e7163bc2feb7513a993ed62bff7dbb83c8f5a01` |
| PyAutoFit | `68ff9bd57781342a6a5ef8321d6419c43ef62f64` |
| PyAutoGalaxy | `ec5ce75d3136bbbdc0e9c183f47044433216b30c` |
| PyAutoLens | `08a05858aa1ef9237c1346b7ccaafc789ed140f0` |
| autolens_profiling (`AP_ROOT`) | `f283c4cd4a00c4ed2cacee55404f7bf1723232ff` |

The shared tree moved between the #537 session (14:xx) and this one (16:xx) — PyAutoNerves
`efe7c04a` → `0e7163bc`, PyAutoFit `6331b800` → `68ff9bd5`, PyAutoGalaxy `6d216c15` → `ec5ce75d`,
PyAutoLens `9468e3e1` → `08a05858`; that is where the extra `--xla_gpu_enable_triton_gemm=false`
comes from. It is identical across *these* eight legs, so this A/B is clean; the caveat is only
for reading absolute rows across the two notes. All four A100 library revisions except PyAutoFit
also match the tree the AdaptSplit pins were taken on locally (local PyAutoFit `08207bad0`).

Unlike the #536/#537 A/B, **no `PYTHONPATH` override was needed**: the shared
`/mnt/ral/jnightin/PyAuto/PyAutoArray` install is already at `47a00e8c`, i.e. it *contains*
`04152bb7`, so both legs ran one revision that carries the split-stencil compaction. The shared
install was **read only** — checked with `git log` / `merge-base --is-ancestor`, never written,
never pulled, never checked out (the subhalo-validation array `342299` was live on it
throughout). Every job printed `autoarray.__file__` and the shared-tree `rev-parse HEAD`:
verified `/mnt/ral/jnightin/PyAuto/PyAutoArray/autoarray/__init__.py` and `47a00e8c…` on all
eight. Jobs were submitted from the detached task worktree
`/mnt/ral/jnightin/autolens_profiling_wt/delaunay-adapt-split-regularization` at `f283c4c`;
the canonical RAL checkout's working tree was not touched.

The HST dataset is tracked in git, so the worktree carried `data/psf/noise_map/tracer`
identically (md5-checked against the #537 worktree). The one non-tracked input, the cached
`dataset/imaging/hst/lensed_source.fits` adapt image, was **copied from the #537 worktree**
(md5 `b8fb722d…`) rather than regenerated: eight concurrent-capable jobs sharing one cache path
is a write race, and copying also means the AdaptSplit legs used the same adapt image as the
earlier A100 rows. It differs from the local CPU cache used for pinning at float-rounding level
only — the pins still reproduce to 5e-10 relative (below).

Every `error/*.err` was 222 bytes (the `nvidia-smi`/preflight banner), `grep -c Traceback` was 0
on every `.out` and `.err`, and `grep -c "truncated to dtype float32"` was 0 on all sixteen
files. All eight JSONs were written; the four breakdown legs also wrote their PNG (the runtime
cell has no per-step chart).

## Pins

Both breakdown cells print `Eager regression assertion PASSED` on their own leg; both runtime
cells record `pinned_drift: []`. The AdaptSplit pins were taken from each cell's **first eager
local CPU run** (2026-09-08, WSL, JAX fp64, PyAutoLens `08a05858a` / PyAutoNerves `0e7163b` /
PyAutoFit `08207bad0` / PyAutoArray `47a00e8c` / PyAutoGalaxy `ec5ce75d`) — the same way the
ConstantSplit pins were taken — and are verified here against the A100.

| Cell | Scheme | Pin (local CPU) | A100 eager | drift (rel) | rtol | outcome |
|---|---|---:|---:|---:|---:|---|
| `breakdown/delaunay` | `constant_split` | 29110.92085793 | 29110.920857157875 | 2.6e-11 | 1e-4 | PASSED (342343) |
| `breakdown/delaunay` | `adapt_split` | 29155.0010494252 | 29155.001059064576 | 3.3e-10 | 1e-4 | PASSED (342342) |
| `breakdown/delaunay_nn` | `constant_split` | 29144.581943885652 | 29144.581943564488 | 1.1e-11 | 1e-4 | PASSED (342341) |
| `breakdown/delaunay_nn` | `adapt_split` | 29348.90938612374 | 29348.909384723986 | 4.8e-11 | 1e-4 | PASSED (342340) |
| `runtime/delaunay` | `constant_split` | 29110.92085793 | — | `pinned_drift: []` | 1e-4 / 1e-3 | PASSED (342347) |
| `runtime/delaunay` | `adapt_split` | 29155.0010494252 | — | `pinned_drift: []` | 1e-4 / 1e-3 | PASSED (342346) |
| `runtime/delaunay_nn` | `constant_split` | 29144.581943885652 | — | `pinned_drift: []` | 1e-4 / 1e-3 | PASSED (342345) |
| `runtime/delaunay_nn` | `adapt_split` | 29348.90938612374 | — | `pinned_drift: []` | 1e-4 / 1e-3 | PASSED (342344) |
| `runtime/delaunay_numba` (hst) | `constant_split` | 29090.527192092646 | *no A100 leg* | — | 1e-6 | CPU-only cell |
| `runtime/delaunay_numba` (hst) | `adapt_split` | 29212.44050977029 | *no A100 leg* | — | 1e-6 | CPU-only cell |
| `runtime/delaunay_numba` (euclid) | `constant_split` | 7215.3687893658935 | *no A100 leg* | — | 1e-6 | CPU-only cell |
| `runtime/delaunay_numba` (euclid) | `adapt_split` | 5579.104036561161 | *no A100 leg* | — | 1e-6 | CPU-only cell |

Two things this table says that are worth stating explicitly:

- **The adapt-image wiring did not move the ConstantSplit numbers.** The DelaunayNN control leg
  reproduced `29144.581943564488` — the identical eager value the #537 legs printed, to the last
  digit — through code that now always builds `AdaptImages` with an adapt image. The Delaunay
  control agrees with its pre-2026-09-08 pin to 2.6e-11 relative. That is the check that the
  wiring is inert for `ConstantSplit`, which is what makes the historical rows comparable.
- **AdaptSplit moves the log evidence by a real, physical margin** — +44.1 nats on Delaunay,
  +204.3 on DelaunayNN — so the pins genuinely discriminate the two schemes (the degenerate
  `AdaptSplit()` default would have moved them by zero).

`delaunay_numba` is CPU/numba only and has no A100 leg by design; its two pins per instrument
were taken locally and are unverified on the A100. The euclid `adapt_split` pin (5579.10) sits
far below its `constant_split` counterpart (7215.37) — that is the point-mass/1250-vertex euclid
fiducial responding to the adaptive weights, and it is a *pin*, not a claim about which value is
better.

## DelaunayNN breakdown — ConstantSplit (342341) vs AdaptSplit (342340)

All values ms per likelihood call. `vmap/16` is `jax.jit(jax.vmap(fn))` at batch 16, reported as
batch time / 16; only the combined inversion-setup block, the `--split-setup` prefixes and the
params→H prefix are re-timed under `vmap`, so the other rows read `—`.

| Step | ConstantSplit | AdaptSplit | Δ | Const `vmap`/16 | Adapt `vmap`/16 |
|---|---:|---:|---:|---:|---:|
| Ray-trace data grid | 0.157 | 0.159 | +0.002 | — | — |
| Ray-trace mesh grid | 0.163 | 0.150 | −0.013 | — | — |
| Lens light images (pre-PSF) | 0.138 | 0.143 | +0.005 | — | — |
| Blurred image (PSF convolution) | 0.849 | 0.853 | +0.004 | — | — |
| Profile-subtracted image | 0.141 | 0.146 | +0.005 | — | — |
| Inversion setup (steps 5–8 combined) | 27.389 | 32.286 | +4.896 | **15.197** | **15.176** |
| Data vector (D) | 0.318 | 0.340 | +0.022 | — | — |
| Curvature matrix (F) | 4.847 | 4.794 | −0.054 | — | — |
| **Regularization matrix (H)** | **0.767** | **4.814** | **+4.047** | −1.608 | −1.575 |
| **Regularized reconstruction (NNLS)** | **30.883** | **39.200** | **+8.317** | — | — |
| Mapped recon + log evidence | 2.279 | 2.230 | −0.049 | — | — |
| **Total step-by-step (unbatched)** | **67.931** | **85.114** | **+17.183** | — | — |

The `vmap` `H` cell is negative on **both** legs (−1.61 / −1.58). That is the compile-boundary
attribution artifact documented at length in `delaunay_nn_constant_split_assembly.md` — the row
is a difference of two independently compiled prefixes and goes negative once the interval is
smaller than the materialisation slack. It is not read as a saving on either leg; the prefix row
below is what carries the assembly.

### Prefixes and the four-way setup split

| Piece | ConstantSplit | AdaptSplit | Δ | Const `vmap`/16 | Adapt `vmap`/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.036 | 1.469 | +0.433 | 0.066 | 0.065 |
| Triangulation + interpolation | 13.485 | 12.792 | −0.693 | 8.792 | 8.787 |
| Mapping matrix | −0.177 | 0.085 | +0.262 | −2.326 | −0.314 |
| Blurred mapping matrix (PSF) | 8.494 | 12.172 | +3.678 | 8.352 | 8.818 |
| *Interpolator prefix (params→step 6)* | *14.521* | *14.261* | *−0.260* | *8.858* | *8.852* |
| *Split-Sibson prefix (params→split mappings)* | *14.373* | *14.473* | *+0.101* | *8.909* | *6.409* |
| **params→H prefix (the witness)** | **15.289** | **19.076** | **+3.787** | **7.250** | **7.277 (+0.027)** |

| Piece | ConstantSplit | AdaptSplit | Const `vmap`/16 | Adapt `vmap`/16 |
|---|---:|---:|---:|---:|
| Split-point Sibson | −0.148 | 0.212 | 0.051 | −2.442 |
| H, split assembly (differenced cell) | 0.916 | 4.602 | −1.659 | 0.868 |

Read this as: **at `vmap` 16 the whole params→H interval is unchanged** (7.250 → 7.277, +0.4 %),
so computing the adapt weights costs nothing measurable once the work is batched. Unbatched it
costs ~+3.8 ms on the prefix and ~+4.0 ms on the `H` step — a launch-bound cost that amortises
away at batch 16. The differenced sub-cells (Split-point Sibson, the assembly cell) swap ±2.5 ms
between the two legs, which is the same prefix-boundary shuffling both notes warn about; their
sums are stable.

Two anchors against the #537 session twelve minutes' worth of node earlier that day: this run's
ConstantSplit control reads **7.250 ms params→H per call at `vmap` 16 against #537's feature-leg
7.260 (0.14 % apart)** and **30.883 ms reconstruction against 30.846 (0.12 %)**. The control leg
reproduces the predecessor closely enough that the two notes' rows are in practice on one scale,
the extra `--xla_gpu_enable_triton_gemm=false` flag notwithstanding.

## Delaunay breakdown — ConstantSplit (342343) vs AdaptSplit (342342)

| Step | ConstantSplit | AdaptSplit | Δ | Const `vmap`/16 | Adapt `vmap`/16 |
|---|---:|---:|---:|---:|---:|
| Ray-trace data grid | 0.200 | 0.160 | −0.040 | — | — |
| Ray-trace mesh grid | 0.146 | 0.174 | +0.028 | — | — |
| Lens light images (pre-PSF) | 0.144 | 0.156 | +0.013 | — | — |
| Blurred image (PSF convolution) | 0.845 | 1.212 | +0.367 | — | — |
| Profile-subtracted image | 0.144 | 0.130 | −0.014 | — | — |
| Inversion setup (steps 5–8 combined) | 20.503 | 25.355 | +4.853 | **13.822** | **13.834** |
| Data vector (D) | 0.337 | 0.340 | +0.003 | — | — |
| Curvature matrix (F) | 4.841 | 4.857 | +0.016 | — | — |
| **Regularization matrix (H)** | **0.083** | **0.116** | **+0.032** | 0.081 | 0.046 |
| **Regularized reconstruction (NNLS)** | **32.273** | **37.101** | **+4.828** | — | — |
| Mapped recon + log evidence | 2.250 | 2.255 | +0.005 | — | — |
| **Total step-by-step (unbatched)** | **61.765** | **71.855** | **+10.091** | — | — |

| Piece | ConstantSplit | AdaptSplit | Δ | Const `vmap`/16 | Adapt `vmap`/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.506 | 1.058 | −0.447 | 0.066 | 0.067 |
| Triangulation + interpolation | 5.716 | 6.039 | +0.323 | 4.990 | 7.453 |
| Mapping matrix | 0.067 | 0.209 | +0.143 | 0.168 | −2.349 |
| Blurred mapping matrix (PSF) | 8.458 | 8.472 | +0.014 | 8.317 | 10.623 |
| *Interpolator prefix (params→step 6)* | *7.222* | *7.097* | *−0.124* | *5.056* | *7.520* |
| params→H prefix | 7.305 | 7.213 | −0.092 | 5.137 | 7.566 |

On the barycentric Delaunay mesh (`K = 4`) the `H` step is 0.08–0.12 ms on both legs and the
**unbatched** params→H prefix is flat to 1.3 % — the adapt weights are simply too cheap to see
next to the Sibson-free interpolation. The `vmap` params→H prefix reads 5.137 → 7.566, but that
whole +2.43 ms sits in the *interpolator* prefix upstream of `H` (5.056 → 7.520), and it is a
boundary shuffle, not work: `Triangulation + interpolation` gains +2.46 while `Mapping matrix`
loses −2.52 on the same leg, and the combined `Inversion setup` block at `vmap` 16 is flat
(13.822 vs 13.834, 0.09 %). Nothing in the regularization path can move an interval that ends
before `reg_split_from` is called; this cell's honest statement is that **AdaptSplit's assembly
cost is below the noise on the Delaunay mesh**, and the four-way split's per-piece attribution
is not stable enough at this scale to say more.

## Whole-likelihood runtime cells

| Cell | | ConstantSplit | AdaptSplit | Δ |
|---|---|---:|---:|---:|
| `delaunay_nn` | Full pipeline (single JIT) | 71.349 | 74.323 | **+2.974 (+4.2 %)** |
| `delaunay_nn` | `vmap` 16, per call | 40.915 | 45.760 | **+4.845 (+11.8 %)** |
| `delaunay` | Full pipeline (single JIT) | 59.753 | 64.511 | **+4.758 (+8.0 %)** |
| `delaunay` | `vmap` 16, per call | 39.678 | 42.467 | **+2.789 (+7.0 %)** |

This is what a production evaluation pays for the regularization production actually uses. The
DelaunayNN ConstantSplit leg (40.915 ms per call at `vmap` 16) sits 0.14 % from the #537 feature
leg's 40.858, so the control is anchored; the AdaptSplit number, 45.76 ms per call, is the new
reference for this cell and is the row `delaunay_nn_hpc_a100_fp64.json` now carries.

The +3 to +5 ms does **not** come from building `H` — the batched params→H prefix is flat to
0.4 %. It comes from the NNLS interior-point loop converging more slowly against a
worse-conditioned `F + H`, which shows up in the breakdown as the +8.3 / +4.8 ms
`Regularized reconstruction` rows. Two consequences worth carrying forward:

- **The NNLS iteration count is now a regularization-dependent cost**, and it is the largest
  single row on both cells (39.2 ms of an 85.1 ms unbatched DelaunayNN step total). Any future
  work on `solve_nnls_primal` — preconditioning, `target_kappa`, warm starts — should be costed
  against the AdaptSplit rows, not the ConstantSplit ones, because production is on AdaptSplit.
- **This was invisible before this task**, because every Delaunay profiling row measured a
  regularization production does not use.

A caveat repeated once, deliberately: the AdaptSplit leg ran first in all four pairs, and the
unbatched-only offset on the regularization-free `Inversion setup` block (+4.9 ms on both cells,
flat under `vmap`) shows this session had run-order or warm-up structure at the several-ms
level. A reversed-order repeat was not run. The `vmap` 16 numbers are the ones to quote; the
NNLS story rests on the reconstruction row, which is +8.3 ms on DelaunayNN — large against that
noise — and is corroborated by the whole-likelihood `vmap` rows on both cells.

## Witness

- **(a) Every cell's AdaptSplit pin passes on its A100 leg, and every ConstantSplit pin passes on
  its control leg — met.** Eight of eight legs: `Eager regression assertion PASSED` on the four
  breakdown legs, `pinned_drift: []` on the four runtime legs, worst relative drift 3.3e-10
  against rtol 1e-4. Exception, stated: `delaunay_numba` is CPU-only and has no A100 leg, so its
  four pins (two instruments × two schemes) are local-only.
- **(b) DelaunayNN AdaptSplit params→H prefix at `vmap` 16 within ~1 ms/call of the same-node
  ConstantSplit control — met, by a wide margin.** 7.277 vs 7.250 ms (+0.027 ms, 0.4 %), against
  the 7.26 ms earlier reference in `delaunay_nn_hpc_a100_fp64_assembly.json`. The #537 split-
  stencil compaction carries over to `AdaptSplit` — expected, since `AdaptSplit` shares
  `pixel_splitted_regularization_matrix_from`, and now measured.
- **Result JSONs carry the `regularization` key — met** on all eight new rows.

Nothing in the witness was missed. What is *not* covered, and was never in it: no A100 leg for
the numba cell; no reversed-order repeat to separate the unbatched offset from a warm-up effect;
no per-iteration NNLS instrumentation to prove the conditioning mechanism directly (the
mechanism is inferred from the solver's `while_loop` structure plus where the cost lands).

## Artifacts

- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.{json,png}` — **AdaptSplit** (342340)
- `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_constant_split.{json,png}` (342341)
- `results/breakdown/imaging/delaunay_hpc_a100_fp64.{json,png}` — **AdaptSplit** (342342)
- `results/breakdown/imaging/delaunay_hpc_a100_fp64_constant_split.{json,png}` (342343)
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64.json` — **AdaptSplit** (342344)
- `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_constant_split.json` (342345)
- `results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64.json` — **AdaptSplit** (342346)
- `results/runtime/imaging/delaunay/delaunay_hpc_a100_fp64_constant_split.json` (342347)
- Preserved ConstantSplit predecessors (renamed, not overwritten):
  `results/breakdown/imaging/delaunay{,_nn}_hpc_a100_fp64_constant_split_2026_09_05.{json,png}`,
  `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64_constant_split_2026_09_05.json`
- `hpc/batch_gpu/submit_{breakdown,runtime}_imaging_delaunay{,_nn}_a100_hst_fp64_{adapt_split,constant_split}`
  (eight submits; none carries the retired `--exclude=euclid-ral-gpu-1` block, per #220)
- Related: `results/notes/delaunay_nn_constant_split_assembly.md` (#536/#537, the rows this
  note's control leg anchors against).
