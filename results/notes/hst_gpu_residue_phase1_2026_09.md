# HST GPU residue — phase 1: where the fused production likelihood actually spends its device time (2026-09)

> **Status correction — 2026-09-18:** Phase 2 is in flight as issue #273. The active task record reports A100 array 343376 submitted on 2026-09-17; that is a submission record, not a current job-status assertion or a completed verdict. The inspected branch contains RTX instrument proofs. Do not choose GPU batching/callback changes until the matched A100 results are harvested and judged. See the [current campaign summary](profiling_campaign_status_2026_09.md).

autolens_profiling issue [#268](https://github.com/PyAutoLabs/autolens_profiling/issues/268),
branch `feature/hst-gpu-residue-p1`. **Phase 1 of the `hst-gpu-non-solver-residue` epic**, and
the first measurement in it. The campaign map says ~21 of the 25.4 ms certified Delaunay A100
call is not the solver — but its largest term, "~13.9 ms mesh / mapper / weights / imaging /
blurring", is **attribution arithmetic across two cells**, not a measurement. Every JAX
decomposition this repo owns (`--split-setup`) is ~11 *separately compiled* prefix programs
whose rows move with fusion; PyAutoArray#531 once made one of them negative.

This phase measures the **fused production program** instead: one `jax.jit`, one process, one
XLA device timeline, every GPU kernel joined to the library source line that emitted it. Three
A100 legs and two RTX 2060 legs. The table sums to the wall.

**The 13.9 ms bucket is refuted.** Mesh, mapper and interpolation weights together are
**0.37 ms** on the A100 — 1.2 % of the call. What that bucket actually contained is the PSF
convolution of the mapping-matrix cube (7.12 ms), the `A.T A` GEMM (4.18 ms), and the device
sitting idle while the host triangulates (5.44 ms of the 8.05 ms idle row).

## The five results

1. **The single largest non-solver item is the device doing nothing.** `device_idle` is
   **8.05 ms (25.3 %)** of the production Delaunay call, and **5.44 ms of it is one gap**,
   sitting exactly between the two halves of `pure_callback.4` — the qhull host round-trip.
   Measured twice independently: as a device-timeline gap (5.44 ms) and as a host-plane span
   (`scipy_delaunay_tri_only`, 4.73 ms). The GPU has nothing to run for 17 % of the call.
2. **The PSF convolution of the mapping-matrix cube is 7.12 ms (22.3 %), and it is already
   compiled once, not twice.** `operated_mapping_matrix_list` is a plain property reached
   twice per evaluation, which made a doubled convolution a live candidate. The HLO census
   settles it: **three `fft` instructions, all sharing one caller chain** (one `rfft2` plus an
   `irfft2` that XLA splits in two). XLA's CSE collapsed the two accesses. There is no doubled
   work to remove — the cost is the cube itself.
3. **`curvature_reg_matrix_rebuilt_every_access` has no JAX/GPU cost to recover.**
   `abstract.py:371` carries exactly **one instruction, a `transpose`** — **no `add` survives**.
   XLA fuses `F + λH` into its consumers, and walking back from every consumer through
   pass-throughs and fusion boundaries reaches **one shared producer**, `cublas-lt-matmul.2` at
   `inversion_util.py:139`. The matrix is built once. That is the JAX-side verdict on
   `draft/bug/autoarray/curvature_reg_matrix_rebuilt_every_access.md`; the numba/CPU side is
   [#267](https://github.com/PyAutoLabs/autolens_profiling/issues/267)'s and is not settled here.
4. **The campaign map's 25.39 ms headline is a pass-budget-2 number.** At the **production**
   budget 7 the same call is **31.5 ms**. The two runs agree where they should: route **b**
   (library PDIP, budget-independent) is **49.67 ms** in the earlier leg against **50.56 ms**
   here — **1.8 %**, on the same cell and the same machine class. The 25.39 → 31.49 ms gap on
   route **d** is five extra active-set passes (~1.2 ms each), not a discrepancy. The
   non-solver residue is then **21.7 ms of 31.9 ms (68 %)** — the map's "~21 ms" was right, but
   the total it sits in is 31.9, not 25.4.
5. **The border-relocator premise in the issue was backwards.** It assumed the profiling cells
   force the relocator on and production ships it off. `autolens_profiling/config/general.yaml`
   declares no `inversion` block, so autoconf falls through to the packaged
   `PyAutoGalaxy/autogalaxy/config/general.yaml:16`, which is **`true`**. Every lensing
   workspace — `autolens_workspace`, `autolens_workspace_test`, `HowToLens` — and
   `euclid_strong_lens_modeling_pipeline/config/general.yaml:15` set it `true` as well. So
   relocator-on **is** production, and it costs **0.100 ms (0.3 %)**.

## Scope — read this before quoting a number

- **What was measured.** Five legs of
  `scripts/imaging/likelihood_breakdown/fixed_light_trace.py`: three on the A100 (SLURM array
  343350, one task each) and two on the RTX 2060. Each builds the S3 system (lens light fixed
  after SLaM `light[1]`, converted to regular profiles and subtracted), times route **b**
  (library PDIP) and route **d** (certified active set at the **production** budget 7 with PDIP
  fallback), then traces 10 steady calls of route **d**'s compiled executable. HST, fp64,
  N=1500, `adapt_split` regularization.
- **Every stage row is a measured sum of GPU kernel durations**, joined to library source
  through the optimized HLO's stack frame index. Nothing is a difference of two compilations.
- **The rows attribute the command-buffers-OFF program.** With XLA's CUDA graphs on, every
  kernel reports `hlo_op=command_buffer_N` and the join recovers only **35–76 %** of device
  duration on the real program (23 % on the spike's toy) instead of 100 %, so the traced executable is compiled with
  `xla_gpu_enable_command_buffer=""`. Both programs are compiled from **one lowering in one
  process** and both walls are recorded. The delta is **+0.4 % / +0.8 %** on the two A100
  Delaunay legs — but **+17.4 % on DelaunayNN**, and that leg's rows must not be read as
  production timings (see its caveat below).
- **The A100 stage rows carry ~9 % profiler inflation.** Reconciliation is computed against the
  **traced** wall, because that is the run the kernel durations came from. On a 32 ms A100 call
  the profiler costs 2.9 ms (34.80 traced vs 31.89 untraced, +9.1 %); on a 1000 ms RTX call it
  is under 1 %. To place an A100 stage row in the production call, scale by
  **31.64 / 34.80 = 0.909** — and treat that as an assumption of uniform overhead, not a
  measurement. The shares are unaffected.
- **`mixed_fusion` is a measurement-resolution limit, not a stage.** A kernel whose fused
  constituents span stages goes there with its constituent set recorded rather than being
  assigned to whichever constituent XLA happened to name.
  `mixed_fusion.prorated_estimate_ms` in the JSONs is an **equal split** and is labelled
  ESTIMATE everywhere; it is never part of the table and never reconciles anything.
- **The RTX 2060 numbers are session-scoped and must not be used to rank anything.** See the
  instability caveat below.
- **Not measured:** rectangular (one `--mesh` flag away, not run), Euclid, the sparse operator,
  JWST, batching, and any optimisation. Only Delaunay and DelaunayNN, only HST, only fp64.

## Provenance and gate

| Legs | Device | Config | Job | Exit | Elapsed |
|---|---|---|---|---|---|
| 1 delaunay, relocator **on** (production) | A100 80GB PCIe, RAL `gpu` | `hpc_a100_fp64_fixed_light_trace` | 343350_0 | **0:0** | 57 s |
| 1 delaunay, relocator **off** | A100 80GB PCIe, RAL `gpu` | `hpc_a100_fp64_fixed_light_trace` | 343350_1 | **0:0** | 56 s |
| 1 delaunay_nn, relocator **on** | A100 80GB PCIe, RAL `gpu` | `hpc_a100_fp64_fixed_light_trace` | 343350_2 | **0:0** | 1 m 30 s |
| 1 delaunay, relocator **on** | RTX 2060 Max-Q 6 GB, driver 580.97 | `local_rtx2060_fp64_fixed_light_trace` | — | 0 | ~5 m |
| 1 delaunay, relocator **off** | RTX 2060 Max-Q 6 GB, driver 580.97 | `local_rtx2060_fp64_fixed_light_trace` | — | 0 | ~5 m |

**Counts: legs run = 5; result JSONs written = 5; non-zero exit = none; OOM = none (A100 peak
1.18–1.19 GB, RTX peak 1.57 GB); wall-clock kill = none; pin FAILED = none; `unjoined` kernel
time = 0.000 ms on all five legs.**

Library revisions on every A100 leg (shared install `/mnt/ral/jnightin/PyAuto`, untouched):
PyAutoNerves `fac8b17b`, PyAutoFit `27d41e7c8`, PyAutoArray `5e2bc0f4`, PyAutoGalaxy
`840ffde0`, PyAutoLens `ccf9295f3`; `autoarray.__version__ = 2026.8.17.1`,
`autoarray.__file__ = /mnt/ral/jnightin/PyAuto/PyAutoArray/autoarray/__init__.py`.
Worktree HEAD `6b60d30`.

| Gate | Result |
|---|---|
| route **d** (certified @ budget 7) == route **b** (library PDIP), rtol 1e-9 | **PASS** on all five legs; A100 2.6e-11 / 3.2e-11 / 2.7e-11, RTX 5.6e-11 / 1.3e-11 |
| injected solver actually reached on the JAX path | asserted per leg — the cell **raises** if the count is zero; non-zero on all five |
| reconciliation: kernel rows + measured `device_idle` vs the traced wall | **−2.57 / −2.47 / −2.01 %** (A100), **−0.10 / −0.22 %** (RTX) — all inside the 5 % witness |
| `unjoined` (a kernel whose `hlo_op` is not an instruction of this module) | **0.000 ms on every leg** — the join is complete, which is what the 5 % is really testing |
| `other` (joined but unmatched by every stage rule) | 0.082 / 0.081 ms on the A100 Delaunay legs (bare XLA `copy`s carrying no source metadata); **1.434 ms on DelaunayNN** — see its caveat |
| `AUTOTUNE_ENTRIES count` | **0** on all three A100 legs — kernels are the library default, not a seeded autotune cache |
| tie back to the previous epic (route **b**, budget-independent) | **49.67 ms** there against **50.56 ms** here, Delaunay A100 N=1500 — **1.8 %** |
| tie back to the previous epic (DelaunayNN, route **b**) | **55.37 ms** there against **59.06 ms** here — 6.7 % |
| `.err` contents | one benign `mask_2d_util.py:564` padding warning per leg; nothing else |

## The A100 table — PRIMARY, this is what phase 2 ranks from

N=1500, HST, fp64, pass budget **7**, 10 traced calls. Percentages are of the untraced wall.

### Task 0 — Delaunay, border relocator **on**: the production configuration

Untraced wall **31.89 ms**; command buffers **ON 31.64 ms** / OFF 31.78 ms (**+0.42 %**);
traced wall 34.80 ms; `sum(kernels) + device_idle` = 33.90 ms (**−2.57 %**).

| stage | ms | % of call | what it is |
|---|---:|---:|---|
| `certified_active_set_solve` | **10.215** | 32.0 | the solver — `cholesky.70` 6.25 ms + two `triangular-solve` 2.08 ms (harness `active_set_steps.py:418-419`) |
| `device_idle` | **8.054** | 25.3 | gaps on the stream; **5.44 ms is one gap at `pure_callback.4`** |
| `psf_convolution_mapping_matrix` | **7.116** | 22.3 | `input_scatter_fusion` 1.99 + `fft.19` 1.76 + `fft.15` 1.37 + `fft.17`, on a (1500,180,180) cube |
| `curvature_matrix_F` | **4.183** | 13.1 | one `cublas-lt-matmul` (`A.T A`), `inversion_util.py:139` |
| `mixed_fusion` | 1.948 | 6.1 | top set `curvature_matrix_F + psf_convolution_mapping_matrix` (1.02 ms) |
| `log_det_regularization` | 0.893 | 2.8 | cuSOLVER `cholesky.120`, reached from `abstract.py:941` |
| `log_det_curvature_reg` | 0.890 | 2.8 | cuSOLVER `cholesky.110`, reached from `abstract.py:894` |
| `mesh_seed_argmin` | 0.224 | 0.7 | `delaunay.py:191` nearest-vertex seed |
| `mesh_locate_walk` | 0.131 | 0.4 | the visibility walk `lax.while_loop` |
| `border_relocator` | 0.100 | 0.3 | |
| `other` | 0.082 | 0.3 | XLA-inserted `copy`s with no source metadata |
| `memset`, `inversion_other`, `regularization_H`, `mesh_qhull_callback` (device side) | 0.064 | 0.2 | |

Host plane: `pure_callback` span **5.146 ms**, of which `scipy_delaunay_tri_only` (qhull)
**4.732 ms**. These are **host** milliseconds and are not a row of the device table — the
device pays for them as the 5.44 ms idle gap above.

### Task 1 — Delaunay, border relocator **off**

Untraced wall **31.58 ms**; command buffers ON 31.29 / OFF 31.53 (**+0.75 %**); reconciliation
**−2.47 %**. Identical to task 0 within noise except the missing `border_relocator` row:
solve 10.219, idle 7.096, convolution 7.087, F 4.209, mixed 1.924, log-dets 0.898 + 0.895,
seed 0.223, walk 0.130, other 0.081.

**The relocator costs 0.100 ms**, and the 0.31 ms difference in the two walls is that plus
run-to-run noise. It is not a lever.

### Task 2 — DelaunayNN, border relocator on — **caveated, do not rank from it**

Untraced wall **47.07 ms**; command buffers **ON 39.70 / OFF 46.62 — +17.4 %**.

| stage | ms | % |
|---|---:|---:|
| `device_idle` | **22.151** | 47.1 |
| `certified_active_set_solve` | 10.260 | 21.8 |
| `psf_convolution_mapping_matrix` | 7.109 | 15.1 |
| `mesh_construction` | **6.947** | 14.8 |
| `curvature_matrix_F` | 4.196 | 8.9 |
| `mixed_fusion` | 2.341 | 5.0 |
| `other` | **1.434** | 3.0 |
| `regularization_H` | 0.660 | 1.4 |
| log-dets, seed, walk, relocator, memset | 2.254 | 4.8 |

Two reasons this leg's *idle* row is not a production number. The command-buffer delta is
**+17.4 %**, so ~6.9 ms of launch overhead that the CUDA graphs remove is sitting in that
22.15 ms; and the traced-vs-untraced gap is +21.97 %. The **kernel** rows are still sound —
`mesh_construction` **6.95 ms** is real, and it is Sibson: `sibson.py:319-320` scatter/gather
fusions over (4096, 32) neighbour tables. The 1.43 ms `other` row is three bare XLA `copy`s of
those same `s32[4096,32]` / `s32[4096]` tables, carrying no source metadata — so the DelaunayNN
mesh construction is really ~8.4 ms once its copies are counted. `regularization_H` is 0.660 ms
here against 0.011 ms on Delaunay (`regularization_util.py:432`, the split assembly).

**DelaunayNN costs 39.70 ms against Delaunay's 31.64 ms** (command buffers on, production
configuration for both), and essentially all of the difference is mesh construction plus its
copies.

## The RTX 2060 table — and why nothing may be ranked from it

Delaunay, N=1500, HST, fp64, budget 7, relocator on: untraced wall **622.24 ms**,
reconciliation **−0.10 %**, `unjoined` 0.000 ms.

| stage | ms | % |
|---|---:|---:|
| `curvature_matrix_F` | 374.58 | 60.2 |
| `psf_convolution_mapping_matrix` | 115.44 | 18.6 |
| `certified_active_set_solve` | 81.40 | 13.1 |
| `device_idle` | 14.06 | 2.3 |
| `mixed_fusion` | 9.43 | 1.5 |
| `log_det_curvature_reg` / `log_det_regularization` | 9.08 / 9.03 | 2.9 |
| `mesh_seed_argmin`, `mesh_locate_walk`, `border_relocator`, rest | 5.06 | 0.8 |

Relocator off: wall 672.96 ms, reconciliation −0.22 %, same shape without the
`border_relocator` row.

> **⚠ These numbers are session-scoped.** The *same* leg, same code, same flags, measured
> **1063 ms** in one session and **622 ms** in another two hours later, with the second value
> reproducing to ~1 % on an immediate repeat (F 374.6 / 378.2 ms, convolution 115.4 / 116.0 ms,
> solve 81.4 / 81.9 ms). The two sessions do **not** differ by a constant factor — F was 47.5 %
> of the call in one and 60.2 % in the other, because the GEMM and the FFT throttle differently
> on a 65 W laptop part. **Absolute milliseconds and even stage shares from this GPU are
> session-scoped.** The RTX legs' job in this phase is to prove the instrument reconciles
> (−0.10 % with zero unjoined time); the A100 is what phase 2 ranks from.

## The HLO census

Taken from the optimized HLO of the same compiled executable the trace ran, on all five legs,
with identical results.

| Question | Answer | Evidence |
|---|---|---|
| How many `(n,n)` `add`s of `F + λH` survive XLA at `abstract.py:371`? | **None.** The line carries exactly one instruction, a `transpose`. | `opcodes_at_line_371 = {"transpose": 1}` |
| Then where is the matrix? | **One shared producer** for every consumer: `cublas-lt-matmul.2` at `inversion_util.py:139`. The three `[ids][:, ids]` gather paths and the log-det Cholesky all resolve to it. | `producers_shared_by_more_than_one_consumer = ["cublas-lt-matmul.2"]` |
| Is the PSF convolution of the mapping-matrix cube compiled twice? | **No — once.** Three `fft` instructions (one `rfft2` at `convolver.py:1222`, an `irfft2` at `:1226` that XLA splits in two), **all sharing one caller chain** through `operated_mapping_matrix_list`. | `fft_convolver_total.callers` |
| Gathers at `abstract.py:613` (edge subset) | **7** — the edge-zeroed branch *is* taken | |
| Gathers at `abstract.py:397` (`curvature_reg_matrix_reduced`) | **0** — the branch returns the matrix directly | |
| Cholesky factorizations | **6**: two in the harness active set (`active_set_steps.py:418`), one per log determinant (`abstract.py:894` and `:941`), two in the never-taken jaxnnls PDIP fallback branch (`jaxnnls/pdip.py:62,106`) | `cholesky.callers` |

The two log determinants are the same three lines of `_log_det_symmetric_from` and are told
apart **only by their caller**; that is why the stage map has a `requires` clause, and why
their 0.890 / 0.893 ms are separate rows rather than one.

### A trap worth recording

A first pass scanned the *whole* stack for `abstract.py:371` and reported **15 "F + λH adds"**.
`curvature_reg_matrix` is `self._xp.add(self.curvature_matrix, self.regularization_matrix)` on
**one line**, and Python evaluates both arguments *at that line* — so `:371` is in the stack of
339 instructions, including the Delaunay interpolation weights, the split regularization matrix
and the adaptive pixel signals. A census predicate must ask **where an instruction was
written** (innermost frame); only a *caller* predicate may scan the stack. Both helpers exist
in `xla_attribution.py` and a regression test fails on the old predicate.

## How the join works — the feasibility spike

1. **Command buffers must be off.** With XLA defaults every GPU kernel reports
   `hlo_op="command_buffer_1"`, because the module is captured into a CUDA graph. Measured on a
   toy jit: **23.0 %** of device duration joinable with graphs on, **100 %** with
   `xla_gpu_enable_command_buffer=""`. Kernel names do not rescue it — a cuBLAS GEMM is
   `void magma_sgemmEx_kernel<...>`, a cuSOLVER Cholesky is `volta_dgemm_64x64_lower_nt`.
2. **Pass the flag per compilation.** `Lowered.compile(compiler_options={...})` is accepted, so
   one lowering compiles twice and both walls are measured in the same process. This is what
   makes the on/off delta a measurement rather than a cross-run comparison.
3. **Source metadata is a stack, not a `source_file=`.** This JAX emits
   `metadata={op_name="..." stack_frame_id=N}` and puts the file/line table in
   `HloModuleProto.stack_frame_index` — **field 17** (`file_names`=1, `function_names`=2,
   `file_locations`=3 `{file_id,func_id,line,col}`, `stack_frames`=4 `{loc_id,parent}`, all
   1-based). Read with a ~70-line protobuf wire reader; no new dependency (`xprof`,
   `tensorboard_plugin_profile` and `perfetto` are not installed, and importing TensorFlow's
   `hlo_pb2` beside a live JAX GPU backend was rejected). This gives the **whole Python stack**,
   which is strictly more than `source_file=` ever gave and is what lets the two log
   determinants be separated.
4. **Fusions keep their constituents' metadata.** A fused kernel's entry instruction carries one
   representative `op_name`, and the `%fused_computation*` it `calls=` holds every constituent
   with its own `op_name` and `stack_frame_id`. A fusion is therefore attributed from the *set*
   of its constituents' stages — which is what makes `mixed_fusion` measurable rather than
   guessed.
5. **Host events nest.** `pure_callback.N` contains `_wrapped_callback` contains
   `pure_callback_impl` contains `scipy_delaunay_tri_only`. Summing every matching host event
   counted the same work five times. They are aggregated by name, and the result cross-checks
   against the device timeline: the largest device-idle gap sits exactly at `pure_callback.N`
   and is the same size as the host qhull call.

## The ranked lever list for phase 2

A100, Delaunay, production configuration (31.64 ms with command buffers on). Stage rows are
quoted as measured; multiply by **0.909** for the production-scaled value if a uniform-overhead
assumption is acceptable.

| # | Lever | A100 ms | % of call | Kind | Trap |
|---|---|---:|---:|---|---|
| 1 | **The qhull `pure_callback` host round-trip** | **5.44** | **17.2** | PyAutoArray | see below |
| 2 | **PSF convolution of the mapping-matrix cube** | **7.12** | **22.3** | harness experiment first, then PyAutoArray | see below |
| 3 | **A second Cholesky of a matrix just factorised** | **0.89** | **2.8** | harness experiment | see below |
| 4 | The `A.T A` GEMM (`curvature_matrix_F`) | 4.18 | 13.1 | not a lever on its own | near the fp64 dense floor |
| 5 | DelaunayNN mesh construction (that mesh only) | 6.95 + 1.43 copies | 17.8 of *its* call | PyAutoArray | mesh-specific; Delaunay is unaffected |
| — | Mesh / mapper / interpolation weights | **0.37** | **1.2** | **not a lever** | the 13.9 ms bucket was not this |
| — | Border relocator | 0.100 | 0.3 | not a lever | and it is production's default |
| — | `mixed_fusion` | 1.95 | 6.1 | not a lever | measurement-resolution limit |

**1 — the qhull callback (5.44 ms, 17.2 %).** The largest non-solver item in the call, and the
device is *idle* for all of it. `_jax_delaunay_tables` (`delaunay.py:139-171`) sends the mesh
points to the host, scipy triangulates, the tables come back. Two shapes a phase-2 prompt could
take: overlap the callback with independent device work (the lens-light image and the data-grid
ray trace do not depend on the triangulation), or triangulate on device. **Traps:**
`pure_callback` has no JVP rule and the module documents that as load-bearing, so any
replacement must keep the non-differentiability story intact; the walk's `lax.while_loop`
consumes the tables directly; and the callback is one host round-trip per evaluation regardless
of N, so this lever does **not** scale away at larger meshes — it gets relatively *worse*.

**2 — the PSF convolution (7.12 ms, 22.3 %).** Not doubled work (result 2 above settles that),
so the lever is the cube: `(1500, 180, 180)` fp64 is 389 MB per FFT, and the row is
`input_scatter_fusion` 1.99 + three `fft` 3.7 ms. A harness experiment should come first — does
a smaller `fft_shape`, a real-space convolution for a compact PSF, or complex64 for the
mapping-matrix path (the image path already takes it under `use_mixed_precision`) beat it? Only
promote to a PyAutoArray change if the harness says yes. **Trap:** `convolver.py` already picks
`next_fast_len`, and the mapping-matrix path deliberately keeps the kernel multiply in
complex128 because the downstream linear algebra needs it (`convolver.py:1195-1205`).

**3 — the re-factorised Cholesky (0.89 ms, 2.8 %).** `log_det_curvature_reg_matrix_term`
(`abstract.py:894`) runs a cuSOLVER Cholesky of `F + λH`; the reconstruction has already
factorised the same matrix moments earlier (here `cholesky.70` in the harness solver, 6.25 ms;
on the library path `Inversion.reconstruction`). Small in absolute terms but it is *pure*
duplication and the cheapest thing on this list to test. **Traps:** the library's docstring
already notes the log-det "uses the Cholesky decomposition which is already computed before
solving the reconstruction" — the trace shows that is not what the JAX path compiles; and the
PDIP fallback branch factorises too (`jaxnnls/pdip.py`), so a shared factor must not assume the
certified branch was taken.

**Settled, do not re-open.** Matrix-free CG / SLQ log-det is a **no-go**
([#247](https://github.com/PyAutoLabs/autolens_profiling/issues/247),
`results/notes/matrix_free_pixelized_2026_09.md`). `reg_adapt` cannot jit on the Delaunay
family. A non-uniform over-sample map **triples** jit compile time, so any lever that touches
the over-sampling must report compile time beside run time.

## What phase 1 did not measure

Rectangular (one `--mesh` flag away, deliberately not run), Euclid, the sparse operator, JWST,
batching, `--vmap`, and any optimisation at all. No PyAutoArray, PyAutoLens, PyAutoGalaxy or
PyAutoFit source was changed. The RTX legs did not include DelaunayNN.

## Artifacts

| File | |
|---|---|
| `results/breakdown/imaging/fixed_light_trace_delaunay_hpc_a100_fp64_fixed_light_trace.{json,png}` | A100 task 0 — the production configuration |
| `results/breakdown/imaging/fixed_light_trace_delaunay_border_off_hpc_a100_fp64_fixed_light_trace.{json,png}` | A100 task 1 |
| `results/breakdown/imaging/fixed_light_trace_delaunay_nn_hpc_a100_fp64_fixed_light_trace.{json,png}` | A100 task 2 |
| `results/breakdown/imaging/fixed_light_trace_delaunay_local_rtx2060_fp64_fixed_light_trace.{json,png}` | RTX, relocator on |
| `results/breakdown/imaging/fixed_light_trace_delaunay_border_off_local_rtx2060_fp64_fixed_light_trace.{json,png}` | RTX, relocator off |
| `scripts/misc/likelihood_breakdown/xla_attribution.py` | HLO index, stack frame index, trace parser, stage map, census |
| `scripts/misc/test/test_xla_attribution.py` | 49 tests, no GPU needed except one skipped end-to-end |
| `scripts/imaging/likelihood_breakdown/fixed_light_trace.py` | the cell |
| `hpc/batch_gpu/submit_breakdown_imaging_fixed_light_trace_delaunay_a100_hst_fp64` | the A100 array submit |
| `hpc/batch_gpu/{output,error}/*.343350_*` | SLURM logs for the three A100 tasks |

Each JSON carries `stage_map_provenance` — the rules and line ranges that produced its table —
and a `stage_audit` naming the instructions that built each row with their full source stack, so
a row can be checked against the HLO rather than trusted. The cell also dumps the optimized HLO
text and the module proto beside the trace, so a disputed census row can be re-derived without a
recompile.
