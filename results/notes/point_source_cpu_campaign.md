# Point-source CPU speed-up campaign — ledger

Issue: [autolens_profiling #297](https://github.com/PyAutoLabs/autolens_profiling/issues/297) (phase 1); [PyAutoArray #568](https://github.com/PyAutoLabs/PyAutoArray/issues/568) (phase 2)  
Branch: `feature/point-source-cpu-p1` (phase 1); `feature/point-source-cpu-p2` (phase 2, PyAutoArray + autolens_profiling); `feature/point-source-cpu-p3` (phase 3, PyAutoArray + PyAutoLens + autolens_profiling)  
Status: phase 1 **DONE** (RAL baseline, job 350580); phase 2 **DONE, ACCEPTED** (vertex-dedup A/B, RAL job 350582: 4.47× simple, 1.98× cluster, bit-identical; library change awaiting merge and release); phase 3 **DONE, ACCEPTED** (static step-0 lattice A/B, RAL jobs 350636 CPU / 350637 A100: CPU control → library 2.0–2.1× simple, 1.43× vmap-4, 5.2× cluster, compile +3–12 %, all 31 gates bit-identical; source-on-vertex tie case PASSED by human decision 2026-09-24, pinned as PyAutoLens `test__source_on_a_step_0_vertex_returns_the_two_true_images`); phase 4 not started  
Instrument: [`scripts/point_source_image/likelihood_breakdown/image_plane.py`](../../scripts/point_source_image/likelihood_breakdown/image_plane.py)
(shipped in #293, see [point_source_shared_likelihood_breakdown.md](point_source_shared_likelihood_breakdown.md))
and [`scripts/cluster/likelihood_breakdown/image_plane.py`](../../scripts/cluster/likelihood_breakdown/image_plane.py)

This campaign aims to make the JAX-CPU PointSolver image-plane likelihood cheaper
for both the galaxy-scale `simple` case and the 13-component, two-source cluster
case that cluster modelling depends on. It runs as one bounded phase at a time.
Phase 1 (first section) produces a quotable CPU baseline on a quiet
RAL node and records the unoptimized library revisions. Phase 2 tests the single
reported lever, the JAX-only throwaway `jnp.unique` vertex deduplication. Phases
3–4 are chosen only by what the measurements show. A September 17 study reported
large gains, but they are **hypotheses to reproduce**, not targets (see
[the reported evidence](point_source_cpu_2026_09_17_reported/README.md)).

## Frozen unoptimized baseline — the revisions the GPU campaign must reproduce

All four phase-1 rows (laptop and RAL, point source and cluster) record the same
library revisions. They are the **unoptimized** state. The A100 campaign
(Mind draft `point_source_image_plane_gpu_breakdown.md`) must take its first
baseline on exactly these revisions, either before phase 2 merges or afterwards
by checking these SHAs out:

| Package | Revision / version |
|---|---|
| PyAutoArray | `22e6d6082e3365b10fb08d96e19ab02a5420e935` |
| PyAutoLens | `2aaa1c1a8bb4a5923edf30a18451fcea330a6bd4` |
| PyAutoGalaxy | `70a61e26cd715cefe7012a38285698091246a8a4` |
| PyAutoFit | `a736840127be7a3c21e15a5c945e94d8f1082983` |
| PyAutoNerves | `1fa613aa8d89a79613009fb66f5b4e2d21deb57a` |
| autolens | `2026.8.17.1` |
| jax / jaxlib | `0.10.2` / `0.10.2` |
| autolens_profiling | `15ceb686ac93f9d5957910ed68f6ee33d7f81b60` (RAL rows); `3052443c…` (laptop point-source row, #293); `8cebf318…` (laptop cluster row) |

**Rule for every phase:** *solved* (`FitPositionsImagePairAllSolved` /
`…RepeatSolved`, analytic β\* source centre) and *plain* (`FitPositionsImagePairAll`
/ `…Repeat`, free source centre) are different likelihood variants. A difference
between them is a property of the likelihood variant, never of the hardware.
Hardware is compared only for the same variant.

---

## Phase 1 — RAL CPU fp64 baseline (2026-09-23) — DONE

### Environment

| | Laptop (`local_cpu_fp64`) | RAL (`hpc_ral_cpu_fp64`) |
|---|---|---|
| Host | `DESKTOP-H143S82`, WSL2, 8 logical cores (`device.cpu_count = 8`) | `euclid-ral-compute-10-2`, partition **`ral`** (CPU partition, not the gpu fallback) |
| CPU | Intel i9-10885H | Intel(R) Xeon(R) Platinum 8490H (`lscpu` in the job log) |
| CPUs | 8 | SLURM `--cpus-per-task=8`, `--mem=32gb`. `device.cpu_count = 236` in the JSON is the **node's** `os.cpu_count()`, not the cgroup allocation |
| Load | Not quiet. The point-source row (2026-09-20, #293) shared the host with a workspace smoke job. The run operator reported laptop loadavg ~2–3 during the cluster row (2026-09-23), but no loadavg is stamped in that JSON | Entry loadavg `0.00 0.00 0.00`; `0.69` at Leg A start, `1.84` at Leg B start, `1.06` at exit (1-min values, 236-CPU node) |
| Threads | `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `NPROC=8` | same |
| `XLA_FLAGS` | `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` | byte-identical (checked across all four JSONs) |
| Precision / backend | fp64 (`JAX_ENABLE_X64=1`), `JAX_PLATFORMS=cpu` | same |
| JAX compile cache | point source: none; cluster: `~/.cache/pyauto_jax` (persistent, state not controlled) | per-job directory `output/jax_cache/point_source_350580`, wiped and recreated by the submit, shared by both legs (which compile different programs) |
| Job | — | SLURM **350580**, `COMPLETED 0:0`, elapsed **00:07:06** |

**Wall clock per leg (RAL, from the job's `date` stamps).** Setup and import
provenance took 67 s (14:23:35 → 14:24:42 BST). Leg A (point source) took 161 s
(→ 14:27:23). Leg B (cluster) took 197 s (→ 14:30:40); about 52 s of that was an
automatic simulation of `dataset/cluster/simple`, which is gitignored and was
absent from the fresh RAL worktree. The regenerated dataset reproduces the
laptop likelihoods exactly (see parity). stderr held only the benign
`No blurring_image provided` convolver warning.

**Limits on what the environment block shows.**

- **`device.cache_fresh: true` is not evidence here.** `_profile_cli.py` sets it
  to `bool(cache_dir) and autotune_cache_entries_at_start == 0`, which counts
  GPU autotune entries. On CPU that count is always 0, so the flag says nothing.
  What does hold is that the submit wiped the per-job
  `JAX_COMPILATION_CACHE_DIR` before Leg A, so the RAL lower/compile seconds
  below are cold compiles. The laptop cluster row ran with the persistent
  `~/.cache/pyauto_jax`, so its compile seconds are not controlled.
- **CPU affinity was not verified.** SLURM allocated 8 CPUs. This run did not
  check whether XLA's CPU intra-op pool was sized to the 8-CPU cgroup/affinity
  mask or to the node's 236 CPUs. Phase 2 must record
  `len(os.sched_getaffinity(0))` and XLA's intra-op thread count.

### Baseline — steady per-call cost

Every number is computed from the four JSONs by a script, not typed by eye. The
figure is `steady_per_call_s`, the mean over `n_repeats` warmed calls:

| Row (steady ms / call) | n_repeats | laptop `local_cpu_fp64` | RAL `hpc_ral_cpu_fp64` | laptop / RAL |
|---|---:|---:|---:|---:|
| Point source, fused solved (`fused_full_solved_s`) | 5 | 62.69 | **24.69** | 2.54× |
| Point source, fused plain control (`fused_full_plain_control_s`) | 5 | 74.35 | **24.72** | 3.01× |
| Cluster, fused plain (`fused_full_plain_s`) | 3 | 218.93 | **128.05** | 1.71× |
| Cluster, fused solved (`fused_full_solved_s`) | 3 | 215.98 | **134.52** | 1.61× |
| Cluster, per-source solve `point_0` (z=1.0) | 3 | 126.51 | **110.29** | 1.15× |
| Cluster, per-source solve `point_1` (z=2.0) | 3 | 146.70 | **115.25** | 1.27× |

Compile phases of the fused rows:

| Fused row | host | lower s | compile s | first call s | steady ms |
|---|---|---:|---:|---:|---:|
| Point source, solved | laptop | 11.50 | 17.64 | 0.0620 | 62.69 |
| Point source, solved | RAL | 2.34 | 4.04 | 0.0276 | 24.69 |
| Point source, plain control | laptop | 6.48 | 13.29 | 0.1249 | 74.35 |
| Point source, plain control | RAL | 1.69 | 3.84 | 0.0254 | 24.72 |
| Cluster, plain | laptop | 43.40 | 43.32 | 0.2363 | 218.93 |
| Cluster, plain | RAL | 25.78 | 20.16 | 0.1292 | 128.05 |
| Cluster, solved | laptop | 76.81 | 69.45 | 0.2606 | 215.98 |
| Cluster, solved | RAL | 33.83 | 28.37 | 0.1305 | 134.52 |

**Measurement limitation carried forward:** these cells record only the mean of
the `n_repeats` steady calls. They record no per-call samples, median or
dispersion. The acceptance contract requires repeated, interleaved A/B medians
with dispersion, and a minimum detectable improvement derived from observed
noise. The phase-1 numbers are therefore a baseline level, not a noise model.
Phase 2's A/B must add that timing (see the handoff).

**Dashboard caution.** The README auto-table rows for `cluster/image_plane`
(4.81 s laptop / 4.53 s RAL) are `build_readme.py`'s existing
`total_step_by_step`: the sum of the plain step-by-step rows. Those are the
source-centre back-trace, the two standalone jitted solves and the two **eager**
plain per-system fit totals (steps 1–3), which the eager fits dominate at
≈2.1–2.3 s each. They are not the fused likelihood. The per-call cost a sampler pays is the fused
128–135 ms above. The `point_source/image_plane` dashboard rows (62.7 / 24.7 ms)
are the fused solved value.

### Parity evidence (quoted from the JSONs)

- **Point source, solved:** eager / JIT `7.743201200876817` / `7.743201200876812`
  on RAL, identical to the laptop row bit for bit. **Plain:** `7.196577317761017` /
  `7.196577317761015`, also identical across hosts. `vmap_batch = 2` returned
  `[7.743201200876812, 7.743201200876812]`. The gradient is finite,
  `gradient_l2 = 1783.302107230914` (laptop `1783.3021072305985`, relative
  difference 1.8e-13). `filtered_finite_positions = 4`.
- **Cluster:** the fused log-likelihoods are identical across hosts to every
  printed digit. Plain eager/JIT: `23.96507508339339` / `23.965075083393216`.
  Solved eager/JIT: `25.56012716672685` / `25.56012716672745`. The within-host
  eager−JIT gaps are 1.7e-13 (plain) and 6.0e-13 (solved). The solved paired
  model positions are equal element for element across hosts, with 3/3 finite
  rows per system (`point_0`, `point_1`). The RAL dataset was regenerated by the
  simulator (above), so this also shows the simulator is deterministic.

### Interpretation

- **The RAL node is ≈2.5× faster than the laptop on the simple cell and
  ≈1.6–1.7× faster on the cluster fused rows.** The laptop/RAL ratio is not one
  number. It is 2.54× (solved) versus 3.01× (plain) on the same simple cell, and
  1.15–1.27× on the standalone cluster solves. A clean CPU-speed difference would
  give a much more uniform ratio. The laptop rows were taken under load, so they
  are load-inflated. **Only the RAL rows are quotable baselines.**
- **On RAL the plain control and the solved likelihood cost the same** (24.72 vs
  24.69 ms, ratio 0.999). The laptop showed plain 19 % slower (74.35 vs 62.69 ms).
  That gap is laptop noise, not a property of either likelihood. On the cluster,
  RAL has solved 5 % above plain (134.52 vs 128.05 ms); at 3 repeats with no
  dispersion recorded, that is not established as a real difference.
- **The cluster solve is the dominant component, but the components do not add
  up to the fused value.** Each standalone per-source solve costs 110–115 ms on
  RAL, against a fused two-source likelihood of 128–135 ms. The two standalone
  solves sum to 225.5 ms, which is 1.76× the fused plain value. So one solve
  costs most of what the fused call costs, which is consistent with the
  September claim that the solve dominates. The solves cannot be subtracted from
  the fused row, though: independently jitted programs get different fusion, and
  the fused program shares work between systems. See the prefix-versus-fused
  caveat in
  [point_source_shared_likelihood_breakdown.md](point_source_shared_likelihood_breakdown.md#reading-the-rows).
  Only a device trace of the fused program can attribute its time.
- On RAL the point-source step-0 ray-trace prefix alone steadies at 22.48 ms of
  the 24.69 ms fused call, and every later prefix sits at 22–24 ms. This fits
  the September report that step 0 (which contains the `jnp.unique` sort) is
  nearly the whole call. It is a prefix reading, not proof: the September study
  had isolated the sort itself.
- **The September 17 within-run ratios remain hypotheses.** It reported 4.9×
  (40.0 → 8.1 ms) on the simple likelihood and 2.4× (117 → 48 ms / source) on the
  cluster solve from removing the throwaway `jnp.unique`. Those came from a
  monkeypatched scratch harness on a loaded laptop. They are **reported**
  ([folder](point_source_cpu_2026_09_17_reported/README.md),
  [note](point_source_cpu_2026_09_17_reported/pointsolver_cpu_research_note.md)),
  not reproduced. Phase 2 must reproduce them on RAL against the baseline above.
  Its steady cluster solves of 110–115 ms are close to the reported
  117–120 ms "current" figures, but those were laptop medians from another
  harness and are not directly comparable.

### Disposition of every CPU assessment recommendation (after phase 1)

- [ ] **Remove the throwaway `jnp.unique` vertex dedup (JAX path, `_plane_triangles`).**
  Unchanged; the reported 4.9× / 2.4× is still a hypothesis. **To be measured in phase 2.**
- [ ] **Numba / sparse rewrite.** Unchanged: not pursued by default. Reopen only
  if post-phase-2 measurements show a substantial unexplained bottleneck.
- [ ] **Step count (`n_steps`).** Unchanged; low priority, never a blind accuracy
  trade. **Phase 4(b)** (initial scale versus refinement count).
- [ ] **Solver grid extent.** Unchanged; needs image-completeness evidence across
  a prior, reported as a separate configuration. **Phase 4(a).**
- [ ] **Fit / chi-squared, β\*, magnification filter share.** Unchanged; re-rank
  after phase 2 and do not assume the reported 0.34 % still holds. **Phase 2 reprofile.**
- [ ] **Duplicate `model_data` solve (XLA CSE).** Unchanged; verify in the fused
  execution (HLO or trace) before proposing Python caching. **Phase 2 reprofile.**
- [ ] **Precompute static initial geometry.** Proposed, not measured. **Phase 3**, only if
  deflections dominate after phase 2.
- [ ] **`MAX_CONTAINING_SIZE` / neighbourhood fan-out.** Not measured; it is a
  correctness knob. **Phase 4(c).**
- [ ] **Cluster dPIE/NFW deflections.** Not measured; any PyAutoGalaxy work is a
  separately scoped phase. **Phase 4(d).**
- [ ] **Unmeasured controls.** vmap batches 1/4/16, post-fix grid/`n_steps`/capacity
  sweeps, the `constant_folding` flag A/B, and direct deflections on the
  276 507-point cluster input. **Phase 2 onward**, as each becomes the next hypothesis.
- [x] **Warm starts / cross-call mutable solver state.** Rejected. The likelihood
  stays pure for arbitrary sampler order, `vmap` and `custom_jvp`.
- [x] **Phase 1 itself.** The reported evidence is recovered and labelled, the
  solved and plain paths are measured separately on both cells, and the RAL
  baseline is published with the frozen revisions.

### Phase 2 handoff

- **The bounded change:** remove only the JAX-path throwaway `jnp.unique` in
  `CoordinateArrayTriangles._vertices_and_indices`, as consumed by
  `AbstractSolver._plane_triangles` (PyAutoArray + PyAutoLens). Do not remove all
  `unique` / `remove_duplicates` calls. The NumPy sibling path, `ShapeSolver`
  and plotting keep their deduplicated `ArrayTriangles`.
- **Red control first.** Show the defect on the original path, then require all of:
  - a bit-identical point-source likelihood (`7.743201200876812` JIT);
  - cluster solved positions equal element for element and fused likelihoods
    unchanged;
  - mask and index semantics unchanged;
  - eager/JIT/`vmap` parity;
  - `custom_jvp` gradient parity (`gradient_l2`).
  Correctness belongs in library tests / `autolens_workspace_test`, not in the
  timing cell.
- **Measurement protocol:**
  - RAL `ral` partition, 8 CPUs, a fresh per-job compile cache.
  - Record `len(os.sched_getaffinity(0))` and XLA's intra-op thread count.
  - Interleaved A/B in one process, with **distinct function objects** and
    `jax.clear_caches()` before each compile (JAX caches traces on function
    identity; the September study was fooled by this once).
  - Report **medians and dispersion** per arm and the minimum detectable
    improvement from the observed noise.
  - Keep the **fused production likelihood** as the control and report prefix
    residuals honestly.
  - Refresh both breakdown rows after the library change ships.
- **GPU ordering:** take the A100 baseline on the frozen revisions above
  **before** the phase-2 library change merges. If that is no longer possible,
  check the listed SHAs out explicitly. Phase 2 also owes a GPU regression check
  for the shared library change (done: RAL A100 job 350587, see phase 2, Decision).

### Reproduce

Laptop, from the autolens_profiling worktree, with the bundle's canonical-main
libraries on `PYTHONPATH` (see follow-ups):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NPROC=8
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib
python scripts/point_source_image/likelihood_breakdown/image_plane.py --config-name local_cpu_fp64
python scripts/cluster/likelihood_breakdown/image_plane.py --config-name local_cpu_fp64
```

RAL (the shared PyAuto install under `/mnt/ral/jnightin/PyAuto` must sit at the
frozen revisions; the job prints them):

```bash
ssh euclid_jump
cd /mnt/ral/jnightin/autolens_profiling && git fetch origin
git worktree add /mnt/ral/jnightin/autolens_profiling_wt/point-source-cpu-p1 feature/point-source-cpu-p1
cd /mnt/ral/jnightin/autolens_profiling_wt/point-source-cpu-p1/hpc/batch_cpu
sinfo -p ral,gpu -N -o "%N %t %C %O"      # ral must be idle; fallback: --partition=gpu, no --gres
sbatch submit_breakdown_point_source_image_plane_ral_cpu_fp64
```

The submit ([`hpc/batch_cpu/submit_breakdown_point_source_image_plane_ral_cpu_fp64`](../../hpc/batch_cpu/submit_breakdown_point_source_image_plane_ral_cpu_fp64))
runs both legs with `python3 -u … --config-name hpc_ral_cpu_fp64`. Job 350580
produced:

- [`results/breakdown/point_source_image/image_plane_hpc_ral_cpu_fp64.json`](../breakdown/point_source_image/image_plane_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/point_source_image/image_plane_hpc_ral_cpu_fp64.png)
- [`results/breakdown/cluster/image_plane_hpc_ral_cpu_fp64.json`](../breakdown/cluster/image_plane_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/cluster/image_plane_hpc_ral_cpu_fp64.png)
- the job log [`point_source_cpu_2026_09_23_ral_job_350580.out`](point_source_cpu_2026_09_23_ral_job_350580.out)

The auto-simulate step also wrote a `results/simulators/cluster_summary_v2026.8.17.1.*`
on RAL; it was not harvested. The laptop counterparts are
[`point_source_image/image_plane_local_cpu_fp64.json`](../breakdown/point_source_image/image_plane_local_cpu_fp64.json)
(#293) and [`cluster/image_plane_local_cpu_fp64.json`](../breakdown/cluster/image_plane_local_cpu_fp64.json).

### Follow-ups found in phase 1 (not fixed)

- **`check_submits.py` misses `python3 -u`.** In
  [`scripts/misc/wall/check_submits.py`](../../scripts/misc/wall/check_submits.py),
  the `_PYTHON_CALL` regex `python3?\s+(scripts/…\.py)` does not match
  `python3 -u scripts/…`. For every `-u` submit, `cells_run()` is empty, so the
  WALL-BASIS cell-coverage rule is **silently skipped**. That affects 87
  submit files under `hpc/`, including this campaign's.
- **Canonical `activate.sh` leaks a feature worktree.** The workspace-root
  `activate.sh` pointed `PYTHONPATH` at the `certified-positive-solver` library
  worktree, not the library mains; the phase-1 run overrode it by hand. A guard
  is recommended, one that refuses or warns when `activate.sh` resolves a library
  outside its canonical checkout. Without it, a baseline can silently measure an
  unmerged library branch.
- **No CI smoke coverage.** The point-source and cluster
  `likelihood_breakdown/image_plane.py` cells are not in the lint workflow's
  smoke step, so a breaking harness change is caught only on the next real run.

---

## Phase 2 — remove the throwaway vertex dedup (2026-09-23) — DONE, ACCEPTED

Library issue: [PyAutoArray #568](https://github.com/PyAutoLabs/PyAutoArray/issues/568).
Library branch: PyAutoArray `feature/point-source-cpu-p2` (on main `11b93476`), with
commits `273e152e` (the fix) and `25894d10` (the tests). Profiling branch:
`feature/point-source-cpu-p2`. Instrument:
[`scripts/point_source_image/likelihood_breakdown/vertex_dedup_ab.py`](../../scripts/point_source_image/likelihood_breakdown/vertex_dedup_ab.py).

### Mechanism (research note §3, now confirmed on current code)

On the JAX path, `PointSolver._plane_triangles` ray-traces `triangles.vertices`
and rebuilds the triangles with `with_vertices`. Both `vertices` and `indices`
came from `CoordinateArrayTriangles._vertices_and_indices`, which on main called
`jnp.unique(flat_triangles, axis=0, return_inverse=True, size=3N, equal_nan=True, fill_value=nan)`.

- Under `jit`, `jnp.unique` needs a static output shape, so it was given `size=3N`.
  That is exactly the input's own row count. The "deduplicated" table therefore
  had 3N rows: the real unique vertices plus NaN fill. On the `simple` step-0
  lattice that is 69 849 rows, of which 28 665 are unique (research note §3).
  The ray trace evaluated all 3N rows, so the dedup saved **zero** deflection
  evaluations.
- What it did cost was a lexicographic sort of 3N fp64 rows at every refinement
  step. The Python code reads the property twice per step (`vertices` and
  `indices`), but XLA CSEs the pair. The optimised HLO holds **one sort per step**:
  one `f64[69849]` sort at step 0 and seven `f64[720]` sorts for steps 1–7
  (240 kept triangles × 3). Read the optimised-HLO count, not the StableHLO
  count (3 → 1), which counts sort definitions inside reused private functions.
- **The fix** (`273e152e`) returns `(triangles.reshape(-1, 2), arange(3N).reshape(-1, 3))`:
  the flat vertex table and an identity index map, with no `jnp.unique`. It is
  **API-neutral**. `vertices`, `indices` and `with_vertices` keep their signatures
  and shapes, since the old table already had 3N rows. `with_vertices(vertices)`
  rebuilds the same triangles, and `_plane_triangles` in PyAutoLens is the only
  solve-path consumer and needs no change.
- **NaN padding.** Padding triangles are NaN in the flat table exactly as before.
  They trace to NaN, and every `Shape.mask` comparison against NaN is false, so
  they are never kept. The kept-triangle set, and hence the likelihood, cannot
  change.
- **The NumPy sibling is unchanged.** `coordinate_array_np.py` calls `np.unique`
  with no `size=`, so it genuinely shrinks the table (69 849 → 28 665 rows) and
  saves deflections. `ShapeSolver`, plotting and every `remove_duplicates` /
  `neighborhood` sort are untouched.

### Environment

| | Laptop witness (`local_cpu_fp64`) | RAL (`hpc_ral_cpu_fp64`) |
|---|---|---|
| Host | `DESKTOP-H143S82`, WSL2 | `euclid-ral-compute-1`, partition **`ral`**, SLURM job **350582**, `COMPLETED`, elapsed 00:11:12 (cell `wall_s` 628 s) |
| CPU | Intel i9-10885H | **AMD EPYC 7763 64-Core** (`lscpu` in the job log). Phase 1 ran on `euclid-ral-compute-10-2`, a Xeon Platinum 8490H: **a different CPU** |
| CPUs | 8 logical; `sched_getaffinity` = 8 | `--cpus-per-task=8`; `sched_getaffinity` = **8**; `os.cpu_count()` = 252 is the node |
| Load (1/5/15-min) | start 3.82 / 3.15 / 2.86 → end 1.53 / 1.89 / 2.61; **not quiet** | job entry 0.00; cell start 0.49 / 0.13 / 0.04 → end 1.54 / 1.43 / 0.84 |
| Threads | `NPROC=8`, `OMP/MKL/OPENBLAS_NUM_THREADS=1` | same |
| `XLA_FLAGS` | `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false` | byte-identical |
| Precision / backend | fp64, `JAX_PLATFORMS=cpu`, jax / jaxlib 0.10.2 | same |
| JAX compile cache | none (`jax_compilation_cache_dir: null`): every route compiles cold | per-job directory `vertex_dedup_ab_350582`, wiped by the submit. `control` compiles cold; `library` lowers to the control program and **hits the cache** (compile 0.4–1.9 s), so its compile seconds are not comparable |
| PyAutoArray | canonical main `11b93476` (pre-fix; see the PYTHONPATH note) | shared install `22e6d608` (the frozen phase-1 revision, pre-fix) |
| PyAutoLens / Galaxy / Fit / Nerves | `2aaa1c1a` / `70a61e26` / `a7368401` / `1fa613aa` | same |
| autolens_profiling | `04351847` (main; the cell was uncommitted, see the provenance defect) | `d97556f8` |
| Protocol | 3 routes (`control`, `nodedup`, `library`), **20 rounds × 5 calls** round-robin with the start rotated each round (100 per-call samples per route), 3 warm calls, 16-instance stream (seed 568; entry 0 = prior medians, the rest `U(0.25, 0.75)` unit vectors), bootstrap 2000 resamples (seed 12345, 90 %) | same |
| Peak RSS | 2.37 GB for the whole process (`/usr/bin/time -v`, all routes and both models), not per route | not recorded |

- **Cache-trap guard.** Each route gets a fresh closure and fresh solver and
  analysis objects, plus `jax.clear_caches()`. The argument is the physical
  parameter vector, with `instance_from_vector(xp=jnp)` inside the trace, so
  constant folding cannot fake work.
- **`library` self-labels.** It is the installed method, not an injected one.
  On both hosts it lowers to the control's sort count, so it reads
  `library_matches: control`.
- **XLA intra-op thread count.** The phase-1 handoff asked for it; it is still
  not recorded. `NPROC=8` and the 8-CPU affinity mask bound it, but the pool size
  itself was not read out.
- **Protocol deviation.** The #568 plan asked for ≥ 20 warmed calls per round;
  the cell ran 5. The 100 samples per route still give bootstrap CIs of ±2 % on
  the ratio, so this does not change the decision.
- **Provenance defect in the laptop JSON (not edited).** In the committed
  [`vertex_dedup_ab_local_cpu_fp64.json`](../breakdown/point_source_image/vertex_dedup_ab_local_cpu_fp64.json),
  the provenance strings `control_source.body` and `nodedup_body` hold the
  **wrong function texts**: `_profiling_root` and the control body respectively.
  The script was edited during the run, and `inspect.getsource` then read stale
  line offsets. The RAL JSON's bodies are correct and are the authoritative record
  of what each route ran. The laptop timings, HLO counts and gates are unaffected:
  the routes execute the function objects, not the recorded text.

### A/B — steady per-call cost

All numbers below come from the two JSONs via a script. The ratio is
control / nodedup of the per-route medians, with the JSON's bootstrap 90 % CI.
**MDI** (minimum detectable improvement) is the control's (p90 − p10) / median.
FLOPs are XLA `cost_analysis()` values.

**RAL `hpc_ral_cpu_fp64` (quotable)**

| Row | control median [p10, p90] ms | nodedup median [p10, p90] ms | library median ms | control / nodedup [90 % CI] | MDI | HLO sorts ctrl → nodedup | FLOPs ctrl → nodedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `simple_solved` | 24.00 [23.43, 24.59] | 5.38 [4.65, 6.25] | 23.92 | **4.47×** [4.37, 4.57] | 4.8 % | 15 → 7 | 15.58M → 7.05M |
| `simple_plain` | 23.76 [23.28, 24.21] | 5.39 [4.91, 6.25] | 23.74 | **4.40×** [4.30, 4.45] | 3.9 % | 15 → 7 | 15.58M → 7.05M |
| `simple_solved_vmap4` (per batch of 4) | 31.31 [29.69, 32.55] | 11.91 [10.77, 13.09] | 30.87 | **2.63×** [2.53, 2.72] | 9.2 % | 15 → 7 | 33.66M → 23.30M |
| `cluster_solved` | 155.22 [142.18, 162.34] | 78.32 [65.79, 87.82] | 154.50 | **1.98×** [1.93, 2.03] | 13.0 % | 25 → 12 | 184.16M → 139.44M |
| `cluster_plain` | 153.85 [141.55, 161.84] | 77.71 [66.12, 88.17] | 152.40 | **1.98×** [1.94, 2.02] | 13.2 % | 25 → 12 | 184.15M → 139.43M |

**Laptop `local_cpu_fp64` (witness, loaded host)**

| Row | control median [p10, p90] ms | nodedup median [p10, p90] ms | library median ms | control / nodedup [90 % CI] | MDI | HLO sorts ctrl → nodedup | FLOPs ctrl → nodedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `simple_solved` | 32.03 [30.68, 35.54] | 5.34 [5.00, 5.99] | 31.91 | **6.00×** [5.91, 6.21] | 15.2 % | 15 → 7 | 15.58M → 7.05M |
| `simple_plain` | 36.79 [34.40, 43.27] | 6.09 [5.61, 7.56] | 36.89 | **6.04×** [5.90, 6.23] | 24.1 % | 15 → 7 | 15.58M → 7.05M |
| `simple_solved_vmap4` (per batch of 4) | 51.92 [46.21, 67.19] | 15.31 [13.82, 32.32] | 50.19 | **3.39×** [3.20, 3.48] | 40.4 % | 15 → 7 | 33.66M → 23.30M |
| `cluster_solved` | 143.43 [133.24, 164.62] | 62.75 [57.44, 71.02] | 142.20 | **2.29×** [2.22, 2.33] | 21.9 % | 25 → 12 | 184.16M → 139.44M |
| `cluster_plain` | 143.75 [138.33, 154.88] | 62.39 [57.84, 69.60] | 145.07 | **2.30×** [2.27, 2.34] | 11.5 % | 25 → 12 | 184.15M → 139.43M |

**Side by side**

| Row | laptop ratio | RAL ratio | RAL paired per-round ratio median [p10, p90] |
|---|---:|---:|---:|
| `simple_solved` | 6.00× | 4.47× | 4.51 [4.17, 4.77] |
| `simple_plain` | 6.04× | 4.40× | 4.37 [4.13, 4.65] |
| `simple_solved_vmap4` | 3.39× | 2.63× | 2.65 [2.46, 2.87] |
| `cluster_solved` | 2.29× | 1.98× | 1.98 [1.86, 2.21] |
| `cluster_plain` | 2.30× | 1.98× | 1.97 [1.86, 2.18] |

The `control / library` ratio is 0.99–1.04 on every row of both hosts, with CIs
straddling 1 except RAL vmap4 at 1.014 [1.000, 1.024]. That confirms the
injected control reproduces the installed pre-fix method.

**Reading the rows.**

- **Sort shapes (RAL, identical on the laptop).**
  - `simple` control: 1 × `f64[69849]`, 7 × `f64[720]` and 7 × `f64[60]`.
    Nodedup keeps only the 7 × `f64[60]` (`neighborhood`).
  - `cluster` control: 1 × `f64[276507]`, 12 × `f64[720]` and 12 × `f64[60]`;
    nodedup keeps the 12 × `f64[60]`. The step-0 image-plane lattice is shared by
    both source systems, so XLA traces its sort once, not once per system.
  - The fix therefore removes the step-0 sort **and** the seven (twelve) `f64[720]`
    per-step sorts, all of them `_vertices_and_indices`. The research note §4
    credited the `f64[720]` sorts to `remove_duplicates`; that attribution does
    not survive this A/B, because only `_vertices_and_indices` differs between routes.
- **The gradient program halves its sorts too:** 30 → 15, and FLOPs 18.75M → 9.67M.
- **Throughput under `vmap`.** At batch 4 the control costs 7.83 ms per
  likelihood on RAL and nodedup 2.98 ms. Against the scalar calls (24.00 / 5.38 ms),
  the control already amortises most of its step-0 sort across the batch. This
  is consistent with the static step-0 lattice being batch-invariant, so the
  sort runs once per batch; that is an interpretation, not traced. The fix
  therefore gains less under `vmap` (2.63×) than per scalar call (4.47×), while
  still leaving the best throughput.
- **Ratios are comparable only within one host.** The RAL simple rows are
  faster than the laptop's, but the RAL cluster rows are **slower** (155 / 78 ms
  against 143 / 62 ms). The cluster program's larger deflection share evidently
  runs worse on 8 EPYC 7763 cores than on the i9; that is a host property, not a
  code one. The loaded laptop inflates the simple control more than nodedup, so
  its 6.0× overstates the gain, and **RAL's 4.47× / 1.98× are the quotable figures.**
- **Against the September hypotheses.**
  - Simple: reported 4.9× (40.0 → 8.1 ms); RAL measures **4.47×** (24.00 → 5.38 ms).
  - Cluster: reported 2.4× per standalone solve (117 → 48 ms); RAL measures
    **1.98×** on the fused two-source likelihood (155 → 78 ms). These are
    different quantities: fused likelihood here, standalone per-source solve then.
  - Direction and order of magnitude reproduce. The exact reported ratios were
    loaded-laptop, monkeypatched numbers and are superseded.

### Cross-check against the phase-1 RAL baseline

| Row | phase-1 fused (mean of `n_repeats`, one instance) | phase-2 control median | gap |
|---|---:|---:|---:|
| simple solved | 24.69 ms | 24.00 ms | **−2.8 %** |
| simple plain | 24.72 ms | 23.76 ms | **−3.9 %** |
| cluster solved | 134.52 ms | 155.22 ms | **+15.4 %** |
| cluster plain | 128.05 ms | 153.85 ms | **+20.1 %** |

- **Simple.** The simple control agrees with phase 1 within 4 %, on the same
  code (`22e6d608`) and the same fused likelihood. That holds even though phase 2
  medians over a 16-instance parameter stream and phase 1 averaged 5 calls at one
  instance. The simple call is dominated by the parameter-independent step-0 sort.
- **Cluster.** The cluster control is 15–20 % above phase 1. Comparing means
  gives +14 % / +20 %, so this is not a median-versus-mean artefact. The code is
  the same (the RAL control and library routes are the `22e6d608` install and
  agree within 1 %). The host is not: phase 1 ran on a Xeon Platinum 8490H node
  and phase 2 on an EPYC 7763 node. The cluster program is dominated by 13-component
  deflections, so the gap is read as a **host difference, not drift**. That fits
  the laptop running the cluster rows faster than this EPYC node. A second
  contributor is that the instance stream varies the lens parameters, while phase 1
  used one instance (the p10–p90 spread is 142–162 ms).
- **Consequence.** Phase-1 and phase-2 rows are one baseline only for the simple
  cell. The cluster before/after claim rests on the in-process A/B (1.98×), not
  on phase-1 128 ms → phase-2 78 ms. The phase-3 refresh must record its node's
  CPU model and compare against a control taken on the same node.

### Correctness gates

All 19 gates in `gate_summary` pass on both hosts.

- **Log-likelihood.** For every row and route pair (control / nodedup / library),
  the value is **bit-identical** (Δ = 0.0 absolute and relative, NaN pattern equal)
  on all 16 stream instances, and deterministic within each route. The fiducial
  simple solved value is `7.743201200876812` on all three routes. The stream
  values agree across hosts.
- **Positions.** Simple solved and cluster solved model positions are `array_equal`
  (max |Δ| = 0.0) on 3 gate instances.
- **Gradient (simple solved).** `jax.grad` is finite and **bit-identical** between
  routes on 3 instances (rtol 1e-10, atol 1e-12, achieved Δ = 0).
- **`vmap`.** vmap-4 equals the scalar calls bit for bit within every route, and
  all routes agree with each other.
- **Library tests and pins** (worker A, against the worktree PyAutoArray):
  - The new `test_autoarray/structures/triangles/test_coordinate_jax.py` has
    10 tests. Its HLO guard **fails on main** ("traced containment carries 39 sort
    op(s)") and passes on the branch. The full PyAutoArray suite passes (1626).
  - The autolens_workspace_test point-source `jax_likelihood` scripts ×4 print
    likelihoods **identical to main**, for example `-83.38049777774609` (JIT),
    `-82.33883111107943` and `-89.71129441664627`. `jax_grad/gradient.py` passes
    all checks.
  - The profiling `image_plane_solved` pin reads eager `7.743201200876817` / JIT
    `7.743201200876812`, identical to main; laptop JIT 41.2 → 7.0 ms.
- **The ulp-coincident-vertex caveat did not materialise** on any control. The
  plan flagged the risk that removing the dedup could change containment for
  vertices shared between triangles. It cannot here, for three reasons:
  1. `jnp.unique` merged only bit-equal rows, and the flat table's duplicates of a
     vertex are bit-equal. The deflection is a deterministic elementwise function,
     so each copy traces to the same bits the merged row did.
  2. Rows that were only ulp-close were never merged in the first place.
  3. `Point.mask` is inclusive (`0 <= α, β, γ <= 1`), so a source exactly on a
     shared edge or vertex is kept by every triangle touching it, with or without
     dedup.

  The bit-identical positions and likelihoods on every instance are the direct
  evidence.

### Decision — ACCEPT

The change is **repeatable and material, with no correctness or compile/memory
regression**.

- **Repeatable and material.** On the quiet RAL node, the simple likelihood
  falls 24.00 → 5.38 ms (**4.47×**, 90 % CI 4.37–4.57) and the two-source cluster
  likelihood falls 155 → 78 ms (**1.98×**, CI 1.93–2.03). The laptop witness
  reproduces the direction on a different CPU (6.0× / 2.3×). The observed
  reductions are 78 % (simple) and 50 % (cluster) of the call, against a minimum
  detectable improvement of 4.8 % and 13 % respectively. That is 16× and 4× the
  noise floor, and every paired per-round ratio's p10 is ≥ 1.86.
- **Compile.** Compile time falls on every row because the programs are smaller
  (HLO 28 583 → 24 404 lines on simple, 70 897 → 62 911 on cluster). On RAL,
  simple solved goes 4.08 → 3.24 s (−20 %) and cluster solved 31.45 → 27.89 s
  (−11 %). Lowering is 2.89 → 2.56 s and 43.1 → 42.0 s; first call is
  28.1 → 5.4 ms and 143 → 77 ms.
- **Memory.** Per-route peak memory is not recorded by this cell. The September
  note measured XLA temp memory 5.20 → 4.58 MB (−12 %). Removing a 3N-row sort
  and its gather cannot add live buffers, so no regression is expected; this is
  not measured here.
- **GPU — no regression; nodedup is 1.7–2.05× faster on an A100 (RAL job
  350587, 2026-09-24).** Same cell and routes, fp64, **20 rounds × 20 calls**
  (400 per-call samples per route, the #568 plan's protocol), one
  NVIDIA A100 80GB PCIe on `euclid-ral-gpu-1`, `JAX_PLATFORMS=cuda`, with the
  backend asserted `gpu` in the log. The imported PyAutoArray was a clone of
  **`feature/point-source-cpu-p2` at `25894d10`** that the submit prepends to
  `PYTHONPATH`. So `library` is the shipped fix: it self-labels
  `library_matches: nodedup`, and its optimised HLO is hash-identical to
  nodedup's on every row. There is no mixed-precision leg, because the
  PointSolver has no mixed-precision switch.

  | Row | control median [p10, p90] ms | nodedup median [p10, p90] ms | library median ms | control / nodedup [90 % CI] | paired per-round ratio median [p10, p90] | compile s ctrl → nodedup |
  |---|---:|---:|---:|---:|---:|---:|
  | `simple_solved` | 1.695 [1.672, 1.733] | 0.843 [0.825, 0.882] | 0.810 | **2.01×** [2.00, 2.02] | 2.01 [1.97, 2.04] | 9.7 → 5.6 |
  | `simple_plain` | 1.684 [1.663, 1.736] | 0.822 [0.805, 0.854] | 0.799 | **2.05×** [2.05, 2.06] | 2.06 [2.01, 2.09] | 9.4 → 5.3 |
  | `simple_solved_vmap4` (per batch of 4) | 1.865 [1.830, 1.951] | 0.958 [0.935, 1.000] | 0.926 | **1.95×** [1.94, 1.95] | 1.95 [1.93, 1.97] | 10.8 → 6.4 |
  | `cluster_solved` | 4.206 [4.166, 4.373] | 2.457 [2.411, 2.661] | 2.364 | **1.71×** [1.71, 1.72] | 1.72 [1.69, 1.73] | 52.7 → 46.5 |
  | `cluster_plain` | 3.860 [3.800, 4.062] | 2.105 [2.076, 2.169] | 2.039 | **1.83×** [1.83, 1.84] | 1.83 [1.82, 1.86] | 43.7 → 37.6 |

  The **correctness gates are all bit-identical on GPU**, with max |Δ| = 0 for every pair:
  log L on all five rows (control vs nodedup and control vs library), solved
  positions (simple 4/4 finite, cluster 3+3, three instances each), the
  simple-solved `jax.grad` (bit-identical, all finite, every component non-zero,
  |g| from 2.80 to 2532) and vmap-4 against scalar within each route. Every route
  is deterministic across repeated instances. Lowering is unchanged (for example
  cluster solved 45.1 → 45.5 s). First call is 23 → 11 ms (simple solved) and
  42 → 27 ms (cluster solved). `library`'s compile (0.5–3.6 s) is a cache hit on
  nodedup's program and is not comparable. Wall time was 696 s.

  One oddity: `library` runs **3–4 % faster than nodedup** on every row (the
  library/nodedup CI is [0.956, 0.976]), even though the two optimised HLOs are
  hash-identical. That is an effect of the second executable instance (its compile
  was a cache hit and its buffers were placed differently), not a code difference.
  It sets a ~4 % floor on what this protocol can resolve on GPU. The
  control/nodedup effect is 20–25× that floor.

  JSON [`vertex_dedup_ab_hpc_ral_a100_fp64.json`](../breakdown/point_source_image/vertex_dedup_ab_hpc_ral_a100_fp64.json)
  and [`.png`](../breakdown/point_source_image/vertex_dedup_ab_hpc_ral_a100_fp64.png).
  Log [`point_source_cpu_2026_09_24_ral_job_350587_vertex_dedup_ab_a100.out`](point_source_cpu_2026_09_24_ral_job_350587_vertex_dedup_ab_a100.out),
  with an empty stderr. Submit
  [`hpc/batch_gpu/submit_breakdown_point_source_vertex_dedup_ab_a100_fp64`](../../hpc/batch_gpu/submit_breakdown_point_source_vertex_dedup_ab_a100_fp64).
  This JSON's `control_source.body` and `nodedup_body` are correct. The cell now
  captures them at import, with a bytecode hash; see the follow-up below.

### Post-fix budget and phase-3 handoff

With the sort gone, the simple likelihood is **deflection-dominated**.

- **FLOP shares (hardware-independent, same code as the research note §4).**
  The bare deflections of one solve cost 2 916 501 (69 849 points) + 7 × 289 599
  (720 points) = 4.94M FLOP. That is **70 %** of nodedup's 7.05M. The remaining
  ~2.1M FLOP is containment and tiling, the seven `f64[60]` `neighborhood` sorts,
  β\*, the magnification filter and χ².
- **Time.** The research note's wall-clock deflection bound (≈ 6 ms, laptop,
  loaded) exceeds the laptop nodedup call itself (5.34 ms). It is an inflated
  upper value and is **not re-measured on RAL**. So the only budget claim made
  here is the FLOP share.
- **Cluster.** The nodedup program still carries 139.4M FLOP. Its direct
  deflections on the 276 507-point step-0 input remain an unmeasured control.
- **Duplicate `model_data` solve: XLA CSE confirmed on current code.** The
  positions-only program (one `model_data` solve) is 15 529 212 FLOP, and the
  full likelihood is 15 582 802. They differ by 53 590 FLOP (0.34 %), and the
  gap is the same 53 590 on nodedup (6 999 844 vs 7 053 434). No second solve is
  compiled, so Python caching would buy nothing at runtime.

**Ranked phase-3/4 levers (proposed, none measured):**

1. **Phase 3 — precompute the static initial lattice.** Build the step-0 unique
   vertices and index map in NumPy at construction and pass them as immutable JAX
   inputs. The step-0 trace would drop from 69 849 to 28 665 points on `simple`,
   a deflection saving of about 1.56M FLOP (≈ 22 % of the nodedup call). It
   scales with the cluster's 276 507-point input, where deflections dominate even
   more. This is the natural successor now that the dynamic sort is gone.
   - Report setup cost, memory and cache invalidation on geometry change.
   - Never cache model-dependent deflections.
   - The `constant_folding` flag A/B belongs here, because the step-0 lattice is
     a compile-time constant.
2. **Phase 4(a) — grid-extent guidance.** The reported 100×100 → 30×30 extent cut
   was 6.2× on the pre-fix code. After the fix it can act only on the 70 %
   deflection share plus containment. It needs image-completeness evidence
   across a prior.
3. **Phase 4(b) — initial scale versus steps.** Step 0 dominates the deflection
   FLOPs (2.92M of 4.94M), and each extra step costs about 0.29M.
4. **Phase 4(c) — `MAX_CONTAINING_SIZE` / neighbourhood fan-out.** This acts only
   on the 720-point steps and the `f64[60]` sorts, the smallest remaining share.
   It is a correctness knob.
5. **Phase 4(d) — cluster dPIE/NFW deflections.** This is the likely residue at
   cluster scale. It is a separately scoped PyAutoGalaxy phase.

**The first phase-3 step:** once PyAutoArray `feature/point-source-cpu-p2` is
merged **and released**, and RAL's shared install is synced (`HPCPullPyAuto`):

1. Re-run the phase-1 breakdown cells (`point_source` and `cluster`
   `likelihood_breakdown/image_plane.py`) on RAL under a new label, so the README
   dashboard rows move. Record the node's CPU model; see the cross-check.
2. Re-run this A/B cell. Its `library` route must then read
   `library_matches: nodedup` (sort count 7 / 12).

Only then measure the lattice precompute against that refreshed control.

### Disposition changes (phase 2)

The phase-1 checklist above is kept as written. These items change:

- [x] **Remove the throwaway `jnp.unique` vertex dedup.** **Shipped (PyAutoArray
  #568 branch) and measured.** RAL 4.47× simple / 1.98× cluster, bit-identical.
  Awaiting merge and release, then the dashboard refresh (phase-3 first step).
- [x] **Duplicate `model_data` solve (XLA CSE).** **Verified in the fused program.**
  Full-likelihood minus positions-only FLOPs is 53 590 (0.34 %) on both routes,
  so no second solve is compiled. Python caching is not a performance fix; closed.
- [ ] **Fit / chi-squared, β\*, magnification filter share.** Re-ranked by FLOPs:
  together with containment and the `neighborhood` sorts, they are the ≈ 30 %
  non-deflection share of the post-fix call. They are not separated further, and
  no lever is proposed.
- [ ] **Precompute static initial geometry.** Now **phase 3's lever**, because
  deflections dominate (70 % of FLOPs). Proposed, not measured.
- [ ] **Step count, solver grid extent, `MAX_CONTAINING_SIZE`, cluster
  deflections.** Unchanged: phase 4(b), 4(a), 4(c) and 4(d).
- [ ] **Unmeasured controls.** The fixed controls are now measured: vmap batch 4
  (2.63×) and scalar. Still open: vmap batches 1/16, the post-fix
  grid/`n_steps`/capacity sweeps, the `constant_folding` flag A/B (moved to phase 3)
  and direct deflections on the 276 507-point cluster input.
- [ ] **Numba / sparse rewrite.** Unchanged: not pursued. No substantial
  unexplained bottleneck remains; the residue is deflections.
- [x] **Warm starts.** Unchanged: rejected.

### Reproduce

**Laptop witness.** Run from the profiling worktree. The bundle's `activate.sh`
puts the worktree PyAutoArray (the fix) on `PYTHONPATH`, so for a `library = control`
run, swap that entry for the canonical-main checkout:

```bash
source ../activate.sh
export PYTHONPATH=${PYTHONPATH/<bundle>\/PyAutoArray/<PyAutoLabs>\/array\/PyAutoArray}   # pre-fix library
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NPROC=8
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib
python3 -u scripts/point_source_image/likelihood_breakdown/vertex_dedup_ab.py --config-name local_cpu_fp64
```

The cell injects both `control` and `nodedup`. Which library is installed only
changes what the `library` route labels itself as.

**RAL.** The shared install stays on main; the cell injects both variants.

```bash
ssh euclid_jump
cd /mnt/ral/jnightin/autolens_profiling && git fetch origin
git worktree add /mnt/ral/jnightin/autolens_profiling_wt/point-source-cpu-p2 feature/point-source-cpu-p2
cd /mnt/ral/jnightin/autolens_profiling_wt/point-source-cpu-p2/hpc/batch_cpu
sinfo -p ral -N -o "%N %t %C %O"      # ral must be idle; fallback: --partition=gpu, no --gres
sbatch submit_breakdown_point_source_vertex_dedup_ab_ral_cpu_fp64
```

The submit is
[`hpc/batch_cpu/submit_breakdown_point_source_vertex_dedup_ab_ral_cpu_fp64`](../../hpc/batch_cpu/submit_breakdown_point_source_vertex_dedup_ab_ral_cpu_fp64).
It uses `ral`, 8 CPUs, a fresh per-job JAX cache, and prints provenance and
affinity. The absent `dataset/cluster/simple` is auto-simulated, as in phase 1.
Job 350582 produced:

- [`results/breakdown/point_source_image/vertex_dedup_ab_hpc_ral_cpu_fp64.json`](../breakdown/point_source_image/vertex_dedup_ab_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/point_source_image/vertex_dedup_ab_hpc_ral_cpu_fp64.png)
- the job log [`point_source_cpu_2026_09_23_ral_job_350582_vertex_dedup_ab.out`](point_source_cpu_2026_09_23_ral_job_350582_vertex_dedup_ab.out)
  (stderr held only the benign `No blurring_image provided` warning)

The laptop witness is
[`vertex_dedup_ab_local_cpu_fp64.json`](../breakdown/point_source_image/vertex_dedup_ab_local_cpu_fp64.json)
and [`.png`](../breakdown/point_source_image/vertex_dedup_ab_local_cpu_fp64.png).
Library: PyAutoArray `feature/point-source-cpu-p2`, commits `273e152e` (fix)
and `25894d10` (tests), on `11b93476`.

### Follow-ups found in phase 2 (not fixed)

- **`jax.grad` of `AnalysisPoint` silently returns all zeros without
  `autofit.jax.register_model(model)`.** The forward values are correct, and
  this reproduces on PyAutoFit / PyAutoLens main. The A/B cell registers every
  model before tracing, so its gradient gate is valid. A user calling `jax.grad`
  directly gets zeros with no error. **This is a library issue candidate**
  (PyAutoFit / PyAutoLens): raise or warn on an unregistered model.
- **The `library` route self-labels, so the cell is safe to re-run after
  release.** Once the fix is installed it must read `library_matches: nodedup`.
  A `control` reading on a released install means the install is stale.
- **The laptop JSON's provenance bodies are wrong** (see Environment). The cell
  should capture `inspect.getsource` at import time, or hash the function's
  bytecode, so that a mid-run edit cannot misrecord them. The JSON is left as
  committed.
- **The cell does not record the XLA intra-op thread count or per-route
  memory**; add both before the phase-3 run. It ran 5 calls per round, not the
  planned ≥ 20.
- **Host pinning.** `ral` jobs land on different CPU models (Xeon 8490H in
  phase 1, EPYC 7763 here). Future cross-phase comparisons need the same node or
  an in-job control; consider `--nodelist` or a `--constraint` on the submits.
- **Still open from phase 1:** the `check_submits.py` `python3 -u` regex gap,
  which also applies to this submit; the `activate.sh` worktree-leak guard
  (this run swapped `PYTHONPATH` by hand again); and CI smoke coverage for the
  breakdown cells, which now includes `vertex_dedup_ab.py`.
- **The GPU regression check is DONE.** RAL A100 job 350587 found no
  regression: nodedup is 1.71–2.05× faster than control, and every gate is
  bit-identical (see Decision, "GPU"). The branch's PyAutoArray `25894d10` was
  imported.
- **Provenance-capture fix (done 2026-09-24).** `vertex_dedup_ab.py` now reads
  `inspect.getsource` of the two bodies **at import**, together with a
  `co_code`/`co_consts` SHA-256 (`bytecode_sha256`). It refuses text that does
  not define the named function, so a mid-run edit can no longer misrecord the
  bodies. The laptop JSON is still left as committed.

---

## Phase 3 — precompute the static step-0 lattice (2026-09-24) — DONE, ACCEPTED (RAL jobs 350636 / 350637)

Plan: PyAutoArray#568 (re-titled, phase-3 plan comment). Human decisions (2026-09-24): tolerance
gate, the geometric 11 859-vertex table, on by default for the JAX PointSolver if the A/B accepts.

### Mechanism

Step 0 of the JAX `PointSolver` deflects `triangles.vertices` of a lattice fixed by the solver
geometry alone (static pytree aux data). After phase 2 that is the flat `(3N, 2)` table: 69 849 rows
for the simple ±9.9″ / 0.2″ lattice (23 283 triangles), of which 28 665 are exact-float distinct and
**11 859 geometrically distinct** — a lattice point computed from up to six triangle centres differs
by ~1 ulp. Verified independently on canonical NumPy `CoordinateArrayTrianglesNp` (28 665 exact,
11 859 at 1e-9 rounding). The cluster 200×200 @ 0.7″ lattice: 276 507 / 110 085 / 46 516.

- PyAutoArray `feature/point-source-cpu-p3` `ad0bf97b`: `static_vertex_table(y_min, y_max, x_min,
  x_max, scale)` — `lru_cache`d NumPy builder keyed on the integer lattice position
  `(cy + f·dy, 2·cx + f·dx)`, first-occurrence floats (same arithmetic as `.triangles`), read-only;
  `CoordinateArrayTriangles(vertex_table=None)`, `for_limits_and_scale(..., static_vertices=False)`.
  Derived lattices and pytree round-trips drop the table. Build 0.13 s / 0.75 MB (simple), 0.35 s /
  2.96 MB (cluster); cache hit ~2.5 µs.
- PyAutoLens `feature/point-source-cpu-p3` `b346b6a0`: `AbstractSolver._initial_triangles` passes
  `static_vertices=True` on the JAX path only.

### Cell

`scripts/point_source/likelihood_breakdown/static_lattice_ab.py` — four routes injected into
`AbstractSolver._initial_triangles` for tracing only: `control` (flat, 69 849), `exact` (28 665,
cell-built), `lattice` (11 859, cell-built), `library` (self-labels from the traced step-0 row
count). Fresh closures + `jax.clear_caches()`; 20 rounds × 20 calls; records FLOPs, `memory_analysis`,
RSS delta, compile / lower / first call, table build and cache-hit time, observed thread count.
`--constant-folding` re-enables XLA folding in a separate process (dummy pass name
`constant_folding_probe_disabled`, so PyAutoNerves does not re-add the real flag) and refuses on CPU
unless an HLO probe shows folding ran. The first probe (`arange * 2 + 1`) could not discriminate —
elementwise arithmetic on constants is folded in both modes — so the probe is a constant `c @ c`
(live with the flag, a literal without it; verified in both modes).

### Laptop witness (WSL2, 8 cores, loadavg 2–6 — load-inflated, NOT quotable)

`results/breakdown/point_source/static_lattice_ab_laptop_cpu_fp64.json` (folding off) and
`static_lattice_ab_constant_folding_laptop_cpu_fp64.json` (folding on), cell `3da2583`, library =
the branch (`library_matches: lattice` in both). Median ms per call, bootstrap 90 % CI of
control / lattice:

| row | control | exact | lattice | library | control/lattice | control/exact |
|---|---|---|---|---|---|---|
| simple solved | 5.92 | 4.20 | 3.11 | 3.07 | 1.91 [1.88, 1.93] | 1.41 |
| simple solved vmap-4 | 16.58 | 11.61 | 11.08 | 10.39 | 1.50 [1.39, 1.57] | 1.43 |
| simple plain | 5.80 | 4.23 | 3.14 | 3.09 | 1.85 [1.83, 1.87] | 1.37 |
| cluster solved | 76.58 | 38.70 | 28.61 | 27.82 | 2.68 [2.54, 2.78] | 1.98 |
| cluster plain | 106.86 | 56.53 | 38.24 | 37.37 | 2.79 [2.73, 2.87] | 1.89 |
| *folding on:* simple solved | 7.82 | 5.21 | 3.95 | 3.95 | 1.98 [1.88, 2.11] | 1.50 |
| *folding on:* cluster solved | 72.66 | 43.11 | 34.22 | 33.39 | 2.12 [2.09, 2.17] | 1.69 |

- FLOPs: simple 7.05 → 3.38 M (−52 %), cluster 139.4 → 39.5 M (−72 %). XLA temp buffer: simple
  4.82 → 1.69 MB, cluster 91.6 → 22.6 MB. No memory regression.
- MDI (control p90−p10 / median) 0.28–0.81 on this rerun: the laptop is not a timing host. An
  earlier run of the same cell (loadavg ~2, JSON superseded) gave the same ratios (1.91 / 2.55).
- **Compile time is the open stop-rule question.** Laptop compile seconds per route are single
  samples under load and scatter both ways (simple solved 5.1 → 5.9 s, simple plain 4.7 → 5.8 s,
  cluster solved 51.5 → 40.5 s, cluster plain 31.7 → 51.8 s; the earlier run: 3.0 → 4.1, 3.2 → 3.3,
  32.6 → 34.2, 22.6 → 28.3 s). Several exceed +20 %. Judge the +20 % rule on the pinned RAL CPU job,
  not on these.
- Folding on does not change the verdict: control gains a little from folding, lattice keeps
  ~2×. One earlier folding-on run spent 555 s compiling the cluster-plain control; the rerun took
  34 s — a laptop outlier, recorded, not reproduced.

### Correctness gates (laptop and library suites)

- Laptop, both runs: 31 / 31 gates pass and are **bit-identical** (log L on every streamed instance,
  solved positions and image counts, `jax.grad` finite and non-zero, vmap-4 vs scalar, simple-plain
  step-0 `containing_indices` sets).
- PyAutoArray suite 1645 passed (+19 static-table test cases); PyAutoLens suite 756 passed, 1 xfailed
  (+16 JAX tests in `test_static_lattice_jax.py`, incl. the pinned tie case, which skip without jax). The shape guard (first
  traced deflection grid 11 859 rows) is red on main (69 849).
- autolens_workspace_test `point_source/jax_likelihood/*.py` ×4 and `jax_grad/gradient.py`: rc 0 on
  the branch, output identical to main except timing lines; no rtol-1e-4 pin moved.
- **Tie finding (PASSED by human decision 2026-09-24; see below).** A source placed *bit-exactly* on a traced step-0
  lattice vertex changes the step-0 kept set (16 / 50 constructed ties) and, in 1 / 13, the image
  count: control 3, lattice 2. The control's two positions at the tie vertex (|p − v| = 9.0e-4 each,
  μ ≈ 4.7) both Newton-converge to the same root, the vertex itself — a duplicate; the lattice path
  returns exactly the system's 2 true images. NumPy returns 7 positions (the same 2 roots); a finer
  precision changes the duplicate count on both paths; nudging the source by 1e-9 gives 2 = 2. Main's
  own flat path is not self-consistent at these ties: eager vs jit differ in the step-0 set on 25 / 25
  and in image count on 22 / 25 (lattice vs flat under jit: 6 / 25 and 1 / 25). Generic and
  near-caustic sources are identical to the flat path.

### RAL environment (quotable)

- **CPU — job 350636**, partition `ral`, pinned `--nodelist=euclid-ral-compute-10-2` (Intel Xeon
  Platinum 8490H, the phase-1 host; phase 2 ran on an EPYC 7763, so only in-job ratios compare across
  phases). 8 CPUs (`sched_affinity` 8, `NPROC=8`, BLAS threads 1), fp64, JAX 0.10.2, loadavg 0.28 at
  start. Two runs in one job: folding **off** (the PyAutoNerves default) then folding **on**
  (`--constant-folding`; the HLO probe confirms folding ran, `agrees_with_request: true`). Walls 637 s
  and 606 s. Empty stderr.
- **A100 — job 350637**, `euclid-ral-gpu-1`, one NVIDIA A100 80GB PCIe, backend asserted `gpu`, fp64,
  folding off, wall 984 s. Empty stderr.
- **Provenance (all three JSONs):** `source_revisions` PyAutoArray `ad0bf97b`, PyAutoLens `b346b6a0`
  (imported from the branch clones `/mnt/ral/jnightin/autolens_profiling_wt/{PyAutoArray,PyAutoLens}_point-source-cpu-p3`,
  refused otherwise), autolens_profiling `3da2583`, PyAutoGalaxy `70a61e26`, PyAutoFit `a7368401`,
  PyAutoNerves `1fa613aa`. `library_matches: lattice` on every row; on the simple rows the library's
  optimised HLO is hash-identical to the cell's lattice route. `all_gates_pass: true`.
- Protocol as the laptop: 20 rounds × 20 calls (400 samples per route), round-robin with rotated
  start, 3 warm calls, 16-instance parameter stream. The library `static_vertex_table` equals the cell's
  lattice table on both geometries; build 34 ms / 0.75 MB (simple) and 184 ms / 2.96 MB (cluster) on the
  Xeon, cache hit 1.2–2.4 µs.

Numbers below are read from the JSONs by script. Ratios are per-route medians with the JSON's bootstrap
90 % CI; **MDI** = control (p90 − p10) / median; compile is `compile_s` of a **single cold compile per
route** (not a median); temp = XLA `memory_analysis().temp_size_in_bytes`; FLOPs = `cost_analysis()`.

**RAL CPU, constant folding off** (`static_lattice_ab_hpc_ral_cpu_fp64.json`)

| Row | control median [p10, p90] ms | exact | lattice | library | control / library [90 % CI] | control / lattice [90 % CI] | paired per-round c/lat median [p10, p90] | MDI | compile s ctrl → lib | temp MB ctrl → lib | FLOPs ctrl → lib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `simple_solved` | 3.540 [3.067, 4.296] | 2.254 | 1.674 | 1.758 | **2.01×** [1.63, 2.09] | 2.11× [1.92, 2.16] | 1.94 [1.79, 2.13] | 34.7 % | 2.91 → 3.14 (+8.0 %) | 4.82 → 1.69 | 7.05M → 3.38M |
| `simple_solved_vmap4` (per batch of 4) | 8.258 [7.128, 9.332] | 6.550 | 5.761 | 5.784 | **1.43×** [1.42, 1.44] | 1.43× [1.42, 1.46] | 1.44 [1.39, 1.63] | 26.7 % | 3.39 → 3.71 (+9.5 %) | 15.47 → 6.76 | 23.30M → 11.83M |
| `simple_plain` | 3.479 [3.047, 4.298] | 2.267 | 1.673 | 1.669 | **2.08×** [1.91, 2.13] | 2.08× [1.89, 2.13] | 1.92 [1.72, 2.17] | 36.0 % | 2.58 → 2.67 (+3.3 %) | 4.82 → 1.69 | 7.05M → 3.38M |
| `cluster_solved` | 47.36 [40.51, 59.02] | 18.82 | 9.211 | 9.085 | **5.21×** [5.10, 5.30] | 5.14× [5.05, 5.26] | 5.20 [4.91, 5.44] | 39.1 % | 22.77 → 24.66 (+8.3 %) | 91.55 → 22.64 | 139.44M → 39.47M |
| `cluster_plain` | 47.60 [41.12, 53.95] | 17.70 | 9.243 | 9.225 | **5.16×** [5.04, 5.26] | 5.15× [5.04, 5.26] | 5.16 [4.85, 5.59] | 27.0 % | 16.69 → 18.72 (+12.2 %) | 91.54 → 22.63 | 139.43M → 39.46M |

**RAL CPU, constant folding on** (`static_lattice_ab_constant_folding_hpc_ral_cpu_fp64.json`)

| Row | control median [p10, p90] ms | exact | lattice | library | control / library [90 % CI] | control / lattice [90 % CI] | paired per-round c/lat median [p10, p90] | MDI | compile s ctrl → lib | temp MB ctrl → lib | FLOPs ctrl → lib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `simple_solved` | 4.093 [3.056, 4.811] | 2.194 | 1.785 | 1.722 | **2.38×** [2.08, 2.45] | 2.29× [1.89, 2.39] | 2.12 [1.72, 2.34] | 42.9 % | 2.85 → 2.91 (+1.9 %) | 4.69 → 1.69 | 6.79M → 3.10M |
| `simple_solved_vmap4` (per batch of 4) | 7.632 [7.099, 10.500] | 5.675 | 4.982 | 5.499 | **1.39×** [1.31, 1.63] | 1.53× [1.34, 1.65] | 1.55 [1.29, 1.72] | 44.6 % | 3.22 → 3.43 (+6.4 %) | 15.34 → 6.76 | 23.02M → 11.53M |
| `simple_plain` | 3.564 [3.035, 4.800] | 2.392 | 1.689 | 1.696 | **2.10×** [1.99, 2.25] | 2.11× [2.01, 2.31] | 2.03 [1.81, 2.40] | 49.5 % | 2.44 → 2.57 (+5.2 %) | 4.69 → 1.69 | 6.79M → 3.10M |
| `cluster_solved` | 39.03 [30.66, 45.27] | 12.96 | 8.506 | 8.599 | **4.54×** [4.39, 4.64] | 4.59× [4.44, 4.70] | 4.58 [4.19, 4.94] | 37.4 % | 21.93 → 24.14 (+10.1 %) | 91.88 → 22.64 | 131.39M → 30.34M |
| `cluster_plain` | 32.34 [27.91, 37.96] | 13.43 | 8.257 | 8.273 | **3.91×** [3.83, 3.99] | 3.92× [3.84, 3.98] | 3.91 [3.58, 4.22] | 31.1 % | 16.28 → 18.97 (+16.5 %) | 91.54 → 22.63 | 131.38M → 30.33M |

**RAL A100, fp64** (`static_lattice_ab_hpc_ral_a100_fp64.json`)

| Row | control median [p10, p90] ms | exact | lattice | library | control / library [90 % CI] | control / lattice [90 % CI] | MDI | compile s ctrl → lib | temp MB ctrl → lib | FLOPs ctrl → lib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `simple_solved` | 0.846 [0.836, 0.881] | 0.843 | 0.839 | 0.842 | **1.01×** [1.00, 1.01] | 1.01× [1.01, 1.01] | 5.3 % | 5.28 → 5.57 (+5.4 %) | 3.36 → 0.39 | 6.60M → 3.07M |
| `simple_solved_vmap4` (per batch of 4) | 0.942 [0.932, 0.974] | 0.922 | 0.925 | 0.925 | **1.02×** [1.02, 1.02] | 1.02× [1.02, 1.02] | 4.5 % | 6.28 → 6.19 (−1.5 %) | 7.83 → 1.53 | 21.98M → 10.82M |
| `simple_plain` | 0.824 [0.813, 0.871] | 0.821 | 0.818 | 0.818 | **1.01×** [1.00, 1.01] | 1.01× [1.01, 1.01] | 7.0 % | 5.19 → 5.10 (−1.6 %) | 3.36 → 0.39 | 6.60M → 3.07M |
| `cluster_solved` | 2.696 [2.601, 2.850] | 2.566 | 2.516 | 2.515 | **1.07×** [1.06, 1.09] | 1.07× [1.06, 1.08] | 9.2 % | 43.13 → 48.84 (+13.2 %) | 42.06 → 6.35 | 120.38M → 34.59M |
| `cluster_plain` | 2.073 [2.013, 2.258] | 2.002 | 1.984 | 1.991 | **1.04×** [1.04, 1.05] | 1.05× [1.04, 1.05] | 11.8 % | 36.11 → 38.76 (+7.4 %) | 42.05 → 6.35 | 120.37M → 34.57M |

**Reading the rows.**

- **CPU gain is material and far outside the noise.** Folding off, the library is 2.0–2.1× faster on the
  simple likelihood (3.54 → 1.76 ms solved, 3.48 → 1.67 ms plain) and **5.2×** on the two-source cluster
  (47.4 → 9.1 ms), with every CI and every paired per-round p10 above 1.6 (simple scalar rows) and 4.8 (cluster).
  `library / lattice` is 0.99–1.05 (CIs straddle or touch 1): the shipped code reproduces the cell's
  lattice route. The cluster gains more because its 276 507-row step-0 input shrinks 5.9× to 46 516
  rows and its 13-component deflections dominate the call.
- **`exact` (28 665 / 110 085 rows) captures about half the gain**; the geometric dedup to 11 859 /
  46 516 rows is the other half (exact / lattice 1.35× simple, 2.0× cluster). The human decision to
  ship the geometric table is borne out.
- **`vmap`-4 gains less (1.43×)**, as in phase 2: the step-0 lattice is batch-invariant, so under `vmap`
  the control already amortises its step-0 work across the batch. Per likelihood the library costs
  1.45 ms at batch 4 against 1.76 ms scalar.
- **Folding on does not change the verdict.** It speeds the control's cluster rows (47.4 → 39.0 /
  32.3 ms) because the flat step-0 table becomes a folded constant; the library is barely affected
  (9.1 → 8.6 / 8.3 ms). The library still wins 2.1–2.4× simple, 1.39× vmap-4 and 3.9–4.5× cluster.
  Folding also trims FLOPs (library cluster 39.5M → 30.3M). Folding stays off by default (PyAutoNerves);
  nothing here argues for flipping it.
- **Compile (the open laptop question) — resolved on the quiet node.** Control → library `compile_s`
  is **+3.3 % to +12.2 %** folding off and **+1.9 % to +16.5 %** folding on (worst: cluster plain 16.28 →
  18.97 s); lowering is unchanged or faster (cluster solved 30.58 → 30.03 s) and first call falls
  (cluster solved 49 → 11 ms). Every row is under the +20 % stop rule. The laptop's readings (+37 % and
  +63 % on some rows, −21 % on another) did not reproduce: they were load noise on single samples. Caveat: each figure is one cold
  compile per route, not a median, so differences of a few percent are not resolved.
- **Memory — no regression, large reduction.** XLA temp 4.82 → 1.69 MB (simple), 91.6 → 22.6 MB
  (cluster), 15.5 → 6.8 MB (vmap-4); on the A100 3.36 → 0.39 MB and 42.1 → 6.35 MB. RSS delta at compile
  falls on every heavy row (cluster solved 861 → 82 MB CPU, 907 → 178 MB A100). The table itself is a
  0.75 / 2.96 MB host constant, built once per geometry.
- **A100 — no regression, a small gain.** The GPU call is launch/latency-bound, not FLOP-bound:
  halving (simple) or cutting 3.5× (cluster) the FLOPs moves the call by only 1–7 % (0.846 → 0.842 ms,
  2.70 → 2.52 ms). All CIs are ≥ 1.00. Compile is −1.6 % to +13.2 % (cluster solved 43.1 → 48.8 s), under
  +20 %.
- **MDI 27–50 % on the Xeon node is large**: it is per-call dispersion (p10–p90 across a 16-instance
  parameter stream and 400 calls), not uncertainty of the median. The effects are 2–5×, and the
  bootstrap CIs of the ratios are a few percent wide.

### Correctness gates (RAL)

All **31 / 31** gates in `gate_summary` pass in all three JSONs, with max |Δ| = 0 everywhere:

- log L on all five rows, every route pair (control vs exact / lattice / library), all 16 stream
  instances, **bit-identical** (NaN pattern equal), deterministic within each route; fiducial simple
  solved `7.743201200876812` on every route (identical to phases 1–2);
- solved positions and image counts: simple 4 finite per instance, cluster 3 + 3, three instances each,
  padded arrays equal, max |Δ| = 0.0;
- simple-solved `jax.grad`: finite, non-zero on every instance, identical between routes (Δ = 0);
- vmap-4 against scalar within each route and across routes: bit-identical;
- simple-plain step-0 `containing_indices` sets identical across routes.

The same holds on CPU with folding on and on the A100.

### Tie finding — PASSED by human decision 2026-09-24

Not exercised by the RAL gates (stream sources are generic). From the laptop study above: a source
placed **bit-exactly on a traced step-0 lattice vertex** can change the step-0 kept set (16 / 50
constructed ties) and, in 1 / 13, the image count — **control 3 vs lattice 2**. The control's third image
is a **duplicate**: its two positions at the tie vertex both Newton-converge to the same root, the vertex
itself; the lattice path returns exactly the system's 2 true images. Main's own flat path is **not
self-consistent** at such ties: eager vs jit differ in image count on **22 / 25** (lattice vs flat under
jit: 1 / 25). Nudging the source by 1e-9 gives 2 images on both paths. The plan's gate asks for
"identical image counts" and so is formally failed on this constructed measure-zero case; accepting it
(as a duplicate-removal, not a lost image) was put to the human at ship.

**Decision (human, 2026-09-24): PASS.** The case is accepted as the removal of a duplicate and pinned
as a regression test, PyAutoLens `test_autolens/point/triangles/test_static_lattice_jax.py::test__source_on_a_step_0_vertex_returns_the_two_true_images`
(branch `feature/point-source-cpu-p3` `972d454e`): the simple SIE of the cell, source on the image of
step-0 vertex v = (−0.8, −1.99185843) (flat vertex 31 732), solved under `jit` with the source a traced
input; the static-lattice solver must return exactly the 2 true images (v and the counter-image
~(0.41013634, 0.89463539)) within 2 × `pixel_scale_precision`. With the source a closed-over constant
instead, the flat path also returns 2 here — one more face of the flat path's instability at ties.

### Decision — ACCEPT

Stop rule (PyAutoArray #568): reject if the simple gain < max(5 %, 2× MDI) with no cluster gain; any
gate fails; compile +20 %; memory or GPU regression.

- **Gain.** Simple solved speed-up +101 % (2.01×) against 2× MDI = 69 %; cluster +421 % (5.21×) against
  78 %. The cluster clause alone rules out rejection. (Read as a fractional time reduction instead —
  50 % simple vs 69 % — the simple row alone would not clear 2× MDI on this high-dispersion node, but the
  cluster's 81 % reduction does, and the paired per-round p10 is 1.79× simple / 4.91× cluster.)
- **Gates.** 31 / 31 bit-identical on CPU (both folding modes) and A100; the tie case above PASSED by
  human decision 2026-09-24 and is pinned as a PyAutoLens test.
- **Compile.** Worst +12.2 % (folding off), +16.5 % (folding on), +13.2 % (A100): all < +20 %.
- **Memory / GPU.** XLA temp memory −56 % to −88 %; A100 1.01–1.07× faster.

Ship library-first (PyAutoArray `ad0bf97b` → PyAutoLens `b346b6a0` + tie test `972d454e` → autolens_profiling), on by default
for the JAX PointSolver as decided on 2026-09-24.

### Post-fix budget and phase-4 handoff

After phase 3 the step-0 lattice is no longer the dominant cost on `simple`. FLOP accounting (folding
off, hardware-independent; the step-0 share is an **estimate** from the marginal FLOP per removed step-0
row between the control and lattice routes):

- **Simple (3.38M FLOP).** The marginal step-0 cost is (7.05 − 3.38)M / (69 849 − 11 859) ≈ 63 FLOP per
  row, so the 11 859-row step 0 is ≈ 0.75M (**≈ 22 %**). The seven refinement steps' deflections
  (7 × 289 599 = 2.03M, phase 2) are now **≈ 60 %**; containment, the `f64[60]` neighbourhood sorts, β\*,
  the magnification filter and χ² share the remaining ≈ 0.6M (≈ 18 %).
- **Cluster (39.47M FLOP).** ≈ 435 FLOP per step-0 row → the 46 516-row step 0 is ≈ 20M (**≈ 51 %**);
  the refinement steps and the rest are ≈ 19M. Deflections of the 13-component lens still dominate
  both halves.
- **GPU.** The A100 call barely moved with FLOPs, so FLOP levers will not show on GPU; any GPU
  work is a launch-count/latency question, separately scoped.

**Re-ranked phase-4 levers (proposed, none measured):**

1. **Simple: initial scale versus refinement steps (old 4(b)).** The seven refinement steps now carry
   ≈ 60 % of the simple call; fewer steps at a finer step 0 (or a coarser final precision where
   admissible) is the largest remaining lever. It is a correctness knob: image completeness and
   position precision across a prior must be shown.
2. **Cluster: dPIE/NFW deflection cost (old 4(d)).** Deflections dominate every cluster step; this is
   a separately scoped PyAutoGalaxy phase.
3. **Grid-extent guidance (old 4(a)).** Now acts on the ≈ 22 % (simple) / ≈ 51 % (cluster) step-0
   share only, plus containment; still needs image-completeness evidence. Stronger at cluster scale.
4. **`MAX_CONTAINING_SIZE` / neighbourhood fan-out (old 4(c)).** Sets the 720-point refinement grids and
   the `f64[60]` sorts, so it now scales the dominant simple share; correctness knob.

Before ranking by wall time, run the phase-1 breakdown cells on the post-phase-3 code (after release)
so the per-step time split, not only FLOPs, backs the choice.

### Reproduce (RAL)

```bash
# RAL worktree of feature/point-source-cpu-p3, with the PyAutoArray / PyAutoLens branch clones beside it
cd hpc/batch_cpu && sbatch submit_breakdown_point_source_static_lattice_ab_ral_cpu_fp64   # folding off, then on
cd ../batch_gpu && sbatch submit_breakdown_point_source_static_lattice_ab_a100_fp64
```

JSONs and PNGs: [`static_lattice_ab_hpc_ral_cpu_fp64`](../breakdown/point_source_image/static_lattice_ab_hpc_ral_cpu_fp64.json)
([png](../breakdown/point_source_image/static_lattice_ab_hpc_ral_cpu_fp64.png)),
[`static_lattice_ab_constant_folding_hpc_ral_cpu_fp64`](../breakdown/point_source_image/static_lattice_ab_constant_folding_hpc_ral_cpu_fp64.json)
([png](../breakdown/point_source_image/static_lattice_ab_constant_folding_hpc_ral_cpu_fp64.png)),
[`static_lattice_ab_hpc_ral_a100_fp64`](../breakdown/point_source_image/static_lattice_ab_hpc_ral_a100_fp64.json)
([png](../breakdown/point_source_image/static_lattice_ab_hpc_ral_a100_fp64.png)).
Logs: [`point_source_cpu_2026_09_24_ral_job_350636_static_lattice_ab.out`](point_source_cpu_2026_09_24_ral_job_350636_static_lattice_ab.out),
[`point_source_cpu_2026_09_24_ral_job_350637_static_lattice_ab_a100.out`](point_source_cpu_2026_09_24_ral_job_350637_static_lattice_ab_a100.out).

## Phase 4 — profile the residue and iterate — NOT STARTED

Rank by new evidence; the phase-3 re-rank above puts (b) initial scale versus refinement count
first for `simple` and (d) cluster dPIE/NFW deflections first at cluster scale, then (a) grid-extent
guidance and (c) `MAX_CONTAINING_SIZE` / neighbourhood fan-out. Any PyAutoGalaxy work is its own phase.

## Phase 4a — re-baseline + solver-config sweep (2026-09-26) — DONE

Issue: [autolens_profiling #314](https://github.com/PyAutoLabs/autolens_profiling/issues/314). Profiling
branch `feature/point-source-cpu-p4`. **Workspace-only research: no library code changed.** Phases 2
and 3 are released in 2026.9.26.1, and 4a measures on that released code.

### Scope decision

Human decision (2026-09-26): this campaign is **single-source only**, the `scripts/point_source_image/`
use case. The two-source cluster rows that phases 1–3 carried leave the campaign. They move to epic
`cluster-pointsolver-speed`, prompt `organs/PyAutoMind/draft/research/autolens_profiling/cluster_pointsolver_speed.md`,
which starts from its own data and a cluster `likelihood_breakdown` baseline. The cluster lever
(d), dPIE/NFW deflections, went with it. 4a measures the three single-source levers: (a) grid
extent, (b) initial scale versus refinement count, and (c) `MAX_CONTAINING_SIZE` / neighbourhood
fan-out. All three are correctness knobs, so every configuration is gated on image completeness
before it is timed.

### Environment (quotable)

- **RAL CPU**, partition `ral`, pinned `--nodelist=euclid-ral-compute-10-4`, Intel Xeon Platinum 8490H
  (the phase-1/3 model; phases 1 and 3 ran on 10-2, so only in-job ratios compare across phases).
  8 CPUs (`sched_affinity` 8, `NPROC=8`, BLAS threads 1), fp64, JAX 0.10.2. XLA_FLAGS are the phase-1
  row's byte for byte (constant folding off).
- **Library code:** only the shared RAL mirror `/mnt/ral/jnightin/PyAuto` was on `PYTHONPATH`, with no
  branch clones. The job refuses to run otherwise and asserts the JSON `source_revisions` against the
  mirror HEADs. `source_revisions` are PyAutoArray `3de624b5`, PyAutoLens `86054bbc`, PyAutoGalaxy
  `70a61e26`, PyAutoFit `dd9fbe0a`, PyAutoNerves `1fa613aa` and autolens_profiling `6c45fec`. The mirror
  was **not** pulled to the 2026.9.26.1 tags, because other sessions' RAL arrays were running on it.
  Its point-source code path is byte-identical to the tags: PyAutoLens differs only by a hooks commit,
  and PyAutoArray only in `util/jax_nnls.py` + nnls config keys.
- **Step 1**, job **356365**: loadavg 0.00 at start, 1 committed run + 4 replicates.
- **Step 2**, job **356367**: loadavg 0.00 → 1.23, wall 787 s, `all_gates_pass: true`.
  - Job 356367 **supersedes 356366**. Job 356366 used the same draws and protocol, but its step-0
    trace/containment split method was flawed, so its artefacts are kept out of the repo.
  - The laptop witness JSON (`solver_config_sweep_laptop_cpu_fp64.json`) is **not quotable**.

### Re-baseline — measured per-step split (job 356365)

Cell: `scripts/point_source_image/likelihood_breakdown/image_plane.py`, `--config-name hpc_ral_cpu_fp64_p4`.
At about 2 ms, one run cannot separate a stage from noise, so the job runs the cell 5 times on the
same node. The committed row is
[`image_plane_hpc_ral_cpu_fp64_p4.json`](../breakdown/point_source_image/image_plane_hpc_ral_cpu_fp64_p4.json)
([png](../breakdown/point_source_image/image_plane_hpc_ral_cpu_fp64_p4.png)); the 4 replicates are in the
gitignored `output/` tree.

- **Fused solved: median 2.095 ms** across the 5 runs (1.988–2.198; the committed row 2.198). Plain: 1.857 ms.
- Fiducial log L is **`7.743201200876812`** under jit, bit-identical to phases 1–3. Eager gives `…817`.
- vmap-2 is bit-identical, and `jax.grad` is finite.

The table differences the per-run median cumulative JIT prefixes against the median fused call.
Rows can go negative under XLA fusion.

| stage | ms | % of fused | per-run spread (ms) |
|---|---:|---:|---|
| source centre β\* | 0.166 | 7.9 | 0.149 … 0.172 |
| **step 0** (11 859-row lattice) | **1.384** | **66.0** | 1.309 … 1.483 |
| steps 1–7 (refinement, summed) | 0.310 | 14.8 | each step −0.16 … 0.46 |
| magnification filter | 0.153 | 7.3 | −0.013 … 0.168 |
| pairing χ² / residual | 0.082 | 3.9 | −0.038 … 0.322 |

**This overturns the phase-3 FLOP ranking.** That ranking estimated the refinement steps at ≈ 60 %
and step 0 at ≈ 22 %. In wall time, step 0 is two thirds of the call and refinement is ≈ 15 %: the
seven 720-row steps are cheap per FLOP. Within refinement, the neighbourhood stage is the largest
part (≈ 0.43 ms summed, ≈ 20 %).

*Caveat, corrected below:* this cell also split step 0 into "ray trace" 0.965 ms and "containment"
0.462 ms. That split is wrong. The cell's ray-trace prefix returns the materialised triangles, so it
counts the `vertices[indices]` gather as trace. The sweep measures the corrected split.

### Solver-config sweep (job 356367)

- Cell: [`scripts/point_source_image/likelihood_breakdown/solver_config_sweep.py`](../../scripts/point_source_image/likelihood_breakdown/solver_config_sweep.py).
- Submit: [`submit_breakdown_point_source_image_solver_config_sweep_ral_cpu_fp64`](../../hpc/batch_cpu/submit_breakdown_point_source_image_solver_config_sweep_ral_cpu_fp64).
- JSON: [`solver_config_sweep_hpc_ral_cpu_fp64.json`](../breakdown/point_source_image/solver_config_sweep_hpc_ral_cpu_fp64.json)
  ([png](../breakdown/point_source_image/solver_config_sweep_hpc_ral_cpu_fp64.png)).
- Log: [`point_source_cpu_2026_09_26_ral_job_356367_solver_config_sweep.out`](point_source_cpu_2026_09_26_ral_job_356367_solver_config_sweep.out).

**Protocol.**
- 26 configurations of the production `FitPositionsImagePairAllSolved` likelihood. Each is its own
  interleaved route, and no route's speed is ever substituted into another's.
- Timing: 20 rounds × 20 calls, 3 warm calls and a 16-instance parameter stream. Routes run
  round-robin with a rotated start. Each route gets a fresh closure and solver plus
  `jax.clear_caches()`.
- Statistics: medians with a 2000-sample bootstrap 90 % CI.
- `MAX_CONTAINING_SIZE` is patched in-process for tracing only. It needed the module constant **and**
  the `ArrayTriangles.__init__` / `for_limits_and_scale` `__defaults__`, because the default argument
  binds at import. It is restored by value, and a gate checks both.

**Completeness method.**
- Reference: a fine solve at extent ±12″, scale 0.05, precision 1e-4 and MCS 60, on **200 seeded draws
  from the full point_source prior** (the source is the solved β\*).
- Stress set: **200 draws outside the prior** (θ_E U(1, 2), ell_comps U(−0.2, 0.2), source U(−0.4, 0.4)).
- Reference floor: a second reference (±11″ / 0.07) agrees with the first on 200/200 + 200/200 draws,
  with max position error 1.6e-4″.
- **Admissible** means 100 % distinct-image multiplicity agreement on the prior draws (distinct at
  0.005″) **and** a max position error of at most **0.002″**. The tolerance is set by the solver, not
  by the data. The default's last-step triangle side is 0.0016″, and the default itself sits at
  8.5e-4″ against the reference. 0.002″ is 0.04 σ of the σ = 0.05″ position noise.
- **Precision-equivalent** means a last-step triangle side no larger than the default's, i.e. the same
  or finer final precision.

**The default is complete.** The default (±9.9″ / 0.2 / 1e-3 / MCS 15 / nd 1) matches the reference's
image multiplicity on **200/200 prior and 200/200 stress** draws. Its max position error is 8.5e-4″
(p99 7.9e-4″). Its |Δ log L| against the reference is at most 0.87 (p99 0.57): that is the default's own
discretisation, and every precision-equivalent config below shares it exactly.

| config | extent ″ | scale | steps | step-0 rows | median ms | speed-up [90 % CI] | admissible | prec.-equiv. | notes |
|---|---:|---:|---:|---:|---:|---:|:-:|:-:|---|
| control (default) | ±9.9 | 0.2 | 8 | 11 859 | 1.824 | 1.00 [0.91, 1.10] | ✓ | ✓ | 200/200 + 200/200 |
| extent | ±6 | 0.2 | 8 | 4 428 | 1.203 | **1.52** [1.47, 1.66] | ✓ | ✓ | log L = control |
| extent | ±4 | 0.2 | 8 | 2 075 | 1.025 | **1.78** [1.72, 1.96] | ✓ | ✓ | log L = control |
| extent | ±3 | 0.2 | 8 | 1 197 | 0.888 | **2.05** [1.98, 2.24] | ✓ | ✓ | log L = control |
| extent | ±2.5 | 0.2 | 8 | 848 | 0.814 | **2.24** [2.16, 2.45] | ✓ | ✓ | log L = control |
| scale | ±9.9 | 0.3 | 9 | 5 400 | 1.337 | 1.36 [1.32, 1.49] | ✓ | ✓ (finer) | finer last side 0.0012″ |
| scale | ±9.9 | 0.4 | 9 | 3 030 | 1.272 | **1.43** [1.39, 1.57] | ✓ | ✓ | log L = control on 200/200 |
| scale | ±9.9 | 0.5 | 9 | 1 944 | 1.079 | 1.69 [1.63, 1.84] | ✓ | ✗ | coarser 0.0020″; stress 198/200 |
| scale | ±9.9 | 0.8 | 10 | 816 | 0.979 | 1.86 | ✗ | ✓ | prior 195/200, stress 185/200 |
| combo | ±4 | 0.4 | 9 | 559 | 0.849 | 2.15 [2.08, 2.35] | ✓ | ✓ | |
| combo | ±3 | 0.4 | 9 | 330 | 0.819 | 2.23 [2.15, 2.43] | ✓ | ✓ | |
| **combo (best)** | **±2.5** | **0.4** | 9 | 243 | **0.771** | **2.37** [2.28, 2.59] | ✓ | ✓ | vmap-16 5.55× |
| combo | ±4 / ±3 | 0.5 | 9 | 385 / 216 | 0.798 / 0.792 | 2.29 / 2.30 | ✓ | ✗ | stress 198/200 |
| combo | ±4 / ±3 | 0.8 | 10 | 161 / 102 | 0.861 / 0.838 | 2.12 / 2.18 | ✗ | ✓ | prior 195/200 |
| finer step 0 | ±9.9 / ±4 | 0.1 | 7 | 46 284 / 7 824 | 5.712 / 1.416 | 0.32 / 1.29 | ✓ | ✓ | loses vs same extent at 0.2 |
| MCS 8 / 10 | ±9.9 | 0.2 | 8 | 11 859 | 1.649 / 1.720 | 1.11 / 1.06 | ✗ | ✓ | prior 124/200, 173/200 |
| nd 0 / nd 2 | ±9.9 | 0.2 | 8 | 11 859 | 1.363 / 2.803 | 1.34 / **0.65** | ✗ / ✓ | ✓ | nd 0: prior 18/200 |
| precision 0.002 / 0.005 | ±9.9 | 0.2 | 7 / 6 | 11 859 | 1.951 / 1.845 | 0.94 / 0.99 | ✓ / ✗ | ✗ | no speed-up |

Further results:

- **vmap** (per likelihood, control → ±2.5/0.4):
  - batch 1: 1.783 → 0.839 ms (2.13×);
  - batch 4: 1.442 → 0.455 ms (3.17×);
  - batch 16: 1.618 → 0.292 ms (**5.55×** [5.48, 5.73]).
  - The step-0 saving grows under batching: the control's vmap gain stalls, and the small lattice's
    gain keeps growing.
- **Compile and memory:** `compile_s` is 3.13 s for the control and 3.63 s for ±2.5/0.4 (+16 %, one
  cold compile each). FLOPs fall from 3.38M to 0.96M, and XLA temp memory from 1.69 MB to 0.29 MB.
- **Images the solver needs:** the reference's images reach max |coord| 1.78″ and max radius 1.81″ on
  the prior, and 2.30″ / 2.52″ on the stress set. That is why ±2.5″ still passes on these draws. It says
  nothing about real galaxy-scale data with larger Einstein radii or offset centres; see the human
  decision below.
- **Gates:** all pass.
  - The fiducial is bit-exact at `7.743201200876812`.
  - The MCS patch takes effect (positions shape `(MCS, 2)`) and is restored.
  - `jax.grad` is finite and non-zero on every instance for the control and the best config.

### Step-0 cost anatomy — the triangle gather

The sweep's step-0 split uses four prefixes: source centre, `jnp.sum(plane.vertices)`, the
materialised plane triangles, and `containing_indices`.

- On the default lattice (11 859 rows / 23 283 triangles), step 0 is **1.48 ms, 81 % of the 1.824 ms
  likelihood**. It divides into:
  - ray trace (deflecting the 11 859 vertices): **0.27 ms**;
  - containment: **1.21 ms, 66 % of the likelihood**.
- Containment includes the `vertices[indices]` gather, `Point.mask` and `jnp.where`. The gather alone,
  materialising the `(23 283, 3, 2)` triangle array, is **≈ 0.90 ms, ≈ 49 % of the likelihood**.
- **The step-0 cost is not deflection arithmetic.** Phase 3 cut the deflected rows to 11 859. What
  remains is the memory-bound gather of 23 283 × 3 vertices and the containment test over every
  triangle.
- The 356365 split above (ray trace 0.965, containment 0.462 ms) is wrong for this reason: its
  ray-trace prefix is the materialised array.
- Under ±2.5/0.4 (243 rows), containment falls to **0.06 ms** and the whole of step 0 to 0.07 ms.
- A code lever that removes the gather should therefore recover most of the extent/scale speed-up
  **without** shrinking the default grid, and without any completeness risk.

### Latent `MAX_CONTAINING_SIZE` overflow (17 > 15)

The sweep also counted containing triangles **uncapped**, in NumPy on the 200 prior draws.

- The default's step-0 count has median 9, p99 15 and **max 17, on prior draw 12**. That exceeds
  `MAX_CONTAINING_SIZE = 15`.
- 23 draws exceed 12, and 1 draw exceeds 15. Steps 1–7 peak at 13 / 11 / 9 / 7 / 5 / 5 / 5.
- Draw 12 still passed completeness, because the truncated entries happened to be the spurious
  fold-line candidates. That is luck, not a guarantee: the cap silently truncates, with no warning.
- Under ±2.5/0.4, the max is exactly 15 at step 1 (p99 15), with no draw over the cap. The headroom is
  zero there too.
- MCS 8 and MCS 10 lose images outright (prior 62 % and 86.5 %). The cap is load-bearing, and 15 is at
  the edge of the prior.

### Disposition of the levers

- **(a) Grid extent — real but a per-dataset choice, not a library default.** ±6″ to ±2.5″ are all
  admissible and give 1.52–2.24× on these draws, and extent-only configs reproduce the control's log L
  exactly. The admissible extent depends on the lens, so this is a **workspace and user setting**,
  backed by a library sanity-check warning. See the human decision below.
- **(b) Initial scale — a finding, within limits.** Scale 0.4 is precision-equivalent (the same 0.0016″
  last side, one extra step), complete on every draw, log-L-identical to the control on 200/200, and
  1.43× at full extent.
  - Scale 0.3 is also admissible, but slower (1.36×).
  - Scale 0.5 coarsens the final precision (0.0020″) and drops 2 stress draws' images.
  - Scale 0.8 is **inadmissible** (prior 195/200).
  - A finer step 0 (0.1) loses everywhere.
  - Combined with extent, 0.4 gives the best row: ±2.5/0.4 at 2.37×.
- **(c) `MAX_CONTAINING_SIZE` / neighbourhood — no speed lever, but a correctness fix.** Lowering MCS
  (8 / 10) and `neighbor_degree` 0 lose images; `neighbor_degree` 2 is 0.65×. The actionable finding is
  the other direction: add **headroom** above the 17 observed.
- **Precision block — no speed-up.** 0.002 is 0.94× and 0.005 is 0.99×: removing one or two 720-row
  steps saves nothing measurable, and both coarsen the positions.
- **Neighbourhood block —** see (c). The refinement steps are ≈ 15 % of wall time, so they are not
  where the time is.

### Human decisions (2026-09-26, live)

The human agreed all three recommendations:
1. Phase 4b is the step-0 gather/containment **code** lever, first.
2. Raise `MAX_CONTAINING_SIZE` headroom from 15 to about 20, measured.
3. The starting-scale finding stands.

On extent, the human said:

> "I think +-3" is a bit small for galaxy scale lenses and I think we would need to update it in
> workspace accordingly for each package. but +-10" still wont do clusters well so I think a
> workspace level task is right. We probabbly need some sort of a sanity check that prints or alerts
> the user? and warning based on the extent of the masked data (or the Point dataset)?"

So there is **no library default change for the extent**. The extent is set per workspace package,
and the library gains a construction-time sanity-check warning.

### Next tasks (filed in PyAutoMind, epic `point-source-cpu-speed`)

1. **Phase 4b**: `draft/feature/autoarray/pointsolver_step0_gather_containment.md`. Remove or cut the
   ≈ 0.9 ms step-0 `vertices[indices]` gather, e.g. regular-lattice arithmetic or slicing, or a fused
   containment. The result must stay bit-identical. Priority high.
2. **Phase 4c**: `draft/feature/autoarray/pointsolver_max_containing_size_headroom.md`. Raise MCS 15 → ~20
   (measure 18 / 20 / 24), consider an overflow counter, and mind the `__defaults__` binding trap.
3. `draft/feature/autolens/pointsolver_extent_sanity_check.md`. A construction-time (non-JAX) warning
   when `PointDataset` positions (or the mask) approach the solver grid edge, plus a softer perf hint
   when the grid is far larger than needed.
4. `draft/feature/autolens_workspace/pointsolver_grid_extent_per_package.md`. Set galaxy-scale
   point-solver grids per workspace package, proposing ±4″ to ±6″ with a 0.4″ initial scale. It
   depends on 3 (library first), and cluster scripts are left alone.

### Reproduce (RAL)

```bash
# RAL worktree of feature/point-source-cpu-p4; library code = the shared mirror only
cd hpc/batch_cpu && sbatch submit_breakdown_point_source_image_image_plane_p4_ral_cpu_fp64      # step 1 (1 + 4 replicates)
sbatch submit_breakdown_point_source_image_solver_config_sweep_ral_cpu_fp64                     # step 2
# or from the laptop: hpc/sync push, hpc/sync submit --cpu <submit_name>, then hpc/sync pull
```

Step-1 log: [`point_source_cpu_2026_09_26_ral_job_356365_image_plane_p4.out`](point_source_cpu_2026_09_26_ral_job_356365_image_plane_p4.out);
submit [`submit_breakdown_point_source_image_image_plane_p4_ral_cpu_fp64`](../../hpc/batch_cpu/submit_breakdown_point_source_image_image_plane_p4_ral_cpu_fp64).

### Follow-ups

- Fix `image_plane.py`'s step-0 prefix split, so that "ray trace" stops at the deflected vertices,
  not the materialised triangles. Otherwise its dashboard row keeps misattributing the gather.
- There is no A100 row for 4a, because no config became a library default. Phase 4b carries an A100
  no-regression row.
- The completeness draws are the workspace prior plus one stress set. The per-package extents
  (task 4) must be checked on each package's own lenses, which the sanity-check warning (task 3) makes
  visible.
