# Point-source CPU speed-up campaign — ledger

Issue: [autolens_profiling #297](https://github.com/PyAutoLabs/autolens_profiling/issues/297) (phase 1); [PyAutoArray #568](https://github.com/PyAutoLabs/PyAutoArray/issues/568) (phase 2)  
Branch: `feature/point-source-cpu-p1` (phase 1); `feature/point-source-cpu-p2` (phase 2, PyAutoArray + autolens_profiling)  
Status: phase 1 **DONE** (RAL baseline, job 350580); phase 2 **DONE, ACCEPTED** (vertex-dedup A/B, RAL job 350582: 4.47× simple, 1.98× cluster, bit-identical; library change awaiting merge and release); phases 3–4 not started  
Instrument: [`scripts/point_source/likelihood_breakdown/image_plane.py`](../../scripts/point_source/likelihood_breakdown/image_plane.py)
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
python scripts/point_source/likelihood_breakdown/image_plane.py --config-name local_cpu_fp64
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

- [`results/breakdown/point_source/image_plane_hpc_ral_cpu_fp64.json`](../breakdown/point_source/image_plane_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/point_source/image_plane_hpc_ral_cpu_fp64.png)
- [`results/breakdown/cluster/image_plane_hpc_ral_cpu_fp64.json`](../breakdown/cluster/image_plane_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/cluster/image_plane_hpc_ral_cpu_fp64.png)
- the job log [`point_source_cpu_2026_09_23_ral_job_350580.out`](point_source_cpu_2026_09_23_ral_job_350580.out)

The auto-simulate step also wrote a `results/simulators/cluster_summary_v2026.8.17.1.*`
on RAL; it was not harvested. The laptop counterparts are
[`point_source/image_plane_local_cpu_fp64.json`](../breakdown/point_source/image_plane_local_cpu_fp64.json)
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
[`scripts/point_source/likelihood_breakdown/vertex_dedup_ab.py`](../../scripts/point_source/likelihood_breakdown/vertex_dedup_ab.py).

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
  [`vertex_dedup_ab_local_cpu_fp64.json`](../breakdown/point_source/vertex_dedup_ab_local_cpu_fp64.json),
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

  JSON [`vertex_dedup_ab_hpc_ral_a100_fp64.json`](../breakdown/point_source/vertex_dedup_ab_hpc_ral_a100_fp64.json)
  and [`.png`](../breakdown/point_source/vertex_dedup_ab_hpc_ral_a100_fp64.png).
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
python3 -u scripts/point_source/likelihood_breakdown/vertex_dedup_ab.py --config-name local_cpu_fp64
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

- [`results/breakdown/point_source/vertex_dedup_ab_hpc_ral_cpu_fp64.json`](../breakdown/point_source/vertex_dedup_ab_hpc_ral_cpu_fp64.json) and [`.png`](../breakdown/point_source/vertex_dedup_ab_hpc_ral_cpu_fp64.png)
- the job log [`point_source_cpu_2026_09_23_ral_job_350582_vertex_dedup_ab.out`](point_source_cpu_2026_09_23_ral_job_350582_vertex_dedup_ab.out)
  (stderr held only the benign `No blurring_image provided` warning)

The laptop witness is
[`vertex_dedup_ab_local_cpu_fp64.json`](../breakdown/point_source/vertex_dedup_ab_local_cpu_fp64.json)
and [`.png`](../breakdown/point_source/vertex_dedup_ab_local_cpu_fp64.png).
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

## Phase 3 — precompute static initial geometry — NOT STARTED

This phase runs only if deflections dominate after phase 2; they do (70 % of the post-fix FLOPs, see the phase-2 handoff). It would build the
initial lattice's unique vertices and index map in NumPy at construction and
reuse them as immutable JAX inputs, never caching model-dependent deflections.

## Phase 4 — profile the residue and iterate — NOT STARTED

Rank by new evidence: (a) grid-extent guidance, (b) initial scale versus
refinement count, (c) `MAX_CONTAINING_SIZE` / neighbourhood fan-out and
(d) cluster dPIE/NFW deflections. Any PyAutoGalaxy work is its own phase.
