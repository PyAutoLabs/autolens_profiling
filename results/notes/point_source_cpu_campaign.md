# Point-source CPU speed-up campaign — ledger

Issue: [autolens_profiling #297](https://github.com/PyAutoLabs/autolens_profiling/issues/297)  
Branch (phase 1): `feature/point-source-cpu-p1`  
Instrument: [`scripts/point_source/likelihood_breakdown/image_plane.py`](../../scripts/point_source/likelihood_breakdown/image_plane.py)
(shipped in #293, see [point_source_shared_likelihood_breakdown.md](point_source_shared_likelihood_breakdown.md))
and [`scripts/cluster/likelihood_breakdown/image_plane.py`](../../scripts/cluster/likelihood_breakdown/image_plane.py)

This campaign aims to make the JAX-CPU PointSolver image-plane likelihood cheaper
for both the galaxy-scale `simple` case and the 13-component, two-source cluster
case that cluster modelling depends on. It runs as one bounded phase at a time.
Phase 1 (this note, first section) produces a quotable CPU baseline on a quiet
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
  for the shared library change.

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

## Phase 2 — remove the throwaway vertex dedup — NOT STARTED

Test the JAX-only throwaway conversion at `_plane_triangles` against the phase-1
RAL baseline, with the red control and protocol in the handoff above. Ship the
bounded library change, then refresh the workspace rows.

## Phase 3 — precompute static initial geometry — NOT STARTED

This phase runs only if deflections dominate after phase 2. It would build the
initial lattice's unique vertices and index map in NumPy at construction and
reuse them as immutable JAX inputs, never caching model-dependent deflections.

## Phase 4 — profile the residue and iterate — NOT STARTED

Rank by new evidence: (a) grid-extent guidance, (b) initial scale versus
refinement count, (c) `MAX_CONTAINING_SIZE` / neighbourhood fan-out and
(d) cluster dPIE/NFW deflections. Any PyAutoGalaxy work is its own phase.
