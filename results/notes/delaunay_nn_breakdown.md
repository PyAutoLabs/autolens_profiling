# DelaunayNN vs Delaunay — A100 likelihood breakdown (2026-09-05)

## Verdict

Swapping barycentric `Delaunay` for Sibson natural-neighbour `DelaunayNN` (cap 32) in
an otherwise identical HST / Hilbert-1500 / MGE-60 / `ConstantSplit` imaging likelihood
costs **2.18×** on the A100 when every step is timed one at a time
(256.4 ms vs 117.4 ms step-sum), and the whole of that gap is geometry: the
"Triangulation + interpolation" row goes 30.0 ms → 122.0 ms (4.07×) and the
regularization-matrix row, which reads the same interpolator's split-point walk,
goes 10.4 ms → 59.8 ms. Under `jax.jit(jax.vmap(...))` at batch 16 — the batch the
Nautilus reference runs use — both collapse: triangulation amortises 30.0 → 6.7 ms
(Delaunay) and 122.0 → 13.4 ms (DelaunayNN), a 1.99× ratio, and substituting the two
vmap-timed rows into the step-sums narrows the whole-likelihood gap to **1.26×**
(78.2 ms vs 98.4 ms per call, a 20.3 ms penalty). That is why Nautilus sees roughly
the same ~61 ms per evaluation for both meshes while the mapper-only geometry
benchmark reads 4.3×: the geometry benchmark measures the unbatched row, and
production never runs it unbatched. This note also corrects the
"Regularization matrix (H)" row, which before 2026-09 timed a host-to-device copy
rather than a JIT step.

## Provenance

| Job | Cell | Node | Start (2026-09-05) | Elapsed | State | Artifact |
|---|---|---|---|---:|---|---|
| 342277 | `likelihood_breakdown/imaging/delaunay` (`--split-setup --vmap-batch 16`) | `euclid-ral-gpu-1` | 17:25:32 | 1:33 | COMPLETED | `results/breakdown/imaging/delaunay_hpc_a100_fp64.{json,png}` |
| 342278 | `likelihood_breakdown/imaging/delaunay_nn` (`--split-setup --vmap-batch 16`) | `euclid-ral-gpu-2` | 17:28:28 | 2:27 | COMPLETED | `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64.{json,png}` |
| 342279 | `likelihood_breakdown/imaging/delaunay_nn` (`--vmap-batch 64`) | `euclid-ral-gpu-1` | 17:25:32 | 2:06 | COMPLETED (vmap phase OOM, caught) | `results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_vmap64.{json,png}` |
| 342280 | `likelihood_runtime/imaging/delaunay_nn` (probe → full) | `euclid-ral-gpu-2` | 17:25:33 | 2:55 | **FAILED** (cuFFT at probe-recommended batch 64) | `results/runtime/imaging/delaunay_nn/vmap_probe_delaunay_nn_batch64_cufft_failed.json` |
| 342281 | `likelihood_runtime/imaging/delaunay_nn` (full only, no probe) | `euclid-ral-gpu-2` | 17:40:28 | 1:27 | COMPLETED | `results/runtime/imaging/delaunay_nn/delaunay_nn_hpc_a100_fp64.json` |

- Hardware: NVIDIA A100 80GB PCIe, RAL `gpu` partition; fp64 (`JAX_ENABLE_X64=True`),
  dense inversion path (`InversionImagingMapping`).
- `PyAutoLens 2026.8.17.1`; `jax 0.10.2` / `jaxlib 0.10.2`; Python 3.12.
- Library source revisions, from the jobs' `=== source revisions ===` block (identical
  across all five jobs):

  | Repo | Revision |
  |---|---|
  | PyAutoNerves | `fc9c474ba36cca0aaf0d40351e65e791b99f9a35` |
  | PyAutoFit | `cdda28b5fe953b674b66cb33d3af484c37b5d994` |
  | PyAutoArray | `e36a5af4c2c86392d2a835c5422ec721818b097b` |
  | PyAutoGalaxy | `6d216c151c914b2ce79af18fbc0348a6fc5425d7` |
  | PyAutoLens | `146a3d7254418eac964edd8b89ada7281ac778e7` |

- RAL worktree: `/mnt/ral/jnightin/autolens_profiling_wt/delaunay-nn-breakdown` on
  `feature/delaunay-nn-breakdown`; jobs 342277–342280 at `8648294`, job 342281 at
  `c5c49c4`.
- Eager log-evidence pins held on the A100 for both cells: **29110.920858** (Delaunay,
  342277) and **29144.581944** (DelaunayNN, 342278 / 342279 / 342281; the runtime cell
  records `pinned_drift: []`).
- **The `euclid-ral-gpu-1` exclusion was dropped on these submits.** MIG mode was
  confirmed `Disabled` on all four GPUs on 2026-09-05 and JAX CUDA init was verified per
  GPU from inside an `srun`; jobs landed on both `gpu-1` (342277, 342279) and `gpu-2`
  (342278, 342280, 342281) and every one produced valid, pin-passing numbers. The
  per-job preflight stays as the backstop; fleet-wide retirement of the exclusion is a
  separate follow-up (`PyAutoMind/draft/maintenance/autolens_profiling/retire_gpu1_mig_exclusion.md`).
- Recorded `xla_flags` on these rows are
  `--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0`. The
  2026-07-10 A100 rows in `preopt_breakdown_baseline.md` carried only the first of the
  two; the Curvature matrix (F) row differs by 4.8 → 25.6 ms between them, which is the
  most visible consequence of comparing across that env difference. Compare like with
  like inside this note, not across notes.

## Like-for-like per-step table

All values ms per likelihood call. "vmap/16" is
`jax.jit(jax.vmap(fn))` over a params pytree broadcast to batch 16, reported as
batch time / 16. Only the combined inversion-setup block, the four `--split-setup`
prefixes and the params→H prefix are re-timed under vmap (that is what `--vmap-batch`
covers); the remaining rows are eager-JIT single-call timings in both columns and are
shown as `—` in the vmap columns, not as zero.

| Step | Delaunay | Delaunay vmap/16 | DelaunayNN | DelaunayNN vmap/16 |
|---|---:|---:|---:|---:|
| Ray-trace data grid | 0.175 | — | 0.190 | — |
| Ray-trace mesh grid | 0.144 | — | 0.155 | — |
| Lens light images (pre-PSF) | 0.122 | — | 0.139 | — |
| Blurred image (PSF convolution) | 0.852 | — | 0.841 | — |
| Profile-subtracted image | 0.130 | — | 0.139 | — |
| **Inversion setup (steps 5–8 combined)** | **45.078** | **15.502** | **136.225** | **22.192** |
| Data vector (D) | 0.307 | — | 0.342 | — |
| Curvature matrix (F) | 25.631 | — | 25.594 | — |
| **Regularization matrix (H)** | **10.412** | **0.746** | **59.823** | **15.839** |
| Regularized reconstruction | 32.321 | — | 30.687 | — |
| Mapped recon + log evidence | 2.226 | — | 2.305 | — |
| **Total step-by-step (unbatched)** | **117.398** | — | **256.440** | — |

Everything outside the inversion setup and H rows is mesh-independent to within
measurement scatter (≤ 9% on rows under 1 ms, ≤ 5% on F and the reconstruction), as it
should be: the two cells differ only in the mapper class.

### Four-way setup split (`--split-setup`)

| Piece | Delaunay | Delaunay vmap/16 | DelaunayNN | DelaunayNN vmap/16 |
|---|---:|---:|---:|---:|
| Border relocation | 1.078 | 0.064 | 1.080 | 0.069 |
| **Triangulation + interpolation** | **29.977** | **6.732** | **121.993** | **13.413** |
| Mapping matrix | 0.734 | 0.030 | −0.605 | 0.216 |
| Blurred mapping matrix (PSF) | 8.173 | 8.372 | 9.381 | 8.262 |
| *Prefix-sum* | *39.962* | *15.198* | *131.849* | *21.960* |
| *Combined block (for comparison)* | *45.078* | *15.502* | *136.225* | *22.192* |
| **Regularization matrix (H)** | **10.412** | **0.746** | **59.823** | **15.839** |

The prefix-sum tracks the combined block to 11% (Delaunay, unbatched), 3%
(DelaunayNN, unbatched) and 1–2% under vmap — no large fusion redistribution.
The **negative −0.605 ms DelaunayNN "Mapping matrix" row** is the by-difference
artifact of two ~132 ms nested prefixes measured 0.5% apart; read it as "below the
noise floor of this decomposition", which is consistent with the +0.734 ms Delaunay
value and with the +0.216 ms it reads under vmap, where the prefixes are ten times
cheaper and the difference resolves.

The one row that does **not** amortise is **Blurred mapping matrix (PSF)**: 8.173 →
8.372 ms (Delaunay) and 9.381 → 8.262 ms (DelaunayNN), i.e. flat per call at batch 16.
It is already a dense FFT-convolution over the full mapping matrix and was saturating
the device at batch 1.

## The H-row correction

**What the old row measured.** Before 2026-09 the "Regularization matrix (H)" row timed
`jnp.array(inversion.regularization_matrix)` — a host-to-device copy of the 1560×1560
fp64 matrix (1500 mesh vertices + 60 MGE linear light profiles; 19.5 MB) that the
*eager NumPy* `FitImaging` had already computed. On the A100 that is ~19.5 MB of PCIe
traffic and no JIT work at all: **14.4 ms** in the 2026-07-10 A100 rows
(`preopt_breakdown_baseline.md`, jobs 330062–330070, recorded in the superseded
`delaunay_hpc_a100_fp64.json` at `v2026.7.6.649`).

**What it measures now.** A prefix function `params_tree → regularization matrix` is
compiled — tracer from the params pytree, traced grids, border relocation,
`Interpolator*`, `Mapper`, then
`ConstantSplit.regularization_matrix_from(linear_obj=mapper, xp=jnp)` — and the row is
reported as the difference

```
t(params -> H) - t(params -> interpolator outputs)
```

where the interpolator prefix is exactly `_setup_prefix_fn(6)`, the same prefix that
terminates "Triangulation + interpolation" in the four-way table. Subtracting it avoids
double-charging the mesh build; the two absolute prefix times are kept in the JSON as
`regularization_matrix_prefix_s` / `interpolator_prefix_s` (and
`*_vmap_per_call_s` under `--vmap-batch`).

**Why the prefix must nest.** `ConstantSplit` reads the interpolator's *split-point*
walk and never the per-query interpolation, so a prefix returning H alone lets XLA
dead-code-eliminate the entire query side of the mapper — the two prefixes stop nesting
and their difference goes negative. Measured on local CPU (DelaunayNN/HST, 2026-09-05)
that gave a 215 ms H prefix against a 391 ms interpolator prefix: a nonsensical
**−175 ms** row. `_setup_prefix_fn(11)` therefore returns the step-6 outputs *alongside*
H, making it a strict superset of `_setup_prefix_fn(6)` and the subtraction meaningful.

**The new values.**

| | H prefix | interpolator prefix | **H row (difference)** |
|---|---:|---:|---:|
| Delaunay, unbatched | 41.467 | 31.055 | **10.412** |
| Delaunay, vmap/16 per call | 7.542 | 6.796 | **0.746** |
| DelaunayNN, unbatched | 182.896 | 123.073 | **59.823** |
| DelaunayNN, vmap/16 per call | 29.321 | 13.482 | **15.839** |

So the honest A100 Delaunay H row is 10.4 ms of real JIT work, not the 14.4 ms of PCIe
the old row reported — a number that happened to be the same order of magnitude and
therefore never looked wrong. Under vmap it falls to 0.75 ms, i.e. H is essentially free
in production for barycentric Delaunay. For DelaunayNN it is the *dominant* remaining
per-call penalty (15.8 ms of the 20.3 ms total, below), because the Sibson split-point
walk over the split-cross points (four per mesh vertex) is as expensive as the main query walk.

The downstream "Regularized reconstruction" and "Mapped recon + log evidence" steps
still consume the inversion's own `regularization_matrix`, so every correctness
assertion and both eager pins are unchanged by this correction.

## Reconciliation

### (a) The mapper-only geometry benchmark vs the breakdown's triangulation row

`results/notes/delaunay_nn_cap_audit.md` (A100, job 334949, 1,200 vertices, 15,000
queries, query chunk 256, ten warm repeats) records **36.6 ms** for barycentric Delaunay
and **157.3 ms** for DelaunayNN cap 32 — a **4.30×** overhead.

This breakdown's *unbatched* "Triangulation + interpolation" rows (1,500 vertices,
15,361 masked / 17,980 over-sampled queries) read **29.977 ms** and **121.993 ms** — a
**4.07×** overhead. The two agree in ratio to 5% and in absolute scale to ~20–30%, which
is what the different vertex and query counts predict. **The geometry benchmark is a
faithful predictor of the unbatched row**, and it was never measuring anything else.

### (b) Why Nautilus sees ~61 ms per evaluation for both meshes

Because Nautilus evaluates in batches of 16, and the batched columns are a different
story:

| | Delaunay | DelaunayNN | ratio |
|---|---:|---:|---:|
| Triangulation + interpolation, unbatched | 29.977 | 121.993 | 4.07× |
| Triangulation + interpolation, vmap/16 per call | 6.732 | 13.413 | **1.99×** |
| Inversion setup block, vmap/16 per call | 15.502 | 22.192 | 1.43× |
| Regularization matrix (H), vmap/16 per call | 0.746 | 15.839 | 21.2× |

Per-call **DelaunayNN penalty at batch 16**, computed from the two JSONs
(`steps_vmap_per_call` differences):

- Inversion setup block: 22.192 − 15.502 = **6.690 ms**
- Regularization matrix (H): 15.839 − 0.746 = **15.093 ms**
- **Total: 21.783 ms per call**

Substituting the two vmap-timed rows into each cell's step-sum (a hybrid figure — the
remaining rows are unbatched timings, and the eight mesh-independent rows contribute
~62 ms to both):

| | step-sum with vmap/16 setup+H rows |
|---|---:|
| Delaunay | 117.398 − 45.078 − 10.412 + 15.502 + 0.746 = **78.156 ms** |
| DelaunayNN | 256.440 − 136.225 − 59.823 + 22.192 + 15.839 = **98.423 ms** |
| ratio | **1.26×** (penalty **20.267 ms/call**) |

The unbatched 2.18× gap becomes 1.26× at batch 16. Against the ~61 ms per-evaluation
Nautilus cost, a 20 ms penalty on a ~78 ms base is within the run-to-run spread of the
sampler's own per-evaluation accounting — which is exactly the observation this note
set out to explain. **The mesh choice is not a 4× tax in production; it is a ~20 ms
per-call tax, three quarters of which is the split-point regularization walk.**

### (c) The full-pipeline runtime cell

| | Delaunay | DelaunayNN |
|---|---:|---:|
| Single JIT, per call | *no A100 row* | **250.284 ms** (342281) |
| vmap/16, per call | *no A100 row* | **82.204 ms** (342281) |
| vmap speedup vs single JIT | — | **3.0×** |

`results/runtime/imaging/delaunay/hst/` holds only `local_cpu_fp64`, `local_cpu_mp` and
their `_sparse` variants — **there is no committed A100 row for the Delaunay runtime
cell**, so the head-to-head at the full-pipeline level is not yet available and this
table has one column. Filing it is the obvious companion run.

The DelaunayNN full-pipeline single-JIT number reproduced across jobs: 250.654 ms and
254.025 ms in the two phases of the failed 342280, 250.284 ms in 342281 — a 1.5% spread.
The vmap/16 per-call 82.204 ms sits 16% *below* the 98.4 ms hybrid step-sum estimate of
§(b), which is the expected direction: the full pipeline is compiled as one program and
XLA fuses across step boundaries that the breakdown deliberately cuts. Treat the
step-sums as an upper bound on, and an attribution of, the fused cost — never as the
fused cost itself.

## The vmap-64 record (job 342279)

The `--vmap-batch 64` job wrote every unbatched row before its batched phase died, so it
is a free replicate of 342278 on a different node (`gpu-1` vs `gpu-2`):

| Row | 342278 (`gpu-2`) | 342279 (`gpu-1`) | Δ |
|---|---:|---:|---:|
| Inversion setup (5–8) | 136.225 | 136.587 | +0.3% |
| Triangulation + interpolation | 121.993 | 123.397 | +1.2% |
| Border relocation | 1.080 | 1.056 | −2.3% |
| Mapping matrix | −0.605 | −0.804 | (both sub-noise) |
| Blurred mapping matrix (PSF) | 9.381 | 9.223 | −1.7% |
| H prefix | 182.896 | 179.265 | −2.0% |
| Interpolator prefix | 123.073 | 124.452 | +1.1% |
| Regularization matrix (H) | 59.823 | 54.813 | −8.4% |
| Curvature matrix (F) | 25.594 | 25.735 | +0.6% |
| **Total step-by-step** | **256.440** | **251.661** | **−1.9%** |

Everything agrees within 2.5%. The one apparent outlier, the 8.4% H row, is arithmetic
rather than physics: it is the difference of two ~180 ms and ~124 ms prefixes that
individually agree to 2.0% and 1.1%, and a 2% wobble on the larger term is 3.6 ms on a
~57 ms difference. **Both nodes produce the same numbers** — the `gpu-1` MIG worry does
not show up in the measurements.

The batched phase OOM'd, caught by the script's guard and recorded in the JSON's
`vmap_error`:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to
allocate 46.86GiB. [executable_name='jit_fn'] [tf-allocator-allocation-error='']
```

preceded in the job's `.err` by XLA's rematerialization pass giving up:

```
W0905 17:27:21 hlo_rematerialization.cc:3231] Can't reduce memory use below
57.03GiB (61237292825 bytes) by rematerialization; only reduced to 68.32GiB
(73361280968 bytes), down from 68.32GiB (73361280968 bytes) originally
```

— i.e. rematerialization recovered *nothing* (68.32 GiB before, 68.32 GiB after)
against a 57.03 GiB target, and the allocator then failed the single 46.86 GiB request.

**Batch 64 on the dense path is out of reach on 80 GB for this cell.** The dense
per-replica footprint measured by the probes is **921.9 MB/replica** for Delaunay
(`results/runtime/imaging/a100_probes/vmap_probe_delaunay.json`) and 771.0 MB/replica
for DelaunayNN; 64 replicas is ~49–59 GB of live mapping matrices before any transient,
which is the same order as the 46.86 GiB the allocator refused. Batch 16 is the right
production number and is what the `VMAP_BATCH` table already carries for both cells.

If larger batches are ever wanted here, the **sparse** (w-tilde operator) path is the
route: `vmap_probe_delaunay_sparse.json` measures **131.5 MB/replica**, seven times
lighter, and the probe recommends the full cap of 64 on it. That is a different
inversion path with its own accuracy story (`sparse_vs_dense_inversion_path.md`), not a
free switch.

## Runtime probe trap: cuFFT at the probe-recommended batch

Job 342280 (the first attempt at the DelaunayNN runtime cell) ran the standard two-phase
`--vmap-probe` → full pattern. Phase A sampled **batch 1 only**, extrapolated
**771.0 MB/replica**, and against a 65 GB budget with a 1.15 safety factor recommended
its cap of **64**. `resolve_vmap_batch` prefers a fresh matching probe over the table,
so Phase B ran at 64 rather than the table's 16 — and died in cuFFT, not in the
allocator:

```
jax.errors.JaxRuntimeError: INTERNAL: RET_CHECK failure
(external/xla/xla/backends/gpu/runtime/fft_thunk.cc:200) fft_plan != nullptr
Failed to create cuFFT batched plan with scratch allocator
[executable_name='jit_full_pipeline_from_params']
```

This is the failure mode `scripts/misc/vram/config.py` already documents inline above
the `VMAP_BATCH` table — *"probe-recommended sizes halved for hst/jwst/ao after cuFFT
scratch-allocator failures at the probe-predicted batch. The static `memory_analysis()`
doesn't account for cuFFT batched-plan scratch"* — and the `("imaging", "delaunay", "jwst")`
row's *"probe said 23, cuFFT failed"*. The DelaunayNN cell is now a fourth instance.

**Workaround used** (job 342281): `--vmap-probe` was removed from
`hpc/batch_gpu/submit_runtime_imaging_delaunay_nn_a100_hst_fp64` with a comment
explaining why, and the probe JSON on RAL was renamed to
`vmap_probe_delaunay_nn_batch64_cufft_failed.json` so `resolve_vmap_batch` finds no
probe and falls back to the table's `("imaging", "delaunay_nn", "hst"): 16`. The job log
confirms `vmap batch_size: 16 (source: table)`. The failed probe is committed under its
renamed path as the evidence.

**Recommended, not implemented** (a follow-up, deliberately out of scope here — changing
`resolve_vmap_batch` semantics mid-campaign would invalidate every other cell's
provenance): either cap the probe recommendation at the table value when a table row
exists, so the probe can only ever *lower* the batch, or give the runtime cells a
`--vmap-batch N` flag matching the breakdown cells' so a submit can pin the batch
without deleting its probe. The first is the smaller change and matches what the table's
comments already say the humans do by hand.

## Environment note: `/home` quota on RAL

Both `gpu-1` jobs (342277, 342279) logged, once per compiled program:

```
jax/_src/compiler.py:833: UserWarning: Error writing persistent compilation cache
entry for 'jit_fn': OSError: [Errno 28] No space left on device:
'/home/jnightin/.cache/pyauto_jax/jit_fn-<hash>-cache'
```

These are **warnings only** — JAX compiles normally and simply fails to persist the
artifact, so the results in this note are unaffected (the pins passed and the numbers
replicate 342278 within 2.5%). The cost is compile time on subsequent runs. The
`/home/jnightin` quota needs clearing; `/home` is small on RAL and the JAX persistent
cache under `~/.cache/pyauto_jax/` is the obvious first thing to prune or relocate to
`/mnt/ral/jnightin`.

## What this means for the optimisation plan

The two latency-bound targets are the **unbatched "Triangulation + interpolation" row**
(30.0 ms Delaunay, 122.0 ms DelaunayNN) and the **H row** that shares its walk (10.4 ms
and 59.8 ms) — together 34% and 71% of their cells' step-sums. Under vmap at batch 16
both amortise hard (6.7 / 13.4 ms and 0.7 / 15.8 ms), so any optimisation aimed at
production throughput must be measured batched or it will chase a cost the sampler does
not pay. The residual that *does* survive batching is DelaunayNN's H row, 15.8 ms per
call and 74% of the mesh's 20.3 ms production penalty: the `ConstantSplit` split-point
walk over the split-cross points (four per mesh vertex), not the main query walk. The point-location
follow-up is
`PyAutoMind/draft/feature/autoarray/delaunay_walk_early_exit_unchunked.md`.

## XLA GPU autotuning A/B (2026-09-05)

`PyAutoNerves/autonerves/jax_wrapper.py` (lines 58–73, introduced in PyAutoNerves
e8d5842, 2026-07-17) appends `--xla_gpu_autotune_level=0` to `XLA_FLAGS` unless the
variable already names a level — the stated rationale being that autotuning dominates
cold GPU compile time "while giving no measurable evaluation speed-up on PyAuto
likelihoods". Every 2026-09-05 A100 job above therefore ran with autotuning **off**,
while the 2026-07-10 A100 tier (`preopt_breakdown_baseline.md`) predates the Nerves
change and ran with autotuning at XLA's default. Between the two tiers the Curvature
matrix (F) row went 4.82 → 25.63 ms while the Cholesky-bound rows did not move; F is a
single dense fp64 GEMM, `(15361 × 1560)ᵀ (15361 × 1560)`. This section is the controlled
A/B that tests whether that is the autotuner.

Both arms are the same commit, the same `autolens 2026.8.17.1`, the same
`configuration` block (15361 masked pixels, 17980 over-sampled, 1500 vertices, dense
path), the same submit scripts bar three lines, and differ only in `XLA_FLAGS`.
Setting the level explicitly is respected by the Nerves wrapper, and
`device_info_dict()` reads `XLA_FLAGS` from the environment at run time, so each JSON
records which arm it is.

| Job | Cell | Arm (`XLA_FLAGS` autotune level) | Node | Start (2026-09-05) | Elapsed | State |
|---|---|---|---|---|---:|---|
| 342277 | `likelihood_breakdown/imaging/delaunay` | **0** (Nerves default) | `euclid-ral-gpu-1` | 17:25:32 | 1:33 | COMPLETED |
| 342282 | `likelihood_breakdown/imaging/delaunay` | **4** (XLA default) | `euclid-ral-gpu-2` | 18:02:52 | 1:42 | COMPLETED |
| 342281 | `likelihood_runtime/imaging/delaunay_nn` | **0** | `euclid-ral-gpu-2` | 17:40:28 | 1:27 | COMPLETED |
| 342283 | `likelihood_runtime/imaging/delaunay_nn` | **4** | `euclid-ral-gpu-2` | 18:04:35 | 1:32 | COMPLETED |

The breakdown pair is cross-node (`gpu-1` vs `gpu-2`); both are `NVIDIA A100 80GB PCIe`
and the 2026-09-05 rows above already replicate across the two nodes to within 2.5%.
The runtime pair is same-node.

### Breakdown cell — per-step (ms per likelihood call, unbatched)

| Step | autotune 0 (342277) | autotune 4 (342282) | Δ | ratio |
|---|---:|---:|---:|---:|
| Ray-trace data grid | 0.175 | 0.170 | −0.005 | 1.03× |
| Ray-trace mesh grid | 0.144 | 0.200 | +0.056 | 0.72× |
| Lens light images (pre-PSF) | 0.122 | 0.122 | +0.000 | 1.00× |
| Blurred image (PSF convolution) | 0.852 | 0.853 | +0.001 | 1.00× |
| Profile-subtracted image | 0.130 | 0.153 | +0.023 | 0.85× |
| Inversion setup (steps 5–8 combined) | 45.078 | 49.502 | +4.424 | 0.91× |
| Data vector (D) | 0.307 | 0.327 | +0.020 | 0.94× |
| **Curvature matrix (F)** | **25.631** | **4.902** | **−20.729** | **5.23×** |
| Regularization matrix (H) | 10.412 | 10.096 | −0.316 | 1.03× |
| Regularized reconstruction | 32.321 | 32.584 | +0.263 | 0.99× |
| Mapped recon + log evidence | 2.226 | 2.213 | −0.013 | 1.01× |
| **Total step-by-step** | **117.398** | **101.125** | **−16.273** | **1.16×** |

### Breakdown cell — four-way setup split (ms per call)

| Piece | autotune 0 | autotune 4 | Δ | autotune 0 vmap/16 | autotune 4 vmap/16 |
|---|---:|---:|---:|---:|---:|
| Border relocation | 1.078 | 1.071 | −0.007 | 0.064 | 0.061 |
| Triangulation + interpolation | 29.977 | 30.408 | +0.431 | 6.732 | 6.626 |
| Mapping matrix | 0.734 | 3.462 | +2.728 | 0.030 | 0.162 |
| **Blurred mapping matrix (PSF)** | **8.173** | **4.950** | **−3.223** | 8.372 | 8.324 |
| *Prefix-sum* | *39.962* | *39.891* | *−0.071* | *15.198* | *15.173* |
| *Combined block* | *45.078* | *49.502* | *+4.424* | *15.502* | *17.926* |
| Regularization matrix (H) | 10.412 | 10.096 | −0.316 | 0.746 | 0.866 |

Two rows move by more than measurement scatter and they move in opposite directions:
the PSF-blurred mapping matrix gains 3.22 ms (8.17 → 4.95 ms, below even the 6.10 ms
of the 2026-07-10 tier) and the by-difference "Mapping matrix" row loses 2.73 ms. Their
prefix-sum is flat to 0.07 ms (39.96 vs 39.89), so the pair is a redistribution inside
the nested prefixes, not a real change; the combined-block row rising 4.42 ms against a
flat prefix-sum is fusion in the fused program and is the same order as the 11%
prefix-vs-combined gap already recorded above. The vmap/16 rows are flat throughout
(≤ 0.12 ms), including F's consumers.

### Breakdown cell — compile-time cost of autotuning (seconds)

| Timer | autotune 0 | autotune 4 | Δ |
|---|---:|---:|---:|
| **`curvature_matrix_jit_compile`** | **0.3789** | **1.7904** | **+1.4115** |
| `inversion_setup_jit_compile` | 12.7481 | 12.9464 | +0.1983 |
| `regularization_matrix_jit_compile` | 3.1960 | 3.1343 | −0.0617 |
| `reconstruction_jit_compile` | 0.4596 | 0.6321 | +0.1725 |
| `log_evidence_jit_compile` | 0.2183 | 0.3780 | +0.1597 |
| *sum of all `_compile` timers* | *23.602* | *25.189* | *+1.588* |
| *sum of all `_lower` timers* | *5.885* | *6.195* | *+0.309* |
| *sum of all `vmap16_first_call` timers* | *28.370* | *30.695* | *+2.325* |
| *SLURM wall clock* | *1:33* | *1:42* | *+0:09* |

The compile cost is concentrated in exactly the step that gained: compiling the
curvature-matrix program costs +1.41 s once and returns 20.73 ms on every call, a
break-even at ~68 calls. Nothing else's compile time moves by more than 0.2 s.

### Runtime cell (`likelihood_runtime/imaging/delaunay_nn`, dense)

| Quantity | autotune 0 (342281) | autotune 4 (342283) | Δ | ratio |
|---|---:|---:|---:|---:|
| Full pipeline, single JIT (ms/call) | 250.284 | 229.176 | −21.108 | 1.09× |
| vmap/16 per call (ms/call) | 82.204 | 62.873 | −19.331 | 1.31× |
| vmap batch time (s, batch 16) | 1.3153 | 1.0060 | −0.3093 | 1.31× |
| vmap speed-up vs single JIT | 3.0× | 3.6× | — | — |
| `full_pipeline_lower` (s) | 5.180 | 3.796 | −1.383 | — |
| `full_pipeline_compile` (s) | 19.719 | 22.289 | +2.571 | — |
| `full_pipeline_first_call` (s) | 0.367 | 0.341 | −0.025 | — |
| `vmap_first_call` (s) | 23.595 | 31.185 | +7.590 | — |
| *one-off total (lower + compile + first calls)* | *48.861* | *57.611* | *+8.750* | — |
| *SLURM wall clock* | *1:27* | *1:32* | *+0:05* | — |

Break-even for the runtime cell is +8.75 s of one-off cost against 19.33 ms saved per
batched call: ~453 likelihood evaluations, i.e. ~29 vmap batches of 16.

### Pins

Both arms of both cells reproduce their pinned figures bit-identically. The breakdown
runs report `figure_of_merit (log_evidence) = 29110.920857378314` in both arms and both
log `Eager regression assertion PASSED: log_evidence matches 29110.920858` (the
step-by-step recomputation agrees to the 12th significant figure: 29110.920857389552 vs
29110.920857389550). The runtime runs both report
`figure_of_merit (log_evidence) = 29144.581943885576`, both log
`Pinned-value check PASSED`, and both JSONs carry `"pinned_drift": []` against
`"pinned_expected": 29144.581943885652`.

### Verdict

Re-enabling XLA GPU autotuning restores the curvature-matrix GEMM: F falls 25.63 →
4.90 ms (5.23×), back to the 4.82 ms of the 2026-07-10 A100 tier, while every
Cholesky-bound row stays flat, the unbatched step-sum falls 117.40 → 101.13 ms and the
DelaunayNN runtime cell falls 250.3 → 229.2 ms single-JIT and 82.20 → 62.87 ms per
vmap/16 call. The cost is one-off compile time — +1.41 s on the curvature-matrix
program alone, +1.59 s across all breakdown compiles and +8.75 s across the runtime
cell's lower/compile/first-call phases — so the flag pays for itself after ~68 calls of
the F step and ~453 batched likelihood evaluations respectively.

## See also

- `results/notes/preopt_breakdown_baseline.md` — the 2026-07-10 A100 tier this note
  supersedes for the Delaunay rows.
- `results/notes/delaunay_nn_cap_audit.md` — the cap-32 decision and the mapper-only
  geometry benchmark reconciled in §(a).
- `results/notes/sparse_vs_dense_inversion_path.md` — the sparse path referenced in the
  vmap-64 record.
- `scripts/misc/vram/config.py` — the `VMAP_BATCH` table and its cuFFT provenance notes.
