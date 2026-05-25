# `vram` — vmap batch_size investigation + per-cell config

This subpackage owns the VRAM-budget logic for the likelihood profiling
sweep on A100 80 GB. Two responsibilities:

1. **Probe** (`vram.probe`) — measure how the compiled vmap program's
   memory footprint scales with batch size, extrapolate the largest batch
   that fits the device's VRAM budget.
2. **Config** (`vram.config`) — the curated per-(dataset, model,
   instrument) batch_size table that runtime cell scripts look up via
   `vmap_batch_for(...)`.

## Why a separate subpackage?

The runtime cell scripts (`likelihood_runtime/{imaging,interferometer,
datacube}/*.py`) used to hard-code `batch_size = 3` everywhere. Production
sampling uses much larger batches (≥ 10), and the right batch size varies
with model/instrument (a 700-px mask AO dataset needs a smaller batch
than a 70-px Euclid one). Splitting the logic out:

- Keeps each runtime cell terse — one import + one call to
  `vmap_batch_for(...)`.
- Centralises the probe / extrapolation math so it can be unit-tested.
- Makes the per-(cell, instrument) batch_size table reviewable as data,
  not as scattered constants.

## Public API

From `vram`:

| Name | Purpose |
|------|---------|
| `vmap_batch_for(dataset, model, instrument)` | Return per-cell batch_size (or `None` if vmap is intentionally skipped or the cell hasn't been probed). |
| `probe_vmap_memory(func, args, batch_sizes=(2, 4))` | JIT-vmap `func` at each batch, read `compiled.memory_analysis()`, return a `ProbeResult`. |
| `recommend_batch_size(probe, vram_budget_gb=70.0, max_batch=64)` | Linear extrapolation → max batch fitting in budget. |
| `write_probe_json(probe, recommended, path)` | Serialise probe + recommendation to JSON. |

## How `probe_vmap_memory` works

1. For each batch size `B` in `batch_sizes` (default `(1, 4)`):
   1. Broadcast each leaf of `args_pytree` along a new leading axis of size `B`.
   2. `jax.jit(jax.vmap(func))(parameters).lower().compile()`.
   3. Read `compiled.memory_analysis().peak_memory_in_bytes` (preferred, on
      modern JAX) or fall back to `output_size + temp_size` (older JAX).
2. Fit a linear model: `peak_mb ≈ overhead + B * per_replica`.
3. `recommend_batch_size` returns
   `floor((budget - safety_factor * overhead) / (safety_factor * per_replica))`,
   capped at `max_batch` (default 64).

**Why `peak_memory_in_bytes` and not `output + temp`?** Those are sequential
phases — peak is the actual maximum simultaneous allocation including XLA
rematerialisation. Summing output+temp double-counts buffer reuse and over-
reports memory. peak is what XLA actually allocates on device.

## Methodology — A100 80 GB budget

- Hard ceiling: **80 GB** on the RAL A100s.
- Soft budget (default): **65 GB**. Leaves ~15 GB for JAX runtime overhead,
  CUDA driver allocations, allocator fragmentation, and per-call activation
  slack that static analysis doesn't fully account for.
- **Safety factor: 1.15×** on `per_replica_mb`. The XLA static estimate
  typically under-counts the real runtime peak by ~10-15% on complex graphs.
- Cap: **64**. XLA compile time scales superlinearly with batch on some
  cells (notably `delaunay`, due to scipy-Delaunay-via-`pure_callback`
  inflating the XLA graph). 64 is roughly where diminishing returns kick in
  for production sampling.

All defaults are configurable via kwargs on `recommend_batch_size` /
`probe_vmap_memory` if a different device or workload needs different limits.

## Batch_sizes selection per cell

XLA recompiles for each new batch_size (no cache reuse). Compile cost varies:

- **mge / pixelization** — ~10 s/compile. Use a multi-point fit:
  `batch_sizes=(1, 4, 16)` catches the ~8/16/32 rematerialisation phase
  transitions JAX may exhibit.
- **delaunay** — 10-30 min/compile on big graphs (scipy.Delaunay via
  `pure_callback` bloats the XLA graph). Use single-point:
  `batch_sizes=(1,)` and accept linear extrapolation. If the chosen batch
  OOMs at run time, manually re-probe at a smaller batch.

## Adding a new instrument

1. Add an entry to the appropriate `INSTRUMENTS` dict in
   `simulators/{imaging,interferometer}.py`.
2. Create a probe SLURM script at
   `z_projects/profiling/hpc/batch_gpu/probe_vmap_<cell>_<instrument>`
   (clone any existing one).
3. Run the probe; pull the resulting JSON to
   `output/runtime/<cell>/<instrument>/vmap_probe.json`.
4. Read `recommended_batch_size` and add the row to `VMAP_BATCH` in
   `vram/config.py`.
5. Re-run the regular profile to confirm the chosen batch holds at
   steady state (vmap completes, doesn't OOM).
