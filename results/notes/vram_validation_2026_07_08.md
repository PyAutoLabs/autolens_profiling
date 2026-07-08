# Phase-2 vram-first validation sweep — 2026-07-08

Local CPU validation of every PreOptimizationTimes campaign cell **before**
A100 time is spent, per the campaign's vram-first rule. Tracked by
[autolens_profiling#54](https://github.com/PyAutoLabs/autolens_profiling/issues/54);
design context: [`design_lock_in.md`](./design_lock_in.md). PyAutoLens
2026.7.6.649, JAX CPU backend, cheapest instrument per cell; sweep artifacts
were scratch-tier (validation, not campaign data) — this note is the durable
record.

## Verdict

**All 9 campaign cells run clean end-to-end on current source** (after the
fixes below). The A100 probe leg (`hpc/batch_gpu/submit_probe_{fast,delaunay}_a100`)
can be dispatched without expecting functional surprises.

| Cell | Instrument | Path | Wall time (CPU) |
|------|-----------|------|-----------------|
| imaging/mge | euclid | dense | 47 s |
| imaging/pixelization | euclid | dense | 401 s |
| imaging/pixelization | euclid | sparse | 194 s |
| imaging/delaunay | euclid | dense | 381 s |
| imaging/delaunay | euclid | sparse | 213 s |
| interferometer/mge | sma | dense | 23 s |
| interferometer/pixelization | sma | dense | 70 s |
| interferometer/delaunay | sma | dense | 124 s |
| datacube/delaunay | sma | — | 265 s |

Also validated live: the `--vmap-probe` pipeline end-to-end on CPU, and the
probe-JSON-over-table resolution (a fresh probe in the cell output dir won,
`source: probe (vmap_probe_mge.json)` in the log).

## Defects flushed (all fixed on the phase-2 branch)

1. **A100 batch table applied on every backend.** Runtime cells vmapped at
   table batch (64) on laptop CPU → OOM-kill (exit 137, pixelization dense at
   97 s); the probe's 65 GB budget default is A100-specific too, so a CPU
   probe *also* recommended 64. Fix: `resolve_vmap_batch` takes the running
   backend and clamps to 3 on non-GPU; probes record their backend and
   resolution ignores probes from a different backend.
2. **Instrument-naive pins.** `imaging/{mge,pixelization,delaunay}` asserted
   an HST-only pinned likelihood/evidence for every `--instrument` — any
   non-HST run failed by construction (mge at 62 s; pixelization sparse died
   at 2499 s *after* 41 min of honest compute, at the final assert).
   Converted to the per-instrument dict pattern `interferometer/mge` had.
3. **Pins were hard asserts** (design defect, user-directed redesign): they
   guard benchmark comparability, not library correctness, and must never
   kill a profiling job — see the record-and-flag section of
   [`design_lock_in.md`](./design_lock_in.md).

## Empirical bonus

Pixelization **sparse at batch 64 on CPU did not OOM where dense did** (it
survived to the final assert), directly confirming the sparse path's smaller
per-replica footprint — the basis for `VMAP_BATCH_SPARSE`'s
fall-back-to-dense-is-conservative rule.

## What phase 3 consumes

- Batch sizes: `vram.resolve_vmap_batch` (probe JSON → table → clamp), sparse
  rows via `VMAP_BATCH_SPARSE` once the A100 probes are ingested.
- The dense table's provenance is 2026-05-24 (`vram.config.PROVENANCE`) —
  ingest the probe-only A100 submits' JSONs before trusting it for the
  campaign (workflow: `vram/README.md` "Probe-only submits + ingest").
- No pins exist yet for euclid/ao/jwst imaging or any interferometer
  instrument — first clean campaign runs should pin them (values are printed
  by every unpinned run).
