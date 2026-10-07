# Streaming scaling: array-free interferometer vs in-memory

These cells measure the memory and time of the two ways to build an
interferometer dataset that is ready for a sparse-operator inversion:

- **Streamed (array-free):** `Interferometer.from_stream(chunks, mask)` accumulates
  the sparse terms one visibility chunk at a time and never holds the full arrays.
- **In memory:** `Interferometer(data, noise_map, uv_wavelengths, mask).apply_sparse_operator()`
  holds every visibility at once.

The question comes from PyAutoLabs Discussion #13, which concerns a 2e8-visibility
ALMA cube. The campaign page is
[`wiki/campaigns/interferometer_streaming.md`](../../../wiki/campaigns/interferometer_streaming.md).

## Scripts

| Script | What it measures |
|--------|------------------|
| `streaming_accumulate.py` | `from_stream` wall time and peak RSS vs N_vis × chunk. Gives s per 1e6 vis, a linear fit `wall = a + b·N` per chunk, and a cProfile top-10 when a rate exceeds 5 s per 1e6 vis. `--extra` rows run only within `--budget-s`. |
| `streaming_in_memory.py` | `apply_sparse_operator()` peak RSS and wall vs N_vis, ascending and stopping at the first failure. One arm uses library defaults and one uses `nufft_chunk_size`. |
| `streaming_parity.py` | `log_evidence` of a 20×20 `RectangularUniform` sparse inversion, streamed vs in memory, as \|Δ\| in nats. |
| `streaming_plot_scaling.py` | Both paths on one figure, from the two JSONs. It runs no measurement. |
| `_streaming.py` | Shared harness: seeded synthetic chunks, the builders, the fresh-child runner (`RLIMIT_AS` cap and timeout), and provenance. |

Every measurement runs in a **fresh child process** with a 10 GB `RLIMIT_AS` cap and a
per-child timeout. An out-of-memory run is therefore a recorded outcome, together with the
innermost PyAuto frame it failed in. `RLIMIT_AS` limits virtual address space, which is
stricter than resident memory, so peak RSS is recorded beside every outcome.

The dataset is synthetic. Chunk `k` is drawn from `default_rng(1234 + k)`, with `uv`
uniform in ±1e5 wavelengths, a 400×400 circular mask at 0.05″/pix (125 676 pixels), and
`TransformerNUFFT`.

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
python scripts/interferometer/delaunay/streaming_in_memory.py
python scripts/interferometer/delaunay/streaming_accumulate.py --n-vis 1e6,4e6,1.6e7 --chunks 65536,4096 --extra 5e7:65536,1e8:65536
python scripts/interferometer/delaunay/streaming_parity.py --n-vis 4e6,5e5
python scripts/interferometer/delaunay/streaming_plot_scaling.py
```

Results are written to `results/streaming_scaling/` as
`<cell>_local_cpu_v<version>.{json,png}`. Here `<version>` is `git describe --tags` of the
PyAutoArray checkout, so `2026.10.4.1+2` means two commits past the tag. The library
`__version__` is not used, because a source checkout reports a stale one. Per-child logs
are written under `output/streaming_scaling/`, which is gitignored.
