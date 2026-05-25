"""VRAM / vmap-batch utilities for the likelihood profiling sweep.

Two responsibilities live here:

1. **Probing** — measure the per-replica VRAM cost of a vmapped likelihood
   function on a given device, so we can pick the largest batch_size that
   fits the device's memory budget. See ``vram.probe``.

2. **Configuration** — the curated table of per-(dataset, model, instrument)
   batch sizes derived from the probe results. Runtime cell scripts import
   ``vram.vmap_batch_for(...)`` to look up the production batch size for
   their cell. See ``vram.config``.

See ``vram/README.md`` for methodology and how to extend.
"""

from vram.config import VMAP_BATCH, vmap_batch_for
from vram.probe import (
    ProbeResult,
    probe_vmap_memory,
    recommend_batch_size,
    write_probe_json,
)

__all__ = [
    "VMAP_BATCH",
    "vmap_batch_for",
    "ProbeResult",
    "probe_vmap_memory",
    "recommend_batch_size",
    "write_probe_json",
]
