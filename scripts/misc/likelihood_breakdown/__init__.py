"""Shared helpers for the per-step likelihood breakdown cells.

Two modules live here:

1. **``timing``** — the JIT/vmap timing harness every breakdown cell shares:
   ``Timer``, ``block``, ``jit_profile``, ``vmap_profile``,
   ``parse_vmap_batch`` and ``split_by_successive_differences``. These were
   copy-pasted per cell until 2026-09-10; the copies drifted (the rectangular
   cell still had a single-array ``block()`` that timed tuple-returning
   prefixes asynchronously) so they now live in one place.

2. **``sparse_steps``** — standalone JAX functions that reproduce
   ``autoarray.inversion.inversion.imaging.sparse.InversionImagingSparse``
   step by step for the *func-list + mapper* case (an MGE lens-light basis
   alongside one pixelized ``Mapper``), so a ``--sparse`` breakdown times the
   real w-tilde steps instead of the dense mapping-matrix ones.

The package is imported as ``from likelihood_breakdown import timing`` —
``scripts/misc`` is already on ``sys.path`` in every cell (see the
``_profiling_root()`` preamble each one carries), the same arrangement
``scripts/misc/vram/`` uses.

See ``README.md`` for the dense <-> sparse row correspondence.
"""

from likelihood_breakdown.timing import (
    Timer,
    block,
    jit_profile,
    parse_vmap_batch,
    split_by_successive_differences,
    vmap_profile,
)

__all__ = [
    "Timer",
    "block",
    "jit_profile",
    "parse_vmap_batch",
    "split_by_successive_differences",
    "vmap_profile",
]
