"""Shared helpers for the per-step likelihood breakdown cells.

Three modules live here:

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

3. **``reconstruction_steps``** — standalone JAX pieces of the *inside* of the
   reconstruction and log-evidence rows: the Jacobi scaling and the PDIP NNLS
   driver (which, unlike the library's ``custom_vjp`` primal, keeps the
   iteration count and the convergence flag), a single Cholesky and a full
   unconstrained Cholesky solve of the same ``F + λH``, the two reduced log-det
   Choleskys, and every term of the evidence. These feed **overlapping**
   sub-rows (``steps_reconstruction_sub_rows``, ``nnls``,
   ``log_evidence_terms``) that are never summed against their parent row.

The package is imported as ``from likelihood_breakdown import timing`` —
``scripts/misc`` is already on ``sys.path`` in every cell (see the
``_profiling_root()`` preamble each one carries), the same arrangement
``scripts/misc/vram/`` uses.

See ``README.md`` for the dense <-> sparse row correspondence.
"""

#: The ``timing`` names this package re-exports, resolved LAZILY (PEP 562).
#:
#: ``timing`` imports ``jax`` at module level, so importing it here made
#: ``import likelihood_breakdown.<anything>`` an import of JAX — including
#: ``fixed_light_system`` and ``call_accounting``, the two modules that exist
#: precisely so a numba CPU cell can measure the likelihood with no JAX in the
#: process (its device block records ``use_jax: false`` and it asserts
#: ``"jax" not in sys.modules`` after its imports).
#:
#: Every consumer in this repo imports the *submodule*
#: (``from likelihood_breakdown import timing``), which never went through these
#: names; the ``__getattr__`` below keeps ``likelihood_breakdown.Timer`` working
#: for anything that does not.
__all__ = [
    "Timer",
    "block",
    "jit_profile",
    "parse_vmap_batch",
    "split_by_successive_differences",
    "vmap_profile",
]


def __getattr__(name):
    if name in __all__:
        from likelihood_breakdown import timing

        return getattr(timing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
