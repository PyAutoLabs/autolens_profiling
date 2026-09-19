"""Harness-level probe of the library's qhull host callback.

Why this module exists
----------------------

Phase 1 of the ``hst-gpu-non-solver-residue`` epic measured 5.44 ms of a
31.6 ms A100 call as one device-idle gap at ``pure_callback.N``: the host
round-trip that runs ``scipy.spatial.Delaunay`` on the source-plane mesh
vertices. The profiler's host plane names the *whole* round trip
(``pure_callback.N``) and the callback body (``scipy_delaunay_tri_only``), but
it cannot say how much of the body is qhull itself and how much is the
adjacency bookkeeping the body does afterwards — and under
``jax.vmap`` the callback is ``vmap_method="sequential"``, so the body runs
once **per lane** and its split is the term that never amortises over a batch.

This module answers that with a scoped wrapper, the same pattern
``library_solver_injection`` uses for the solver: nothing in PyAutoArray
moves, the wrapper returns the library function's own arrays bit-for-bit, and
it is restored on exit including on an exception.

Where the rebind goes, and why it takes
---------------------------------------

``autoarray/inversion/mesh/interpolator/delaunay.py`` builds the callback as::

    jax.pure_callback(
        lambda pts: scipy_delaunay_tri_only(np.asarray(pts)), ...
    )

The lambda resolves ``scipy_delaunay_tri_only`` as a **module global at call
time**, so rebinding the module attribute is picked up by the next callback
execution — the rebind does *not* have to be in place before lowering, and a
program lowered outside the context is still probed when it is *run* inside
it. The compiled program is unchanged either way: the callback is host Python
invoked at run time, not traced code.

That said, the caller should not take this on trust. Every context manager
yields a :class:`HostCallbackProbe` whose ``calls`` counter is the evidence,
and a cell that measures a callback split is expected to assert it is
non-zero — a probe that never fires would otherwise report ``0.0 ms`` of qhull
as though it had measured it.

What is timed
-------------

``qhull_s``
    ``scipy.spatial.Delaunay(points)`` alone.
``tables_s``
    Everything else the library body does — allocating the padded arrays,
    casting ``tri.simplices`` / ``tri.neighbors`` to int32, and the three-pass
    ``vertex_simplex`` scatter.

The split is re-implemented here rather than instrumented in place because the
library function is a single straight-line body with no seam to hook; the
re-implementation is pinned against the library's own output, array for array,
by ``scripts/misc/test/test_host_callback_probe.py``. If the library body ever
grows a step this wrapper does not have, that test fails rather than the
numbers quietly drifting.

Non-finite lanes
----------------

The library returns the ``-1``-padded arrays without calling qhull when the
points are not all finite (a NaN lane under ``vmap`` must not abort the whole
batch). The wrapper reproduces that early return exactly and records the call
with ``qhull_s = 0.0`` and ``finite = False``, so a batch whose cost is really
"one lane went NaN" cannot be read as a cheap triangulation.
"""

from __future__ import annotations

import contextlib
import functools
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LIBRARY_QHULL_DOTTED",
    "PROBE_HOST_EVENT_FRAGMENTS",
    "CallbackRecord",
    "HostCallbackProbe",
    "qhull_probe",
]

#: The dotted name this module rebinds. Recorded verbatim into the results
#: JSON so a reader of the numbers knows exactly what was wrapped.
LIBRARY_QHULL_DOTTED = "autoarray.inversion.mesh.interpolator.delaunay.scipy_delaunay_tri_only"

#: Name fragments the profiler's host plane uses for the probe's own wrapper, in
#: place of the library's ``scipy_delaunay_tri_only``. The profiler names a host
#: event from the code object, not from ``__name__``, so a trace taken with the
#: probe entered has NO event containing "scipy_delaunay" and a matcher written
#: for phase 1's trace reports 0.0 ms of qhull without saying anything.
PROBE_HOST_EVENT_FRAGMENTS = ("host_callback_probe.py", "_timed_tri_only")


@dataclass
class CallbackRecord:
    """One execution of the qhull host callback."""

    n_points: int
    qhull_s: float
    tables_s: float
    finite: bool

    @property
    def total_s(self) -> float:
        return self.qhull_s + self.tables_s

    def as_dict(self) -> dict:
        return {
            "n_points": int(self.n_points),
            "qhull_ms": self.qhull_s * 1e3,
            "tables_ms": self.tables_s * 1e3,
            "total_ms": self.total_s * 1e3,
            "finite": bool(self.finite),
        }


@dataclass
class HostCallbackProbe:
    """Mutable record of every qhull callback taken while the probe is entered."""

    records: list[CallbackRecord] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.records)

    def reset(self) -> None:
        """Drop every record so the next window is counted on its own.

        The per-*traced-call* callback count is the whole point of the probe —
        it is what says the ``vmap`` program pays ``B`` serial host round trips
        per batched call — and that number is only readable if the caller can
        zero the counter immediately before the window it means to measure.
        """
        self.records.clear()

    def summary(self) -> dict:
        """Totals over the records currently held, plus the per-call means."""
        n = self.calls
        qhull_ms = sum(r.qhull_s for r in self.records) * 1e3
        tables_ms = sum(r.tables_s for r in self.records) * 1e3
        return {
            "calls": n,
            "qhull_ms_total": qhull_ms,
            "tables_ms_total": tables_ms,
            "total_ms_total": qhull_ms + tables_ms,
            "qhull_ms_mean": qhull_ms / n if n else 0.0,
            "tables_ms_mean": tables_ms / n if n else 0.0,
            "total_ms_mean": (qhull_ms + tables_ms) / n if n else 0.0,
            "non_finite_calls": sum(1 for r in self.records if not r.finite),
            "n_points": sorted({int(r.n_points) for r in self.records}),
        }


def _timed_tri_only(points_np) -> tuple[tuple, float, float, bool]:
    """``scipy_delaunay_tri_only``'s own body, with the qhull call timed apart.

    Returns ``(arrays, qhull_s, tables_s, finite)``. The arrays are the same
    three the library returns, built by the same steps in the same order.
    """
    t0 = time.perf_counter()
    points_np = np.asarray(points_np)
    N = points_np.shape[0]
    simplices_padded = -np.ones((2 * N, 3), dtype=np.int32)
    simplex_neighbors = -np.ones((2 * N, 3), dtype=np.int32)
    vertex_simplex = -np.ones(N, dtype=np.int32)

    if not np.isfinite(points_np).all():
        tables_s = time.perf_counter() - t0
        return (simplices_padded, simplex_neighbors, vertex_simplex), 0.0, tables_s, False

    from scipy.spatial import Delaunay

    t_qhull = time.perf_counter()
    tri = Delaunay(points_np)
    qhull_s = time.perf_counter() - t_qhull

    simplices = tri.simplices.astype(np.int32)
    T = simplices.shape[0]
    simplices_padded[:T] = simplices
    simplex_neighbors[:T] = tri.neighbors.astype(np.int32)

    simplex_ids = np.arange(T, dtype=np.int32)
    for k in range(3):
        vertex_simplex[simplices[:, k]] = simplex_ids

    tables_s = (time.perf_counter() - t0) - qhull_s
    return (simplices_padded, simplex_neighbors, vertex_simplex), qhull_s, tables_s, True


@contextlib.contextmanager
def qhull_probe(*, reimplement: bool = True):
    """Rebind the library's qhull callback to a timing wrapper.

    ``reimplement=True`` (the default) runs this module's own copy of the
    library body so ``scipy.spatial.Delaunay`` can be timed apart from the
    table building. ``reimplement=False`` calls the library function untouched
    and times only the whole body — the fallback to reach for if the library
    body ever changes, because it can never disagree with the library, but it
    reports ``qhull_ms = 0.0`` and ``tables_ms`` equal to the whole call.

    Yields a :class:`HostCallbackProbe`. The original function is restored on
    exit, including on an exception.
    """
    from autoarray.inversion.mesh.interpolator import delaunay as _delaunay

    original = _delaunay.scipy_delaunay_tri_only
    probe = HostCallbackProbe()

    # ``functools.wraps`` keeps ``__name__``/``__doc__`` for anything that
    # introspects the attribute, but it does NOT rename the host-plane event: the
    # profiler builds that name from the CODE OBJECT (``$<file>:<line> <co_name>``),
    # so under the probe phase 1's ``$delaunay.py:90 scipy_delaunay_tri_only``
    # event becomes ``$host_callback_probe.py:<line> wrapped``. A cell that reads
    # a qhull span off the xplane must therefore match THIS module's names too, or
    # it silently reports 0.0 ms of qhull -- which is what
    # :data:`PROBE_HOST_EVENT_FRAGMENTS` is for.
    if reimplement:

        @functools.wraps(original)
        def wrapped(points_np):
            arrays, qhull_s, tables_s, finite = _timed_tri_only(points_np)
            probe.records.append(
                CallbackRecord(
                    n_points=int(np.asarray(points_np).shape[0]),
                    qhull_s=qhull_s,
                    tables_s=tables_s,
                    finite=finite,
                )
            )
            return arrays

    else:

        @functools.wraps(original)
        def wrapped(points_np):
            points_np = np.asarray(points_np)
            t0 = time.perf_counter()
            arrays = original(points_np)
            total_s = time.perf_counter() - t0
            probe.records.append(
                CallbackRecord(
                    n_points=int(points_np.shape[0]),
                    qhull_s=0.0,
                    tables_s=total_s,
                    finite=bool(np.isfinite(points_np).all()),
                )
            )
            return arrays

    _delaunay.scipy_delaunay_tri_only = wrapped
    try:
        yield probe
    finally:
        _delaunay.scipy_delaunay_tri_only = original
