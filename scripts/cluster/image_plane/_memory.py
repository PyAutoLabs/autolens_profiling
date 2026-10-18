"""
Process memory snapshots for the cluster PointSolver sweep (autolens_profiling#436).

RAL pilot job 412567 was OOM-killed during the A2744 leg (MaxRSS 66.6 GB at row 8 of 25,
``--mem=64gb``) with the growth unattributed. ``snapshot`` records, at each progress line,
the current resident set (``/proc/self/status`` VmRSS), the process peak
(``resource.getrusage`` ru_maxrss) and the count and total bytes of live JAX arrays, so the
next log separates retained device buffers (live arrays grow) from retained compiled
executables / host-side state (RSS grows, live arrays flat).

Importable without running the sweep; JAX is optional (``jax=None`` skips the live-array
fields). Every probe is best-effort: a missing ``/proc`` or a changed JAX API yields
``None``, never an exception.
"""

from __future__ import annotations

import resource
import sys


def current_rss_mb(status_path: str = "/proc/self/status") -> float | None:
    """Current resident set size (VmRSS) in MB, or ``None`` where ``/proc`` is unavailable."""
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except OSError:
        return None
    return None


def peak_rss_mb() -> float | None:
    """Process peak RSS in MB (``ru_maxrss``: kB on Linux, bytes on macOS)."""
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def live_arrays(jax) -> tuple[int | None, int | None]:
    """``(count, total bytes)`` of live JAX arrays, ``(None, None)`` if the API is unavailable."""
    if jax is None:
        return None, None
    try:
        arrays = jax.live_arrays()
    except Exception:  # noqa: BLE001
        return None, None
    total = 0
    for a in arrays:
        try:
            total += int(a.nbytes)
        except Exception:  # noqa: BLE001
            pass
    return len(arrays), total


def snapshot(jax=None) -> dict:
    """One memory record: ``rss_mb``, ``peak_rss_mb``, ``live_arrays``, ``live_array_bytes``."""
    n, nbytes = live_arrays(jax)
    rss = current_rss_mb()
    peak = peak_rss_mb()
    return {
        "rss_mb": None if rss is None else round(rss, 1),
        "peak_rss_mb": None if peak is None else round(peak, 1),
        "live_arrays": n,
        "live_array_bytes": nbytes,
    }


def describe(snap: dict) -> str:
    """Compact progress-line suffix for a ``snapshot``."""

    def mb(v):
        return "n/a" if v is None else f"{v:.0f} MB"

    live = (
        "live arrays n/a"
        if snap.get("live_arrays") is None
        else f"live arrays {snap['live_arrays']} ({snap['live_array_bytes'] / 2**20:.1f} MB)"
    )
    return f"[rss {mb(snap.get('rss_mb'))}, peak {mb(snap.get('peak_rss_mb'))}, {live}]"
