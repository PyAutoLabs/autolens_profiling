"""The warm-up validity check every timing verdict applies (#362, fix phase 6, row P3).

Why this module exists
----------------------

``scripts/imaging/pixelized/fixed_light_numba.py::_warm_to_steady_state`` warms
each row until the median of the last ``window`` calls is within ``tolerance``
of the median of the previous ``window`` calls, or until ``max_calls`` is
reached. It records the whole sequence and a ``steady`` flag, and it logs
"NEVER SETTLED" when the flag is false. Before this phase nothing read that flag.
A row timed on a host that never settled could still PASS or FAIL the ABBA
overhead gate (P1), and it could still be promoted or written as a measured
NO_LEVER (P2).

Section (a) of ``results/notes/timing_noise_audit_2026_10.md`` makes "warm-up
never settled" one of the validity checks whose failure makes a verdict
INCONCLUSIVE. :func:`warmup_unsettled_reason` is that check. It is shared so the
overhead verdict, the promotion block and the dashboard ask the same question
in the same words.

The rule
--------

- No warm-up record at all (``None`` or not a dict) returns ``None``. That is a
  measurement taken without this protocol, such as the CI fixture or a runtime
  cell. It is judged by its own rules, not by this one.
- A record whose ``steady`` is ``True`` returns ``None``.
- Anything else returns a reason string beginning :data:`WARMUP_UNSETTLED`.
  That covers ``steady`` false, and a record with no ``steady`` boolean, because
  a record that cannot show it settled has not shown it.

The function is pure and deterministic: it does no timing and keeps no state.
It does not change the steady-state rule itself. The limits of that rule are
stated under P3 in the audit note.
"""

from __future__ import annotations

import math
import statistics

#: The prefix of every reason this module returns, so readers can match on it.
WARMUP_UNSETTLED = "warm-up never settled"


def _medians(sequence, window):
    """(recent, previous) window medians of the recorded sequence, or None."""
    try:
        values = [float(v) for v in sequence]
        window = int(window)
    except (TypeError, ValueError):
        return None
    if window < 1 or len(values) < 2 * window or not all(math.isfinite(v) for v in values):
        return None
    return (
        statistics.median(values[-window:]),
        statistics.median(values[-2 * window : -window]),
    )


def warmup_unsettled_reason(warmup) -> str | None:
    """``None`` when the warm-up settled (or none was recorded), else why it did not.

    ``warmup`` is the record ``_warm_to_steady_state`` writes under
    ``rows[*].warmup``: ``sequence_s``, ``n_calls``, ``steady``, ``window``,
    ``tolerance`` and ``max_calls``.
    """
    if not isinstance(warmup, dict):
        return None
    steady = warmup.get("steady")
    if steady is True:
        return None
    if not isinstance(steady, bool):
        return f"{WARMUP_UNSETTLED}: the warm-up record carries no steady flag"

    sequence = warmup.get("sequence_s") or []
    n_calls = warmup.get("n_calls", len(sequence))
    window = warmup.get("window")
    tolerance = warmup.get("tolerance")
    detail = f"{n_calls} call(s)"
    medians = _medians(sequence, window)
    if medians is not None:
        recent, previous = medians
        change = abs(recent - previous) / max(previous, 1e-300)
        detail += (
            f", last-{window} median {recent * 1e3:.3f} ms vs previous-{window} "
            f"{previous * 1e3:.3f} ms ({change:.1%}"
        )
        detail += f" > {float(tolerance):.0%})" if tolerance is not None else ")"
    return (
        f"{WARMUP_UNSETTLED} ({detail}): the row was timed on a host that had not "
        f"reached steady state, so no timing verdict on it is resolved"
    )
