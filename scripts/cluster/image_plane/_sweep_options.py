"""
CLI option parsers for ``robust_settings_sweep.py`` (autolens_profiling#439).

The sweep runs at import, so its option parsing lives here to be unit-tested
without running a sweep. NumPy-free; no PyAutoLens import.
"""

from __future__ import annotations

DEFAULT_SCALES = (0.5, 0.7, 1.0, 2.0)


def parse_scales(text: str | None, default=DEFAULT_SCALES) -> tuple[float, ...]:
    """``"1.0,0.7,0.37"`` -> ``(1.0, 0.7, 0.37)``; ``None`` / empty -> ``default``.

    Order is kept (rows are built in this order); duplicates and non-positive or
    non-numeric entries are refused.
    """
    if text is None or not str(text).strip():
        return tuple(float(s) for s in default)
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"--scales: {item!r} is not a number") from exc
        if not value > 0.0:
            raise ValueError(f"--scales: {item!r} must be positive")
        if value in out:
            raise ValueError(f"--scales: {item!r} given twice")
        out.append(value)
    if not out:
        raise ValueError("--scales: no scale given")
    return tuple(out)


def parse_draw_ids(text: str | None) -> list[int] | None:
    """``"53,73,81"`` -> ``[53, 73, 81]`` (sorted, unique); ``None`` / empty -> ``None``."""
    if text is None or not str(text).strip():
        return None
    ids = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"--draw-ids: {item!r} must be >= 0")
        ids.add(value)
    return sorted(ids) or None
