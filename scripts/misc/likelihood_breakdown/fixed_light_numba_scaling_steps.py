"""Exclusive wall-time attribution for the production sparse inversion path.

The instrumentation in this module is deliberately local to a context manager:
no PyAutoArray source is changed, and every patched descriptor is restored on
exit (including exceptional exit).  Put the context around one complete
``Analysis.log_likelihood_function`` call so ``other`` is its unclassified
outer remainder.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

CATEGORIES = (
    "curvature",
    "regularization",
    "solve",
    "logdet_H",
    "logdet_FH",
    "other",
)


class _TimedDescriptor:
    """Delegate to *descriptor* without changing its property/cache semantics."""

    def __init__(self, descriptor, timings: InversionTimings, category: str):
        self.descriptor = descriptor
        self.timings = timings
        self.category = category
        self.__doc__ = getattr(descriptor, "__doc__", None)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        with self.timings.measure(self.category):
            return self.descriptor.__get__(instance, owner)


class _TimedDataDescriptor(_TimedDescriptor):
    """Timed delegate for properties which are data descriptors."""

    def __set__(self, instance, value):
        setter = getattr(self.descriptor, "__set__", None)
        if setter is None:
            raise AttributeError("can't set attribute")
        setter(instance, value)

    def __delete__(self, instance):
        deleter = getattr(self.descriptor, "__delete__", None)
        if deleter is None:
            raise AttributeError("can't delete attribute")
        deleter(instance)


class _TimedCallable:
    """Callable wrapper used at the actual positive-only NNLS boundary."""

    def __init__(self, function, timings: InversionTimings, category: str):
        self.function = function
        self.timings = timings
        self.category = category

    def __call__(self, *args, **kwargs):
        with self.timings.measure(self.category):
            return self.function(*args, **kwargs)


@dataclass
class InversionTimings:
    """Millisecond totals populated by :func:`timing_scope`."""

    total_ms: float = 0.0
    inclusive_ms: dict[str, float] = field(
        default_factory=lambda: {category: 0.0 for category in CATEGORIES}
    )
    exclusive_ms: dict[str, float] = field(
        default_factory=lambda: {category: 0.0 for category in CATEGORIES}
    )
    _stack: list[list[Any]] = field(default_factory=list, repr=False)

    @contextmanager
    def measure(self, category: str):
        """Measure one possibly nested operation, excluding timed children."""
        if category not in CATEGORIES or category == "other":
            raise ValueError(f"unknown timed category: {category!r}")
        frame = [category, time.perf_counter_ns(), 0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - frame[1]
            popped = self._stack.pop()
            if popped is not frame:  # defensive: a corrupt stack must be visible
                raise RuntimeError("timing stack exited out of order")
            exclusive_ns = max(0, elapsed_ns - frame[2])
            self.inclusive_ms[category] += elapsed_ns / 1_000_000
            self.exclusive_ms[category] += exclusive_ns / 1_000_000
            if self._stack:
                self._stack[-1][2] += elapsed_ns


_PATCH_LOCK = threading.RLock()


def _production_targets() -> tuple[tuple[type, str, str], ...]:
    from autoarray.inversion.inversion import inversion_util
    from autoarray.inversion.inversion.abstract import AbstractInversion
    from autoarray.inversion.inversion.imaging_numba.sparse import (
        InversionImagingSparseNumba,
    )

    return (
        (InversionImagingSparseNumba, "curvature_matrix", "curvature"),
        (AbstractInversion, "regularization_matrix", "regularization"),
        (inversion_util, "reconstruction_positive_only_from", "solve"),
        (AbstractInversion, "log_det_regularization_matrix_term", "logdet_H"),
        (AbstractInversion, "log_det_curvature_reg_matrix_term", "logdet_FH"),
    )


@contextmanager
def timing_scope(
    targets: Sequence[tuple[type, str, str]] | None = None,
):
    """Instrument one likelihood evaluation and yield its timing record.

    ``targets`` exists for focused tests; production callers should omit it.
    Contexts are serialized because descriptors are process-global.  Exact
    original objects are restored in reverse order even when evaluation raises.
    """
    timings = InversionTimings()
    originals: list[tuple[type, str, Any]] = []
    with _PATCH_LOCK:
        try:
            for owner, name, category in targets or _production_targets():
                original = vars(owner)[name]
                originals.append((owner, name, original))
                if isinstance(owner, type):
                    wrapper_type = (
                        _TimedDataDescriptor
                        if hasattr(type(original), "__set__")
                        else _TimedDescriptor
                    )
                    wrapper = wrapper_type(original, timings, category)
                else:
                    wrapper = _TimedCallable(original, timings, category)
                setattr(owner, name, wrapper)
            started_ns = time.perf_counter_ns()
            yield timings
        finally:
            if "started_ns" in locals():
                timings.total_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                classified = sum(timings.exclusive_ms[category] for category in CATEGORIES[:-1])
                timings.exclusive_ms["other"] = max(0.0, timings.total_ms - classified)
                timings.inclusive_ms["other"] = timings.exclusive_ms["other"]
            for owner, name, descriptor in reversed(originals):
                setattr(owner, name, descriptor)


def _cell_value(cell: Mapping[str, Any], key: str) -> Any:
    value = cell.get(key)
    if value is None and isinstance(cell.get("timings"), Mapping):
        value = cell["timings"].get(key)
    return value


def summarize_scaling(
    cells: Iterable[Mapping[str, Any]],
    *,
    size_key: str = "source_pixels",
    value_key: str = "total_ms",
) -> dict[str, Any]:
    """Describe measured, failed, and missing cells and fit ``time ~ size**p``.

    Every input cell is represented in ``cells``.  Only positive finite values
    with status ``ok``/``measured`` enter the log-log fit.
    """
    rows: list[dict[str, Any]] = []
    points: list[tuple[float, float]] = []
    counts: Counter[str] = Counter()
    for cell in cells:
        status = str(cell.get("status", "missing"))
        size = _cell_value(cell, size_key)
        value = _cell_value(cell, value_key)
        valid = status in {"ok", "measured"}
        try:
            x, y = float(size), float(value)
            valid = valid and x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)
        except (TypeError, ValueError):
            valid = False
        effective_status = "measured" if valid else (status if status != "ok" else "failed")
        counts[effective_status] += 1
        rows.append({"status": effective_status, size_key: size, value_key: value})
        if valid:
            points.append((x, y))

    exponent = r_squared = None
    if len(points) >= 2:
        xs = [math.log(point[0]) for point in points]
        ys = [math.log(point[1]) for point in points]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator > 0:
            exponent = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
            intercept = y_mean - exponent * x_mean
            residual = sum((y - (intercept + exponent * x)) ** 2 for x, y in zip(xs, ys))
            total = sum((y - y_mean) ** 2 for y in ys)
            r_squared = (
                1.0 if total == 0 and residual == 0 else (1.0 - residual / total if total else None)
            )

    return {
        "cells": rows,
        "status_counts": dict(counts),
        "measured_points": len(points),
        "log_log_exponent": exponent,
        "r_squared": r_squared,
    }
