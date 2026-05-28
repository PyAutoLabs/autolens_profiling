"""Metrics collected during a first-class PyAutoFit search profiling run.

The runner wraps an analysis instance via ``attach_viz_timer``, runs the
search, then calls ``collect_metrics`` to assemble the per-cell result dict.

Two metric sources:

1. **Visualization wall-time** — accumulated across every call to the
   analysis's visualize-family methods plus the search's
   ``plot_results``. The framework writes a per-update visualization
   time into ``search.summary`` but only the *last* update's value, so
   accumulating in-process is the only way to get a total.

2. **Sampler/search statistics** — read post-hoc from the returned
   ``Result.samples`` (log_evidence, max log L, posterior count, total
   samples). The framework already persists these to disk; we just
   surface them in the JSON.

Viz wall-time is intentionally *separate* from total search wall-time so
the JSON can answer both questions: "how long did the full first-class
fit take?" and "how much of that was visualization?".
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VizTimer:
    """Accumulates wall-time spent inside wrapped visualize callables.

    Calls are not assumed to be re-entrant; each enter pushes a fresh
    start onto a stack so that nested ``visualize_*`` paths (combined →
    individual) don't double-count if PyAutoFit ever changes which calls
    which.
    """

    total_s: float = 0.0
    n_calls: int = 0
    _stack: list[float] = field(default_factory=list)

    def __enter__(self) -> "VizTimer":
        self._stack.append(time.perf_counter())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._stack:
            return
        start = self._stack.pop()
        # Only the outermost frame contributes to the accumulator so we
        # don't double-count if visualize_combined() internally calls
        # visualize().
        if not self._stack:
            self.total_s += time.perf_counter() - start
            self.n_calls += 1


def _wrap_method(target: Any, attr: str, timer: VizTimer) -> None:
    """Wrap ``target.attr`` so each call accumulates wall-time into ``timer``.

    No-op if the attribute does not exist (older PyAutoLens analyses may
    not implement every visualize-family hook).
    """
    fn = getattr(target, attr, None)
    if fn is None:
        return

    def wrapped(self, *args, **kwargs):
        with timer:
            return fn(*args, **kwargs)

    setattr(target, attr, types.MethodType(wrapped, target))


def attach_viz_timer(analysis: Any, search: Any) -> VizTimer:
    """Wrap every visualize-family hook on ``analysis`` and ``search``.

    Hooks captured:

    - ``analysis.visualize_before_fit`` and
      ``analysis.visualize_before_fit_combined`` — fire once at the
      start of the search, *outside* the SearchUpdater's per-update
      timer.
    - ``analysis.visualize`` and ``analysis.visualize_combined`` — fire
      every full update during the sampling loop.
    - ``search.plot_results`` — search-specific plots (e.g. Nautilus
      corner plots via anesthetic), called from the SearchUpdater.

    Returns the timer; read ``timer.total_s`` after the fit completes.
    """
    timer = VizTimer()
    for attr in (
        "visualize_before_fit",
        "visualize_before_fit_combined",
        "visualize",
        "visualize_combined",
    ):
        _wrap_method(analysis, attr, timer)
    _wrap_method(search, "plot_results", timer)
    return timer


@dataclass
class RunMetrics:
    """Headline numbers a profiling cell writes to its JSON."""

    total_wall_s: float
    viz_wall_s: float
    sampler_wall_s: float
    likelihood_evals: int
    time_per_eval_ms: float
    log_evidence: float
    max_log_likelihood: float
    posterior_samples: int


def collect_metrics(
    *,
    result: Any,
    total_wall_s: float,
    viz_wall_s: float,
) -> RunMetrics:
    """Assemble the headline metric block from a finished ``search.fit`` result.

    ``sampler_wall_s = total_wall_s - viz_wall_s`` keeps things honest
    relative to per-call counters that might disagree with the
    framework's own timer.
    """
    samples = result.samples
    total_samples = int(samples.total_samples)

    try:
        log_evidence = float(samples.log_evidence)
    except (AttributeError, TypeError):
        log_evidence = float("nan")

    try:
        max_log_likelihood = float(samples.max_log_likelihood_sample.log_likelihood)
    except AttributeError:
        max_log_likelihood = float("nan")

    try:
        posterior_samples = int(len(samples.parameter_lists))
    except (AttributeError, TypeError):
        posterior_samples = 0

    sampler_wall_s = max(total_wall_s - viz_wall_s, 0.0)
    time_per_eval_ms = (
        sampler_wall_s / max(total_samples, 1) * 1e3 if total_samples else float("nan")
    )

    return RunMetrics(
        total_wall_s=total_wall_s,
        viz_wall_s=viz_wall_s,
        sampler_wall_s=sampler_wall_s,
        likelihood_evals=total_samples,
        time_per_eval_ms=time_per_eval_ms,
        log_evidence=log_evidence,
        max_log_likelihood=max_log_likelihood,
        posterior_samples=posterior_samples,
    )
