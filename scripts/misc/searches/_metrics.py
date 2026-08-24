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

import json
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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

    def __enter__(self) -> VizTimer:
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


def _disable_method(target: Any, attr: str, timer: VizTimer) -> None:
    """Replace ``target.attr`` with a no-op that counts the skipped call.

    Used by the ``disable_viz`` path: the hook is never executed, so
    ``timer.total_s`` stays ~0 while ``n_calls`` still records how many
    visualizations *would* have run.
    """
    if getattr(target, attr, None) is None:
        return

    def skipped(self, *args, **kwargs):
        timer.n_calls += 1
        return None

    setattr(target, attr, types.MethodType(skipped, target))


def attach_viz_timer(analysis: Any, search: Any, disable: bool = False) -> VizTimer:
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

    With ``disable=True`` every hook is replaced by a no-op instead of a
    timing wrapper. This is for *convergence* benchmarks (e.g. the group
    cell), where pre-fit visualization of an 8-galaxy model costs ~1 hour
    of pure wall-clock before the optimizer takes a single step and tells
    us nothing about the question being asked. The resulting JSON reports
    ``viz_wall_s ~ 0`` with ``viz_n_calls`` still counting the skips, and
    the summary records ``viz_disabled: true`` so such a row is never
    mistaken for a real end-to-end viz measurement.

    Returns the timer; read ``timer.total_s`` after the fit completes.
    """
    timer = VizTimer()
    bind = _disable_method if disable else _wrap_method
    for attr in (
        "visualize_before_fit",
        "visualize_before_fit_combined",
        "visualize",
        "visualize_combined",
    ):
        bind(analysis, attr, timer)
    bind(search, "plot_results", timer)
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
    # W4 / issue #161 (Phase 1 targets registry, schema v2) additions — all
    # additive, so every pre-existing v1-shaped caller of collect_metrics
    # keeps working with these at their defaults.
    stored_samples: int = 0
    gradient_evals: int | None = None
    kish_ess: float | None = None
    evals_per_ess: float | None = None
    ess_per_min: float | None = None


def _kish_ess(weight_list: Any) -> float | None:
    """Kish effective sample size ``(sum(w))**2 / sum(w**2)``.

    Scale-invariant (multiplying every weight by a constant leaves the ratio
    unchanged), so it does not matter whether ``weight_list`` is normalised.
    Uniform weights over ``n`` samples give ``n``; all weight on one sample
    gives ``1``. Returns ``None`` when ``weight_list`` is absent or empty —
    MAP optimizers (MultiStart*) have no weighted posterior.
    """
    if weight_list is None:
        return None
    weights = np.asarray(weight_list, dtype=float)
    if weights.size == 0:
        return None
    sum_w = float(weights.sum())
    sum_w2 = float(np.square(weights).sum())
    if sum_w2 == 0.0:
        return None
    return (sum_w**2) / sum_w2


def collect_metrics(
    *,
    result: Any,
    total_wall_s: float,
    viz_wall_s: float,
    is_multi_start: bool = False,
    n_starts: int | None = None,
    multi_start_total_steps: int | None = None,
) -> RunMetrics:
    """Assemble the headline metric block from a finished ``search.fit`` result.

    ``sampler_wall_s = total_wall_s - viz_wall_s`` keeps things honest
    relative to per-call counters that might disagree with the
    framework's own timer.

    ``likelihood_evals`` (W4 / issue #161, Phase 1): for a nested sampler
    (Nautilus / NSS), ``samples.total_samples`` already counts every
    likelihood call including rejected proposals, so it is used directly. For
    a ``MultiStart*`` gradient search it is WRONG — ``total_samples`` there is
    a small posterior-storage count (frequently just ``n_starts`` or fewer),
    not the number of gradient/likelihood evaluations actually made. The
    correct reject-inclusive count is ``total_steps * n_starts`` (one
    likelihood + one gradient evaluation per lane per step —
    ``search_internal["total_steps"]`` from
    ``searches._per_lane.capture_search_internal``): e.g. 178 steps x 256
    starts = 45,568 evals, not the handful of stored samples. Pass
    ``is_multi_start=True`` with both ``n_starts`` and
    ``multi_start_total_steps`` to apply the fix; if either is ``None``
    (nested samplers, or a MultiStart run whose ``search_internal`` capture
    is unavailable — e.g. an older PyAutoFit predating the counters) it
    falls back to the old ``total_samples`` reading rather than raising, so
    a completed run's summary is never discarded over a missing diagnostic.

    ``stored_samples`` is always ``total_samples`` — the raw posterior/best-
    point storage count, kept distinct from the (corrected) evaluation count
    so the two questions ("how many evals ran" vs "how many samples are
    stored") never collapse into the same, ambiguous field again.
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

    likelihood_evals = total_samples
    gradient_evals = None
    if is_multi_start and n_starts is not None and multi_start_total_steps is not None:
        likelihood_evals = int(multi_start_total_steps) * int(n_starts)
        gradient_evals = likelihood_evals  # one gradient eval per likelihood eval per lane-step

    time_per_eval_ms = (
        sampler_wall_s / max(likelihood_evals, 1) * 1e3 if likelihood_evals else float("nan")
    )

    kish_ess = None if is_multi_start else _kish_ess(getattr(samples, "weight_list", None))
    evals_per_ess = likelihood_evals / kish_ess if kish_ess is not None and kish_ess > 0 else None
    ess_per_min = (
        kish_ess / (sampler_wall_s / 60.0) if kish_ess is not None and sampler_wall_s > 0 else None
    )

    return RunMetrics(
        total_wall_s=total_wall_s,
        viz_wall_s=viz_wall_s,
        sampler_wall_s=sampler_wall_s,
        likelihood_evals=likelihood_evals,
        time_per_eval_ms=time_per_eval_ms,
        log_evidence=log_evidence,
        max_log_likelihood=max_log_likelihood,
        posterior_samples=posterior_samples,
        stored_samples=total_samples,
        gradient_evals=gradient_evals,
        kish_ess=kish_ess,
        evals_per_ess=evals_per_ess,
        ess_per_min=ess_per_min,
    )


def load_summary(path: str | Path) -> dict:
    """Load a results JSON, normalising a v1-shaped summary to v2 in memory.

    W4 / issue #161 (Phase 1). A v1 summary (no ``schema_version`` key, or
    ``schema_version`` != 2) predates the Phase 1 schema additions and lacks
    ``target`` / ``algorithm`` / ``hardware``. This loader synthesises
    best-effort versions of those three blocks from the v1 keys that ARE
    present (``sampler`` / ``dataset_class`` / ``model`` / ``instrument`` /
    ``config_name`` / ``device`` / ``sampler_config`` / ``use_mixed_precision``)
    so a caller can treat every summary — old or new — uniformly without a
    schema-version branch of its own. The synthesised ``target`` block has
    ``target_id: None`` — a v1 run predates ``_targets.py`` and has no
    provenance to hash, so fabricating an id would be worse than omitting
    one. The file on disk is NEVER rewritten by this function; normalisation
    happens only in the returned ``dict``.
    """
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") == 2:
        return data

    normalised = dict(data)
    normalised["schema_version"] = normalised.get("schema_version", 1)

    if "target" not in normalised:
        cell = "/".join(
            str(normalised.get(k, "?")) for k in ("dataset_class", "model", "instrument")
        )
        normalised["target"] = {
            "target_id": None,
            "cell": cell,
            "model_dim": (normalised.get("model_summary") or {}).get("free_parameters"),
            "priors_ref": None,
            "note": "synthesised by load_summary() from a v1 (pre-schema-v2) results JSON",
        }
    if "algorithm" not in normalised:
        sampler_config = normalised.get("sampler_config") or {}
        normalised["algorithm"] = {
            "name": normalised.get("sampler"),
            "config_id": normalised.get("config_name"),
            "settings": sampler_config,
            "seed": sampler_config.get("seed"),
        }
    if "hardware" not in normalised:
        normalised["hardware"] = {
            "tier": normalised.get("config_name"),
            "precision": "mp" if normalised.get("use_mixed_precision") else "fp64",
            "device": normalised.get("device"),
        }
    return normalised
