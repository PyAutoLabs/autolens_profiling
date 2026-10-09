"""JIT / vmap timing harness shared by the ``likelihood_breakdown`` cells.

Lifted verbatim from ``scripts/imaging/delaunay/likelihood_breakdown.py`` (the
most-evolved copy) on 2026-09-10 so the three imaging cells stop carrying
divergent copies of the same four functions. Two behaviours from that copy are
load-bearing and are the reason the rectangular cell's older local copy had to
go:

- ``block()`` synchronises **every leaf** of a returned pytree. A
  tuple-returning prefix slipped through the older ``hasattr(x,
  "block_until_ready")`` test and was therefore timed asynchronously.
- ``jit_profile`` times ``lower`` and ``compile`` separately, so compile time
  is recoverable from the result JSON rather than only from SLURM stdout.

Neither ``Timer`` nor ``jit_profile`` is a module-level singleton: each cell
owns its ``Timer`` and its ``jit_records`` dict and passes them in, so a cell
can still keep its own section names.

``steady_median_profile`` (autolens_profiling#371) is the steady-state
statistic. It runs at least five warm calls and then times N calls one by one,
returning the median with p10 / p90.

``jit_profile``'s ``steady_per_call_s`` is the mean of one block of
``n_repeats`` calls taken right after the first call. On the A100 that block
can land in the post-compile transient: the source-plane cell read 0.642 ms
against a steady 0.267 ms median (jobs 366912 / 366914). It is kept unchanged
for continuity with every committed row. The median is an addition beside it,
never a replacement.

``headline_steady_median`` (timing-noise audit P8, autolens_profiling#362) is
the one call every runtime cell makes after its ``full_pipeline`` block: it
takes the steady median of the already compiled pipeline and returns the
``full_pipeline_single_jit_median*`` fields written beside the legacy
``full_pipeline_single_jit``. Compile time stays in ``jit_profile``'s
``lower`` / ``compile`` / ``first_call`` sections; the median only times warm
calls of the compiled callable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager

import jax


class Timer:
    """Accumulates named timing measurements and prints a summary."""

    def __init__(self):
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def section(self, label: str):
        """Context manager that records wall-clock time for *label*."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        print(f"  [{label}] {elapsed:.4f} s")

    def summary(self):
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        max_label = max(len(r[0]) for r in self.records)
        total = 0.0
        for label, elapsed in self.records:
            print(f"  {label:<{max_label}}  {elapsed:>10.4f} s")
            total += elapsed
        print("-" * 70)
        print(f"  {'TOTAL':<{max_label}}  {total:>10.4f} s")
        print("=" * 70)


def block(x):
    """Force synchronisation on every JAX array in *x* (array or pytree).

    Tuple-returning prefixes (the multi-output stages of the ``--split-setup``
    walk) slip through a bare ``hasattr(x, "block_until_ready")`` test and are
    then timed asynchronously — exactly the artifact the H-row attribution fix
    removes elsewhere. Blocking over ``tree_leaves`` makes every timed step
    synchronous on the same terms.
    """
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def jit_profile(
    func: Callable,
    label: str,
    *args,
    n_repeats: int = 10,
    timer: Timer,
    jit_records: MutableMapping[str, dict] | None = None,
    median_n_warm: int = 0,
    median_n_timed: int = 0,
):
    """JIT-compile *func*, time lower / compile / first call / steady state.

    Records four ``timer`` sections (``{label}_lower``, ``{label}_compile``,
    ``{label}_first_call``, ``{label}_steady_x{n_repeats}``) and, when
    *jit_records* is given, stores the same four numbers under *label* as
    ``{lower_s, compile_s, first_call_s, steady_per_call_s}`` so the cell can
    write compile time into its result JSON.

    ``median_n_timed > 0`` opts in to the steady median
    (:func:`steady_median_profile`, with ``median_n_warm`` of at least 5). It
    runs *after* the block above and adds no ``timer`` section, so
    ``steady_per_call_s`` and ``timer.records[-1]`` are exactly what they are
    without it. The result is stored as ``jit_records[label]["steady_median"]``.

    Returns ``(compiled, result)``.
    """
    jitted = jax.jit(func)

    with timer.section(f"{label}_lower"):
        lowered = jitted.lower(*args)
    lower_s = timer.records[-1][1]

    with timer.section(f"{label}_compile"):
        compiled = lowered.compile()
    compile_s = timer.records[-1][1]

    with timer.section(f"{label}_first_call"):
        result = compiled(*args)
        block(result)
    first_call_s = timer.records[-1][1]

    with timer.section(f"{label}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            result = compiled(*args)
            block(result)

    per_call = timer.records[-1][1] / n_repeats
    print(f"    -> per-call avg: {per_call:.6f} s")

    steady_median = None
    if median_n_timed > 0:
        steady_median = steady_median_profile(
            compiled, *args, n_warm=median_n_warm, n_timed=median_n_timed
        )
        print(
            f"    -> steady median: {steady_median['median_s']:.6f} s "
            f"(p10 {steady_median['p10_s']:.6f}, p90 {steady_median['p90_s']:.6f}; "
            f"{steady_median['n_warm']} warm, {steady_median['n_timed']} timed)"
        )

    if jit_records is not None:
        jit_records[label] = {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "steady_per_call_s": float(per_call),
        }
        if steady_median is not None:
            jit_records[label]["steady_median"] = steady_median

    return compiled, result


#: Minimum warm calls before the steady median is timed. The diagnostic that
#: found the A100 transient (job 366914) warmed with 5.
MIN_STEADY_WARM = 5


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted sequence (numpy's default)."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def steady_median_profile(
    compiled: Callable,
    *args,
    n_warm: int = MIN_STEADY_WARM,
    n_timed: int = 200,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Steady-state per-call statistic of an already compiled callable.

    The callable runs ``n_warm`` (at least :data:`MIN_STEADY_WARM`) untimed warm
    calls. It then runs ``n_timed`` calls, each timed on its own with ``block``
    inside the timed interval. Returns ``{n_warm, n_timed, median_s, p10_s,
    p90_s, mean_s, statistic}``.

    This is the statistic of the 2026-09-28 diagnostic (job 366914: 5 warm,
    400 timed; median 0.267 ms, p10 0.260, p90 0.279). It records no ``Timer``
    section, so it can follow ``jit_profile`` without moving
    ``timer.records[-1]``.
    """
    if n_warm < MIN_STEADY_WARM:
        raise ValueError(f"n_warm must be >= {MIN_STEADY_WARM}, got {n_warm}")
    if n_timed < 1:
        raise ValueError(f"n_timed must be >= 1, got {n_timed}")
    for _ in range(n_warm):
        block(compiled(*args))
    samples = []
    for _ in range(n_timed):
        start = clock()
        block(compiled(*args))
        samples.append(clock() - start)
    ordered = sorted(samples)
    return {
        "n_warm": int(n_warm),
        "n_timed": int(n_timed),
        "median_s": _quantile(ordered, 0.5),
        "p10_s": _quantile(ordered, 0.1),
        "p90_s": _quantile(ordered, 0.9),
        "mean_s": float(sum(samples) / len(samples)),
        "statistic": "median of individually timed calls after warm calls",
    }


#: Timed-call budget of the headline steady median, seconds. ``n_timed`` is the
#: budget over the block mean, clamped to [HEADLINE_MEDIAN_MIN_TIMED,
#: HEADLINE_MEDIAN_MAX_TIMED]: a 0.3 ms A100 call gets the 200 of job 366914, a
#: 4 s CPU call gets 20 (about 100 s with its 5 warm calls).
HEADLINE_MEDIAN_BUDGET_S = 30.0
HEADLINE_MEDIAN_MAX_TIMED = 200
HEADLINE_MEDIAN_MIN_TIMED = 20
#: Above this block mean (s per call) no headline median is taken: its 5 warm + 20 timed calls
#: would cost more than ~50 s (a 48.8 s laptop call would add ~20 min and could push a run past
#: ``--per-run-timeout`` or an HPC wall). The post-compile transient the median guards against is
#: a sub-millisecond-to-millisecond effect; the row keeps the legacy headline, labelled as such.
HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S = 2.0

#: What the legacy ``full_pipeline_single_jit`` is, written into every protocol block.
SINGLE_JIT_BLOCK_MEAN_IS = (
    "mean of one 10-call block after one first call (first block after compile)"
)


def headline_median_n_timed(block_mean_s: float | None) -> int:
    """Timed calls for the headline median, sized from the block mean (see the budget above)."""
    if block_mean_s is None or not block_mean_s > 0 or block_mean_s != block_mean_s:
        return HEADLINE_MEDIAN_MAX_TIMED
    n = int(HEADLINE_MEDIAN_BUDGET_S // block_mean_s)
    return max(HEADLINE_MEDIAN_MIN_TIMED, min(HEADLINE_MEDIAN_MAX_TIMED, n))


def single_jit_median_fields(steady: Mapping, prefix: str = "full_pipeline_single_jit") -> dict:
    """The result-JSON fields of a steady median written beside the legacy block mean.

    ``<prefix>_median`` (seconds) is the key ``catalogue_adapters.DIRECT`` already maps to the
    ``runtime`` / ``single_jit_median`` metric. ``<prefix>_median_ms`` / ``_p10_ms`` / ``_p90_ms``
    / ``_median_protocol`` are the keys the source-plane solved cell has written since #371 and
    that ``build_dashboard`` reads. ``<prefix>`` itself is never written here.
    """
    return {
        f"{prefix}_median": float(steady["median_s"]),
        f"{prefix}_median_ms": float(steady["median_s"]) * 1000.0,
        f"{prefix}_p10_ms": float(steady["p10_s"]) * 1000.0,
        f"{prefix}_p90_ms": float(steady["p90_s"]) * 1000.0,
        f"{prefix}_median_protocol": {
            "n_warm": int(steady["n_warm"]),
            "n_timed": int(steady["n_timed"]),
            "statistic": steady["statistic"],
            f"{prefix}_is": SINGLE_JIT_BLOCK_MEAN_IS,
            "issue": "autolens_profiling#371; timing-noise audit P8 (#362)",
        },
    }


def headline_steady_median(
    compiled: Callable,
    *args,
    block_mean_s: float | None = None,
    n_warm: int = MIN_STEADY_WARM,
    n_timed: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Steady median of a cell's compiled ``full_pipeline``, as result-JSON fields.

    Call it right after the cell has read ``full_pipeline_per_call`` off the
    ``jit_profile`` block, with that block mean as *block_mean_s*: it adds no
    ``Timer`` section, so the legacy statistic is untouched. Returns
    :func:`single_jit_median_fields` of :func:`steady_median_profile`, or ``{}`` (no median; the
    legacy headline stands) when *block_mean_s* exceeds :data:`HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S`
    and *n_timed* is not given.
    """
    if (
        n_timed is None
        and block_mean_s is not None
        and block_mean_s > HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S
    ):
        print(
            f"  steady median skipped: block mean {block_mean_s:.3f} s > "
            f"{HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S:g} s per call (legacy headline stands)"
        )
        return {}
    if n_timed is None:
        n_timed = headline_median_n_timed(block_mean_s)
    steady = steady_median_profile(compiled, *args, n_warm=n_warm, n_timed=n_timed, clock=clock)
    print(
        f"  steady median = {steady['median_s']:.6f} s "
        f"(p10 {steady['p10_s']:.6f}, p90 {steady['p90_s']:.6f}; "
        f"{steady['n_warm']} warm, {steady['n_timed']} timed)"
        + (f"; block mean {block_mean_s:.6f} s" if block_mean_s is not None else "")
    )
    return single_jit_median_fields(steady)


def vmap_profile(
    func: Callable,
    label: str,
    params_batched,
    n: int,
    *,
    n_repeats: int = 10,
    timer: Timer,
    jit_records: MutableMapping[str, dict] | None = None,
) -> float:
    """Time ``jax.jit(jax.vmap(func))`` on *params_batched*; return per-call seconds.

    ``params_batched`` is the cell's params pytree broadcast to a leading batch
    axis of *n*. The returned number is amortized (batch time / n), which is
    what the ``--vmap-batch`` columns report. ``first_call_s`` is recorded into
    *jit_records* alongside the steady-state number; there is no separate
    lower/compile split here because the first call is what pays them.
    """
    fn = jax.jit(jax.vmap(func))

    with timer.section(f"{label}_vmap{n}_first_call"):
        block(fn(params_batched))
    first_call_s = timer.records[-1][1]

    with timer.section(f"{label}_vmap{n}_steady_x{n_repeats}"):
        for _ in range(n_repeats):
            block(fn(params_batched))

    batch_time = timer.records[-1][1] / n_repeats
    per_call = batch_time / n
    print(f"    -> batch {n}: {batch_time * 1000:9.3f} ms; per call: {per_call * 1000:9.3f} ms")

    if jit_records is not None:
        jit_records[f"{label}_vmap{n}"] = {
            "first_call_s": float(first_call_s),
            "batch_per_call_s": float(batch_time),
            "steady_per_call_s": float(per_call),
        }

    return per_call


def parse_vmap_batch(argv: Sequence[str]) -> int | None:
    """Parse ``--vmap-batch N`` / ``--vmap-batch=N`` out of *argv*.

    Read straight from ``sys.argv`` rather than added to
    ``_profile_cli.parse_profile_cli`` because it is a breakdown-only flag —
    the runtime cells resolve their batch from the VRAM table / probe JSON
    instead, and the shared parser stays the sweep-driver contract.
    """
    for i, arg in enumerate(argv):
        if arg == "--vmap-batch" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--vmap-batch="):
            return int(arg.split("=", 1)[1])
    return None


def split_by_successive_differences(
    prefix_per_call: Mapping[int, float],
    labels: Mapping[int, str],
) -> dict[str, float]:
    """Attribute a nested prefix walk to per-stage rows by successive differences.

    ``prefix_per_call`` maps a stage key to the timed ``params -> <stage>``
    prefix cost; ``labels`` maps the same keys to row names. Stages are walked
    in ascending key order and each row is ``t(stage) - t(previous stage)``,
    with the first row measured from zero.

    The differences inherit the XLA fusion caveat: work can move across a
    prefix boundary, so a small negative row is noise and the comparable
    aggregate is the absolute prefix, not the differenced row.
    """
    out: dict[str, float] = {}
    prev = 0.0
    for key in sorted(labels):
        out[labels[key]] = prefix_per_call[key] - prev
        prev = prefix_per_call[key]
    return out
