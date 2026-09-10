"""JIT / vmap timing harness shared by the ``likelihood_breakdown`` cells.

Lifted verbatim from ``scripts/imaging/likelihood_breakdown/delaunay.py`` (the
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
):
    """JIT-compile *func*, time lower / compile / first call / steady state.

    Records four ``timer`` sections (``{label}_lower``, ``{label}_compile``,
    ``{label}_first_call``, ``{label}_steady_x{n_repeats}``) and, when
    *jit_records* is given, stores the same four numbers under *label* as
    ``{lower_s, compile_s, first_call_s, steady_per_call_s}`` so the cell can
    write compile time into its result JSON.

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

    if jit_records is not None:
        jit_records[label] = {
            "lower_s": float(lower_s),
            "compile_s": float(compile_s),
            "first_call_s": float(first_call_s),
            "steady_per_call_s": float(per_call),
        }

    return compiled, result


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
