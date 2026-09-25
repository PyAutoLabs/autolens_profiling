"""Profiling-only pre-solve policy for the production NNLS memo.

The wrapper deliberately leaves both production solve paths unchanged.  It
only decides whether the existing memo entry may be returned to production.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from likelihood_breakdown import fixed_light_draws_steps as draws

CALIBRATION = {
    "seed": 278,
    "nearby_n": 8,
    "broad_n": 8,
    "walk_sigma": 0.05,
    "thresholds": [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 2.0],
    "repeats": 3,
    "selection": (
        "minimum max(guarded broad / cold broad, guarded nearby / memo nearby), "
        "numerical passing candidates only; ties smaller threshold"
    ),
}


@dataclass(frozen=True)
class LockedPolicy:
    """A threshold selected from calibration and immutable during evaluation."""

    threshold: float


def select_threshold(rows: list[dict]) -> float:
    """Select the declared minimax threshold from numerical calibration rows."""
    passing = [row for row in rows if row.get("numerical_passed") is True]
    if not passing:
        raise ValueError("no numerically passing calibration candidate")
    selected = min(
        passing,
        key=lambda row: (
            max(float(row["broad_ratio"]), float(row["nearby_ratio"])),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"])


def build_rows(seed: int, n: int, nearby: bool) -> list[dict]:
    """Build deterministic broad draws or a cumulative nearby walk.

    The nearby sequence starts at its first nonzero increment, rather than at a
    shared fiducial.  Neither family clips its offsets.
    """
    kind = "nearby" if nearby else "broad"
    increments = draws.random_offsets(
        seed,
        n,
        sigma_scale=CALIBRATION["walk_sigma"] if nearby else draws.RANDOM_SIGMA_SCALE,
    )
    offsets = {parameter: 0.0 for parameter in draws.RANDOM_PARAMETERS}
    rows = []
    for index, increment in enumerate(increments):
        if nearby:
            offsets = {
                parameter: offsets[parameter] + increment[parameter]
                for parameter in draws.RANDOM_PARAMETERS
            }
            row_offsets = dict(offsets)
        else:
            row_offsets = dict(increment)
        rows.append(
            {
                "name": f"seed{seed}_{kind}_{index:02d}",
                "kind": kind,
                "seed": int(seed),
                "index": index,
                "offsets": row_offsets,
                "mass_cls": "Isothermal",
                "sigma_scale": (CALIBRATION["walk_sigma"] if nearby else draws.RANDOM_SIGMA_SCALE),
                "clipping": False,
            }
        )
    return rows


@dataclass
class PolicyState:
    """State owned by one policy traversal."""

    records: list[dict] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    reconstruction_cache: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict)
    last_solve_args: tuple = field(default_factory=tuple)
    last_solve_kwargs: dict = field(default_factory=dict)
    had_memo_entry: bool = False
    _counterfactual_context: object = field(default=None, repr=False)

    @contextlib.contextmanager
    def counterfactual_scope(self, enabled: bool) -> Iterator[None]:
        """Replay the last diagnostic call from its pre-decision memo state."""
        if self._counterfactual_context is None or not self.last_solve_kwargs:
            raise RuntimeError("no diagnostic solve has been captured")
        with self._counterfactual_context(bool(enabled)):
            yield


def residual_precheck_score(
    matrix: np.ndarray,
    vector: np.ndarray,
    previous_reconstruction: np.ndarray,
    passive_set: np.ndarray,
) -> float:
    """Return the dimensionless one-sided KKT residual of a previous solution."""
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    previous_reconstruction = np.asarray(previous_reconstruction)
    passive_set = np.asarray(passive_set)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    if vector.shape != (n,) or previous_reconstruction.shape != (n,):
        raise ValueError("matrix, vector and reconstruction dimensions must agree")
    if passive_set.ndim != 1 or not np.issubdtype(passive_set.dtype, np.integer):
        raise ValueError("passive_set must be a one-dimensional integer array")
    if passive_set.size and (int(passive_set.min()) < 0 or int(passive_set.max()) >= n):
        raise ValueError("passive_set contains an out-of-range index")
    if not (
        np.all(np.isfinite(matrix))
        and np.all(np.isfinite(vector))
        and np.all(np.isfinite(previous_reconstruction))
    ):
        raise ValueError("precheck inputs must be finite")

    try:
        with np.errstate(over="raise", invalid="raise"):
            product = matrix @ previous_reconstruction
            residual = product - vector
    except FloatingPointError as error:
        raise ValueError("precheck product and residual must be finite") from error
    if not np.all(np.isfinite(product)) or not np.all(np.isfinite(residual)):
        raise ValueError("precheck product and residual must be finite")
    passive = np.zeros(n, dtype=bool)
    passive[passive_set] = True
    passive_violation = float(np.max(np.abs(residual[passive]))) if np.any(passive) else 0.0
    inactive_violation = (
        float(np.max(np.maximum(-residual[~passive], 0.0))) if np.any(~passive) else 0.0
    )
    violation = max(passive_violation, inactive_violation)
    scale = max(
        float(np.linalg.norm(product, ord=np.inf)),
        float(np.linalg.norm(vector, ord=np.inf)),
    )
    if scale == 0.0:
        return 0.0 if violation == 0.0 else math.inf
    return violation / scale


@contextlib.contextmanager
def policy_scope(threshold: float, diagnostics: bool = False) -> Iterator[PolicyState]:
    """Apply one isolated, bounded eligibility policy around production NNLS."""
    from autoarray.inversion.inversion import inversion_util, nnls_memo

    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    original_outer = inversion_util.reconstruction_positive_only_from
    original_get = nnls_memo.passive_set_get
    production_memo = nnls_memo._nnls_passive_set_memo
    memo_snapshot = dict(production_memo)
    old_environment = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    state = PolicyState()
    current: list[tuple[np.ndarray, np.ndarray, str | None]] = []
    max_entries = nnls_memo._NNLS_PASSIVE_SET_MEMO_MAX_ENTRIES
    last_memo_snapshot: dict | None = None

    @contextlib.contextmanager
    def counterfactual_context(enabled: bool) -> Iterator[None]:
        if last_memo_snapshot is None:
            raise RuntimeError("no diagnostic solve has been captured")
        guarded_outer = inversion_util.reconstruction_positive_only_from
        guarded_get = nnls_memo.passive_set_get
        guarded_memo = dict(production_memo)
        guarded_environment = os.environ.get("AUTOARRAY_NNLS_WARM_START")
        inversion_util.reconstruction_positive_only_from = original_outer
        nnls_memo.passive_set_get = original_get
        production_memo.clear()
        production_memo.update(last_memo_snapshot)
        os.environ["AUTOARRAY_NNLS_WARM_START"] = "1" if enabled else "0"
        try:
            yield
        finally:
            inversion_util.reconstruction_positive_only_from = guarded_outer
            nnls_memo.passive_set_get = guarded_get
            production_memo.clear()
            production_memo.update(guarded_memo)
            if guarded_environment is None:
                os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
            else:
                os.environ["AUTOARRAY_NNLS_WARM_START"] = guarded_environment

    state._counterfactual_context = counterfactual_context

    def record(*, score, reason: str, accepted: bool, elapsed_ns: int) -> None:
        if diagnostics:
            state.records.append(
                {
                    "score": score,
                    "reason": reason,
                    "accepted": accepted,
                    "precheck_ms": elapsed_ns / 1.0e6,
                }
            )

    def policy_get(key: str, n: int):
        started = time.perf_counter_ns() if diagnostics else None
        entry = original_get(key=key, n=n)
        score = None
        reason = "missing_entry"
        accepted = False
        if entry is not None and current and key == current[-1][2]:
            matrix, vector, _ = current[-1]
            previous = state.reconstruction_cache.get(key)
            if previous is None:
                reason = "missing_reconstruction"
            else:
                try:
                    score = residual_precheck_score(matrix, vector, previous, entry.passive_set)
                except (TypeError, ValueError, FloatingPointError):
                    reason = "incompatible_or_nonfinite"
                else:
                    accepted = score <= threshold
                    reason = "accepted" if accepted else "score_above_threshold"
        elif entry is not None:
            reason = "incompatible_lookup"

        if entry is not None and not accepted:
            nnls_memo.memo_drop(key=key)
            state.reconstruction_cache.pop(key, None)
            state.rejected += 1
            result = None
        elif entry is not None:
            state.accepted += 1
            result = entry
        else:
            result = None
        if diagnostics:
            elapsed_ns = time.perf_counter_ns() - started
            record(score=score, reason=reason, accepted=accepted, elapsed_ns=elapsed_ns)
        return result

    # The library's #566 `solver` / `stats` and #572 `preconditioning` keywords are
    # accepted (the inversion passes them by name) and forwarded only to a library
    # that declares them.
    _outer_params = inspect.signature(original_outer).parameters

    def policy_outer(
        data_vector,
        curvature_reg_matrix,
        settings=None,
        xp=np,
        fingerprint=None,
        factor=None,
        solver="pdip",
        stats=None,
        preconditioning="jacobi",
    ):
        nonlocal last_memo_snapshot
        use_policy = (
            xp is np
            and settings is not None
            and settings.nnls_warm_start_memo
            and fingerprint is not None
        )
        key = (
            nnls_memo.memo_key(n=np.asarray(data_vector).shape[0], fingerprint=fingerprint)
            if use_policy
            else None
        )
        if diagnostics:
            last_memo_snapshot = dict(production_memo)
            state.last_solve_args = ()
            state.last_solve_kwargs = {
                "data_vector": data_vector,
                "curvature_reg_matrix": curvature_reg_matrix,
                "settings": settings,
                "xp": xp,
                "fingerprint": fingerprint,
                "factor": factor,
            }
            state.had_memo_entry = key is not None and key in last_memo_snapshot
        current.append((curvature_reg_matrix, data_vector, key))
        try:
            result = original_outer(
                data_vector,
                curvature_reg_matrix,
                settings=settings,
                xp=xp,
                fingerprint=fingerprint,
                factor=factor,
                **{
                    name: value
                    for name, value in (
                        ("solver", solver),
                        ("stats", stats),
                        ("preconditioning", preconditioning),
                    )
                    if name in _outer_params
                },
            )
        finally:
            current.pop()

        if key is not None and key in production_memo:
            state.reconstruction_cache[key] = np.asarray(result).copy()
            state.reconstruction_cache.move_to_end(key)
            while len(state.reconstruction_cache) > max_entries:
                state.reconstruction_cache.popitem(last=False)
        elif key is not None:
            state.reconstruction_cache.pop(key, None)
        for stale_key in set(state.reconstruction_cache).difference(production_memo):
            state.reconstruction_cache.pop(stale_key, None)
        return result

    inversion_util.reconstruction_positive_only_from = policy_outer
    nnls_memo.passive_set_get = policy_get
    production_memo.clear()
    os.environ["AUTOARRAY_NNLS_WARM_START"] = "1"
    try:
        yield state
    finally:
        if inversion_util.reconstruction_positive_only_from is policy_outer:
            inversion_util.reconstruction_positive_only_from = original_outer
        if nnls_memo.passive_set_get is policy_get:
            nnls_memo.passive_set_get = original_get
        production_memo.clear()
        production_memo.update(memo_snapshot)
        state.reconstruction_cache.clear()
        if old_environment is None:
            os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
        else:
            os.environ["AUTOARRAY_NNLS_WARM_START"] = old_environment
