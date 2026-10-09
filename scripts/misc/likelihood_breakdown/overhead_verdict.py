"""One verdict for the ABBA instrumentation-overhead measurement (#362, fix phase 1).

Why this module exists
----------------------

The same instrument — ``call_accounting`` timed against a clean call in A B B A
blocks — used to be judged twice, in two units, by two pieces of code that did
not know about each other:

- the CI test ``test_call_accounting_covers_a_real_likelihood_call`` gated the
  mean block *ratio* at 1.031 with a one-sided Student-t interval, printed its
  INCONCLUSIVE (silent under ``pytest -q``) and could PASS on a ratio below 1;
- the production cell ``scripts/imaging/pixelized/fixed_light_numba.py`` gated a
  point estimate of the *milliseconds* of excess against 12 ms, with no interval,
  and raised on FAIL, losing the row's JSON.

``results/notes/timing_noise_audit_2026_10.md`` (rows T1 and P1, section (a)) is
the audit; this is its fix phase 1. Both callers now import
:func:`abba_overhead_verdict` from here, and the test asserts it is the same
function object the cell imports.

The rule
--------

The measured effect is the instrument's **excess over the clean call, in
milliseconds**: ``(ratio - 1) * clean_mean_s * 1e3``, the unit the budget is
written in (the instrument's cost is a fixed number of wrapper invocations per
call, so a ratio budget tightens every time the call gets shorter). Its
one-sided small-sample Student-t bounds at ``confidence`` over the per-block
ratios are compared with the budget, in this order:

1. invalid input (empty, non-1-D, non-finite or non-positive ratios; a
   non-finite or non-positive clean mean or budget) raises ``ValueError``;
2. ``FAIL_GROSS`` when the mean ratio exceeds ``gross_ratio``, whatever the
   interval width or block count — noise never masks a catastrophic regression;
3. ``INCONCLUSIVE`` when the row's warm-up record shows it never settled
   (``likelihood_breakdown.warmup_gate.warmup_unsettled_reason``; fix phase 6,
   row P3). The interval is not computed against the budget: a row timed on a
   host that had not reached steady state resolves neither a PASS nor a FAIL.
   The gross guard above still fires on such a row. A caller that passes no
   warm-up record (the CI fixture) is unaffected;
4. ``INCONCLUSIVE`` when fewer than ``min_blocks`` blocks were measured;
5. ``INCONCLUSIVE`` (reason "host-noise signature") when the mean ratio is
   **resolved** below 1, i.e. the upper bound of the excess is below zero. No
   instrumentation makes a call faster, so a resolved negative excess says the
   blocks are not measuring the instrument; it is never a measured pass. A mean
   below 1 whose interval still reaches 0 is consistent with a near-zero cost
   and is judged on its bounds like any other;
6. ``PASS`` when the upper bound is at or below the budget;
7. ``FAIL`` when the lower bound is above the budget;
8. ``INCONCLUSIVE`` otherwise — the interval straddles the budget.

The interval is valid only if the block ratios are roughly normal and
independent; at three to five blocks it cannot detect that they are not
(section (a) of the audit note states the limits). The function is pure and
deterministic: it does no timing, reads no clock and keeps no state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from likelihood_breakdown.warmup_gate import warmup_unsettled_reason

PASS = "PASS"
FAIL = "FAIL"
FAIL_GROSS = "FAIL_GROSS"
INCONCLUSIVE = "INCONCLUSIVE"

#: Every verdict :func:`abba_overhead_verdict` can return.
VERDICTS = (PASS, FAIL, FAIL_GROSS, INCONCLUSIVE)

#: The verdicts a caller must treat as a failed gate.
FAILING_VERDICTS = (FAIL, FAIL_GROSS)


@dataclass(frozen=True)
class OverheadVerdict:
    """The verdict and the numbers it was decided on.

    ``overhead_ms`` is the point estimate of the excess; ``lower_ms`` and
    ``upper_ms`` are its one-sided bounds at ``confidence`` (``nan`` when fewer
    than two blocks exist, so no spread can be estimated).
    """

    verdict: str
    mean_ratio: float
    overhead_ms: float
    lower_ms: float
    upper_ms: float
    n_blocks: int
    budget_ms: float
    confidence: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def abba_overhead_verdict(
    block_ratios,
    clean_mean_s: float,
    budget_ms: float,
    *,
    gross_ratio: float = 1.5,
    confidence: float = 0.95,
    min_blocks: int = 3,
    warmup: dict | None = None,
) -> OverheadVerdict:
    """Judge ABBA block ratios against a millisecond budget of instrument excess.

    Parameters
    ----------
    block_ratios
        One ``mean(B) / mean(A)`` ratio per A B B A block.
    clean_mean_s
        The row's own mean clean (uninstrumented) call, in seconds. It converts
        the ratio's excess over 1 into the milliseconds the budget is written in.
    budget_ms
        The largest acceptable instrument excess, in milliseconds.
    gross_ratio
        A mean ratio above this is ``FAIL_GROSS`` whatever the interval.
    confidence
        One-sided confidence of the Student-t bounds.
    min_blocks
        Below this many blocks the verdict is ``INCONCLUSIVE``.
    warmup
        The row's warm-up record (``rows[*].warmup``), or ``None`` when the
        measurement has none. A record that never settled makes any verdict
        short of ``FAIL_GROSS`` ``INCONCLUSIVE``.

    See the module docstring for the rule, in order.
    """
    ratios = np.asarray(block_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0 or not np.all(np.isfinite(ratios)):
        raise ValueError("ABBA block ratios must be a non-empty finite 1-D vector")
    if np.any(ratios <= 0.0):
        raise ValueError("ABBA block ratios must be positive")
    clean_mean_s = _positive_finite("clean_mean_s", clean_mean_s)
    budget_ms = _positive_finite("budget_ms", budget_ms)
    gross_ratio = _positive_finite("gross_ratio", gross_ratio)
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence!r}")
    if int(min_blocks) < 2:
        raise ValueError(f"min_blocks must be at least 2, got {min_blocks!r}")

    n_blocks = int(ratios.size)
    mean_ratio = float(np.mean(ratios))
    scale_ms = clean_mean_s * 1e3
    overhead_ms = (mean_ratio - 1.0) * scale_ms

    if n_blocks >= 2:
        from scipy.stats import t

        se = float(np.std(ratios, ddof=1)) / math.sqrt(n_blocks)
        margin = float(t.ppf(confidence, df=n_blocks - 1)) * se
        lower_ms = (mean_ratio - margin - 1.0) * scale_ms
        upper_ms = (mean_ratio + margin - 1.0) * scale_ms
    else:
        lower_ms = upper_ms = float("nan")

    def _verdict(verdict: str, reason: str) -> OverheadVerdict:
        return OverheadVerdict(
            verdict=verdict,
            mean_ratio=mean_ratio,
            overhead_ms=overhead_ms,
            lower_ms=lower_ms,
            upper_ms=upper_ms,
            n_blocks=n_blocks,
            budget_ms=budget_ms,
            confidence=float(confidence),
            reason=reason,
        )

    bounds = f"[{lower_ms:.3f}, {upper_ms:.3f}] ms"
    if mean_ratio > gross_ratio:
        return _verdict(
            FAIL_GROSS,
            f"mean ratio {mean_ratio:.4f} exceeds the gross bound {gross_ratio}; noise "
            f"never masks a catastrophic regression",
        )
    unsettled = warmup_unsettled_reason(warmup)
    if unsettled is not None:
        return _verdict(INCONCLUSIVE, f"{unsettled} (excess bounds {bounds}, not judged)")
    if n_blocks < min_blocks:
        return _verdict(
            INCONCLUSIVE,
            f"{n_blocks} block(s) < {min_blocks}: too few to resolve the {budget_ms:g} ms budget",
        )
    if upper_ms < 0.0:
        return _verdict(
            INCONCLUSIVE,
            f"host-noise signature: the mean ratio {mean_ratio:.4f} is resolved below 1 "
            f"(excess bounds {bounds}); no instrumentation makes a call faster, so this "
            f"is not a measured pass",
        )
    if upper_ms <= budget_ms:
        return _verdict(PASS, f"upper bound {upper_ms:.3f} ms <= budget {budget_ms:g} ms")
    if lower_ms > budget_ms:
        return _verdict(FAIL, f"lower bound {lower_ms:.3f} ms > budget {budget_ms:g} ms")
    return _verdict(
        INCONCLUSIVE,
        f"excess bounds {bounds} straddle the {budget_ms:g} ms budget",
    )
