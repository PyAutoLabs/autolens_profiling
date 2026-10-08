"""One verdict for pre-registered A/B go / lever rules, and tie sets (#362, fix phase 3).

Why this module exists
----------------------

``results/notes/timing_noise_audit_2026_10.md`` found the campaign A/B rules
deciding on point estimates (rows C7 and P2), or holding an interval without an
INCONCLUSIVE state (C6), or naming a single "best" by argmax over statistically
tied configurations (C10). An unresolved difference was then written as a
measured negative ("a valid **measured** NO_LEVER", ``go: false``). This module
is the one rule those cells now share, beside the overhead rule in
:mod:`likelihood_breakdown.overhead_verdict`. Every caller imports
:func:`ab_rule_verdict` / :func:`tie_set` from here, and
``scripts/misc/test/test_ab_verdict.py`` asserts each cell holds the same
function object.

The rule
--------

A pre-registered rule is a conjunction of criteria. Each criterion is a measured
effect (a saving in ms, a fraction, a ratio) with an interval ``[lower, upper]``
at a stated confidence, and a bar it must be ``at_least`` or ``at_most``. In
order:

1. a red **correctness gate** gives ``NO_GO`` before any timing is read — a
   route that computes the wrong answer is not a lever, whatever its speed;
2. **invalid inputs** (a non-finite point or bound, ``lower > upper``, a
   negative or non-integer sample count) give ``INCONCLUSIVE``;
3. fewer than ``min_n`` independent units (rounds or blocks) give
   ``INCONCLUSIVE``;
4. per criterion: ``GO`` when the **whole interval** and the point estimate
   clear the bar; ``NO_GO`` when the **whole interval** is on the bad side;
   ``INCONCLUSIVE`` otherwise (the interval straddles the bar);
5. the rule is ``NO_GO`` if any criterion is ``NO_GO`` (the conjunction cannot
   hold), ``GO`` if every criterion is ``GO``, and ``INCONCLUSIVE`` otherwise.

``INCONCLUSIVE`` is a status, never a measured negative. It carries the
resolvable effect size of each criterion (``mdi``, the interval half-width in the
criterion's unit — the C12 minimum-detectable-improvement idea): an effect
smaller than that cannot be told from the bar by this run. Bars are the callers'
pre-registered values; nothing here raises or lowers one.

Argmax selection
----------------

:func:`tie_set` replaces "the fastest configuration". The leader is the best
point estimate; the tie set is every candidate whose interval overlaps the
leader's. A single ``best`` is named only when the tie set is the leader alone.

Limits (stated in the audit note, section (a))
----------------------------------------------

- The verdict is only as good as the interval it is given. An iid bootstrap over
  autocorrelated calls is too narrow (fix phase 4 resamples whole rounds).
- No multiple-comparison adjustment is applied to the per-criterion
  confidence. A conjunction of k criteria at 90 % each is conservative for GO
  (every interval must clear) but each NO_GO is a 90 % statement, and a tie set
  over many configurations is wider, not narrower, than any adjusted family
  would make it. Holm / Bonferroni is a recorded follow-up, not applied here.

The functions are pure and deterministic: they do no timing and read no clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

GO = "GO"
NO_GO = "NO_GO"
INCONCLUSIVE = "INCONCLUSIVE"

#: Every verdict :func:`ab_rule_verdict` can return.
VERDICTS = (GO, NO_GO, INCONCLUSIVE)

AT_LEAST = "at_least"
AT_MOST = "at_most"
DIRECTIONS = (AT_LEAST, AT_MOST)

#: The audit note's minimum for an A/B lever rule: at least 5 paired rounds
#: (section (a), "A/B lever / go rules").
MIN_AB_ROUNDS = 5

#: The campaigns' interval confidence (their 90 % bootstrap CIs).
AB_CONFIDENCE = 0.90


@dataclass(frozen=True)
class Criterion:
    """One measured effect, its interval and its pre-registered bar."""

    name: str
    point: float
    lower: float
    upper: float
    bar: float
    direction: str = AT_LEAST
    unit: str = ""


@dataclass(frozen=True)
class CriterionVerdict:
    name: str
    verdict: str
    point: float
    lower: float
    upper: float
    bar: float
    direction: str
    unit: str
    mdi: float
    reason: str


@dataclass(frozen=True)
class ABVerdict:
    """The rule's verdict, the per-criterion verdicts and the numbers behind them."""

    verdict: str
    reason: str
    n: int | None
    min_n: int
    confidence: float
    criteria: tuple[CriterionVerdict, ...]
    gates: dict = field(default_factory=dict)

    @property
    def go(self) -> bool:
        return self.verdict == GO

    def as_dict(self) -> dict:
        out = asdict(self)
        out["criteria"] = {c.name: asdict(c) for c in self.criteria}
        return out


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _fmt(value: float, unit: str) -> str:
    return f"{value:.4g}{(' ' + unit) if unit else ''}"


def criterion_verdict(criterion: Criterion) -> CriterionVerdict:
    """Judge one criterion's interval against its bar (step 4 of the rule)."""
    c = criterion
    if c.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {c.direction!r}")
    if not _finite(c.bar):
        raise ValueError(f"criterion {c.name!r}: the bar must be finite, got {c.bar!r}")

    def _out(verdict: str, reason: str, mdi: float) -> CriterionVerdict:
        return CriterionVerdict(
            name=c.name,
            verdict=verdict,
            point=float(c.point) if _finite(c.point) else float("nan"),
            lower=float(c.lower) if _finite(c.lower) else float("nan"),
            upper=float(c.upper) if _finite(c.upper) else float("nan"),
            bar=float(c.bar),
            direction=c.direction,
            unit=c.unit,
            mdi=mdi,
            reason=reason,
        )

    if not _finite(c.point, c.lower, c.upper):
        return _out(INCONCLUSIVE, f"{c.name}: invalid input (non-finite point or bound)", math.nan)
    point, lower, upper, bar = float(c.point), float(c.lower), float(c.upper), float(c.bar)
    if lower > upper:
        return _out(INCONCLUSIVE, f"{c.name}: invalid input (lower bound > upper)", math.nan)
    mdi = (upper - lower) / 2.0
    interval = f"[{_fmt(lower, c.unit)}, {_fmt(upper, c.unit)}]"
    sign = ">=" if c.direction == AT_LEAST else "<="
    if c.direction == AT_LEAST:
        clears = lower >= bar and point >= bar
        fails = upper < bar
    else:
        clears = upper <= bar and point <= bar
        fails = lower > bar
    if clears:
        return _out(GO, f"{c.name} {interval} clears {sign} {_fmt(bar, c.unit)}", mdi)
    if fails:
        return _out(
            NO_GO,
            f"{c.name} {interval} is wholly on the wrong side of {sign} {_fmt(bar, c.unit)}",
            mdi,
        )
    return _out(
        INCONCLUSIVE,
        f"{c.name} {interval} straddles the bar {sign} {_fmt(bar, c.unit)} "
        f"(point {_fmt(point, c.unit)}; resolvable effect +/-{_fmt(mdi, c.unit)})",
        mdi,
    )


def ab_rule_verdict(
    criteria: Sequence[Criterion],
    *,
    n,
    min_n: int = MIN_AB_ROUNDS,
    confidence: float = AB_CONFIDENCE,
    gates: Mapping[str, bool] | None = None,
) -> ABVerdict:
    """Judge a pre-registered conjunction of criteria (the module docstring's rule).

    Parameters
    ----------
    criteria
        The rule's criteria; every one must clear for ``GO``.
    n
        The number of independent units the intervals were computed over (rounds
        or blocks — never individual calls).
    min_n
        Below this ``n`` the verdict is ``INCONCLUSIVE``.
    confidence
        The intervals' confidence, recorded with the verdict.
    gates
        Correctness gates by name; any ``False`` is ``NO_GO`` before timing.
    """
    criteria = tuple(criteria)
    if not criteria:
        raise ValueError("a rule needs at least one criterion")
    if int(min_n) < 2:
        raise ValueError(f"min_n must be at least 2, got {min_n!r}")
    gates = {str(k): bool(v) for k, v in (gates or {}).items()}
    judged = tuple(criterion_verdict(c) for c in criteria)

    try:
        n_int = int(n)
        n_valid = n_int == n and n_int >= 0
    except (TypeError, ValueError, OverflowError):
        n_int, n_valid = None, False

    def _out(verdict: str, reason: str) -> ABVerdict:
        return ABVerdict(
            verdict=verdict,
            reason=reason,
            n=n_int if n_valid else None,
            min_n=int(min_n),
            confidence=float(confidence),
            criteria=judged,
            gates=gates,
        )

    red = [name for name, ok in gates.items() if not ok]
    if red:
        return _out(NO_GO, f"correctness gate(s) {red} red: not a lever, whatever its timing")
    if not n_valid:
        return _out(INCONCLUSIVE, f"invalid input: sample count {n!r}")
    invalid = [c for c in judged if c.reason.startswith(f"{c.name}: invalid input")]
    if invalid:
        return _out(INCONCLUSIVE, "; ".join(c.reason for c in invalid))
    if n_int < min_n:
        mdis = ", ".join(f"{c.name} +/-{_fmt(c.mdi, c.unit)}" for c in judged)
        return _out(
            INCONCLUSIVE,
            f"n = {n_int} < {min_n}: too few independent units to resolve the rule "
            f"(resolvable effect {mdis})",
        )
    failed = [c for c in judged if c.verdict == NO_GO]
    if failed:
        return _out(NO_GO, "; ".join(c.reason for c in failed))
    if all(c.verdict == GO for c in judged):
        return _out(GO, "; ".join(c.reason for c in judged))
    return _out(
        INCONCLUSIVE,
        "; ".join(c.reason for c in judged if c.verdict == INCONCLUSIVE)
        + " — INCONCLUSIVE, not a measured negative",
    )


@dataclass(frozen=True)
class TieSet:
    """An argmax that refuses to name a winner its intervals cannot separate."""

    leader: str | None
    members: tuple[str, ...]
    best: str | None
    reason: str
    higher_is_better: bool

    @property
    def resolved(self) -> bool:
        return self.best is not None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["members"] = list(self.members)
        out["resolved"] = self.resolved
        return out


def tie_set(
    candidates: Mapping[str, Sequence[float]],
    *,
    higher_is_better: bool = True,
    n=None,
    min_n: int = MIN_AB_ROUNDS,
) -> TieSet:
    """The leader by point estimate and every candidate its interval cannot separate.

    ``candidates`` maps a name to ``(point, lower, upper)``. A candidate whose
    interval overlaps the leader's is in the tie set; a candidate with a
    non-finite point or bound cannot be excluded and is in it too. ``best`` is
    the leader only when the tie set is the leader alone.

    ``n`` is the number of independent units (rounds) the intervals were computed
    over. Below ``min_n`` no interval can exclude anything (#362 fix phase 4: a
    round bootstrap over 3 rounds has 10 distinct resamples), so every candidate
    is in the tie set and ``best`` is ``None``.
    """
    items = {str(name): tuple(v) for name, v in candidates.items()}
    for name, v in items.items():
        if len(v) != 3:
            raise ValueError(f"candidate {name!r}: expected (point, lower, upper), got {v!r}")
    if not items:
        return TieSet(None, (), None, "no candidates", higher_is_better)
    finite = {k: tuple(float(x) for x in v) for k, v in items.items() if _finite(*v)}
    if not finite:
        members = tuple(items)
        return TieSet(None, members, None, "no candidate has a finite interval", higher_is_better)
    sign = 1.0 if higher_is_better else -1.0
    leader = max(finite, key=lambda name: (sign * finite[name][0], name))
    if n is not None:
        try:
            n_ok = int(n) == n and int(n) >= int(min_n)
        except (TypeError, ValueError, OverflowError):
            n_ok = False
        if not n_ok:
            return TieSet(
                leader,
                tuple(
                    sorted(
                        items,
                        key=lambda k: (-(sign * finite[k][0]) if k in finite else math.inf, k),
                    )
                ),
                None,
                f"n = {n!r} < {min_n} independent units: no interval can separate the "
                f"candidates, so every one is in the tie set",
                higher_is_better,
            )
    _, lead_lo, lead_hi = finite[leader]

    def _overlaps(name: str) -> bool:
        if name not in finite:
            return True
        _, lo, hi = finite[name]
        return hi >= lead_lo if higher_is_better else lo <= lead_hi

    members = tuple(
        sorted(
            (k for k in items if k == leader or _overlaps(k)),
            key=lambda k: (-(sign * finite[k][0]) if k in finite else math.inf, k),
        )
    )
    if members == (leader,):
        return TieSet(
            leader,
            members,
            leader,
            f"{leader}'s interval is clear of every other candidate's",
            higher_is_better,
        )
    return TieSet(
        leader,
        members,
        None,
        f"the intervals of {list(members)} overlap the point leader {leader}'s: a tie set, "
        f"not a single best",
        higher_is_better,
    )


def paired_block_ratio_interval(reference, candidate, *, confidence: float = AB_CONFIDENCE):
    """Two-sided Student-t interval on the mean of paired per-block ratios.

    ``reference`` and ``candidate`` are per-block summaries measured on the same
    instances in the same order (paired by instance). Returns
    ``(point, lower, upper, n)`` for ``candidate / reference``; the bounds are
    ``nan`` when fewer than two blocks exist or any input is invalid, which
    :func:`ab_rule_verdict` reads as INCONCLUSIVE.
    """
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    nan = float("nan")
    if (
        ref.ndim != 1
        or ref.shape != cand.shape
        or ref.size == 0
        or not (np.all(np.isfinite(ref)) and np.all(np.isfinite(cand)))
        or np.any(ref <= 0.0)
        or np.any(cand <= 0.0)
    ):
        return nan, nan, nan, int(min(ref.size, cand.size))
    ratios = cand / ref
    n = int(ratios.size)
    point = float(np.mean(ratios))
    if n < 2:
        return point, nan, nan, n
    from scipy.stats import t

    se = float(np.std(ratios, ddof=1)) / math.sqrt(n)
    margin = float(t.ppf(0.5 + confidence / 2.0, df=n - 1)) * se
    return point, point - margin, point + margin, n
