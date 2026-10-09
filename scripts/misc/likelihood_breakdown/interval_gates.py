"""Interval verdicts for the memo-policy, scaling and log-det gates (#362, fix phase 3b).

Why this module exists
----------------------

``results/notes/timing_noise_audit_2026_10.md`` rows C1, C3, C4 and C5 found
four campaign gates labelling a result from point estimates alone:

- **C1** ``fixed_light_numba_memo_policy.matched_counterfactual`` classified a
  memo/cold median ratio of 4 paired repeats as "beneficial" (< 0.97) or
  "harmful" (> 1.03), and those flags were summed into ``false_accepts`` /
  ``false_rejects`` as if they were measured decision errors;
- **C3** the memo-policy verdict was GO when six point ratios held and
  NO_LEVER otherwise, with no interval and no multiple-comparison logic;
- **C4** ``fixed_light_numba_scaling.evaluate_group``'s ``breakdown_reconciles``
  was ``|observed median / clean median - 1| <= 0.05`` on points, feeding
  ``status: PASS``;
- **C5** ``fixed_light_trace``'s log-det lever ``clears_threshold`` was two
  point savings of the same quantity both >= ``LOGDET_LEVER_MS``.

Each gate now reads an interval through the shared tools — the paired
whole-round bootstrap (:mod:`likelihood_breakdown.round_bootstrap`) on its own
pairing unit, and the verdict (:mod:`likelihood_breakdown.ab_verdict`) — and
has an INCONCLUSIVE state. No bar, band, repeat count or protocol is changed.

Pairing units (each gate's own data layout)
-------------------------------------------

- C1: the 4 solver-only repeats of one transition; cold and memo alternate
  order inside a repeat, so a repeat is the round (1 call per arm).
- C3: the ``--n-repeats`` traversals of one evaluation; every repeat runs the
  three lanes once in a rotated order (``SCHEDULE``), so a repeat is the round
  (1 sequence total per lane).
- C4: the ``--n-repeats`` repeats of one group; each repeat runs a clean and an
  instrumented traversal of the lane (order counterbalanced), so a repeat is
  the round.
- C5: the 7 interleaved rounds of one task; each round is one 10-call block of
  the library control then one of the candidate. The ``jit_profile`` estimator
  is one 10-call block per arm: no interval can be computed for it.

Every function is pure and deterministic (fixed bootstrap seeds); none reads a
clock. ``scripts/misc/test/test_interval_gates.py`` holds the witnesses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from likelihood_breakdown.ab_verdict import (
    AB_CONFIDENCE,
    AT_LEAST,
    AT_MOST,
    GO,
    INCONCLUSIVE,
    MIN_AB_ROUNDS,
    NO_GO,
    Criterion,
    ab_rule_verdict,
    bootstrap_criterion,
    conjoin_verdicts,
    holm_family_verdict,
)
from likelihood_breakdown.logdet_reuse_injection import LOGDET_LEVER_MS
from likelihood_breakdown.round_bootstrap import (
    RESAMPLING,
    round_median_ratio,
    round_median_saving,
)

PASS = "PASS"
FAIL = "FAIL"

# ---------------------------------------------------------------------------
# C1: matched counterfactual classification
# ---------------------------------------------------------------------------

#: The pre-registered 3 % neutral band of ``matched_counterfactual`` (unchanged).
MATCHED_BENEFICIAL_BELOW = 0.97
MATCHED_HARMFUL_ABOVE = 1.03
#: Bootstrap seed (the memo-policy issue, #280).
MATCHED_SEED = 280

BENEFICIAL = "beneficial"
HARMFUL = "harmful"
NEUTRAL = "neutral"
#: Every class :func:`matched_classification` can return.
MATCHED_CLASSES = (BENEFICIAL, HARMFUL, NEUTRAL, INCONCLUSIVE)


def matched_classification(
    samples_ms: Mapping[str, Sequence[float]],
    *,
    seed: int = MATCHED_SEED,
    min_n: int = MIN_AB_ROUNDS,
) -> dict:
    """C1: classify one transition's memo/cold ratio only when its interval resolves.

    ``samples_ms`` is ``{"cold": [...], "memo": [...]}``, one solver call per arm
    per repeat, in repeat order (paired by repeat). The interval is the paired
    round bootstrap of ``median(memo) / median(cold)`` (the published point).

    ``beneficial`` when the whole interval and the point are <= 0.97;
    ``harmful`` when they are >= 1.03; ``neutral`` when the interval is wholly
    inside (0.97, 1.03); ``INCONCLUSIVE`` otherwise, and whenever fewer than
    ``min_n`` repeats exist (the audit's 5-round minimum; the cell records 4).
    """
    cold, memo = list(samples_ms["cold"]), list(samples_ms["memo"])
    if len(cold) != len(memo) or not cold:
        raise ValueError("cold and memo must hold the same, non-zero number of paired repeats")
    n = len(cold)
    ratio = round_median_ratio(memo, cold, n_rounds=n, seed=seed)
    interval = (ratio["ratio"], ratio["ci90_low"], ratio["ci90_high"])
    beneficial = ab_rule_verdict(
        [Criterion("memo_over_cold", *interval, MATCHED_BENEFICIAL_BELOW, AT_MOST)],
        n=n,
        min_n=min_n,
    )
    harmful = ab_rule_verdict(
        [Criterion("memo_over_cold", *interval, MATCHED_HARMFUL_ABOVE, AT_LEAST)],
        n=n,
        min_n=min_n,
    )
    if beneficial.verdict == GO:
        cls, reason = BENEFICIAL, beneficial.reason
    elif harmful.verdict == GO:
        cls, reason = HARMFUL, harmful.reason
    elif beneficial.verdict == NO_GO and harmful.verdict == NO_GO:
        cls = NEUTRAL
        reason = (
            f"memo_over_cold [{interval[1]:.4g}, {interval[2]:.4g}] is wholly inside the "
            f"({MATCHED_BENEFICIAL_BELOW}, {MATCHED_HARMFUL_ABOVE}) neutral band"
        )
    else:
        cls = INCONCLUSIVE
        reason = beneficial.reason if beneficial.verdict == INCONCLUSIVE else harmful.reason
    return {
        "classification": cls,
        "memo_over_cold": interval[0],
        "ci90_low": interval[1],
        "ci90_high": interval[2],
        "effective_n": n,
        "min_n": int(min_n),
        "resampling": RESAMPLING,
        "band": [MATCHED_BENEFICIAL_BELOW, MATCHED_HARMFUL_ABOVE],
        "reason": reason,
    }


def matched_decision_counts(rows: Iterable[tuple[bool, str]]) -> dict:
    """C1: the matched transitions' resolved classifications and INCONCLUSIVE count.

    ``rows`` holds ``(precheck accepted, classification)`` per eligible
    transition. The counts are resolved classifications of the solver-only
    counterfactual, never measured decision errors: an INCONCLUSIVE transition
    is neither a correct nor a wrong decision.
    """
    rows = [(bool(a), str(c)) for a, c in rows]
    unknown = sorted({c for _, c in rows} - set(MATCHED_CLASSES))
    if unknown:
        raise ValueError(f"unknown classification(s) {unknown}")
    return {
        "eligible_transitions": len(rows),
        "accepted": sum(a for a, _ in rows),
        "rejected": sum(not a for a, _ in rows),
        "resolved_harmful_accepted": sum(a and c == HARMFUL for a, c in rows),
        "resolved_beneficial_rejected": sum(not a and c == BENEFICIAL for a, c in rows),
        "resolved_beneficial_accepted": sum(a and c == BENEFICIAL for a, c in rows),
        "resolved_harmful_rejected": sum(not a and c == HARMFUL for a, c in rows),
        "resolved_neutral": sum(c == NEUTRAL for _, c in rows),
        "inconclusive": sum(c == INCONCLUSIVE for _, c in rows),
        "qualification": (
            "resolved classifications of a solver-only counterfactual (paired round-bootstrap "
            f"90 % interval vs the {MATCHED_BENEFICIAL_BELOW}/{MATCHED_HARMFUL_ABOVE} band, "
            f">= {MIN_AB_ROUNDS} repeats); not measured decision-error rates. An INCONCLUSIVE "
            "transition is neither a correct nor a wrong decision."
        ),
    }


# ---------------------------------------------------------------------------
# C3: the six-target memo-policy verdict, family-wise
# ---------------------------------------------------------------------------

#: The pre-registered targets (unchanged): name -> (evaluation, reference lane, bar).
#: Each is ``median(guarded) / median(reference) <= bar`` over the sequence totals.
MEMO_POLICY_TARGETS = {
    "graded_5pct_vs_memo": ("graded", "memo", 0.95),
    "permuted_5pct_vs_memo": ("permuted", "memo", 0.95),
    "graded_3pct_vs_cold": ("graded", "cold", 1.03),
    "permuted_3pct_vs_cold": ("permuted", "cold", 1.03),
    "broad_holdout_3pct_vs_cold": ("broad_holdout", "cold", 1.03),
    "nearby_holdout_3pct_vs_memo": ("nearby_holdout", "memo", 1.03),
}
MEMO_POLICY_SEED = 280

#: The memo-policy verdicts: the family's GO / NO_GO / INCONCLUSIVE, named as the cell does.
MEMO_POLICY_VERDICTS = {GO: "GO", NO_GO: "NO_LEVER", INCONCLUSIVE: INCONCLUSIVE}


def memo_policy_family(
    evaluations: Mapping[str, Mapping],
    *,
    seed: int = MEMO_POLICY_SEED,
    min_n: int = MIN_AB_ROUNDS,
    confidence: float = AB_CONFIDENCE,
) -> dict:
    """C3: the six targets as one Holm family (``ab_verdict.holm_family_verdict``).

    ``evaluations`` maps each evaluation name to its record carrying
    ``sequence_totals_ms`` (lane -> one total per repeat, paired by repeat). The
    verdict is ``GO`` only if all six resolve in favour, ``NO_LEVER`` only if at
    least one resolves against, ``INCONCLUSIVE`` otherwise.
    """
    members, n_rounds, targets = {}, [], {}
    for name, (run, reference, bar) in MEMO_POLICY_TARGETS.items():
        totals = evaluations[run]["sequence_totals_ms"]
        guarded, ref = list(totals["guarded"]), list(totals[reference])
        if len(guarded) != len(ref) or not guarded:
            raise ValueError(f"{name}: guarded and {reference} totals are not paired by repeat")
        ratio, boots = round_median_ratio(
            guarded, ref, n_rounds=len(guarded), seed=seed, return_boots=True
        )
        members[name] = bootstrap_criterion(name, ratio["ratio"], boots, bar, AT_MOST)
        n_rounds.append(len(guarded))
        targets[name] = {
            "evaluation": run,
            "ratio": f"guarded / {reference}",
            "bar": bar,
            "point": ratio["ratio"],
            "ci90_low": ratio["ci90_low"],
            "ci90_high": ratio["ci90_high"],
            "effective_n": ratio["effective_n"],
        }
    family = holm_family_verdict(members, n=min(n_rounds), min_n=min_n, confidence=confidence)
    out = family.as_dict()
    out["verdict"] = MEMO_POLICY_VERDICTS[family.verdict]
    out["targets"] = targets
    out["resampling"] = RESAMPLING
    out["target_verdicts"] = {
        c.name: c.verdict for c in family.adjusted.criteria
    }  # per target, at its Holm level
    return out


# ---------------------------------------------------------------------------
# C4: breakdown reconciliation
# ---------------------------------------------------------------------------

#: The pre-registered reconciliation band (unchanged): |observed / clean - 1| <= 0.05.
RECONCILE_TOLERANCE = 0.05
#: Bootstrap seed (the scaling declaration's seed).
RECONCILE_SEED = 601
_RECONCILE_FROM_AB = {GO: PASS, NO_GO: FAIL, INCONCLUSIVE: INCONCLUSIVE}
_RECONCILE_TO_AB = {v: k for k, v in _RECONCILE_FROM_AB.items()}


def breakdown_reconciliation(
    observed_totals: Sequence[float],
    clean_totals: Sequence[float],
    *,
    seed: int = RECONCILE_SEED,
    min_n: int = MIN_AB_ROUNDS,
) -> dict:
    """C4: ``median(observed) / median(clean) - 1`` against ``+-0.05`` on an interval.

    ``observed_totals`` (instrumented) and ``clean_totals`` hold one sequence
    total per repeat, paired by repeat. ``PASS`` when the whole interval is inside
    the band, ``FAIL`` when it is wholly outside one side, ``INCONCLUSIVE``
    otherwise and below ``min_n`` repeats.
    """
    observed, clean = list(observed_totals), list(clean_totals)
    if len(observed) != len(clean) or not observed:
        raise ValueError("observed and clean totals must be paired by repeat")
    n = len(observed)
    r = round_median_ratio(observed, clean, n_rounds=n, seed=seed)
    point, low, high = r["ratio"] - 1.0, r["ci90_low"] - 1.0, r["ci90_high"] - 1.0
    verdict = ab_rule_verdict(
        [
            Criterion("relative_at_most_plus_band", point, low, high, RECONCILE_TOLERANCE, AT_MOST),
            Criterion(
                "relative_at_least_minus_band", point, low, high, -RECONCILE_TOLERANCE, AT_LEAST
            ),
        ],
        n=n,
        min_n=min_n,
    )
    return {
        "verdict": _RECONCILE_FROM_AB[verdict.verdict],
        "relative": point,
        "ci90_low": low,
        "ci90_high": high,
        "tolerance": RECONCILE_TOLERANCE,
        "effective_n": n,
        "min_n": int(min_n),
        "resampling": RESAMPLING,
        "reason": verdict.reason,
    }


def conjoin_reconciliations(verdicts: Sequence[str]) -> str:
    """C4: a cell's lanes together — PASS only if every lane is PASS, FAIL if one is FAIL.

    ``ab_verdict.conjoin_verdicts`` in the PASS / FAIL / INCONCLUSIVE vocabulary.
    """
    return _RECONCILE_FROM_AB[conjoin_verdicts([_RECONCILE_TO_AB[v] for v in verdicts])]


# ---------------------------------------------------------------------------
# C5: the log-det lever
# ---------------------------------------------------------------------------

#: Bootstrap seed (the log-det issue, #303).
LOGDET_SEED = 303


def logdet_lever_verdict(
    interleaved: Mapping,
    saving_jit_profile_ms: float,
    *,
    threshold_ms: float = LOGDET_LEVER_MS,
    seed: int = LOGDET_SEED,
    min_n: int = MIN_AB_ROUNDS,
) -> dict:
    """C5: the pre-registered "both estimators >= threshold" rule, on intervals.

    The interleaved estimator's interval is the paired round bootstrap of
    ``median(library) - median(candidate)`` over its rounds (one 10-call block
    per arm per round). The ``jit_profile`` estimator is one 10-call block per
    arm: no interval exists, so it cannot resolve and is INCONCLUSIVE by
    construction. The rule is their conjunction
    (``ab_verdict.conjoin_verdicts``): a resolved NO_GO on the interleaved
    interval is NO_GO; GO needs both resolved, which the current protocol cannot
    give. ``clears_threshold`` is ``True`` / ``False`` only when resolved, else
    ``"INCONCLUSIVE"``.
    """
    library = list(interleaved["library_ms"])
    candidate = list(interleaved["candidate_ms"])
    if len(library) != len(candidate) or not library:
        raise ValueError("interleaved library and candidate blocks must be paired by round")
    n = len(library)
    s = round_median_saving(library, candidate, n_rounds=n, seed=seed, scale=1.0)
    ab = ab_rule_verdict(
        [
            Criterion(
                "saving_interleaved_median_ms",
                s["saved_ms"],
                s["ci90_low"],
                s["ci90_high"],
                threshold_ms,
                AT_LEAST,
                "ms",
            )
        ],
        n=n,
        min_n=min_n,
    )
    jit = {
        "verdict": INCONCLUSIVE,
        "point_ms": float(saving_jit_profile_ms),
        "effective_n": 1,
        "reason": (
            "jit_profile is one 10-call block per arm: no interval can be computed, so this "
            "estimator cannot resolve against the threshold (INCONCLUSIVE by construction)"
        ),
    }
    verdict = conjoin_verdicts([ab.verdict, jit["verdict"]])
    return {
        "verdict": verdict,
        "clears_threshold": {GO: True, NO_GO: False, INCONCLUSIVE: INCONCLUSIVE}[verdict],
        "threshold_ms": threshold_ms,
        "interleaved": {
            "verdict": ab.verdict,
            "saving_ms": s["saved_ms"],
            "ci90_low": s["ci90_low"],
            "ci90_high": s["ci90_high"],
            "effective_n": n,
            "min_n": int(min_n),
            "resampling": RESAMPLING,
            "reason": ab.reason,
        },
        "jit_profile": jit,
        "reason": (
            "the interleaved interval resolves below the threshold: not a lever"
            if verdict == NO_GO
            else f"interleaved: {ab.reason}; jit_profile: {jit['reason']}"
        ),
    }
