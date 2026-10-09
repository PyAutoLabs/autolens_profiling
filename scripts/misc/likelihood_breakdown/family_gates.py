"""Family-wise verdicts for the multi-comparison gates, and two remainders (#362, fix phase 9).

Why this module exists
----------------------

``results/notes/timing_noise_audit_2026_10.md`` section (a), "Multiple
comparisons", found three gates whose verdict reads many comparisons, each
judged at 90 % with no adjustment:

- **C6** ``backward_pass_ab._phase2c_rule``: four routes x two lanes x two
  criteria on one host, from which the phase-2c decision picks the GO routes;
- **C10** ``solver_config_sweep._fastest``: a tie set over many
  configurations, from which a single "best admissible" is named;
- **C11** ``numba_interferometer/bakeoff.py``'s kill gate: the best of six
  numba kernels against ``rfft2_numpy`` in four cells, on point medians with no
  interval at all ("ratio > 1.3").

Each is now judged under the one family-wise policy stated in
:mod:`likelihood_breakdown.ab_verdict` (Holm, family-wise 90 %, family = the
comparisons one verdict or claim rests on), through
:func:`~likelihood_breakdown.ab_verdict.holm_family_verdict` and
:func:`~likelihood_breakdown.ab_verdict.holm_tie_set`, on paired whole-round
bootstrap draws (:mod:`likelihood_breakdown.round_bootstrap`). The unadjusted
reading is recorded beside every family-wise one.

Two remainders of the audit live here too:

- **C12** ``gpu_bottleneck_map._mdi``: the split-half minimum detectable
  improvement moves from an iid bootstrap of the first half of the calls
  against the second half onto a paired round bootstrap of each round's first
  calls against its last calls (:func:`round_split_half_mdi`);
- **P2** ``fixed_light_numba``'s promotion: the two rows are separate passes,
  so drift between them is not cancelled by the paired-block interval;
  :func:`between_row_drift` records it and the promotion is INCONCLUSIVE when
  the rows drift.

Families (what each verdict rests on)
-------------------------------------

- C6: one host's run — every timed (route, lane) x {saved ms, ratio}
  criterion. The phase-2c decision is "which routes are GO"; a false GO on any
  of them is the error to control. A route is GO for the host only when it is
  GO in every lane (the decision was taken per route, on both lanes).
- C10: one sweep's candidates — the leader against each other candidate
  (``holm_tie_set``).
- C11: one bake-off — every pinned numba kernel in every sma / alma cell
  against that cell's ``rfft2_numpy``. "passed" (numba survives) is an "any"
  claim: some member resolves GO at its Holm level; "tripped" (kill) needs
  every member resolved NO_GO; anything else is INCONCLUSIVE.

Every function is pure and deterministic (fixed seeds); none reads a clock.
``scripts/misc/test/test_family_gates.py`` holds the witnesses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from likelihood_breakdown.ab_verdict import (
    AB_CONFIDENCE,
    AT_LEAST,
    AT_MOST,
    GO,
    INCONCLUSIVE,
    MIN_AB_ROUNDS,
    NO_GO,
    ab_rule_verdict,
    bootstrap_criterion,
    bootstrap_interval,
    conjoin_verdicts,
    holm_family_verdict,
    holm_tie_set,
)
from likelihood_breakdown.round_bootstrap import (
    RESAMPLING,
    ROUND_BOOTSTRAP_CONFIDENCE,
    ROUND_BOOTSTRAP_SAMPLES,
    as_rounds,
    round_median_ratio,
    round_median_saving,
)

#: How a family-wise record names its procedure.
FAMILY_METHOD = "holm"


def _n_ok(n, min_n) -> bool:
    try:
        return int(n) == n and int(n) >= int(min_n)
    except (TypeError, ValueError, OverflowError):
        return False


def _any_claim(verdicts: Sequence[str], n, min_n) -> str:
    """An "any member clears" claim from final per-member verdicts (the policy's shape)."""
    if not verdicts or not _n_ok(n, min_n):
        return INCONCLUSIVE
    if GO in verdicts:
        return GO
    if all(v == NO_GO for v in verdicts):
        return NO_GO
    return INCONCLUSIVE


# ---------------------------------------------------------------------------
# C6: the backward-pass phase-2c rule, one host's routes x lanes as one family
# ---------------------------------------------------------------------------


def phase2c_family(
    grad_rows: Mapping[str, Mapping],
    *,
    routes: Sequence[str],
    saved_bar_ms: float,
    fraction_bar: float,
    seed: int,
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    min_n: int = MIN_AB_ROUNDS,
    confidence: float = AB_CONFIDENCE,
) -> dict:
    """C6: every timed (lane, route) x {saved ms, ratio} criterion of one host as one Holm family.

    ``grad_rows`` maps each lane to its ``<lane>_grad`` row as the cell writes it
    (``per_call_ms`` in ms per route, ``timed_routes``, ``control``,
    ``n_rounds``). The draws are the cell's own: the ratio of the route at
    position ``i`` of ``timed_routes`` uses ``seed + i`` and its saving
    ``seed + 100 + i``, so every 90 % interval equals the one the row records.
    A route that was not timed (its correctness gate is red) is NO_GO before
    timing and is not a member.

    Returns ``per_lane[lane][route]`` (``verdict`` family-wise,
    ``verdict_unadjusted``, the criteria at their Holm levels), ``per_route``
    (GO only when GO in every lane), ``go_routes`` and the ``family`` record.
    """
    lanes = list(grad_rows)
    members, units, timed_pairs = {}, [], []
    for lane in lanes:
        row = grad_rows[lane]
        control = row["control"]
        timed = list(row["timed_routes"])
        n_rounds = int(row["n_rounds"])
        control_ms = np.asarray(row["per_call_ms"][control], dtype=float)
        for route in routes:
            if route == control or route not in timed:
                continue
            i = timed.index(route)
            route_ms = np.asarray(row["per_call_ms"][route], dtype=float)
            ratio, ratio_boots = round_median_ratio(
                route_ms,
                control_ms,
                n_rounds=n_rounds,
                seed=seed + i,
                samples=samples,
                return_boots=True,
            )
            saved, saved_boots = round_median_saving(
                control_ms,
                route_ms,
                n_rounds=n_rounds,
                seed=seed + 100 + i,
                scale=1.0,
                samples=samples,
                return_boots=True,
            )
            members[f"{lane}/{route}/saved_ms"] = bootstrap_criterion(
                f"{lane}/{route}/saved_ms",
                saved["saved_ms"],
                saved_boots,
                saved_bar_ms,
                AT_LEAST,
                "ms",
            )
            members[f"{lane}/{route}/ratio"] = bootstrap_criterion(
                f"{lane}/{route}/ratio", ratio["ratio"], ratio_boots, 1.0 - fraction_bar, AT_MOST
            )
            units.append(n_rounds)
            timed_pairs.append((lane, route))

    per_lane: dict = {lane: {} for lane in lanes}
    for lane in lanes:
        timed = set(grad_rows[lane]["timed_routes"])
        for route in routes:
            if route != grad_rows[lane]["control"] and route not in timed:
                per_lane[lane][route] = {
                    "verdict": NO_GO,
                    "verdict_unadjusted": NO_GO,
                    "reason": "not timed: its correctness gate is red",
                }
    if not members:
        family_record = {"method": FAMILY_METHOD, "k": 0, "verdict": INCONCLUSIVE}
    else:
        n = min(units)
        family = holm_family_verdict(members, n=n, min_n=min_n, confidence=confidence)
        final = {c.name: c for c in family.adjusted.criteria}
        flat = {c.name: c for c in family.unadjusted.criteria}
        for lane, route in timed_pairs:
            names = (f"{lane}/{route}/saved_ms", f"{lane}/{route}/ratio")
            adjusted = ab_rule_verdict(
                [members[k](family.member_confidence[k]) for k in names],
                n=n,
                min_n=min_n,
                confidence=confidence,
                gates={"correctness": True},
            )
            unadjusted = ab_rule_verdict(
                [members[k](confidence) for k in names],
                n=n,
                min_n=min_n,
                confidence=confidence,
                gates={"correctness": True},
            )
            per_lane[lane][route] = {
                "verdict": adjusted.verdict,
                "verdict_unadjusted": unadjusted.verdict,
                "reason": adjusted.reason,
                "member_confidence": {
                    k.rsplit("/", 1)[1]: family.member_confidence[k] for k in names
                },
                "criteria": {
                    k.rsplit("/", 1)[1]: {
                        "point": final[k].point,
                        "lower": final[k].lower,
                        "upper": final[k].upper,
                        "bar": final[k].bar,
                        "verdict": final[k].verdict,
                        "unadjusted_90": [flat[k].lower, flat[k].upper],
                    }
                    for k in names
                },
            }
        family_record = {
            "method": FAMILY_METHOD,
            "family": "routes x lanes x {saved_ms, ratio} of this host's run",
            "k": len(members),
            "n": n,
            "family_confidence": confidence,
            "levels": list(family.levels),
            "steps": [dict(s) for s in family.steps],
            "resampling": RESAMPLING,
        }

    per_route = {}
    for route in routes:
        verdicts = [per_lane[lane][route]["verdict"] for lane in lanes if route in per_lane[lane]]
        flat_verdicts = [
            per_lane[lane][route]["verdict_unadjusted"] for lane in lanes if route in per_lane[lane]
        ]
        if not verdicts:
            continue
        per_route[route] = {
            "verdict": conjoin_verdicts(verdicts),
            "verdict_unadjusted": conjoin_verdicts(flat_verdicts),
        }
    return {
        "per_lane": per_lane,
        "per_route": per_route,
        "go_routes": sorted(r for r, v in per_route.items() if v["verdict"] == GO),
        "go_routes_unadjusted": sorted(
            r for r, v in per_route.items() if v["verdict_unadjusted"] == GO
        ),
        "family": family_record,
    }


# ---------------------------------------------------------------------------
# C10: the solver sweep's tie set at the family-wise level
# ---------------------------------------------------------------------------


def sweep_tie_set(
    rows: Mapping[str, Mapping],
    names: Sequence[str],
    *,
    n_rounds: int,
    seed: int,
    control: str = "control",
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    min_n: int = MIN_AB_ROUNDS,
    confidence: float = AB_CONFIDENCE,
):
    """C10: :func:`holm_tie_set` over the candidates' speed-up draws (control / config).

    ``rows`` is the sweep's ``rows`` (``per_call_ms`` per configuration, in the
    order the configurations were timed); the draws are the cell's own (seed
    ``seed + position of the configuration in rows``), so every 90 % interval
    equals the recorded ``speedup_vs_control``. Returns a
    :class:`~likelihood_breakdown.ab_verdict.FamilyTieSet`.
    """
    order = list(rows)
    control_ms = np.asarray(rows[control]["per_call_ms"], dtype=float)
    candidates = {}
    for name in names:
        ratio, boots = round_median_ratio(
            control_ms,
            np.asarray(rows[name]["per_call_ms"], dtype=float),
            n_rounds=n_rounds,
            seed=seed + order.index(name),
            samples=samples,
            return_boots=True,
        )
        candidates[name] = bootstrap_interval(ratio["ratio"], boots)
    return holm_tie_set(
        candidates, higher_is_better=True, n=n_rounds, min_n=min_n, confidence=confidence
    )


# ---------------------------------------------------------------------------
# C11: the numba-interferometer kill gate, kernels x cells as one family
# ---------------------------------------------------------------------------

#: The pre-registered kill-gate margin (unchanged): numba must beat rfft2 by > 1.3x.
KILL_GATE_THRESHOLD = 1.3
KILL_GATE_INSTRUMENTS = ("sma", "alma")
KILL_GATE_REFERENCE = "rfft2_numpy"
#: Bootstrap seed (the bake-off's synthetic-input seed date, 2026-09-07).
KILL_GATE_SEED = 20260907
#: The gate's three outcomes, named as the cell does.
KILL_GATE_VERDICTS = {GO: "passed", NO_GO: "tripped", INCONCLUSIVE: INCONCLUSIVE}


def _timed_rounds(record: Mapping) -> list[float]:
    """The bake-off's timed rounds: round 0 is discarded unless it is the only one."""
    samples = list(record.get("all_rounds_s") or [])
    return samples[1:] if len(samples) > 1 else samples


def kill_gate_point(cells: Mapping[str, Mapping], *, numba_kernels: Sequence[str]) -> dict:
    """The pre-phase-9 rule, kept for continuity: best numba median vs rfft2 median > 1.3x."""
    per_cell, tripped = {}, True
    for key, cell in cells.items():
        if key.split("/")[0] not in KILL_GATE_INSTRUMENTS:
            continue
        rfft = cell["kernels"].get(KILL_GATE_REFERENCE, {}).get("median_s")
        best_name, best = None, None
        for name in numba_kernels:
            median = cell["kernels"].get(name, {}).get("median_s")
            if median is not None and (best is None or median < best):
                best, best_name = median, name
        ratio = (rfft / best) if (rfft and best) else None
        per_cell[key] = {
            "best_numba_kernel": best_name,
            "best_numba_s": best,
            "rfft2_numpy_s": rfft,
            "speedup_rfft2_over_numba": ratio,
        }
        if ratio is not None and ratio > KILL_GATE_THRESHOLD:
            tripped = False
    return {"kill_gate": "tripped" if tripped else "passed", "per_cell": per_cell}


def kill_gate_family(
    cells: Mapping[str, Mapping],
    *,
    numba_kernels: Sequence[str],
    threshold: float = KILL_GATE_THRESHOLD,
    seed: int = KILL_GATE_SEED,
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    min_n: int = MIN_AB_ROUNDS,
    confidence: float = AB_CONFIDENCE,
) -> dict:
    """C11: every measured numba kernel x sma / alma cell vs ``rfft2_numpy`` as one Holm family.

    A member is ``median(rfft2) / median(kernel)`` over the cell's timed rounds
    (every kernel runs once per round, round-robin, so rounds pair the arms),
    with a paired round-bootstrap interval, against ``> threshold``. The gate
    is "passed" (numba survives) when any member resolves GO at its Holm level,
    "tripped" (kill) when every member resolves NO_GO, INCONCLUSIVE otherwise or
    below ``min_n`` timed rounds. Unpinned, skipped and unmeasured kernels are
    not members (they have no rounds), and are listed.
    """
    members, units, records, excluded = {}, [], {}, {}
    j = 0
    for key, cell in cells.items():
        if key.split("/")[0] not in KILL_GATE_INSTRUMENTS:
            continue
        kernels = cell["kernels"]
        reference = _timed_rounds(kernels.get(KILL_GATE_REFERENCE, {}))
        for name in numba_kernels:
            record = kernels.get(name, {})
            timed = _timed_rounds(record)
            member = f"{key}/{name}"
            if not timed or record.get("median_s") is None or not reference:
                excluded[member] = record.get("status", "not measured")
                continue
            j += 1
            if len(timed) != len(reference):
                excluded[member] = (
                    f"{len(timed)} timed rounds vs {len(reference)} for {KILL_GATE_REFERENCE}: "
                    "not paired by round"
                )
                continue
            ratio, boots = round_median_ratio(
                reference,
                timed,
                n_rounds=len(timed),
                seed=seed + j,
                samples=samples,
                return_boots=True,
            )
            members[member] = bootstrap_criterion(
                member, ratio["ratio"], boots, threshold, AT_LEAST
            )
            units.append(len(timed))
            records[member] = {
                "speedup_rfft2_over_kernel": ratio["ratio"],
                "ci90": [ratio["ci90_low"], ratio["ci90_high"]],
                "effective_n": ratio["effective_n"],
            }
    point = kill_gate_point(cells, numba_kernels=numba_kernels)
    out = {
        "rule": (
            f"numba survives ('passed') iff some pinned numba kernel beats {KILL_GATE_REFERENCE} "
            f"by > {threshold}x at sma or alma on a paired round-bootstrap interval, Holm over "
            f"kernels x cells at family-wise {confidence:.0%}; 'tripped' (kill) iff every member "
            f"is resolved <= {threshold}x; INCONCLUSIVE otherwise or below {min_n} timed rounds"
        ),
        "threshold": threshold,
        "kill_gate_point": point["kill_gate"],
        "per_cell_point": point["per_cell"],
        "members": records,
        "not_members": excluded,
        "resampling": RESAMPLING,
    }
    if not members:
        out.update(
            {
                "kill_gate": INCONCLUSIVE,
                "kill_gate_unadjusted": INCONCLUSIVE,
                "reason": "no numba kernel has timed rounds paired with rfft2_numpy",
            }
        )
        return out
    n = min(units)
    family = holm_family_verdict(members, n=n, min_n=min_n, confidence=confidence)
    final = {c.name: c for c in family.adjusted.criteria}
    flat = {c.name: c for c in family.unadjusted.criteria}
    adjusted = _any_claim([c.verdict for c in final.values()], n, min_n)
    unadjusted = _any_claim([c.verdict for c in flat.values()], n, min_n)
    for name, rec in records.items():
        rec["verdict"] = final[name].verdict
        rec["verdict_unadjusted"] = flat[name].verdict
        rec["member_confidence"] = family.member_confidence[name]
        rec["interval_at_member_confidence"] = [final[name].lower, final[name].upper]
    if not _n_ok(n, min_n):
        reason = (
            f"{n} timed rounds < {min_n}: too few independent rounds to resolve the kill gate "
            f"(the point rule reads '{point['kill_gate']}') — INCONCLUSIVE by construction, not "
            "a measured pass or kill"
        )
    elif adjusted == GO:
        reason = f"{sorted(k for k, c in final.items() if c.verdict == GO)} resolve > {threshold}x"
    elif adjusted == NO_GO:
        reason = f"every member resolves <= {threshold}x"
    else:
        reason = "no member resolves > the margin and not every member resolves below it"
    out.update(
        {
            "kill_gate": KILL_GATE_VERDICTS[adjusted],
            "kill_gate_unadjusted": KILL_GATE_VERDICTS[unadjusted],
            "reason": reason,
            "family": {
                "method": FAMILY_METHOD,
                "family": "pinned numba kernels x sma / alma cells",
                "k": len(members),
                "n": n,
                "family_confidence": confidence,
                "levels": list(family.levels),
                "steps": [dict(s) for s in family.steps],
            },
        }
    )
    return out


# ---------------------------------------------------------------------------
# C12: the split-half minimum detectable improvement on paired rounds
# ---------------------------------------------------------------------------


def round_split_half_mdi(
    samples_ms,
    *,
    n_rounds: int,
    seed: int,
    samples: int = ROUND_BOOTSTRAP_SAMPLES,
    confidence: float = ROUND_BOOTSTRAP_CONFIDENCE,
    min_n: int = MIN_AB_ROUNDS,
) -> dict:
    """C12: the 90 % half-width of a null A/B of one program against itself, paired by round.

    ``samples_ms`` is one route's per-call times in round order (``n_calls`` per
    round). Each round's first ``n_calls // 2`` calls are arm A and its last
    ``n_calls // 2`` arm B (the middle call is dropped when ``n_calls`` is odd):
    two adjacent blocks of the same round, the slots a real A/B of two routes
    occupies. The ratio ``median(A) / median(B)`` is bootstrapped over whole
    rounds with the same indices for both arms; the MDI is the larger distance
    of the interval from 1. Fewer than 2 calls per round or ``min_n`` rounds
    gives ``mdi`` NaN with the reason.
    """
    arr = as_rounds(samples_ms, n_rounds)
    n_calls = arr.shape[1]
    half = n_calls // 2
    out = {
        "estimator": "paired round bootstrap of median(first half of each round's calls) / "
        "median(last half)",
        "n_rounds": int(arr.shape[0]),
        "calls_per_arm_per_round": int(half),
        "resampling": RESAMPLING,
    }
    if half < 1 or arr.shape[0] < int(min_n):
        out.update(
            {
                "mdi": math.nan,
                "reason": f"{arr.shape[0]} rounds x {n_calls} calls: needs >= {min_n} rounds and "
                ">= 2 calls per round",
            }
        )
        return out
    ratio = round_median_ratio(
        arr[:, :half],
        arr[:, n_calls - half :],
        n_rounds=arr.shape[0],
        seed=seed,
        samples=samples,
        confidence=confidence,
    )
    out.update(
        {
            "mdi": float(max(abs(ratio["ci90_high"] - 1.0), abs(1.0 - ratio["ci90_low"]))),
            "ratio": ratio["ratio"],
            "ci90": [ratio["ci90_low"], ratio["ci90_high"]],
            "effective_n": ratio["effective_n"],
        }
    )
    return out


# ---------------------------------------------------------------------------
# P2: between-row drift of the promotion's two passes
# ---------------------------------------------------------------------------

#: A row's own clean median may move this much between its halves (half the 5 % bar).
ROW_DRIFT_TOLERANCE = 0.025
#: The 1-minute load may change this much across the rows (two runnable tasks).
LOAD_CHANGE_TOLERANCE = 2.0


def _halves_drift(sequence) -> float | None:
    s = np.asarray(sequence or [], dtype=float)
    if s.size < 4 or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        return None
    half = s.size // 2
    return float(np.median(s[s.size - half :]) / np.median(s[:half]) - 1.0)


def between_row_drift(
    rows: Mapping[str, Mapping],
    *,
    load_average_at_start=None,
    load_average_at_end=None,
    row_tolerance: float = ROW_DRIFT_TOLERANCE,
    load_tolerance: float = LOAD_CHANGE_TOLERANCE,
) -> dict:
    """P2: does the host drift across the promotion's rows? (``drifted`` True / False)

    The reference and candidate rows are separate passes, so a host change
    between them is charged to the candidate and the paired-block interval
    does not cover it. Two recorded signals, each against a stated tolerance:

    - **within-row drift**: each row's clean-call median of its last half over
      its first half, minus 1; beyond ``row_tolerance`` the host was not
      stationary on the scale of the rule while the rows ran;
    - **load change**: the range of every recorded 1-minute load average (the
      run's start / end and each row's ``load_average_at_row_start`` /
      ``_end`` where recorded) beyond ``load_tolerance``.

    A signal that cannot be computed (no sequence, no load) is recorded as
    unassessed and does not by itself mark drift.
    """
    reasons, per_row, loads = [], {}, []
    for label, la in (("run start", load_average_at_start), ("run end", load_average_at_end)):
        if la:
            loads.append((label, float(la[0])))
    for key, row in rows.items():
        drift = _halves_drift(row.get("call_ms_sequence"))
        per_row[key] = {"within_row_drift": drift}
        if drift is None:
            per_row[key]["note"] = "fewer than 4 clean calls: within-row drift unassessed"
        elif abs(drift) > row_tolerance:
            reasons.append(
                f"{key}'s clean median moved {drift:+.2%} between its halves "
                f"(> {row_tolerance:.1%})"
            )
        for field in ("load_average_at_row_start", "load_average_at_row_end"):
            la = row.get(field)
            if la:
                loads.append((f"{key} {field.rsplit('_', 1)[1]}", float(la[0])))
    if loads:
        values = [v for _, v in loads]
        load_range = max(values) - min(values)
        if load_range > load_tolerance:
            reasons.append(
                f"the 1-minute load moved by {load_range:.2f} across the rows "
                f"(> {load_tolerance:.1f}): {dict(loads)}"
            )
    else:
        load_range = None
    return {
        "drifted": bool(reasons),
        "reasons": reasons,
        "per_row": per_row,
        "load_1min": dict(loads),
        "load_range_1min": load_range,
        "row_tolerance": row_tolerance,
        "load_tolerance": load_tolerance,
        "note": (
            "the rows are separate passes: drift between them is not cancelled by the "
            "paired-block interval, so a drifting host makes the promotion INCONCLUSIVE"
        ),
    }
