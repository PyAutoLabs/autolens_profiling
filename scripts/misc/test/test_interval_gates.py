"""Witnesses for the memo-policy, scaling and log-det interval gates (#362, fix phase 3b).

``scripts/misc/likelihood_breakdown/interval_gates.py`` gives rows C1, C3, C4 and
C5 of ``results/notes/timing_noise_audit_2026_10.md`` an interval and an
INCONCLUSIVE state, through the shared ``round_bootstrap`` and ``ab_verdict``
tools; ``ab_verdict.holm_family_verdict`` is the family-wise policy C3 uses (and
phase 9 reuses). Everything here is deterministic: synthetic numbers are fixed
arrays, bootstraps use fixed seeds, and no clock is read. The committed rows are
re-judged as facts; no JSON is rewritten.
"""

from __future__ import annotations

import ast
import glob
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_MISC = str(ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from likelihood_breakdown import ab_verdict, interval_gates  # noqa: E402
from likelihood_breakdown.ab_verdict import (  # noqa: E402
    AT_LEAST,
    AT_MOST,
    GO,
    INCONCLUSIVE,
    NO_GO,
    Criterion,
    conjoin_verdicts,
    holm_family_verdict,
    holm_levels,
)
from likelihood_breakdown.interval_gates import (  # noqa: E402
    BENEFICIAL,
    FAIL,
    HARMFUL,
    MEMO_POLICY_TARGETS,
    NEUTRAL,
    PASS,
    breakdown_reconciliation,
    logdet_lever_verdict,
    matched_classification,
    matched_decision_counts,
    memo_policy_family,
)

RESULTS = ROOT / "results/breakdown/imaging"
MEMO_CELL = ROOT / "scripts/imaging/pixelized/fixed_light_numba_memo_policy.py"
SCALING_CELL = ROOT / "scripts/imaging/pixelized/fixed_light_numba_scaling.py"
TRACE_CELL = ROOT / "scripts/imaging/pixelized/fixed_light_trace.py"
MEMO_ROW = (
    RESULTS / "fixed_light_numba_memo_policy_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s5b.json"
)

#: A fixed, symmetric spread for synthetic repeats (no random draws).
SPREAD = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


def _repeats(ratio: float, spread: float, n: int = 6, base: float = 100.0):
    """``(reference, candidate)`` with candidate / reference = ratio + spread * SPREAD."""
    offsets = np.resize(SPREAD, n)
    return [base] * n, list(base * (ratio + spread * offsets))


# ---------------------------------------------------------------------------
# The shared family-wise policy (ab_verdict.holm_*), with exact intervals
# ---------------------------------------------------------------------------


def _normal(name, point, se, bar, direction=AT_MOST):
    """A ``confidence -> Criterion`` with the exact normal interval ``point +- z se``."""
    from statistics import NormalDist

    def at(level):
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return Criterion(name, point, point - z * se, point + z * se, bar, direction)

    return at


def test_holm_levels_step_down_from_bonferroni_to_unadjusted():
    assert holm_levels(6) == pytest.approx((1 - 0.1 / 6, 0.98, 0.975, 1 - 0.1 / 3, 0.95, 0.90))
    assert holm_levels(1) == pytest.approx((0.90,))
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            holm_levels(bad)
    with pytest.raises(ValueError):
        holm_levels(3, confidence=1.0)


def test_holm_steps_down_where_bonferroni_would_stop():
    # A resolves at 95 % (z = 3 standard errors clear); B only at 90 % (z = 1.8).
    members = {
        "a": _normal("a", 0.90, 0.05 / 3.0, 0.95),
        "b": _normal("b", 0.90, 0.05 / 1.8, 0.95),
    }
    family = holm_family_verdict(members, n=6)
    assert family.verdict == GO
    assert family.member_confidence == pytest.approx({"a": 0.95, "b": 0.90})
    assert [s["m"] for s in family.steps] == [2, 1]
    # Bonferroni (both at 95 %) would leave b unresolved.
    assert ab_verdict.criterion_verdict(members["b"](0.95)).verdict == INCONCLUSIVE


def test_holm_family_unadjusted_go_is_inconclusive_family_wise():
    # Six targets each resolve at 90 % (z = 1.7) but none at 98.3 %: unadjusted GO,
    # family-wise INCONCLUSIVE, never a measured negative.
    members = {f"t{i}": _normal(f"t{i}", 0.90, 0.05 / 1.7, 0.95) for i in range(6)}
    family = holm_family_verdict(members, n=6)
    assert family.unadjusted.verdict == GO
    assert family.verdict == INCONCLUSIVE
    assert len(family.steps) == 1 and family.steps[0]["resolved"] == {}
    assert "family-wise rule governs" in family.reason
    assert json.loads(json.dumps(family.as_dict(), allow_nan=False))["verdict"] == INCONCLUSIVE


def test_holm_family_one_resolved_against_is_no_go_whatever_the_rest():
    members = {
        "for": _normal("for", 0.90, 0.001, 0.95),
        "against": _normal("against", 1.10, 0.001, 0.95),
        "open": _normal("open", 0.95, 0.05, 0.95),
    }
    family = holm_family_verdict(members, n=6)
    assert family.verdict == NO_GO
    assert family.adjusted.criteria[2].verdict == INCONCLUSIVE


def test_holm_family_below_min_n_and_red_gates():
    members = {"a": _normal("a", 0.5, 0.001, 0.95)}
    assert holm_family_verdict(members, n=4).verdict == INCONCLUSIVE
    assert holm_family_verdict(members, n=6, gates={"numerical": False}).verdict == NO_GO
    with pytest.raises(ValueError):
        holm_family_verdict({}, n=6)
    with pytest.raises(ValueError):
        holm_family_verdict({"x": _normal("y", 0.5, 0.01, 0.95)}, n=6)


def test_conjoin_verdicts():
    assert conjoin_verdicts([GO, GO]) == GO
    assert conjoin_verdicts([GO, INCONCLUSIVE]) == INCONCLUSIVE
    assert conjoin_verdicts([INCONCLUSIVE, NO_GO]) == NO_GO
    with pytest.raises(ValueError):
        conjoin_verdicts([])
    with pytest.raises(ValueError):
        conjoin_verdicts(["PASS"])


# ---------------------------------------------------------------------------
# C1: matched counterfactual classification
# ---------------------------------------------------------------------------


def test_c1_a_point_below_097_with_a_straddling_interval_is_inconclusive():
    cold = [100.0] * 6
    memo = [88.0, 101.0, 93.0, 99.0, 95.0, 97.0]  # median ratio 0.96: the old flag fired
    out = matched_classification({"cold": cold, "memo": memo})
    assert out["memo_over_cold"] == pytest.approx(0.96)
    assert out["ci90_low"] < 0.97 < out["ci90_high"]
    assert out["classification"] == INCONCLUSIVE


@pytest.mark.parametrize(
    "ratio,expected", [(0.50, BENEFICIAL), (1.70, HARMFUL), (1.00, NEUTRAL)], ids=str
)
def test_c1_clearly_resolved_cases(ratio, expected):
    cold, memo = _repeats(ratio, 0.002)
    out = matched_classification({"cold": cold, "memo": memo})
    assert out["classification"] == expected
    assert out["effective_n"] == 6 and out["resampling"] == "paired whole rounds"


def test_c1_four_repeats_are_inconclusive_by_construction():
    # The cell records 4 repeats: below the audit's 5-round minimum, whatever the effect.
    cold, memo = _repeats(0.50, 0.002, n=4)
    out = matched_classification({"cold": cold, "memo": memo})
    assert out["classification"] == INCONCLUSIVE
    assert "n = 4 < 5" in out["reason"]
    with pytest.raises(ValueError):
        matched_classification({"cold": [1.0, 2.0], "memo": [1.0]})


def test_c1_counts_are_resolved_classifications_never_decision_errors():
    rows = [
        (True, HARMFUL),
        (False, BENEFICIAL),
        (True, INCONCLUSIVE),
        (False, INCONCLUSIVE),
        (False, NEUTRAL),
    ]
    out = matched_decision_counts(rows)
    assert out["resolved_harmful_accepted"] == 1
    assert out["resolved_beneficial_rejected"] == 1
    assert out["inconclusive"] == 2
    assert out["resolved_neutral"] == 1
    assert out["eligible_transitions"] == 5 and out["accepted"] == 2
    assert not {"false_accepts", "false_rejects"} & set(out)
    assert "not measured decision-error rates" in out["qualification"]
    with pytest.raises(ValueError):
        matched_decision_counts([(True, "maybe")])


# ---------------------------------------------------------------------------
# C3: the six-target memo-policy family
# ---------------------------------------------------------------------------


def _evaluations(spreads=None, **points):
    """Synthetic sequence totals: each target's point ratio, each run's relative spread.

    The guarded lane is ``1000 (1 + spread * SPREAD)`` and each reference lane a
    constant, so every target's ratio is ``point (1 + spread * SPREAD)`` per repeat
    and its median is exactly ``point``. Defaults clear every bar by 0.03.
    """
    spreads = {run: 0.0005 for run, _, _ in MEMO_POLICY_TARGETS.values()} | (spreads or {})
    evaluations = {}
    for name, (run, reference, bar) in MEMO_POLICY_TARGETS.items():
        lanes = evaluations.setdefault(run, {"sequence_totals_ms": {}})["sequence_totals_ms"]
        lanes["guarded"] = list(1000.0 * (1.0 + spreads[run] * SPREAD))
        lanes[reference] = [1000.0 / points.get(name, bar - 0.03)] * 6
    for lanes in evaluations.values():
        for lane in ("cold", "memo"):
            lanes["sequence_totals_ms"].setdefault(lane, [1000.0] * 6)
    return evaluations


def test_c3_clearly_resolved_go():
    family = memo_policy_family(_evaluations())
    assert family["verdict"] == "GO"
    assert set(family["target_verdicts"].values()) == {GO}


def test_c3_one_target_resolved_against_is_no_lever():
    family = memo_policy_family(_evaluations(broad_holdout_3pct_vs_cold=1.06))
    assert family["verdict"] == "NO_LEVER"
    assert family["target_verdicts"]["broad_holdout_3pct_vs_cold"] == NO_GO


def test_c3_a_point_target_met_with_a_straddling_interval_is_inconclusive():
    # graded guarded / memo point 0.94 <= 0.95 (the old rule would GO), scatter +-3.75 %.
    evaluations = _evaluations({"graded": 0.015}, graded_5pct_vs_memo=0.94)
    family = memo_policy_family(evaluations)
    target = family["targets"]["graded_5pct_vs_memo"]
    assert target["point"] <= 0.95 < target["ci90_high"]
    assert family["verdict"] == INCONCLUSIVE


def test_c3_holm_witness_unadjusted_go_but_family_wise_inconclusive():
    # Every target: relative spread 1 %, point bar / 1.0175 — its 90 % upper bound is
    # below the bar and its 98.3 % (Holm's first level) upper bound above it.
    runs = {run for run, _, _ in MEMO_POLICY_TARGETS.values()}
    evaluations = _evaluations(
        {run: 0.01 for run in runs},
        **{name: bar / 1.0175 for name, (_, _, bar) in MEMO_POLICY_TARGETS.items()},
    )
    family = memo_policy_family(evaluations)
    for name, (_, _, bar) in MEMO_POLICY_TARGETS.items():
        target = family["targets"][name]
        assert target["point"] < target["ci90_high"] <= bar, name
    assert family["unadjusted"]["verdict"] == GO
    assert family["verdict"] == INCONCLUSIVE
    assert family["steps"][0]["confidence"] == pytest.approx(1 - 0.1 / 6)
    assert family["steps"][0]["resolved"] == {}


def test_c3_the_committed_memo_policy_row_stays_no_lever_family_wise():
    row = json.loads(MEMO_ROW.read_text())
    family = memo_policy_family(row["evaluations"])
    assert row["verdict"] == "NO_LEVER"
    assert family["verdict"] == "NO_LEVER"
    # Every target resolves at the first (Bonferroni) Holm level, matching its point.
    assert len(family["steps"]) == 1
    expected = {k: (GO if v else NO_GO) for k, v in row["performance_targets"].items()}
    assert family["target_verdicts"] == expected
    for name, (run, reference, _) in MEMO_POLICY_TARGETS.items():
        stored = row["evaluations"][run][f"guarded_over_{reference}"]
        assert family["targets"][name]["point"] == pytest.approx(stored, rel=1e-12)


def test_c1_the_committed_matched_transitions_are_all_inconclusive():
    row = json.loads(MEMO_ROW.read_text())
    counts = {}
    for name, run in row["evaluations"].items():
        rows = []
        for record in run["diagnostics"]["guarded"].values():
            matched = record.get("matched_counterfactual", {})
            if matched.get("eligible"):
                out = matched_classification(matched["samples_ms"])
                assert out["memo_over_cold"] == pytest.approx(matched["memo_over_cold"])
                rows.append((record["precheck"]["accepted"], out["classification"]))
        counts[name] = matched_decision_counts(rows)
    assert sum(c["eligible_transitions"] for c in counts.values()) == 128
    assert all(c["inconclusive"] == c["eligible_transitions"] for c in counts.values())
    published = sum(
        run["matched_decisions"]["false_accepts"] + run["matched_decisions"]["false_rejects"]
        for run in row["evaluations"].values()
    )
    assert published == 15  # 1 "false accept" + 14 "false rejects", now unresolved


# ---------------------------------------------------------------------------
# C4: breakdown reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio,spread,expected",
    [
        (1.002, 0.0005, PASS),
        (1.10, 0.0005, FAIL),
        (0.90, 0.0005, FAIL),
        (1.04, 0.012, INCONCLUSIVE),  # point inside the band, interval crosses 1.05
        (1.06, 0.012, INCONCLUSIVE),  # point outside the band, interval crosses back
    ],
    ids=str,
)
def test_c4_reconciliation_on_an_interval(ratio, spread, expected):
    clean, observed = _repeats(ratio, spread, base=1000.0)
    out = breakdown_reconciliation(observed, clean)
    assert out["verdict"] == expected
    assert out["relative"] == pytest.approx(ratio - 1.0, abs=spread * 3)


def test_c4_too_few_repeats_and_unpaired_input():
    clean, observed = _repeats(1.0, 0.0005, n=4, base=1000.0)
    assert breakdown_reconciliation(observed, clean)["verdict"] == INCONCLUSIVE
    with pytest.raises(ValueError):
        breakdown_reconciliation([1.0, 2.0], [1.0])


def _scaling_driver():
    os.environ["AUTOLENS_PROFILING_SMOKE"] = "1"
    spec = importlib.util.spec_from_file_location("scaling_status_under_test", SCALING_CELL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c4_status_pass_requires_a_resolved_pass():
    run_status = _scaling_driver().run_status
    ok = {"numerical": True, "breakdown": True, "single_thread": True, "complete": True}
    assert run_status(ok, [PASS, PASS]) == "PASS"
    unresolved = dict(ok, breakdown=False)
    assert run_status(unresolved, [PASS, INCONCLUSIVE]) == "INCONCLUSIVE"
    assert run_status(unresolved, [FAIL, INCONCLUSIVE]) == "INCOMPLETE_OR_FAIL"
    assert run_status(dict(unresolved, numerical=False), [INCONCLUSIVE]) == "INCOMPLETE_OR_FAIL"


SCALING_ROWS = sorted(
    glob.glob(str(RESULTS / "fixed_light_numba_scaling_delaunay_*_n[0-9]*[0-9].json"))
)


@pytest.mark.parametrize("path", SCALING_ROWS, ids=lambda p: Path(p).stem[-5:])
def test_c4_the_committed_scaling_rows_resolve_pass(path):
    row = json.loads(Path(path).read_text())
    assert row["status"] == "PASS"
    for group in row["groups"].values():
        for summary in group["summary"].values():
            out = breakdown_reconciliation(
                summary["observed_sequence_totals_ms"], summary["sequence_totals_ms"]
            )
            assert out["verdict"] == PASS, out["reason"]
            assert abs(out["ci90_low"]) < 0.01 and abs(out["ci90_high"]) < 0.01


def test_c4_five_committed_scaling_cells():
    assert len(SCALING_ROWS) == 5


# ---------------------------------------------------------------------------
# C5: the log-det lever
# ---------------------------------------------------------------------------


def _interleaved(saving: float, spread: float, n: int = 7):
    library = [26.5] * n
    candidate = list(26.5 - saving + spread * np.resize(SPREAD, n))
    return {"library_ms": library, "candidate_ms": candidate}


def test_c5_a_point_saving_above_threshold_with_a_straddling_interval_is_inconclusive():
    out = logdet_lever_verdict(_interleaved(0.6, 0.08), 0.6)
    assert out["interleaved"]["saving_ms"] >= 0.5 > out["interleaved"]["ci90_low"]
    assert out["clears_threshold"] == INCONCLUSIVE


def test_c5_a_saving_resolved_below_the_threshold_does_not_clear():
    out = logdet_lever_verdict(_interleaved(0.1, 0.01), 0.1)
    assert out["interleaved"]["verdict"] == NO_GO
    assert out["clears_threshold"] is False


def test_c5_a_resolved_saving_cannot_clear_without_a_jit_profile_interval():
    # The interleaved interval clears 0.5 ms; the one-block jit_profile estimator has
    # no interval, so the pre-registered conjunction stays INCONCLUSIVE.
    out = logdet_lever_verdict(_interleaved(2.0, 0.01), 2.0)
    assert out["interleaved"]["verdict"] == GO
    assert out["jit_profile"]["verdict"] == INCONCLUSIVE
    assert out["clears_threshold"] == INCONCLUSIVE


def test_c5_too_few_rounds_and_unpaired_input():
    assert logdet_lever_verdict(_interleaved(0.1, 0.01, n=4), 0.1)["clears_threshold"] == (
        INCONCLUSIVE
    )
    with pytest.raises(ValueError):
        logdet_lever_verdict({"library_ms": [1.0], "candidate_ms": []}, 0.0)


def _logdet_rows():
    rows = {}
    for path in sorted(RESULTS.glob("fixed_light_trace_*logdet*.json")):
        payload = json.loads(path.read_text())

        def find(node):
            if isinstance(node, dict):
                if isinstance(node.get("lever"), dict) and "clears_threshold" in node["lever"]:
                    return node
                for value in node.values():
                    found = find(value)
                    if found:
                        return found
            return None

        rows[path.name.removeprefix("fixed_light_trace_").split("_fp64")[0]] = find(payload)
    return rows


#: Re-judged facts: every committed lever row was published clears_threshold False.
LOGDET_EXPECTED = {
    "delaunay_logdet_schur_k32_hpc_a100": False,
    "delaunay_logdet_schur_k64_hpc_a100": False,
    "delaunay_logdet_schur_k256_hpc_a100": False,
    "rectangular_logdet_schur_k32_hpc_a100": False,
    "rectangular_logdet_schur_k64_hpc_a100": False,
    "rectangular_logdet_schur_k256_hpc_a100": False,
    "delaunay_logdet_schur_k32_local_rtx2060": False,
    "delaunay_logdet_schur_k64_local_rtx2060": INCONCLUSIVE,
    "delaunay_logdet_schur_k256_local_rtx2060": INCONCLUSIVE,
}


def test_c5_the_committed_logdet_rows_are_rejudged():
    rows = _logdet_rows()
    levers = {k: v for k, v in rows.items() if v["kind"] == "lever"}
    assert set(levers) == set(LOGDET_EXPECTED)
    for name, block in levers.items():
        assert block["lever"]["clears_threshold"] is False
        out = logdet_lever_verdict(
            block["whole_call_ms"]["interleaved"], block["lever"]["saving_jit_profile_ms"]
        )
        assert out["clears_threshold"] == LOGDET_EXPECTED[name], (name, out["reason"])
        assert math.isclose(
            out["interleaved"]["saving_ms"], block["lever"]["saving_interleaved_median_ms"]
        )


# ---------------------------------------------------------------------------
# The cells read the shared gates and no point-only comparison is left
# ---------------------------------------------------------------------------


def _source_calls(cell: Path) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(ast.parse(cell.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "interval_gates"
    }


@pytest.mark.parametrize(
    "cell,calls,gone",
    [
        (
            MEMO_CELL,
            {"matched_classification", "matched_decision_counts", "memo_policy_family"},
            ("ratio < 0.97", "ratio > 1.03", '"false_accepts"', '"false_rejects"'),
        ),
        (
            SCALING_CELL,
            {"breakdown_reconciliation"},
            ("abs(observed_median / clean_median - 1) <= 0.05",),
        ),
        (
            TRACE_CELL,
            {"logdet_lever_verdict"},
            ("_ld_saving_jit >= logdet_reuse_injection.LOGDET_LEVER_MS",),
        ),
    ],
    ids=lambda v: v.stem if isinstance(v, Path) else "",
)
def test_each_cell_judges_its_gate_through_interval_gates(cell, calls, gone):
    source = cell.read_text()
    assert calls <= _source_calls(cell), cell.name
    for text in gone:
        assert text not in source, (cell.name, text)
    tree = ast.parse(source)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"ab_rule_verdict", "holm_family_verdict", "round_median_ratio"}


def test_c4_lanes_conjoin_through_the_shared_conjunction():
    conjoin = interval_gates.conjoin_reconciliations
    assert conjoin([PASS, PASS]) == PASS
    assert conjoin([PASS, INCONCLUSIVE]) == INCONCLUSIVE
    assert conjoin([INCONCLUSIVE, FAIL]) == FAIL
