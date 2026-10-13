"""Witnesses for the shared A/B go / lever verdict and tie sets (#362, fix phase 3).

``scripts/misc/likelihood_breakdown/ab_verdict.py`` is the one rule behind the
pre-registered A/B rules of ``results/notes/timing_noise_audit_2026_10.md`` rows
C6, C7, C10 and P2. Everything here is deterministic: synthetic call times are
drawn from a fixed seed and bootstrapped with a fixed seed, and no clock is read.

The cells are module-level JAX / numba scripts, so they are never imported here.
Their ``from likelihood_breakdown.ab_verdict import ...`` statements and their
rule functions are lifted from the AST and executed on synthetic or committed
rows, so the code under test is the cells' own.
"""

from __future__ import annotations

import ast
import json
import math
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

from likelihood_breakdown import ab_verdict, family_gates  # noqa: E402
from likelihood_breakdown.ab_verdict import (  # noqa: E402
    AT_LEAST,
    AT_MOST,
    GO,
    INCONCLUSIVE,
    NO_GO,
    Criterion,
    ab_rule_verdict,
    paired_block_ratio_interval,
    tie_set,
)

PYTREE_CELL = ROOT / "scripts/point_source_source/source_plane/pytree_input_ab.py"
BACKWARD_CELL = ROOT / "scripts/point_source_source/source_plane/backward_pass_ab.py"
SWEEP_CELL = ROOT / "scripts/point_source_image/image_plane/solver_config_sweep.py"
NUMBA_CELL = ROOT / "scripts/imaging/pixelized/fixed_light_numba.py"
CELLS = (PYTREE_CELL, BACKWARD_CELL, SWEEP_CELL, NUMBA_CELL)

#: The phase-2b / 2c bars (issues #322, #325), unchanged by fix phase 3.
GO_MIN_SAVED_MS = 0.05
GO_MIN_FRACTION = 0.15

SEED = 12345
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Synthetic, seeded A/B samples and their bootstrap intervals
# ---------------------------------------------------------------------------


def _calls(median_ms: float, n_rounds: int, n_calls: int, scatter: float, seed: int):
    """Lognormal call times in seconds, ``n_rounds * n_calls`` of them, fixed seed."""
    rng = np.random.default_rng(seed)
    return median_ms * 1e-3 * np.exp(rng.normal(0.0, scatter, n_rounds * n_calls))


def _bootstrap(control, route, seed: int):
    """The cells' estimator: medians, iid percentile bootstrap, 90 % intervals."""
    ctl = np.asarray(control) * 1e3
    rte = np.asarray(route) * 1e3
    rng = np.random.default_rng(seed)
    saved = np.empty(N_BOOT)
    ratio = np.empty(N_BOOT)
    for i in range(N_BOOT):
        c = np.median(rng.choice(ctl, ctl.size))
        r = np.median(rng.choice(rte, rte.size))
        saved[i] = c - r
        ratio[i] = r / c
    return {
        "saved": (
            float(np.median(ctl) - np.median(rte)),
            float(np.percentile(saved, 5)),
            float(np.percentile(saved, 95)),
        ),
        "ratio": (
            float(np.median(rte) / np.median(ctl)),
            float(np.percentile(ratio, 5)),
            float(np.percentile(ratio, 95)),
        ),
    }


def _phase_rule(control, route, *, n_rounds: int, seed: int = SEED):
    """The phase-2c rule shape: saved ms >= 0.05 AND route / control <= 0.85."""
    b = _bootstrap(control, route, seed)
    return ab_rule_verdict(
        [
            Criterion("saved_ms", *b["saved"], GO_MIN_SAVED_MS, AT_LEAST, "ms"),
            Criterion("ratio", *b["ratio"], 1.0 - GO_MIN_FRACTION, AT_MOST),
        ],
        n=n_rounds,
    )


def test_a_clear_30_percent_saving_is_go():
    control = _calls(1.0, 20, 20, 0.03, SEED)
    route = _calls(0.70, 20, 20, 0.03, SEED + 1)
    verdict = _phase_rule(control, route, n_rounds=20)
    assert verdict.verdict == GO, verdict.reason
    assert verdict.go is True
    assert all(c.verdict == GO for c in verdict.criteria)


def test_a_clear_0_percent_saving_is_no_go():
    control = _calls(1.0, 20, 20, 0.03, SEED)
    route = _calls(1.0, 20, 20, 0.03, SEED + 1)
    verdict = _phase_rule(control, route, n_rounds=20)
    assert verdict.verdict == NO_GO, verdict.reason
    assert verdict.go is False


def test_a_15_percent_point_estimate_straddling_the_bar_is_inconclusive():
    """The point sits on the bar; the interval straddles it: never a measured no-go."""
    control = _calls(1.0, 6, 5, 0.12, SEED)
    route = _calls(0.85, 6, 5, 0.12, SEED + 1)
    verdict = _phase_rule(control, route, n_rounds=6)
    ratio = next(c for c in verdict.criteria if c.name == "ratio")
    assert ratio.lower < 0.85 < ratio.upper
    assert ratio.verdict == INCONCLUSIVE
    assert verdict.verdict == INCONCLUSIVE, verdict.reason
    assert "not a measured negative" in verdict.reason
    assert math.isfinite(ratio.mdi) and ratio.mdi > 0.0
    assert "resolvable effect" in ratio.reason


def test_two_configurations_with_overlapping_intervals_are_a_tie_set():
    """Seeded speed-up intervals of two near-equal configurations overlap: no single best."""
    control = _calls(1.0, 20, 10, 0.05, SEED)
    candidates = {}
    for i, (name, ms) in enumerate((("a", 0.50), ("b", 0.51), ("slow", 0.80))):
        b = _bootstrap(control, _calls(ms, 20, 10, 0.05, SEED + 10 + i), SEED + 20 + i)
        point, lo, hi = b["ratio"]
        candidates[name] = (1.0 / point, 1.0 / hi, 1.0 / lo)  # control / config speed-up
    result = tie_set(candidates)
    assert set(result.members) == {"a", "b"}, result.reason
    assert result.best is None and not result.resolved
    assert result.leader in {"a", "b"}
    assert "not a single best" in result.reason


def test_a_leader_clear_of_every_other_interval_is_named_best():
    result = tie_set({"a": (2.0, 1.95, 2.05), "b": (1.5, 1.4, 1.6)})
    assert result.best == "a" and result.members == ("a",)
    lower_better = tie_set({"a": (2.0, 1.95, 2.05), "b": (1.5, 1.4, 1.6)}, higher_is_better=False)
    assert lower_better.best == "b"


def test_a_candidate_with_a_non_finite_interval_cannot_be_excluded():
    result = tie_set({"a": (2.0, 1.95, 2.05), "b": (1.0, float("nan"), 1.1)})
    assert result.best is None and set(result.members) == {"a", "b"}
    assert tie_set({}).best is None


def test_n_below_the_minimum_is_inconclusive_even_on_a_clear_effect():
    criterion = Criterion("fraction", 0.30, 0.28, 0.32, GO_MIN_FRACTION, AT_LEAST)
    verdict = ab_rule_verdict([criterion], n=4)
    assert verdict.verdict == INCONCLUSIVE
    assert "4 < 5" in verdict.reason
    assert ab_rule_verdict([criterion], n=5).verdict == GO


@pytest.mark.parametrize(
    "criterion, n",
    [
        (Criterion("f", float("nan"), 0.1, 0.2, 0.15), 10),
        (Criterion("f", 0.15, float("-inf"), 0.2, 0.15), 10),
        (Criterion("f", 0.15, 0.2, 0.1, 0.15), 10),
        (Criterion("f", 0.30, 0.28, 0.32, 0.15), None),
        (Criterion("f", 0.30, 0.28, 0.32, 0.15), -1),
        (Criterion("f", 0.30, 0.28, 0.32, 0.15), 5.5),
        (Criterion("f", 0.30, 0.28, 0.32, 0.15), float("inf")),
    ],
)
def test_invalid_inputs_are_inconclusive(criterion, n):
    verdict = ab_rule_verdict([criterion], n=n)
    assert verdict.verdict == INCONCLUSIVE
    assert "invalid input" in verdict.reason


def test_a_red_correctness_gate_is_no_go_before_any_timing():
    criterion = Criterion("fraction", 0.30, 0.28, 0.32, GO_MIN_FRACTION, AT_LEAST)
    verdict = ab_rule_verdict([criterion], n=20, gates={"correctness": False})
    assert verdict.verdict == NO_GO
    assert "correctness" in verdict.reason


def test_one_resolved_failure_makes_the_conjunction_no_go():
    """A conjunction cannot hold once one criterion is wholly on the wrong side."""
    verdict = ab_rule_verdict(
        [
            Criterion("saved_ms", 0.04, 0.035, 0.045, GO_MIN_SAVED_MS, AT_LEAST, "ms"),
            Criterion("fraction", 0.16, 0.12, 0.20, GO_MIN_FRACTION, AT_LEAST),
        ],
        n=20,
    )
    assert verdict.verdict == NO_GO
    assert "saved_ms" in verdict.reason


def test_programming_errors_raise():
    with pytest.raises(ValueError):
        ab_rule_verdict([], n=10)
    with pytest.raises(ValueError):
        ab_rule_verdict([Criterion("f", 0.3, 0.2, 0.4, 0.15, "sideways")], n=10)
    with pytest.raises(ValueError):
        ab_rule_verdict([Criterion("f", 0.3, 0.2, 0.4, 0.15)], n=10, min_n=1)
    with pytest.raises(ValueError):
        tie_set({"a": (1.0, 2.0)})


def test_paired_block_interval_handles_too_few_and_invalid_blocks():
    point, lo, hi, n = paired_block_ratio_interval([1.0], [0.9])
    assert n == 1 and point == pytest.approx(0.9) and math.isnan(lo) and math.isnan(hi)
    point, lo, hi, n = paired_block_ratio_interval([1.0, -1.0], [0.9, 0.9])
    assert math.isnan(point) and math.isnan(lo)
    point, lo, hi, n = paired_block_ratio_interval([1.0, 1.0, 1.0], [0.9, 0.9])
    assert math.isnan(point)


# ---------------------------------------------------------------------------
# Every cell holds the same function objects and defines no copy of the rule
# ---------------------------------------------------------------------------


def _imports_of(cell: Path) -> dict:
    tree = ast.parse(cell.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "likelihood_breakdown.ab_verdict"
    ]
    assert nodes, f"{cell.name} does not import likelihood_breakdown.ab_verdict"
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(cell), "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


@pytest.mark.parametrize("cell", CELLS, ids=lambda p: p.stem)
def test_each_cell_imports_the_shared_function_objects(cell):
    imported = _imports_of(cell)
    assert imported, cell
    for name, obj in imported.items():
        assert obj is getattr(ab_verdict, name), f"{cell.name}: {name} is not the shared object"
    # Since fix phase 9 a cell may reach the shared rule through family_gates (C6, C10).
    family = _module_imports(cell, "likelihood_breakdown.family_gates")
    for name, obj in family.items():
        assert obj is getattr(family_gates, name), f"{cell.name}: {name} is not the shared object"
    assert {"ab_rule_verdict", "tie_set"} & set(imported) or family
    defined = {
        node.name
        for node in ast.walk(ast.parse(cell.read_text()))
        if isinstance(node, ast.FunctionDef)
    }
    assert not defined & {
        "ab_rule_verdict",
        "tie_set",
        "criterion_verdict",
        "holm_tie_set",
        "holm_family_verdict",
    }, cell.name


def _module_imports(cell: Path, module: str) -> dict:
    """The names a cell's ``from <module> import ...`` statements bind (may be none)."""
    nodes = [
        node
        for node in ast.parse(cell.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module == module
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(cell), "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def _lift(cell: Path, names: tuple[str, ...], namespace: dict) -> dict:
    """Execute the named top-level functions / assignments of a cell in ``namespace``."""
    tree = ast.parse(cell.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
    assert body, f"{cell.name}: none of {names} found"
    namespace.update(_imports_of(cell))
    namespace.update(_module_imports(cell, "likelihood_breakdown.round_bootstrap"))
    namespace.update(_module_imports(cell, "likelihood_breakdown.family_gates"))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(cell), "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# C7 — the pytree phase-2b rule, the cell's own code
# ---------------------------------------------------------------------------


def _pytree_namespace(rows: dict, n_rounds: int = 20) -> dict:
    return _lift(
        PYTREE_CELL,
        ("GO_MIN_SAVED_MS", "GO_MIN_FRACTION", "_median_saving_ms", "_phase2b_rule"),
        {
            "np": np,
            "rows": rows,
            "LANES": ("solved", "plain"),
            "BOOTSTRAP_SAMPLES": N_BOOT,
            "N_ROUNDS": n_rounds,
        },
    )


def _pytree_row(control_s, route_s, n_rounds):
    b = _bootstrap(control_s, route_s, SEED)
    p_ms, f_ms = float(np.median(control_s) * 1e3), float(np.median(route_s) * 1e3)
    ratio_point, ratio_lo, ratio_hi = (
        1.0 / b["ratio"][0],
        1.0 / b["ratio"][2],
        1.0 / b["ratio"][1],
    )
    return {
        "n_rounds": n_rounds,
        "stats": {"pytree": {"median_ms": p_ms}},
        "saved_ms_pytree_minus_flat_vector": p_ms - f_ms,
        "saved_ms_pytree_minus_flat_vector_ci90": {
            "saved_ms": b["saved"][0],
            "ci90_low": b["saved"][1],
            "ci90_high": b["saved"][2],
        },
        "ratio_pytree_over_flat_vector": {
            "ratio": ratio_point,
            "ci90_low": ratio_lo,
            "ci90_high": ratio_hi,
        },
    }


def test_the_pytree_cell_rule_reads_intervals():
    """The cell's own ``_phase2b_rule``: GO, NO_GO and INCONCLUSIVE on seeded rows."""
    rows = {
        "solved_forward": _pytree_row(
            _calls(0.30, 20, 20, 0.03, SEED), _calls(0.21, 20, 20, 0.03, SEED + 1), 20
        ),
        "plain_forward": _pytree_row(
            _calls(0.30, 20, 20, 0.03, SEED), _calls(0.30, 20, 20, 0.03, SEED + 1), 20
        ),
    }
    rule = _pytree_namespace(rows)["_phase2b_rule"]()
    assert rule["per_lane"]["solved"]["verdict"] == GO
    assert rule["per_lane"]["solved"]["go"] is True
    assert rule["per_lane"]["plain"]["verdict"] == NO_GO
    rows["plain_forward"] = _pytree_row(
        _calls(1.0, 6, 5, 0.12, SEED), _calls(0.85, 6, 5, 0.12, SEED + 1), 6
    )
    rule = _pytree_namespace(rows)["_phase2b_rule"]()
    assert rule["per_lane"]["plain"]["verdict"] == INCONCLUSIVE
    assert rule["per_lane"]["plain"]["go"] is False


@pytest.mark.parametrize(
    "config, expected",
    [
        ("hpc_ral_cpu_fp64", {"solved": NO_GO, "plain": NO_GO}),
        ("hpc_a100_fp64", {"solved": GO, "plain": GO}),
    ],
)
def test_the_committed_pytree_rows_are_rejudged_unchanged(config, expected):
    """Re-judged as facts (#362 fix phase 3): the committed go / no-go calls all resolve.

    The deciding RAL CPU lanes save 0.0385 / 0.0452 ms with paired round-bootstrap
    90 % intervals [0.0360, 0.0414] / [0.0438, 0.0470] ms (fix phase 4; the iid
    intervals of fix phase 3 were [0.0367, 0.0408] / [0.0439, 0.0469]), wholly
    below the 0.05 ms bar: the recorded no-go is a measured NO_GO. The A100 lanes'
    GO resolves too.
    """
    data = json.loads(
        (ROOT / f"results/breakdown/point_source_source/pytree_input_ab_{config}.json").read_text()
    )
    rows = {key: dict(row) for key, row in data["rows"].items()}
    ns = _pytree_namespace(rows)
    for lane in ("solved", "plain"):
        row = rows[f"{lane}_forward"]
        row["saved_ms_pytree_minus_flat_vector_ci90"] = ns["_median_saving_ms"](
            np.asarray(row["per_call_ms"]["pytree"]) * 1e-3,
            np.asarray(row["per_call_ms"]["flat_vector"]) * 1e-3,
            SEED + 3,
        )
    rule = ns["_phase2b_rule"]()
    for lane, verdict in expected.items():
        assert rule["per_lane"][lane]["verdict"] == verdict, rule["per_lane"][lane]
        assert rule["per_lane"][lane]["go"] == data["phase2b_rule"]["per_lane"][lane]["go"]


# ---------------------------------------------------------------------------
# C6 — the backward-pass phase-2c rule, re-judged on its committed intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, changed",
    [
        ("hpc_ral_gpunode_cpu_fp64", {}),
        ("hpc_a100_fp64", {}),
        (
            "local_cpu_fp64",
            {("solved", "rev_analytic"): INCONCLUSIVE, ("plain", "rev_analytic"): INCONCLUSIVE},
        ),
    ],
)
def test_the_committed_backward_pass_rows_are_rejudged_unchanged(config, changed):
    """Every committed phase-2c route on the deciding and A100 rows resolves as published.

    Since fix phase 9 the verdict is family-wise (Holm over the host's 16 criteria)
    on the paired round draws; the two laptop ``rev_analytic`` rows do not resolve
    (the laptop does not decide; the phase-2c decision, ``fwd``, is unchanged).
    """
    data = json.loads(
        (ROOT / f"results/breakdown/point_source_source/backward_pass_ab_{config}.json").read_text()
    )
    rows = data["rows"]
    rule = _lift(
        BACKWARD_CELL,
        ("GO_MIN_SAVED_MS", "GO_MIN_FRACTION", "_phase2c_rule"),
        {
            "rows": rows,
            "LANES": tuple(data["phase2c_rule"]["per_lane"]),
            "GRAD_ROUTES": {
                route: None for lane in data["phase2c_rule"]["per_lane"].values() for route in lane
            }
            | {"rev": None},
            "BOOTSTRAP_SEED": SEED,
            "BOOTSTRAP_SAMPLES": 2000,
        },
    )["_phase2c_rule"]()
    for lane, per_route in data["phase2c_rule"]["per_lane"].items():
        for route, old in per_route.items():
            new = rule["per_lane"][lane][route]
            expected = changed.get((lane, route), GO if old["go"] else NO_GO)
            assert new["verdict"] == expected, (lane, route, new.get("verdict_reason"))
    assert "fwd" in rule["go_routes"]


# ---------------------------------------------------------------------------
# C10 — the solver sweep's "best admissible", re-judged as tie sets
# ---------------------------------------------------------------------------

_SWEEP_TIE_SETS = {
    "solver_config_sweep_hpc_ral_cpu_fp64": (
        "e2.5_s0.4",
        {"e2.5_s0.4", "e2.5_s0.2", "e3_s0.4", "e3_s0.3", "e4_s0.4"},
    ),
    # 3 rounds < MIN_AB_ROUNDS: nothing can be excluded (fix phase 4).
    "solver_config_sweep_laptop_cpu_fp64": ("e3_s0.4", None),
    "solver_config_sweep_mcs_hpc_ral_a100_fp64": ("mcs20", {"mcs20"}),
    "solver_config_sweep_mcs_hpc_ral_cpu_fp64": ("mcs18", {"mcs18", "mcs20"}),
    "solver_config_sweep_mcs_laptop_cpu_fp64": ("mcs18", {"mcs18", "mcs20"}),
    "solver_config_sweep_step0_hpc_ral_a100_fp64": (
        "step0_gather",
        {"step0_gather", "step0_structured", "step0_components"},
    ),
    "solver_config_sweep_step0_hpc_ral_cpu_epyc7702_fp64": (
        "step0_structured",
        {"step0_structured", "step0_components"},
    ),
    "solver_config_sweep_step0_hpc_ral_cpu_fp64": (
        "step0_components",
        {"step0_components", "step0_structured"},
    ),
    "solver_config_sweep_step0_laptop_cpu_fp64": ("step0_components", {"step0_components"}),
}

#: Fix phase 9: the members the Holm-adjusted tie set adds to the unadjusted one.
_SWEEP_FAMILY_ADDS = {
    "solver_config_sweep_hpc_ral_cpu_fp64": {"e3_s0.2", "e4_s0.3"},
    "solver_config_sweep_mcs_hpc_ral_cpu_fp64": {"mcs24"},
    "solver_config_sweep_mcs_laptop_cpu_fp64": {"mcs24"},
}


@pytest.mark.parametrize("stem", sorted(_SWEEP_TIE_SETS))
def test_the_committed_sweep_best_is_rejudged_as_a_tie_set(stem):
    """The cell's own ``_fastest`` on the committed rows: 7 of 9 "best" are tie sets.

    Since fix phase 9 ``_fastest`` is the family-wise tie set on the paired round
    draws; its ``.unadjusted`` is the fix phase 3 / 4 tie set (the same sets on the
    round intervals), and the family-wise one adds the members of
    ``_SWEEP_FAMILY_ADDS``. No named "best" changes.
    """
    data = json.loads((ROOT / f"results/breakdown/point_source_image/{stem}.json").read_text())
    rows = data["rows"]
    candidates = [
        n
        for n in rows
        if n != "control"
        and data["admissibility"][n]
        and rows[n]["config"]["block"] != "precision"
        and data["precision_equivalent"][n]
    ]
    fastest = _lift(
        SWEEP_CELL,
        ("_fastest",),
        {
            "rows": rows,
            "N_ROUNDS": data["protocol"]["n_rounds"],
            "BOOTSTRAP_SEED": SEED,
            "BOOTSTRAP_SAMPLES": 2000,
        },
    )["_fastest"]
    result = fastest(candidates)
    leader, members = _SWEEP_TIE_SETS[stem]
    if members is None:
        members = set(candidates)
    assert result.leader == leader == data["best_admissible"]
    assert set(result.unadjusted.members) == members
    assert result.unadjusted.best == (leader if members == {leader} else None)
    family = members | _SWEEP_FAMILY_ADDS.get(stem, set())
    assert set(result.members) == family
    assert result.best == (leader if family == {leader} else None)
