"""Witnesses for the paired whole-round bootstrap (#362, fix phase 4).

``scripts/misc/likelihood_breakdown/round_bootstrap.py`` replaces the A/B cells'
iid, unpaired call bootstrap (audit note rows C6, C8, C9 and the reported CIs).
Everything here is deterministic: synthetic rounds are drawn from fixed seeds and
bootstrapped with fixed seeds, and no clock is read. The cells are never imported
(they are module-level JAX scripts); their bootstrap helpers and rules are lifted
from the AST, so the code under test is the cells' own.
"""

from __future__ import annotations

import ast
import json
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

from likelihood_breakdown import round_bootstrap  # noqa: E402
from likelihood_breakdown.ab_verdict import GO, INCONCLUSIVE, NO_GO  # noqa: E402
from likelihood_breakdown.round_bootstrap import (  # noqa: E402
    RESAMPLING,
    as_rounds,
    round_median_ratio,
    round_median_saving,
)

SS = ROOT / "scripts/point_source_source/source_plane"
IP = ROOT / "scripts/point_source_image/image_plane"
#: Every cell whose reported median-ratio interval is the round bootstrap, and the
#: helper names it keeps (thin wrappers over the shared functions).
CELLS = {
    SS / "pytree_input_ab.py": ("_median_ratio", "_median_saving_ms"),
    SS / "backward_pass_ab.py": ("_median_ratio", "_median_saving_ms"),
    SS / "gradient_mode_crossover.py": ("_ratio_boots",),
    SS / "gradient_mode_library_ab.py": ("_ratio_boots",),
    IP / "solver_config_sweep.py": ("_median_ratio",),
    IP / "static_lattice_ab.py": ("_median_ratio",),
    IP / "vertex_dedup_ab.py": ("_median_ratio",),
}

SEED = 12345
TRUE_RATIO = 0.8


def _correlated_rounds(seed: int, *, n_rounds=20, n_calls=20, round_sd=0.08, call_sd=0.02):
    """Two arms timed in rounds; each (round, arm) block shares one random offset.

    That is the within-round correlation of a round-robin A/B: an arm's
    ``n_calls`` consecutive calls see the same frequency step / neighbour load.
    The population median ratio route / control is exactly ``TRUE_RATIO``.
    """
    rng = np.random.default_rng(seed)
    shape = (n_rounds, n_calls)
    control = np.exp(rng.normal(0.0, round_sd, (n_rounds, 1)) + rng.normal(0.0, call_sd, shape))
    route = TRUE_RATIO * np.exp(
        rng.normal(0.0, round_sd, (n_rounds, 1)) + rng.normal(0.0, call_sd, shape)
    )
    return control.ravel(), route.ravel()


def _iid_ratio(numerator, denominator, seed: int, samples: int):
    """The cells' pre-fix-phase-4 estimator: iid, unpaired call resampling."""
    rng = np.random.default_rng(seed)
    boots = np.empty(samples)
    for i in range(samples):
        boots[i] = np.median(rng.choice(numerator, numerator.size)) / np.median(
            rng.choice(denominator, denominator.size)
        )
    return float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


# ---------------------------------------------------------------------------
# The witness: within-round correlation
# ---------------------------------------------------------------------------


def test_the_iid_interval_is_visibly_narrower_than_the_round_interval():
    control, route = _correlated_rounds(0)
    rounds = round_median_ratio(route, control, n_rounds=20, seed=SEED)
    iid_low, iid_high = _iid_ratio(route, control, SEED, 2000)
    round_width = rounds["ci90_high"] - rounds["ci90_low"]
    assert (iid_high - iid_low) < 0.5 * round_width, (iid_low, iid_high, rounds)
    assert rounds["effective_n"] == 20
    assert rounds["resampling"] == RESAMPLING == "paired whole rounds"


def test_the_round_interval_covers_the_true_ratio_under_a_fixed_seed():
    control, route = _correlated_rounds(4)
    rounds = round_median_ratio(route, control, n_rounds=20, seed=SEED)
    assert rounds["ci90_low"] <= TRUE_RATIO <= rounds["ci90_high"], rounds
    iid_low, iid_high = _iid_ratio(route, control, SEED, 2000)
    assert not iid_low <= TRUE_RATIO <= iid_high, "the iid interval misses it on this seed"


def test_round_coverage_is_near_nominal_and_iid_coverage_is_not():
    """Over 40 seeded datasets: round ~0.85, iid ~0.33 at a nominal 0.90."""
    covered_round = covered_iid = 0
    for seed in range(40):
        control, route = _correlated_rounds(seed)
        r = round_median_ratio(route, control, n_rounds=20, seed=SEED, samples=500)
        covered_round += r["ci90_low"] <= TRUE_RATIO <= r["ci90_high"]
        lo, hi = _iid_ratio(route, control, SEED, 500)
        covered_iid += lo <= TRUE_RATIO <= hi
    assert covered_round / 40 >= 0.75
    assert covered_iid / 40 <= 0.5


def test_rounds_are_paired_across_arms():
    """A shared round effect cancels: same indices for both arms, so a narrow ratio."""
    rng = np.random.default_rng(SEED)
    shared = np.exp(rng.normal(0.0, 0.2, (20, 1)))
    control = (shared * np.exp(rng.normal(0.0, 0.01, (20, 10)))).ravel()
    route = (TRUE_RATIO * shared * np.exp(rng.normal(0.0, 0.01, (20, 10)))).ravel()
    paired = round_median_ratio(route, control, n_rounds=20, seed=SEED)
    assert paired["ci90_high"] - paired["ci90_low"] < 0.05
    assert paired["ci90_low"] <= TRUE_RATIO <= paired["ci90_high"]


def test_saving_and_boots_and_layout():
    control, route = _correlated_rounds(3)
    saving = round_median_saving(control, route, n_rounds=20, seed=SEED, scale=1.0)
    assert saving["ci90_low"] < saving["saved_ms"] < saving["ci90_high"]
    assert saving["effective_n"] == 20
    out, boots = round_median_ratio(route, control, n_rounds=20, seed=SEED, return_boots=True)
    assert boots.shape == (2000,) and out["bootstrap_samples"] == 2000
    assert as_rounds(np.arange(6.0), 3).shape == (3, 2)
    with pytest.raises(ValueError):
        as_rounds(np.arange(7.0), 3)
    with pytest.raises(ValueError):
        round_median_ratio(np.ones((10, 2)), np.ones((12, 2)), n_rounds=10, seed=SEED)


# ---------------------------------------------------------------------------
# Every cell holds the same function objects and no iid copy
# ---------------------------------------------------------------------------


def _import_namespace(cell: Path) -> dict:
    nodes = [
        node
        for node in ast.parse(cell.read_text()).body
        if isinstance(node, ast.ImportFrom)
        and node.module
        in ("likelihood_breakdown.round_bootstrap", "likelihood_breakdown.ab_verdict")
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(cell), "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def _lift(cell: Path, names, namespace: dict) -> dict:
    body = [
        node
        for node in ast.parse(cell.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    namespace.update(_import_namespace(cell))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(cell), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("cell", sorted(CELLS), ids=lambda p: p.stem)
def test_each_cell_uses_the_shared_round_bootstrap(cell):
    imported = _import_namespace(cell)
    shared = {
        k: v for k, v in imported.items() if k in ("round_median_ratio", "round_median_saving")
    }
    assert shared, f"{cell.name} does not import the round bootstrap"
    for name, obj in shared.items():
        assert obj is getattr(round_bootstrap, name)
    source = cell.read_text()
    assert "rng.choice(num, num.size)" not in source, f"{cell.name} kept an iid bootstrap"
    assert "rng.choice(ctl, ctl.size)" not in source and "rng.choice(slow" not in source


@pytest.mark.parametrize("cell", sorted(CELLS), ids=lambda p: p.stem)
def test_each_cell_helper_returns_the_shared_round_interval(cell):
    control, route = _correlated_rounds(5)
    ns = _lift(cell, CELLS[cell], {"np": np, "N_ROUNDS": 20, "BOOTSTRAP_SAMPLES": 2000})
    expected = round_median_ratio(route, control, n_rounds=20, seed=SEED)
    helper = CELLS[cell][0]
    got = ns[helper](route, control, SEED)
    got = got[0] if isinstance(got, tuple) else got
    assert got == expected


# ---------------------------------------------------------------------------
# Re-judged committed rows (recorded as facts in the audit note)
# ---------------------------------------------------------------------------


def _ms_to_s(values) -> np.ndarray:
    return np.asarray(values, dtype=float) * 1e-3


def test_the_committed_pytree_ral_cpu_saving_stays_below_the_bar():
    data = json.loads(
        (
            ROOT / "results/breakdown/point_source_source/pytree_input_ab_hpc_ral_cpu_fp64.json"
        ).read_text()
    )
    expected = {"solved": (0.03597, 0.04135), "plain": (0.04381, 0.04702)}
    for lane, (low, high) in expected.items():
        row = data["rows"][f"{lane}_forward"]
        saved = round_median_saving(
            _ms_to_s(row["per_call_ms"]["pytree"]),
            _ms_to_s(row["per_call_ms"]["flat_vector"]),
            n_rounds=row["n_rounds"],
            seed=SEED + 3,
        )
        assert saved["ci90_low"] == pytest.approx(low, abs=2e-5)
        assert saved["ci90_high"] == pytest.approx(high, abs=2e-5)
        assert saved["ci90_high"] < 0.05


def _backward_rule_on_round_intervals(config: str) -> tuple[dict, dict]:
    path = ROOT / f"results/breakdown/point_source_source/backward_pass_ab_{config}.json"
    data = json.loads(path.read_text())
    cell = SS / "backward_pass_ab.py"
    ns = _lift(
        cell,
        (
            "GO_MIN_SAVED_MS",
            "GO_MIN_FRACTION",
            "_median_ratio",
            "_median_saving_ms",
            "_phase2c_rule",
        ),
        {"np": np, "BOOTSTRAP_SAMPLES": 2000, "BOOTSTRAP_SEED": SEED},
    )
    rows = {}
    for key, row in data["rows"].items():
        row = dict(row)
        if key.endswith("_grad"):
            ns["N_ROUNDS"] = row["n_rounds"]
            times = {r: _ms_to_s(v) for r, v in row["per_call_ms"].items()}
            control = row["control"]
            row["vs_control"] = {
                route: {
                    "ratio_route_over_control": ns["_median_ratio"](
                        times[route], times[control], SEED + i
                    ),
                    "saved_ms": ns["_median_saving_ms"](
                        times[control], times[route], SEED + 100 + i
                    ),
                }
                for i, route in enumerate(row["timed_routes"])
                if route != control
            }
        rows[key] = row
    ns["rows"] = rows
    ns["LANES"] = tuple(data["phase2c_rule"]["per_lane"])
    ns["GRAD_ROUTES"] = {
        route: None for lane in data["phase2c_rule"]["per_lane"].values() for route in lane
    } | {"rev": None}
    return data, ns["_phase2c_rule"]()


@pytest.mark.parametrize(
    "config, changed",
    [
        ("hpc_ral_gpunode_cpu_fp64", {}),
        ("hpc_a100_fp64", {}),
        ("local_cpu_fp64", {("solved", "rev_analytic"): INCONCLUSIVE}),
    ],
)
def test_the_committed_backward_pass_rows_on_round_intervals(config, changed):
    """C6 on the paired round bootstrap: one laptop GO becomes INCONCLUSIVE.

    ``local_cpu`` solved ``rev_analytic`` was GO on an iid ratio interval
    [0.810, 0.846]; the round interval [0.805, 0.854] straddles the 0.85 bar.
    The laptop row does not decide (RAL CPU decides); every RAL row is unchanged.
    """
    data, rule = _backward_rule_on_round_intervals(config)
    for lane, per_route in data["phase2c_rule"]["per_lane"].items():
        for route, old in per_route.items():
            new = rule["per_lane"][lane][route]["verdict"]
            expected = changed.get((lane, route), GO if old["go"] else NO_GO)
            assert new == expected, (lane, route, rule["per_lane"][lane][route]["verdict_reason"])


def test_the_committed_sweep_tie_sets_hold_on_round_intervals():
    """C10 on the round bootstrap: the RAL CPU tie set is the same five configurations."""
    data = json.loads(
        (
            ROOT / "results/breakdown/point_source_image/solver_config_sweep_hpc_ral_cpu_fp64.json"
        ).read_text()
    )
    rows = data["rows"]
    n_rounds = data["protocol"]["n_rounds"]
    ns = _lift(
        IP / "solver_config_sweep.py",
        ("_median_ratio", "_fastest"),
        {"np": np, "N_ROUNDS": n_rounds, "BOOTSTRAP_SAMPLES": 2000},
    )
    control = _ms_to_s(rows["control"]["per_call_ms"])
    rejudged = {}
    for i, name in enumerate(rows):
        row = dict(rows[name])
        row["speedup_vs_control"] = ns["_median_ratio"](
            control, _ms_to_s(row["per_call_ms"]), SEED + i
        )
        rejudged[name] = row
    ns["rows"] = rejudged
    candidates = [
        n
        for n in rows
        if n != "control"
        and data["admissibility"][n]
        and rows[n]["config"]["block"] != "precision"
        and data["precision_equivalent"][n]
    ]
    result = ns["_fastest"](candidates)
    assert result.best is None
    assert set(result.members) == {"e2.5_s0.4", "e2.5_s0.2", "e3_s0.4", "e3_s0.3", "e4_s0.4"}
