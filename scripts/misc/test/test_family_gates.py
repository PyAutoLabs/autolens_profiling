"""Witnesses for the family-wise policy and the C12 / P2 remainders (#362, fix phase 9).

``scripts/misc/likelihood_breakdown/family_gates.py`` judges the multi-comparison
gates of ``results/notes/timing_noise_audit_2026_10.md`` (rows C6, C10, C11)
under one family-wise policy (Holm at family-wise 90 %, stated in
``ab_verdict``'s docstring), moves C12's split-half MDI onto the paired round
bootstrap, and records P2's between-row drift. Everything here is
deterministic: synthetic timings come from fixed seeds and every bootstrap has a
fixed seed; no clock is read. The cells are never imported; their imports and
helpers are read from the AST.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_MISC = str(ROOT / "scripts" / "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

from likelihood_breakdown import family_gates  # noqa: E402
from likelihood_breakdown.ab_verdict import (  # noqa: E402
    INCONCLUSIVE,
    NO_GO,
    holm_tie_set,
    tie_set,
)
from likelihood_breakdown.family_gates import (  # noqa: E402
    between_row_drift,
    kill_gate_family,
    phase2c_family,
    round_split_half_mdi,
)

BACKWARD_CELL = ROOT / "scripts/point_source_source/source_plane/backward_pass_ab.py"
SWEEP_CELL = ROOT / "scripts/point_source_image/image_plane/solver_config_sweep.py"
BAKEOFF_CELL = ROOT / "scripts/misc/numba_interferometer/bakeoff.py"
GPU_CELL = ROOT / "scripts/point_source_image/image_plane/gpu_bottleneck_map.py"
NUMBA_CELL = ROOT / "scripts/imaging/pixelized/fixed_light_numba.py"
NUMBA_KERNELS = ("reference", "hoisted", "symmetric", "two_stage", "direct_conv", "source_loop")
SEED = 12345


# ---------------------------------------------------------------------------
# Wiring: every cell holds the shared function objects
# ---------------------------------------------------------------------------

_WIRED = {
    BACKWARD_CELL: "phase2c_family",
    SWEEP_CELL: "sweep_tie_set",
    BAKEOFF_CELL: "kill_gate_family",
    GPU_CELL: "round_split_half_mdi",
    NUMBA_CELL: "between_row_drift",
}


def _module_imports(cell: Path, module: str) -> dict:
    nodes = [
        node
        for node in ast.parse(cell.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module == module
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(cell), "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


@pytest.mark.parametrize("cell", sorted(_WIRED), ids=lambda p: p.stem)
def test_each_cell_imports_the_shared_family_gate(cell):
    imported = _module_imports(cell, "likelihood_breakdown.family_gates")
    assert _WIRED[cell] in imported, cell.name
    for name, obj in imported.items():
        assert obj is getattr(family_gates, name), f"{cell.name}: {name} is not the shared object"
    defined = {
        node.name
        for node in ast.walk(ast.parse(cell.read_text()))
        if isinstance(node, ast.FunctionDef)
    }
    assert not defined & {_WIRED[cell], "holm_tie_set", "holm_family_verdict"}, cell.name


def test_the_bakeoff_no_longer_decides_on_a_point_ratio():
    """C11: the kill gate is the family-wise function; the point rule is only recorded."""
    source = BAKEOFF_CELL.read_text()
    assert "kill_gate_family(" in source
    assert "ratio > 1.3" not in source
    assert "tripped = " not in source


# ---------------------------------------------------------------------------
# Holm tie set (C10): a "best" needs separation at the family-wise level
# ---------------------------------------------------------------------------


def _normal(point: float, se: float):
    """A ``confidence -> (point, lower, upper)`` with exact normal two-sided intervals."""

    def at(level: float):
        z = float(norm.ppf(0.5 + level / 2.0))
        return point, point - z * se, point + z * se

    return at


def test_holm_witness_unadjusted_names_a_best_and_the_family_does_not():
    """Five rivals each separate from the leader at 90 % but none at Holm's 98 %.

    Leader 1.10, rivals 1.02, s.e. 0.02: the 90 % intervals are 0.066 wide
    apart-to-touch (separated by 0.08), so the unadjusted tie set names the
    leader; at 1 - 0.1 / 5 = 98 % they need 0.093, none separates, and the
    family-wise tie set keeps all six.
    """
    candidates = {"leader": _normal(1.10, 0.02)}
    candidates.update({f"rival{i}": _normal(1.02, 0.02) for i in range(5)})
    result = holm_tie_set(candidates, n=20)
    assert result.unadjusted.best == "leader"
    assert result.best is None
    assert set(result.members) == set(candidates)
    assert result.steps[0]["m"] == 5 and result.steps[0]["confidence"] == pytest.approx(0.98)
    assert result.steps[0]["separated"] == []
    assert "unadjusted 90 % would name leader" in result.reason


def test_holm_tie_set_names_a_leader_that_separates_at_the_adjusted_level():
    candidates = {"leader": _normal(1.50, 0.02)}
    candidates.update({f"rival{i}": _normal(1.02 + 0.01 * i, 0.02) for i in range(5)})
    result = holm_tie_set(candidates, n=20)
    assert result.best == "leader" and result.members == ("leader",)
    assert set(result.excluded_at) == {f"rival{i}" for i in range(5)}


def test_holm_tie_set_steps_down_after_a_separation():
    """One far rival separates at 1 - 0.1/2 = 95 %; the near one then gets 90 % and separates."""
    candidates = {
        "leader": _normal(1.10, 0.02),
        "far": _normal(0.80, 0.02),
        "near": _normal(1.025, 0.02),  # 0.075 apart: > 0.066 (90 %), < 0.078 (95 %)
    }
    result = holm_tie_set(candidates, n=20)
    assert result.excluded_at["far"] == pytest.approx(0.95)
    assert result.excluded_at["near"] == pytest.approx(0.90)
    assert result.best == "leader"


def test_holm_tie_set_contains_the_unadjusted_tie_set_and_keeps_non_finite_members():
    candidates = {
        "leader": _normal(1.10, 0.02),
        "a": _normal(1.03, 0.02),
        "b": _normal(0.70, 0.02),
        "broken": lambda level: (math.nan, math.nan, math.nan),
    }
    result = holm_tie_set(candidates, n=20)
    assert set(result.unadjusted.members) <= set(result.members)
    assert "broken" in result.members and result.best is None
    plain = tie_set({k: f(0.9) for k, f in candidates.items()}, n=20)
    assert result.unadjusted == plain


def test_holm_tie_set_below_the_minimum_rounds_keeps_every_candidate():
    candidates = {"leader": _normal(2.0, 0.01), "slow": _normal(1.0, 0.01)}
    result = holm_tie_set(candidates, n=4)
    assert result.best is None and set(result.members) == set(candidates)


# ---------------------------------------------------------------------------
# C6: one host's routes x lanes x criteria as one family
# ---------------------------------------------------------------------------


def _grad_row(ratio: float, seed: int, *, n_rounds: int = 20, n_calls: int = 5) -> dict:
    """A ``<lane>_grad`` row: control ``rev`` at 1 ms, two routes at ``ratio`` with round noise."""
    rng = np.random.default_rng(seed)
    control = np.exp(rng.normal(0.0, 0.02, (n_rounds, n_calls)))
    rng = np.random.default_rng(seed + 1)
    per_call = {"rev": control.ravel().tolist()}
    for route in ("fwd", "rev_analytic"):
        per_call[route] = (
            (control * ratio * np.exp(rng.normal(0.0, 0.03, (n_rounds, 1)))).ravel().tolist()
        )
    return {
        "control": "rev",
        "timed_routes": ["rev", "fwd", "rev_analytic"],
        "n_rounds": n_rounds,
        "per_call_ms": per_call,
    }


def _phase2c(ratio: float, seed: int, **kw) -> dict:
    rows = {"solved": _grad_row(ratio, 10 + seed, **kw), "plain": _grad_row(ratio, 20 + seed, **kw)}
    return phase2c_family(
        rows,
        routes=("fwd", "rev_jacrev", "rev_analytic", "fwd_analytic"),
        saved_bar_ms=0.05,
        fraction_bar=0.15,
        seed=SEED,
    )


def test_c6_holm_witness_unadjusted_go_but_family_wise_not():
    """A 0.83 ratio with +-3 % round noise: unadjusted 90 % names fwd GO, Holm does not.

    The two absent routes are NO_GO by their (red) gates and are not members; the
    eight timed criteria are one family: the saved-ms ones resolve at 1 - 0.1/8 and
    the ratios, straddling 0.85 at the stepped-down levels, are left open.
    """
    result = _phase2c(0.83, 2)
    assert result["go_routes_unadjusted"] == ["fwd"]
    assert result["go_routes"] == []
    assert result["per_route"]["fwd"]["verdict"] == INCONCLUSIVE
    assert result["per_route"]["fwd"]["verdict_unadjusted"] == "GO"
    assert result["family"]["k"] == 8
    assert result["per_lane"]["solved"]["rev_jacrev"]["verdict"] == NO_GO


def test_c6_clear_cases_resolve_family_wise():
    clear = _phase2c(0.60, 0)
    assert clear["go_routes"] == ["fwd", "rev_analytic"]
    flat = _phase2c(1.00, 0)
    assert flat["go_routes"] == []
    assert {v["verdict"] for v in flat["per_route"].values()} == {NO_GO}


def test_c6_below_the_minimum_rounds_is_inconclusive():
    result = _phase2c(0.60, 0, n_rounds=4)
    assert result["go_routes"] == []
    assert result["per_route"]["fwd"]["verdict"] == INCONCLUSIVE


@pytest.mark.parametrize(
    "config, go_routes, changed",
    [
        ("hpc_ral_gpunode_cpu_fp64", ["fwd", "fwd_analytic", "rev_analytic"], {}),
        ("hpc_a100_fp64", ["fwd", "fwd_analytic"], {}),
        ("local_cpu_fp64", ["fwd", "fwd_analytic"], {("plain", "rev_analytic"): INCONCLUSIVE}),
    ],
)
def test_c6_committed_rows_rejudged_family_wise(config, go_routes, changed):
    """Facts: the deciding (EPYC) row's GO routes are unchanged; one laptop NO_GO is open.

    ``changed`` is against the fix phase 4 round-interval verdicts
    (``verdict_unadjusted``); EPYC plain ``rev_analytic`` (ratio upper 0.844) does not
    resolve at 1 - 0.1/16 but does at the last Holm step (90 %), so it stays GO.
    """
    data = json.loads(
        (ROOT / f"results/breakdown/point_source_source/backward_pass_ab_{config}.json").read_text()
    )
    rows = {lane: data["rows"][f"{lane}_grad"] for lane in ("solved", "plain")}
    result = phase2c_family(
        rows,
        routes=("fwd", "rev_jacrev", "rev_analytic", "fwd_analytic"),
        saved_bar_ms=0.05,
        fraction_bar=0.15,
        seed=SEED,
    )
    assert result["family"]["k"] == 16
    assert result["go_routes"] == go_routes
    for lane, per_route in result["per_lane"].items():
        for route, v in per_route.items():
            expected = changed.get((lane, route), v["verdict_unadjusted"])
            assert v["verdict"] == expected, (config, lane, route, v["reason"])


# ---------------------------------------------------------------------------
# C11: the kill gate, kernels x cells as one family ("any" claim)
# ---------------------------------------------------------------------------


def _bakeoff_cells(speed: float, seed: int, *, n_rounds: int = 8, sd: float = 0.04) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for cell in ("sma/delaunay", "sma/rect", "alma/delaunay", "alma/rect"):
        base = np.exp(rng.normal(0.0, 0.01, n_rounds))
        kernels = {"rfft2_numpy": {"median_s": 1.0, "all_rounds_s": base.tolist()}}
        for name in NUMBA_KERNELS:
            samples = (base / speed * np.exp(rng.normal(0.0, sd, n_rounds))).tolist()
            kernels[name] = {"median_s": float(np.median(samples[1:])), "all_rounds_s": samples}
        out[cell] = {"kernels": kernels}
    return out


def test_c11_holm_witness_unadjusted_passed_but_family_wise_inconclusive():
    """24 kernels at a true 1.30x (7 timed rounds): one lucky member clears 1.3 at 90 %.

    The point rule and the unadjusted "any" read "passed"; Holm over the 24
    members does not resolve any of them above the margin.
    """
    gate = kill_gate_family(_bakeoff_cells(1.30, 0), numba_kernels=NUMBA_KERNELS)
    assert gate["kill_gate_point"] == "passed"
    assert gate["kill_gate_unadjusted"] == "passed"
    assert gate["kill_gate"] == INCONCLUSIVE
    assert gate["family"]["k"] == 24 and gate["family"]["n"] == 7


def test_c11_clear_cases():
    assert kill_gate_family(_bakeoff_cells(2.0, 0), numba_kernels=NUMBA_KERNELS)["kill_gate"] == (
        "passed"
    )
    assert kill_gate_family(_bakeoff_cells(0.8, 0), numba_kernels=NUMBA_KERNELS)["kill_gate"] == (
        "tripped"
    )


def test_c11_four_timed_rounds_are_inconclusive_by_construction():
    """The default ``--reps 5`` discards round 0 and times 4: below the 5-round minimum."""
    gate = kill_gate_family(_bakeoff_cells(3.0, 0, n_rounds=5), numba_kernels=NUMBA_KERNELS)
    assert gate["kill_gate"] == INCONCLUSIVE and gate["kill_gate_point"] == "passed"
    assert "4 timed rounds < 5" in gate["reason"]


@pytest.mark.parametrize("stem", ["bakeoff_v2026.8.17.1", "bakeoff_machine_drift_sma_v2026.8.17.1"])
def test_c11_committed_kill_gate_rejudged(stem):
    """Fact: both committed bake-offs read "passed" on points and INCONCLUSIVE on intervals.

    Every cell has 4 timed rounds (``--reps 5``). For context only, not a verdict:
    with the minimum lowered to 4 the family-wise gate would read "passed"
    (e.g. sma / Delaunay ``direct_conv`` 5.93x, interval [5.46, 6.11]).
    """
    data = json.loads((ROOT / f"results/breakdown/interferometer/{stem}.json").read_text())
    assert data["kill_gate"] == "passed"
    gate = kill_gate_family(data["cells"], numba_kernels=NUMBA_KERNELS)
    assert gate["kill_gate_point"] == "passed"
    assert gate["kill_gate"] == INCONCLUSIVE and gate["family"]["n"] == 4
    context = kill_gate_family(data["cells"], numba_kernels=NUMBA_KERNELS, min_n=4)
    assert context["kill_gate"] == "passed"


# ---------------------------------------------------------------------------
# C12: the split-half MDI on paired rounds
# ---------------------------------------------------------------------------


def _iid_split_half_mdi(samples_ms, seed: int) -> float:
    """The pre-phase-9 ``_mdi``: iid bootstrap, run's first half of calls vs its second half."""
    s = np.asarray(samples_ms, dtype=float)
    half = s.size // 2
    a, b = s[:half], s[half : 2 * half]
    rng = np.random.default_rng(seed)
    boots = np.array(
        [np.median(rng.choice(a, a.size)) / np.median(rng.choice(b, b.size)) for _ in range(2000)]
    )
    lo, hi = np.percentile(boots, [5, 95])
    return float(max(abs(hi - 1.0), abs(1.0 - lo)))


def test_c12_slow_drift_inflates_the_iid_floor_and_not_the_paired_one():
    """20 rounds x 20 calls, a 10 % linear drift across the run, 1 % call noise.

    The old split-half charges the drift between the run's halves to the floor
    (~5 %); pairing two halves of the same round cancels it (< 1 %).
    """
    rng = np.random.default_rng(7)
    drift = np.linspace(1.0, 1.10, 20)[:, None]
    calls = (drift * np.exp(rng.normal(0.0, 0.01, (20, 20)))).ravel()
    old = _iid_split_half_mdi(calls, SEED + 7)
    new = round_split_half_mdi(calls, n_rounds=20, seed=SEED + 7)
    assert old > 0.04
    assert new["mdi"] < 0.01
    assert new["effective_n"] == 20 and new["calls_per_arm_per_round"] == 10


def test_c12_too_few_rounds_or_calls_gives_no_mdi():
    assert math.isnan(round_split_half_mdi(np.ones(9), n_rounds=3, seed=1)["mdi"])
    assert math.isnan(round_split_half_mdi(np.ones(10), n_rounds=10, seed=1)["mdi"])


def test_c12_the_cell_helper_is_the_shared_mdi():
    tree = ast.parse(GPU_CELL.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_mdi"]
    namespace = _module_imports(GPU_CELL, "likelihood_breakdown.family_gates")
    namespace.update({"N_ROUNDS": 20, "BOOTSTRAP_SEED": SEED})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GPU_CELL), "exec"), namespace)
    calls = np.exp(np.random.default_rng(3).normal(0.0, 0.02, 400))
    assert namespace["_mdi"](calls) == round_split_half_mdi(calls, n_rounds=20, seed=SEED + 7)


@pytest.mark.parametrize(
    "stem, old, new",
    [
        ("gpu_bottleneck_map_hpc_ral_a100_fp64", 0.0545, 0.0094),
        ("gpu_bottleneck_map_hpc_ral_a100_fp32_whatif", 0.0652, 0.0100),
    ],
)
def test_c12_committed_mdi_rejudged(stem, old, new):
    """Fact: the A100 MDI the GPU memo quotes (5.45 %) is ~0.94 % on paired rounds."""
    data = json.loads((ROOT / f"results/breakdown/point_source_image/{stem}.json").read_text())
    on = data["baseline"]["rows"]["command_buffers_on"]["per_call_ms"]
    assert data["baseline"]["mdi_same_program"] == pytest.approx(old, abs=1e-4)
    mdi = round_split_half_mdi(on, n_rounds=data["protocol"]["rounds"], seed=SEED + 7)
    assert mdi["mdi"] == pytest.approx(new, abs=1e-4)


# ---------------------------------------------------------------------------
# P2: between-row drift
# ---------------------------------------------------------------------------


def _row(ms: float, *, drift: float = 0.0, n: int = 32) -> dict:
    trend = np.linspace(1.0, 1.0 + drift, n)
    return {"call_ms_sequence": (ms * trend).tolist()}


def test_p2_stationary_rows_do_not_drift():
    record = between_row_drift(
        {"b": _row(236.0), "d": _row(200.0)},
        load_average_at_start=[1.0, 1.0, 1.0],
        load_average_at_end=[1.5, 1.0, 1.0],
    )
    assert record["drifted"] is False and record["reasons"] == []


def test_p2_a_row_whose_clean_median_moves_drifts():
    record = between_row_drift({"b": _row(236.0, drift=0.08), "d": _row(200.0)})
    assert record["drifted"] is True
    assert "b's clean median moved" in record["reasons"][0]


def test_p2_a_load_change_across_the_rows_drifts():
    rows = {"b": _row(236.0), "d": _row(200.0)}
    rows["d"]["load_average_at_row_end"] = [4.5, 2.0, 1.0]
    record = between_row_drift(
        rows, load_average_at_start=[1.0, 1.0, 1.0], load_average_at_end=[1.2, 1.0, 1.0]
    )
    assert record["drifted"] is True and record["load_range_1min"] == pytest.approx(3.5)


def test_p2_missing_records_are_unassessed_not_drift():
    record = between_row_drift({"b": {}, "d": {"call_ms_sequence": [1.0, 1.0]}})
    assert record["drifted"] is False
    assert record["load_range_1min"] is None
    assert "unassessed" in record["per_row"]["b"]["note"]


def test_p2_committed_s4b_rows_do_not_drift():
    """Fact: the one committed promotion pair is stationary (-1.08 % / +1.00 %, load 1.0 -> 1.0)."""
    data = json.loads(
        (
            ROOT
            / "results/breakdown/imaging"
            / "fixed_light_numba_delaunay_hpc_ral_cpu_fp64_fixed_light_numba_s4b_warm_t1.json"
        ).read_text()
    )
    record = between_row_drift(
        data["rows"],
        load_average_at_start=data["contention"]["load_average_at_start"],
        load_average_at_end=data["contention"]["load_average_at_end"],
    )
    assert record["drifted"] is False
    assert record["per_row"]["b_sparse_numba"]["within_row_drift"] == pytest.approx(
        -0.0108, abs=1e-4
    )
    assert record["per_row"]["d_perm_sparse_numba"]["within_row_drift"] == pytest.approx(
        0.0100, abs=1e-4
    )
