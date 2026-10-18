"""``scripts/cluster/image_plane/``: per-draw candidate positions in the sweep JSON and the
oracle verification figure (autolens_profiling#436).

``_judge.py`` is NumPy only (no JAX, no PyAutoLens); the figure smoke builds a one-galaxy
isothermal tracer and a hand-made sweep fixture, so it never loads a cluster dataset or
runs the sweep.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_cluster_oracle_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[2] / "cluster" / "image_plane"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import _judge  # noqa: E402
import oracle_figure  # noqa: E402

TOL = 0.02


class _FlatEquation:
    """``residual`` = distance to a fixed list of exact roots (enough for the verdicts)."""

    def __init__(self, roots):
        self.roots = np.asarray(roots, dtype=float)

    def residual(self, positions, plane_index, beta):
        p = np.asarray(positions, dtype=float)
        d = p[:, None, :] - self.roots[None, :, :]
        k = np.argmin(np.hypot(d[..., 0], d[..., 1]), axis=1)
        return d[np.arange(len(p)), k]


def _root(y, x, mu, parity):
    return {
        "position": [y, x],
        "magnification": mu,
        "parity": parity,
        "singular_values": [1.0, 0.5],
        "accepted": True,
    }


def _oracle_source():
    return {
        "roots": [
            _root(1.0, 0.0, 10.0, 1),  # covered
            _root(-1.0, 0.0, -5.0, -1),  # duplicated
            _root(0.0, 3.0, 4.0, 1),  # missed
            _root(0.0, -0.05, 0.01, 1),  # below the 0.1 threshold: never judged
            {**_root(5.0, 5.0, 2.0, 1), "accepted": False},  # rejected by the oracle
        ]
    }


def _candidates():
    return np.array(
        [
            [1.001, 0.0],  # covers root 0
            [-1.0, 0.008],  # two distinct candidates on root 1 -> duplicated
            [-1.0, -0.008],
            [0.0, -0.05],  # attributed to the below-threshold root -> ignored
            [7.0, -7.0],  # spurious
            [np.nan, np.nan],  # padding
        ]
    )


def _judged():
    src = _oracle_source()
    eq = _FlatEquation([r["position"] for r in src["roots"]])
    return _judge.judge(src, _candidates(), TOL, (-10, 10, -10, 10), eq, 1, (0.0, 0.0))


def test_judge_counts_unchanged():
    j = _judged()
    assert j["n_roots_considered"] == 3
    assert j["n_roots_below_threshold"] == 1
    assert (j["n_covered"], j["n_missed"], j["n_duplicated"], j["n_spurious"]) == (1, 1, 1, 1)
    assert j["n_candidates_raw"] == 5 and j["n_candidates_distinct"] == 5
    assert j["n_candidates_for_ignored_roots"] == 1
    assert j["multiplicity_match"] is False  # 4 relevant candidates vs 3 roots
    assert j["unresolved"] == 2


def test_per_draw_source_entry_schema():
    e = _judge.per_draw_source_entry(_judged())
    assert set(e) == set(_judge.COUNT_KEYS) | set(_judge.POSITION_KEYS)
    assert e["tol"] == TOL
    verdicts = [r["verdict"] for r in e["roots"]]
    assert verdicts == ["covered", "duplicated", "missed"]
    for r in e["roots"]:
        assert set(r) == {"position", "magnification", "parity", "verdict", "matched"}
        assert len(r["position"]) == 2
        for m in r["matched"]:
            assert set(m) == {"position", "error"} and m["error"] <= TOL
    covered, duplicated, missed = e["roots"]
    assert len(covered["matched"]) == 1
    assert covered["matched"][0]["error"] == pytest.approx(0.001, abs=1e-9)
    assert len(duplicated["matched"]) == 2 and missed["matched"] == []
    errs = [m["error"] for m in duplicated["matched"]]
    assert errs == sorted(errs)
    assert sorted(map(tuple, e["duplicated_candidates"])) == [(-1.0, -0.008), (-1.0, 0.008)]
    assert [s["position"] for s in e["spurious"]] == [[7.0, -7.0]]
    assert set(e["spurious"][0]) == {"position", "residual", "nearest_root_distance"}
    assert [s["position"] for s in e["ignored_candidates"]] == [[0.0, -0.05]]
    json.dumps(e, allow_nan=False)  # serialisable, no NaN / inf leaks


def test_per_draw_source_entry_none_and_cap():
    assert _judge.per_draw_source_entry(None) is None
    j = _judged()
    j["spurious_all"] = [
        {"position": [float(i), 0.0], "residual": 1.0, "nearest_root_distance": 1.0}
        for i in range(50)
    ]
    e = _judge.per_draw_source_entry(j)
    assert len(e["spurious"]) == _judge.MAX_RECORDED_CANDIDATES
    assert e["n_spurious"] == j["n_spurious"]  # counts stay exact


def test_close_groups_and_zoom_window():
    pts = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.4], [20.0, 20.0], [40.0, 0.0], [41.0, 0.0]]
    groups = oracle_figure.close_groups(pts, 3.0)
    assert sorted(groups) == [[0, 1, 2], [4, 5]]
    triplet = next(g for g in groups if 0 in g)
    cy, cx, hw = oracle_figure.zoom_window([pts[i] for i in triplet])
    assert (cy, cx) == pytest.approx((0.25, 0.2))
    assert hw == pytest.approx(0.375)  # 0.75 x span: the triplet fills ~2/3 of the panel
    assert oracle_figure.zoom_window([[0, 0], [1e-4, 0]], tol=0.01)[2] == pytest.approx(0.1)


def test_marker_and_bar_helpers():
    assert oracle_figure.root_marker_size(0.01) == oracle_figure.root_marker_size(0.1) == 25.0
    assert oracle_figure.root_marker_size(-1000.0) == pytest.approx(165.0)
    assert oracle_figure.kpc_bar(60.0, 0.16) == 50.0  # ~9" at z=0.5 on a 60" field
    assert oracle_figure.kpc_bar(160.0, 0.16) == 200.0
    assert oracle_figure.nearest_curve_distance(
        (0.0, 0.0), [(np.array([3.0, 4.0]), np.array([4.0, 0.0]), "-")]
    ) == pytest.approx(4.0)
    assert oracle_figure.nearest_curve_distance((0.0, 0.0), []) is None


# ---------------------------------------------------------------------------
# Figure smoke on a hand-made fixture
# ---------------------------------------------------------------------------


def _fixture_summary():
    """An SIS (theta_E = 1") at z=0.5, one source at z=1 with beta = (0, 0.2):
    images at x = 1.2 (parity +) and x = -0.8 (parity -)."""
    roots = [_root(0.0, 1.2, 6.0, 1), _root(0.0, -0.8, -4.0, -1)]
    oracle_src = {"beta": [0.0, 0.2], "roots": roots, "stability": {"stable": True}}
    src = {"roots": roots}
    eq = _FlatEquation([r["position"] for r in roots])
    good = _judge.judge(src, [[0.0, 1.2005], [0.0, -0.8]], TOL, (-3, 3, -3, 3), eq, 1, (0, 0.2))
    bad = _judge.judge(src, [[0.0, 1.2], [2.0, 2.0]], TOL, (-1, 1.5, -1, 1.5), eq, 1, (0, 0.2))
    row = lambda status, limits, j: {  # noqa: E731
        "status": status,
        "static": {"limits": list(limits), "last_step_triangle_side": TOL},
        "per_draw": [{"draw": 0, "set": "prior", "sources": [_judge.per_draw_source_entry(j)]}],
    }
    return {
        "dataset": "simple",
        "pilot": True,
        "config_name": "fixture",
        "quotable": False,
        "oracle": {
            "per_draw": [
                {"draw": 0, "plane_indices": [1], "betas": [[0.0, 0.2]], "sources": [oracle_src]}
            ]
        },
        "rows": {
            "control": row("admissible", (-3, 3, -3, 3), good),
            "tight": row("diagnostic", (-1, 1.5, -1, 1.5), bad),
        },
        "draws": {"set_per_draw": ["prior"], "vectors": [[]]},
    }


def test_select_rows_and_paths(tmp_path):
    s = _fixture_summary()
    assert oracle_figure.rows_with_positions(s) == ["control", "tight"]
    assert oracle_figure.select_rows(s, 0, [0]) == ["control", "tight"]  # tight has a miss
    assert oracle_figure.select_rows(s, 0, [0], ["tight"]) == ["tight"]
    with pytest.raises(SystemExit):
        oracle_figure.select_rows(s, 0, [0], ["nope"])
    assert oracle_figure.plane_sources(s["oracle"]["per_draw"][0]) == {1: [0]}
    assert oracle_figure.figure_path(tmp_path, s, 0, 1).name == (
        "oracle_figure_simple_pilot_fixture_draw0_plane1.png"
    )


def test_oracle_figure_smoke(tmp_path):
    pytest.importorskip("autolens.cluster.plot.cluster_plots")
    import autolens as al

    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)
            ),
            al.Galaxy(redshift=1.0),
        ]
    )
    case = SimpleNamespace(
        observed_positions=[np.array([[0.0, 1.2], [0.0, -0.8]])], source_names=["fixture_src"]
    )
    path, info = oracle_figure.render(
        _fixture_summary(), case, 0, 1, tmp_path, cc_pixels=40, tracer=tracer
    )
    assert path.exists() and path.stat().st_size > 10_000
    assert info["rows"]["control"]["covered"] == 2
    assert info["rows"]["tight"]["missed"] == 1 and info["rows"]["tight"]["spurious"] == 1
    assert len(info["groups"]) == 1 and info["groups"][0]["n"] == 2  # 2" pair < 3" radius
