"""Cluster PointSolver starting-grid follow-up (autolens_profiling#439): the ``viz_cluster``
case, the sweep's ``--scales`` / ``--draw-ids`` parsing and the separation-rule analysis.

The sweep runs at import, so its wiring is checked on the source text (as in
``test_cluster_sweep_memory.py``); the separation rule is tested on synthetic tables.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/cells/test/test_cluster_starting_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
_DIR = ROOT / "scripts" / "cluster" / "image_plane"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import _datasets  # noqa: E402
import _sweep_options  # noqa: E402
import separation_rule as sr  # noqa: E402

SWEEP = (_DIR / "robust_settings_sweep.py").read_text()

# ---------------------------------------------------------------------------
# viz_cluster case
# ---------------------------------------------------------------------------


def test__viz_constants_shape_and_digest():
    assert _datasets.VIZ_N_COMPONENTS == 13
    assert len(_datasets.viz_member_cores()) == 12
    assert _datasets.VIZ_SOURCE_REDSHIFTS == [1.0, 2.0]
    assert _datasets.VIZ_SCALING_RCUT_EXPONENT == pytest.approx(0.7)
    digest = _datasets.viz_constants_digest()
    assert len(digest) == 64 and digest == _datasets.viz_constants_digest()


def test__viz_case_truth_gate_and_sources():
    pytest.importorskip("autolens")
    case = _datasets.load_viz_cluster()
    assert case.name == "viz_cluster"
    assert case.source_names == ["point_0", "point_1"]
    assert case.source_redshifts == [1.0, 2.0]
    gate = case.provenance["truth_gate"]
    assert gate["pass"], gate
    assert gate["n_lens_galaxies"] == 13
    assert case.provenance["constants_sha256"] == _datasets.viz_constants_digest()
    assert case.provenance["copied_at_sha"] == _datasets.VIZ_COPIED_AT_SHA
    # positions come from the oracle on the truth tracer, never the triangle solver
    assert "_oracle.solve" in case.provenance["positions"]["producer"]
    assert [len(p) for p in case.observed_positions] == [5, 2]
    centres = case.mass_centres(case.reference_vector)
    assert len(centres) == 13
    assert sum(c["kind"] == "scaling" for c in centres) == 10
    assert sum(c["kind"] == "host" for c in centres) == 1
    # the claim's image: point_0 has an image ~0.39" from main_1's core
    d, _ = sr.nearest(case.observed_positions[0], [[10.0, 8.0]])
    assert d.min() == pytest.approx(0.39, abs=0.01)


# ---------------------------------------------------------------------------
# sweep options
# ---------------------------------------------------------------------------


def test__parse_scales():
    assert _sweep_options.parse_scales(None) == (0.5, 0.7, 1.0, 2.0)
    assert _sweep_options.parse_scales("") == (0.5, 0.7, 1.0, 2.0)
    assert _sweep_options.parse_scales("1.0, 0.7,0.37 ,0.2") == (1.0, 0.7, 0.37, 0.2)
    for bad in ("1.0,abc", "0.5,-1", "0", "1.0,1"):
        with pytest.raises(ValueError):
            _sweep_options.parse_scales(bad)


def test__parse_draw_ids():
    assert _sweep_options.parse_draw_ids(None) is None
    assert _sweep_options.parse_draw_ids("81, 53,73,53") == [53, 73, 81]
    with pytest.raises(ValueError):
        _sweep_options.parse_draw_ids("-1")


def test__sweep_wiring_keeps_phase1_defaults():
    assert "SCALES = _sweep_options.parse_scales(_args.scales)" in SWEEP
    assert "PRECISIONS = (0.01, 0.001)" in SWEEP
    assert '"viz_cluster"' in SWEEP and "load_viz_cluster" in SWEEP
    assert '"--release-executables"' in SWEEP and "jax.clear_caches()" in SWEEP
    assert '"draw_id": DRAW_IDS[d["draw"]]' in SWEEP
    assert '"timing_mode"' in SWEEP
    assert '"simple": 50, "a2744": 30' in SWEEP


# ---------------------------------------------------------------------------
# separation rule (synthetic)
# ---------------------------------------------------------------------------


def test__scale_summary_monotone_and_censoring():
    s = sr.scale_summary({0.2: True, 0.37: True, 0.5: False, 1.0: False})
    assert s["coarsest_safe"] == 0.37 and s["coarsest_covering"] == 0.37
    assert s["censoring"] is None and s["monotone"] and s["finest_failing"] == 0.5
    assert sr.scale_summary({0.5: True, 1.0: True})["censoring"] == "lower"
    assert sr.scale_summary({0.5: False, 1.0: False})["censoring"] == "upper"
    nm = sr.scale_summary({0.5: False, 1.0: True})
    assert nm["coarsest_covering"] == 1.0 and nm["coarsest_safe"] is None
    assert nm["monotone"] is False


def test__required_c():
    recs = [
        {"id": "a", "sep": 0.39, "failing_scales": [0.4, 0.5, 1.0]},
        {"id": "b", "sep": 2.0, "failing_scales": []},
        {"id": "c", "sep": 0.33, "failing_scales": [0.2]},
    ]
    out = sr.required_c(recs, "sep")
    assert out["c_required"] == pytest.approx(0.33 / 0.2)
    assert out["binding"]["id"] == "c" and out["n_failures"] == 4
    assert (
        sr.required_c([{"id": "b", "sep": 1.0, "failing_scales": []}], "sep")["c_required"] is None
    )


def _root(y, x, mu, verdict="covered"):
    return {"position": [y, x], "magnification": mu, "accepted": True, "verdict": verdict}


def _row(scale, verdicts):
    return {
        "config": {
            "block": "geometry",
            "extent": "production",
            "backend": "jit",
            "max_containing_size": 20,
            "precision": 0.001,
            "scale": scale,
        },
        "verdict": {"n_roots_considered": 3, "n_missed": verdicts.count("missed")},
        "per_draw": [
            {
                "sources": [
                    {
                        "roots": [
                            _root(0.0, 0.0, 5.0, verdicts[0]),
                            _root(0.0, 0.3, -5.0, verdicts[1]),
                            _root(5.0, 5.0, 2.0, verdicts[2]),
                        ],
                        "n_spurious": 0,
                    }
                ]
            }
        ],
    }


def _sweep(rows, draw_ids=None):
    return {
        "dataset": "synthetic",
        "_file": "synthetic.json",
        "production_solver": {"pixel_scale_precision": 0.001},
        "draws": {
            "n_judged": 1,
            "draw_ids": draw_ids,
            "draw_id_filter": draw_ids,
            "set_per_draw": ["stress"],
            "vectors": [[0.0]],
        },
        "oracle": {
            "per_draw": [
                {
                    "sources": [
                        {
                            "roots": [
                                _root(0.0, 0.0, 5.0),
                                _root(0.0, 0.3, -5.0),
                                _root(5.0, 5.0, 2.0),
                                _root(9.0, 9.0, 0.05),  # below threshold: never considered
                            ]
                        }
                    ]
                }
            ]
        },
        "rows": {f"s{s}": _row(s, v) for s, v in rows.items()},
    }


def _centres(_vector):
    return [
        {"name": "main_0", "kind": "main", "centre": [5.0, 5.4]},
        {"name": "host_halo", "kind": "host", "centre": [0.0, 0.05]},  # excluded
    ]


def test__records_from_sweep_separations_and_scales():
    sweep = _sweep(
        {
            1.0: ["missed", "missed", "covered"],
            0.5: ["covered", "missed", "covered"],
            0.2: ["covered", "covered", "covered"],
        }
    )
    (rec,) = sr.records_from_sweep(sweep, _centres, "g")
    assert rec["n_roots"] == 3
    assert rec["min_image_image_sep"] == pytest.approx(0.3)
    assert rec["min_image_core_sep"] == pytest.approx(0.4)
    assert rec["nearest_core"] == "main_0"
    rec = sr.finalise(rec)
    assert rec["coarsest_safe"] == 0.2 and rec["failing_scales"] == [0.5, 1.0]
    assert rec["min_sep"] == pytest.approx(0.3)
    rule = sr.rule_test([rec])
    # root 2 (0.3" from root 1) misses at 0.5: c >= 0.3 / 0.5
    assert rule["per_root"]["min_feature_sep"]["c_required"] == pytest.approx(0.6)


def test__merge_adds_scales_and_flags_conflicts():
    a = sr.records_from_sweep(_sweep({1.0: ["missed", "covered", "covered"]}), _centres, "coarse")
    b = sr.records_from_sweep(
        _sweep({1.0: ["covered"] * 3, 0.5: ["covered"] * 3}, draw_ids=[0]), _centres, "fine"
    )
    merged, conflicts = sr.merge(a + b)
    assert len(merged) == 1 and merged[0]["group"] == "fine"
    assert set(merged[0]["per_scale_covers"]) == {"1.0", "0.5"}
    assert conflicts and conflicts[0]["scale"] == 1.0


def test__parse_a2744_log_rows_and_invalid():
    text = "\n".join(
        [
            "[  3941.0 s] row control                    diagnostic cov 67/69 [prior 49/49] "
            "[stress 18/20] miss 2 dup 0 spur 0 mult 0.9 overflow 9/20 maxerr 0.0009476 "
            "[rss 20122 MB, peak 56695 MB, live arrays 5 (0.0 MB)]",
            "[  922.6 s] row production_s0.5_p0.01      diagnostic cov 66/69 [prior 48/49] "
            "[stress 18/20] miss 3 dup 0 spur 0 mult 0.86 overflow 7/20 maxerr 0.0077 "
            "[rss 8349 MB, peak 22077 MB, live arrays 5 (0.0 MB)]",
            "[ 10823.4 s] row tight_s1_p0.001: INVALID (route error JaxRuntimeError: INTERNAL: "
            "Failed to materialize symbols: { x }) [rss 39412 MB, peak 81528 MB, live arrays 5 (0.0 MB)]",
            "[ 11462.8 s] row control_mcs40: INVALID (route error JaxRuntimeError: INTERNAL: "
            "boom) [rss 39354 MB, peak 81528 MB, live arrays 5 (0.0 MB)]",
        ]
    )
    rows = sr.parse_a2744_log(text)
    assert [r["config"] for r in rows] == [
        "control",
        "production_s0.5_p0.01",
        "tight_s1_p0.001",
        "control_mcs40",
    ]
    c = rows[0]
    assert c["scale"] == 1.0 and c["precision"] == 0.001 and c["extent"] == "production"
    assert c["n_missed"] == 2 and c["n_missed_prior"] == 0 and c["n_missed_stress"] == 2
    assert rows[1]["scale"] == 0.5 and rows[1]["n_missed_prior"] == 1
    assert rows[2]["status"] == "invalid" and rows[2]["extent"] == "tight"
    assert rows[3]["mcs"] == 40 and "JaxRuntimeError" in rows[3]["error"]


def test__nearest_excludes_self():
    d, i = sr.nearest([[0, 0], [0, 1], [5, 5]], [[0, 0], [0, 1], [5, 5]], exclude_self=True)
    assert np.allclose(d[:2], 1.0) and list(i[:2]) == [1, 0]
