"""Focused tests for the phase-6 numba scaling timing helper."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from likelihood_breakdown import fixed_light_numba_scaling_steps as steps


def _solve():
    return sum(range(20_000))


_FUNCTIONS = SimpleNamespace(solve=_solve)


def _driver():
    os.environ["AUTOLENS_PROFILING_SMOKE"] = "1"
    path = (
        Path(__file__).resolve().parents[2]
        / "imaging/likelihood_breakdown/fixed_light_numba_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("fixed_light_numba_scaling_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_timing_targets_the_numba_imaging_dispatch():
    targets = steps._production_targets()

    assert targets[0][0].__name__ == "InversionImagingSparseNumba"
    assert targets[0][1:] == ("curvature_matrix", "curvature")
    assert targets[2][1:] == ("reconstruction_positive_only_from", "solve")


class _TimedExample:
    @property
    def curvature(self):
        _FUNCTIONS.solve()
        return 1


def test_timing_scope_is_exclusive_and_restores_the_exact_descriptor():
    original = _TimedExample.__dict__["curvature"]
    original_solve = _FUNCTIONS.solve
    targets = (
        (_TimedExample, "curvature", "curvature"),
        (_FUNCTIONS, "solve", "solve"),
    )

    with steps.timing_scope(targets=targets) as timings:
        example = _TimedExample()
        example.timings = timings
        assert example.curvature == 1

    assert _TimedExample.__dict__["curvature"] is original
    assert _FUNCTIONS.solve is original_solve
    assert timings.total_ms >= timings.inclusive_ms["curvature"]
    assert timings.inclusive_ms["curvature"] >= timings.exclusive_ms["curvature"]
    assert timings.exclusive_ms["solve"] > 0
    assert timings.exclusive_ms["curvature"] + timings.exclusive_ms["solve"] == pytest.approx(
        timings.inclusive_ms["curvature"], abs=0.05
    )
    assert sum(timings.exclusive_ms.values()) == pytest.approx(timings.total_ms, abs=0.05)


def test_timing_scope_restores_after_error_and_records_is_do_not_leak():
    original = _TimedExample.__dict__["curvature"]
    targets = ((_TimedExample, "curvature", "curvature"),)

    with pytest.raises(RuntimeError):
        with steps.timing_scope(targets=targets) as failed:
            raise RuntimeError("likelihood failed")

    assert _TimedExample.__dict__["curvature"] is original
    assert failed.total_ms > 0

    with steps.timing_scope(targets=targets) as fresh:
        pass

    assert fresh is not failed
    assert fresh.inclusive_ms["curvature"] == 0
    assert _TimedExample.__dict__["curvature"] is original


def test_summarize_scaling_keeps_missing_and_failed_cells_visible():
    summary = steps.summarize_scaling(
        [
            {"status": "ok", "source_pixels": 100, "total_ms": 4.0},
            {"status": "ok", "source_pixels": 200, "total_ms": 16.0},
            {"status": "failed", "source_pixels": 300, "total_ms": None},
            {"status": "missing", "source_pixels": 400},
        ]
    )

    assert summary["status_counts"] == {"measured": 2, "failed": 1, "missing": 1}
    assert len(summary["cells"]) == 4
    assert summary["log_log_exponent"] == pytest.approx(2.0)
    assert summary["r_squared"] == pytest.approx(1.0)


def test_driver_declaration_is_deterministic_and_shared_across_sizes():
    driver = _driver()

    first = driver.declaration(2)
    second = driver.declaration(2)
    assert first == second
    assert first["requested_pixels"] == list(driver.PIXELS)


def test_driver_paired_gates_reject_bad_observer_and_active_set(monkeypatch):
    driver = _driver()
    diagnostics = {
        0: {
            "evidence": 10.0,
            "reconstruction": [1.0],
            "solve_dimension": 1,
            "solver": {"seed_source": "zero", "warm_start_fallback": False},
        }
    }
    monkeypatch.setattr(
        driver,
        "diagnostics",
        lambda *args: {
            0: {
                **diagnostics[0],
                "reconstruction": list(diagnostics[0]["reconstruction"]),
                "solver": dict(diagnostics[0]["solver"]),
            }
        },
    )
    monkeypatch.setattr(driver.replay, "compare_solutions", lambda *args: {"passed": False})

    clean = [[{"elapsed_ms": 1.0, "index": 0, "evidence": 10.0}]]
    observed = [
        [
            {
                "elapsed_ms": 1.0,
                "index": 0,
                "evidence": 12.0,
                "exclusive_ms": {category: 0.0 for category in steps.CATEGORIES},
            }
        ]
    ]
    monkeypatch.setattr(
        driver,
        "timed",
        lambda cells, enabled, instrumented=False: (observed if instrumented else clean)[0],
    )

    result = driver.evaluate_group([object()], repeats=1)

    assert result["observer_agreement"] is False
    assert result["numerical_passed"] is False


def test_aggregator_retains_failed_and_missing_cells_and_log_axes(tmp_path, monkeypatch):
    driver = _driver()
    measured = {
        "smoke": False,
        "gates": {"numerical": True, "single_thread": True, "complete": True},
        "groups": {
            group: {"summary": {lane: {"ms_per_model": 2.0} for lane in ("cold", "memo")}}
            for group in ("nearby", "broad")
        },
        "peak_rss_kib": 1024,
        "status": "PASS",
    }
    failed = {**measured, "gates": {**measured["gates"], "numerical": False}}
    for n, payload in ((500, measured), (1000, failed)):
        path = tmp_path / f"fixed_light_numba_scaling_delaunay_cfg_n{n}.json"
        path.write_text(json.dumps(payload))

    import matplotlib.axes

    scales = []
    original = matplotlib.axes.Axes.set_xscale
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "set_xscale",
        lambda axis, scale, *args, **kwargs: (
            scales.append(scale),
            original(axis, scale, *args, **kwargs),
        )[1],
    )
    assert driver.aggregate(tmp_path, "cfg") == 0
    report = json.loads((tmp_path / "fixed_light_numba_scaling_summary_cfg.json").read_text())

    counts = report["lanes"]["nearby/cold"]["status_counts"]
    assert counts == {"measured": 1, "failed": 1, "missing": 3}
    assert scales == ["log", "log"]
