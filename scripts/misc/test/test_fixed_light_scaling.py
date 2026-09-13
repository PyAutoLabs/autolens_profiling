"""Unit tests for the fixed-lens-light source-pixel scaling aggregator (#257).

Synthetic manifest + minimal result JSONs in a tmp dir. No JAX, no real result
files: the point is the **folding** — which legs the aggregator finds, what it
does with a leg that died, the power-law exponent it fits, and the threshold
crossing it refuses to extrapolate — plus the cell's default flags, which are
what keeps the phase-0 A100 JSONs reproducible.

The module is loaded straight off its path rather than as
``likelihood_breakdown.fixed_light_scaling_table``: that package's ``__init__``
imports ``timing``, which imports JAX, and this file has no business paying a
JAX import to parse JSON.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_scaling.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys as _sys
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_MODULE_PATH = ROOT / "scripts" / "misc" / "likelihood_breakdown" / "fixed_light_scaling_table.py"
_spec = importlib.util.spec_from_file_location("fixed_light_scaling_table_under_test", _MODULE_PATH)
fst = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = fst
_spec.loader.exec_module(fst)

CELL_PATH = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light.py"


# ---------------------------------------------------------------------------
# fixtures — the smallest JSON carrying the keys the aggregator reads
# ---------------------------------------------------------------------------


def _leg_json(
    *,
    source_pixels: int,
    certified_ms: float,
    safe_ms: float,
    unconstrained_ms: float,
    pdip_ms: float,
    library_s3_ms: float,
    certifying_budget: int = 5,
    safe_budget: int = 11,
    peak_bytes: int | None = 1_600_000_000,
) -> dict:
    return {
        "autolens_version": "2026.8.17.1",
        "device": {"hostname": "test"},
        "machine": {"cpu_model": "test"},
        "precision": {"pins_mode": "fp64", "tau_rel": 1e-9},
        "instrument": "hst",
        "configuration": {
            "source_pixels": source_pixels,
            "inversion_path": "dense",
            "safe_budget": safe_budget,
            "pins_mode": "fp64",
        },
        "s3": {
            "rows": {
                "curvature_reg_matrix build (dense)": 0.05,
                "Cholesky solve (unconstrained)": unconstrained_ms / 1e3,
                "NNLS PDIP (cell-driven, max_iter 50)": pdip_ms / 1e3,
                "Log det Cholesky (F+λH reduced)": 0.003,
                "Log det Cholesky (H reduced)": 0.003,
            },
            "nnls": {"iterations": 15},
        },
        "active_set": {
            "smallest_certifying_budget": certifying_budget,
            "safe_budget": safe_budget,
            "safe_budget_ms": safe_ms,
            "budgets": [
                {"pass_budget": certifying_budget, "ms": certified_ms, "certified": True},
                {"pass_budget": safe_budget, "ms": safe_ms, "certified": True},
            ],
        },
        "library_row": {"status": "ok", "s3_ms": library_s3_ms, "s0_ms": library_s3_ms * 1.3},
        "peak_bytes": {"after_all": peak_bytes},
    }


@pytest.fixture
def sweep(tmp_path):
    """A rectangular RTX sweep: three legs measured, one OOM, one never run.

    Times are an exact ``ms = 1e-6 * N**2`` power law so the fitted exponent has
    a known answer.
    """
    results = tmp_path / "results"
    results.mkdir()
    manifest = tmp_path / "manifest.jsonl"

    lines = []
    for n in (500, 1000, 1500):
        built = fst.built_pixels("rectangular", n)
        payload = _leg_json(
            source_pixels=built,
            certified_ms=1e-6 * built**2,
            safe_ms=2e-6 * built**2,
            unconstrained_ms=1e-7 * built**2,
            pdip_ms=5e-6 * built**2,
            library_s3_ms=1e-5 * built**2,
        )
        name = f"fixed_light_rectangular_n{built}_local_rtx2060_fp64_fixed_light.json"
        (results / name).write_text(json.dumps(payload))
        lines.append(
            {"mesh": "rectangular", "n": n, "hardware": "local_rtx2060_fp64", "status": "ok"}
        )

    lines.append(
        {"mesh": "rectangular", "n": 2500, "hardware": "local_rtx2060_fp64", "status": "oom"}
    )
    # n=4000 is deliberately absent from the manifest: never launched.
    manifest.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    return results, manifest


# ---------------------------------------------------------------------------
# built-pixel arithmetic — the filename the cell actually writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested,expected",
    [(500, 484), (1000, 1024), (1500, 1521), (2500, 2500), (4000, 3969)],
)
def test_rectangular_rounds_to_the_nearest_square(requested, expected):
    assert fst.built_pixels("rectangular", requested) == expected


@pytest.mark.parametrize("requested", [500, 1000, 1500, 2500, 4000])
def test_delaunay_builds_exactly_what_was_requested(requested):
    assert fst.built_pixels("delaunay", requested) == requested


def test_json_candidate_uses_the_built_count_not_the_request():
    assert fst.json_candidates("rectangular", 500, "cfg") == ["fixed_light_rectangular_n484_cfg"]
    assert fst.json_candidates("delaunay", 500, "cfg") == ["fixed_light_delaunay_n500_cfg"]


# ---------------------------------------------------------------------------
# collection — a leg that died is still a row
# ---------------------------------------------------------------------------


def test_collect_finds_every_cell_of_the_grid(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    assert len(legs) == len(fst.HARDWARE) * len(fst.MESHES) * len(fst.SIZES)


def test_measured_legs_are_ok_and_carry_their_built_count(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    leg = legs[("rectangular", 1500, "local_rtx2060_fp64")]
    assert leg.status == "ok"
    assert leg.fits is True
    assert leg.built == 1521
    assert leg.certifying_budget == 5
    assert leg.safe_budget == 11


def test_an_oom_leg_is_a_row_not_a_gap(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    leg = legs[("rectangular", 2500, "local_rtx2060_fp64")]
    assert leg.status == "oom"
    assert leg.fits is False
    assert leg.fits_cell() == "OOM"
    assert all(value is None for value in leg.metrics().values())


def test_a_leg_never_launched_is_distinguishable_from_one_that_died(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    assert legs[("rectangular", 4000, "local_rtx2060_fp64")].status == "missing"
    assert legs[("rectangular", 4000, "local_rtx2060_fp64")].fits_cell() == "not run"


def test_a_manifest_claiming_ok_without_a_json_is_not_believed(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"mesh": "delaunay", "n": 500, "hardware": "local_cpu_fp64", "status": "ok"})
        + "\n"
    )
    legs = fst.collect(results, manifest)
    assert legs[("delaunay", 500, "local_cpu_fp64")].status == "failed"


def test_an_absent_manifest_is_not_an_error(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    legs = fst.collect(results, tmp_path / "nope.jsonl")
    assert all(leg.status == "missing" for leg in legs.values())


def test_peak_bytes_becomes_gigabytes(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    leg = legs[("rectangular", 500, "local_rtx2060_fp64")]
    assert leg.peak_gb() == pytest.approx(1_600_000_000 / 1024**3)
    assert leg.fits_cell().startswith("✓ 1.4")


def test_a_backend_without_memory_stats_still_fits(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    payload = _leg_json(
        source_pixels=500,
        certified_ms=1.0,
        safe_ms=2.0,
        unconstrained_ms=0.5,
        pdip_ms=5.0,
        library_s3_ms=50.0,
        peak_bytes=None,
    )
    name = "fixed_light_delaunay_n500_local_cpu_fp64_fixed_light.json"
    (results / name).write_text(json.dumps(payload))
    legs = fst.collect(results, tmp_path / "nope.jsonl")
    assert legs[("delaunay", 500, "local_cpu_fp64")].fits_cell() == "✓"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_metrics_read_the_certifying_and_the_safe_budget_rows(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    metrics = legs[("rectangular", 1000, "local_rtx2060_fp64")].metrics()
    built = 1024
    assert metrics["certified_certifying_ms"] == pytest.approx(1e-6 * built**2)
    assert metrics["certified_safe_ms"] == pytest.approx(2e-6 * built**2)
    assert metrics["unconstrained_ms"] == pytest.approx(1e-7 * built**2)
    assert metrics["pdip_ms"] == pytest.approx(5e-6 * built**2)
    assert metrics["library_s3_ms"] == pytest.approx(1e-5 * built**2)


def test_a_missing_safe_budget_row_is_none_not_an_error(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    payload = _leg_json(
        source_pixels=500,
        certified_ms=1.0,
        safe_ms=2.0,
        unconstrained_ms=0.5,
        pdip_ms=5.0,
        library_s3_ms=50.0,
    )
    payload["active_set"]["budgets"] = [{"pass_budget": 5, "ms": 1.0, "certified": True}]
    payload["active_set"]["safe_budget_ms"] = None
    name = "fixed_light_delaunay_n500_local_cpu_fp64_fixed_light.json"
    (results / name).write_text(json.dumps(payload))
    legs = fst.collect(results, tmp_path / "nope.jsonl")
    metrics = legs[("delaunay", 500, "local_cpu_fp64")].metrics()
    assert metrics["certified_safe_ms"] is None
    assert metrics["certified_certifying_ms"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the exponent fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0, 2.5])
def test_exponent_fit_recovers_a_clean_power_law(alpha):
    points = [(float(n), 1e-6 * n**alpha) for n in (500, 1000, 1500, 2500, 4000)]
    fit = fst.fit_exponent(points)
    assert fit["alpha"] == pytest.approx(alpha, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 5


def test_exponent_fit_on_noisy_data_is_close_but_not_perfect():
    points = [
        (float(n), 1e-6 * n**2 * (1.0 + 0.05 * ((-1) ** i)))
        for i, n in enumerate((500, 1000, 1500, 2500, 4000))
    ]
    fit = fst.fit_exponent(points)
    assert fit["alpha"] == pytest.approx(2.0, abs=0.1)
    assert 0.9 < fit["r_squared"] < 1.0


def test_exponent_fit_refuses_a_single_point():
    fit = fst.fit_exponent([(1500.0, 12.0)])
    assert fit["alpha"] is None
    assert fit["r_squared"] is None
    assert fit["n_points"] == 1


def test_exponent_fit_ignores_missing_and_nonpositive_values():
    points = [(500.0, None), (1000.0, 0.0), (1500.0, 2.25), (3000.0, 9.0)]
    fit = fst.fit_exponent(points)
    assert fit["n_points"] == 2
    assert fit["alpha"] == pytest.approx(2.0, rel=1e-9)


def test_exponent_fit_of_a_flat_row_is_zero():
    points = [(float(n), 7.0) for n in (500, 1000, 2000)]
    fit = fst.fit_exponent(points)
    assert fit["alpha"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# the threshold crossing — interpolated between measured points, never beyond
# ---------------------------------------------------------------------------


def test_crossing_is_exact_on_a_power_law():
    points = [(float(n), 1e-4 * n**2) for n in (500, 1000, 1500, 2500, 4000)]
    # 1e-4 * N^2 = 100 ms  ->  N = 1000
    assert fst.crossing_n(points, 100.0) == pytest.approx(1000.0, rel=1e-9)


def test_crossing_above_the_measured_range_is_not_extrapolated():
    points = [(float(n), 1e-6 * n**2) for n in (500, 1000, 1500)]
    assert fst.crossing_n(points, 1000.0) is None


def test_crossing_below_the_measured_range_is_not_extrapolated():
    points = [(float(n), 1e-2 * n**2) for n in (500, 1000, 1500)]
    assert fst.crossing_n(points, 1.0) is None


def test_crossing_needs_two_points():
    assert fst.crossing_n([(1500.0, 100.0)], 100.0) is None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_rendered_table_names_every_row_and_marks_the_dead_legs(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    text = fst.render(legs, "local_rtx2060_fp64")
    for _key, label in fst.TABLE_ROWS:
        assert label in text
    assert "OOM" in text
    assert "not run" in text
    assert "rectangular" in text and "delaunay" in text


def test_rendered_table_prints_the_built_count_beside_the_request(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    text = fst.render(legs, "local_rtx2060_fp64")
    assert "500 (484)" in text
    assert "1500 (1521)" in text


def test_rendered_table_prints_the_fitted_exponent(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    text = fst.render(legs, "local_rtx2060_fp64")
    # Every synthetic row is an exact N^2 law.
    assert "| 2.00 |" in text


def test_rendered_table_says_not_measured_rather_than_extrapolating(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    text = fst.render(legs, "local_rtx2060_fp64")
    assert "never extrapolates" in text


def test_a_hardware_with_no_legs_renders_an_empty_but_valid_table(sweep):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    text = fst.render(legs, "hpc_a100_fp64")
    assert "hpc_a100_fp64" in text
    assert "not run" in text


def test_figure_is_not_written_when_nothing_was_measured(sweep, tmp_path):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    path = tmp_path / "empty.png"
    assert fst.save_figure(legs, "hpc_a100_fp64", path) is False
    assert not path.exists()


def test_figure_is_written_when_something_was_measured(sweep, tmp_path):
    results, manifest = sweep
    legs = fst.collect(results, manifest)
    path = tmp_path / "drawn.png"
    assert fst.save_figure(legs, "local_rtx2060_fp64", path) is True
    assert path.exists() and path.stat().st_size > 0


def test_main_writes_one_markdown_per_hardware(sweep, tmp_path):
    results, manifest = sweep
    out = tmp_path / "out"
    rc = fst.main(
        [
            "--results-dir",
            str(results),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out),
            "--hardware",
            "local_rtx2060_fp64",
        ]
    )
    assert rc == 0
    assert (out / "fixed_light_scaling_local_rtx2060_fp64.md").exists()
    assert (out / "fixed_light_scaling_local_rtx2060_fp64.png").exists()


# ---------------------------------------------------------------------------
# the cell's defaults — what keeps the phase-0 A100 JSONs reproducible
# ---------------------------------------------------------------------------


def _cell_source() -> str:
    return CELL_PATH.read_text()


def test_pins_defaults_to_fp64():
    assert '_cell_parser.add_argument("--pins", choices=("fp64", "none"), default="fp64")' in (
        _cell_source()
    )


def test_pins_are_asserted_only_at_the_fiducial_source_pixel_count():
    source = _cell_source()
    # The fiducial gate is a separate branch from the pins-mode gate, so a
    # non-fiducial N skips the assertion whatever the pins mode.
    assert "if not PINS_ASSERT:" in source
    assert "elif not PIN_IS_FIDUCIAL:" in source
    assert "PIN_IS_FIDUCIAL = n_source_pixels == FIDUCIAL_SOURCE_PIXELS[MESH]" in source


def test_default_pass_budgets_are_unchanged_from_phase_0():
    source = _cell_source()
    assert '_cell_parser.add_argument("--pass-budget-max", type=int, default=8)' in source
    # The safe budget is REPORTED from the sweep, never appended to it: a default
    # run measures exactly the budgets phase 0 measured.
    assert "PASS_BUDGETS = tuple(range(1, PASS_BUDGET_MAX + 1))" in source


def test_safe_budget_defaults_to_phase_3s_zero_fallback_budgets():
    source = _cell_source()
    assert 'PHASE3_SAFE_BUDGET = {"rectangular": 11, "delaunay": 7, "delaunay_nn": 7}' in source


def test_tau_rel_is_the_fp64_default_unless_mixed_precision_is_on():
    source = _cell_source()
    assert "if USE_MIXED_PRECISION:" in source
    assert "TAU_REL = _EPS_F32 * math.sqrt(float(n_image_pixels))" in source
    assert "TAU_REL = active_set_steps.TAU_REL_DEFAULT" in source


def test_every_active_set_call_carries_the_re_derived_tau_rel():
    source = _cell_source()
    calls = re.findall(r"active_set_steps\.active_set_masked_jax\((.*?)\)", source, re.S)
    assert calls, "the cell no longer calls active_set_masked_jax"
    for call in calls:
        assert "tau_rel=TAU_REL" in call


def test_the_cell_records_the_machine_and_precision_blocks():
    source = _cell_source()
    assert '"machine": machine_info_dict(),' in source
    assert '"tau_rel_basis": TAU_REL_BASIS,' in source


def test_the_safe_budget_row_is_never_interpolated():
    source = _cell_source()
    assert '"safe_budget_measured": bool(_safe_budget_entry is not None),' in source
    assert "measured, not extrapolated, so the row is null" in source


def test_hardware_labels_cover_the_four_legs_of_the_phase():
    assert set(fst.HARDWARE) == {
        "hpc_a100_fp64",
        "local_rtx2060_fp64",
        "local_rtx2060_mp",
        "local_cpu_fp64",
    }


def test_sweep_sizes_are_the_phase_prompt_s_five():
    assert fst.SIZES == (500, 1000, 1500, 2500, 4000)


def test_math_import_is_used_for_the_log_log_fit():
    # A guard against the fit quietly becoming linear: the exponent must be a
    # log-log slope, so `math.log` has to appear in the fit.
    source = _MODULE_PATH.read_text()
    fit_block = source.split("def fit_exponent")[1].split("def crossing_n")[0]
    assert "math.log" in fit_block
    assert math.log(1.0) == 0.0
