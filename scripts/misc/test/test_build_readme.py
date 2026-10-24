"""``build_readme.py`` reads the mirrored results tree natively, keyed on the v3 identity."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_TOOLING = Path(__file__).resolve().parents[1] / "tooling"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("build_readme")


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_runtime_cells_are_mirrored_cells_then_misc(tmp_path):
    results = tmp_path / "results"
    _write(results / "imaging/rectangular/hst/comparison.json", {"configs": {}})
    _write(results / "misc/likelihood_runtime/point_source_source/x_solved/comparison.json", {})
    _write(results / "misc/hazards/a/b/comparison.json", {})  # not a runtime group
    cells = [c.cell_id for c in br._scan_runtime_cells(results)]
    assert cells == ["imaging/rectangular/hst", "point_source_source/x_solved"]


def test_breakdown_rows_carry_the_v3_identity_and_unresolved_rows(tmp_path):
    results = tmp_path / "results"
    cell = results / "imaging/delaunay/hst/jax_gpu_sparse_mixed/hpc_a100"
    _write(cell / "breakdown.json", {"autolens_version": "1.2", "total_step_by_step": 0.07})
    _write(cell / "experiments/arm/breakdown.json", {"total_step_by_step": 9.0})  # never current
    _write(
        results / "breakdown/cluster/image_plane_hpc_ral_cpu_fp64.json",
        {"autolens_version": "1.1", "total_step_by_step": 4.5},
    )
    rows = br._scan_breakdown_rows(results)
    assert [(r.cell, r.setup, r.host, r.sparse) for r in rows] == [
        ("imaging/delaunay", "jax_gpu_sparse_mixed", "hpc_a100", True),
        ("cluster/image_plane", None, "hpc_ral_cpu_fp64", False),
    ]
    table = br._render_breakdown_table(rows)
    assert "| `imaging/delaunay` | hst | `jax_gpu_sparse_mixed` | hpc_a100 | sparse" in table
    assert "| `cluster/image_plane` | — | _not recorded_ | hpc_ral_cpu_fp64 |" in table


def test_archived_baselines_join_the_current_cell(tmp_path):
    root = tmp_path
    _write(
        root / "catalogue/series_id_map.json",
        {"cells": {"imaging/pixelization/hst": "imaging/rectangular/hst"}},
    )
    _write(
        root / "archive/2026/07/PreOptimizationTimes/imaging/pixelization/hst/comparison.json", {}
    )
    _write(root / "archive/2026/07/PreOptimizationTimes/imaging/mge/hst/comparison.json", {})
    baselines = br._scan_baselines(root / "archive")
    assert list(baselines) == ["PreOptimizationTimes"]
    assert sorted(c.cell_id for c in baselines["PreOptimizationTimes"]) == [
        "imaging/mge/hst",
        "imaging/rectangular/hst",
    ]


def test_archived_artifacts_are_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(br, "ARCHIVE_ROOT", tmp_path / "archive")
    _write(
        tmp_path / "archive/2026/05/pre_2026_09/simulators/imaging_summary_v2026.5.14.2.json", {}
    )
    _write(tmp_path / "results/misc/simulators/imaging_summary_v2026.10.1.1.json", {})
    _write(tmp_path / "results/lens/deflections/total_summary_hst_v2026.10.1.1.json", {})
    arts = br._scan_artifacts()
    labels = sorted((a.section, a.version_label) for a in arts)
    assert labels == [
        ("lens", "v2026.10.1.1"),
        ("simulators", "v2026.10.1.1"),
        ("simulators", "v2026.5.14.2 (archived)"),
    ]


def test_published_rows_without_a_v3_cell_stay_listed_archived_ones_labelled(tmp_path):
    results = tmp_path / "results"
    archive = tmp_path / "archive/2026/08/hostless_summaries/breakdown/imaging"
    # an archived group with no newer release anywhere: listed, labelled archived
    _write(archive / "mge_breakdown_hst_v2026.8.17.1.json", {"total_step_by_step": 0.04})
    # an archived release superseded by a newer one the map moved into a mirrored cell
    _write(archive / "delaunay_breakdown_hst_v2026.7.6.649.json", {"total_step_by_step": 9.0})
    cell = results / "imaging/delaunay/hst/jax_cpu_dense_fp64/local_laptop/breakdown.json"
    _write(cell, {"autolens_version": "2026.8.17.1", "total_step_by_step": 3.3})
    _write(
        tmp_path / "catalogue/migration_day_zero.json",
        {
            "entries": [
                {
                    "old": "results/breakdown/imaging/delaunay_breakdown_hst_v2026.8.17.1.json",
                    "new": cell.relative_to(tmp_path).as_posix(),
                }
            ]
        },
    )
    # a current row with no derivable cell (results/misc/likelihood_breakdown/): plain
    _write(
        results / "misc/likelihood_breakdown/datacube/inversion_hpc_a100_mp.json",
        {"autolens_version": "2026.7.6.649", "instrument": "alma_high"},
    )
    rows = br._legacy_breakdown_rows(results)
    labels = sorted((r.cell, r.host, r.version_label) for r in rows)
    assert labels == [
        ("datacube/inversion", "hpc_a100_mp", "v2026.7.6.649"),
        ("imaging/mge", "local_cpu_fp64", "v2026.8.17.1 (archived)"),
    ]
