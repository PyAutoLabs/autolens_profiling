"""runtime_estimates export, setup-baseline closure and the day-zero coverage witness
(autolens_profiling#503, likelihood lifecycle phase 3, day-zero write 4/4)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[1]
REPO = TOOLING.parents[2]
PULSE = REPO.parent / "PyAutoPulse"
spec = importlib.util.spec_from_file_location(
    "export_runtime_estimates", TOOLING / "export_runtime_estimates.py"
)
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)

needs_pulse = pytest.mark.skipif(
    not (PULSE / rx.SNAPSHOT).is_file(), reason="needs a PyAutoPulse checkout beside the repo"
)


@needs_pulse
def test_the_committed_export_is_current():
    assert rx.main(["--root", str(REPO), "--pulse", str(PULSE), "--check"]) == 0


@needs_pulse
def test_one_estimate_per_stamped_runtime_cell():
    snap = rx.load_snapshot(PULSE)
    rows = rx.estimates(snap)
    assert len(rows) == 116
    designations = json.loads((REPO / "catalogue/pulse_designations.json").read_text())
    assert {r["decision_id"] for r in rows if r["decision_id"]} == {
        d["id"] for d in designations["designations"]
    }
    held = [r for r in rows if r["decision_id"] is None]
    assert {(r["likelihood"], r["hardware_class"]) for r in held} == {
        ("datacube/delaunay", "gpu_a100")
    }
    assert {r["role"] for r in held} == {"provisional_optimal", "provisional_secondary"}
    assert all(r["ms_per_eval"] and r["ms_per_eval"] > 0 for r in rows)
    assert all(r["record_id"] in {s["id"] for s in snap["records"]} for r in rows)


@needs_pulse
def test_a_tampered_snapshot_is_refused(tmp_path):
    snap = json.loads((PULSE / rx.SNAPSHOT).read_text())
    bad = copy.deepcopy(snap)
    bad["records"][0]["values"] = {"x": 1.0}
    (tmp_path / rx.SNAPSHOT).parent.mkdir(parents=True)
    (tmp_path / rx.SNAPSHOT).write_text(json.dumps(bad))
    with pytest.raises(rx.ExportError, match="records_sha256"):
        rx.load_snapshot(tmp_path)


def test_the_yaml_scalars_round_trip():
    yaml = pytest.importorskip("yaml")
    for value in (None, True, 3, 0.25, "a: b", 'say "hi"', "2026.10.25.1"):
        assert yaml.safe_load(f"k: {rx._scalar(value)}")["k"] == value


def test_no_optimal_planned_cell_is_not_measured():
    """The coverage witness (PyAutoPulse ``setup_browser.not_measured``): every optimal planned
    cell has a current record on its key."""
    doc = json.loads((REPO / "dashboard/catalogue.json").read_text())
    current = {
        (
            r["likelihood"]["dataset_type"],
            r["likelihood"]["variant"],
            r["instrument"],
            r["setup_id"],
            r["host_class"],
            r["model_variant"],
            r["measurement"],
        )
        for r in doc["records"]
        if r["status"] == "current"
    }
    optimal = [c for c in doc["planned_cells"] if c["class"] == "optimal"]
    assert len(optimal) == 148
    missing = [
        c
        for c in optimal
        if (
            c["likelihood"]["dataset_type"],
            c["likelihood"]["variant"],
            c["instrument"],
            c["setup_id"],
            c["host_class"],
            c["model_variant"],
            c["measurement"],
        )
        not in current
    ]
    assert missing == []
    coverage = json.loads((REPO / "dashboard/coverage.json").read_text())
    assert {c["state"] for c in coverage["cells"] if c["class"] == "optimal"} == {"current"}


def test_the_setup_baseline_campaign_is_closed_by_day_zero():
    closure = json.loads((REPO / "baseline/campaign.json").read_text())
    assert (closure["schema"], closure["status"]) == ("setup-baseline-closure", "closed")
    assert (
        closure["superseded_by"]["tag"]
        == json.loads((REPO / "catalogue/day_zero.json").read_text())["tag"]
    )
    assert len(closure["answers"]) == 6
    draft = json.loads((REPO / closure["draft"]["path"]).read_text())
    assert draft["unknowns"] == [a["unknown"] for a in closure["answers"]]
    assert not (REPO / "scripts/misc/tooling/baseline_readiness.py").exists()
