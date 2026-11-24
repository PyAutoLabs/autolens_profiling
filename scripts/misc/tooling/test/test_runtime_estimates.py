"""runtime_estimates export, setup-baseline closure and the day-zero coverage witness
(autolens_profiling#503, likelihood lifecycle phase 3, day-zero write 4/4)."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
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


#: The v2 size fields each instrument was timed at (evidence ``configuration`` blocks).
IMAGING_PIXELS = {"euclid": 3841, "hst": 15361, "jwst": 42737, "ao": 384753}
INTERFEROMETER = {
    "sma": (3852, 190),
    "alma": (15380, 1_000_000),
    "alma_high": (61572, 5_000_000),
    "jvla": (384852, 25_000_000),
}


@needs_pulse
def test_v2_size_fields_match_each_instrument():
    rows = rx.estimates(rx.load_snapshot(PULSE), REPO)
    assert all({"n_pixels_masked", "n_vis", "source_pixels"} <= set(r) for r in rows)
    for r in rows:
        dataset_type, variant = r["likelihood"].split("/")
        if dataset_type == "imaging":
            assert r["n_pixels_masked"] == IMAGING_PIXELS[r["instrument"]], r["record_id"]
            assert r["n_vis"] is None
        if dataset_type in ("interferometer", "datacube"):
            pixels, vis = INTERFEROMETER[r["instrument"]]
            assert r["n_vis"] == vis, r["record_id"]
            if variant != "mge":
                assert r["n_pixels_masked"] == pixels, r["record_id"]
        if dataset_type in ("cluster", "weak", "point_source_image", "point_source_source"):
            assert (r["n_pixels_masked"], r["n_vis"], r["source_pixels"]) == (None,) * 3
        expected = {"rectangular": 1225, "delaunay": 1250, "delaunay_nn": 1250}.get(variant)
        assert r["source_pixels"] == expected, r["record_id"]
    assert {r["likelihood"] for r in rows if not (r["n_pixels_masked"] or r["n_vis"])} == {
        "cluster/image_plane",
        "cluster/source_plane",
        "weak/isothermal",
        "point_source_image/image_plane",
        "point_source_source/source_plane",
    }


@needs_pulse
def test_a_missing_evidence_file_is_refused(tmp_path):
    with pytest.raises(rx.ExportError, match="evidence"):
        rx.estimates(rx.load_snapshot(PULSE), tmp_path)


@needs_pulse
def test_v2_header_carries_the_measured_efficiency_curve():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((REPO / rx.OUT).read_text())
    assert (doc["schema"], doc["version"]) == ("runtime-estimates", 2)
    eta = doc["parallel_efficiency"]
    assert eta["curve"] == {1: 1.0, 2: 0.68, 4: 0.62, 8: 0.55, 16: 0.36, 32: 0.28}
    assert (eta["likelihood"], eta["setup_id"], eta["kind"]) == (
        "imaging/rectangular",
        "numba_cpu_sparse_fp64",
        "measured",
    )
    assert "wall/model.py" in eta["source"]
    assert set(doc["size_fields"]) == {"n_pixels_masked", "n_vis", "source_pixels"}
    assert doc["cells"] == len(doc["estimates"]) == 116
    assert all(set(doc["size_fields"]) <= set(e) for e in doc["estimates"])


def test_the_embedded_curve_matches_autolens_inference():
    """The exporter's copy of the curve equals autolens_inference's, when that checkout is beside
    this repo (CI has none; parsed, never imported)."""
    import ast

    model = REPO.parent / "autolens_inference/scripts/misc/wall/model.py"
    if not model.is_file():
        pytest.skip("needs an autolens_inference checkout beside the repo")
    for node in ast.parse(model.read_text()).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == (
            "PARALLEL_EFFICIENCY"
        ):
            assert ast.literal_eval(node.value) == rx.PARALLEL_EFFICIENCY
            return
    pytest.fail("PARALLEL_EFFICIENCY not found in wall/model.py")


def _evidence_root(root: Path) -> Path:
    """A stand-in repo root holding only the evidence files the snapshot's runtime records name."""
    for record in rx.load_snapshot(PULSE)["records"]:
        if record["measurement"] == "runtime":
            target = root / record["evidence"]["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / record["evidence"]["path"], target)
    (root / "catalogue").mkdir(parents=True, exist_ok=True)
    return root


@needs_pulse
def test_copy_to_writes_the_identical_file(tmp_path, capsys):
    root = _evidence_root(tmp_path / "profiling")
    lens = tmp_path / "PyAutoLens"
    (lens / "autolens").mkdir(parents=True)
    argv = ["--root", str(root), "--pulse", str(PULSE), "--copy-to", str(lens)]
    assert rx.main(argv) == 0
    copied = lens / rx.COPY_TARGET
    assert copied.read_text() == (root / rx.OUT).read_text() == (REPO / rx.OUT).read_text()
    assert rx.main(["--root", str(root), "--pulse", str(PULSE), "--check"]) == 0


def test_copy_to_refuses_a_checkout_without_the_package(tmp_path, capsys):
    assert rx.main(["--pulse", str(tmp_path), "--copy-to", str(tmp_path)]) == 1
    assert "no autolens/" in capsys.readouterr().out
    assert not (tmp_path / rx.COPY_TARGET).exists()


@needs_pulse
def test_check_reports_drift(tmp_path, capsys):
    root = _evidence_root(tmp_path / "profiling")
    (root / rx.OUT).write_text((REPO / rx.OUT).read_text().replace("version: 2", "version: 1"))
    assert rx.main(["--root", str(root), "--pulse", str(PULSE), "--check"]) == 1
    assert "STALE" in capsys.readouterr().out


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
