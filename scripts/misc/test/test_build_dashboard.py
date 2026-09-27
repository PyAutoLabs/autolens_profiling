"""``build_dashboard.py``: the scanner, the qualification rules, the drift badge, the outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_TOOLING = Path(__file__).resolve().parents[1] / "tooling"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve the defining module through sys.modules
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_dashboard")


def _prov(host="euclid-ral-gpu-2", loadavg=0.5, job="360001"):
    return {
        "provenance_schema": 1,
        "captured_at": "2026-09-27T00:00:00Z",
        "host": host,
        "slurm": {"job_id": job, "array_job_id": None, "array_task_id": None},
        "loadavg_at_import": [loadavg, loadavg, loadavg],
        "loadavg_at_write": [loadavg, loadavg, loadavg],
        "profiling_revision": "abc",
        "library_revisions": {},
        "library_versions": {},
        "dependency_versions": {},
    }


def _row(version, single, vmap=None, backend="gpu", hostname=None, prov=None, instrument="hst"):
    device = {"backend": backend}
    if hostname:
        device["hostname"] = hostname
    if prov:
        device["provenance"] = prov
    d = {
        "autolens_version": version,
        "device": device,
        "instrument": instrument,
        "full_pipeline_single_jit": single,
    }
    if vmap is not None:
        d["vmap"] = {"batch_size": 16, "per_call": vmap}
    return d


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "hpc").mkdir(parents=True)
    (root / "hpc" / "release_sweep.conf").write_text(
        "RELEASE_SWEEP_NODE=euclid-ral-gpu-2\nRELEASE_SWEEP_LOADAVG_CAP=8.0\n"
    )
    rt = root / "results" / "runtime" / "imaging" / "delaunay"
    rt.mkdir(parents=True)
    # versioned local rows (no provenance): two releases, steady
    (rt / "delaunay_likelihood_summary_hst_v2026.8.17.1.json").write_text(
        json.dumps(_row("2026.8.17.1", 4.5, backend="cpu"))
    )
    (rt / "delaunay_likelihood_summary_hst_v2026.9.27.1.json").write_text(
        json.dumps(_row("2026.9.27.1", 4.7, backend="cpu"))
    )
    # A100 rows: old without provenance, new on the reference host -> drifted 2.2x
    (rt / "delaunay_hpc_a100_fp64.json").write_text(
        json.dumps(_row("2026.8.17.1", 0.065, vmap=0.0425, hostname="euclid-ral-gpu-2"))
    )
    (rt / "hst").mkdir()
    (rt / "hst" / "comparison.json").write_text(
        json.dumps(
            {"configs": {"hpc_a100_fp64": _row("2026.9.27.1", 0.143, vmap=0.09, prov=_prov())}}
        )
    )
    # a loaded-host row -> refused; an off-host row -> hollow
    bd_dir = root / "results" / "breakdown" / "imaging"
    bd_dir.mkdir(parents=True)
    (bd_dir / "mge_hpc_a100_fp64.json").write_text(
        json.dumps(_row("2026.9.27.1", 0.008, prov=_prov(loadavg=190.0, job="357321")))
    )
    (bd_dir / "mge_hpc_a100_mp.json").write_text(
        json.dumps(_row("2026.9.27.1", 0.007, prov=_prov(host="euclid-ral-gpu-1")))
    )
    # a versioned breakdown pair with an improvement
    (bd_dir / "pixelization_breakdown_hst_v2026.8.17.1.json").write_text(
        json.dumps(
            {
                "autolens_version": "2026.8.17.1",
                "device": {"backend": "cpu"},
                "instrument": "hst",
                "total_step_by_step": 9.0,
            }
        )
    )
    (bd_dir / "pixelization_breakdown_hst_v2026.9.27.1.json").write_text(
        json.dumps(
            {
                "autolens_version": "2026.9.27.1",
                "device": {"backend": "cpu"},
                "instrument": "hst",
                "total_step_by_step": 3.0,
            }
        )
    )
    # ledger tree: ignored
    (root / "results" / "notes").mkdir()
    (root / "results" / "notes" / "x.md").write_text("# x\n")
    return root


def test_grammar_matches_build_readme():
    br = _load("build_readme")
    assert bd.ARTIFACT_RE.pattern == br.ARTIFACT_RE.pattern
    assert bd.CONFIG_TAGGED_RE.pattern == br.CONFIG_TAGGED_RE.pattern
    assert bd.CONFIG_ORDER == br.CONFIG_ORDER


def test_conf_is_read(tmp_path):
    root = _tree(tmp_path)
    assert bd.read_release_sweep_conf(root) == {"node": "euclid-ral-gpu-2", "loadavg_cap": 8.0}
    assert bd.read_release_sweep_conf(tmp_path) == {"node": None, "loadavg_cap": None}


def test_series_qualification_and_drift(tmp_path):
    root = _tree(tmp_path)
    series, refused = bd.build_series(root, bd.read_release_sweep_conf(root))
    by_key = {s["key"]: s for s in series}
    a100 = by_key["runtime:imaging/delaunay/hst:hpc_a100_fp64"]
    assert [p["version"] for p in a100["points"]] == ["2026.8.17.1", "2026.9.27.1"]
    assert (
        a100["points"][0]["qualified"] is False and "no provenance" in a100["points"][0]["reason"]
    )
    assert a100["points"][1]["qualified"] is True and a100["points"][1]["job"] == "360001"
    assert a100["drift"]["status"] == "drifted" and a100["drift"]["ratio"] == 2.2
    local = by_key["runtime:imaging/delaunay/hst:local_cpu_fp64"]
    assert local["drift"]["status"] == "steady"
    pix = by_key["breakdown:imaging/pixelization/hst:local_cpu_fp64"]
    assert pix["drift"]["status"] == "improved" and pix["points"][-1]["single_jit_s"] == 3.0
    off = by_key["breakdown:imaging/mge/hst:hpc_a100_mp"]
    assert (
        off["points"][0]["qualified"] is False
        and "off the reference host" in off["points"][0]["reason"]
    )
    assert "breakdown:imaging/mge/hst:hpc_a100_fp64" not in by_key
    assert (
        len(refused) == 1
        and refused[0]["job"] == "357321"
        and "above the cap" in refused[0]["reason"]
    )


def test_drift_edge_cases():
    assert bd.drift([])["status"] == "no-data"
    one = [{"version": "1", "single_jit_s": 1.0}]
    assert bd.drift(one)["status"] == "single-release"
    tiny = [{"version": "1", "single_jit_s": 0.0002}, {"version": "2", "single_jit_s": 0.0005}]
    assert bd.drift(tiny)["status"] == "steady"  # 2.5x but under the 1 ms floor


def test_state_feed_contract(tmp_path):
    root = _tree(tmp_path)
    series, _ = bd.build_series(root, bd.read_release_sweep_conf(root))
    state = bd.build_state(series, "2026-09-27T00:00:00Z")
    for key in (
        "schema_version",
        "organ",
        "repo",
        "status",
        "headline",
        "updated",
        "pages_url",
        "items",
    ):
        assert key in state
    assert state["status"] == "yellow" and state["items"][0]["severity"] == "yellow"
    assert "2.2x" in state["items"][0]["text"] and state["items"][0]["prompt"].startswith(
        "/profiling triage"
    )
    assert bd.build_state([], "2026-09-27T00:00:00Z")["status"] == "grey"


def test_build_writes_and_check_is_idempotent(tmp_path, capsys):
    root = _tree(tmp_path)
    assert bd.main(["--root", str(root), "--check"]) == 1  # nothing committed yet
    assert bd.main(["--root", str(root)]) == 0
    out = root / "dashboard"
    assert (
        (out / "index.html").is_file()
        and (out / "series.json").is_file()
        and (out / "state.json").is_file()
    )
    page = (out / "index.html").read_text()
    assert (
        "imaging/delaunay/hst" in page
        and "drifted" in page
        and 'class="hollow"' in page
        and "refused rows" in page
    )
    assert "<title>" in page and "prefers-color-scheme" in page
    assert bd.main(["--root", str(root), "--check"]) == 0
    # data change -> stale
    (
        root
        / "results"
        / "runtime"
        / "imaging"
        / "delaunay"
        / "delaunay_likelihood_summary_hst_v2026.9.27.1.json"
    ).write_text(json.dumps(_row("2026.9.27.1", 40.0, backend="cpu")))
    assert bd.main(["--root", str(root), "--check"]) == 1
    assert "STALE" in capsys.readouterr().out


def test_real_tree_renders():
    root = _TOOLING.parents[2]
    outputs = bd.build(root, generated="2026-09-27T00:00:00Z")
    doc = json.loads(outputs["series.json"])
    assert doc["reference_host"] == "euclid-ral-gpu-2"
    assert len(doc["series"]) > 20
    assert json.loads(outputs["state.json"])["organ"] == "profiling"
