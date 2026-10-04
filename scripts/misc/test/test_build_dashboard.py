"""``build_dashboard.py``: the scanner, the qualification rules, the drift badge, the outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
    assert state["organ"] == "autolens_profiling"
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
    assert json.loads(outputs["state.json"])["organ"] == "autolens_profiling"


# --- profiling-summary v1 (the PyAutoPulse read contract) -------------------


def _summary(tmp_path, revision="a" * 40):
    root = _tree(tmp_path)
    series, refused = bd.build_series(root, bd.read_release_sweep_conf(root))
    conf = bd.read_release_sweep_conf(root)
    return bd.build_summary(series, refused, conf, "2026-09-27T00:00:00Z", revision)


def test_summary_envelope_and_records(tmp_path):
    s = _summary(tmp_path)
    assert bd.validate_summary(s) == []
    assert (s["schema"], s["version"]) == ("profiling-summary", 1)
    assert (s["project"], s["scope"]) == ("autolens_profiling", "release-runtime")
    assert s["generated_at"] == "2026-09-27T00:00:00Z"
    assert s["evidence_updated_at"] == "2026-09-27T00:00:00Z"  # newest release date in the tree
    assert s["valid_until"] is None and s["producer_revision"] == "a" * 40
    assert s["comparison_policy"]["id"] == "runtime-drift-2x-1ms"
    assert s["comparison_policy"]["ratio"] == 2.0 and s["comparison_policy"]["floor_s"] == 0.001
    ids = [r["id"] for r in s["records"]]
    assert len(ids) == len(set(ids)) == s["coverage"]["observed"]["points"]
    a100 = next(r for r in s["records"] if r["id"].endswith(":hpc_a100_fp64@2026.9.27.1"))
    assert a100["axis"] == "runtime" and a100["unit"] == "s"
    assert a100["measurement"] == {"single_jit_s": 0.143, "vmap_per_call_s": 0.09}
    assert a100["identity"]["tier"] == "hpc" and a100["identity"]["device"] == "a100"
    assert a100["identity"]["precision"] == "float64"
    assert a100["identity"]["release_date"] == "2026-09-27"
    assert a100["provenance"]["qualified"] is True and a100["provenance"]["job"] == "360001"
    assert a100["evidence"] == {
        "path": "results/runtime/imaging/delaunay/hst/comparison.json",
        "fragment": "hpc_a100_fp64",
    }
    old = next(r for r in s["records"] if r["id"].endswith(":hpc_a100_fp64@2026.8.17.1"))
    # unknown provenance stays null + reason, never a default (the host is the row's hostname)
    assert old["provenance"]["job"] is None and old["provenance"]["loadavg"] is None
    assert old["provenance"]["has_provenance"] is False and old["provenance"]["qualified"] is False
    assert "no provenance" in old["provenance"]["reason"]
    assert old["evidence"]["fragment"] is None


def test_summary_comparisons_are_the_producers_verdict(tmp_path):
    s = _summary(tmp_path)
    by_key = {c["comparison_key"]: c for c in s["comparisons"]}
    drifted = by_key["runtime:imaging/delaunay/hst:hpc_a100_fp64"]
    assert drifted["status"] == "drifted" and drifted["ratio"] == 2.2
    assert (drifted["baseline"], drifted["candidate"]) == ("2026.8.17.1", "2026.9.27.1")
    # the baseline row has no provenance: a drift flag, not regression evidence
    assert drifted["qualified"] is False and any("baseline" in r for r in drifted["reasons"])
    assert (
        by_key["runtime:imaging/delaunay/hst:local_cpu_fd64".replace("fd", "fp")]["status"]
        == "flat"
    )
    assert by_key["breakdown:imaging/pixelization/hst:local_cpu_fp64"]["status"] == "improved"
    single = by_key["breakdown:imaging/mge/hst:hpc_a100_mp"]
    assert single["status"] == "insufficient" and "one release only" in single["reasons"]
    assert all(c["policy"] == "runtime-drift-2x-1ms" for c in s["comparisons"])


def test_summary_refused_rows_are_excluded_not_recorded(tmp_path):
    s = _summary(tmp_path)
    excluded = s["coverage"]["excluded"]
    assert len(excluded) == 1 and "above the cap" in excluded[0]["reason"]
    assert excluded[0]["id"] == "breakdown:imaging/mge/hst:hpc_a100_fp64@2026.9.27.1"
    assert not any(r["id"] == excluded[0]["id"] for r in s["records"])
    assert any("coverage.excluded" in line for line in s["limitations"])


def test_summary_valid_empty_producer(tmp_path):
    s = bd.build_summary([], [], {"node": None, "loadavg_cap": None}, "2026-09-27T00:00:00Z", None)
    assert bd.validate_summary(s) == []
    assert s["records"] == [] and s["comparisons"] == []
    assert s["coverage"]["observed"] == {"series": 0, "points": 0, "releases": 0, "cells": 0}
    assert s["evidence_updated_at"] is None
    assert s["evidence_updated_at_reason"] == "no included measurement"
    assert any("producer_revision is null" in line for line in s["limitations"])


def test_summary_validator_rejections(tmp_path):
    import copy

    good = _summary(tmp_path)
    assert bd.validate_summary(good) == []

    dup = copy.deepcopy(good)
    dup["records"][1]["id"] = dup["records"][0]["id"]
    assert any("duplicate record id" in f for f in bd.validate_summary(dup))

    nan = copy.deepcopy(good)
    nan["records"][0]["measurement"]["single_jit_s"] = float("nan")
    assert any("non-finite measurement" in f for f in bd.validate_summary(nan))

    inf = copy.deepcopy(good)
    inf["comparisons"][0]["ratio"] = float("inf")
    assert any("non-finite ratio" in f for f in bd.validate_summary(inf))

    unsafe = copy.deepcopy(good)
    unsafe["records"][0]["evidence"]["path"] = "../../etc/passwd"
    assert any("not a safe repo-relative path" in f for f in bd.validate_summary(unsafe))
    unsafe["records"][0]["evidence"]["path"] = "/mnt/ral/jnightin/results/x.json"
    assert any("not a safe repo-relative path" in f for f in bd.validate_summary(unsafe))

    empty = copy.deepcopy(good)
    empty["records"][0]["measurement"] = {"single_jit_s": None, "vmap_per_call_s": None}
    assert any("carries no measurement" in f for f in bd.validate_summary(empty))

    version = copy.deepcopy(good)
    version["version"] = 2
    assert any("version" in f for f in bd.validate_summary(version))

    missing = copy.deepcopy(good)
    del missing["coverage"]
    assert any("missing required field 'coverage'" in f for f in bd.validate_summary(missing))

    date = copy.deepcopy(good)
    date["generated_at"] = "2026-09-27 00:00:00"
    assert any("ISO-8601" in f for f in bd.validate_summary(date))
    date["generated_at"] = "2026-09-27T00:00:00Z"
    date["valid_until"] = "2026-09-26T00:00:00Z"
    assert any("precedes generated_at" in f for f in bd.validate_summary(date))

    status = copy.deepcopy(good)
    status["comparisons"][0]["status"] = "regressed"
    assert any("not in the contract" in f for f in bd.validate_summary(status))


def test_summary_is_written_checked_and_the_other_outputs_are_untouched(tmp_path):
    root = _tree(tmp_path)
    before = bd.build(root, generated="2026-09-27T00:00:00Z", revision="b" * 40)
    assert set(before) == {"series.json", "state.json", "summary.json", "index.html"}
    assert bd.main(["--root", str(root)]) == 0
    out = root / "dashboard"
    summary = json.loads((out / "summary.json").read_text())
    assert bd.validate_summary(summary) == []
    # not a git checkout: null revision + a limitation line, and --check is still idempotent
    assert summary["producer_revision"] is None
    assert bd.main(["--root", str(root), "--check"]) == 0
    # the organ-facing file never alters the page, the trend data or the cockpit feed
    stamp = bd._existing_stamp(out)
    again = bd.build(root, generated=stamp, revision=None)
    for name in ("series.json", "state.json", "index.html"):
        assert again[name] == (out / name).read_text()
    assert "summary.json" not in (out / "series.json").read_text()
    assert json.loads((out / "state.json").read_text())["organ"] == "autolens_profiling"


def test_real_tree_summary_is_valid():
    root = _TOOLING.parents[2]
    outputs = bd.build(root, generated="2026-09-27T00:00:00Z", revision="c" * 40)
    summary = json.loads(outputs["summary.json"])
    assert bd.validate_summary(summary) == []
    assert summary["coverage"]["observed"]["points"] == len(summary["records"]) > 20
    assert summary["coverage"]["observed"]["series"] == len(summary["comparisons"])
    assert len(summary["coverage"]["excluded"]) == len(
        json.loads(outputs["series.json"])["refused"]
    )


def test_gpu_single_jit_headline_is_labelled_first_block_after_compile():
    """#371 option (a): label the A100 single-JIT block; do not re-base the value."""
    gpu = bd._point(_row("2026.9.27.2", 0.000642, vmap=5.6e-6), "2026.9.27.2", "x.json")
    assert gpu["single_jit_s"] == 0.000642  # value unchanged
    assert gpu["headline_note"] == bd.FIRST_BLOCK_NOTE == "first block after compile"
    assert "single_jit_median_s" not in gpu

    cpu = bd._point(_row("2026.9.27.2", 0.000258, backend="cpu"), "2026.9.27.2", "x.json")
    assert "headline_note" not in cpu and "single_jit_median_s" not in cpu

    # The sweep aggregator's ``full_pipeline_per_call`` alias of the same statistic is labelled too.
    alias = _row("2026.9.27.2", 0.000642)
    alias["full_pipeline_per_call"] = 0.000642
    assert bd._point(alias, "2026.9.27.2", "c.json#hpc_a100_fp64")["headline_note"]

    # A GPU row headlined by a different statistic is not.
    other = {"autolens_version": "1", "device": {"backend": "gpu"}, "total_step_by_step": 0.2}
    assert "headline_note" not in bd._point(other, "1", "b.json")


def test_steady_median_field_rides_beside_the_headline():
    row = _row("2026.10.4.1", 0.000642)
    row["full_pipeline_single_jit_median_ms"] = 0.267
    p = bd._point(row, "2026.10.4.1", "x.json")
    assert p["single_jit_s"] == 0.000642
    assert p["single_jit_median_s"] == pytest.approx(0.000267)
    cell = bd._per_call_html(p)
    assert (
        "0.64 ms" in cell
        and "first block after compile" in cell
        and "steady median 0.27 ms" in cell
    )
