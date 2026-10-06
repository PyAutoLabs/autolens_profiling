"""Baseline screening stays read-only and never promotes archived or weak evidence."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
module_spec = importlib.util.spec_from_file_location(
    "baseline_readiness", ROOT / "scripts/misc/tooling/baseline_readiness.py"
)
baseline = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(baseline)


@pytest.fixture
def draft():
    return baseline.read_json(ROOT / "baseline/campaign.json")


@pytest.fixture
def frozen(draft):
    spec = copy.deepcopy(draft)
    spec["status"] = "frozen"
    spec["software"] = {
        "revisions": {k: "a" * 40 for k in baseline.REPOS},
        "environment_lock_sha256": "b" * 64,
    }
    spec["collection"] = {
        "not_before_utc": "2026-01-01T00:00:00Z",
        "not_after_utc": "2026-02-01T00:00:00Z",
        "authorization_reference": "synthetic unit fixture; never a campaign authorization",
    }
    for device in spec["devices"].values():
        for k in ["host", "processor", "accelerator", "driver", "cpu_affinity"]:
            device[k] = "synthetic"
        device["ram_bytes"] = device["device_memory_bytes"] = 10000
        device["environment_sha256"] = "c" * 64
    for setup in spec["setups"]:
        setup["configuration"] = {
            "geometry": {"shape": [10, 10]},
            "model": {"name": "test"},
            "solver": {"method": "test"},
            "parameters": [1, 2],
            "seed": 1,
        }
        setup["input_sha256"] = {"synthetic.fits": "d" * 64}
        setup["correctness"] = {
            "reference_sha256": "e" * 64,
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-6,
        }
    assert baseline.validate(spec) == []
    return spec


@pytest.fixture
def evidence(frozen, tmp_path):
    (tmp_path / "catalogue").mkdir()
    for name in ["registry.json", "script_routes.json"]:
        (tmp_path / "catalogue" / name).write_bytes((ROOT / "catalogue" / name).read_bytes())
    cell = next(
        c
        for c in baseline.enumerate_cells(frozen)["cells"]
        if c["measurement"] == "runtime"
        and c["device"] == "cpu"
        and c["capability"] == "unverified"
    )
    record = {
        "schema": "baseline-observation",
        "version": 1,
        "campaign_id": frozen["campaign_id"],
        "spec_sha256": baseline.fingerprint(frozen),
        "cell_id": cell["id"],
        "origin": "fresh-campaign",
        "observed_at": "2026-01-05T00:00:00Z",
        "context": copy.deepcopy(baseline.context(frozen, cell)),
        "resources": {"load_per_allocated_cpu": [0.01] * 10},
        "outcome": "measured",
        "fresh_process": True,
        "synchronized": True,
        "warmup_calls": 3,
        "correctness": {
            "reference_sha256": "e" * 64,
            "max_absolute_error": 1e-8,
            "max_relative_error": 1e-8,
        },
        "samples": {"single_call": [0.1] * 10},
    }
    return frozen, tmp_path, cell, record


def screen(evidence):
    spec, root, cell, record = evidence
    path = root / "results/campaigns" / spec["campaign_id"] / "observation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    receipt = {
        "schema": "baseline-evidence-receipt",
        "version": 1,
        "records": [
            {
                "cell_id": cell["id"],
                "artifact": "observation.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    result = baseline.report(spec, receipt, root)
    row = next(r for r in result["cells"] if r["cell_id"] == cell["id"])
    return row, result, receipt


def test_inventory_is_complete_and_unknowns_do_not_become_ready(draft):
    plan = baseline.enumerate_cells(draft)
    assert len(plan["cells"]) == 29 * 3 * 2 * 6
    assert len({c["id"] for c in plan["cells"]}) == len(plan["cells"])
    assert plan["status"] == "blocked"
    assert plan["counts"] == {"unverified": 620, "unsupported": 366, "not_applicable": 58}
    assert all(
        c["capability"] == "unsupported"
        for c in plan["cells"]
        if c["measurement"].startswith("compile_")
    )
    assert any("software.revisions" in b for b in plan["blockers"])
    assert any("configuration" in b for b in plan["blockers"])
    assert len({c["setup_id"] for c in plan["cells"] if c["runtime_sweep_dispatch"]}) == 22


def test_empty_report_is_missing_never_accepted(draft):
    report = baseline.report(
        draft, {"schema": "baseline-evidence-receipt", "version": 1, "records": []}
    )
    assert report["accepted"] is False
    assert report["status"] == "blocked"
    assert report["counts"] == {"missing": 986, "not_applicable": 58}


@pytest.mark.parametrize(
    "change",
    [
        "partition",
        "missing_setup",
        "duplicate_setup",
        "method_unit",
        "revision",
        "threads_bool",
        "missing_key",
        "reversed_window",
        "null_configuration",
        "nan_load",
        "scope_null",
        "method_null",
        "configuration_bool",
    ],
)
def test_invalid_frozen_spec_fails_closed(frozen, change):
    if change == "partition":
        frozen["devices"]["cpu"]["partition"] = "gpu"
    elif change == "missing_setup":
        frozen["setups"].pop()
    elif change == "duplicate_setup":
        frozen["setups"][-1] = frozen["setups"][0]
    elif change == "method_unit":
        frozen["measurements"][0]["unit"] = "MiB"
    elif change == "revision":
        frozen["software"]["revisions"]["PyAutoLens"] = "main"
    elif change == "threads_bool":
        frozen["devices"]["cpu"]["threads"] = True
    elif change == "missing_key":
        del frozen["protocol"]["sync"]
    elif change == "reversed_window":
        frozen["collection"]["not_after_utc"] = "2025-01-01T00:00:00Z"
    elif change == "null_configuration":
        frozen["setups"][0]["configuration"]["parameters"] = [None]
    elif change == "scope_null":
        frozen["protocol"]["scope"] = None
    elif change == "method_null":
        frozen["measurements"][0]["description"] = None
    elif change == "configuration_bool":
        frozen["setups"][0]["configuration"]["geometry"] = True
    elif change == "nan_load":
        frozen["devices"]["cpu"]["max_load_per_allocated_cpu"] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        baseline.validate(frozen)


def test_inventory_edit_requires_explicit_refreeze(frozen, tmp_path):
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue/registry.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        baseline.validate(frozen, tmp_path)


def test_valid_fixture_is_only_candidate_and_other_cells_stay_missing(evidence):
    row, report, _ = screen(evidence)
    assert row["status"] == "candidate_for_human_review"
    assert report["status"] == "blocked" and report["accepted"] is False
    assert report["counts"]["missing"] == 985
    assert "not independently observed" in report["qualification"]


@pytest.mark.parametrize(
    "change",
    [
        "archived",
        "campaign",
        "hash",
        "time",
        "future",
        "context",
        "context_bool",
        "load",
        "samples",
        "sample_bool",
        "negative",
        "nan",
        "witness",
        "witness_reference",
        "warmup",
        "sync",
        "process",
        "unit",
        "cache",
    ],
)
def test_weak_or_mismatched_observation_rejected(evidence, change):
    record = evidence[3]
    if change == "archived":
        record["origin"] = "archive"
    elif change == "campaign":
        record["campaign_id"] = "another"
    elif change == "hash":
        record["spec_sha256"] = "0" * 64
    elif change == "time":
        record["observed_at"] = "2025-01-01T00:00:00Z"
    elif change == "future":
        record["observed_at"] = "2999-01-01T00:00:00Z"
    elif change == "context":
        record["context"]["precision"] = "mixed"
    elif change == "context_bool":
        record["context"]["protocol"]["batch_size"] = True
    elif change == "load":
        record["resources"]["load_per_allocated_cpu"][0] = 1
    elif change == "samples":
        record["samples"]["single_call"].pop()
    elif change == "sample_bool":
        record["samples"]["single_call"][0] = True
    elif change == "negative":
        record["samples"]["single_call"][0] = -1
    elif change == "nan":
        record["samples"]["single_call"][0] = float("nan")
    elif change == "witness":
        record["correctness"]["max_absolute_error"] = 1
    elif change == "witness_reference":
        record["correctness"]["reference_sha256"] = "f" * 64
    elif change == "warmup":
        record["warmup_calls"] = 0
    elif change == "sync":
        record["synchronized"] = False
    elif change == "process":
        record["fresh_process"] = "true"
    elif change == "unit":
        record["context"]["measurement"]["unit"] = "B"
    elif change == "cache":
        record["context"]["measurement"]["cache_state"] = "cold"
    assert screen(evidence)[0]["status"] == "rejected"


def test_cpu_exclusion_is_fresh_timeout_not_performance(evidence):
    record = evidence[3]
    record["outcome"] = "cpu_timeout"
    record["elapsed_seconds"] = 1800
    record["resources"]["load_per_allocated_cpu"] = [0.01]
    del record["samples"]
    row, report, _ = screen(evidence)
    assert row["status"] == "cpu_exclusion_for_human_review"
    assert report["accepted"] is False
    record["elapsed_seconds"] = 1799
    assert screen(evidence)[0]["status"] == "rejected"


@pytest.mark.parametrize(
    "change", ["duplicate", "unsafe", "absolute", "symlink", "hash", "unknown", "unsupported"]
)
def test_receipt_guards(evidence, change):
    spec, root, cell, _ = evidence
    _, _, receipt = screen(evidence)
    row = receipt["records"][0]
    if change == "duplicate":
        receipt["records"].append(copy.deepcopy(row))
    elif change == "unsafe":
        row["artifact"] = "../observation.json"
    elif change == "absolute":
        row["artifact"] = "/etc/passwd"
    elif change == "symlink":
        path = root / "results/campaigns" / spec["campaign_id"] / "escape.json"
        path.symlink_to(root / "catalogue/registry.json")
        row["artifact"] = "escape.json"
    elif change == "hash":
        row["sha256"] = "0" * 64
    elif change == "unknown":
        row["cell_id"] = "unknown"
    elif change == "unsupported":
        row["cell_id"] = next(
            c["id"]
            for c in baseline.enumerate_cells(spec)["cells"]
            if c["capability"] == "unsupported"
        )
    report = baseline.report(spec, receipt, root)
    assert report["accepted"] is False
    if change == "unknown":
        assert report["receipt_errors"] == ["unknown cell: unknown"]
    else:
        assert report["counts"]["rejected"] == 1


def test_json_duplicates_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
        path.write_text(text)
        with pytest.raises(ValueError):
            baseline.read_json(path)


def test_cli_is_stdlib_and_read_only(tmp_path):
    script = ROOT / "scripts/misc/tooling/baseline_readiness.py"
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "enumerate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "blocked"
    assert not list(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "validate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "report"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "requires --receipt" in result.stderr
