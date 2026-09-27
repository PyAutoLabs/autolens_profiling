"""``check_results_layout.py``: the notes-only rule and the provenance ratchet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tooling" / "check_results_layout.py"
_spec = importlib.util.spec_from_file_location("check_results_layout", _TOOL)
crl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crl)


def _provenance():
    return {k: None for k in crl.REQUIRED_PROVENANCE_KEYS}


def _tree(tmp_path: Path) -> Path:
    """A minimal consistent tree: one ledger, one allowlisted pack, one compliant JSON."""
    root = tmp_path / "repo"
    (root / "results" / "notes" / "clipper_campaign").mkdir(parents=True)
    (root / "results" / "notes" / "alpha.md").write_text("# alpha\n")
    (root / "results" / "notes" / "clipper_campaign" / "README.md").write_text("# pack\n")
    (root / "results" / "notes" / "clipper_campaign" / "bench.json").write_text("{}")
    (root / "results" / "breakdown" / "imaging").mkdir(parents=True)
    (root / "results" / "breakdown" / "imaging" / "cell_hpc_a100_fp64.json").write_text(
        json.dumps({"device": {"backend": "gpu", "provenance": _provenance()}, "rows": []})
    )
    (root / "results" / "breakdown" / "imaging" / "sidecar_job1.json").write_text(
        json.dumps({"job": 1, "status": "measured"})
    )
    return root


def test_consistent_tree_passes(tmp_path):
    assert crl.check(_tree(tmp_path)) == []


def test_stray_log_and_sidecar_under_notes_fail(tmp_path):
    root = _tree(tmp_path)
    (root / "results" / "notes" / "job_1.out").write_text("log\n")
    (root / "results" / "notes" / "side_job1.json").write_text("{}")
    failures = crl.check(root)
    assert len(failures) == 2
    assert all(f.startswith("(a)") for f in failures)
    assert any("job_1.out" in f for f in failures)
    assert any("side_job1.json" in f for f in failures)


def test_unlisted_folder_under_notes_fails(tmp_path):
    root = _tree(tmp_path)
    (root / "results" / "notes" / "new_pack").mkdir()
    failures = crl.check(root)
    assert failures == ["(a) folder under results/notes/ not allowlisted: results/notes/new_pack/"]


def test_device_json_without_provenance_fails(tmp_path):
    root = _tree(tmp_path)
    bad = root / "results" / "breakdown" / "imaging" / "cell_local_cpu_fp64.json"
    bad.write_text(json.dumps({"device": {"backend": "cpu"}}))
    failures = crl.check(root)
    assert len(failures) == 1
    assert failures[0].startswith("(b) results/breakdown/imaging/cell_local_cpu_fp64.json")
    assert "captured_at" in failures[0]


def test_partial_provenance_names_missing_keys(tmp_path):
    root = _tree(tmp_path)
    prov = _provenance()
    del prov["slurm"]
    del prov["library_revisions"]
    p = root / "results" / "breakdown" / "imaging" / "cell_partial.json"
    p.write_text(json.dumps({"device": {"provenance": prov}}))
    failures = crl.check(root)
    assert len(failures) == 1
    assert "missing slurm, library_revisions" in failures[0]


def test_grandfather_manifest_exempts_and_ratchets(tmp_path):
    root = _tree(tmp_path)
    old = root / "results" / "runtime" / "imaging" / "old_hpc_a100_fp64.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"device": {"backend": "gpu"}}))
    assert len(crl.check(root)) == 1
    n = crl.write_grandfather(root)
    assert n == 1
    manifest = (root / crl.MANIFEST).read_text()
    assert manifest.startswith("#")
    assert "results/runtime/imaging/old_hpc_a100_fp64.json" in manifest
    assert crl.check(root) == []
    # A dead manifest line is a failure: the ratchet only tightens.
    old.unlink()
    failures = crl.check(root)
    assert len(failures) == 1
    assert failures[0].startswith("(c) dead line")


def test_json_in_allowlisted_pack_and_without_device_are_ignored(tmp_path):
    root = _tree(tmp_path)
    (root / "results" / "breakdown" / "imaging" / "comparison.json").write_text(
        json.dumps({"cells": []})
    )
    (root / "results" / "breakdown" / "imaging" / "broken.json").write_text("{not json")
    assert crl.check(root) == []


@pytest.mark.parametrize("flag", ["--check", ""])
def test_cli_reports_and_exit_codes(tmp_path, capsys, flag):
    root = _tree(tmp_path)
    (root / "results" / "notes" / "job_2.out").write_text("log\n")
    argv = ["--root", str(root)] + ([flag] if flag else [])
    code = crl.main(argv)
    out = capsys.readouterr().out
    assert "1 failure(s)" in out
    assert code == (1 if flag else 0)


def test_cli_ok_line(tmp_path, capsys):
    root = _tree(tmp_path)
    assert crl.main(["--root", str(root)]) == 0
    assert "check_results_layout: OK" in capsys.readouterr().out


def test_real_tree_is_consistent():
    """The committed tree passes: the gate is green at the policy's birth."""
    root = _TOOL.parents[3]
    assert crl.check(root) == []
