"""``_profile_cli.provenance_dict``: the block every device-recording result JSON carries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("_profile_cli", _ROOT / "_profile_cli.py")
_cli = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_profile_cli", _cli)
_spec.loader.exec_module(_cli)

_LAYOUT = _ROOT / "scripts" / "misc" / "tooling" / "check_results_layout.py"
_lspec = importlib.util.spec_from_file_location("check_results_layout", _LAYOUT)
_layout = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(_layout)


def test_keys_match_the_layout_gate():
    prov = _cli.provenance_dict()
    assert tuple(prov) == _cli.PROVENANCE_KEYS
    assert set(_layout.REQUIRED_PROVENANCE_KEYS) <= set(prov)
    assert _layout.missing_provenance({"provenance": prov}) == []


def test_shape_and_types(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    prov = _cli.provenance_dict()
    assert prov["provenance_schema"] == 1
    assert prov["captured_at"].endswith("Z")
    assert isinstance(prov["host"], str) and prov["host"]
    assert prov["slurm"] == {"job_id": None, "array_job_id": None, "array_task_id": None}
    assert len(prov["loadavg_at_import"]) == 3 and len(prov["loadavg_at_write"]) == 3
    assert prov["loadavg_at_import"] == _cli.LOADAVG_AT_IMPORT
    assert isinstance(prov["profiling_revision"], str)
    assert "autolens_profiling" in prov["library_revisions"]
    for name in ("jax", "numpy", "numba", "nufftax"):
        assert name in prov["dependency_versions"]


def test_slurm_env_is_recorded(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "360001")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "360000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    prov = _cli.provenance_dict()
    assert prov["slurm"] == {"job_id": "360001", "array_job_id": "360000", "array_task_id": "3"}


def test_never_raises_when_helpers_fail(monkeypatch):
    """A provenance failure must never lose a measurement: fields degrade to strings/None."""

    def boom(*_a, **_k):
        raise RuntimeError("no git here")

    monkeypatch.setattr(_cli, "_source_revisions", boom)
    monkeypatch.setattr(_cli.os, "getloadavg", boom)
    prov = _cli.provenance_dict()
    assert prov["library_revisions"].startswith("unavailable")
    assert prov["loadavg_at_write"] is None
    assert tuple(prov) == _cli.PROVENANCE_KEYS
