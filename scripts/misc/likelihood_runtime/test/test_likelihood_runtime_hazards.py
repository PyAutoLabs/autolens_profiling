"""Tests for the first hazards-index consumer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def _aggregate_module():
    path = _profiling_root() / "scripts" / "misc" / "likelihood_runtime" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("likelihood_runtime_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pixelization_consumer_reads_only_relevant_stable_ids(tmp_path):
    index = tmp_path / "hazards_index.json"
    index.write_text(
        json.dumps(
            {
                "findings": {
                    "likelihood.relevant": {"subject_name": "imaging_pixelization"},
                    "matrix.relevant": {"subject_name": "curvature_matrix"},
                    "component.unrelated": {"subject_name": "power_law"},
                }
            }
        )
    )
    assert _aggregate_module()._hazard_findings_for_cell(index, "imaging/pixelization/hst") == [
        "likelihood.relevant",
        "matrix.relevant",
    ]


def test_missing_or_invalid_index_soft_fails_with_warning(tmp_path):
    module = _aggregate_module()
    with pytest.warns(UserWarning, match="hazard index unavailable"):
        assert (
            module._hazard_findings_for_cell(tmp_path / "missing.json", "imaging/pixelization")
            == []
        )
