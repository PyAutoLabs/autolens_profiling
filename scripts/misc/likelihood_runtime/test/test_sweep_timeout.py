"""A sweep timeout during the fit.png phase must not mark a completed row unusable."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

# Loaded by file path under its own name: ``likelihood_runtime`` also names the per-cell scripts
# (``scripts/<type>/<variant>/likelihood_runtime.py``), which other tests put on sys.path, and
# importing one of those runs a profile.
_spec = importlib.util.spec_from_file_location(
    "likelihood_runtime_sweep_under_test", Path(__file__).resolve().parents[1] / "sweep.py"
)
_sweep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _sweep  # dataclasses resolve their module by name
_spec.loader.exec_module(_sweep)
row_completed_since = _sweep.row_completed_since


def test_row_finished_this_run_with_its_pinned_check_is_complete(tmp_path):
    t0 = time.time() - 1
    row = tmp_path / "runtime.json"
    row.write_text(json.dumps({"full_pipeline_single_jit": 0.1, "pinned_drift": []}))
    assert row_completed_since(row, t0)


def test_early_row_missing_row_or_older_row_is_not_complete(tmp_path):
    row = tmp_path / "runtime.json"
    assert not row_completed_since(row, time.time())
    row.write_text(json.dumps({"vmap": "PENDING", "full_pipeline_single_jit": 0.1}))
    assert not row_completed_since(row, time.time() - 1)  # early write: no pinned check yet
    row.write_text(json.dumps({"pinned_drift": []}))
    assert not row_completed_since(row, time.time() + 60)  # written before this run started
    row.write_text("{not json")
    assert not row_completed_since(row, time.time() - 1)
