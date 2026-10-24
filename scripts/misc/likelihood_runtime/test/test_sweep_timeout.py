"""A sweep timeout during the fit.png phase must not mark a completed row unusable."""

from __future__ import annotations

import json
import time

from likelihood_runtime.sweep import row_completed_since


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
