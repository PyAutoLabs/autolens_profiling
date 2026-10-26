"""The 2a peak-memory backfill (autolens_profiling#462): sacct + walls.tsv -> row sidecar."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bf = _load("peak_rss_backfill", ROOT / "scripts/misc/campaigns/peak_rss_backfill.py")
p2a = _load("imaging_p2a", ROOT / "scripts/misc/campaigns/imaging_p2a.py")


def _dump(*, empty=(), swap_wall=None):
    """A synthetic sacct + walls dump covering every batch-1 leg (and batch 0's tasks)."""
    sacct = ["JobID|JobName|State|Elapsed|MaxRSS|AllocCPUS|NodeList"]
    walls = []
    for instrument, array in bf.BATCH1_RAL_ARRAYS.items():
        for i, task in enumerate(p2a.batch1_ral_cpu(instrument)):
            job = f"{array}_{i}"
            rss = "" if job in empty else f"{(i + 1) * 1000000}K"
            sacct += [
                f"{job}|x|COMPLETED|00:01:00||8|n",
                f"{job}.batch|batch|COMPLETED|00:01:00|{rss}|8|n",
            ]
            walls.append(f"WALL\t{900000 + i}\t{i}\t60\t{task.cell}")
    for instrument, job in bf.BATCH1_A100_JOBS.items():
        sacct += [
            f"{job}|x|COMPLETED|01:00:00||8|g",
            f"{job}.batch|batch|COMPLETED|01:00:00|2G|8|g",
        ]
        for i, task in enumerate(p2a.batch1_a100(instrument)):
            cell = swap_wall if (swap_wall and (job, i) == ("412944", 0)) else task.cell
            walls.append(f"WALL\t{job}\t{i}\t60\t{cell}")
    for _cell, (_w, _e, job) in p2a.BATCH0_WALLS.items():
        sacct.append(f"{job}.batch|batch|COMPLETED|00:01:00|512M|8|n")
    return "\n".join(sacct) + "\n@@WALLS\n" + "\n".join(walls) + "\n"


def test_maxrss_units():
    assert bf.maxrss_kib("12727812K") == 12727812
    assert bf.maxrss_kib("25334M") == 25334 * 1024
    assert bf.maxrss_kib("1.5G") == 1.5 * 1024**2
    assert bf.maxrss_kib("") is None
    assert bf.gb(1e6) == 1.024


def test_ral_legs_are_exact_per_task_and_a100_legs_job_level():
    doc = bf.build(_dump(), exists=lambda path: True)
    rows = doc["rows"]
    assert len(rows) == 106 and doc["legs_without_row"] == []
    ral = p2a.batch1_ral_cpu("hst")
    for i, task in enumerate(ral):
        entry = rows[task.output]
        assert entry["sacct_job"] == f"412925_{i}"
        assert entry["peak_rss_basis"] == "sacct_maxrss"
        assert entry["peak_rss_gb"] == bf.gb((i + 1) * 1e6)
        assert entry["cpus_per_eval"] == (1 if task.backend == "numba" else 8)
    for task in p2a.batch1_a100("ao"):
        entry = rows[task.output]
        assert entry["sacct_job"] == "412927"
        assert entry["peak_rss_basis"] == "job_maxrss"
        assert "upper bound" in entry["peak_rss_note"]
        assert entry["devices_per_eval"] == (1 if task.device == "gpu" else 0)


def test_empty_maxrss_is_null_with_reason():
    doc = bf.build(_dump(empty={"412924_4"}), exists=lambda path: True)
    task = p2a.batch1_ral_cpu("euclid")[4]
    entry = doc["rows"][task.output]
    assert entry["peak_rss_gb"] is None and entry["peak_rss_basis"] is None
    assert "no MaxRSS" in entry["peak_rss_reason"]


def test_batch0_is_supporting_only():
    doc = bf.build(_dump(), exists=lambda path: True)
    cell = "imaging/delaunay/ao numba_cpu_sparse_fp64@hpc_ral_cpu [runtime]"
    task = next(t for t in p2a.planned_tasks() if t.cell == cell)
    entry = doc["rows"][task.output]
    assert entry["supporting_batch0"]["sacct_job"] == "412877_3"
    assert entry["peak_rss_gb"] != entry["supporting_batch0"]["peak_rss_gb"]


def test_missing_row_is_listed_not_invented():
    gone = p2a.batch1_a100("ao")[0]
    doc = bf.build(_dump(), exists=lambda path: path != gone.output)
    assert gone.output not in doc["rows"]
    assert doc["legs_without_row"] == [gone.cell]


def test_walls_mismatch_refuses():
    with pytest.raises(ValueError, match="walls.tsv"):
        bf.build(_dump(swap_wall="imaging/mge/hst x@hpc_a100 [runtime]"), exists=lambda p: True)


def test_committed_sidecar_covers_every_committed_2a_row():
    doc = json.loads((ROOT / bf.SIDECAR).read_text())
    tasks = p2a.planned_tasks()
    committed = {t.output for t in tasks if (ROOT / t.output).is_file()}
    # Rows written after #462 (the two ao gpu-dense breakdowns rerun in #466) carry their own
    # ``resources`` block and need no sidecar entry; the sidecar stays batch 1's record.
    own = {o for o in committed if "resources" in json.loads((ROOT / o).read_text())}
    assert set(doc["rows"]) == committed - own
    assert len(committed - own) == 104
    assert {t.cell for t in tasks if t.output not in committed - own} == set(
        doc["legs_without_row"]
    )
    for entry in doc["rows"].values():
        assert entry["cpus_per_eval"] in (1, 8)
        assert (entry["peak_rss_gb"] is None) == (entry["peak_rss_basis"] is None)
