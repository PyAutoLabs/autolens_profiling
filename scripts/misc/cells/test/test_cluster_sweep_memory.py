"""``scripts/cluster/image_plane/_memory.py`` and the sweep's memory JSON fields
(autolens_profiling#436, after RAL job 412567 was OOM-killed at A2744 row 8/25).

The sweep runs at import, so its wiring is checked on the source text: every row
stores ``rss_mb`` / ``memory``, the summary stores ``peak_rss_mb`` / ``memory_trace``,
and the pilot submit exposes ``PILOT_LEGS`` with the resized ``--mem`` / ``--time``.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/cells/test/test_cluster_sweep_memory.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
_DIR = ROOT / "scripts" / "cluster" / "image_plane"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import _memory  # noqa: E402

SWEEP = (_DIR / "robust_settings_sweep.py").read_text()
PILOT = (
    ROOT / "hpc" / "batch_cpu" / "submit_breakdown_cluster_robust_settings_pilot_ral_cpu_fp64"
).read_text()
SNAPSHOT_KEYS = {"rss_mb", "peak_rss_mb", "live_arrays", "live_array_bytes"}


class _Arr:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test__snapshot_keys_and_json_serialisable():
    snap = _memory.snapshot()
    assert set(snap) == SNAPSHOT_KEYS
    assert snap["live_arrays"] is None and snap["live_array_bytes"] is None
    json.dumps(snap)
    if Path("/proc/self/status").exists():
        assert snap["rss_mb"] > 0
        assert snap["peak_rss_mb"] >= snap["rss_mb"] * 0.5


def test__current_rss_reads_vmrss(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t 9 kB\nVmRSS:\t  204800 kB\n")
    assert _memory.current_rss_mb(str(status)) == pytest.approx(200.0)
    assert _memory.current_rss_mb(str(tmp_path / "missing")) is None
    (tmp_path / "norss").write_text("Name:\tpython\n")
    assert _memory.current_rss_mb(str(tmp_path / "norss")) is None


def test__live_arrays_counts_and_bytes():
    fake = SimpleNamespace(live_arrays=lambda: [_Arr(8), _Arr(1024), object()])
    assert _memory.live_arrays(fake) == (3, 1032)
    assert _memory.snapshot(fake)["live_array_bytes"] == 1032


def test__live_arrays_api_missing_is_none():
    assert _memory.live_arrays(SimpleNamespace()) == (None, None)
    assert _memory.live_arrays(None) == (None, None)


def test__describe():
    line = _memory.describe(
        {"rss_mb": 1500.4, "peak_rss_mb": 2048.0, "live_arrays": 3, "live_array_bytes": 2**21}
    )
    assert line == "[rss 1500 MB, peak 2048 MB, live arrays 3 (2.0 MB)]"
    assert "n/a" in _memory.describe(
        {"rss_mb": None, "peak_rss_mb": None, "live_arrays": None, "live_array_bytes": None}
    )


def test__sweep_stores_row_and_summary_memory_fields():
    assert 'row["rss_mb"] = snap["rss_mb"]' in SWEEP
    assert 'row["memory"] = snap' in SWEEP
    assert '"peak_rss_mb": _memory.snapshot()["peak_rss_mb"]' in SWEEP
    assert '"memory_trace": MEMORY_TRACE' in SWEEP
    # every row exit (unusable, invalid, judged) records its snapshot
    assert SWEEP.count("+ row_mem(row, name)") == 3
    for label in ("oracle calibration", "oracle draw", "uncapped counts", "timing done"):
        assert f'mem("{label}' in SWEEP or f'mem(f"{label}' in SWEEP


def test__pilot_submit_legs_and_sizing():
    assert re.search(r"^#SBATCH --mem=256gb$", PILOT, re.M)
    assert re.search(r"^#SBATCH --time=6:00:00$", PILOT, re.M)
    assert 'PILOT_LEGS="${PILOT_LEGS:-simple,a2744}"' in PILOT
    assert "412567" in PILOT
    assert '[[ ",$PILOT_LEGS," == *,simple,* ]]' in PILOT
    assert '[[ ",$PILOT_LEGS," == *,a2744,* ]]' in PILOT
