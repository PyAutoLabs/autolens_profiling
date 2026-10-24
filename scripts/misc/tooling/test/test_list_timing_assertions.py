"""``list_timing_assertions.py``: the timing-noise inventory lister (#362).

Synthetic sources with known answers: which comparisons are timing thresholds,
which are validity checks, which constants are timing-gate settings, and that
``--check`` names a key the note does not mention. No timing is measured.
"""

from __future__ import annotations

import sys as _sys
import textwrap
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_tooling = _profiling_root() / "scripts" / "misc" / "tooling"
if str(_tooling) not in _sys.path:
    _sys.path.insert(0, str(_tooling))

import list_timing_assertions as lta  # noqa: E402

SOURCE = textwrap.dedent(
    """
    MAX_OVERHEAD_MS = 12.0
    N_REPEATS = 5
    PIXEL_SCALE = 0.05
    WARMUP_WINDOW = 3


    def gate(overhead_ms, wall_s, n_pixels, elapsed_s, incl_s):
        ok = overhead_ms <= MAX_OVERHEAD_MS
        fast = wall_s / 2.0 < 0.95
        capped = n_pixels > 10000
        sane = elapsed_s >= 0.0
        identity = abs(incl_s - wall_s) <= 1e-9
        return ok, fast, capped, sane, identity


    speedup = 0.1
    go = speedup >= 0.05
    """
)


def _sites(tmp_path):
    root = tmp_path
    (root / "scripts").mkdir()
    path = root / "scripts" / "cell.py"
    path.write_text(SOURCE)
    return lta.scan_file(path, root)


def test_threshold_validity_and_constant_kinds(tmp_path):
    sites = _sites(tmp_path)
    by_text = {s.text: s for s in sites}

    assert by_text["overhead_ms <= MAX_OVERHEAD_MS"].kind == "threshold"
    assert by_text["overhead_ms <= MAX_OVERHEAD_MS"].key == "scripts/cell.py::gate"
    assert by_text["wall_s / 2.0 < 0.95"].kind == "threshold"
    assert by_text["speedup >= 0.05"].key == "scripts/cell.py::<module>"
    # A sign check and a 1e-9 identity are validity, not noise cutoffs.
    assert by_text["elapsed_s >= 0.0"].kind == "validity"
    assert by_text["abs(incl_s - wall_s) <= 1e-9"].kind == "validity"
    # A size cap names no timing quantity and is not listed.
    assert "n_pixels > 10000" not in by_text

    constants = {s.key for s in sites if s.kind == "constant"}
    assert constants == {
        "scripts/cell.py::MAX_OVERHEAD_MS",
        "scripts/cell.py::N_REPEATS",
        "scripts/cell.py::WARMUP_WINDOW",
    }


def test_check_names_exactly_the_missing_required_keys(tmp_path):
    sites = _sites(tmp_path)
    required = lta.required_keys(sites)
    assert "scripts/cell.py::gate" in required

    note = "\n".join(f"`{key}`" for key in required if key != "scripts/cell.py::N_REPEATS")
    assert lta.missing_from_note(sites, note) == ["scripts/cell.py::N_REPEATS"]
    assert lta.missing_from_note(sites, note + "\n`scripts/cell.py::N_REPEATS`") == []


def test_the_committed_inventory_covers_the_repo():
    """The live witness of the lint step: every key in the repo is in the note."""
    assert lta.NOTE.exists()
    assert lta.missing_from_note(lta.scan(), lta.NOTE.read_text()) == []
