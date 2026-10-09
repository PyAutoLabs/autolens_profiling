"""Timing-noise audit fix phase 6 (#362): the witness band (P5) and the dashboard's warm-up rule (P3).

The warm-up witnesses that run the cell's own ``_warm_to_steady_state`` (the T3 helper) and carry
an unsettled record into the overhead verdict (P1) and the promotion decision (P2) live beside the
T3 tests in ``test_fixed_light_numba.py``. This file pins the two remaining consumers:

- ``_production_config.witness_verdict`` (P5) judges a cold-eval median against the production
  range only on a reference host class (``build_dashboard.is_reference_host_class``). Off it, the
  verdict is ``INCONCLUSIVE`` with the reason "off reference host class (<class>)". The band
  (decision 2 on #235) is unchanged.
- ``build_dashboard.qualify`` (P6) does not qualify a point whose payload records a warm-up that
  never settled.

Everything here is deterministic: no timing, no clock, no compute.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MISC = Path(__file__).resolve().parents[1]
_ROOT = _MISC.parents[1]
sys.path.insert(0, str(_MISC))
sys.path.insert(0, str(_ROOT))

from likelihood_breakdown.warmup_gate import WARMUP_UNSETTLED  # noqa: E402

import _production_config as pc  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_dashboard", _MISC / "tooling" / "build_dashboard.py")

#: The four cells that call the witness, each passing its ``--config-name`` as host class.
WITNESS_CELLS = [
    _ROOT / "scripts/imaging" / family / f"likelihood_{kind}_numba.py"
    for family in ("rectangular", "delaunay")
    for kind in ("runtime", "breakdown")
]


# ---------------------------------------------------------------------------
# P5: the production-representative witness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument, median_s, expected",
    [
        # HST band [0.52 / 1.5, 0.81 x 1.5] = [0.347, 1.215] s; Euclid [0.46, 1.695] s.
        ("hst", 0.80, "PASS"),
        ("hst", 1.463, "FAIL"),  # the committed laptop Delaunay HST median, above the band
        ("hst", 0.30, "FAIL"),
        ("euclid", 0.70, "PASS"),
        ("euclid", 0.2346, "FAIL"),  # the committed laptop Delaunay Euclid median, below
    ],
)
def test_a_reference_host_class_row_keeps_its_pass_or_fail(instrument, median_s, expected):
    result = pc.witness_verdict(median_s, instrument, "hpc_ral_cpu_fp64")
    assert result["verdict"] == expected
    assert result["in_band"] is (expected == "PASS")
    assert result["host_class"] == "hpc_ral_cpu_fp64"


@pytest.mark.parametrize("host_class", ["local_cpu_fp64", "local_cpu_mp", "local_gpu_fp64", None])
@pytest.mark.parametrize("median_s", [0.80, 1.463])
def test_a_laptop_row_is_inconclusive_whatever_the_band_says(host_class, median_s):
    result = pc.witness_verdict(median_s, "hst", host_class)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == f"off reference host class ({host_class or 'untagged'})"
    # Where the median fell is still recorded for the reader.
    assert result["in_band"] is (median_s == 0.80)


def test_the_band_is_unchanged_and_the_host_rule_is_the_dashboards():
    assert pc.WITNESS_FACTOR == 1.5
    assert pc.witness_verdict(0.8, "hst", "hpc_ral_cpu_fp64")["allowed_s"] == pytest.approx(
        [0.52 / 1.5, 0.81 * 1.5]
    )
    for config in ("hpc_ral_cpu_fp64", "hpc_a100_mp", "local_cpu_fp64", "local_gpu_mp"):
        assert pc._is_reference_host_class(config) is bd.is_reference_host_class(config)
    assert pc.witness_verdict(0.8, "nirspec", "hpc_ral_cpu_fp64")["verdict"] == "no_reference"


@pytest.mark.parametrize("cell", WITNESS_CELLS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_witness_cell_passes_its_host_class(cell):
    source = cell.read_text()
    assert 'witness_verdict(timing["cold_eval_median_s"], instrument, _cli.config_name)' in source
    assert '"INCONCLUSIVE"' in source and "witness['reason']" in source


def test_the_committed_witness_rows_rejudge_as_inconclusive():
    """Facts (fix phase 6): all 8 committed rows ran untagged on the laptop (2026-09-08).

    4 PASS and 4 FAIL as published; all 8 are INCONCLUSIVE under the host-class rule, and
    their in-band reading is the published one.
    """
    rows = sorted(
        p
        for p in (_ROOT / "results").rglob("*_numba_*_v2026.8.17.1*.json")
        if "witness" in json.loads(p.read_text())
    )
    assert len(rows) == 8
    published = []
    for path in rows:
        payload = json.loads(path.read_text())
        old = payload["witness"]
        published.append(old["verdict"])
        new = pc.witness_verdict(old["measured_cold_eval_median_s"], old["instrument"], None)
        assert new["verdict"] == "INCONCLUSIVE"
        assert new["in_band"] is (old["verdict"] == "PASS")
    assert sorted(published) == ["FAIL"] * 4 + ["PASS"] * 4


# ---------------------------------------------------------------------------
# P3 at the dashboard: an unsettled warm-up never qualifies
# ---------------------------------------------------------------------------

_CONF = {"node": "euclid-ral-gpu-2", "loadavg_cap": 8.0}
_UNSETTLED = {
    "sequence_s": [1.0 / (1.0 + 0.5 * i) for i in range(12)],
    "n_calls": 12,
    "steady": False,
    "window": 3,
    "tolerance": 0.10,
    "max_calls": 12,
}


def _payload(warmup):
    payload = {
        "full_pipeline_single_jit": 0.1,
        "device": {
            "backend": "cpu",
            "provenance": {
                "host": "euclid-ral-gpu-2",
                "slurm": {"job_id": "1"},
                "loadavg_at_import": [0.5, 0.5, 0.5],
            },
        },
    }
    if warmup is not None:
        payload["warmup"] = warmup
    return payload


@pytest.mark.parametrize(
    "warmup, qualified",
    [
        (None, True),
        (12.3, True),  # a scalar warm-up time (quick_update style) is not a record
        (dict(_UNSETTLED, steady=True), True),
        (_UNSETTLED, False),
        ({}, False),  # a record that cannot show it settled has not shown it
    ],
)
def test_a_point_whose_warmup_never_settled_is_unqualified(warmup, qualified):
    point = bd._point(_payload(warmup), "1", "results/x.json")
    ok, reason, refused = bd.qualify(point, "hpc_ral_cpu_fp64", _CONF)
    assert ok is qualified and refused is False
    if not qualified:
        assert reason.startswith(WARMUP_UNSETTLED)
    else:
        assert "warmup_unsettled" not in point


def test_a_per_row_warmup_record_is_read_too():
    """``fixed_light_numba.py`` writes its records under ``rows[*].warmup``."""
    payload = _payload(None)
    payload["rows"] = {"b": {"warmup": dict(_UNSETTLED, steady=True)}, "d": {"warmup": _UNSETTLED}}
    assert bd.warmup_unsettled(payload).startswith(WARMUP_UNSETTLED)
    payload["rows"]["d"]["warmup"] = dict(_UNSETTLED, steady=True)
    assert bd.warmup_unsettled(payload) is None
