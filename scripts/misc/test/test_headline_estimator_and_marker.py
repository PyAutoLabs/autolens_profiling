"""Timing-noise audit fix phase 5 (#362): the headline estimator (P8) and the GPU-only marker (P9).

Deterministic witnesses, no timing:

- P8: an injected-clock post-compile transient moves the legacy 10-call block mean 2.4x while the
  steady median of the same compiled callable does not; the dashboard headlines the median where a
  row records it, labels it, and drift never compares a median with a block mean.
- P9: a ``--per-run-timeout`` marker records INCONCLUSIVE with host, loads and timeout; it renders
  GPU-only only when it qualifies (reference host class, host and load recorded, load under the
  cap, the pinned node); ``--skip-existing`` re-measures an unqualified marker; every committed
  marker renders inconclusive.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_MISC = Path(__file__).resolve().parents[1]
_ROOT = _MISC.parents[1]
sys.path.insert(0, str(_MISC))

from likelihood_breakdown import timing  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_dashboard", _MISC / "tooling" / "build_dashboard.py")
adapters = _load("catalogue_adapters", _MISC / "tooling" / "catalogue_adapters.py")

_CONF = {"node": "euclid-ral-gpu-2", "loadavg_cap": 8.0}

# The A100 source-plane numbers of #371 scaled x10 so a 2.4x move also clears the 1 ms drift floor.
STEADY_S = 0.00267
TRANSIENT_EXCESS_S = 0.0375  # one slow first-block call: block mean 6.42 ms = 2.4x steady


class _Counted:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.calls


class _Clock:
    """Call i of the timed loop lasts ``durations[i]`` seconds (two reads per timed call)."""

    def __init__(self, durations):
        self.durations = list(durations)
        self.now = 0.0
        self.reads = 0

    def __call__(self):
        if self.reads % 2 == 1:
            self.now += self.durations[self.reads // 2]
        self.reads += 1
        return self.now


def _release_row(version: str, block: list[float], timed: list[float]) -> dict:
    """A runtime row as a cell writes it: the legacy block mean plus the helper's median fields."""
    block_mean = sum(block) / len(block)
    fields = timing.headline_steady_median(
        _Counted(), block_mean_s=block_mean, n_timed=len(timed), clock=_Clock(timed)
    )
    return {
        "autolens_version": version,
        "device": {"backend": "gpu"},
        "full_pipeline_single_jit": block_mean,
        **fields,
    }


def _qualified_series(points: list[dict]) -> dict:
    pts = []
    for p in points:
        p = dict(p, host="euclid-ral-gpu-2", loadavg=0.5, has_provenance=True)
        ok, reason, _ = bd.qualify(p, "hpc_a100_fp64", _CONF)
        pts.append(dict(p, qualified=ok, reason=reason))
    return {"key": "runtime:c:hpc_a100_fp64", "points": pts, "drift": bd.drift(pts)}


# --- P8: headline estimator -------------------------------------------------------------------


def test_injected_transient_moves_the_block_mean_2p4x_and_not_the_median():
    clean_block = [STEADY_S] * 10
    transient_block = [STEADY_S + TRANSIENT_EXCESS_S] + [STEADY_S] * 9
    # The timed calls after >= 5 warm calls: steady, with one late spike the median ignores.
    timed = [STEADY_S] * 39 + [STEADY_S * 10]

    before = _release_row("2026.10.1.1", clean_block, timed)
    after = _release_row("2026.10.2.1", transient_block, timed)

    ratio_block = after["full_pipeline_single_jit"] / before["full_pipeline_single_jit"]
    ratio_median = (
        after["full_pipeline_single_jit_median"] / before["full_pipeline_single_jit_median"]
    )
    assert ratio_block == pytest.approx(2.4045, abs=1e-3)
    assert ratio_median == pytest.approx(1.0)
    assert after["full_pipeline_single_jit_median_ms"] == pytest.approx(STEADY_S * 1000)

    # Legacy rows (no median recorded): the 2.4x transient alone flags the series as drifted.
    legacy = [
        bd._point(
            {k: v for k, v in r.items() if not k.startswith("full_pipeline_single_jit_")}, v, "x"
        )
        for r, v in ((before, "2026.10.1.1"), (after, "2026.10.2.1"))
    ]
    assert bd.drift(legacy)["status"] == "drifted"

    # Rows that record the median: the headline is the median, labelled, and it does not drift.
    points = [bd._point(r, r["autolens_version"], "x") for r in (before, after)]
    assert [p["headline_estimator"] for p in points] == [bd.ESTIMATOR_MEDIAN] * 2
    assert points[1]["single_jit_s"] == pytest.approx(STEADY_S)
    assert points[1]["single_jit_block_mean_s"] == pytest.approx(STEADY_S * 2.4045, rel=1e-3)
    d = bd.drift(points)
    assert d["status"] == "steady" and d["ratio"] == pytest.approx(1.0)
    # Still one summary per release: inside the band is insufficient, not flat (P7 unchanged).
    c = bd._summary_comparison(_qualified_series(points))
    assert c["status"] == "insufficient" and c["reasons"] == [bd.SINGLE_SAMPLE_NULL_REASON]


def test_helper_adds_no_timer_section_and_writes_the_known_keys():
    fields = timing.headline_steady_median(
        _Counted(), block_mean_s=0.1, n_timed=7, clock=_Clock([1.0] * 7)
    )
    assert set(fields) == {
        "full_pipeline_single_jit_median",
        "full_pipeline_single_jit_median_ms",
        "full_pipeline_single_jit_p10_ms",
        "full_pipeline_single_jit_p90_ms",
        "full_pipeline_single_jit_median_protocol",
    }
    assert "full_pipeline_single_jit" not in fields  # the legacy key is never written here
    proto = fields["full_pipeline_single_jit_median_protocol"]
    assert proto["n_warm"] == timing.MIN_STEADY_WARM and proto["n_timed"] == 7
    assert proto["full_pipeline_single_jit_is"] == timing.SINGLE_JIT_BLOCK_MEAN_IS


@pytest.mark.parametrize(
    "block_mean, n",
    [(0.000267, 200), (0.15, 200), (0.5, 60), (4.0, 20), (60.0, 20), (None, 200), (0.0, 200)],
)
def test_median_timed_calls_are_sized_from_the_block_mean(block_mean, n):
    assert timing.headline_median_n_timed(block_mean) == n


def test_every_runtime_cell_that_headlines_the_block_mean_writes_the_median():
    cells = [
        "imaging/delaunay/likelihood_runtime.py",
        "imaging/delaunay_nn/likelihood_runtime.py",
        "imaging/mge/likelihood_runtime.py",
        "imaging/rectangular/likelihood_runtime.py",
        "interferometer/delaunay/likelihood_runtime.py",
        "interferometer/mge/likelihood_runtime.py",
        "interferometer/rectangular/likelihood_runtime.py",
        "point_source_image/image_plane/likelihood_runtime.py",
        "point_source_image/image_plane/likelihood_runtime_solved.py",
        "point_source_source/source_plane/likelihood_runtime.py",
        "point_source_source/source_plane/likelihood_runtime_solved.py",
    ]
    for rel in cells:
        src = (_ROOT / "scripts" / rel).read_text(encoding="utf-8")
        assert "from likelihood_breakdown.timing import headline_steady_median" in src, rel
        n_sites = src.count('"full_pipeline_single_jit": full_pipeline_per_call,')
        assert n_sites >= 1 and src.count("**full_pipeline_median,") == n_sites, rel


def test_median_reaches_the_v2_catalogue_on_its_existing_metric():
    """No v2 vocabulary change: the adapter's pre-existing ``single_jit_median`` metric picks it up."""
    row = {"full_pipeline_single_jit": 0.00642, "full_pipeline_single_jit_median": 0.00267}
    got = {m[1]: m for m in adapters.metrics(row, "profile", "")}
    assert got["single_jit_block"][3] == 0.00642 and got["single_jit_block"][5] is None
    assert got["single_jit_median"][3] == 0.00267 and got["single_jit_median"][5] == "median"


def test_drift_never_compares_a_median_with_a_block_mean():
    legacy = {"version": "1", "single_jit_s": 0.010}
    median = {"version": "2", "single_jit_s": 0.030, "headline_estimator": bd.ESTIMATOR_MEDIAN}
    d = bd.drift([legacy, median])
    assert d == {"status": "estimator-mismatch", "ratio": None, "from": "1", "to": "2"}
    c = bd._summary_comparison(_qualified_series([dict(legacy), dict(median)]))
    assert c["status"] == "insufficient" and c["ratio"] is None
    assert c["reasons"] == [bd.ESTIMATOR_MISMATCH_REASON]
    # Two medians compare as usual.
    both = [dict(median, version="1", single_jit_s=0.010), median]
    assert bd.drift(both)["status"] == "drifted"
    assert set(bd._COMPARISON_STATUS.values()) == {"drifted", "improved", "flat", "insufficient"}


# --- P9: the GPU-only marker -------------------------------------------------------------------


def _marker(**kw):
    m = {
        "cpu_unusable": True,
        "verdict": "INCONCLUSIVE",
        "outcome": "timeout",
        "timeout_seconds": 3600.0,
        "host": "euclid-ral-gpu-2",
        "loadavg_at_start": 0.5,
        "loadavg_at_timeout": 0.7,
    }
    m.update(kw)
    return m


def test_a_qualified_marker_renders_gpu_only():
    v = bd.marker_verdict(_marker(), "hpc_ral_cpu_fp64", _CONF)
    assert v == {"gpu_only": True, "label": bd.MARKER_GPU_ONLY, "reason": None}


@pytest.mark.parametrize(
    "marker, config, reason",
    [
        (_marker(loadavg_at_timeout=12.0), "hpc_ral_cpu_fp64", "loadavg 12.0 above the cap 8"),
        (
            _marker(loadavg_at_start=9.5, loadavg_at_timeout=0.4),
            "hpc_ral_cpu_fp64",
            "above the cap",
        ),
        (
            _marker(loadavg_at_start=None, loadavg_at_timeout=None),
            "hpc_ral_cpu_fp64",
            "provenance carries no load average",
        ),
        (_marker(host=None), "hpc_ral_cpu_fp64", "provenance carries no host"),
        (_marker(host="laptop"), "local_cpu_fp64", "not a reference host class"),
        (_marker(host="euclid-ral-gpu-1"), "hpc_ral_cpu_fp64", "off the reference host"),
        (
            {"cpu_unusable": True, "reason": "wall-clock timeout after 3600s — GPU-only"},
            "local_cpu_fp64",
            "written before fix phase 5",
        ),
    ],
)
def test_an_unqualified_marker_is_never_rendered_gpu_only(marker, config, reason):
    v = bd.marker_verdict(marker, config, _CONF)
    assert v["gpu_only"] is False and v["label"] == bd.MARKER_TIMED_OUT
    assert reason in v["reason"]


def test_a_non_timeout_marker_is_labelled_did_not_finish():
    oom = {
        "cpu_unusable": True,
        "reason": "OOM-killed at 2858s (fp64 peak exceeds 15GB laptop RAM)",
    }
    v = bd.marker_verdict(oom, "local_cpu_fp64", _CONF)
    assert v["gpu_only"] is False and v["label"] == bd.MARKER_NOT_FINISHED


def test_every_committed_marker_renders_inconclusive():
    markers = sorted((_ROOT / "results").rglob("*.unusable.json"))
    assert len(markers) == 4
    labels = {}
    for path in markers:
        m = json.loads(path.read_text(encoding="utf-8"))
        v = bd.marker_verdict(m, m["config_name"], bd.read_release_sweep_conf(_ROOT))
        assert v["gpu_only"] is False
        labels[path.relative_to(_ROOT / "results" / "runtime").as_posix()] = v["label"]
    assert labels == {
        "datacube/delaunay/alma/delaunay_local_cpu_fp64.unusable.json": bd.MARKER_TIMED_OUT,
        "interferometer/delaunay/alma/delaunay_local_cpu_fp64.unusable.json": bd.MARKER_NOT_FINISHED,
        "interferometer/delaunay/alma_high/delaunay_local_cpu_fp64.unusable.json": bd.MARKER_TIMED_OUT,
        "interferometer/delaunay/alma_high/delaunay_local_cpu_mp.unusable.json": bd.MARKER_NOT_FINISHED,
    }


def _sweep():
    return _load("p9_sweep", _MISC / "likelihood_runtime" / "sweep.py")


def test_a_timeout_writes_an_inconclusive_marker_with_host_and_load(monkeypatch, tmp_path):
    sweep = _sweep()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(sweep.subprocess, "run", timeout)
    monkeypatch.setattr(sweep.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(sweep, "_loadavg_1min", lambda: 3.25)
    script = sweep.runtime_path("imaging", "mge", sweep._REPO_ROOT)
    ok, _, _ = sweep._run_one(
        "python", script, sweep.CONFIGS[0], tmp_path, False, per_run_timeout=1
    )
    assert ok
    marker = json.loads((tmp_path / "mge_local_cpu_fp64.unusable.json").read_text())
    assert marker["verdict"] == "INCONCLUSIVE" and marker["outcome"] == "timeout"
    assert marker["host"] == "laptop" and marker["timeout_seconds"] == 1
    assert marker["loadavg_at_start"] == marker["loadavg_at_timeout"] == 3.25
    assert "GPU-only" not in marker["reason"] and "inconclusive" in marker["reason"]
    v = bd.marker_verdict(marker, marker["config_name"], _CONF)
    assert v["gpu_only"] is False and v["label"] == bd.MARKER_TIMED_OUT


def _skip_existing(monkeypatch, tmp_path, verdict=None):
    sweep = _sweep()
    runs = []

    def fake_run(cmd, **kwargs):
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    if verdict is not None:
        monkeypatch.setattr(sweep, "existing_marker_verdict", lambda *a: verdict)
    argv = ["sweep", "--only", "imaging/mge/hst", "--skip-gpu", "--skip-mp", "--skip-existing"]
    monkeypatch.setattr(sys, "argv", [*argv, "--output-root", str(tmp_path)])
    out_dir = tmp_path / "imaging" / "mge" / "hst"
    out_dir.mkdir(parents=True)
    return sweep, runs, out_dir


def test_skip_existing_re_measures_an_unqualified_marker(monkeypatch, tmp_path):
    sweep, runs, out_dir = _skip_existing(monkeypatch, tmp_path)
    legacy = {"cpu_unusable": True, "reason": "wall-clock timeout after 3600s — GPU-only"}
    (out_dir / "mge_local_cpu_fp64.unusable.json").write_text(json.dumps(legacy))
    assert sweep.main() == 0
    assert len(runs) == 1  # the marker did not stop the run


def test_skip_existing_honours_only_a_qualified_marker_or_a_result(monkeypatch, tmp_path):
    gpu_only = {"gpu_only": True, "label": bd.MARKER_GPU_ONLY, "reason": None}
    sweep, runs, out_dir = _skip_existing(monkeypatch, tmp_path, verdict=gpu_only)
    (out_dir / "mge_local_cpu_fp64.unusable.json").write_text(json.dumps(_marker()))
    assert sweep.main() == 0 and runs == []

    sweep, runs, out_dir = _skip_existing(monkeypatch, tmp_path / "b")
    (out_dir / "mge_local_cpu_fp64.json").write_text("{}")
    assert sweep.main() == 0 and runs == []
