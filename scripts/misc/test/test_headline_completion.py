"""Timing-noise audit fix phase 10 (#362): headline completion.

Deterministic witnesses, no timing and no compute:

- the README runtime table reads the dashboard's estimator rule: the steady median where a row
  records one (labelled), the median of independent runs where a row carries them, else the
  legacy value; a table with no such row renders exactly as before;
- the interferometer breakdown cells, the datacube cell and the two ``mge_mass`` cells take the
  headline steady median; the datacube's lands beside ``full_pipeline_cube_single_jit`` and the
  dashboard headlines it; the ``mge_mass`` cells keep their old ``full_pipeline_single_jit``
  (already a median) and say what it is;
- ``wall/rates.py`` bounds the median's extra calls at 50 s per run and ``check_submits.py``
  adds that bound per invocation, so an estimate can only grow;
- ``single_jit_repeats``: a release backed by >= 2 independent runs gets a repeat summary, and a
  comparison inside the 2x band between two such endpoints is ``flat``, while the same numbers
  from single runs stay ``insufficient``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MISC = Path(__file__).resolve().parents[1]
_ROOT = _MISC.parents[1]
for _p in (_MISC, _MISC / "tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from likelihood_breakdown import timing  # noqa: E402
from wall import check_submits, rates  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_dashboard", _MISC / "tooling" / "build_dashboard.py")
readme = _load("build_readme", _MISC / "tooling" / "build_readme.py")
adapters = _load("catalogue_adapters", _MISC / "tooling" / "catalogue_adapters.py")
aggregate = _load("aggregate_p10", _MISC / "likelihood_runtime" / "aggregate.py")

_CONF = {"node": "euclid-ral-gpu-2", "loadavg_cap": 8.0}
_HOST = "euclid-ral-gpu-2"


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


def _run(single, job, *, version="2026.10.1.1", host=_HOST, load=0.5, median_ms=None, **extra):
    """One run's payload as a sweep row writes it (an A100 row with provenance)."""
    payload = {
        "autolens_version": version,
        "full_pipeline_single_jit": single,
        "device": {
            "backend": "gpu",
            "provenance": {
                "host": host,
                "slurm": {"job_id": job},
                "loadavg_at_import": [load, load, load],
            },
        },
    }
    if median_ms is not None:
        payload["full_pipeline_single_jit_median_ms"] = median_ms
    payload.update(extra)
    return payload


def _with_repeats(runs: list[dict]) -> dict:
    """The row (run 1) with its repeat runs listed, as ``aggregate._attach_repeat_runs`` does."""
    listed = [dict(r, source=f"row.repeat{i + 1}.json") for i, r in enumerate(runs)]
    return dict(runs[0], **{bd.REPEAT_RUNS_FIELD: listed})


def _series(points: list[dict], config: str = "hpc_a100_fp64") -> dict:
    pts = []
    for p in points:
        ok, reason, _ = bd.qualify(p, config, _CONF)
        pts.append(dict(p, qualified=ok, reason=reason))
    return {"key": f"runtime:c:{config}", "points": pts, "drift": bd.drift(pts)}


# --- the producer: prefix / what the legacy key is --------------------------------------------


def test_the_helper_writes_beside_any_legacy_key_and_says_what_it_is():
    fields = timing.headline_steady_median(
        _Counted(),
        block_mean_s=0.5,
        n_timed=3,
        clock=_Clock([0.4] * 3),
        prefix="full_pipeline_cube_single_jit",
        block_mean_is=timing.single_jit_block_is(3),
    )
    assert set(fields) == {
        "full_pipeline_cube_single_jit_median",
        "full_pipeline_cube_single_jit_median_ms",
        "full_pipeline_cube_single_jit_p10_ms",
        "full_pipeline_cube_single_jit_p90_ms",
        "full_pipeline_cube_single_jit_median_protocol",
    }
    proto = fields["full_pipeline_cube_single_jit_median_protocol"]
    assert proto["full_pipeline_cube_single_jit_is"] == (
        "mean of one 3-call block after one first call (first block after compile)"
    )
    assert timing.single_jit_block_is(10) == timing.SINGLE_JIT_BLOCK_MEAN_IS
    # The >2 s rule holds for every opted-in cell: nothing is run above it.
    fn = _Counted()
    assert timing.headline_steady_median(fn, block_mean_s=2.5, prefix="x") == {}
    assert fn.calls == 0


def test_the_cube_median_reaches_v2_on_an_additive_metric():
    row = {"full_pipeline_cube_single_jit": 0.5, "full_pipeline_cube_single_jit_median": 0.4}
    got = {m[1]: m for m in adapters.metrics(row, "profile", "")}
    assert got["cube_single_jit"][3] == 0.5 and got["cube_single_jit"][5] is None
    assert got["cube_single_jit_median"][3] == 0.4 and got["cube_single_jit_median"][5] == "median"


# --- the cells' wiring -------------------------------------------------------------------------


def _src(rel: str) -> str:
    return (_ROOT / "scripts" / rel).read_text(encoding="utf-8")


def test_the_breakdown_cells_take_the_median_beside_their_block():
    mge = _src("interferometer/mge/likelihood_breakdown.py")
    assert "timing.headline_steady_median(" in mge
    assert "block_mean_is=timing.single_jit_block_is(N_REPEATS)" in mge
    assert mge.count("**full_pipeline_median,") == 1  # the top-level headline only
    pix = _src("misc/likelihood_breakdown/interferometer_pixelized.py")
    assert 'state["full_pipeline_median"] = timing.headline_steady_median(' in pix
    assert "block_mean_is=timing.single_jit_block_is(n_repeats)" in pix
    assert '**state["full_pipeline_median"],' in pix


def test_the_datacube_cell_takes_the_median_beside_its_cube_block():
    src = _src("datacube/delaunay/likelihood_runtime.py")
    assert 'prefix="full_pipeline_cube_single_jit"' in src
    assert "block_mean_is=single_jit_block_is(_full_cube_n_repeats)" in src
    assert src.count("**full_pipeline_median,") == 1
    assert "full_pipeline_median = {}" in src  # the CUBE_FULL_JIT-off branch writes none


@pytest.mark.parametrize(
    "rel, legacy_line",
    [
        (
            "imaging/mge_mass/likelihood_runtime_jax.py",
            '"full_pipeline_single_jit": on["steady_median_s"],',
        ),
        (
            "imaging/rectangular/likelihood_runtime_numba_mge_mass.py",
            '"full_pipeline_single_jit": on_median_s,',
        ),
    ],
)
def test_the_mge_mass_cells_keep_their_old_key_and_add_the_shared_median(rel, legacy_line):
    src = _src(rel)
    # Continuity: the old key keeps the value it has always had (a median, not a block mean).
    assert legacy_line in src
    assert src.count("**full_pipeline_median,") == 1
    assert "from likelihood_breakdown.timing import headline_steady_median" in src
    assert "block_mean_is=SINGLE_JIT_KEY_IS" in src
    assert (
        "SINGLE_JIT_KEY_IS = (" in src and "median" in src.split("SINGLE_JIT_KEY_IS = (")[1][:200]
    )


def test_every_opted_in_script_is_in_the_wall_basis_set():
    sites = {
        f"scripts/{p.relative_to(_ROOT / 'scripts').as_posix()}"
        for p in (_ROOT / "scripts").rglob("*.py")
        if "misc/test" not in p.as_posix()
        and "headline_steady_median(" in p.read_text(encoding="utf-8")
        and "def headline_steady_median" not in p.read_text(encoding="utf-8")
    }
    # The pixelized breakdown harness runs inside the interferometer delaunay / rectangular cells.
    sites.discard("scripts/misc/likelihood_breakdown/interferometer_pixelized.py")
    sites |= {
        "scripts/interferometer/delaunay/likelihood_breakdown.py",
        "scripts/interferometer/rectangular/likelihood_breakdown.py",
    }
    assert sites == set(rates.HEADLINE_MEDIAN_SCRIPTS)


# --- the README runtime table ------------------------------------------------------------------


def test_the_readme_headline_follows_the_dashboard_estimator_rule():
    legacy = _run(0.00642, "1")
    assert readme._config_headline(legacy) == (0.00642, "")
    median = _run(0.00642, "1", median_ms=0.267)
    seconds, label = readme._config_headline(median)
    assert seconds == pytest.approx(0.000267) and label == "median"
    # The aggregator's alias of the block mean is still the single-jit headline it aliases.
    aliased = dict(median, full_pipeline_per_call=0.00642)
    assert readme._config_headline(aliased)[0] == pytest.approx(0.000267)
    # The same rule as the dashboard's point.
    point = bd._point(median, "2026.10.1.1", "x")
    assert point["single_jit_s"] == pytest.approx(seconds)
    assert point["headline_estimator"] == bd.ESTIMATOR_MEDIAN
    # A median not beside the headline it summarises is not the headline.
    other = {"full_pipeline_per_call": 0.5, "full_pipeline_single_jit_median_ms": 1.0}
    assert readme._config_headline(other) == (0.5, "")
    # A row with no runtime headline is still "—", whatever else it carries.
    assert readme._config_headline({"total_step_by_step": 0.1}) == (None, "")


def test_the_readme_labels_a_median_and_adds_the_footnote_only_then(tmp_path):
    def table(configs):
        path = tmp_path / "comparison.json"
        path.write_text(json.dumps({"configs": configs}))
        return readme._render_runtime_table([readme.RuntimeCell(("a", "b"), path)], {})

    plain = table({"hpc_a100_fp64": _run(0.00642, "1")})
    assert "6.4 ms |" in plain and readme.HEADLINE_FOOTNOTE not in plain and "_(" not in plain
    labelled = table({"hpc_a100_fp64": _run(0.00642, "1", median_ms=0.267)})
    assert "267 μs _(median)_" in labelled and readme.HEADLINE_FOOTNOTE in labelled


def test_no_committed_runtime_row_carries_a_median_or_repeats_so_the_readme_is_unchanged():
    for path in sorted((_ROOT / "results").rglob("comparison.json")):
        configs = json.loads(path.read_text()).get("configs", {})
        for cfg in configs.values():
            assert readme._config_headline(cfg) == (readme._config_legacy_seconds(cfg), ""), path
            assert bd.REPEAT_RUNS_FIELD not in cfg, path


# --- the dashboard: the cube median -------------------------------------------------------------


def test_the_dashboard_headlines_the_cube_median_and_keeps_the_gpu_note_off_it():
    row = _run(None, "1", full_pipeline_cube_single_jit=0.5)
    del row["full_pipeline_single_jit"]
    p = bd._point(row, "2026.10.1.1", "x")
    assert p["single_jit_s"] == 0.5 and "headline_estimator" not in p
    row["full_pipeline_cube_single_jit_median_ms"] = 400.0
    p = bd._point(row, "2026.10.1.1", "x")
    assert p["single_jit_s"] == pytest.approx(0.4)
    assert p["headline_estimator"] == bd.ESTIMATOR_MEDIAN and p["single_jit_block_mean_s"] == 0.5
    assert "headline_note" not in p  # the A100 first-block note stays the single-jit key's


# --- single_jit_repeats: flat becomes reachable --------------------------------------------------


def _release(values: list[float], version: str, first_job: int) -> dict:
    runs = [_run(v, str(first_job + i), version=version) for i, v in enumerate(values)]
    payload = _with_repeats(runs) if len(runs) > 1 else runs[0]
    return bd._point(payload, version, f"results/runtime/c/row_{version}.json")


def test_repeat_endpoints_publish_flat_and_single_runs_stay_insufficient():
    # The same per-run numbers, 1.17x apart: inside the 2x band either way.
    a, b = [0.0100, 0.0110, 0.0105], [0.0120, 0.0125, 0.0118]
    repeated = _series([_release(a, "2026.10.1.1", 100), _release(b, "2026.10.2.1", 200)])
    single = _series([_release(a[:1], "2026.10.1.1", 100), _release(b[:1], "2026.10.2.1", 200)])
    assert repeated["drift"]["status"] == single["drift"]["status"] == "steady"
    pts = repeated["points"]
    assert [p[bd.REPEAT_SUMMARY_FIELD] for p in pts] == [3, 3]
    assert pts[0]["single_jit_s"] == 0.0105 and pts[1]["single_jit_s"] == 0.0120
    assert all(p["qualified"] for p in pts)
    flat = bd._summary_comparison(repeated)
    assert flat["status"] == "flat" and flat["reasons"] == [bd.FLAT_BAND_REASON]
    insufficient = bd._summary_comparison(single)
    assert insufficient["status"] == "insufficient"
    assert insufficient["reasons"] == [bd.SINGLE_SAMPLE_NULL_REASON]
    # One repeat endpoint is not enough.
    mixed = _series([_release(a, "2026.10.1.1", 100), _release(b[:1], "2026.10.2.1", 200)])
    assert bd._summary_comparison(mixed)["status"] == "insufficient"


def test_a_gross_move_between_repeat_endpoints_keeps_its_status_without_the_caveat():
    up = _series([_release([0.010, 0.011], "1", 1), _release([0.030, 0.031], "2", 9)])
    c = bd._summary_comparison(up)
    assert c["status"] == "drifted" and bd.SINGLE_SAMPLE_REASON not in c["reasons"]


@pytest.mark.parametrize(
    "second, why",
    [
        (_run(0.011, "100"), "the same job and source is one run, not two"),
        (_run(0.011, "101", host="euclid-ral-gpu-1"), "another host is not a repeat on this one"),
        (_run(0.011, "101", median_ms=10.5), "another estimator is not the compared metric"),
        (_run(None, "101"), "a run with no headline"),
    ],
)
def test_runs_that_are_not_independent_repeats_of_the_metric_do_not_count(second, why):
    first = _run(0.010, "100")
    listed = [dict(first, source="row.json"), dict(second, source="row.json")]
    if "101" in json.dumps(second):
        listed[1]["source"] = "row.repeat2.json"
    payload = dict(first, **{bd.REPEAT_RUNS_FIELD: listed})
    assert bd.repeat_summary(payload) is None, why
    assert bd.REPEAT_SUMMARY_FIELD not in bd._point(payload, "1", "x"), why


def test_qualification_judges_the_noisiest_repeat_run():
    runs = [_run(0.010, "1", load=0.5), _run(0.011, "2", load=9.5)]
    p = bd._point(_with_repeats(runs), "1", "x")
    assert p["loadavg"] == 9.5
    ok, reason, refused = bd.qualify(p, "hpc_a100_fp64", _CONF)
    assert not ok and refused and "above the cap" in reason


def test_the_aggregator_lists_repeat_runs_of_the_same_release_only(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    row = _run(0.010, "1", version="2026.10.1.1")
    (cell / "x_hpc_a100_fp64.json").write_text(json.dumps(row))
    (cell / "x_hpc_a100_fp64.repeat2.json").write_text(json.dumps(_run(0.012, "2")))
    (cell / "x_hpc_a100_fp64.repeat3.json").write_text(json.dumps(_run(0.011, "3")))
    other = _run(0.5, "4", version="2026.9.1.1")
    (cell / "x_hpc_a100_fp64.repeat4.json").write_text(json.dumps(other))
    configs = aggregate._aggregate_cell(cell)["configs"]
    assert list(configs) == ["hpc_a100_fp64"]  # repeat files are never rows of their own
    entry = configs["hpc_a100_fp64"]
    runs = entry[aggregate.REPEAT_RUNS_FIELD]
    assert [r["source"] for r in runs] == [
        "x_hpc_a100_fp64.json",
        "x_hpc_a100_fp64.repeat2.json",
        "x_hpc_a100_fp64.repeat3.json",
    ]
    assert aggregate.REPEAT_RUNS_FIELD == bd.REPEAT_RUNS_FIELD
    p = bd._point(entry, "2026.10.1.1", "comparison.json#hpc_a100_fp64")
    assert p[bd.REPEAT_SUMMARY_FIELD] == 3 and p["single_jit_s"] == 0.011
    # A row with no repeat files is aggregated exactly as before.
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "x_hpc_a100_fp64.json").write_text(json.dumps(row))
    assert (
        aggregate.REPEAT_RUNS_FIELD
        not in aggregate._aggregate_cell(lone)["configs"]["hpc_a100_fp64"]
    )


def test_the_readme_reads_a_repeat_summary_with_its_label():
    runs = [_run(v, str(i)) for i, v in enumerate([0.010, 0.012, 0.011])]
    seconds, label = readme._config_headline(_with_repeats(runs))
    assert seconds == 0.011 and label == "median of 3 runs"


# --- the wall basis -------------------------------------------------------------------------------


def test_the_wall_constants_mirror_the_producer():
    assert rates.HEADLINE_MEDIAN_MIN_STEADY_WARM == timing.MIN_STEADY_WARM
    assert rates.HEADLINE_MEDIAN_BUDGET_S == timing.HEADLINE_MEDIAN_BUDGET_S
    assert rates.HEADLINE_MEDIAN_MIN_TIMED == timing.HEADLINE_MEDIAN_MIN_TIMED
    assert rates.HEADLINE_MEDIAN_MAX_TIMED == timing.HEADLINE_MEDIAN_MAX_TIMED
    assert rates.HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S == timing.HEADLINE_MEDIAN_MAX_BLOCK_MEAN_S


def test_the_extra_wall_bound_covers_every_block_mean():
    bound = rates.headline_median_extra_wall_s()
    assert bound == 50.0
    means = [1e-5 * 1.01**k for k in range(1400)] + [2.0]
    for m in means:
        extra = rates.headline_median_extra_wall_s(m)
        n = timing.headline_median_n_timed(m) if m <= 2.0 else 0
        assert extra == pytest.approx((timing.MIN_STEADY_WARM + n) * m if n else 0.0)
        assert extra <= bound + 1e-9
    assert rates.headline_median_extra_wall_s(2.5) == 0.0  # no median above the cut-off


def test_the_one_fixed_n_timed_cell_stays_inside_the_bound():
    """source_plane_solved keeps n_timed=200 (#371): its 205 calls fit the 50 s bound."""
    explicit = [
        p
        for p in (_ROOT / "scripts").rglob("*.py")
        if "misc/test" not in p.as_posix() and "n_timed=200" in p.read_text(encoding="utf-8")
    ]
    assert [p.name for p in explicit] == ["likelihood_runtime_solved.py"]
    worst = 0.0
    for path in (_ROOT / "results/runtime/point_source_source/source_plane_solved").glob("*.json"):
        v = json.loads(path.read_text()).get("full_pipeline_single_jit")
        if isinstance(v, float):
            worst = max(worst, v)
    assert (
        0 < worst and (timing.MIN_STEADY_WARM + 200) * worst < rates.headline_median_extra_wall_s()
    )


_SUBMIT = """#!/bin/bash -l
# WALL-BASIS:
#   cell: {cell}  device: cpu  precision: fp64
#   source: measured-wall  wall: 100  ref: laptop  headroom: 1.25{extra}

#SBATCH --time={time}

python3 {script} --instrument simple
"""


def _check(script: str, cell: str, time: str, extra: str = "") -> list[str]:
    return check_submits.check_text(
        _SUBMIT.format(script=script, cell=cell, time=time, extra=extra)
    )


def test_a_median_cell_needs_the_extra_wall_and_a_plain_cell_does_not():
    median = "scripts/point_source_source/source_plane/likelihood_runtime.py"
    plain = "scripts/imaging/rectangular/likelihood_runtime_numba.py"
    cell_m = "point_source_source/source_plane/simple"
    cell_p = "imaging/pixelization_numba/simple"
    # 125 s was enough for 1.25 x 100 s before the median; it no longer is (1.25 x 150 s).
    assert _check(plain, cell_p, "0:02:05") == []
    problems = _check(median, cell_m, "0:02:05")
    assert len(problems) == 1 and "estimated 150 s wall" in problems[0]
    assert _check(median, cell_m, "0:03:08") == []
    # A wall measured with the median in it says so and is not padded twice.
    assert _check(median, cell_m, "0:02:05", extra="  median: included") == []
    # A loop that runs the cell k times declares it.
    assert "estimated 250 s" in _check(median, cell_m, "0:02:05", extra="  median-runs: 3")[0]


def test_the_a100_solved_submit_counts_both_invocations():
    text = (
        _ROOT / "hpc/batch_gpu/submit_runtime_point_source_source_source_plane_solved_a100_fp64"
    ).read_text()
    assert check_submits.median_runs(text) == {("point_source_source", "source_plane_solved"): 2}
    assert check_submits.check_text(text) == []
