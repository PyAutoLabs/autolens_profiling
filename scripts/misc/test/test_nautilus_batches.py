"""The captured-batch file format and lane rules (``likelihood_breakdown.nautilus_batches``).

Pure numpy: no JAX, no Nautilus run. The recorder's patching is exercised by the
capture cell itself; these pin the rules the replay's numbers depend on.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import nautilus_batches as nb  # noqa: E402


class _Sampler:
    def __init__(self, n_bounds, explored, n_like):
        self.bounds = [object()] * n_bounds
        self.explored = explored
        self.n_like = n_like


def _recorder(sizes, states):
    rec = nb.BatchRecorder()
    rng = np.random.default_rng(0)
    for size, (n_bounds, explored) in zip(sizes, states):
        rec.note_sampler_state(_Sampler(n_bounds, explored, rec.n_lanes))
        params = rng.normal(size=(size, 7))
        rec.record(params, params.sum(axis=1), wall_s=0.1)
    return rec


def test__phase_of():
    assert nb.phase_of(1, False) == "prior"
    assert nb.phase_of(0, False) == "prior"
    assert nb.phase_of(3, False) == "exploration"
    assert nb.phase_of(3, True) == "sampling"


def test__save_load_round_trip(tmp_path):
    rec = _recorder([4, 4, 3], [(1, False), (2, False), (2, True)])
    summary = nb.save(tmp_path / "b.npz", rec, {"parameter_paths": list("abcdefg")})
    assert summary["n_calls"] == 3 and summary["n_lanes"] == 11
    assert summary["call_sizes"] == {"3": 1, "4": 2}
    data = nb.load(tmp_path / "b.npz")
    assert data["parameters"].shape == (11, 7)
    assert data["call_index"].tolist() == [0] * 4 + [1] * 4 + [2] * 3
    assert data["lane_in_call"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert data["call_phase"].tolist() == ["prior", "exploration", "sampling"]
    assert data["call_n_like_before"].tolist() == [0, 4, 8]
    assert data["meta"]["parameter_paths"] == list("abcdefg")
    np.testing.assert_allclose(data["figure_of_merit"], data["parameters"].sum(axis=1))


def test__record_refuses_a_batch_axis_mismatch():
    rec = nb.BatchRecorder()
    with pytest.raises(AssertionError):
        rec.record(np.zeros((4, 7)), np.zeros(3), wall_s=0.0)


def test__save_refuses_an_empty_capture(tmp_path):
    with pytest.raises(ValueError):
        nb.save(tmp_path / "b.npz", nb.BatchRecorder(), {})


def test__sample_calls_spans_the_run():
    assert nb.sample_calls(10, None).tolist() == list(range(10))
    assert nb.sample_calls(10, 50).tolist() == list(range(10))
    sampled = nb.sample_calls(101, 5).tolist()
    assert sampled == [0, 25, 50, 75, 100]


def test__timed_window_starts_at_the_first_post_prior_call():
    call_index = np.repeat(np.arange(5), 4)
    call_phase = np.array(["prior", "prior", "exploration", "exploration", "sampling"])
    window = nb.timed_window(call_index, call_phase, 4)
    assert window.tolist() == [8, 9, 10, 11]  # exactly call 2
    window = nb.timed_window(call_index, call_phase, 6)
    assert call_index[window].tolist() == [2, 2, 2, 2, 3, 3]
    # Not enough post-prior lanes: the window slides back, never repeats a lane.
    window = nb.timed_window(call_index, call_phase, 16)
    assert window.tolist() == list(range(4, 20))
    with pytest.raises(ValueError):
        nb.timed_window(call_index, call_phase, 21)


def test__chunked_pads_the_last_chunk_and_reports_real_rows():
    chunks = nb.chunked(np.arange(7), 3)
    assert [c.tolist() for c, _ in chunks] == [[0, 1, 2], [3, 4, 5], [6, 6, 6]]
    assert [n for _, n in chunks] == [3, 3, 1]


def test__rate_summary_and_groups():
    certified = [True, True, False, True]
    passes = [0, 3, 16, 2]
    summary = nb.rate_summary(certified, passes, budget=16)
    assert summary["n_uncertified"] == 1
    assert summary["uncertified_rate"] == 0.25
    assert summary["max_passes"] == 16
    assert summary["n_at_budget"] == 1
    assert summary["passes_histogram"] == {"0": 1, "2": 1, "3": 1, "16": 1}
    groups = nb.grouped_rate_summaries(certified, passes, 16, ["a", "a", "b", "b"])
    assert groups["a"]["n_uncertified"] == 0 and groups["b"]["n_uncertified"] == 1


def test__early_late_split_is_the_median_replayed_call():
    labels = nb.early_late(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
    assert labels.tolist() == ["early", "early", "late", "late"]
