"""``likelihood_breakdown.timing``: the opt-in steady median beside ``jit_profile``'s block.

autolens_profiling#371 (option (a)). ``jit_profile``'s ``steady_per_call_s`` is the mean of one
``n_repeats`` block right after the first call. On the A100 it can land in the post-compile
transient (source-plane 0.642 ms vs a steady 0.267 ms median, jobs 366912 / 366914). These tests
check three things:
- the steady median counts and times exactly the calls it says it does;
- it refuses fewer than 5 warm calls;
- opting in leaves the old statistic and ``timer.records`` exactly as they are without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from likelihood_breakdown import timing  # noqa: E402


class _Counted:
    """A fake compiled callable that counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.calls


class _Clock:
    """A fake clock: call i (0-based) of the timed loop lasts ``durations[i]`` seconds."""

    def __init__(self, durations):
        self.durations = list(durations)
        self.now = 0.0
        self.reads = 0

    def __call__(self):
        # Two reads per timed call: start, then stop.
        if self.reads % 2 == 1:
            self.now += self.durations[self.reads // 2]
        self.reads += 1
        return self.now


def test_steady_median_counts_warm_and_timed_calls_and_reports_quantiles():
    fn = _Counted()
    durations = [float(i) for i in range(1, 11)]  # 1..10 s, median 5.5
    clock = _Clock(durations)

    out = timing.steady_median_profile(fn, "x", n_warm=5, n_timed=10, clock=clock)

    assert fn.calls == 15  # 5 warm (untimed) + 10 timed
    assert clock.reads == 20  # only the timed calls read the clock
    assert out["n_warm"] == 5 and out["n_timed"] == 10
    assert out["median_s"] == pytest.approx(5.5)
    assert out["p10_s"] == pytest.approx(1.9)  # numpy-default linear quantile
    assert out["p90_s"] == pytest.approx(9.1)
    assert out["mean_s"] == pytest.approx(5.5)


def test_steady_median_is_robust_to_a_transient_the_block_mean_is_not():
    # One 10x-slow call (the post-compile transient) among steady 1 s calls.
    durations = [10.0] + [1.0] * 19
    out = timing.steady_median_profile(_Counted(), n_warm=5, n_timed=20, clock=_Clock(durations))
    assert out["median_s"] == pytest.approx(1.0)
    assert out["mean_s"] == pytest.approx(29.0 / 20)


@pytest.mark.parametrize("n_warm", [0, 1, 4])
def test_steady_median_refuses_fewer_than_five_warm_calls(n_warm):
    with pytest.raises(ValueError, match="n_warm must be >= 5"):
        timing.steady_median_profile(_Counted(), n_warm=n_warm, n_timed=10)


def test_steady_median_refuses_zero_timed_calls():
    with pytest.raises(ValueError, match="n_timed must be >= 1"):
        timing.steady_median_profile(_Counted(), n_warm=5, n_timed=0)


def _profile(**kw):
    import jax.numpy as jnp

    timer = timing.Timer()
    records: dict = {}
    _, result = timing.jit_profile(
        lambda x: jnp.sum(x * x), "f", jnp.arange(4.0), timer=timer, jit_records=records, **kw
    )
    return timer, records, result


def test_jit_profile_without_opt_in_is_unchanged():
    timer, records, result = _profile()
    assert [r[0] for r in timer.records] == ["f_lower", "f_compile", "f_first_call", "f_steady_x10"]
    assert set(records["f"]) == {"lower_s", "compile_s", "first_call_s", "steady_per_call_s"}
    assert records["f"]["steady_per_call_s"] == pytest.approx(timer.records[-1][1] / 10)
    assert float(result) == 14.0


def test_jit_profile_opt_in_adds_the_median_beside_the_old_statistic():
    timer, records, result = _profile(median_n_warm=5, median_n_timed=7)
    # No extra timer section: timer.records[-1] is still the 10-call block that cells divide by 10.
    assert [r[0] for r in timer.records] == ["f_lower", "f_compile", "f_first_call", "f_steady_x10"]
    rec = records["f"]
    assert rec["steady_per_call_s"] == pytest.approx(timer.records[-1][1] / 10)
    med = rec["steady_median"]
    assert med["n_warm"] == 5 and med["n_timed"] == 7
    assert med["p10_s"] <= med["median_s"] <= med["p90_s"]
    assert float(result) == 14.0


def test_jit_profile_opt_in_refuses_too_few_warm_calls():
    with pytest.raises(ValueError, match="n_warm must be >= 5"):
        _profile(median_n_warm=2, median_n_timed=7)
