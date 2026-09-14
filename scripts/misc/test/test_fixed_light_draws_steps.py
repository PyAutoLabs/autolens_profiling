"""Unit tests for the graded Δlog L draw set.

Phase 3 of the ``fixed-lens-light-profiling`` epic (autolens_profiling#255).
**NumPy only** — the draw set is a construction over a scalar callback, so every
property that matters here (does the bisection land on its target, is the random
family reproducible, is the fallback rate arithmetic right) is testable without
an autolens, JAX or dataset import. The expensive end — the S0→S3 rebuild — is
the cell's, and is pinned there by the fiducial pass-count / PDIP-iteration pins.

Three tiers:

1. **The bisection**, against synthetic monotone offsets whose answer is known
   in closed form. A bisection that reports "reached" while landing somewhere
   else is the failure this tier catches.
2. **The random family**, seed-reproducibility and the σ it actually draws at.
3. **The derived tables** — the fallback rate from a hand-made pass-count list,
   including the ``None`` (never certified) case, which must count as a fallback
   at *every* budget. And two static checks on the cell: that ``--pins none``
   withdraws verdicts rather than skipping comparisons, and that the
   slope-promotion check gates the slope walk.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_draws_steps.py
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


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import fixed_light_draws_steps as flds  # noqa: E402

CELL = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_draws.py"


# ---------------------------------------------------------------------------
# 1. The bisection
# ---------------------------------------------------------------------------


def _quadratic(k: float):
    """``Δlog L = -k δ²`` — a locally quadratic likelihood, the seed's own model."""

    def evaluate(delta: float) -> float:
        return -k * float(delta) ** 2

    return evaluate


def _quartic(k: float):
    """``Δlog L = -k δ⁴`` — monotone but *not* the model the seed assumes."""

    def evaluate(delta: float) -> float:
        return -k * float(delta) ** 4

    return evaluate


@pytest.mark.parametrize("target", [-10.0, -100.0, -1000.0, -1.0e4])
@pytest.mark.parametrize("evaluate_from,k", [(_quadratic, 5.0e6), (_quartic, 1.0e8)])
def test_bisection_lands_within_20_percent_of_its_target(evaluate_from, k, target):
    """Every target is reached to the declared 20 % tolerance, inside the budget."""
    result = flds.bisect_to_target(
        evaluate_from(k),
        target,
        probe_delta=0.016,
        max_delta=1.2,
        max_evals=12,
        rtol=0.2,
    )
    assert result.reached is True
    assert abs(result.achieved - target) <= 0.2 * abs(target)
    assert result.n_evaluations <= 12
    assert result.delta > 0.0


def test_bisection_seed_is_exact_on_a_quadratic():
    """On the model the seed assumes, the very first prediction is the answer.

    Not a micro-optimisation test: the whole reason the walk is affordable (the
    A100 leg rebuilds a full S0 fit per evaluation, ~5 s each) is that the
    quadratic seed starts *inside* the bracket instead of climbing to it.
    """
    result = flds.bisect_to_target(
        _quadratic(5.0e6), -1000.0, probe_delta=0.016, max_delta=1.2, max_evals=12, rtol=0.2
    )
    assert result.n_evaluations <= 2
    assert result.achieved == pytest.approx(-1000.0, rel=1e-6)


def test_bisection_shares_its_cache_across_targets():
    """A cache passed in is used: the second target pays only for new deltas."""
    cache: dict = {}
    calls = {"n": 0}

    def evaluate(delta: float) -> float:
        calls["n"] += 1
        return -5.0e6 * float(delta) ** 2

    flds.bisect_to_target(evaluate, -100.0, probe_delta=0.016, max_delta=1.2, cache=cache, rtol=0.2)
    first = calls["n"]
    assert len(cache) == first

    # Re-running the same target costs nothing at all: every delta is cached.
    flds.bisect_to_target(evaluate, -100.0, probe_delta=0.016, max_delta=1.2, cache=cache, rtol=0.2)
    assert calls["n"] == first


def test_bisection_reports_not_reached_rather_than_relabelling():
    """A target beyond ``max_delta`` is reported with what it landed on."""
    result = flds.bisect_to_target(
        _quadratic(1.0),  # far too shallow to reach -1e4 within max_delta
        -1.0e4,
        probe_delta=0.01,
        max_delta=0.05,
        max_evals=12,
        rtol=0.2,
    )
    assert result.reached is False
    assert result.achieved > -1.0e4  # shallower than the target
    assert abs(result.delta) <= 0.05


def test_bisection_rejects_a_positive_target():
    with pytest.raises(ValueError):
        flds.bisect_to_target(_quadratic(1.0), 10.0, probe_delta=0.01, max_delta=1.0)


def test_walk_draws_covers_every_parameter_and_target():
    """The walk family is the parameter × target grid, named and provenanced."""

    def evaluate_walk(parameter: str, delta: float) -> float:
        return -5.0e6 * float(delta) ** 2

    draws = flds.walk_draws(
        evaluate_walk,
        parameters=("einstein_radius", "slope"),
        targets=(-10.0, -1000.0),
        max_evals=12,
    )
    assert len(draws) == 4
    assert {d.parameter for d in draws} == {"einstein_radius", "slope"}
    assert all(d.kind == "walk" for d in draws)
    # The slope walk is the only one that promotes the mass.
    assert {d.mass_cls for d in draws if d.parameter == "slope"} == {"PowerLaw"}
    assert {d.mass_cls for d in draws if d.parameter == "einstein_radius"} == {"Isothermal"}
    # Each draw carries the Δlog L it achieved, not the one it aimed at.
    for draw in draws:
        assert draw.achieved_d_log_l is not None
        assert abs(draw.achieved_d_log_l - draw.target) <= 0.2 * abs(draw.target)


# ---------------------------------------------------------------------------
# 2. The random family
# ---------------------------------------------------------------------------


def test_random_draws_are_seed_reproducible():
    """Same seed, same draws — the property that makes the legs comparable."""
    a = flds.random_offsets(0, 24)
    b = flds.random_offsets(0, 24)
    assert a == b

    c = flds.random_offsets(1, 24)
    assert c != a


def test_random_draws_are_a_prefix_of_a_longer_set():
    """``n`` draws are the first ``n`` of ``n + k`` — extending never re-rolls."""
    assert flds.random_offsets(0, 8) == flds.random_offsets(0, 24)[:8]


def test_random_draws_move_every_parameter_at_the_declared_sigma():
    offsets = flds.random_offsets(0, 4000)
    assert set(offsets[0]) == set(flds.RANDOM_PARAMETERS)
    for parameter in flds.RANDOM_PARAMETERS:
        expected = flds.RANDOM_SIGMA_SCALE * flds.PRIOR_SIGMA[parameter]
        measured = float(np.std([o[parameter] for o in offsets]))
        assert measured == pytest.approx(expected, rel=0.1)


def test_random_draws_do_not_move_the_slope():
    """The random family is the Isothermal early-search sample."""
    assert "slope" not in flds.RANDOM_PARAMETERS
    for draw in flds.random_draws(0, 4):
        assert draw.mass_cls == "Isothermal"
        assert "slope" not in draw.offsets


# ---------------------------------------------------------------------------
# 3. The derived tables
# ---------------------------------------------------------------------------


def test_fallback_table_from_a_hand_made_pass_count_list():
    """The number the phase exists for, computed from a list whose answer is known."""
    pass_counts = [2, 2, 3, 5, 7, 8, None]  # 7 draws
    rows = flds.fallback_table(pass_counts, budgets=(2, 4, 6, 7))
    by_budget = {r["budget"]: r for r in rows}

    # budget 2 -> everything above 2 falls back: 3, 5, 7, 8 and the None = 5/7
    assert by_budget[2]["n_fallback"] == 5
    assert by_budget[2]["fallback_rate"] == pytest.approx(5 / 7)
    # budget 4 -> 5, 7, 8, None = 4/7
    assert by_budget[4]["n_fallback"] == 4
    # budget 6 -> 7, 8, None = 3/7
    assert by_budget[6]["n_fallback"] == 3
    # budget 7 -> 8, None = 2/7
    assert by_budget[7]["n_fallback"] == 2
    for row in rows:
        assert row["n_draws"] == 7
        assert row["n_never_certified"] == 1


def test_fallback_table_counts_never_certified_at_every_budget():
    """A draw that never certified is a fallback at the *largest* budget too."""
    rows = flds.fallback_table([None, None], budgets=(2, 100))
    assert [r["n_fallback"] for r in rows] == [2, 2]
    assert [r["fallback_rate"] for r in rows] == [1.0, 1.0]


def test_fallback_table_of_an_all_easy_set_is_zero():
    rows = flds.fallback_table([1, 2, 2], budgets=(2, 4))
    assert [r["fallback_rate"] for r in rows] == [0.0, 0.0]


def test_rank_correlation_matches_the_monotone_cases():
    assert flds.rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert flds.rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert flds.rank_correlation([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert flds.rank_correlation([1, 2], [1, 2]) is None  # too few points


def test_rank_correlation_averages_ties():
    # Ties must not become an arbitrary order: [1, 1, 2] against [5, 5, 9] is
    # perfectly monotone once ties are averaged.
    assert flds.rank_correlation([1, 1, 2, 3], [5, 5, 9, 11]) == pytest.approx(1.0)


def test_quantile_summary_is_none_safe():
    assert flds.quantile_summary([]) == {"n": 0}
    assert flds.quantile_summary([None, None]) == {"n": 0}
    summary = flds.quantile_summary([1.0, 2.0, 3.0, 4.0], (0.5, 1.0))
    assert summary["n"] == 4
    assert summary["q50"] == pytest.approx(2.5)
    assert summary["q100"] == pytest.approx(4.0)
    assert summary["min"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Mass construction — the alias and the promotion guard
# ---------------------------------------------------------------------------


BASE = {
    "centre_0": 0.0,
    "centre_1": 0.0,
    "ell_comps_0": 0.05,
    "ell_comps_1": 0.02,
    "einstein_radius": 1.6,
    "slope": 2.0,
}


def test_mass_from_refuses_a_slope_offset_on_an_isothermal():
    """An Isothermal has no slope — a silent no-op here would fake a slope walk."""
    with pytest.raises(ValueError):
        flds.mass_from(BASE, {"slope": 0.1}, mass_cls="Isothermal")


def test_mass_from_rejects_an_unknown_parameter():
    with pytest.raises(KeyError):
        flds.mass_from(BASE, {"not_a_parameter": 0.1})


def test_fiducial_mass_values_keys_are_the_walk_and_random_parameters():
    """Every parameter the draw set moves must exist in the fiducial dict."""
    keys = set(BASE)
    for parameter in flds.RANDOM_PARAMETERS:
        assert parameter in keys
    for parameter in flds.WALK_PARAMETERS:
        alias = "centre_1" if parameter == "centre_x" else parameter
        assert alias in keys


# ---------------------------------------------------------------------------
# 5. Static checks on the cell
# ---------------------------------------------------------------------------


def test_cell_gates_the_slope_walk_on_the_promotion_check():
    """The slope rows must be withdrawn, not measured, if the promotion drifted."""
    text = CELL.read_text()
    assert "SLOPE_PROMOTION_OK" in text
    assert 'p != "slope" or SLOPE_PROMOTION_OK' in text


def test_cell_pins_none_records_rather_than_skips():
    """``--pins none`` withdraws verdicts; every comparison is still written."""
    text = CELL.read_text()
    assert '"RECORDED"' in text
    assert "pins_recorded" in text
    # The verdict is conditional on PINS_ASSERT in every pin path.
    assert text.count("if PINS_ASSERT else") >= 2


def test_cell_never_certified_is_a_null_not_a_max():
    """A draw that ran out of passes is ``null``, so the fallback table sees it."""
    text = CELL.read_text()
    assert (
        "int(run.passes_to_certification) if run.passes_to_certification is not None else None"
        in text
    )


def test_cell_declares_the_phase_0_fiducial_pins():
    text = CELL.read_text()
    assert '"rectangular": 7' in text and '"delaunay": 2' in text
    assert '"rectangular": 15' in text and '"delaunay": 17' in text
