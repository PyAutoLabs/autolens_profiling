"""Unit tests for the ``SEARCHES_POSITIONS*`` env-var plumbing (Phase 4 Stage 1,
issue #159).

Covers: env-var parsing/validation, the composed arm tag, the recorded
``positions_settings()`` block, the ``dataset_class`` support guard (raises
rather than silently ignoring), and — the load-bearing correctness property —
that a positions-on search and a positions-off search of the "same" cell
resolve to a DIFFERENT ``output_path`` *and* a different ``identifier``. See
``_setup.py``'s "PositionsLH plumbing" section docstring for why this matters:
PyAutoFit's identifier hashes only ``[search, model, unique_tag]``, so the
``Analysis`` object (and therefore whether a positions penalty is attached) is
never part of the hash on its own.

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_positions.py -q
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_misc_dir = str(_ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import os
from pathlib import Path

import pytest
from searches._samplers import (
    assert_disjoint_output_paths,
    build_multi_start,
    build_nautilus,
    multi_start_unique_tag,
)
from searches._setup import (
    _positions_likelihood_list_for,
    build_for_cell,
    positions_arm_tag,
    positions_enabled,
    positions_settings,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# env-var parsing / validation
# ---------------------------------------------------------------------------


def test_positions_default_off(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    assert positions_enabled() is False
    assert positions_arm_tag() is None
    assert positions_settings() == {"enabled": False}


def test_positions_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "maybe")
    with pytest.raises(ValueError):
        positions_enabled()


def test_positions_threshold_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_THRESHOLD", "not-a-number")
    with pytest.raises(ValueError):
        positions_settings()


def test_positions_factor_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_FACTOR", "not-a-number")
    with pytest.raises(ValueError):
        positions_settings()


# ---------------------------------------------------------------------------
# arm tag composition
# ---------------------------------------------------------------------------


def test_arm_tag_fixed_default(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.delenv("SEARCHES_POSITIONS_THRESHOLD", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS_FACTOR", raising=False)
    assert positions_arm_tag() == "pos_t0.3_f1e8"


def test_arm_tag_auto_custom_factor(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_THRESHOLD", "auto")
    monkeypatch.setenv("SEARCHES_POSITIONS_FACTOR", "1e5")
    assert positions_arm_tag() == "pos_tauto0.2_f1e5"


def test_settings_auto_collapses_to_floor(monkeypatch):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_THRESHOLD", "auto")
    settings = positions_settings()
    assert settings["threshold_mode"] == "auto"
    assert settings["threshold_value"] == pytest.approx(0.2)
    assert settings["source"] == "simulator_truth_positions"
    assert settings["target_class"] == 3


# ---------------------------------------------------------------------------
# dataset_class support guard — never silently ignore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_class,model_type,instrument",
    [
        ("point_source", "image_plane", "simple"),
        ("cluster", "source_plane", "simple"),
        ("group", "mge", "hst"),
    ],
)
def test_unsupported_dataset_class_raises_not_ignores(
    monkeypatch, dataset_class, model_type, instrument
):
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    with pytest.raises(NotImplementedError):
        build_for_cell(
            dataset_class=dataset_class,
            model_type=model_type,
            instrument=instrument,
            use_jax=False,
        )


def test_positions_likelihood_list_none_when_off(monkeypatch):
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    assert _positions_likelihood_list_for("imaging", "hst", Path("dataset/imaging/hst")) is None


# ---------------------------------------------------------------------------
# truth positions — derived-and-committed dataset/imaging/hst/positions.json
# ---------------------------------------------------------------------------


def test_truth_positions_committed_and_reused(monkeypatch):
    """dataset/imaging/hst/positions.json must exist (derived once, committed)
    and loading it must not re-derive (no second write / no crash on a
    read-only checkout)."""
    positions_path = _ROOT / "dataset" / "imaging" / "hst" / "positions.json"
    assert positions_path.exists(), (
        "dataset/imaging/hst/positions.json is missing — it should have been "
        "derived once from tracer.json and committed (see _setup.py's "
        "_truth_positions_for)."
    )
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_THRESHOLD", "0.3")
    monkeypatch.setenv("SEARCHES_POSITIONS_FACTOR", "1e8")
    positions_likelihood_list = _positions_likelihood_list_for(
        "imaging", "hst", positions_path.parent
    )
    assert positions_likelihood_list is not None
    assert len(positions_likelihood_list) == 1
    positions_lh = positions_likelihood_list[0]
    assert len(positions_lh.positions) >= 2
    assert positions_lh.threshold == pytest.approx(0.3)
    assert positions_lh.log_likelihood_penalty_factor == pytest.approx(1e8)


# ---------------------------------------------------------------------------
# THE correctness guard: positions-on vs positions-off never collide
# ---------------------------------------------------------------------------


def test_multi_start_unique_tag_differs_on_vs_off(monkeypatch):
    monkeypatch.delenv("SEARCHES_SEED", raising=False)
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    tag_off = multi_start_unique_tag("imaging", "mge")
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    tag_on = multi_start_unique_tag("imaging", "mge")
    assert tag_off is None
    assert tag_on is not None
    assert tag_on != tag_off


def test_multi_start_search_output_path_and_identifier_differ_on_vs_off(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    search_off = build_multi_start(
        sampler="multi_start_adam",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    search_on = build_multi_start(
        sampler="multi_start_adam",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    assert_disjoint_output_paths(search_off, search_on)


def test_nautilus_search_output_path_and_identifier_differ_on_vs_off(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    monkeypatch.delenv("SEARCHES_POSITIONS", raising=False)
    search_off = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    search_on = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    assert_disjoint_output_paths(search_off, search_on)


def test_two_positions_arms_with_different_factor_also_differ(monkeypatch):
    """Two positions-ON arms that differ only in factor must ALSO resolve to
    distinct output_path/identifier — the arm tag has to encode the whole
    positions config, not just the on/off bit."""
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    monkeypatch.setenv("SEARCHES_POSITIONS", "on")
    monkeypatch.setenv("SEARCHES_POSITIONS_FACTOR", "1e5")
    search_a = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    monkeypatch.setenv("SEARCHES_POSITIONS_FACTOR", "1e8")
    search_b = build_nautilus(
        sampler="nautilus",
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    assert_disjoint_output_paths(search_a, search_b)
