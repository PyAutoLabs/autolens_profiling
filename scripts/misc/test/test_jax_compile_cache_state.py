"""Unit tests for ``jax_compile/probe.py`` cache-state derivation.

``cache_state`` is what makes a warm compile machine-identifiable. Before it,
warmness lived only in the free-text ``tag`` (~40 ad-hoc spellings across the
committed corpus), so the dashboard that has to track *warm* compile could not
tell which rows were warm.

No JAX dependency — the functions under test are pure. Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_jax_compile_cache_state.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_jax_compile = _profiling_root() / "scripts" / "misc" / "jax_compile"
if str(_jax_compile) not in _sys.path:
    _sys.path.insert(0, str(_jax_compile))

import probe  # noqa: E402


def test_no_cache_dir_is_none():
    assert probe.cache_state_from(0, 0, None) == "none"
    assert probe.cache_state_from(0, 0, "") == "none"


def test_a_compile_that_writes_an_entry_is_cold():
    """A new entry means XLA compiled it rather than reusing one: a MISS."""
    assert probe.cache_state_from(0, 1, "/tmp/c") == "cold"
    assert probe.cache_state_from(7, 8, "/tmp/c") == "cold"


def test_a_compile_that_writes_nothing_into_a_populated_cache_is_warm():
    assert probe.cache_state_from(8, 8, "/tmp/c") == "warm"


def test_configured_but_empty_and_nothing_written_is_unknown():
    """Not silently 'warm': an empty cache cannot have been hit."""
    assert probe.cache_state_from(0, 0, "/tmp/c") == "unknown"


def test_cache_dir_alone_does_not_imply_warm():
    """The trap the tag-parsing approach fell into.

    ``cache_dir`` is non-empty on COLD runs too — the cold run is the one that
    populates the cache — so presence of a cache dir says nothing about
    warmness. Only the entry-count delta does.
    """
    cold = probe.cache_state_from(0, 1, "/tmp/c")
    warm = probe.cache_state_from(1, 1, "/tmp/c")
    assert cold != warm
    assert (cold, warm) == ("cold", "warm")


def test_entry_count_counts_files_recursively(tmp_path):
    assert probe.cache_entry_count(tmp_path) == 0
    (tmp_path / "a").write_text("x")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "b").write_text("y")
    assert probe.cache_entry_count(tmp_path) == 2


def test_entry_count_tolerates_a_missing_or_unset_dir(tmp_path):
    assert probe.cache_entry_count(None) == 0
    assert probe.cache_entry_count(tmp_path / "does_not_exist") == 0


def test_a_cold_then_warm_pair_yields_exactly_one_of_each(tmp_path):
    """The acceptance shape: one invocation each, no tag parsing anywhere."""
    cache = tmp_path / "jax_cache"
    cache.mkdir()

    before = probe.cache_entry_count(cache)
    (cache / "entry-0").write_text("compiled")  # stands in for the XLA write
    first = probe.cache_state_from(before, probe.cache_entry_count(cache), cache)

    before = probe.cache_entry_count(cache)
    second = probe.cache_state_from(before, probe.cache_entry_count(cache), cache)

    assert (first, second) == ("cold", "warm")


def test_host_state_records_the_load_provenance():
    """Compile happens on host cores, so load is not bookkeeping."""
    hs = probe.host_state()
    assert set(hs) == {"cpu_count", "load_avg_1m"}
    assert hs["cpu_count"] is None or hs["cpu_count"] >= 1
