"""Unit tests for the access-counting harness (``call_accounting.py``).

**No imaging, no JAX.** The harness is the instrument the numba CPU
decomposition is read through, so it is pinned on synthetic classes where the
right answer is known by construction rather than measured: the exclusive times
must be additive, a cached site must be seen exactly once, a plain ``property``
must be seen on every access, and ``uninstall`` must put back the *same object*
it took.

The last one is the quiet failure mode. A harness that restores an equal-but-not-
identical descriptor leaves a permanent wrapper on a library class for the rest
of the process, and every subsequent timing in that process is measured through
it — so the test asserts ``cls.__dict__[name] is original``, not equality.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_call_accounting.py
"""

from __future__ import annotations

import functools
import sys as _sys
import time
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

from likelihood_breakdown import call_accounting as ca  # noqa: E402

BURN = 0.004


def _burn(seconds: float = BURN) -> float:
    """Spend ``seconds`` of wall clock so a timing is well above the clock floor."""
    end = time.perf_counter() + seconds
    total = 0.0
    while time.perf_counter() < end:
        total += 1.0
    return total


@pytest.fixture(autouse=True)
def _always_uninstall():
    """No test may leave a wrapper behind on a class another test reads."""
    yield
    ca.uninstall()


# ---------------------------------------------------------------------------
# A synthetic nested-property class — the arithmetic, with a known answer
# ---------------------------------------------------------------------------


class Nested:
    """``root -> (branch -> leaf, leaf)``, three plain properties, all timed."""

    @property
    def leaf(self):
        return _burn()

    @property
    def branch(self):
        _burn()
        return self.leaf

    @property
    def root(self):
        _burn()
        return self.branch + self.leaf


def _nested_spec():
    return [
        ca.descriptor_site("root", Nested, "root"),
        ca.descriptor_site("branch", Nested, "branch"),
        ca.descriptor_site("leaf", Nested, "leaf"),
    ]


def test_exclusive_times_sum_to_the_top_level_inclusive_time():
    """``sum(excl_s) == incl_s`` of the root — the identity the coverage gate rests on.

    The unattributed fraction the cell reports is ``call - sum(excl_s)``. If the
    exclusive times were not additive by construction that number would be an
    artefact of the harness rather than a measurement of the library.
    """
    ca.install(_nested_spec())
    _ = Nested().root
    snap = ca.snapshot()
    ca.uninstall()

    total_exclusive = sum(entry["excl_s"] for entry in snap.values())
    assert abs(total_exclusive - snap["root"]["incl_s"]) <= 1e-9

    # ...and it is a genuine decomposition, not one site doing all the work.
    assert snap["leaf"]["excl_s"] > 0.0
    assert snap["branch"]["excl_s"] > 0.0
    assert snap["root"]["excl_s"] > 0.0
    assert snap["root"]["excl_s"] < snap["root"]["incl_s"]


def test_a_plain_property_is_seen_on_every_access():
    """A data descriptor has no instance-dict shortcut: three accesses, three calls."""
    ca.install(_nested_spec())
    obj = Nested()
    _ = obj.leaf
    _ = obj.leaf
    _ = obj.leaf
    snap = ca.snapshot()
    ca.uninstall()

    assert snap["leaf"]["n_calls"] == 3
    assert snap["root"]["n_calls"] == 0
    # A site that was never reached is reported with zeros, not absent.
    assert snap["root"]["incl_s"] == 0.0


def test_functools_cached_property_is_seen_exactly_once():
    class WithFunctoolsCache:
        @functools.cached_property
        def value(self):
            return _burn()

    ca.install([ca.descriptor_site("value", WithFunctoolsCache, "value", cached=True)])
    obj = WithFunctoolsCache()
    first = obj.value
    assert obj.value == first
    assert obj.value == first
    snap = ca.snapshot()
    ca.uninstall()

    assert snap["value"]["n_calls"] == 1
    # The library's caching still works through the wrapper: the value is in
    # the instance dict, which is why access two and three never reach us.
    assert "value" in obj.__dict__
    assert "value" in ca.declared_cached_labels()


def test_autonerves_cached_property_is_seen_exactly_once():
    """The hand-rolled ``CachedProperty`` is a different class with the same contract."""
    autonerves = pytest.importorskip("autonerves")

    class WithNervesCache:
        @autonerves.cached_property
        def value(self):
            return _burn()

    assert autonerves.cached_property is not functools.cached_property

    ca.install([ca.descriptor_site("value", WithNervesCache, "value", cached=True)])
    obj = WithNervesCache()
    _ = obj.value
    _ = obj.value
    _ = obj.value
    snap = ca.snapshot()
    ca.uninstall()

    assert snap["value"]["n_calls"] == 1
    assert "value" in obj.__dict__


def test_re_entrancy_does_not_double_count():
    """A site beneath itself adds its inclusive time once, and stays additive.

    ``incl_s`` accrues only at the outermost frame of a recursive site, so an
    ``n``-deep recursion cannot report ``n`` times its own elapsed time; the
    exclusive sum is unaffected because each frame still hands its whole
    inclusive time to its parent.
    """
    import types

    module = types.ModuleType("_call_accounting_recursion_fixture")

    def countdown(n):
        _burn(0.002)
        if n == 0:
            return 0
        return module.countdown(n - 1)

    module.countdown = countdown

    ca.install([ca.function_site("countdown", module, "countdown")])
    started = time.perf_counter()
    module.countdown(4)
    wall = time.perf_counter() - started
    snap = ca.snapshot()
    ca.uninstall()

    assert snap["countdown"]["n_calls"] == 5
    assert snap["countdown"]["incl_s"] <= wall + 1e-9
    assert abs(snap["countdown"]["excl_s"] - snap["countdown"]["incl_s"]) <= 1e-9
    assert module.countdown is countdown


def test_uninstall_restores_the_descriptor_by_identity():
    """Equality is not enough: the library class must hold the *same object* again."""
    original_root = Nested.__dict__["root"]
    original_leaf = Nested.__dict__["leaf"]

    ca.install(_nested_spec())
    assert Nested.__dict__["root"] is not original_root
    ca.uninstall()

    assert Nested.__dict__["root"] is original_root
    assert Nested.__dict__["leaf"] is original_leaf
    assert ca.installed_labels() == []


def test_double_install_raises():
    ca.install(_nested_spec())
    with pytest.raises(RuntimeError, match="already live"):
        ca.install(_nested_spec())
    ca.uninstall()
    assert Nested.__dict__["root"] is not None


def test_function_wrapping_round_trips_and_captures_stats():
    """A module attribute is rebound, timed, restored — and its ``stats`` dict kept.

    ``fnnls_cholesky`` fills a ``stats`` dict its caller owns, and
    ``reconstruction_positive_only_from`` adds two more keys to that same dict
    *after* the solver returns. The wrapper keeps the reference, not a copy, so
    the cell reads the finished dict rather than a half-written one.
    """
    import types

    module = types.ModuleType("_call_accounting_solver_fixture")

    def solve(a, b, stats=None):
        _burn()
        if stats is not None:
            stats["outer_iterations"] = 7
            stats["inner_iterations"] = 19
            stats["passive_set"] = [0, 2, 5]
            stats["n_passive"] = 3
            stats["warm_start_errors"] = 1
        return a + b

    module.solve = solve

    ca.install([ca.function_site("solve", module, "solve")])
    stats: dict = {}
    assert module.solve(2, 3, stats=stats) == 5
    # The caller's post-hoc keys, written after the wrapped call returned.
    stats["seed_source"] = "dense"
    stats["warm_start_fallback"] = False

    snap = ca.snapshot()
    captured = ca.captured_stats()
    ca.uninstall()

    assert module.solve is solve
    assert snap["solve"]["n_calls"] == 1
    assert snap["solve"]["incl_s"] > 0.0
    assert captured["solve"]["outer_iterations"] == 7
    assert captured["solve"]["inner_iterations"] == 19
    assert captured["solve"]["passive_set"] == [0, 2, 5]
    assert captured["solve"]["seed_source"] == "dense"
    assert captured["solve"]["warm_start_fallback"] is False


def test_a_missing_site_records_zeros_rather_than_vanishing():
    """A formalism-specific site must be a ``n_calls: 0`` row, not an absent label."""

    class Dense:
        @property
        def shared(self):
            return 1

    ca.install(
        [
            ca.descriptor_site("shared", Dense, "shared"),
            ca.descriptor_site("sparse_only", Dense, "psf_weighted_data", missing_ok=True),
        ]
    )
    _ = Dense().shared
    snap = ca.snapshot()
    ca.uninstall()

    assert snap["sparse_only"] == {"excl_s": 0.0, "incl_s": 0.0, "n_calls": 0}
    assert snap["shared"]["n_calls"] == 1
    assert "sparse_only" not in ca.installed_labels()


def test_a_missing_site_without_missing_ok_is_loud_and_leaves_nothing_behind():
    """A typo'd site name is an error, and a failed install uninstalls itself."""
    original_root = Nested.__dict__["root"]

    with pytest.raises(AttributeError, match="no descriptor"):
        ca.install(
            [
                ca.descriptor_site("root", Nested, "root"),
                ca.descriptor_site("typo", Nested, "not_a_property"),
            ]
        )

    assert Nested.__dict__["root"] is original_root
    assert ca.installed_labels() == []


def test_reset_zeroes_the_counters_without_uninstalling():
    ca.install(_nested_spec())
    obj = Nested()
    _ = obj.leaf
    assert ca.snapshot()["leaf"]["n_calls"] == 1

    ca.reset()
    assert ca.snapshot()["leaf"]["n_calls"] == 0
    assert ca.snapshot()["leaf"]["excl_s"] == 0.0

    _ = obj.leaf
    _ = obj.leaf
    snap = ca.snapshot()
    ca.uninstall()
    assert snap["leaf"]["n_calls"] == 2


def test_call_accounting_imports_without_jax():
    """The harness is stdlib + numpy: a cell that uses it imports no JAX runtime."""
    source = (_misc / "likelihood_breakdown" / "call_accounting.py").read_text()
    assert "import jax" not in source
