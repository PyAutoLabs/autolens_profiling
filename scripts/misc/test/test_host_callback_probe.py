"""Unit tests for ``likelihood_breakdown.host_callback_probe`` (#273).

**CPU only, no JAX, no GPU.** The probe re-implements the library's qhull
callback body so ``scipy.spatial.Delaunay`` can be timed apart from the
adjacency bookkeeping. A re-implementation that drifts from the library body
would report a plausible split of the wrong work, so the central test here is
array-for-array parity against the library function itself.

What a broken probe would look like, and what catches it:

- the wrapper returns *its own* triangulation rather than the library's
  (a different qhull option set, a different padding convention) —
  ``test__wrapper_returns_the_library_arrays_bit_for_bit``;
- the rebind never takes, so a cell reports ``0.0 ms`` of qhull as though it
  had measured it — ``test__probe_counts_every_call`` and
  ``test__probe_is_restored_on_exit``;
- the non-finite early return is lost, so a NaN lane raises out of a
  ``pure_callback`` and aborts the whole batch —
  ``test__non_finite_points_pass_through_without_qhull``.
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

from likelihood_breakdown import host_callback_probe as hcp  # noqa: E402

delaunay_mod = pytest.importorskip(
    "autoarray.inversion.mesh.interpolator.delaunay",
    reason="PyAutoArray is not importable in this environment",
)


def _points(n: int = 200, seed: int = 7) -> np.ndarray:
    """A random source-plane-like point cloud, float64 ``(n, 2)``."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, size=(n, 2))


def test__wrapper_returns_the_library_arrays_bit_for_bit():
    points = _points()
    expected = delaunay_mod.scipy_delaunay_tri_only(points)

    with hcp.qhull_probe() as probe:
        got = delaunay_mod.scipy_delaunay_tri_only(points)

    assert probe.calls == 1
    assert len(got) == len(expected) == 3
    for name, a, b in zip(
        ("simplices_padded", "simplex_neighbors", "vertex_simplex"), got, expected
    ):
        assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b), (
            f"{name}: the probe's re-implementation of the library body no longer "
            f"reproduces it — its qhull/tables split would describe different work"
        )


def test__probe_records_a_positive_qhull_and_tables_split():
    with hcp.qhull_probe() as probe:
        delaunay_mod.scipy_delaunay_tri_only(_points(400))

    (record,) = probe.records
    assert record.n_points == 400
    assert record.finite is True
    assert record.qhull_s > 0.0, "qhull was not timed — the split is meaningless"
    assert record.tables_s > 0.0, "the table building was not timed"
    summary = probe.summary()
    assert summary["calls"] == 1
    assert summary["qhull_ms_total"] == pytest.approx(record.qhull_s * 1e3)
    assert summary["non_finite_calls"] == 0
    assert summary["n_points"] == [400]


def test__probe_counts_every_call_and_reset_zeroes_the_window():
    """Per-traced-call counts are the evidence that vmap pays B serial trips."""
    points = _points(120)
    with hcp.qhull_probe() as probe:
        for _ in range(4):
            delaunay_mod.scipy_delaunay_tri_only(points)
        assert probe.calls == 4

        probe.reset()
        assert probe.calls == 0
        assert probe.summary()["calls"] == 0
        assert probe.summary()["qhull_ms_mean"] == 0.0

        for _ in range(3):
            delaunay_mod.scipy_delaunay_tri_only(points)
        assert probe.calls == 3


def test__non_finite_points_pass_through_without_qhull():
    """A NaN lane must not raise: a qhull exception aborts the whole vmap batch."""
    points = _points(50)
    points[3, 1] = np.nan

    expected = delaunay_mod.scipy_delaunay_tri_only(points)
    with hcp.qhull_probe() as probe:
        got = delaunay_mod.scipy_delaunay_tri_only(points)

    (record,) = probe.records
    assert record.finite is False
    assert record.qhull_s == 0.0
    for a, b in zip(got, expected):
        assert np.array_equal(a, b)
    assert all(np.all(a == -1) for a in got), (
        "a non-finite lane must come back as the -1 padding convention, untouched"
    )
    assert probe.summary()["non_finite_calls"] == 1


def test__pass_through_mode_calls_the_library_and_times_the_whole_body():
    points = _points(150)
    with hcp.qhull_probe(reimplement=False) as probe:
        got = delaunay_mod.scipy_delaunay_tri_only(points)

    (record,) = probe.records
    assert record.qhull_s == 0.0, "pass-through mode cannot see inside the body"
    assert record.tables_s > 0.0
    for a, b in zip(got, delaunay_mod.scipy_delaunay_tri_only(points)):
        assert np.array_equal(a, b)


def test__the_wrapper_keeps_the_library_functions_attribute_name():
    """``functools.wraps``, so anything reading ``__name__`` still sees the library's."""
    original = delaunay_mod.scipy_delaunay_tri_only
    for reimplement in (True, False):
        with hcp.qhull_probe(reimplement=reimplement):
            assert delaunay_mod.scipy_delaunay_tri_only.__name__ == original.__name__, (
                f"reimplement={reimplement}: the wrapper no longer carries the library "
                f"function's __name__"
            )


def test__the_probe_declares_the_host_event_names_it_substitutes():
    """The profiler names host events from the CODE OBJECT, not ``__name__``.

    So under the probe the xplane has no ``scipy_delaunay_tri_only`` event at
    all, and a cell matching phase 1's name reports 0.0 ms of qhull without
    saying anything. The published fragments are what a cell matches instead,
    so they must actually name this module's wrapper.
    """
    assert hcp.PROBE_HOST_EVENT_FRAGMENTS, "no substitute host-event names published"
    source_file = _Path(hcp.__file__).name
    assert source_file in hcp.PROBE_HOST_EVENT_FRAGMENTS, (
        f"the profiler's host events are named `$<file>:<line> <co_name>`, so "
        f"{source_file!r} must be one of the published fragments"
    )
    with hcp.qhull_probe() as probe:
        delaunay_mod.scipy_delaunay_tri_only(_points(40))
    assert probe.calls == 1
    co_names = {hcp._timed_tri_only.__code__.co_name}
    assert co_names & set(hcp.PROBE_HOST_EVENT_FRAGMENTS), (
        f"the timed body's code name {co_names} is not among the published fragments, "
        f"so a cell matching them would miss the event the profiler actually emits"
    )


def test__the_cell_matches_the_probes_host_event_names():
    """Or every batched leg reports `host_callback.qhull_ms: 0.0` in silence."""
    cell = (
        _profiling_root() / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_trace.py"
    ).read_text()
    assert "host_callback_probe.PROBE_HOST_EVENT_FRAGMENTS" in cell, (
        "fixed_light_trace.py no longer folds the probe's host-event names into its "
        "qhull matcher, so its xplane qhull row would read 0.0 ms under the probe"
    )


def test__probe_is_restored_on_exit_including_on_an_exception():
    original = delaunay_mod.scipy_delaunay_tri_only

    with hcp.qhull_probe():
        assert delaunay_mod.scipy_delaunay_tri_only is not original
    assert delaunay_mod.scipy_delaunay_tri_only is original

    with pytest.raises(RuntimeError, match="deliberate"):
        with hcp.qhull_probe():
            raise RuntimeError("deliberate")
    assert delaunay_mod.scipy_delaunay_tri_only is original


def test__the_library_callback_resolves_the_module_global_at_call_time():
    """The rebind takes because ``_jax_delaunay_tables`` looks the name up late.

    If the callback ever captures the function at lowering time instead (a
    default argument, a module-level alias, ``functools.partial``), a probe
    entered after lowering would silently measure nothing — so this reads the
    source rather than trusting the docstring.
    """
    import inspect

    source = inspect.getsource(delaunay_mod._jax_delaunay_tables)
    assert "lambda pts: scipy_delaunay_tri_only(" in source, (
        "the qhull callback no longer resolves scipy_delaunay_tri_only as a module "
        "global inside a lambda — host_callback_probe's rebind may no longer be seen "
        "by an already-lowered program"
    )


def test__the_dotted_name_the_probe_rebinds_is_the_one_that_exists():
    module_path, _, attribute = hcp.LIBRARY_QHULL_DOTTED.rpartition(".")
    import importlib

    module = importlib.import_module(module_path)
    assert hasattr(module, attribute), (
        f"{hcp.LIBRARY_QHULL_DOTTED} does not exist — the recorded provenance "
        f"would name something the probe never patched"
    )
