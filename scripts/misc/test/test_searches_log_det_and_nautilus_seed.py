"""W2/W8: Nautilus seed override and the GPU gradient-cell slogdet default."""

from __future__ import annotations

import sys
from pathlib import Path

_MISC = Path(__file__).resolve().parents[1]
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from searches import _runner  # noqa: E402
from searches._samplers import nautilus_seed  # noqa: E402


def test__nautilus_seed_default_none_and_env(monkeypatch):
    monkeypatch.delenv("SEARCHES_NAUTILUS_SEED", raising=False)
    assert nautilus_seed() is None
    monkeypatch.setenv("SEARCHES_NAUTILUS_SEED", "7")
    assert nautilus_seed() == 7


def test__log_det_default_is_slogdet_only_for_gpu_gradient_pixelized(monkeypatch):
    monkeypatch.delenv("SEARCHES_LOG_DET_METHOD", raising=False)
    monkeypatch.setattr(_runner, "_jax_backend_is_gpu", lambda: True)
    kw = dict(dataset_class="imaging", use_jax=True)
    assert (
        _runner.resolve_log_det_method(
            sampler="multi_start_prodigy_autoconv", model_type="delaunay", **kw
        )
        == "slogdet"
    )
    assert _runner.resolve_log_det_method(sampler="nuts", model_type="knn", **kw) == "slogdet"
    # nested samplers keep the packaged default so truth bars do not move
    assert _runner.resolve_log_det_method(sampler="nautilus", model_type="delaunay", **kw) is None
    assert _runner.resolve_log_det_method(sampler="nss", model_type="delaunay", **kw) is None
    # parametric cells have no inversion log-det
    assert (
        _runner.resolve_log_det_method(
            sampler="multi_start_prodigy_autoconv", model_type="mge", **kw
        )
        is None
    )
    # numpy path
    assert (
        _runner.resolve_log_det_method(
            sampler="multi_start_prodigy_autoconv",
            model_type="delaunay",
            dataset_class="imaging",
            use_jax=False,
        )
        is None
    )


def test__log_det_default_stays_cholesky_on_cpu(monkeypatch):
    monkeypatch.delenv("SEARCHES_LOG_DET_METHOD", raising=False)
    monkeypatch.setattr(_runner, "_jax_backend_is_gpu", lambda: False)
    assert (
        _runner.resolve_log_det_method(
            sampler="multi_start_prodigy_autoconv",
            dataset_class="imaging",
            model_type="delaunay",
            use_jax=True,
        )
        is None
    )


def test__log_det_env_override_wins(monkeypatch):
    monkeypatch.setattr(_runner, "_jax_backend_is_gpu", lambda: True)
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    assert (
        _runner.resolve_log_det_method(
            sampler="multi_start_prodigy_autoconv",
            dataset_class="imaging",
            model_type="delaunay",
            use_jax=True,
        )
        == "cholesky"
    )
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    assert (
        _runner.resolve_log_det_method(
            sampler="nautilus", dataset_class="imaging", model_type="mge", use_jax=True
        )
        == "slogdet"
    )
