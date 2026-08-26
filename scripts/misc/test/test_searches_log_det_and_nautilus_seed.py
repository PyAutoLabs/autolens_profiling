"""W2/W8: Nautilus seed override, the GPU gradient-cell slogdet default, and
the ``log_det_method`` arm tag (#175).

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_log_det_and_nautilus_seed.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

_MISC = Path(__file__).resolve().parents[1]
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from searches import _runner  # noqa: E402
from searches._samplers import (  # noqa: E402
    assert_disjoint_output_paths,
    build_multi_start,
    multi_start_unique_tag,
    nautilus_seed,
)


def _clear_arm_env(monkeypatch):
    """Clear every other arm knob so only ``log_det`` varies across a test."""
    for var in (
        "SEARCHES_SEED",
        "SEARCHES_POSITIONS",
        "SEARCHES_BIJECTOR",
        "SEARCHES_LOG_DET_METHOD",
    ):
        monkeypatch.delenv(var, raising=False)


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


# ---------------------------------------------------------------------------
# unique_tag composition — the #175 correctness guard
# ---------------------------------------------------------------------------


def test_unique_tag_unset_log_det_matches_pre_fix_tag(monkeypatch):
    """The byte-identity control. Every arm that never set
    ``SEARCHES_LOG_DET_METHOD`` — which is every arm recorded before #175, and
    every arm running on the W8-resolved default — must keep EXACTLY its
    pre-#175 tag, or its recorded output path moves."""
    _clear_arm_env(monkeypatch)
    assert multi_start_unique_tag("imaging", "knn") is None

    monkeypatch.setenv("SEARCHES_SEED", "3")
    monkeypatch.setenv("SEARCHES_N_STARTS", "16")
    monkeypatch.setenv("SEARCHES_N_STEPS", "3000")
    assert multi_start_unique_tag("imaging", "delaunay_adapt_split") == "n16_s3000_seed3"


def test_unique_tag_cholesky_differs_from_slogdet(monkeypatch):
    """RAL job 340576 in one assertion: two arms differing only in
    ``log_det_method`` must not resolve to the same tag."""
    _clear_arm_env(monkeypatch)
    monkeypatch.setenv("SEARCHES_SEED", "3")
    monkeypatch.setenv("SEARCHES_N_STARTS", "16")
    monkeypatch.setenv("SEARCHES_N_STEPS", "3000")

    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    tag_cholesky = multi_start_unique_tag("imaging", "delaunay_adapt_split")
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    tag_slogdet = multi_start_unique_tag("imaging", "delaunay_adapt_split")

    assert tag_cholesky == "n16_s3000_seed3_ld_cholesky"
    assert tag_slogdet == "n16_s3000_seed3_ld_slogdet"
    assert tag_cholesky != tag_slogdet


def test_unique_tag_log_det_is_normalised(monkeypatch):
    """Case/whitespace must not split one arm across two output paths —
    ``_runner.resolve_log_det_method`` normalises the same way."""
    _clear_arm_env(monkeypatch)
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "  SLOGDET ")
    assert multi_start_unique_tag("imaging", "knn") == "ld_slogdet"


def test_unique_tag_log_det_alone_tags_an_unseeded_cell(monkeypatch):
    """An unseeded, positions-off, none-bijector arm that DID set an explicit
    override still gets a tag — it must never share the un-overridden cell's
    output path."""
    _clear_arm_env(monkeypatch)
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    assert multi_start_unique_tag("imaging", "knn") == "ld_cholesky"


def test_multi_start_search_output_path_and_identifier_differ_cholesky_vs_slogdet(
    monkeypatch,
):
    """The load-bearing regression this composition exists to prevent: a
    slogdet arm must NOT resume the cholesky arm's ``.completed`` fit."""
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")
    _clear_arm_env(monkeypatch)

    kwargs = dict(
        sampler="multi_start_prodigy",
        dataset_class="imaging",
        model_type="knn",
        instrument="hst",
        config_name="unit_test",
        use_jax=False,
    )
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "cholesky")
    search_cholesky = build_multi_start(**kwargs)
    monkeypatch.setenv("SEARCHES_LOG_DET_METHOD", "slogdet")
    search_slogdet = build_multi_start(**kwargs)

    assert_disjoint_output_paths(search_cholesky, search_slogdet)
