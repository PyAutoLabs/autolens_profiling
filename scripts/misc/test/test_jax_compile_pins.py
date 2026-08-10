"""Unit tests for warm-compile pin derivation (``jax_compile/update_pins.py``).

The pins are the regression surface: a cache or autotune setting that stops
applying shows up as a pinned warm compile moving. What matters most here is
what the pins REFUSE to do — pool measurements that are not comparable.

No JAX dependency. Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_jax_compile_pins.py
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

import pins as pins_mod  # noqa: E402
import update_pins  # noqa: E402


def _rec(**kw):
    base = {
        "transform": "vag",
        "compile_s": 2.0,
        "dataset_class": "imaging",
        "model_type": "mge",
        "instrument": "hst",
        "hardware": "local_cpu",
        "hostname": "laptop",
        "jax_version": "0.10.2",
        "mixed_precision": False,
        "cache_state": "warm",
        "tag": "t",
        "timestamp": "2026-07-01T00:00:00",
    }
    base.update(kw)
    return base


def test_only_warm_rows_are_pinnable():
    """A cold row is the cost the cache exists to remove — pinning it would
    enshrine exactly what this arc watches for."""
    derived = update_pins.derive(
        [
            _rec(cache_state="cold", compile_s=117.0),
            _rec(cache_state="none", compile_s=117.0),
            _rec(cache_state="unknown", compile_s=117.0),
            _rec(cache_state="warm", compile_s=2.3),
        ]
    )
    assert len(derived) == 1
    assert derived[0]["compile_s"] == 2.3


def test_hostname_splits_the_key_within_one_hardware_label():
    """The measured hazard: `hardware` is only ever local_cpu / local_gpu_<dev>,
    so one label spans a laptop AND a 32-core RAL node. Pooling them compares
    machines whose compile times differ by the factor the pins exist to keep
    apart."""
    derived = update_pins.derive(
        [
            _rec(hostname="laptop", compile_s=4.3),
            _rec(hostname="euclid-ral-compute-22", compile_s=1.8),
        ]
    )
    assert len(derived) == 2
    assert {p["compile_s"] for p in derived} == {4.3, 1.8}


def test_each_comparability_field_splits_the_key():
    base = _rec()
    for field, other in (
        ("hardware", "local_gpu_X"),
        ("hostname", "other-host"),
        ("jax_version", "0.11.0"),
        ("mixed_precision", True),
    ):
        derived = update_pins.derive([base, _rec(**{field: other})])
        assert len(derived) == 2, f"{field} must split the pin key"


def test_most_recent_warm_row_wins():
    """A pin states what warm costs NOW; averaging would blend a pre- and
    post-regression world into one number that describes neither."""
    derived = update_pins.derive(
        [
            _rec(compile_s=2.3, timestamp="2026-07-01T00:00:00", tag="old"),
            _rec(compile_s=9.9, timestamp="2026-08-01T00:00:00", tag="new"),
            _rec(compile_s=5.5, timestamp="2026-07-15T00:00:00", tag="mid"),
        ]
    )
    assert len(derived) == 1
    assert derived[0]["compile_s"] == 9.9
    assert derived[0]["source_tag"] == "new"


def test_pin_carries_its_provenance():
    derived = update_pins.derive([_rec(tag="census-warm", timestamp="2026-07-28T16:04:53")])
    pin = derived[0]
    assert pin["source_tag"] == "census-warm"
    assert pin["source_timestamp"] == "2026-07-28T16:04:53"


def test_incomplete_records_are_not_pinned():
    assert update_pins.derive([_rec(instrument=None)]) == []
    assert update_pins.derive([_rec(hostname=None)]) == []


def test_derivation_is_deterministic():
    records = [
        _rec(transform="vag"),
        _rec(transform="jit"),
        _rec(hostname="b"),
        _rec(model_type="pixelization"),
    ]
    assert update_pins.derive(records) == update_pins.derive(list(reversed(records)))


def test_key_str_names_the_host_not_just_the_hardware():
    key = pins_mod.pin_key(_rec(hostname="euclid-ral-compute-22"))
    assert "euclid-ral-compute-22" in pins_mod.key_str(key)


def test_committed_pins_are_current():
    """`update_pins.py --check` is the gate; this catches a corpus edit that
    lands without re-deriving."""
    assert update_pins.derive(update_pins.corpus()) == pins_mod.load()
