"""Warm-compile pins — the compile-axis equivalent of runtime's ``pinned_expected``.

A pin is "what a warm compile of this cell/transform costs here". It exists so a
**cache regression** is detectable: the persistent compilation cache turned
117.0 s into 2.3 s (CPU MGE ``vag``) and 5517.8 s into 937.1 s (A100 end-to-end),
and both wins are *settings* that can silently stop applying.

The comparability key
---------------------

A pin means nothing outside one ``(hardware, jax_version, mixed_precision,
cache_state)``. Compile timings are host-load-sensitive — README.md records the
first measurements being wrong by up to **7x** (851 s vs 117 s for the same
compile) purely from host load, because XLA compiles on the host cores — and the
corpus already mixes a 32-core RAL allocation with laptop rows.

So the key is part of the pin's identity, not metadata attached to it. Comparing
across it produces confident nonsense: a cold row against a warm pin, an A100 row
against a CPU pin, or rows either side of a jax version bump (cache keys include
the jax version, so a bump recompiles **once by design**).

Storage
-------

``pins.json`` beside this module, written by ``update_pins.py``. JSON rather than
a python module because it is generated data that wants clean diffs; the
consumers (``build_readme.py``, PyAutoBrain's profiling conductor) both read it
without importing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

PINS_PATH = Path(__file__).resolve().parent / "pins.json"

# The fields a pin is only comparable within. Order is fixed so the derived key
# string is stable across runs and processes.
#
# `hostname` is in here for a measured reason, not for completeness: `hardware`
# is only ever `local_cpu` / `local_gpu_<device>`, so a single `local_cpu` label
# currently spans a laptop (66 records) AND a 32-core RAL node (12). Without
# hostname the key silently pools two machines whose compile times differ by the
# very factor this module exists to keep apart.
COMPARABILITY_FIELDS = (
    "hardware",
    "hostname",
    "jax_version",
    "mixed_precision",
    "cache_state",
)

# The fields identifying WHAT was measured, inside one comparability key.
CELL_FIELDS = ("dataset_class", "model_type", "instrument", "transform")

PIN_FIELDS = COMPARABILITY_FIELDS + CELL_FIELDS


def comparability_key(rec: dict) -> tuple:
    return tuple(rec.get(f) for f in COMPARABILITY_FIELDS)


def pin_key(rec: dict) -> tuple:
    """Full identity of a pin: comparability key + what was measured."""
    return tuple(rec.get(f) for f in PIN_FIELDS)


def key_str(key: tuple) -> str:
    """Human-readable rendering, e.g. for dashboards and agent output."""
    parts = dict(zip(PIN_FIELDS, key))
    cell = "/".join(
        str(parts[f]) for f in ("dataset_class", "model_type", "instrument") if parts.get(f)
    )
    return (
        f"{cell} [{parts.get('transform')}] "
        f"@ {parts.get('hardware')}/{parts.get('hostname')} jax{parts.get('jax_version')}"
        f"{' mp' if parts.get('mixed_precision') else ''} {parts.get('cache_state')}"
    )


def load(path: Path | None = None) -> list[dict]:
    p = path or PINS_PATH
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    pins = data.get("pins") if isinstance(data, dict) else data
    return pins if isinstance(pins, list) else []


def as_map(pins: list[dict]) -> dict[tuple, dict]:
    return {pin_key(p): p for p in pins}
