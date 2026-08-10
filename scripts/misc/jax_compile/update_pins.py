"""Derive ``pins.json`` — the expected warm compile per cell/transform.

Only ``cache_state == "warm"`` rows are pinnable. A cold row is the compile the
cache exists to remove, so pinning one would enshrine the cost this arc is
watching for; ``none`` and ``unknown`` rows cannot be placed on either side of
that line at all.

Pins are **sticky**. A new key is pinned from its most recent warm row, but an
existing pin is never silently moved — ``--repin`` is required. This is the
whole point: if the pin auto-followed the newest measurement, then running this
script after a cache regression would quietly BAKE THE REGRESSION IN, and the
surveillance would report all-clear forever. A pin is a committed expectation,
the compile analogue of the runtime results' ``pinned_expected``.

Averaging is likewise avoided: it would blend a pre- and post-regression world
into a number describing neither. The source row's tag and timestamp travel with
the pin so any number traces back to the measurement that set it.

Run from the autolens_profiling root::

    python scripts/misc/jax_compile/update_pins.py --check   # report, write nothing
    python scripts/misc/jax_compile/update_pins.py --write    # add pins for NEW keys
    python scripts/misc/jax_compile/update_pins.py --write --repin  # also move existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pins import PIN_FIELDS, PINS_PATH, as_map, key_str, load, pin_key  # noqa: E402

RESULTS = _HERE / "results"


def corpus() -> list[dict]:
    """Every compile-probe record. Sibling instruments are skipped by schema."""
    out: list[dict] = []
    if not RESULTS.is_dir():
        return out
    for path in sorted(RESULTS.glob("*/*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            # export_probe.py / trace_profile.py share this tree with a
            # different schema and describe no compile of their own.
            if isinstance(rec, dict) and "compile_s" in rec:
                out.append(rec)
    return out


def derive(
    records: list[dict], existing: list[dict] | None = None, repin: bool = False
) -> list[dict]:
    """Pins for the corpus. Existing pin VALUES are preserved unless `repin`."""
    best: dict[tuple, dict] = {}
    for rec in records:
        if rec.get("cache_state") != "warm":
            continue
        if any(rec.get(f) in (None, "") for f in PIN_FIELDS if f != "mixed_precision"):
            continue
        key = pin_key(rec)
        prev = best.get(key)
        if prev is None or str(rec.get("timestamp") or "") >= str(prev.get("timestamp") or ""):
            best[key] = rec

    pins = [
        {
            **{f: rec.get(f) for f in PIN_FIELDS},
            "compile_s": rec.get("compile_s"),
            "source_tag": rec.get("tag"),
            "source_timestamp": rec.get("timestamp"),
        }
        for rec in best.values()
    ]
    if existing and not repin:
        # Sticky: a key that is already pinned keeps its committed value, so a
        # regression cannot be absorbed by re-running this script.
        old = as_map(existing)
        pins = [old[pin_key(p)] if pin_key(p) in old else p for p in pins]

    # Deterministic order so re-deriving an unchanged corpus is a no-op diff.
    pins.sort(key=lambda p: tuple(str(p.get(f)) for f in PIN_FIELDS))
    return pins


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report; exit 1 if pins would change")
    g.add_argument("--write", action="store_true", help="write pins.json")
    ap.add_argument(
        "--repin",
        action="store_true",
        help="also MOVE existing pins to the latest warm row (deliberate act: "
        "this is how a confirmed-benign change is accepted, and how a "
        "regression would be baked in if used carelessly)",
    )
    a = ap.parse_args(argv)

    records = corpus()
    existing = load()
    pins = derive(records, existing=existing, repin=a.repin)

    new_map, old_map = as_map(pins), as_map(existing)
    added = [k for k in new_map if k not in old_map]
    removed = [k for k in old_map if k not in new_map]
    changed = [
        k
        for k in new_map
        if k in old_map and new_map[k].get("compile_s") != old_map[k].get("compile_s")
    ]

    warm = sum(1 for r in records if r.get("cache_state") == "warm")
    print(f"compile pins: {len(pins)} from {warm} warm record(s) of {len(records)} total")
    if changed and not a.repin:
        print("  (existing pins are sticky — pass --repin to move them)")
    for label, keys in (("added", added), ("changed", changed), ("removed", removed)):
        for k in keys:
            print(f"  {label}: {key_str(k)}")

    if not (added or removed or changed):
        print("  pins.json is current")
        return 0
    if a.check:
        print("ERROR: pins.json is stale — run with --write and commit.", file=sys.stderr)
        return 1

    PINS_PATH.write_text(json.dumps({"schema": 1, "pins": pins}, indent=2) + "\n")
    print(f"  wrote {PINS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
