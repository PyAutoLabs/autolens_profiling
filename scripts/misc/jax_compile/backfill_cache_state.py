"""One-shot migration: add ``cache_state`` to compile records written before it.

``cache_state`` is derived from probe behaviour going forward (see
``probe.py::cache_state_from``). The records already committed predate that, so
this backfills them from the evidence that IS recoverable — and, crucially,
leaves the rest ``unknown`` rather than guessing.

What is exact:

* ``cache_dir == ""`` -> ``none``. The cache was not configured, full stop.

What is inferred, and only from an unambiguous tag:

* a tag whose cold/warm marker is unambiguous (``*-cold``, ``*-warm``,
  ``*-warm2``, ``*-warm-retry``) -> that state.

Everything else stays ``unknown``. The corpus carries ~40 ad-hoc tag spellings
and the whole point of ``cache_state`` is that tags cannot be trusted, so a
migration that tag-parsed aggressively would bake in exactly the error the
field exists to remove.

Run from the autolens_profiling root::

    python scripts/misc/jax_compile/backfill_cache_state.py --check   # report only
    python scripts/misc/jax_compile/backfill_cache_state.py --write
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# Anchored to the END of the tag so `mb_homo_cold_laxmap_gpu` does not match as
# a cold row on a substring that happens to appear mid-tag; an unanchored
# search is how tag parsing goes wrong in the first place.
_WARM_RE = re.compile(r"warm\d*(?:-retry)?$", re.I)
_COLD_RE = re.compile(r"cold$", re.I)


def state_for(rec: dict) -> str:
    if not rec.get("cache_dir"):
        return "none"
    tag = str(rec.get("tag") or "")
    if _WARM_RE.search(tag):
        return "warm"
    if _COLD_RE.search(tag):
        return "cold"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report the split, write nothing")
    g.add_argument("--write", action="store_true", help="apply the backfill in place")
    a = ap.parse_args()

    tally: Counter[str] = Counter()
    unknown_tags: Counter[str] = Counter()
    touched = 0

    for path in sorted(RESULTS.glob("*/*.json")):
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            continue
        changed = False
        for rec in records:
            if not isinstance(rec, dict) or "cache_state" in rec:
                continue
            # Sibling instruments (export_probe.py, trace_profile.py) share this
            # tree with a different schema; they have no compile of their own to
            # describe, so they are left alone.
            if "compile_s" not in rec:
                continue
            state = state_for(rec)
            rec["cache_state"] = state
            tally[state] += 1
            if state == "unknown":
                unknown_tags[str(rec.get("tag") or "")] += 1
            changed = True
        if changed and a.write:
            path.write_text(json.dumps(records, indent=2) + "\n")
            touched += 1

    print(f"cache_state backfill ({'WRITE' if a.write else 'CHECK'}):")
    for state, n in sorted(tally.items()):
        print(f"  {state:8s} {n}")
    if unknown_tags:
        print("  unknown, by tag (left unknown on purpose):")
        for tag, n in unknown_tags.most_common():
            print(f"    {tag!r}: {n}")
    if a.write:
        print(f"  rewrote {touched} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
