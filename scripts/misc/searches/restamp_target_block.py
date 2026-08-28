"""Recompute a recorded results JSON's schema-v2 ``target`` block in place.

Why this exists (issue #182). ``_targets._positions_block`` used to build a
target's ``positions`` block from the module DEFAULTS (``fixed`` / ``0.3`` /
``1e8``) because ``Target.positions`` is only ``"off"``/``"on"`` — the arm's
threshold mode/value and factor live in ``SEARCHES_POSITIONS_THRESHOLD`` /
``SEARCHES_POSITIONS_FACTOR``. So the Phase-4 diagnostic arms
``pos_t0.3_f1e8``, ``pos_t0.3_f1e5`` and ``pos_tauto0.2_f1e8`` — three
genuinely different objectives, correctly given three distinct output
directories by ``_setup.positions_arm_tag()`` — were all stamped with the
SAME ``target_id``. ``_positions_block`` now takes the resolved positions
block as an argument; this script re-derives affected rows' ``target`` blocks
with the corrected function so no hash is ever hand-edited.

It is a **re-derivation, not a patch**: every field of the new block comes
from ``_targets.target_block`` applied to the row's own recorded
configuration (model type, instrument, precision, and the resolved
``positions`` block the run wrote into the artifact). Nothing is copied from
the old block, and a row whose configuration has not changed meaning
re-stamps to the byte-identical id it already carried — which is the control
this script prints for every file, and the reason ``--write`` is opt-in.

``priors_ref`` is expected to change on every re-stamp: it is the content
hash of ``_targets.py`` itself, and ``_targets.py`` is what changed.

Usage (from the ``autolens_profiling/`` root)::

    # report only (default) — old vs new id for every path
    python3 scripts/misc/searches/restamp_target_block.py \\
        results/searches/multi_start_prodigy_autoconv/imaging/mge/hst/*_pos_*.json

    # write the new blocks back
    python3 scripts/misc/searches/restamp_target_block.py --write <paths...>

    # CI-style: exit non-zero if any path's recorded id is stale
    python3 scripts/misc/searches/restamp_target_block.py --check <paths...>

Rows this script REFUSES rather than guesses at:

- a row with no schema-v2 ``target`` block, or one whose ``target_id`` is
  ``null`` (a cell the Phase-1 registry does not cover);
- a row carrying a ``target_override`` — the model it ran was mutated by a
  diagnostic prior override, so the registry model this script rebuilds is
  not the model that was hashed;
- a cell outside the registry's ``imaging``/``hst`` coverage.

Each refusal is printed with its reason and counted; none is silent.
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
for _p in (str(_ROOT), str(_ROOT / "scripts" / "misc")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

from searches import _setup, _targets  # noqa: E402

_MODEL_CACHE: dict[str, object] = {}


class Refusal(Exception):
    """This row cannot be re-stamped from its own record."""


def _target_for_payload(payload: dict):
    """The registry ``Target`` a recorded row corresponds to.

    Mirrors ``_runner._target_for_cell`` exactly — same registry-key
    convention, same ``imaging``/``hst``-only coverage — so a re-stamp
    reproduces the id the runner would have written, not a near-miss.
    """
    dataset_class = payload.get("dataset_class")
    instrument = payload.get("instrument")
    if dataset_class != "imaging" or instrument != "hst":
        raise Refusal(
            f"cell {dataset_class}/{payload.get('model')}/{instrument} is outside the "
            f"Phase-1 TARGETS registry (imaging/hst only)"
        )
    if payload.get("target_override") is not None:
        raise Refusal(
            "row carries a target_override (diagnostic prior arm): the model that was "
            "hashed is not the registry model this script would rebuild"
        )
    positions = payload.get("positions")
    if not isinstance(positions, dict) or "enabled" not in positions:
        raise Refusal("row has no top-level 'positions' block to re-derive the arm from")
    key = _targets._target_key(
        payload.get("model"),
        "on" if positions["enabled"] else "off",
        "mp" if payload.get("use_mixed_precision") else "fp64",
    )
    target = _targets.TARGETS.get(key)
    if target is None:
        raise Refusal(f"no registry target for key {key!r}")
    return target, positions


def _model_for(target) -> object:
    if target.name not in _MODEL_CACHE:
        _, model, _ = _setup.build_for_cell(target=target, use_jax=False)
        _MODEL_CACHE[target.name] = model
    return _MODEL_CACHE[target.name]


def restamp_payload(payload: dict) -> dict:
    """The corrected ``target`` block for one already-loaded results payload.

    Gated by a **reproduction control**: this environment must first
    reproduce the id the row already carries, using the OLD (defaults-derived)
    positions block. If it cannot, the difference is not the #182 positions
    defect — it is this machine hashing different inputs (a regenerated
    ``lensed_source.fits`` adapt-image cache, drifted priors, different
    dataset files) — and overwriting the recorded id would launder a foreign
    measurement into a row measured elsewhere. Such a row is refused, not
    re-stamped.
    """
    old = payload.get("target")
    if not isinstance(old, dict) or old.get("target_id") is None:
        raise Refusal("row has no schema-v2 target block with a non-null target_id")
    target, positions = _target_for_payload(payload)
    model = _model_for(target)
    dataset_path = _ROOT / "dataset" / payload["dataset_class"] / payload["instrument"]

    block = _targets.target_block(target, model, dataset_path, positions)
    # Either the row still carries its pre-#182 id (this environment reproduces
    # the run's inputs and the only thing that moved is the positions block), or
    # it already carries the corrected one (a re-run of this script — idempotent).
    # Anything else means this machine is hashing different inputs.
    id_before_fix = _targets.target_id(target, model, dataset_path, None)
    if old["target_id"] not in (id_before_fix, block["target_id"]):
        raise Refusal(
            f"reproduction control failed: this environment computes {id_before_fix} the "
            f"OLD way and {block['target_id']} the corrected way, but the row carries "
            f"{old['target_id']} — neither. This machine does not hash the same inputs the "
            f"run did (adapt-image cache / dataset files / priors), so the id difference is "
            f"not the #182 positions defect and must not be re-stamped here."
        )
    return block


def restamp_file(path: Path, *, write: bool) -> dict:
    payload = json.loads(Path(path).read_text())
    new = restamp_payload(payload)
    old = payload["target"]
    changed = old.get("target_id") != new.get("target_id")
    if write:
        payload["target"] = new
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "path": str(path),
        "old_target_id": old.get("target_id"),
        "new_target_id": new.get("target_id"),
        "old_priors_ref": old.get("priors_ref"),
        "new_priors_ref": new.get("priors_ref"),
        "changed": changed,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", type=Path, help="results JSON paths")
    p.add_argument("--write", action="store_true", help="write the recomputed blocks back")
    p.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any path's recorded target_id is stale (implies no --write)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.check and args.write:
        raise SystemExit("--check and --write are mutually exclusive")

    changed, unchanged, refused = [], [], []
    for path in args.paths:
        try:
            record = restamp_file(path, write=args.write)
        except Refusal as exc:
            refused.append((path, str(exc)))
            print(f"  REFUSED  {path}\n           {exc}")
            continue
        (changed if record["changed"] else unchanged).append(record)
        marker = "CHANGED " if record["changed"] else "same    "
        print(f"  {marker} {path}")
        if record["changed"]:
            print(f"           {record['old_target_id']}  ->  {record['new_target_id']}")

    print(
        f"\n{len(changed)} changed, {len(unchanged)} unchanged, {len(refused)} refused "
        f"({'written' if args.write else 'dry run — pass --write to apply'})"
    )
    if args.check and changed:
        raise SystemExit(f"{len(changed)} row(s) carry a stale target_id — re-run with --write.")


if __name__ == "__main__":
    main()
