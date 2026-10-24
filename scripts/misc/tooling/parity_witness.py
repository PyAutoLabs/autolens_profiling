"""The ``parity.json`` witness convention and its validator (stdlib only).

A parity witness records whether one mirrored cell's log-likelihood agrees with a reference
log-likelihood for the same dataset and model. It sits beside the cell's measurements:

    results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]/parity.json

(older witnesses under the cell's ``history/<release_pin>_<YYYY-MM-DD>/``). The day-zero rule
(lifecycle report §3.6) wants a parity witness on every day-zero record; ``build_catalogue.py``
turns each witness into a ``profiling-summary@3`` record with ``measurement: parity`` and
``coverage_matrix.py`` reports which planned cells have one. See results/README.md "Parity
witness" for the field reference.

Run ``python scripts/misc/tooling/parity_witness.py --check`` to validate every
``results/**/parity.json``. A witness is written by the run that measured it; this module never
computes a log-likelihood and never fills one in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

SCHEMA = "parity-witness"
VERSION = 1
STATUSES = ("match", "diverged", "not_assessed")
#: The v3 record's headline metric and its companion value.
METRIC = "abs_delta_log_likelihood"
NUMBERS = ("log_likelihood", "reference_log_likelihood", "tolerance")


def _finite(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _utc(value):
    if not _text(value) or not (value.endswith("Z") or value.endswith("+00:00")):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def delta(doc: dict) -> float:
    """|log_likelihood - reference_log_likelihood| of a valid assessed witness."""
    return abs(doc["log_likelihood"] - doc["reference_log_likelihood"])


def validate(doc) -> list[str]:
    """Every problem with one witness; an empty list means valid."""
    if not isinstance(doc, dict):
        return ["parity witness must be a JSON object"]
    errors = []
    if doc.get("schema") != SCHEMA or doc.get("version") != VERSION:
        errors.append(f"schema/version must be {SCHEMA!r}/{VERSION}")
    status = doc.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {', '.join(STATUSES)}")
    reference = doc.get("reference")
    if not isinstance(reference, dict) or not _text(reference.get("source")):
        errors.append("reference.source must name where the reference log-likelihood came from")
    else:
        try:
            date.fromisoformat(reference.get("date") or "")
        except (TypeError, ValueError):
            errors.append("reference.date must be an ISO date (YYYY-MM-DD)")
    if not _utc(doc.get("measured_at")):
        errors.append("measured_at must be an ISO-8601 UTC timestamp")
    if not _text(doc.get("autolens_version")):
        errors.append("autolens_version (the release pin the witness ran on) is required")
    pixels = doc.get("source_pixels")
    if pixels is not None and not (
        isinstance(pixels, int) and not isinstance(pixels, bool) and pixels > 0
    ):
        errors.append("source_pixels must be a positive integer when present")
    if status == "not_assessed":
        if not _text(doc.get("reason")):
            errors.append("a not_assessed witness needs a reason")
        for key in NUMBERS:
            if doc.get(key) is not None and not _finite(doc[key]):
                errors.append(f"{key} must be finite or null")
    elif status in ("match", "diverged"):
        for key in NUMBERS:
            if not _finite(doc.get(key)):
                errors.append(f"{key} must be a finite number for status {status}")
        if _finite(doc.get("tolerance")) and doc["tolerance"] < 0:
            errors.append("tolerance must be non-negative")
        if not errors:
            within = delta(doc) <= doc["tolerance"]
            if within != (status == "match"):
                errors.append(
                    f"status {status} disagrees with |delta log L| = {delta(doc):.6g} vs "
                    f"tolerance {doc['tolerance']:.6g}"
                )
    return errors


def witnesses(root: Path) -> list[Path]:
    return sorted((root / "results").rglob("parity.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--check", action="store_true", help="validate every results/**/parity.json"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    bad = 0
    found = witnesses(root)
    for path in found:
        try:
            errors = validate(json.loads(path.read_text()))
        except ValueError as exc:
            errors = [f"not JSON: {exc}"]
        for error in errors:
            print(f"parity: {path.relative_to(root).as_posix()}: {error}")
        bad += bool(errors)
    if bad:
        print(f"parity: FAILED ({bad} of {len(found)} witnesses invalid)")
        return 1
    print(f"parity: {len(found)} witness(es) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
