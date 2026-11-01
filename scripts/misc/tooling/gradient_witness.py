"""The ``gradient.json`` witness convention and its validator (stdlib only).

A gradient witness records, for one mirrored cell, what a gradient evaluation of the
likelihood costs beside the forward evaluation and whether the gradient is correct
(likelihood lifecycle phase 6, autolens_profiling#505). It sits beside the cell's other
measurements::

    results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]/gradient.json

It is written by ``scripts/misc/gradient/harness.py`` from a run that measured it;
``build_catalogue.py`` turns each witness into a ``profiling-summary@3`` record with
``measurement: gradient``. Run ``python scripts/misc/tooling/gradient_witness.py --check`` to
validate every ``results/**/gradient.json``.

The finite-difference rule is pre-registered (human ruling 2026-11-01) and lives here so the
harness and the validator share one copy: central differences in fp64 along
``FD_DIRECTIONS`` seeded random unit directions, at steps ``h`` and ``h/2`` combined by
Richardson extrapolation; the gradient passes when
``|FD - grad.v| / max(|grad.v|, FD_TINY) <= FD_TOLERANCE`` in every direction. A hazard is
flagged when value_and_grad costs more than ``HAZARD_GRAD_OVER_FORWARD`` times the forward
call, or the FD check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "gradient-witness"
VERSION = 1
#: Pre-registered FD rule (human ruling 2026-11-01, autolens_profiling#505).
FD_DIRECTIONS = 3
FD_STEP = 1e-3
FD_TOLERANCE = 1e-4
FD_TINY = 1e-8
FD_SEED = 20261101
#: Hazard: value_and_grad over forward, single call (epic phase 6 card: "~4x").
HAZARD_GRAD_OVER_FORWARD = 4.0
FD_STATUSES = ("pass", "fail", "not_assessed")
#: Timing keys (seconds). The single-call value_and_grad median is the v3 headline.
TIMINGS = (
    "forward.single_call_s",
    "forward.first_call_s",
    "forward.vmap_per_call_s",
    "value_and_grad.single_call_s",
    "value_and_grad.first_call_s",
    "value_and_grad.vmap_per_call_s",
)
HEADLINE = "value_and_grad.single_call_s"
REQUIRED_TIMINGS = ("forward.single_call_s", HEADLINE)


def _finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def fd_relative_error(fd: float, analytic: float) -> float:
    """The FD rule's relative error of one direction."""
    return abs(fd - analytic) / max(abs(analytic), FD_TINY)


def richardson(d_h: float, d_half: float) -> float:
    """Richardson extrapolation of two central differences at steps h and h/2."""
    return (4.0 * d_half - d_h) / 3.0


def fd_status(directions: list[dict]) -> str:
    """``pass`` when every direction meets the tolerance, else ``fail``."""
    if len(directions) != FD_DIRECTIONS:
        return "fail"
    return "pass" if all(d["rel_error"] <= FD_TOLERANCE for d in directions) else "fail"


def hazard(grad_over_forward: float | None, status: str) -> dict:
    """The hazard block: flagged on grad/forward > the threshold or a failed FD check."""
    reasons = []
    if _finite(grad_over_forward) and grad_over_forward > HAZARD_GRAD_OVER_FORWARD:
        reasons.append(
            f"value_and_grad costs {grad_over_forward:.2f}x the forward call "
            f"(threshold {HAZARD_GRAD_OVER_FORWARD:g}x)"
        )
    if status == "fail":
        reasons.append("finite-difference check failed")
    return {
        "flag": bool(reasons),
        "reasons": reasons,
        "threshold_grad_over_forward": HAZARD_GRAD_OVER_FORWARD,
    }


def validate(doc) -> list[str]:
    """Errors in one gradient witness (empty when it is valid)."""
    if not isinstance(doc, dict):
        return ["not an object"]
    errors = []
    if doc.get("schema") != SCHEMA or doc.get("version") != VERSION:
        errors.append(f"schema/version must be {SCHEMA}@{VERSION}")
    for key in ("autolens_version", "measured_at", "setup_id", "point"):
        if not _text(doc.get(key)):
            errors.append(f"{key} is required")
    timings = doc.get("timings")
    if not isinstance(timings, dict):
        errors.append("timings must be an object")
        timings = {}
    for key in REQUIRED_TIMINGS:
        if not (_finite(timings.get(key)) and timings[key] > 0):
            errors.append(f"timings.{key} must be a positive number")
    for key, value in timings.items():
        if key not in TIMINGS:
            errors.append(f"timings.{key} is not a gradient timing")
        elif value is not None and not (_finite(value) and value > 0):
            errors.append(f"timings.{key} must be positive or null")
    if not (_finite(doc.get("log_likelihood"))):
        errors.append("log_likelihood must be a finite number")
    gradient = doc.get("gradient")
    if not isinstance(gradient, dict) or not isinstance(gradient.get("finite"), bool):
        errors.append("gradient.finite must be a boolean")
    fd = doc.get("fd")
    if not isinstance(fd, dict) or fd.get("status") not in FD_STATUSES:
        errors.append(f"fd.status must be one of {FD_STATUSES}")
        return errors
    if fd["status"] == "not_assessed":
        if not _text(fd.get("reason")):
            errors.append("fd.reason is required when fd.status is not_assessed")
    else:
        directions = fd.get("directions")
        if not isinstance(directions, list) or not all(
            isinstance(d, dict) and all(_finite(d.get(k)) for k in ("analytic", "fd", "rel_error"))
            for d in directions
        ):
            errors.append("fd.directions must list {analytic, fd, rel_error} numbers")
        else:
            for d in directions:
                if abs(d["rel_error"] - fd_relative_error(d["fd"], d["analytic"])) > 1e-12 * max(
                    1.0, d["rel_error"]
                ):
                    errors.append("fd.directions rel_error disagrees with fd and analytic")
                    break
            if fd_status(directions) != fd["status"]:
                errors.append(f"fd.status {fd['status']} disagrees with the pre-registered rule")
        if gradient and gradient.get("finite") is False and fd["status"] == "pass":
            errors.append("a non-finite gradient cannot pass the FD check")
    ratio = doc.get("grad_over_forward")
    expected_ratio = (
        timings[HEADLINE] / timings["forward.single_call_s"]
        if all(_finite(timings.get(k)) for k in REQUIRED_TIMINGS)
        else None
    )
    if expected_ratio is not None and not (
        _finite(ratio) and abs(ratio - expected_ratio) <= 1e-9 * expected_ratio
    ):
        errors.append("grad_over_forward must equal the single-call value_and_grad / forward")
    if doc.get("hazard") != hazard(ratio, fd["status"]):
        errors.append("hazard disagrees with the pre-registered rule")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted((args.root / "results").rglob("gradient.json"))
    bad = 0
    for path in paths:
        try:
            errors = validate(json.loads(path.read_text()))
        except ValueError as exc:
            errors = [f"unreadable ({exc})"]
        for error in errors:
            print(f"{path.relative_to(args.root)}: {error}")
        bad += bool(errors)
    print(f"gradient: {len(paths) - bad} witness(es) valid" + (f", {bad} invalid" if bad else ""))
    return 1 if (args.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
