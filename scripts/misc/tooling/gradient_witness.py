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

Version 2 (human ruling 2026-11-01, autolens_profiling#509) adds the FD step sweep: along the same
seeded directions, the central difference at every step in ``SWEEP_STEPS`` (1e-1 down to 1e-8).
Each direction is classed (``sweep_class``) as ``agree`` (``SWEEP_RUN`` consecutive steps within
``FD_TOLERANCE`` of ``grad.v``), ``zero`` (the likelihood is flat: ``SWEEP_RUN`` consecutive
central differences exactly 0), ``mismatch`` (``SWEEP_RUN`` consecutive central differences
agree with each other within ``FD_TOLERANCE`` but not with ``grad.v``: a wrong gradient) or
``no_plateau`` (none: a likelihood not smooth, or too noisy, at any step). The sweep passes when
every direction agrees, and in a version-2 witness the sweep, not the h/h2 check, decides the FD
verdict (``fd_verdict``) the hazard and the day-zero stamp read; the h/h2 check is kept beside
it unchanged, for comparison with version 1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "gradient-witness"
VERSION = 2
#: Versions the validator accepts (version 1 has no ``fd_sweep``; its h/h2 check decides).
VERSIONS = (1, 2)
#: Pre-registered FD rule (human ruling 2026-11-01, autolens_profiling#505).
FD_DIRECTIONS = 3
FD_STEP = 1e-3
FD_TOLERANCE = 1e-4
FD_TINY = 1e-8
FD_SEED = 20261101
#: Hazard: value_and_grad over forward, single call (epic phase 6 card: "~4x").
HAZARD_GRAD_OVER_FORWARD = 4.0
FD_STATUSES = ("pass", "fail", "not_assessed")
#: The FD step sweep (version 2; human ruling 2026-11-01, autolens_profiling#509).
SWEEP_STEPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8)
SWEEP_RUN = 3
#: Direction classes, and the order a failed sweep names its mechanism in.
SWEEP_CLASSES = ("agree", "mismatch", "zero", "no_plateau")
SWEEP_MECHANISMS = {
    "mismatch": "the FD converges away from grad.v (wrong gradient)",
    "zero": "the likelihood is flat at small steps (piecewise constant)",
    "no_plateau": "no step agrees (likelihood not smooth or too noisy)",
}
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


def _longest_run(flags) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def sweep_class(analytic: float, centrals: list[float]) -> str:
    """One direction's sweep class from its central differences (``SWEEP_STEPS`` order)."""
    if _longest_run(fd_relative_error(c, analytic) <= FD_TOLERANCE for c in centrals) >= SWEEP_RUN:
        return "agree"
    if _longest_run(c == 0.0 for c in centrals) >= SWEEP_RUN:
        return "zero"
    steady = (
        fd_relative_error(centrals[i], centrals[i - 1]) <= FD_TOLERANCE
        for i in range(1, len(centrals))
    )
    if _longest_run(steady) >= SWEEP_RUN - 1:
        return "mismatch"
    return "no_plateau"


def sweep_status(classes: list[str]) -> str:
    """``pass`` when every direction agrees, else ``fail``."""
    if len(classes) != FD_DIRECTIONS:
        return "fail"
    return "pass" if all(c == "agree" for c in classes) else "fail"


def fd_verdict(doc: dict) -> str:
    """The FD status that decides the hazard and the day-zero stamp: the sweep's in a version-2
    witness, the h/h2 check's in a version-1 witness."""
    sweep = doc.get("fd_sweep")
    return sweep["status"] if isinstance(sweep, dict) else doc["fd"]["status"]


def hazard(grad_over_forward: float | None, status: str, sweep_classes=None) -> dict:
    """The hazard block: flagged on grad/forward > the threshold or a failed FD check. With
    ``sweep_classes`` (version 2) the sweep decides the FD reason and names its mechanisms."""
    reasons = []
    if _finite(grad_over_forward) and grad_over_forward > HAZARD_GRAD_OVER_FORWARD:
        reasons.append(
            f"value_and_grad costs {grad_over_forward:.2f}x the forward call "
            f"(threshold {HAZARD_GRAD_OVER_FORWARD:g}x)"
        )
    if sweep_classes is not None:
        for mechanism in SWEEP_CLASSES[1:]:
            n = sum(c == mechanism for c in sweep_classes)
            if n:
                reasons.append(
                    f"FD step sweep {mechanism} in {n}/{len(sweep_classes)} direction(s): "
                    + SWEEP_MECHANISMS[mechanism]
                )
        if status == "fail" and not any(c != "agree" for c in sweep_classes):
            reasons.append("FD step sweep failed")
    elif status == "fail":
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
    if doc.get("schema") != SCHEMA or doc.get("version") not in VERSIONS:
        errors.append(f"schema/version must be {SCHEMA}@{'/'.join(map(str, VERSIONS))}")
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
    sweep_classes = None
    if doc.get("version") == 2:
        sweep_errors, sweep_classes = _validate_sweep(doc.get("fd_sweep"), gradient)
        errors.extend(sweep_errors)
    elif "fd_sweep" in doc:
        errors.append("fd_sweep belongs to a version-2 witness")
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
    status = doc["fd_sweep"]["status"] if sweep_classes is not None else fd["status"]
    if doc.get("hazard") != hazard(ratio, status, sweep_classes):
        errors.append("hazard disagrees with the pre-registered rule")
    return errors


def _validate_sweep(sweep, gradient) -> tuple[list[str], list[str] | None]:
    """Errors in a version-2 ``fd_sweep`` block, and its direction classes when it is sound."""
    if not isinstance(sweep, dict) or sweep.get("status") not in FD_STATUSES:
        return [f"fd_sweep.status must be one of {FD_STATUSES}"], None
    if sweep["status"] == "not_assessed":
        if not _text(sweep.get("reason")):
            return ["fd_sweep.reason is required when fd_sweep.status is not_assessed"], None
        return [], []
    directions = sweep.get("directions")
    if not isinstance(directions, list) or not all(
        isinstance(d, dict)
        and _finite(d.get("analytic"))
        and isinstance(d.get("steps"), list)
        and [r.get("step") for r in d["steps"] if isinstance(r, dict)] == list(SWEEP_STEPS)
        and all(_finite(r.get("central")) for r in d["steps"])
        for d in directions
    ):
        return [
            "fd_sweep.directions must list {analytic, steps: [{step, central}]} over SWEEP_STEPS"
        ], None
    errors = []
    classes = [sweep_class(d["analytic"], [r["central"] for r in d["steps"]]) for d in directions]
    if [d.get("class") for d in directions] != classes:
        errors.append("fd_sweep direction classes disagree with the sweep rule")
    if sweep_status(classes) != sweep["status"]:
        errors.append(f"fd_sweep.status {sweep['status']} disagrees with the sweep rule")
    if gradient and gradient.get("finite") is False and sweep["status"] == "pass":
        errors.append("a non-finite gradient cannot pass the FD step sweep")
    return errors, classes


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
