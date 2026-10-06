"""Read-only campaign enumeration and evidence screening; never run or accept profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
REPOS = (
    "autolens_profiling",
    "PyAutoNerves",
    "PyAutoFit",
    "PyAutoArray",
    "PyAutoGalaxy",
    "PyAutoLens",
)
METHODS = {
    "runtime": ("runtime", "single_call", "s", "warm", "synchronized_single_call"),
    "breakdown": ("breakdown", "components", "s", "warm", "synchronized_components"),
    "compile_cold": ("compile", "compile", "s", "cold", "fresh_process_compile"),
    "compile_warm": ("compile", "compile", "s", "warm", "fresh_process_compile"),
    "host_peak_rss": ("memory", "host_peak_rss", "MiB", "warm", "isolated_process_peak_rss"),
    "device_peak_allocated": (
        "memory",
        "device_peak_allocated",
        "B",
        "warm",
        "allocator_peak_allocated",
    ),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def hash_value(value, length=64):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value)


def timestamp(value):
    require(isinstance(value, str) and value.endswith("Z"), "timestamp must be UTC with Z suffix")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def contained(root, relative):
    require(isinstance(relative, str), "artifact path must be a string")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), "artifact escapes campaign directory")
    return resolved


def validate(spec, root=ROOT):
    """Validate structure and return unresolved scientific choices, never resolve them."""
    require(
        spec["schema"] == "profiling-baseline-campaign"
        and type(spec["version"]) is int
        and spec["version"] == 1,
        "unsupported campaign schema",
    )
    require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", spec["campaign_id"]), "invalid campaign id")
    require(spec["status"] in ("draft", "frozen"), "invalid campaign status")
    blockers = []

    def known(value, path, predicate):
        if value is None:
            blockers.append(f"unresolved: {path}")
        else:
            require(predicate(value), f"invalid {path}")

    require(
        set(spec["inventory"]) == {"catalogue/registry.json", "catalogue/script_routes.json"},
        "inventory must pin registry and routes",
    )
    for path, expected in spec["inventory"].items():
        require(
            hash_value(expected)
            and hashlib.sha256((root / path).read_bytes()).hexdigest() == expected,
            f"inventory changed: {path}; explicitly review and re-freeze campaign",
        )
    registry = read_json(root / "catalogue/registry.json")
    require(set(spec["software"]["revisions"]) == set(REPOS), "incomplete software revision map")
    for repo, revision in spec["software"]["revisions"].items():
        known(revision, f"software.revisions.{repo}", lambda v: hash_value(v, 40))
    known(
        spec["software"]["environment_lock_sha256"], "software.environment_lock_sha256", hash_value
    )
    for key in ("not_before_utc", "not_after_utc", "authorization_reference"):
        known(
            spec["collection"][key],
            f"collection.{key}",
            lambda v: isinstance(v, str) and bool(v.strip()),
        )
    start, end = (spec["collection"][key] for key in ("not_before_utc", "not_after_utc"))
    if start is not None:
        timestamp(start)
    if end is not None:
        timestamp(end)
    if start is not None and end is not None:
        require(timestamp(start) < timestamp(end), "collection window must increase")
    require(
        set(spec["devices"]) == set(registry["devices"]), "device coverage differs from catalogue"
    )
    for name, device in spec["devices"].items():
        for key in ("host", "processor", "accelerator", "driver", "cpu_affinity"):
            known(
                device[key],
                f"devices.{name}.{key}",
                lambda v: isinstance(v, str) and bool(v.strip()),
            )
        for key in ("ram_bytes", "device_memory_bytes"):
            known(
                device[key],
                f"devices.{name}.{key}",
                lambda v, k=key, d=name: integer(v, 1 if k == "ram_bytes" or d != "cpu" else 0),
            )
        known(device["environment_sha256"], f"devices.{name}.environment_sha256", hash_value)
        require(integer(device["threads"], 1), "threads must be positive integer")
        require(
            device["backend"] == ("cpu" if name == "cpu" else "cuda"), "backend/device mismatch"
        )
        require(
            device["partition"] == ("ral" if name == "cpu" else "gpu"),
            "CPU arrays must use ral; GPU arrays use gpu",
        )
        require(number(device["max_load_per_allocated_cpu"]), "invalid host load bound")
        require(
            integer(device["max_other_gpu_processes"]) and device["max_other_gpu_processes"] == 0,
            "reference GPU must be exclusive",
        )
    require(spec["precisions"] == ["fp64", "mixed"], "explicit fp64/mixed coverage required")
    protocol = spec["protocol"]
    require(protocol["scope"] == "single_call; no vmap batching", "unsupported protocol scope")
    require(
        isinstance(protocol["cpu_exclusion_policy"], str)
        and bool(protocol["cpu_exclusion_policy"].strip()),
        "missing CPU exclusion policy",
    )
    for key, minimum in [
        ("warmup_calls", 1),
        ("repetitions", 2),
        ("batch_size", 1),
        ("cpu_timeout_seconds", 1),
    ]:
        require(integer(protocol[key], minimum), f"invalid protocol.{key}")
    require(
        protocol["batch_size"] == 1
        and protocol["sync"] == "all_output_leaves_before_and_after"
        and protocol["aggregation"] == "median_with_raw_samples",
        "unsupported timing protocol",
    )
    require(len(spec["measurements"]) == len(METHODS), "incomplete measurement methods")
    require({m["id"] for m in spec["measurements"]} == set(METHODS), "measurement ids differ")
    for m in spec["measurements"]:
        require(
            isinstance(m["description"], str) and bool(m["description"].strip()),
            "missing method description",
        )
        require(
            tuple(m[k] for k in ("axis", "metric", "unit", "cache_state", "method"))
            == METHODS[m["id"]],
            f"method/unit mismatch: {m['id']}",
        )
    cells = {c["id"]: c for c in registry["cells"]}
    require(
        len(spec["setups"]) == len(cells) and {s["id"] for s in spec["setups"]} == set(cells),
        "setup coverage differs from catalogue",
    )
    for setup in spec["setups"]:
        cell = cells[setup["id"]]
        require(
            setup["dataset"] == cell["dataset"]
            and setup["instrument"] == cell["instrument"]
            and setup["model"] == registry["model_aliases"].get(cell["model"], cell["model"]),
            "setup identity mismatch",
        )
        known(
            setup["configuration"],
            f"{setup['id']}.configuration",
            lambda v: (
                isinstance(v, dict)
                and bool(v)
                and all(k in v for k in ("geometry", "model", "solver", "parameters", "seed"))
                and all(
                    isinstance(v[k], dict) and bool(v[k]) for k in ("geometry", "model", "solver")
                )
                and isinstance(v["parameters"], (dict, list))
                and bool(v["parameters"])
                and integer(v["seed"])
                and not unresolved(v)
            ),
        )
        known(
            setup["input_sha256"],
            f"{setup['id']}.input_sha256",
            lambda v: (
                isinstance(v, dict)
                and bool(v)
                and all(isinstance(k, str) and k and hash_value(h) for k, h in v.items())
            ),
        )
        witness = setup["correctness"]
        known(
            witness["reference_sha256"], f"{setup['id']}.correctness.reference_sha256", hash_value
        )
        for key in ("absolute_tolerance", "relative_tolerance"):
            known(witness[key], f"{setup['id']}.correctness.{key}", number)
    if spec["status"] != "frozen":
        blockers.insert(0, "draft: later human campaign freeze and compute authorization required")
    return blockers


def unresolved(value):
    if isinstance(value, dict):
        return not value or any(unresolved(v) for v in value.values())
    if isinstance(value, list):
        return not value or any(unresolved(v) for v in value)
    return value is None or value == ""


def enumerate_cells(spec, root=ROOT):
    blockers = validate(spec, root)
    routes = read_json(root / "catalogue/script_routes.json")
    cells = []
    for setup in spec["setups"]:
        for device in spec["devices"]:
            for precision in spec["precisions"]:
                for method in spec["measurements"]:
                    route = None
                    state, reason = (
                        "unverified",
                        "No demonstrated producer for the frozen measurement protocol.",
                    )
                    if method["id"].startswith("compile_"):
                        route = routes["compile_probe"]["path"]
                        if routes["compile_probe"]["builder_status"] == "unavailable":
                            state, reason = "unsupported", routes["compile_probe"]["reason"]
                    elif method["id"] == "device_peak_allocated" and device == "cpu":
                        state, reason = (
                            "not_applicable",
                            "CPU has no device allocator; host RSS remains a separate target.",
                        )
                    elif method["id"] in ("runtime", "breakdown"):
                        measurement = (
                            "likelihood_runtime"
                            if method["id"] == "runtime"
                            else "likelihood_breakdown"
                        )
                        match = next(
                            (
                                r
                                for r in routes["routes"]
                                if r["dataset"] == setup["dataset"]
                                and r["model"] == setup["model"]
                                and r["measurement"] == measurement
                            ),
                            None,
                        )
                        if match:
                            route = match["path"]
                        else:
                            state, reason = (
                                "unsupported",
                                "No canonical script route for this cell/measurement.",
                            )
                    runtime_dispatch = any(
                        r["dataset"] == setup["dataset"]
                        and r["path"] == route
                        and setup["instrument"] in r["instruments"]
                        for r in routes["runtime_cells"]
                    )
                    cells.append(
                        {
                            "id": "/".join([setup["id"], device, precision, method["id"]]),
                            "setup_id": setup["id"],
                            "device": device,
                            "precision": precision,
                            "measurement": method["id"],
                            "unit": method["unit"],
                            "capability": state,
                            "reason": reason,
                            "script": route,
                            "runtime_sweep_dispatch": runtime_dispatch,
                        }
                    )
    return {
        "schema": "baseline-readiness-plan",
        "version": 1,
        "campaign_id": spec["campaign_id"],
        "spec_sha256": fingerprint(spec),
        "status": "blocked" if blockers else "specified_for_human_review",
        "blockers": blockers,
        "counts": dict(Counter(c["capability"] for c in cells)),
        "cells": cells,
    }


def context(spec, cell):
    return {
        "software": spec["software"],
        "setup": next(s for s in spec["setups"] if s["id"] == cell["setup_id"]),
        "hardware": spec["devices"][cell["device"]],
        "device": cell["device"],
        "precision": cell["precision"],
        "protocol": spec["protocol"],
        "measurement": next(m for m in spec["measurements"] if m["id"] == cell["measurement"]),
    }


def check_observation(spec, cell, record):
    require(
        record["schema"] == "baseline-observation"
        and type(record["version"]) is int
        and record["version"] == 1,
        "unsupported observation schema",
    )
    require(
        record["campaign_id"] == spec["campaign_id"]
        and record["spec_sha256"] == fingerprint(spec)
        and record["cell_id"] == cell["id"],
        "campaign/spec/cell mismatch",
    )
    require(
        record["origin"] == "fresh-campaign",
        "archived/imported observations cannot be baseline candidates",
    )
    at = timestamp(record["observed_at"])
    require(
        timestamp(spec["collection"]["not_before_utc"])
        <= at
        <= timestamp(spec["collection"]["not_after_utc"])
        and at <= datetime.now(UTC),
        "observation outside collection window or in future",
    )
    expected = context(spec, cell)
    require(
        canonical(record["context"]) == canonical(expected),
        "observed context differs from frozen specification",
    )
    n = 1 if record["outcome"] == "cpu_timeout" else spec["protocol"]["repetitions"]
    resources = record["resources"]
    loads = resources["load_per_allocated_cpu"]
    require(
        isinstance(loads, list)
        and len(loads) == n
        and all(
            number(v) and v <= expected["hardware"]["max_load_per_allocated_cpu"] for v in loads
        ),
        "host load missing or exceeded",
    )
    if cell["device"] != "cpu":
        peers = resources["other_gpu_processes"]
        require(
            isinstance(peers, list)
            and len(peers) == n
            and all(type(v) is int and v == 0 for v in peers),
            "GPU contention missing or detected",
        )
    if record["outcome"] == "cpu_timeout":
        require(
            cell["device"] == "cpu" and cell["measurement"] == "runtime",
            "CPU timeout exclusions apply only to CPU runtime",
        )
        require(
            number(record["elapsed_seconds"], spec["protocol"]["cpu_timeout_seconds"]),
            "CPU timeout below frozen limit",
        )
        require("samples" not in record, "timeout must not masquerade as timing samples")
        return "cpu_exclusion_for_human_review"
    require(record["outcome"] == "measured", "invalid observation outcome")
    require(
        record["fresh_process"] is True and record["synchronized"] is True,
        "fresh process/synchronization not witnessed",
    )
    require(
        integer(record["warmup_calls"])
        and (
            record["warmup_calls"] == 0
            if cell["measurement"].startswith("compile_")
            else record["warmup_calls"] >= spec["protocol"]["warmup_calls"]
        ),
        "insufficient warmup",
    )
    witness = record["correctness"]
    require(
        witness["reference_sha256"] == expected["setup"]["correctness"]["reference_sha256"],
        "correctness reference mismatch",
    )
    for key, tolerance in [
        ("max_absolute_error", "absolute_tolerance"),
        ("max_relative_error", "relative_tolerance"),
    ]:
        require(
            number(witness[key]) and witness[key] <= expected["setup"]["correctness"][tolerance],
            f"correctness witness failed: {key}",
        )
    samples = record["samples"]
    require(isinstance(samples, dict) and bool(samples), "raw samples missing")
    if cell["measurement"] != "breakdown":
        require(
            set(samples) == {expected["measurement"]["metric"]}, "measurement sample key mismatch"
        )
    else:
        require(
            record["component_accounting"] in ("exclusive", "cumulative"),
            "component accounting not declared",
        )
    for name, values in samples.items():
        require(
            isinstance(name, str)
            and bool(name)
            and isinstance(values, list)
            and len(values) == n
            and all(number(v) for v in values),
            "invalid sample values or repetition count",
        )
    if cell["measurement"].startswith("compile_"):
        require(
            record["cache_state_witness"] == expected["measurement"]["cache_state"],
            "cache-state witness missing",
        )
    return "candidate_for_human_review"


def report(spec, receipt, root=ROOT):
    plan = enumerate_cells(spec, root)
    require(
        receipt["schema"] == "baseline-evidence-receipt"
        and type(receipt["version"]) is int
        and receipt["version"] == 1,
        "unsupported receipt schema",
    )
    require(isinstance(receipt["records"], list), "receipt records must be a list")
    known = {c["id"]: c for c in plan["cells"]}
    indexed = {}
    errors = []
    for row in receipt["records"]:
        require(
            isinstance(row, dict) and isinstance(row.get("cell_id"), str), "invalid receipt row"
        )
        indexed.setdefault(row["cell_id"], []).append(row)
        if row["cell_id"] not in known:
            errors.append(f"unknown cell: {row['cell_id']}")
    campaign_root = root / "results/campaigns" / spec["campaign_id"]
    results = []
    for cell in plan["cells"]:
        rows = indexed.get(cell["id"], [])
        status, reasons = "missing", [cell["reason"]]
        if cell["capability"] == "not_applicable":
            status = "not_applicable"
        if rows:
            try:
                require(len(rows) == 1, "duplicate evidence for cell")
                require(
                    cell["capability"] not in ("not_applicable", "unsupported"),
                    "no supported measurement capability",
                )
                require(not plan["blockers"], "campaign specification unresolved")
                row = rows[0]
                artifact = contained(campaign_root, row["artifact"])
                # Also confine the campaign root itself, including symlinks.
                require(
                    artifact.is_relative_to(
                        root.resolve() / "results/campaigns" / spec["campaign_id"]
                    ),
                    "artifact escaped campaign root",
                )
                require(
                    hash_value(row["sha256"])
                    and hashlib.sha256(artifact.read_bytes()).hexdigest() == row["sha256"],
                    "artifact hash mismatch",
                )
                status = check_observation(spec, cell, read_json(artifact))
                reasons = []
            except (KeyError, TypeError, ValueError, OSError) as exc:
                status, reasons = "rejected", [str(exc)]
        results.append({"cell_id": cell["id"], "status": status, "reasons": reasons})
    counts = dict(Counter(r["status"] for r in results))
    complete = (
        not plan["blockers"]
        and not errors
        and not (counts.get("missing") or counts.get("rejected"))
    )
    return {
        "schema": "baseline-acceptance-report",
        "version": 1,
        "campaign_id": spec["campaign_id"],
        "spec_sha256": plan["spec_sha256"],
        "status": "ready_for_human_review" if complete else "blocked",
        "accepted": False,
        "qualification": "Structural screening only. Producer freshness, hardware, synchronization, scientific adequacy and correctness-reference independence require human verification; declarations are not independently observed here.",
        "blockers": plan["blockers"],
        "receipt_errors": errors,
        "counts": counts,
        "cells": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "enumerate", "report"))
    parser.add_argument("--spec", type=Path, default=ROOT / "baseline/campaign.json")
    parser.add_argument(
        "--receipt", type=Path, help="Explicit evidence receipt, required for report"
    )
    args = parser.parse_args(argv)
    try:
        spec = read_json(args.spec)
        if args.command == "report":
            require(args.receipt is not None, "report requires --receipt; no result autodiscovery")
            result = report(spec, read_json(args.receipt))
        else:
            result = enumerate_cells(spec)
            if args.command == "validate":
                result.pop("cells")
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(args.command != "enumerate" and result["status"] == "blocked")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"baseline readiness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
