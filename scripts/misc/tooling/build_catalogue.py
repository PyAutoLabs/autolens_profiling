"""Generate the setup-oriented profiling evidence catalogue (stdlib only).

Run ``python scripts/misc/tooling/build_catalogue.py [--check]``. The registry
owns navigation and explicit reference choices; adapters only transcribe
recorded measurements. No script imports, timings, latest-row choice or
scientific acceptance occur here. See catalogue/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogue_adapters import metrics, number, pointer, rows

REGISTRY = "catalogue/registry.json"
UNKNOWN = "Not recorded in this legacy evidence; no current default substituted."
REQUIRED_SETTINGS = {
    "image_shape": "pixel",
    "image_pixels_masked": "count",
    "source_pixels": "count",
    "psf_shape": "pixel",
    "oversampling": "dimensionless",
    "regularization": "name",
    "solver": "name",
    "preloads": "name",
    "n_vis": "count",
    "transformer": "name",
}
CONFIG_KEYS = (
    "configuration",
    "regularization",
    "transformer",
    "solver",
    "settings",
    "dataset",
    "adapt_image",
)
PRIVATE_KEY = re.compile(
    r"(^|_)(path|dir|directory|command|cache_dir|cwd|argv|traceback|error)($|_)"
)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def safe_path(path):
    return (
        text(path)
        and not path.startswith("/")
        and not any(c in path for c in "\\:\n\r\0")
        and all(p not in ("", ".", "..") for p in path.split("/"))
    )


def source_path(root, path):
    if not safe_path(path):
        raise ValueError(f"Unsafe catalogue path: {path!r}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Catalogue path escapes checkout: {path}")
    return target


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[
        :20
    ]


def load_json(path):
    return json.loads(path.read_text())


def utc(value):
    if not text(value) or not (value.endswith("Z") or value.endswith("+00:00")):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
    except ValueError:
        return None


def clean(value):
    """Only public, finite metadata; local paths stay in the original artifact."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not PRIVATE_KEY.search(k)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        if value.startswith(("/", "~/", "C:\\")) or re.search(r"/(?:home|mnt|tmp|Users)/", value):
            return None
        return value
    if value is None or isinstance(value, bool) or number(value):
        return value
    # Negative finite scientific configuration values (centres etc.) are valid.
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        import math

        return value if math.isfinite(value) else None
    return None


def registry(root):
    doc = load_json(source_path(root, REGISTRY))
    if doc.get("schema") != "profiling-catalogue-registry" or doc.get("version") != 1:
        raise ValueError("Unsupported catalogue registry schema/version")
    for key in (
        "sources",
        "cells",
        "devices",
        "slots",
        "references",
        "overrides",
        "hazard_bindings",
        "recommendations",
    ):
        if not isinstance(doc.get(key), list):
            raise ValueError(f"registry.{key} must be a list")
    if not isinstance(doc.get("families"), dict) or not doc["families"]:
        raise ValueError("registry.families must declare the scientific navigation")
    seen = set()
    for cell in doc["cells"]:
        if (
            not text(cell.get("id"))
            or cell["id"] in seen
            or cell.get("dataset") not in doc["families"]
            or not text(cell.get("model"))
        ):
            raise ValueError(f"Invalid/duplicate declared cell: {cell}")
        seen.add(cell["id"])
    for spec in doc["sources"]:
        # Globs are trusted declarative paths, but never absolute/traversing.
        if not safe_path(spec.get("glob")) or spec.get("adapter") not in ("profile", "compile"):
            raise ValueError(f"Invalid source declaration: {spec}")
    for spec in doc["overrides"]:
        if not safe_path(spec.get("prefix", "").rstrip("/")) or spec.get("adapter") not in (
            "profile",
            "compile",
            "streaming",
            "hazards",
            "inventory",
        ):
            raise ValueError(f"Invalid source override: {spec}")
    if not doc["devices"] or any(not text(d) for d in doc["devices"]):
        raise ValueError("registry.devices must name target devices")
    for slot in doc["slots"]:
        units = {
            "runtime": {"s"},
            "breakdown": {"s"},
            "compile": {"s"},
            "memory": {"B", "KiB", "MiB", "GiB"},
        }
        if slot.get("unit") not in units.get(slot.get("axis"), set()) or not text(
            slot.get("metric")
        ):
            raise ValueError(f"Invalid baseline slot: {slot}")
    for ref in doc["references"]:
        if (
            not safe_path(ref.get("path"))
            or not isinstance(ref.get("pointer"), str)
            or not text(ref.get("reason"))
        ):
            raise ValueError(f"Invalid reference declaration: {ref}")
    return doc


def route(path, payload, reg):
    parts = Path(path).parts
    dataset = payload.get("dataset_class")
    if not text(dataset):
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), str) else None
    if dataset == "datacube_img":
        dataset = "multi_dataset"
    if dataset not in reg["families"]:
        dataset = next((p for p in parts if p in reg["families"]), "experiments")
    model = payload.get("model_type") or payload.get("model")
    if not text(model):
        config = payload.get("configuration") or {}
        model = config.get("mesh") if isinstance(config, dict) else None
    if not text(model):
        model = next((m for m in reg["model_patterns"] if m in Path(path).stem), None)
    if not text(model):
        model = parts[2] if dataset == "lens" and len(parts) > 3 else "other"
    model = reg["model_aliases"].get(model, model)
    instrument = payload.get("instrument") if text(payload.get("instrument")) else None
    if instrument is None:
        instrument = next(
            (p for p in parts if p in {c["instrument"] for c in reg["cells"] if c["instrument"]}),
            None,
        )
    return dataset, model, instrument


def settings(payload, context):
    values = {}
    config = payload.get("configuration")
    if isinstance(config, dict):
        values.update(clean(config))
    for key in (*REQUIRED_SETTINGS, *CONFIG_KEYS[1:]):
        if key in payload:
            values[key] = clean(payload[key])
    for key in (
        "ndim",
        "n_batch",
        "vmap_batch",
        "batch_size",
        "transform",
        "inversion_path",
        "use_mixed_precision",
        "mixed_precision",
    ):
        if key in payload:
            values[key] = clean(payload[key])
    if isinstance(payload.get("vmap"), dict) and "batch_size" in payload["vmap"]:
        values["vmap_batch_size"] = clean(payload["vmap"]["batch_size"])
    values.update(clean(context))
    aliases = {
        "delaunay_vertices": "source_pixels",
        "pixels_in_mask": "image_pixels_masked",
        "over_sampled_pixels": "oversampled_pixels",
    }
    for old, new in aliases.items():
        if old in values and new not in values:
            values[new] = values[old]
    result = {}
    for key in sorted(set(values) | set(REQUIRED_SETTINGS)):
        value = values.get(key)
        result[key] = {"value": value, "unit": REQUIRED_SETTINGS.get(key, "recorded")}
        if value is None:
            result[key]["reason"] = UNKNOWN
    return result


def identity(payload, path):
    device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
    provenance = device.get("provenance") if isinstance(device.get("provenance"), dict) else {}
    software = {}
    for key in ("library_versions", "library_revisions", "dependency_versions"):
        values = provenance.get(key)
        if isinstance(values, dict):
            software.update({f"{name}.{key}": val for name, val in values.items() if text(val)})
    for key in ("source_revisions", "release_tags", "versions"):
        values = payload.get(key)
        if isinstance(values, dict):
            software.update({name: val for name, val in values.items() if text(val)})
    version = payload.get("autolens_version") or payload.get("release_version")
    if text(version):
        software["PyAutoLens"] = version
    else:
        version = None
    if text(payload.get("jax_version")):
        software["jax"] = payload["jax_version"]
    hardware = payload.get("hardware")
    backend = device.get("backend") or payload.get("backend")
    # Only explicit, documented config/hardware labels are used as hints.
    label = next(
        (x for x in (payload.get("_config_label"), hardware, Path(path).stem) if text(x)), ""
    )
    precision = None
    mixed = payload.get("use_mixed_precision", payload.get("mixed_precision"))
    if not isinstance(mixed, bool) and isinstance(payload.get("precision"), dict):
        mixed = payload["precision"].get("use_mixed_precision")
    if isinstance(mixed, bool):
        precision = "mixed" if mixed else "float64"
    elif "_fp64" in label:
        precision = "float64"
    elif re.search(r"_mp(?:_|$)", label):
        precision = "mixed"
    if not text(backend) and text(hardware):
        backend = "cpu" if hardware == "local_cpu" else "gpu" if "gpu" in hardware else None
    gpu_name = (device.get("nvidia_smi") or "").split(",")[0]
    dev = (
        "a100"
        if "A100" in gpu_name or "a100" in label
        else "gpu"
        if backend in ("gpu", "cuda")
        else "cpu"
        if backend == "cpu"
        else None
    )
    ident = {
        "device": dev,
        "backend": backend if text(backend) else None,
        "precision": precision,
        "library_version": version,
        "software": software,
        "hardware_details": clean({k: v for k, v in device.items() if k != "provenance"}),
    }
    ident["unknowns"] = {k: UNKNOWN for k, v in ident.items() if v is None}
    host = provenance.get("host") or device.get("hostname") or payload.get("hostname")
    measured_at = utc(provenance.get("captured_at") or payload.get("timestamp"))
    prov = {
        "host": host if text(host) else None,
        "measured_at": measured_at,
        "has_provenance": bool(provenance),
        "qualified": False,
        "unknowns": {},
    }
    for key in ("host", "measured_at"):
        if prov[key] is None:
            prov["unknowns"][key] = UNKNOWN
    # Source validity and scientific acceptance are separate. No legacy row is
    # promoted by this exporter, even when it carries provenance/passing pins.
    return ident, prov


def method(payload, axis, metric, statistic, repeats):
    valid_n = isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0
    result = {
        "statistic": statistic,
        "repetitions": repeats if valid_n else None,
        "warmup": None,
        "synchronization": None,
        "cache_state": payload.get("cache_state") if text(payload.get("cache_state")) else None,
    }
    result["unknowns"] = {k: UNKNOWN for k, v in result.items() if v is None}
    result["id"] = "legacy/" + digest([axis, metric, result])
    return result


def selection(record, reason):
    result = {k: record[k] for k in ("setup_id", "axis", "metric", "unit", "identity")}
    result.update(
        id="selected/" + record["id"],
        method_id=record["method"]["id"],
        host=record["provenance"]["host"],
        status="unreviewed",
        record_id=record["id"],
        reason=reason,
    )
    if result["host"] is None:
        result["unknowns"] = {"host": UNKNOWN}
    return result


def build(root, generated, revision):
    root = Path(root).resolve()
    reg = registry(root)
    setups, records, selections, inventory, estimates, findings = [], [], [], [], [], []
    planned_cells = []
    refs = {(r["path"], r["pointer"]): r for r in reg["references"]}
    if len(refs) != len(reg["references"]):
        raise ValueError("Duplicate explicit reference declaration")
    used_refs = set()
    files = {}
    for spec in reg["sources"]:
        for p in sorted(root.glob(spec["glob"])):
            path = p.relative_to(root).as_posix()
            source_path(root, path)
            if path in files:
                raise ValueError(f"Overlapping catalogue source declarations: {path}")
            files[path] = spec["adapter"]
    for path, adapter in sorted(files.items()):
        override = next((s for s in reg["overrides"] if path.startswith(s["prefix"])), {})
        adapter = override.get("adapter", adapter)
        doc = load_json(source_path(root, path))
        source = {
            "path": path,
            "sha256": hashlib.sha256(source_path(root, path).read_bytes()).hexdigest(),
            "adapter": adapter,
            "record_count": 0,
            "reasons": [],
        }
        if adapter == "hazards":
            block = doc.get("findings", {}) if isinstance(doc, dict) else {}
            items = (
                block.items()
                if isinstance(block, dict)
                else enumerate(block)
                if isinstance(block, list)
                else []
            )
            for finding_key, finding in items:
                if not isinstance(finding, dict):
                    continue
                fid = finding.get("finding_id")
                if text(fid):
                    findings.append(
                        {
                            "id": fid,
                            "title": finding.get("title", fid),
                            "description": finding.get("summary", ""),
                            "subject": finding.get("subject"),
                            "subject_name": finding.get("subject_name"),
                            "evidence": {
                                "path": path,
                                "fragment": pointer("/findings", finding_key),
                            },
                            "status": "applicability_unknown",
                            "reason": "Legacy finding does not identify measured library versions and an exact profiling setup; no global applicability inferred.",
                        }
                    )
            source["reasons"].append(
                "Findings retained separately until explicit versioned setup bindings exist."
            )
        for payload, base, context in rows(doc, adapter):
            dataset, model, instrument = route(path, payload, reg)
            dataset, model = override.get("dataset", dataset), override.get("model", model)
            config = settings(payload, context)
            # Legacy schemas do not promise complete scientific identity. A
            # source/row boundary prevents a false join despite equal partial settings.
            cid = digest([path, base, config])
            setup_id = f"{dataset}/{model}/{instrument or 'unspecified'}/{cid}"
            setup = {
                "id": setup_id,
                "label": f"{model} · {instrument or 'instrument unspecified'}",
                "dataset": dataset,
                "model": model,
                "instrument": instrument,
                "configuration_id": cid,
                "configuration": config,
                "role": "reference_candidate" if (path, base) in refs else "legacy_or_experimental",
                "evidence": {"path": path, "fragment": base or None},
                "identity_limitations": "Isolated to source row: legacy metadata is insufficient for cross-file joins.",
            }
            if instrument is None:
                setup["unknowns"] = {"instrument": UNKNOWN}
            ident, prov = identity(payload, path)
            if adapter == "streaming" and payload.get("outcome") != "OK":
                setups.append(setup)
                status = "unusable" if "MEM" in str(payload.get("outcome")) else "failed"
                selections.append(
                    missing_slot(
                        setup,
                        ident,
                        prov["host"],
                        "compile",
                        "dataset_setup_wall",
                        "s",
                        status,
                        str(payload.get("outcome", "Unknown outcome")),
                    )
                )
                source["reasons"].append(
                    f"{base}: unsuccessful run; no successful timing or peak-memory measurement exported."
                )
                continue
            if isinstance(payload.get("samples"), list) and "per_replica_mb" in payload:
                estimates.append(
                    {
                        "evidence": {"path": path, "fragment": base or None},
                        "dataset": dataset,
                        "model": model,
                        "instrument": instrument,
                        "kind": "xla_static_buffer_estimate",
                        "per_replica": {
                            "value": clean(payload.get("per_replica_mb")),
                            "unit": "MiB",
                        },
                        "recommended_batch_size": clean(payload.get("recommended_batch_size")),
                        "reason": "Compiler memory_analysis estimate; not observed peak VRAM or a validated batch recommendation.",
                    }
                )
                source["reasons"].append(
                    "Static VRAM estimate preserved outside measured memory records."
                )
            emitted = []
            for axis, metric, unit, value, anchor, statistic, n in metrics(payload, adapter, base):
                if not number(value):
                    source["reasons"].append(
                        f"{anchor}: missing/non-finite/negative measurement refused."
                    )
                    continue
                if not ident["software"]:
                    source["reasons"].append(
                        f"{anchor}: no measured software version/revision; cannot satisfy v2 record identity."
                    )
                    continue
                rid = "measurement/" + digest([path, anchor, axis, metric])
                emitted.append(
                    {
                        "id": rid,
                        "setup_id": setup_id,
                        "run_id": "legacy/" + digest([path, base]),
                        "axis": axis,
                        "metric": metric,
                        "unit": unit,
                        "measurement": {metric: value},
                        "identity": ident,
                        "method": method(payload, axis, metric, statistic, n),
                        "provenance": prov,
                        "validation": {
                            "status": "unreviewed",
                            "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
                        },
                        "evidence": {"path": path, "fragment": anchor},
                        "adapter": adapter,
                    }
                )
            if emitted:
                setups.append(setup)
                records.extend(emitted)
                source["record_count"] += len(emitted)
                if (path, base) in refs:
                    used_refs.add((path, base))
                    selections.extend(selection(r, refs[(path, base)]["reason"]) for r in emitted)
        source["status"] = "mapped" if source["record_count"] else "indexed_only"
        if not source["record_count"] and not source["reasons"]:
            source["reasons"].append(
                "No supported measured fields; retained as experimental/auxiliary evidence, not silently dropped."
            )
        inventory.append(source)
    if used_refs != set(refs):
        raise ValueError(
            f"Explicit references did not resolve to measurements: {sorted(set(refs) - used_refs)}"
        )
    # The expected benchmark matrix exists independently of file discovery.
    # No legacy row is automatically asserted to fill a future baseline slot.
    for cell in reg["cells"]:
        sid = "baseline/" + cell["id"]
        setup = {
            "id": sid,
            "label": cell["id"] + " · baseline pending",
            "dataset": cell["dataset"],
            "model": cell["model"],
            "instrument": cell["instrument"],
            "configuration_id": "baseline-not-defined",
            "configuration": settings({}, {}),
            "role": "planned_baseline",
        }
        if setup["instrument"] is None:
            setup["unknowns"] = {
                "instrument": "Instrument-independent or not yet declared by the baseline campaign."
            }
        setups.append(setup)
        for device in reg["devices"]:
            for slot in reg["slots"]:
                planned_cells.append(
                    {
                        "setup_id": sid,
                        "device": device,
                        **slot,
                        "status": "not_measured",
                        "record_id": None,
                        "reason": "Baseline stack, method and reference host not yet chosen; legacy evidence has not been accepted as this baseline.",
                    }
                )
    # These extension sections preserve evidence that cannot honestly satisfy
    # v2's measured-version/applicability or peak-memory semantics yet.
    result = {
        "schema": "profiling-summary",
        "version": 2,
        "project": "autolens_profiling",
        "scope": "setup-catalogue",
        "generated_at": generated,
        "evidence_updated_at": None,
        "evidence_updated_at_reason": "No common measurement wall-clock across legacy evidence; inspect each record.",
        "valid_until": None,
        "producer_revision": revision,
        "comparison_policy": {"id": "no-temporal-comparisons"},
        "coverage": {
            "expected": {
                "cells": len(selections),
                "planned_slots": len(planned_cells),
                "baseline_cells": len(reg["cells"]),
            },
            "observed": {
                "setups": len(setups),
                "records": len(records),
                "selected": sum(s["record_id"] is not None for s in selections),
            },
            "excluded": [],
        },
        "setups": setups,
        "records": records,
        "selections": selections,
        "comparisons": [],
        "hazards": [],
        "recommendations": [],
        "limitations": [
            "Legacy evidence remains unreviewed; no new baseline has been accepted.",
            "Source-row isolation prevents cross-file joins where complete configuration identity is unknown.",
            "The v1 production feed remains published during migration; this is the companion v2 catalogue.",
            "Indexed-only evidence and static estimates are not successful measurements or measured peak VRAM.",
        ],
        "inventory": inventory,
        "planned_cells": planned_cells,
        "static_memory_estimates": estimates,
        "unbound_findings": dedupe_findings(findings),
        "navigation": script_inventory(root, reg),
    }
    if (root / "catalogue/script_routes.json").exists():
        result["script_routes"] = source_routes(root)
    apply_bindings(root, reg, result)
    validate_local(root, result)
    return result


def missing_slot(setup, ident, host, axis, metric, unit, status, reason):
    seed = [setup["id"], ident, host, axis, metric]
    out = {
        "id": "slot/" + digest(seed),
        "setup_id": setup["id"],
        "axis": axis,
        "metric": metric,
        "unit": unit,
        "method_id": "pending-baseline/" + axis,
        "identity": ident,
        "host": host,
        "record_id": None,
        "status": status,
        "reason": reason,
    }
    if host is None:
        out["unknowns"] = {"host": "No reference host chosen/measured for this cell."}
    return out


def dedupe_findings(items):
    # Prefer the canonical consumer index when it contains the same finding.
    result = {}
    for item in sorted(items, key=lambda x: x["evidence"]["path"].endswith("hazards_index.json")):
        result[item["id"]] = item
    return list(sorted(result.values(), key=lambda x: x["id"]))


def source_routes(root):
    # The stdlib reader validates routes without importing any scientific leaf.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from _script_routes import load_routes

    return load_routes(root)


def script_inventory(root, reg):
    result = []
    declared = (
        source_routes(root)["routes"] if (root / "catalogue/script_routes.json").exists() else []
    )
    canonical = {row["path"]: row for row in declared}
    legacy = {row["legacy"] for row in declared}
    for path in sorted((root / "scripts").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(root / "scripts").parts
        if "__pycache__" in parts or path.name == "__init__.py" or rel in legacy:
            continue
        if parts[0] == "misc":
            if parts[1] in ("test", "tooling"):
                continue
            category = "shared_measurement_tools"
        else:
            category = "scientific_entrypoint"
        if rel in canonical:
            row = canonical[rel]
            result.append(
                {
                    "path": rel,
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "measurement": row["measurement"],
                    "legacy_paths": [row["legacy"]],
                    "category": "scientific_helper" if path.name.startswith("_") else category,
                }
            )
        else:
            dataset, model, _instrument = route(rel, {}, reg)
            result.append({"path": rel, "dataset": dataset, "model": model, "category": category})
    return result


def apply_bindings(root, reg, result):
    # No global hazard or recommendation applicability is guessed. A maintainer
    # can bind source findings to exact exported setup IDs and measured versions.
    by_finding = {f["id"]: f for f in result["unbound_findings"]}
    for binding in reg["hazard_bindings"]:
        if binding.get("finding_id") not in by_finding:
            raise ValueError("Unknown hazard binding finding")
        finding = by_finding[binding["finding_id"]]
        result["hazards"].append(
            {
                "id": finding["id"],
                "title": finding["title"],
                "description": finding["description"],
                "status": "unknown",
                "applies_to": binding["applies_to"],
                "evidence": [finding["evidence"]],
            }
        )
    for rec in reg["recommendations"]:
        # A recommendation needs explicit supporting record IDs and validation;
        # it is not extracted from narrative headlines by pattern matching.
        result["recommendations"].append(rec)


def resolve_pointer(doc, fragment):
    """Resolve a JSON pointer, including escaped keys, against original evidence."""
    if fragment in (None, ""):
        return doc
    if not isinstance(fragment, str) or not fragment.startswith("/"):
        raise ValueError("Invalid evidence JSON pointer")
    for part in fragment[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        doc = doc[int(key)] if isinstance(doc, list) else doc[key]
    return doc


def validate_local(root, doc):
    """Producer integrity checks; CI also uses the independent Pulse reader."""
    json.dumps(doc, allow_nan=False)
    indexed = {}
    for key in ("setups", "records", "selections", "hazards", "recommendations"):
        values = {v["id"]: v for v in doc[key]}
        if len(values) != len(doc[key]):
            raise ValueError(f"Duplicate {key} id")
        indexed[key] = values
    sources = {}
    for record in doc["records"]:
        if record["setup_id"] not in indexed["setups"]:
            raise ValueError("Unknown setup reference")
        ev = record["evidence"]
        if ev["path"] not in sources:
            sources[ev["path"]] = load_json(source_path(root, ev["path"]))
        value = resolve_pointer(sources[ev["path"]], ev["fragment"])
        if value != record["measurement"][record["metric"]]:
            raise ValueError("Measurement does not match original evidence")
    for item in doc["selections"]:
        if item["setup_id"] not in indexed["setups"]:
            raise ValueError("Unknown selection setup")
        if item["record_id"] is not None:
            ref = indexed["records"][item["record_id"]]
            if (
                any(item[k] != ref[k] for k in ("setup_id", "axis", "metric", "unit", "identity"))
                or item["method_id"] != ref["method"]["id"]
                or item["host"] != ref["provenance"]["host"]
                or item["status"] != ref["validation"]["status"]
            ):
                raise ValueError("Selection identity mismatch")
    for item in doc["hazards"] + doc["recommendations"]:
        scope = item["applies_to"]
        if (
            not scope.get("setup_ids")
            or not scope.get("library_versions")
            or any(not text(v) for v in scope["library_versions"])
            or not text(scope.get("limitations"))
            or not isinstance(scope.get("constraints"), dict)
            or not item.get("evidence")
            or not text(item.get("title"))
            or not text(item.get("description"))
        ):
            raise ValueError("Finding needs explicit versioned applicability and evidence")
        if any(s not in indexed["setups"] for s in item["applies_to"]["setup_ids"]):
            raise ValueError("Unknown finding applicability setup")
        for ev in item["evidence"]:
            if not source_path(root, ev["path"]).is_file():
                raise ValueError("Missing finding evidence")
            if ev["path"].endswith(".json"):
                try:
                    resolve_pointer(load_json(source_path(root, ev["path"])), ev.get("fragment"))
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ValueError("Invalid finding evidence pointer") from exc

    for rec in doc["recommendations"]:
        validation = rec.get("validation", {})
        if validation.get("status") not in ("unreviewed", "accepted", "rejected") or not text(
            validation.get("reason")
        ):
            raise ValueError("Recommendation needs explicit validation")
        if not rec.get("record_ids"):
            raise ValueError("Recommendation needs supporting records")
        for rid in rec["record_ids"]:
            record = indexed["records"].get(rid)
            if (
                record is None
                or record["setup_id"] not in rec["applies_to"]["setup_ids"]
                or record["identity"]["library_version"]
                not in rec["applies_to"]["library_versions"]
                or (
                    validation["status"] == "accepted"
                    and record["validation"]["status"] != "accepted"
                )
            ):
                raise ValueError("Recommendation support is missing, mismatched or unaccepted")


def render_outputs(root, generated, revision):
    """Index plus independently valid setup shards; no 36 MB landing fetch."""
    import copy

    full = build(root, generated, revision)
    by_setup = {}
    for record in full["records"]:
        by_setup.setdefault(record["setup_id"], []).append(record)
    outputs = {}
    shards = []
    envelope_keys = (
        "schema",
        "version",
        "project",
        "scope",
        "generated_at",
        "evidence_updated_at",
        "evidence_updated_at_reason",
        "valid_until",
        "producer_revision",
        "comparison_policy",
        "limitations",
    )
    for setup in full["setups"]:
        records = by_setup.get(setup["id"], [])
        if not records:
            continue
        selected = [s for s in full["selections"] if s["setup_id"] == setup["id"]]
        shard = {k: full[k] for k in envelope_keys}
        shard.update(
            setups=[setup],
            records=records,
            selections=selected,
            hazards=[],
            recommendations=[],
            comparisons=[],
            coverage={
                "expected": {"cells": len(selected)},
                "observed": {
                    "setups": 1,
                    "records": len(records),
                    "selected": sum(s["record_id"] is not None for s in selected),
                },
                "excluded": [],
            },
        )
        path = "catalogue/shards/" + digest(setup["id"]) + ".json"
        content = json.dumps(shard, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        outputs[path] = content
        shards.append(
            {
                "setup_id": setup["id"],
                "path": path,
                "records": len(records),
                "axes": sorted({r["axis"] for r in records}),
                "devices": sorted(
                    {r["identity"]["device"] or "device not recorded" for r in records}
                ),
                "precisions": sorted(
                    {r["identity"]["precision"] or "precision not recorded" for r in records}
                ),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    index = copy.deepcopy(full)
    referenced = {s["record_id"] for s in full["selections"] if s["record_id"] is not None}
    for recommendation in full["recommendations"]:
        referenced.update(recommendation["record_ids"])
    index["records"] = [r for r in full["records"] if r["id"] in referenced]
    index["coverage"]["observed"]["records"] = len(index["records"])
    index["evidence_shards"] = shards
    index["archive_record_count"] = len(full["records"])
    validate_local(root, index)
    outputs["catalogue.json"] = (
        json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return outputs


def obsolete_shards(out, outputs):
    return [
        p
        for p in (out / "catalogue/shards").glob("*.json")
        if re.fullmatch(r"[a-f0-9]{20}\.json", p.name)
        and p.relative_to(out).as_posix() not in outputs
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--validate-with",
        type=Path,
        help="Pulse checkout: validate using the independent merged contract reader",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = root / "dashboard/catalogue.json"
    # Reuse the established deterministic render stamp and producer revision.
    legacy = load_json(root / "dashboard/summary.json")
    outputs = render_outputs(root, legacy["generated_at"], legacy["producer_revision"])
    if args.validate_with:
        sys.path.insert(0, str(args.validate_with.resolve()))
        from pulse.summary import classify

        for name, content in outputs.items():
            outcome, errors = classify(json.loads(content), ("profiling-summary", 2))
            if outcome != "ok":
                raise ValueError(
                    f"Pulse v2 validation failed for {name}: " + "; ".join(errors[:20])
                )
    stale = [
        name
        for name, content in outputs.items()
        if not (out.parent / name).is_file() or (out.parent / name).read_text() != content
    ]
    obsolete = obsolete_shards(out.parent, outputs)
    if args.check:
        if stale or obsolete:
            print(
                f"catalogue: STALE ({len(stale)} changed/missing, {len(obsolete)} obsolete shards); run build_catalogue.py"
            )
            return 1
        print("catalogue: current")
        return 0
    for name, content in outputs.items():
        path = out.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for path in obsolete:
        path.unlink()
    doc = json.loads(outputs["catalogue.json"])
    print(
        f"catalogue: wrote {len(doc['setups'])} setups, {doc['archive_record_count']} measurements, {len(doc['inventory'])} evidence files, {len(outputs) - 1} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
