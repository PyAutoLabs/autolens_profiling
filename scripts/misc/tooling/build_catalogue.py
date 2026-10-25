"""Generate the profiling evidence catalogues (stdlib only).

Run ``python scripts/misc/tooling/build_catalogue.py [--check]``. Two documents come out of
one pass over the mirrored results tree:

- ``dashboard/catalogue.json``: the ``profiling-summary@3`` lifecycle catalogue Pulse reads
  (likelihood, setup_id, host_class, class/status by the day-zero rule; see
  ``catalogue_lifecycle.py``).
- ``dashboard/catalogue/setups.json`` + ``catalogue/shards/``: the companion v2 setup catalogue
  behind this repo's setup browser and setup wiki.

The registry owns navigation and explicit reference choices; adapters only transcribe
recorded measurements. No script imports, timings, latest-row choice or scientific acceptance
occur here. See catalogue/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalogue_lifecycle as lc
import parity_witness
from catalogue_adapters import metrics, number, pointer, rows

REGISTRY = "catalogue/registry.json"
#: The companion v2 index, relative to ``dashboard/`` (its shards sit beside it).
SETUPS_INDEX = "catalogue/setups.json"
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
    cell = lc.mirrored(path)
    if cell is not None:
        # The mirrored tree names its likelihood and instrument; payload labels never override.
        model = reg["model_aliases"].get(cell["variant"], cell["variant"])
        return cell["dataset_type"], model, cell["instrument"]
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


def identity(payload, path, releases=None, revisions_from=None):
    """``revisions_from``: the payload whose provenance names the release (default ``payload``)."""
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
    tag = lc.release_from_revisions(payload if revisions_from is None else revisions_from, releases)
    if tag is not None and tag != version:
        software["release_pin_basis"] = (
            f"library_revisions equal tag {tag}'s SHAs ({lc.RELEASE_REVISIONS}); the source "
            f"checkout stamped autolens_version {version}"
        )
        version = tag
    if text(version):
        software["PyAutoLens"] = version
    else:
        version = None
    if text(payload.get("jax_version")):
        software["jax"] = payload["jax_version"]
    hardware = payload.get("hardware")
    backend = device.get("backend") or payload.get("backend")
    # Only explicit, documented config/hardware labels are used as hints; a mirrored cell's
    # folder names are its setup and host class.
    cell = lc.mirrored(path)
    stem = f"{cell['setup_id']}_{cell['host_class']}" if cell else Path(path).stem
    label = next((x for x in (payload.get("_config_label"), hardware, stem) if text(x)), "")
    precision = None
    mixed = payload.get("use_mixed_precision", payload.get("mixed_precision"))
    if not isinstance(mixed, bool) and isinstance(payload.get("precision"), dict):
        mixed = payload["precision"].get("use_mixed_precision")
    if isinstance(mixed, bool):
        precision = "mixed" if mixed else "float64"
    elif "_fp64" in label:
        precision = "float64"
    elif re.search(r"_(?:mp|mixed)(?:_|$)", label):
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
    for path, adapter in sorted(files.items(), key=lambda kv: Path(kv[0])):
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
        cell = lc.mirrored(path)
        if cell is not None and cell["file_measurement"] == "parity":
            source["reasons"].append(
                "Parity witness: read into the v3 lifecycle catalogue as measurement parity, "
                "not a v2 timing."
            )
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
            # Tooling, the pytest conftest and each group's test/ dir are not measurement tools.
            if parts[1] in ("tooling", "conftest.py") or "test" in parts[2:-1]:
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
    # Discovery is editorial navigation, not scientific applicability. Keep it
    # on unbound findings and never turn it into a version-qualified hazard.
    discovery = reg.get("hazard_discovery", [])
    if not isinstance(discovery, list):
        raise ValueError("registry.hazard_discovery must be a list")
    known_models = {(s["dataset"], s["model"]) for s in result["setups"]}
    seen = set()
    for item in discovery:
        fid = item.get("finding_id")
        models = item.get("models")
        shared = item.get("shared")
        if (
            fid not in by_finding
            or fid in seen
            or not isinstance(models, list)
            or not isinstance(shared, bool)
            or shared == bool(models)
            or not text(item.get("reason"))
        ):
            raise ValueError("Invalid/duplicate hazard discovery entry")
        scopes = set()
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("Invalid hazard discovery model")
            scope = (model.get("dataset"), model.get("model"))
            if scope not in known_models or scope in scopes:
                raise ValueError("Unknown/duplicate hazard discovery model")
            scopes.add(scope)
        seen.add(fid)
        by_finding[fid]["discovery"] = {
            "models": models,
            "shared": shared,
            "reason": item["reason"],
        }
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


# ---------------------------------------------------------------- profiling-summary@3 ---

#: First-commit dates (UTC) of the result files with an undated row: {path: "YYYY-MM-DD"}.
#: Written by ``--refresh-commit-dates`` (needs git history); read, never computed, at build
#: and ``--check`` time, because CI checks out without history.
COMMIT_DATES = "catalogue/measured_at_first_commit.json"


def commit_dates(root):
    path = source_path(root, COMMIT_DATES)
    if not path.is_file():
        return {}
    doc = load_json(path)
    if doc.get("schema") != "measured-at-first-commit" or not isinstance(doc.get("dates"), dict):
        raise ValueError(f"Unsupported {COMMIT_DATES}")
    return doc["dates"]


class UndatedRows(ValueError):
    """Undated rows whose file has no recorded first-commit date."""

    def __init__(self, paths):
        self.paths = sorted(set(paths))
        super().__init__(
            f"{len(self.paths)} result file(s) have undated rows and no first-commit date in "
            f"{COMMIT_DATES} (e.g. {self.paths[0]}); run "
            "`python scripts/misc/tooling/build_catalogue.py --refresh-commit-dates` "
            "in a checkout with full git history and commit the file"
        )


V3_LIMITATIONS = [
    "Phase-0 labels (autolens_profiling#442): nothing was re-measured. A current record is the "
    "canonical cell row on a reference host class for a planned baseline cell; it is not an "
    "accepted baseline and carries no day_zero stamp.",
    "No designation decision exists yet: provisional Optimal rows are class secondary with "
    "'provisional_optimal: true' in their limitations (R3); planned_cells carry the "
    "provisional class.",
    "Standard source-pixel counts are Delaunay 1250 and rectangular 1225 (35x35). No committed "
    "row is at the standard size; current pixelized rows carry a size_fallback limitation.",
    "Records whose key has no current row (laptop-only, experiment-only, or outside the "
    "planned cells) are not rendered; coverage.excluded counts them per key. Every row stays "
    "in the companion v2 setup catalogue (dashboard/catalogue/setups.json).",
    "Historical recommendations and hazard findings name v2 setups and stay in the companion "
    "catalogue until they are re-bound to v3 setup_ids.",
    "Parity records (measurement parity) transcribe a cell's parity.json witness: metric "
    "abs_delta_log_likelihood = |log L - reference log L| beside its tolerance, the signed "
    "values and the witness status in the record's parity block. A not_assessed witness has no "
    "value and yields no record; parity_witnesses lists every witness file.",
]


def planned_baseline_cells(reg):
    """{(dataset_type, variant, instrument)} of the registry's planned baseline cells."""
    cells = set()
    for cell in reg["cells"]:
        instrument = cell.get("instrument") or lc.DEFAULT_INSTRUMENTS.get(cell["dataset"])
        if not text(instrument):
            raise ValueError(f"v3 planned cell needs an instrument: {cell['id']}")
        cells.add((cell["dataset"], cell["model"], instrument))
    return cells


def lifecycle_candidates(root, reg, path, doc, adapter, dates=None, undated=None, releases=None):
    """(v3 records, reasons) for one mirrored source file; nothing is labelled yet."""
    cell = lc.mirrored(path)
    out, reasons = [], []
    if cell is None:
        return out, ["Outside the mirrored tree: no lifecycle key; companion catalogue only."]
    if cell["file_measurement"] is None:
        return out, ["Job sidecar, not a measurement file: no lifecycle record."]
    figure = source_path(root, path).with_name("fit.png")
    figures = []
    if figure.is_file():
        figures.append(
            {
                "kind": "fit",
                "path": figure.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
            }
        )
    cls, notes = lc.record_class(
        cell["dataset_type"], cell["variant"], cell["host_class"], cell["setup_id"]
    )
    for payload, base, _context in rows(doc, adapter):
        groups = {}
        for axis, metric, unit, value, anchor, statistic, n in metrics(payload, adapter, base):
            if not number(value):
                reasons.append(f"{anchor}: missing/non-finite/negative measurement refused.")
                continue
            measurement = lc.measurement_of(axis, metric, cell["file_measurement"])
            groups.setdefault(measurement, {}).setdefault(
                metric, (value, anchor, statistic, n, unit)
            )
        if not groups:
            continue
        ident, prov = identity(payload, path, releases)
        if not ident["software"]:
            reasons.append(f"{base or '/'}: no measured software version; no v3 record.")
            continue
        measured_at, row_notes = prov["measured_at"], list(notes)
        if measured_at is None:
            # v3 needs a UTC measured_at; a row without one is dated by the day its file was
            # first committed (an upper bound on when it was measured), and says so.
            day = (dates or {}).get(path)
            if day is None:
                if undated is not None:
                    undated.append(path)
                continue
            measured_at = f"{day}T00:00:00Z"
            row_notes.append(
                f"measured_at_derived: the row records no timestamp; {day} is the UTC date its "
                "file was first committed to git, an upper bound on the measurement date"
            )
        pixels = lc.size_pixels(cell["variant"], payload)
        for measurement, found in sorted(groups.items()):
            values = {k: v[0] for k, v in sorted(found.items())}
            head = lc.headline(measurement, values)
            _value, _anchor, statistic, n, unit = found[head]
            out.append(
                {
                    "id": "lifecycle/" + digest([path, base, measurement]),
                    "run_id": "legacy/" + digest([path, base]),
                    "likelihood": {
                        "dataset_type": cell["dataset_type"],
                        "variant": cell["variant"],
                    },
                    "instrument": cell["instrument"],
                    "regime": lc.regime(cell["variant"], pixels),
                    "setup": {**cell["setup"], "threads": None},
                    "setup_id": cell["setup_id"],
                    "host_class": cell["host_class"],
                    "model_variant": cell["model_variant"],
                    "measurement": measurement,
                    "metric": head,
                    "unit": unit,
                    "values": values,
                    "class": cls,
                    "status": None,
                    "superseded_by": None,
                    "day_zero": None,
                    "figures": figures,
                    "decision_id": None,
                    "release_pin": ident["library_version"],
                    "measured_at": measured_at,
                    "identity": ident,
                    "method": method(payload, measurement, head, statistic, n),
                    "provenance": prov,
                    "validation": {
                        "status": "unreviewed",
                        "reason": "Legacy evidence labelled by the phase-0 day-zero rule; not an accepted baseline.",
                    },
                    "evidence": {"path": path, "fragment": base or None},
                    "pointers": {k: v[1] for k, v in sorted(found.items())},
                    "location": cell["location"],
                    "limitations": list(row_notes),
                    "_rank": (
                        lc.size_rank(cell["variant"], pixels),
                        lc.FILE_ORDER[cell["file_measurement"]],
                    ),
                    "_pixels": pixels,
                }
            )
    return out, reasons


def parity_candidate(root, path, doc, releases=None):
    """(v3 records, witness entry) for one mirrored ``parity.json`` (parity_witness.py)."""
    cell = lc.mirrored(path)
    errors = parity_witness.validate(doc)
    if errors:
        raise ValueError(f"Invalid parity witness {path}: " + "; ".join(errors))
    entry = {
        "path": path,
        "status": doc["status"],
        "likelihood": {"dataset_type": cell["dataset_type"], "variant": cell["variant"]},
        "instrument": cell["instrument"],
        "setup_id": cell["setup_id"],
        "host_class": cell["host_class"],
        "model_variant": cell["model_variant"],
        "location": cell["location"],
        "record_id": None,
    }
    if doc["status"] == "not_assessed":
        return [], entry
    cls, notes = lc.record_class(
        cell["dataset_type"], cell["variant"], cell["host_class"], cell["setup_id"]
    )
    # A witness records no library revisions: its release is the one its cell's rows name,
    # when they stamp the same autolens_version (the same run wrote them).
    sibling = None
    for name in ("runtime.json", "breakdown.json"):
        candidate = source_path(root, path).with_name(name)
        if candidate.is_file():
            payload = load_json(candidate)
            if payload.get("autolens_version") == doc["autolens_version"]:
                sibling = payload
                break
    ident, prov = identity(doc, path, releases, revisions_from=sibling or {})
    prov = {**prov, "measured_at": doc["measured_at"]}
    prov["unknowns"] = {k: v for k, v in prov["unknowns"].items() if k != "measured_at"}
    pixels = lc.size_pixels(cell["variant"], doc)
    record = {
        "id": "lifecycle/" + digest([path, "", "parity"]),
        "run_id": "parity/" + digest([path]),
        "likelihood": entry["likelihood"],
        "instrument": cell["instrument"],
        "regime": lc.regime(cell["variant"], pixels),
        "setup": {**cell["setup"], "threads": None},
        "setup_id": cell["setup_id"],
        "host_class": cell["host_class"],
        "model_variant": cell["model_variant"],
        "measurement": "parity",
        "metric": parity_witness.METRIC,
        "unit": "dimensionless",
        "values": {parity_witness.METRIC: parity_witness.delta(doc), "tolerance": doc["tolerance"]},
        "class": cls,
        "status": None,
        "superseded_by": None,
        "day_zero": None,
        "figures": [],
        "decision_id": None,
        "release_pin": ident["library_version"],
        "measured_at": doc["measured_at"],
        "identity": ident,
        "method": {
            "id": "parity-witness",
            "statistic": None,
            "repetitions": None,
            "warmup": None,
            "synchronization": None,
            "cache_state": None,
            "unknowns": {
                k: "A parity witness compares one log-likelihood evaluation; not a timing."
                for k in ("statistic", "repetitions", "warmup", "synchronization", "cache_state")
            },
        },
        "provenance": prov,
        "validation": {
            "status": "unreviewed",
            "reason": f"Parity witness status {doc['status']} (|delta log L| against tolerance, "
            "recorded by the producing run); not an accepted baseline.",
        },
        "evidence": {"path": path, "fragment": None},
        "parity": {
            "status": doc["status"],
            "log_likelihood": doc["log_likelihood"],
            "reference_log_likelihood": doc["reference_log_likelihood"],
            "reference": doc["reference"],
        },
        "pointers": {"tolerance": "/tolerance"},
        "location": cell["location"],
        "limitations": list(notes),
        "_rank": (lc.size_rank(cell["variant"], pixels), lc.FILE_ORDER["parity"]),
        "_pixels": pixels,
    }
    entry["record_id"] = record["id"]
    return [record], entry


def label_lifecycle(records, planned, refs):
    """Assign status/superseded_by by the day-zero rule; return (kept, excluded entries)."""
    by_key = {}
    for record in records:
        by_key.setdefault(lc_key(record), []).append(record)
    current = {}
    for key, items in by_key.items():
        eligible = [
            r
            for r in items
            if r["location"] == "cell"
            and r["host_class"] not in lc.LOCAL_HOSTS
            and (
                (r["likelihood"]["dataset_type"], r["likelihood"]["variant"], r["instrument"])
                in planned
                or r["evidence"]["path"] in refs
            )
        ]
        if eligible:
            best = min(eligible, key=lambda r: (r["_rank"], r["id"]))
            best["status"] = "current"
            if best["_rank"][0][0] != 0:
                best["limitations"].append(
                    lc.size_note(best["likelihood"]["variant"], best["_pixels"])
                )
            current[key] = best
    kept, orphans = [], {}
    for key, items in sorted(by_key.items()):
        for record in items:
            if record["status"] == "current":
                kept.append(record)
                continue
            target = current.get(key)
            if target is None and record["host_class"] in lc.LOCAL_HOSTS:
                ref_host = lc.REFERENCE_HOSTS[record["setup"]["device"]]
                target = current.get(lc_key({**record, "host_class": ref_host}))
            if target is None:
                orphans.setdefault(key, []).append(record)
                continue
            record["status"] = "superseded"
            record["superseded_by"] = target["id"]
            kept.append(record)
    excluded = []
    for key, items in sorted(orphans.items()):
        first = items[0]
        why = (
            "laptop/local-GPU rows only (R7: never current)"
            if all(r["host_class"] in lc.LOCAL_HOSTS for r in items)
            else "scan/experiment/history rows only (never current)"
            if all(r["location"] != "cell" for r in items)
            else "likelihood x instrument is not a planned baseline cell"
        )
        excluded.append(
            {
                "id": "no-current/" + digest(key),
                "reason": f"No current record on this key: {why}. Kept in the companion v2 catalogue.",
                "record_count": len(items),
                "key": json.loads(key),
                "evidence": {"path": first["evidence"]["path"]},
            }
        )
    for record in kept:
        record.pop("_rank")
        record.pop("_pixels")
    return sorted(kept, key=lambda r: r["id"]), excluded


def lc_key(record):
    return json.dumps(
        {
            k: record[k]
            for k in (
                "likelihood",
                "instrument",
                "setup_id",
                "host_class",
                "model_variant",
                "measurement",
            )
        },
        sort_keys=True,
    )


def archived_count(root, path, doc, adapter):
    """The v3 records an archived file would yield (counted, never rendered)."""
    count = 0
    stem = Path(path).stem
    for payload, base, _context in rows(doc, adapter):
        found = {
            lc.measurement_of(axis, metric, stem if stem in lc.FILE_MEASUREMENTS else None)
            for axis, metric, _u, value, *_ in metrics(payload, adapter, base)
            if number(value)
        }
        count += len(found)
    return count


def build_lifecycle(root, generated, revision, dates=None):
    """The ``profiling-summary@3`` document (Pulse REFERENCE.md "profiling-summary v3")."""
    root = Path(root).resolve()
    reg = registry(root)
    dates = commit_dates(root) if dates is None else dates
    releases = lc.release_revisions(root)
    undated = []
    planned = planned_baseline_cells(reg)
    refs = {r["path"] for r in reg["references"]}
    candidates, archived, witnesses = [], 0, []
    for spec in reg["sources"]:
        if spec["adapter"] != "profile":
            continue
        for p in sorted(root.glob(spec["glob"])):
            path = p.relative_to(root).as_posix()
            override = next((s for s in reg["overrides"] if path.startswith(s["prefix"])), {})
            adapter = override.get("adapter", spec["adapter"])
            if adapter not in ("profile", "streaming"):
                continue
            doc = load_json(source_path(root, path))
            if path.startswith("archive/"):
                archived += archived_count(root, path, doc, adapter)
                continue
            cell = lc.mirrored(path)
            if cell is not None and cell["file_measurement"] == "parity":
                found, witness = parity_candidate(root, path, doc, releases)
                witnesses.append(witness)
            else:
                found, _reasons = lifecycle_candidates(
                    root, reg, path, doc, adapter, dates, undated, releases
                )
            candidates.extend(found)
    if undated:
        raise UndatedRows(undated)
    records, excluded = label_lifecycle(candidates, planned, refs)
    planned_cells = []
    for dataset_type, variant, instrument in sorted(planned):
        for host_class in ("hpc_a100", "hpc_ral_cpu"):
            for sid, cls in lc.planned_setups(dataset_type, variant, host_class).items():
                for measurement in ("runtime", "breakdown"):
                    planned_cells.append(
                        {
                            "likelihood": {"dataset_type": dataset_type, "variant": variant},
                            "instrument": instrument,
                            "setup_id": sid,
                            "host_class": host_class,
                            "model_variant": "default",
                            "measurement": measurement,
                            "class": cls,
                        }
                    )
    counts = {s: sum(r["status"] == s for r in records) for s in ("current", "superseded")}
    doc = {
        "schema": "profiling-summary",
        "version": 3,
        "project": "autolens_profiling",
        "scope": "likelihood-lifecycle",
        "generated_at": generated,
        "evidence_updated_at": None,
        "evidence_updated_at_reason": "No common measurement wall-clock across legacy evidence; inspect each record's measured_at.",
        "valid_until": None,
        "producer_revision": revision,
        "comparison_policy": {"id": "no-temporal-comparisons"},
        "coverage": {
            "expected": {"cells": len(planned_cells)},
            "observed": {"records": len(records), **counts},
            "excluded": excluded,
        },
        "records": records,
        "planned_cells": planned_cells,
        "archive_record_count": archived,
        "parity_witnesses": witnesses,
        "standard_source_pixels": dict(sorted(lc.STANDARD_SOURCE_PIXELS.items())),
        "comparisons": [],
        "hazards": [],
        "recommendations": [],
        "limitations": V3_LIMITATIONS,
    }
    validate_lifecycle(root, doc)
    return doc


def validate_lifecycle(root, doc):
    """Producer integrity checks for v3; CI also runs the independent Pulse reader."""
    json.dumps(doc, allow_nan=False)
    ids = {r["id"] for r in doc["records"]}
    if len(ids) != len(doc["records"]):
        raise ValueError("Duplicate lifecycle record id")
    current, sources = {}, {}
    for record in doc["records"]:
        if record["status"] == "current":
            if lc_key(record) in current:
                raise ValueError(f"Second current record on one key: {record['id']}")
            if record["host_class"] in lc.LOCAL_HOSTS or record["location"] != "cell":
                raise ValueError(f"Ineligible current record: {record['id']}")
            current[lc_key(record)] = record["id"]
        elif record["status"] != "superseded" or record["superseded_by"] not in ids:
            raise ValueError(f"Superseded record without a target: {record['id']}")
        if record["class"] == "optimal" and not text(record["decision_id"]):
            raise ValueError("class optimal needs a designation decision_id")
        ev = record["evidence"]
        if ev["path"] not in sources:
            sources[ev["path"]] = load_json(source_path(root, ev["path"]))
        for metric, fragment in record["pointers"].items():
            if resolve_pointer(sources[ev["path"]], fragment) != record["values"][metric]:
                raise ValueError("Lifecycle value does not match original evidence")
        if record["measurement"] == "parity":
            witness = sources[ev["path"]]
            if (
                record["values"][parity_witness.METRIC] != parity_witness.delta(witness)
                or record["parity"]["status"] != witness["status"]
            ):
                raise ValueError("Parity record does not match its parity.json witness")


def render_outputs(root, generated, revision):
    """The v3 lifecycle catalogue, plus the companion v2 index and its setup shards."""
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
                "axis_devices": {
                    axis: sorted(
                        {
                            r["identity"]["device"] or "device not recorded"
                            for r in records
                            if r["axis"] == axis
                        }
                    )
                    for axis in sorted({r["axis"] for r in records})
                },
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
    outputs[SETUPS_INDEX] = (
        json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    lifecycle = build_lifecycle(root, generated, revision)
    outputs["catalogue.json"] = (
        json.dumps(lifecycle, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return outputs


def obsolete_shards(out, outputs):
    return [
        p
        for p in (out / "catalogue/shards").glob("*.json")
        if re.fullmatch(r"[a-f0-9]{20}\.json", p.name)
        and p.relative_to(out).as_posix() not in outputs
    ]


def first_commit_date(root, path, moved):
    """UTC date the file was first added to git under its own name.

    A file the day-zero ``git mv`` moved is looked up at its exact pre-move path from
    ``catalogue/migration_day_zero.json`` (``git log --follow`` mis-tracks some of these into an
    older, similar file); any other file at its own path. Refresh mode only.
    """
    import subprocess

    target = moved.get(path, path)
    log = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%cI", "--", target],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not log:
        raise ValueError(f"No add commit in git history for {target} (shallow clone?)")
    oldest = min(datetime.fromisoformat(stamp) for stamp in log)
    return oldest.astimezone(UTC).date().isoformat()


def refresh_commit_dates(root):
    """Record the first-commit date of every result file that has an undated row."""
    try:
        build_lifecycle(root, "1970-01-01T00:00:00Z", "0" * 40, dates={})
        paths = []
    except UndatedRows as missing:
        paths = missing.paths
    day_zero = load_json(source_path(root, "catalogue/migration_day_zero.json"))
    moved = {e["new"]: e["old"] for e in day_zero["entries"] if e["new"] != e["old"]}
    dates = {path: first_commit_date(root, path, moved) for path in paths}
    doc = {
        "schema": "measured-at-first-commit",
        "version": 1,
        "note": "UTC date each result file was first committed (git log --diff-filter=A, at "
        "its pre-move path for files the day-zero move renamed); the measured_at of its "
        "undated rows, an upper bound. Regenerate with build_catalogue.py "
        "--refresh-commit-dates.",
        "dates": dict(sorted(dates.items())),
    }
    target = source_path(root, COMMIT_DATES)
    target.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"catalogue: wrote {COMMIT_DATES} ({len(dates)} files)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--validate-with",
        type=Path,
        help="Pulse checkout: validate with its independent contract reader (v3 lifecycle, v2 companion)",
    )
    parser.add_argument(
        "--refresh-commit-dates",
        action="store_true",
        help=f"rewrite {COMMIT_DATES} from git history (first-add date, through the day-zero move)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.refresh_commit_dates:
        return refresh_commit_dates(root)
    out = root / "dashboard/catalogue.json"
    # Reuse the established deterministic render stamp and producer revision.
    legacy = load_json(root / "dashboard/summary.json")
    try:
        outputs = render_outputs(root, legacy["generated_at"], legacy["producer_revision"])
    except UndatedRows as missing:
        print(f"catalogue: FAILED: {missing}")
        return 1
    if args.validate_with:
        sys.path.insert(0, str(args.validate_with.resolve()))
        from pulse.summary import classify

        for name, content in outputs.items():
            version = 3 if name == "catalogue.json" else 2
            outcome, errors = classify(json.loads(content), ("profiling-summary", version))
            if outcome != "ok":
                raise ValueError(
                    f"Pulse v{version} validation failed for {name}: " + "; ".join(errors[:20])
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
    doc = json.loads(outputs[SETUPS_INDEX])
    v3 = json.loads(outputs["catalogue.json"])
    observed = v3["coverage"]["observed"]
    print(
        f"catalogue: wrote v3 {observed['current']} current + {observed['superseded']} superseded "
        f"records, {v3['archive_record_count']} archived, {len(v3['planned_cells'])} planned cells; "
        f"companion v2 {len(doc['setups'])} setups, {doc['archive_record_count']} measurements, "
        f"{len(doc['inventory'])} evidence files, {len(outputs) - 2} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
