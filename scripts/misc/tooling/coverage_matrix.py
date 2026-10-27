"""The lifecycle coverage matrix: planned cells x what the catalogue holds (stdlib only).

Run ``python scripts/misc/tooling/coverage_matrix.py [--check]``. It reads the
``profiling-summary@3`` catalogue (``dashboard/catalogue.json``) and the machine copy of the
lifecycle report's §3.3 provisional designations (``catalogue/provisional_designations.json``)
and writes ``dashboard/coverage.json`` + ``dashboard/coverage.md``: per likelihood x class x
measurement, how many planned cells are current, stale, superseded or missing, plus whether
each planned setup has a parity witness. ``--check`` fails when the committed outputs differ
from a fresh render, or when a planned cell's class disagrees with the §3.3 copy.

Cell states (the PyAutoBrain profiling conductor's vocabulary, ``CELL_STATES``):

- **current**: the key has a ``current`` record on a reference host class, measured at the
  standard size, whose ``release_pin`` is not behind the catalogue pin.
- **stale**: as current, but the record's ``release_pin`` is behind the catalogue pin (or is
  not recorded). The catalogue pin is the envelope's ``release_pin`` when it declares one, else
  the newest ``release_pin`` any record carries: the same rule the Brain conductor applies.
- **superseded**: the key has only ``superseded`` records.
- **missing**: nothing on the key, **or** its current record is a ``size_fallback`` row
  (human ruling 2026-10-23: a standard-size cell covered only by a non-standard size row is
  missing until the standard size is measured).

Coverage is standard-size only: a declared size variant's rows (``catalogue/size_variants.json``,
a ``size_variant`` limitation; current beside the standard row on their own key) and its
parity witnesses are not read here.

A parity witness (``parity_witness.py``) is reported per planned setup cell: ``match`` /
``diverged`` from its current ``measurement: parity`` record, ``not_assessed`` from the
catalogue's ``parity_witnesses`` list, else ``missing``. Nothing here re-measures, ranks or
designates; class and status are the producer's declarations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CATALOGUE = "dashboard/catalogue.json"
DESIGNATIONS = "catalogue/provisional_designations.json"
OUT_JSON = "dashboard/coverage.json"
OUT_MD = "dashboard/coverage.md"
CELL_STATES = ("current", "stale", "superseded", "missing")
PARITY_STATES = ("match", "diverged", "not_assessed", "missing")
LOCAL_PREFIX = "local_"
SIZE_FALLBACK = "size_fallback"
SIZE_VARIANT = "size_variant"


def pin_order(pin):
    """Sortable form of a release pin (``2026.10.16.2``); None when unparsable."""
    if not isinstance(pin, str):
        return None
    try:
        return tuple(int(part) for part in pin.split("."))
    except ValueError:
        return None


def key_of(row, measurement=None):
    lk = row["likelihood"]
    return (
        lk["dataset_type"],
        lk["variant"],
        row["instrument"],
        row["setup_id"],
        row["host_class"],
        row.get("model_variant", "default"),
        measurement or row["measurement"],
    )


def cell_name(key):
    dataset_type, variant, instrument, setup_id, host, model_variant, measurement = key
    note = "" if model_variant == "default" else f" {model_variant}"
    return f"{dataset_type}/{variant}/{instrument} {setup_id}@{host}{note} [{measurement}]"


# ---- §3.3 provisional designations (machine copy) ----


def load_designations(doc):
    if doc.get("schema") != "provisional-designations" or doc.get("version") != 1:
        raise ValueError(f"{DESIGNATIONS}: unsupported schema/version")
    for name, table in doc["tables"].items():
        for host in ("hpc_a100", "hpc_ral_cpu", "laptop"):
            if set(table.get(host, {})) != {"optimal", "secondary"}:
                raise ValueError(f"{DESIGNATIONS}: table {name} lacks {host} optimal/secondary")
    for row in doc["rows"]:
        for route in row["routes"]:
            if route["table"] not in doc["tables"]:
                raise ValueError(f"{DESIGNATIONS}: route names unknown table {route['table']}")
    return doc


def table_for(doc, dataset_type, variant):
    """The §3.3 table of a likelihood: an exact variant route beats a ``*`` route."""
    wildcard = None
    for row in doc["rows"]:
        for route in row["routes"]:
            if route["dataset_type"] != dataset_type:
                continue
            if route["variants"] == "*":
                wildcard = wildcard or route["table"]
            elif variant in route["variants"]:
                return route["table"]
    return wildcard


def designated_class(doc, dataset_type, variant, host_class, setup_id):
    """``optimal`` / ``secondary`` / ``experimental`` per the §3.3 copy."""
    name = table_for(doc, dataset_type, variant)
    if name is None or "_mixed" in setup_id:
        return "experimental"
    host = {"local_laptop": "laptop", "local_gpu": "hpc_a100"}.get(host_class, host_class)
    table = doc["tables"][name][host]
    if setup_id in table["optimal"]:
        return "optimal"
    return "secondary" if setup_id in table["secondary"] else "experimental"


# ---- the matrix ----


def catalogue_pin(catalogue):
    pin = catalogue.get("release_pin")
    if pin_order(pin) is not None:
        return pin, "envelope"
    pins = [r.get("release_pin") for r in catalogue["records"]]
    pins = [p for p in pins if pin_order(p) is not None]
    return (max(pins, key=pin_order), "newest record") if pins else (None, None)


def is_size_fallback(record):
    return any(str(note).startswith(SIZE_FALLBACK) for note in record.get("limitations", []))


def is_size_variant(record):
    """A declared size variant's row: measured beside the standard size, never coverage."""
    return any(str(note).startswith(SIZE_VARIANT) for note in record.get("limitations", []))


def cell_state(records, pin):
    """(state, reason, record id) of one planned key from its records."""
    current = [
        r
        for r in records
        if r["status"] == "current" and not r["host_class"].startswith(LOCAL_PREFIX)
    ]
    if current:
        record = current[0]
        if is_size_fallback(record):
            return "missing", f"size_fallback ({record['regime']})", record["id"]
        own = pin_order(record.get("release_pin"))
        if pin is not None and (own or ()) < pin_order(pin):
            return "stale", f"release_pin {record.get('release_pin')} behind {pin}", record["id"]
        return "current", None, record["id"]
    if records:
        return "superseded", "only superseded records", None
    return "missing", "no record", None


def build(catalogue, designations):
    designations = load_designations(designations)
    pin, pin_source = catalogue_pin(catalogue)
    by_key = {}
    for record in catalogue["records"]:
        if not is_size_variant(record):
            by_key.setdefault(key_of(record), []).append(record)
    planned = {}
    for cell in catalogue["planned_cells"]:
        planned[key_of(cell)] = cell["class"]
    cells, groups, mismatches = [], {}, []
    for key, declared in sorted(planned.items()):
        state, reason, record = cell_state(by_key.get(key, []), pin)
        expected = designated_class(designations, key[0], key[1], key[4], key[3])
        if expected != declared:
            mismatches.append({"cell": cell_name(key), "planned": declared, "s3_3": expected})
        cells.append(
            {
                "cell": cell_name(key),
                "likelihood": f"{key[0]}/{key[1]}",
                "instrument": key[2],
                "setup_id": key[3],
                "host_class": key[4],
                "model_variant": key[5],
                "measurement": key[6],
                "class": declared,
                "state": state,
                "reason": reason,
                "record": record,
            }
        )
        group = groups.setdefault(
            (f"{key[0]}/{key[1]}", declared, key[6]), dict.fromkeys(CELL_STATES, 0)
        )
        group[state] += 1

    # Parity witnesses, one per planned setup cell (the key without its measurement).
    witnesses = {}
    for entry in catalogue.get("parity_witnesses", []):
        if entry.get("location") == "cell" and not entry.get("size_variant"):
            witnesses[key_of(entry, "parity")] = entry
    setup_cells = sorted({k[:6] for k in planned})
    parity_cells, parity_groups = [], {}
    for base in setup_cells:
        key = (*base, "parity")
        current = [r for r in by_key.get(key, []) if r["status"] == "current"]
        if current:
            state, record = current[0]["parity"]["status"], current[0]["id"]
        elif key in witnesses:
            state, record = witnesses[key]["status"], None
        else:
            state, record = "missing", None
        cls = max(
            (planned[k] for k in planned if k[:6] == base),
            key=("experimental", "secondary", "optimal").index,
        )
        parity_cells.append(
            {"cell": cell_name(key), "class": cls, "state": state, "record": record}
        )
        group = parity_groups.setdefault(
            (f"{base[0]}/{base[1]}", cls), dict.fromkeys(PARITY_STATES, 0)
        )
        group[state] += 1

    totals = {s: sum(c["state"] == s for c in cells) for s in CELL_STATES}
    totals["missing_size_fallback"] = sum(
        c["state"] == "missing" and (c["reason"] or "").startswith(SIZE_FALLBACK) for c in cells
    )
    # Parity rows are tallied per setup cell above (their key is never a planned cell), so they
    # are not "unplanned" (#451 review).
    unplanned = sorted(
        r["id"]
        for r in catalogue["records"]
        if r["status"] == "current"
        and r["measurement"] != "parity"
        and not is_size_variant(r)
        and key_of(r) not in planned
    )
    return {
        "schema": "profiling-coverage",
        "version": 1,
        "sources": {"catalogue": CATALOGUE, "designations": DESIGNATIONS},
        "catalogue_generated_at": catalogue.get("generated_at"),
        "catalogue_release_pin": pin,
        "release_pin_source": pin_source,
        "rules": {
            "current": "a current record on a reference host, at the standard size, whose "
            "release_pin is not behind the catalogue pin",
            "stale": "a current standard-size record whose release_pin is behind the catalogue "
            "pin (the envelope release_pin, else the newest record pin) or is not recorded",
            "superseded": "only superseded records on the key",
            "missing": "no record, or the current record is a size_fallback row (human ruling "
            "2026-10-23: standard-size cells covered only by size_fallback rows are missing)",
            "parity": "per planned setup cell: match/diverged from its current parity record, "
            "not_assessed from parity_witnesses, else missing",
        },
        "totals": {
            "planned_cells": len(cells),
            **totals,
            "records_superseded": sum(r["status"] == "superseded" for r in catalogue["records"]),
            "current_records_unplanned": len(unplanned),
        },
        "matrix": [
            {"likelihood": lk, "class": cls, "measurement": m, **group}
            for (lk, cls, m), group in sorted(groups.items())
        ],
        "parity": {
            "totals": {
                "setup_cells": len(parity_cells),
                **{s: sum(c["state"] == s for c in parity_cells) for s in PARITY_STATES},
            },
            "matrix": [
                {"likelihood": lk, "class": cls, **group}
                for (lk, cls), group in sorted(parity_groups.items())
            ],
            "cells": parity_cells,
        },
        "class_mismatches": mismatches,
        "unplanned_current_records": unplanned,
        "cells": cells,
    }


def render_md(cov):
    t, p = cov["totals"], cov["parity"]["totals"]
    lines = [
        "# Lifecycle coverage",
        "",
        "<!-- Generated by scripts/misc/tooling/coverage_matrix.py from dashboard/catalogue.json "
        "and catalogue/provisional_designations.json; do not edit. -->",
        "",
        f"Catalogue generated {cov['catalogue_generated_at']}; catalogue pin "
        f"`{cov['catalogue_release_pin']}` ({cov['release_pin_source']}).",
        "",
        "| State | Rule |",
        "|---|---|",
    ]
    lines += [f"| {state} | {cov['rules'][state]} |" for state in CELL_STATES]
    lines += [
        "",
        "## Totals",
        "",
        "| Planned cells | Current | Stale | Superseded | Missing (of which size_fallback) "
        "| Superseded records |",
        "|---|---|---|---|---|---|",
        f"| {t['planned_cells']} | {t['current']} | {t['stale']} | {t['superseded']} | "
        f"{t['missing']} ({t['missing_size_fallback']}) | {t['records_superseded']} |",
        "",
        "## Likelihood x class x measurement",
        "",
        "| Likelihood | Class | Measurement | Current | Stale | Superseded | Missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in cov["matrix"]:
        lines.append(
            f"| {row['likelihood']} | {row['class']} | {row['measurement']} | {row['current']} "
            f"| {row['stale']} | {row['superseded']} | {row['missing']} |"
        )
    lines += [
        "",
        "## Parity witnesses",
        "",
        cov["rules"]["parity"].capitalize()
        + " (convention: the Parity witness section of the results README).",
        "",
        "| Setup cells | Match | Diverged | Not assessed | Missing |",
        "|---|---|---|---|---|",
        f"| {p['setup_cells']} | {p['match']} | {p['diverged']} | {p['not_assessed']} | "
        f"{p['missing']} |",
        "",
        "| Likelihood | Class | Match | Diverged | Not assessed | Missing |",
        "|---|---|---|---|---|---|",
    ]
    for row in cov["parity"]["matrix"]:
        lines.append(
            f"| {row['likelihood']} | {row['class']} | {row['match']} | {row['diverged']} | "
            f"{row['not_assessed']} | {row['missing']} |"
        )
    if cov["class_mismatches"]:
        lines += ["", "## Class mismatches against §3.3", ""]
        lines += [
            f"- {m['cell']}: planned {m['planned']}, §3.3 {m['s3_3']}"
            for m in cov["class_mismatches"]
        ]
    return "\n".join(lines) + "\n"


def render(root):
    catalogue = json.loads((root / CATALOGUE).read_text())
    if catalogue.get("schema") != "profiling-summary" or catalogue.get("version") != 3:
        raise ValueError(f"{CATALOGUE} is not a profiling-summary@3 catalogue")
    designations = json.loads((root / DESIGNATIONS).read_text())
    cov = build(catalogue, designations)
    return cov, {
        OUT_JSON: json.dumps(cov, indent=1, sort_keys=True) + "\n",
        OUT_MD: render_md(cov),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cov, outputs = render(root)
    if cov["class_mismatches"]:
        print(
            f"coverage: FAILED: {len(cov['class_mismatches'])} planned cell class(es) disagree "
            f"with {DESIGNATIONS}, e.g. {cov['class_mismatches'][0]}"
        )
        return 1
    stale = [
        n for n, c in outputs.items() if not (root / n).is_file() or (root / n).read_text() != c
    ]
    if args.check:
        if stale:
            print(f"coverage: STALE ({', '.join(stale)}); run coverage_matrix.py")
            return 1
        print("coverage: current")
        return 0
    for name, content in outputs.items():
        (root / name).write_text(content)
    t = cov["totals"]
    print(
        f"coverage: {t['planned_cells']} planned cells: {t['current']} current, {t['stale']} "
        f"stale, {t['superseded']} superseded, {t['missing']} missing "
        f"({t['missing_size_fallback']} size_fallback); parity "
        f"{cov['parity']['totals']['missing']}/{cov['parity']['totals']['setup_cells']} missing"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
