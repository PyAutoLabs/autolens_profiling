"""Export ``catalogue/runtime_estimates.yaml`` from the PyAutoPulse day-zero snapshot (stdlib only).

Run from the repo root, with a PyAutoPulse checkout beside it::

    python scripts/misc/tooling/export_runtime_estimates.py --pulse ../PyAutoPulse
    python scripts/misc/tooling/export_runtime_estimates.py --pulse ../PyAutoPulse --check

The estimator of likelihood lifecycle phase 5E (``al.Setup.estimate``) reads a copy of this file
as ``autolens/config/runtime_estimates.yaml``. Each entry is one stamped day-zero runtime cell:
the likelihood, hardware class and instrument, the setup and its role (optimal / secondary from
the Pulse designation, or the held cell's provisional role), the measured milliseconds per
likelihood evaluation, what one evaluation occupies, and the projected throughput of one node.
Nothing is measured, ranked or extrapolated here; the snapshot's ``records_sha256`` is verified
before anything is read. Expected evaluation counts come from Insight (phase 7), not from here.
autolens_profiling#503.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = "catalogue/runtime_estimates.yaml"
SNAPSHOT = "snapshots/day_zero/lens.json"
HARDWARE = {"hpc_ral_cpu": "cpu_many_core", "hpc_a100": "gpu_a100"}
ROLE_ORDER = ("optimal", "secondary", "provisional_optimal", "provisional_secondary")
HEADER = (
    "# generated: scripts/misc/tooling/export_runtime_estimates.py from the PyAutoPulse\n"
    "# day-zero snapshot; do not edit. Copied into PyAutoLens as\n"
    "# autolens/config/runtime_estimates.yaml by likelihood lifecycle phase 5E.\n"
)


class ExportError(ValueError):
    pass


def _digest(value) -> str:
    """The snapshot's ``records_sha256`` (PyAutoPulse ``pulse/freeze.py``)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _sig(value, digits: int = 6):
    return None if value is None else float(f"{value:.{digits}g}")


def _role(record: dict) -> str:
    if record["decision_id"]:
        return record["class"]
    for note in record["limitations"]:
        for role in ("provisional_optimal", "provisional_secondary"):
            if note.startswith(f"{role}:"):
                return role
    raise ExportError(f"{record['id']}: stamped record with neither a decision nor a hold")


def load_snapshot(pulse: Path) -> dict:
    snap = json.loads((Path(pulse) / SNAPSHOT).read_text())
    if snap.get("schema") != "pulse-frozen-snapshot" or snap.get("label") != "day_zero":
        raise ExportError(f"{SNAPSHOT} is not the day_zero frozen snapshot")
    if snap.get("records_sha256") != _digest(snap["records"]):
        raise ExportError(f"{SNAPSHOT}: records changed since the freeze (records_sha256)")
    return snap


def estimates(snap: dict) -> list[dict]:
    rows = []
    for r in snap["records"]:
        if r["measurement"] != "runtime":
            continue
        throughput = r.get("projected_node_throughput") or {}
        node = next(
            (h for h in throughput.get("hosts") or [] if h.get("host_class") == r["host_class"]),
            {},
        )
        resources = r.get("resources") or {}
        per_call = throughput.get("per_call_s")
        rows.append(
            {
                "likelihood": f"{r['likelihood']['dataset_type']}/{r['likelihood']['variant']}",
                "hardware_class": HARDWARE[r["host_class"]],
                "host_class": r["host_class"],
                "instrument": r["instrument"],
                "setup_id": r["setup_id"],
                "role": _role(r),
                "decision_id": r["decision_id"],
                "ms_per_eval": _sig(per_call * 1e3) if per_call else None,
                "per_eval_source": throughput.get("per_call_source"),
                "cpus_per_eval": resources.get("cpus_per_eval"),
                "devices_per_eval": resources.get("devices_per_eval"),
                "peak_rss_gb": _sig(resources.get("peak_rss_gb"), 4),
                "workers_per_node": node.get("workers"),
                "bound_by": node.get("bound_by"),
                "calls_per_s_per_node": _sig(node.get("calls_per_s"), 4),
                "record_id": r["id"],
            }
        )
    rows.sort(
        key=lambda e: (
            e["likelihood"],
            e["hardware_class"],
            e["instrument"],
            ROLE_ORDER.index(e["role"]),
            e["setup_id"],
        )
    )
    return rows


def _scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)  # a double-quoted YAML scalar


def render(snap: dict) -> str:
    rows = estimates(snap)
    efficiency = next(
        (
            (r.get("projected_node_throughput") or {}).get("parallel_efficiency")
            for r in snap["records"]
            if (r.get("projected_node_throughput") or {}).get("parallel_efficiency")
        ),
        {},
    )
    lines = [
        "schema: runtime-estimates",
        "version: 1",
        f"tag: {_scalar(snap['tag'])}",
        "source:",
        '  repo: "PyAutoLabs/PyAutoPulse"',
        f"  path: {_scalar(SNAPSHOT)}",
        f"  profiling_commit: {_scalar(snap['commit'])}",
        f"  records_sha256: {_scalar(snap['records_sha256'])}",
        "parallel_efficiency:",
        f"  value: {_scalar(efficiency.get('value'))}",
        f"  kind: {_scalar(efficiency.get('kind'))}",
        f"  source: {_scalar(efficiency.get('source'))}",
        f"cells: {len(rows)}",
        "estimates:",
    ]
    for row in rows:
        first = True
        for key, value in row.items():
            lines.append(f"{'  - ' if first else '    '}{key}: {_scalar(value)}")
            first = False
    return HEADER + "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--pulse", type=Path, required=True, help="a PyAutoPulse checkout holding the snapshot"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        text = render(load_snapshot(args.pulse))
    except (OSError, ExportError) as exc:
        print(f"runtime_estimates: FAILED: {exc}")
        return 1
    out = args.root.resolve() / OUT
    current = out.is_file() and out.read_text() == text
    if args.check:
        print(
            "runtime_estimates: current"
            if current
            else f"runtime_estimates: STALE; run export_runtime_estimates.py --pulse {args.pulse}"
        )
        return 0 if current else 1
    out.write_text(text)
    print(f"runtime_estimates: wrote {OUT} ({text.count(chr(10) + '  - ')} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
