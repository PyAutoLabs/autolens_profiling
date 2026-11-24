"""Export ``catalogue/runtime_estimates.yaml`` from the PyAutoPulse day-zero snapshot (stdlib only).

Run from the repo root, with a PyAutoPulse checkout beside it::

    python scripts/misc/tooling/export_runtime_estimates.py --pulse ../PyAutoPulse
    python scripts/misc/tooling/export_runtime_estimates.py --pulse ../PyAutoPulse --check
    python scripts/misc/tooling/export_runtime_estimates.py --pulse ../PyAutoPulse \
        --copy-to ../PyAutoLens

The estimator of likelihood lifecycle phase 5E (``al.Setup.estimate``) reads a copy of this file
as ``autolens/config/runtime_estimates.yaml``. Each entry is one stamped day-zero runtime cell:
the likelihood, hardware class and instrument, the setup and its role (optimal / secondary from
the Pulse designation, or the held cell's provisional role), the measured milliseconds per
likelihood evaluation, what one evaluation occupies, and the projected throughput of one node.
Nothing is measured, ranked or extrapolated here; the snapshot's ``records_sha256`` is verified
before anything is read. Expected evaluation counts come from Insight (phase 7), not from here.
autolens_profiling#503.

Version 2 (likelihood lifecycle phase 5E-1, PyAutoLens#887) adds the dataset size each cell was
timed at, so a library can match a live dataset to an instrument row: ``n_pixels_masked``,
``n_vis`` and ``source_pixels``, read from the ``configuration`` block of the record's own
evidence file in this repo (the ``evidence.path`` the snapshot names), null where that block does
not record one. The header's ``parallel_efficiency`` is the measured numba-under-Nautilus curve
(:data:`PARALLEL_EFFICIENCY`). ``--copy-to`` writes the identical file into a PyAutoLens checkout.
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
#: numba's parallel efficiency under Nautilus's multiprocessing, by core count: a copy of
#: ``PARALLEL_EFFICIENCY`` in autolens_inference ``scripts/misc/wall/model.py`` (phase-4 cores
#: sweep: imaging/rectangular/hst, numba_cpu_sparse_fp64, source_pix[1], seed 0; RAL jobs
#: 414141, 413913, 414142, 413911, 414143, 414144). Embedded, not imported, because CI checks this
#: export without an autolens_inference checkout; test_runtime_estimates.py compares the two when
#: that checkout sits beside this repo.
PARALLEL_EFFICIENCY: dict[int, float] = {1: 1.0, 2: 0.68, 4: 0.62, 8: 0.55, 16: 0.36, 32: 0.28}
PARALLEL_EFFICIENCY_CELL = ("imaging/rectangular", "numba_cpu_sparse_fp64")
PARALLEL_EFFICIENCY_SOURCE = (
    "autolens_inference phase 4 (scripts/misc/wall/model.py PARALLEL_EFFICIENCY)"
)

#: The v2 size fields: row key -> (``configuration`` key(s) in the evidence file, meaning). The
#: first key present wins.
SIZE_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "n_pixels_masked": (
        ("image_pixels_masked",),
        "unmasked image pixels the cell was timed at (per dataset for multi_dataset, per "
        "channel for datacube; the real-space mask for interferometer)",
    ),
    "n_vis": (
        ("visibilities",),
        "visibilities the cell was timed at (per channel for datacube)",
    ),
    "source_pixels": (
        ("source_pixels_requested", "source_pixels"),
        "source-plane mesh pixels the profiling config fixes (the requested count; a Delaunay "
        "mesh may realise a few more vertices)",
    ),
}
COPY_TARGET = "autolens/config/runtime_estimates.yaml"
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


def sizes(record: dict, root: Path) -> dict:
    """The v2 size fields from the ``configuration`` block of the record's evidence file."""
    path = Path(root) / record["evidence"]["path"]
    try:
        config = json.loads(path.read_text()).get("configuration") or {}
    except (OSError, ValueError) as exc:
        raise ExportError(f"{record['id']}: evidence {record['evidence']['path']}: {exc}") from exc
    out = {}
    for field, (keys, _) in SIZE_FIELDS.items():
        value = next((config[k] for k in keys if config.get(k) is not None), None)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ExportError(f"{record['id']}: {field} = {value!r} is not an integer")
        out[field] = value
    return out


def estimates(snap: dict, root: Path = ROOT) -> list[dict]:
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
                **sizes(r, root),
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


def render(snap: dict, root: Path = ROOT) -> str:
    rows = estimates(snap, root)
    lines = [
        "schema: runtime-estimates",
        "version: 2",
        f"tag: {_scalar(snap['tag'])}",
        "source:",
        '  repo: "PyAutoLabs/PyAutoPulse"',
        f"  path: {_scalar(SNAPSHOT)}",
        f"  profiling_commit: {_scalar(snap['commit'])}",
        f"  records_sha256: {_scalar(snap['records_sha256'])}",
        "parallel_efficiency:",
        "  curve:",
        *(f"    {cores}: {_scalar(eta)}" for cores, eta in sorted(PARALLEL_EFFICIENCY.items())),
        f"  likelihood: {_scalar(PARALLEL_EFFICIENCY_CELL[0])}",
        f"  setup_id: {_scalar(PARALLEL_EFFICIENCY_CELL[1])}",
        '  kind: "measured"',
        f"  source: {_scalar(PARALLEL_EFFICIENCY_SOURCE)}",
        "size_fields:",
        *(f"  {field}: {_scalar(meaning)}" for field, (_, meaning) in SIZE_FIELDS.items()),
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
    parser.add_argument(
        "--copy-to",
        type=Path,
        metavar="PYAUTOLENS",
        help=f"also write the identical file to <PYAUTOLENS>/{COPY_TARGET}",
    )
    args = parser.parse_args(argv)
    if args.copy_to is not None and args.check:
        parser.error("--copy-to writes files; it cannot be combined with --check")
    if args.copy_to is not None and not (args.copy_to / "autolens").is_dir():
        print(f"runtime_estimates: FAILED: {args.copy_to} has no autolens/ package directory")
        return 1
    try:
        text = render(load_snapshot(args.pulse), args.root.resolve())
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
    if args.copy_to is not None:
        target = args.copy_to / COPY_TARGET
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"runtime_estimates: copied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
