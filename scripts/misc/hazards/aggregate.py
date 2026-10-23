"""Aggregate per-check hazard records into the consumer-facing index.

Run from the repository root::

    python scripts/misc/hazards/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_MISC_ROOT = _profiling_root() / "scripts" / "misc"
if str(_MISC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MISC_ROOT))

from hazards._record import SCHEMA_VERSION, Finding, findings_from_record  # noqa: E402


def build_index_from_findings(findings: list[Finding]) -> dict:
    indexed = {}
    for finding in sorted(findings, key=lambda item: item.finding_id):
        record = finding.to_dict()
        record["record"] = str(finding.record_relative_path)
        indexed[finding.finding_id] = record
    return {"schema_version": SCHEMA_VERSION, "findings": indexed}


def build_index_from_records(output_root: Path) -> dict:
    indexed: dict[str, dict] = {}
    for path in sorted(output_root.rglob("*.json")):
        if path.name == "hazards_index.json":
            continue
        for finding in findings_from_record(path):
            finding_id = finding.get("finding_id")
            if not finding_id:
                continue
            finding["record"] = str(path.relative_to(output_root))
            indexed[finding_id] = finding
    return {"schema_version": SCHEMA_VERSION, "findings": dict(sorted(indexed.items()))}


def write_index(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def render_markdown_table(payload: dict) -> str:
    rows = [
        "| Finding | Subject | Hazard | Risk basis | Backends |",
        "|---|---|---|---|---|",
    ]
    for finding_id, finding in payload.get("findings", {}).items():
        bases = ", ".join(
            sorted({measurement["basis"] for measurement in finding.get("measurements", [])})
        )
        rows.append(
            f"| `{finding_id}` | `{finding.get('subject')}` | "
            f"`{finding.get('hazard_class')}` | {bases or '—'} | "
            f"{', '.join(finding.get('backends', [])) or '—'} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = _profiling_root()
    output_root = args.output_root or repo_root / "results" / "misc" / "hazards"
    payload = build_index_from_records(output_root)
    index_path = output_root / "hazards_index.json"
    write_index(payload, index_path)
    print(render_markdown_table(payload))
    print(f"\n-> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
