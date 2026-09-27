"""Gate: the results tree keeps to the artefact policy in ``results/README.md``.

Run from the repo root::

    python scripts/misc/tooling/check_results_layout.py                     # report
    python scripts/misc/tooling/check_results_layout.py --check             # exit 1 on any failure
    python scripts/misc/tooling/check_results_layout.py --write-grandfather # (re)write the manifest

Checks
------

(a) ``results/notes/`` is the ledger tree: its top level holds ``*.md`` files only, plus
    the frozen evidence-pack folders named in ``ALLOWED_NOTES_DIRS`` (each with its own
    README). Job logs belong under ``results/logs/<campaign>/``; JSON sidecars belong
    beside the result JSONs they describe.
(b) Every result JSON under ``results/`` that records its JAX device (a top-level
    ``device`` object, i.e. written through ``_profile_cli.device_info_dict``) carries the
    provenance block ``device.provenance`` with every key in ``REQUIRED_PROVENANCE_KEYS``
    — unless the file is listed in the grandfather manifest
    ``results/provenance_grandfathered.txt`` (result JSONs written before the block
    existed). The manifest is a ratchet: lines may be removed, never added by hand;
    ``--write-grandfather`` regenerates it from the files that currently lack the block
    and is meant to run once, at the policy's birth.
(c) Every manifest line names a file that still exists (a moved or deleted file leaves a
    dead line; drop it).

Pure stdlib -- no PyAuto* imports, no PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Folders under results/notes/ that are not ledgers: recovered evidence packs, frozen
#: with their own README. Nothing new is added here; a new campaign's artefacts go
#: beside its result JSONs.
ALLOWED_NOTES_DIRS = frozenset(
    {
        "clipper_campaign",
        "point_source_cpu_2026_09_17_reported",
    }
)

#: The keys ``_profile_cli.provenance_dict`` writes; the wiki index and the run-time
#: dashboard read them. Keep in step with ``_profile_cli.PROVENANCE_KEYS``.
REQUIRED_PROVENANCE_KEYS = (
    "provenance_schema",
    "captured_at",
    "host",
    "slurm",
    "loadavg_at_import",
    "loadavg_at_write",
    "profiling_revision",
    "library_revisions",
    "library_versions",
    "dependency_versions",
)

MANIFEST = Path("results") / "provenance_grandfathered.txt"
MANIFEST_HEADER = """\
# Result JSONs written before the provenance block existed (2026-09-28).
# check_results_layout.py --check exempts these paths from the `device.provenance`
# requirement. This is a ratchet: remove a line when its file gains the block or is
# deleted; never add one by hand. Regenerate only with --write-grandfather.
"""


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def _result_jsons(root: Path):
    """Every JSON under results/ outside the ledger tree, sorted, repo-relative."""
    results = root / "results"
    if not results.is_dir():
        return []
    out = []
    for p in sorted(results.rglob("*.json")):
        rel = p.relative_to(root)
        if rel.parts[:2] == ("results", "notes"):
            continue
        out.append(rel)
    return out


def _device_block(root: Path, rel: Path):
    """The top-level ``device`` object of a result JSON, or None (absent / not an object)."""
    try:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    device = data.get("device")
    return device if isinstance(device, dict) else None


def missing_provenance(device: dict) -> list[str]:
    """Which required provenance keys a ``device`` block lacks (all of them if no block)."""
    prov = device.get("provenance")
    if not isinstance(prov, dict):
        return list(REQUIRED_PROVENANCE_KEYS)
    return [k for k in REQUIRED_PROVENANCE_KEYS if k not in prov]


def read_manifest(root: Path) -> list[str]:
    path = root / MANIFEST
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def check(root: Path) -> list[str]:
    """Return a list of human-readable failures (empty when the tree is consistent)."""
    root = root.resolve()
    failures: list[str] = []

    # (a) results/notes/ holds ledgers only.
    notes = root / "results" / "notes"
    if notes.is_dir():
        for entry in sorted(notes.iterdir()):
            rel = entry.relative_to(root)
            if entry.is_dir():
                if entry.name not in ALLOWED_NOTES_DIRS:
                    failures.append(f"(a) folder under results/notes/ not allowlisted: {rel}/")
            elif entry.suffix != ".md":
                failures.append(
                    f"(a) non-markdown file under results/notes/: {rel} "
                    "(logs -> results/logs/<campaign>/, sidecars -> beside their result JSON)"
                )

    # (b) every device-recording result JSON carries the provenance block.
    grandfathered = set(read_manifest(root))
    for rel in _result_jsons(root):
        device = _device_block(root, rel)
        if device is None or str(rel) in grandfathered:
            continue
        missing = missing_provenance(device)
        if missing:
            failures.append(
                f"(b) {rel}: device.provenance missing {', '.join(missing)} "
                "(write it through _profile_cli.device_info_dict)"
            )

    # (c) the manifest names only files that exist.
    for line in sorted(grandfathered):
        if not (root / line).is_file():
            failures.append(f"(c) dead line in {MANIFEST}: {line} (file gone; drop the line)")

    return failures


def write_grandfather(root: Path) -> int:
    """Write the manifest from the device-recording JSONs that currently lack the block."""
    root = root.resolve()
    lines = []
    for rel in _result_jsons(root):
        device = _device_block(root, rel)
        if device is not None and missing_provenance(device):
            lines.append(str(rel))
    (root / MANIFEST).write_text(MANIFEST_HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on any failure")
    parser.add_argument(
        "--write-grandfather",
        action="store_true",
        help="regenerate results/provenance_grandfathered.txt from the current tree",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="repo root (default: the ruff.toml dir)"
    )
    args = parser.parse_args(argv)
    root = args.root or _profiling_root()

    if args.write_grandfather:
        n = write_grandfather(root)
        print(f"check_results_layout: wrote {MANIFEST} ({n} grandfathered result JSONs)")
        return 0

    failures = check(root)
    if failures:
        print(f"check_results_layout: {len(failures)} failure(s)")
        for f in failures:
            print(f"  {f}")
    else:
        n_json = len(_result_jsons(root))
        n_gf = len(read_manifest(root))
        print(
            f"check_results_layout: OK (results/notes/ is markdown-only; "
            f"{n_json} result JSONs, {n_gf} grandfathered)"
        )
    return 1 if (failures and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
