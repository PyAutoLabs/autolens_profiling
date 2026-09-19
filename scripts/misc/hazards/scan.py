"""Run the numerical-hazard vertical slice and compare it with committed findings.

Examples::

    python scripts/misc/hazards/scan.py
    python scripts/misc/hazards/scan.py --subject component --backend jax
    python scripts/misc/hazards/scan.py --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
MISC_ROOT = REPO_ROOT / "scripts" / "misc"
if str(MISC_ROOT) not in sys.path:
    sys.path.insert(0, str(MISC_ROOT))

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    import hazards

    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

from hazards._record import (  # noqa: E402
    Finding,
    compare_finding_ids,
    load_index,
    write_grouped_findings,
)
from hazards._report import write_ell_comps_seed_summary, write_finding_plots  # noqa: E402
from hazards.aggregate import (  # noqa: E402
    build_index_from_records,
    render_markdown_table,
    write_index,
)
from hazards.checks import CHECKS  # noqa: E402
from hazards.checks._base import ScanContext  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--subject",
        choices=("all", "component", "matrix", "likelihood"),
        default="all",
    )
    parser.add_argument("--backend", choices=("both", "numpy", "jax"), default="both")
    parser.add_argument("--sample-count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "hazards")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-run without writing; exit 1 only when a new semantic finding ID appears.",
    )
    parser.add_argument(
        "--inject-finding",
        action="store_true",
        help="Inject a synthetic new ID to exercise the regression failure path.",
    )
    return parser.parse_args(argv)


def _selected_backends(name: str) -> tuple[str, ...]:
    return ("numpy", "jax") if name == "both" else (name,)


def _workspace_root(repo_root: Path) -> Path:
    return next(
        (path for path in repo_root.parents if (path / ".pyauto-root").is_file()),
        repo_root.parent,
    )


def run_scan(args: argparse.Namespace) -> list[Finding]:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    context = ScanContext(
        repo_root=REPO_ROOT,
        workspace_root=_workspace_root(REPO_ROOT),
        output_root=args.output_root,
        backends=_selected_backends(args.backend),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    findings = []
    for check in CHECKS:
        if check.applies_to(args.subject):
            print(f"[{check.subject}] {check.name}", flush=True)
            findings.extend(check.run(context))
    return findings


def _check(findings: list[Finding], args: argparse.Namespace) -> int:
    index_path = args.output_root / "hazards_index.json"
    baseline = load_index(index_path)
    baseline_ids = {
        finding_id
        for finding_id, finding in baseline.items()
        if args.subject == "all" or finding.get("subject") == args.subject
    }
    current_ids = {finding.finding_id for finding in findings}
    if args.inject_finding:
        current_ids.add("synthetic.regression.injected-finding")
    diff = compare_finding_ids(current_ids, baseline_ids)
    print(f"persistent: {len(diff.persistent)}")
    for finding_id in diff.new:
        print(f"NEW: {finding_id}")
    for finding_id in diff.resolved:
        print(f"resolved: {finding_id}")
    if diff.has_new:
        print(
            "ERROR: new numerical-hazard finding ID(s); inspect before accepting.", file=sys.stderr
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    findings = run_scan(args)
    if args.check:
        return _check(findings, args)

    written = write_grouped_findings(findings, args.output_root)
    plots = write_finding_plots(findings, args.output_root)
    payload = build_index_from_records(args.output_root)
    index_path = args.output_root / "hazards_index.json"
    write_index(payload, index_path)
    seed_path = args.output_root / "ell_comps_clamp.md"
    seed = next(
        (
            finding
            for finding in findings
            if finding.finding_id == "component.ell_comps.magnitude-saturation"
        ),
        None,
    )
    if seed is not None:
        write_ell_comps_seed_summary(seed, seed_path)

    import autolens as al

    from hazards._profiles import discover_profile_registry, write_coverage

    coverage_path = args.output_root / "component" / "profile_registry_coverage.json"
    write_coverage(discover_profile_registry(al), coverage_path)

    print(render_markdown_table(payload))
    print(
        f"\nwrote {len(written)} record(s), {len(plots)} plot(s), "
        f"{index_path.relative_to(REPO_ROOT)} and {coverage_path.relative_to(REPO_ROOT)}"
        + (f", and {seed_path.relative_to(REPO_ROOT)}" if seed is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
