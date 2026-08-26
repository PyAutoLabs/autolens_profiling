"""Aggregate per-config JSONs for swept search cells into comparison.{json,png}.

Walks the four-level layout written by ``searches/sweep.py``::

    <output_root>/<sampler>/<dataset_class>/<model>/<instrument>/<config_name>.json

For each ``<instrument>`` directory, emits a ``comparison.json`` (per-config
dict) and a ``comparison.png`` (grouped bar chart of the headline metrics
across configs: total_wall_s, viz_wall_s, sampler_wall_s, time_per_eval_ms).

**Eval-counter comparability (issue #177).** ``likelihood_evals`` counts
different things in different rows — see ``searches._metrics.eval_counter_basis``
— so the eval-derived metrics (``likelihood_evals``, ``time_per_eval_ms``) are
only ever rendered WITHIN one basis, never across two. A cell holding more than
one basis is reported loudly and exits non-zero rather than emitting a chart
that invites the comparison. Wall metrics are raw timers and stay comparable
throughout; ``max_log_likelihood`` and ``log_evidence`` are untouched.

Usage::

    # All cells under the default output root
    python searches/aggregate.py

    # One cell only
    python searches/aggregate.py --cell nautilus/imaging/mge/hst
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from searches._metrics import (  # noqa: E402
    EVAL_BASIS_UNKNOWN,
    basis_conflicts,
    eval_basis_label,
    eval_counter_basis,
    load_summary,
)

_REPO_ROOT = _profiling_root()
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "searches"


_CONFIG_ORDER = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)


# Raw timers — comparable across every row regardless of eval-counter basis.
_WALL_METRICS_FOR_BAR_CHART: tuple[tuple[str, str], ...] = (
    ("total_wall_s", "Total wall (s)"),
    ("sampler_wall_s", "Sampler wall (s)"),
    ("viz_wall_s", "Viz wall (s)"),
)

# How many filenames the conflict banner prints per basis before summarising.
_BANNER_MAX_NAMES = 6

# Derived from likelihood_evals, so only comparable WITHIN one basis (#177).
_EVAL_METRICS_FOR_BAR_CHART: tuple[tuple[str, str], ...] = (("time_per_eval_ms", "Per-eval (ms)"),)

_METRICS_FOR_BAR_CHART: tuple[tuple[str, str], ...] = (
    _WALL_METRICS_FOR_BAR_CHART + _EVAL_METRICS_FOR_BAR_CHART
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Root output dir. Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--cell",
        nargs="+",
        default=None,
        metavar="SAMPLER/CLASS/MODEL/INSTRUMENT",
        help=("Only aggregate these cells; default = auto-discover under --output-root."),
    )
    return p.parse_args()


def _is_search_payload(path: Path) -> bool:
    """Whether a JSON under a cell dir is a search run written by _runner.py.

    Mirrors the predicate ``build_readme._scan_search_artifacts`` already
    uses — identity comes from the payload's ``sampler`` key, never from the
    filename. Discovery previously accepted a cell only if it held one of the
    six ``_CONFIG_ORDER`` stems, which no longer describes how runs are named:
    a sweep arm is written as e.g. ``hpc_hpc_a100_fp64_n256_seed0.json``, so
    every MultiStart cell — including the one holding the mixed-basis pair
    this module now guards (#177) — was skipped by an auto-discovered run and
    reachable only via an explicit ``--cell``. Aggregation already read those
    files; only discovery disagreed.
    """
    if path.name == "comparison.json":
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "sampler" in payload


def _discover_cells(output_root: Path) -> list[tuple[str, str, str, str]]:
    """Find every <sampler>/<ds>/<model>/<instrument> dir with config JSONs."""
    cells: list[tuple[str, str, str, str]] = []
    if not output_root.exists():
        return cells

    def _has_config_json(d: Path) -> bool:
        return any(_is_search_payload(p) for p in d.glob("*.json"))

    for sampler_dir in sorted(output_root.iterdir()):
        if not sampler_dir.is_dir():
            continue
        for ds_dir in sorted(sampler_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for model_dir in sorted(ds_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for inst_dir in sorted(model_dir.iterdir()):
                    if inst_dir.is_dir() and _has_config_json(inst_dir):
                        cells.append((sampler_dir.name, ds_dir.name, model_dir.name, inst_dir.name))
    return cells


def _read_config(json_path: Path) -> dict:
    """Load one per-config JSON, normalised to the v2 shape in memory.

    ``load_summary`` synthesises the v2 ``target``/``algorithm``/``hardware``
    blocks for a v1 payload and stamps ``schema_version`` (missing key -> 1),
    so downstream code reads one shape. It preserves the recorded
    ``schema_version``, which is what ``eval_counter_basis`` reads — the
    normalisation deliberately does NOT pretend a v1 eval count is a v2 one.
    """
    data = load_summary(json_path)
    data.setdefault("config_name", json_path.stem)
    return data


def _aggregate_cell(cell_dir: Path) -> dict:
    configs: dict[str, dict] = {}
    for json_path in sorted(cell_dir.glob("*.json")):
        if json_path.name == "comparison.json":
            continue
        try:
            configs[json_path.stem] = _read_config(json_path)
        except Exception as exc:
            sys.stderr.write(f"  warn: failed to read {json_path}: {exc}\n")

    ordered: dict[str, dict] = {}
    for name in _CONFIG_ORDER:
        if name in configs:
            ordered[name] = configs.pop(name)
    for name in sorted(configs):
        ordered[name] = configs[name]

    bases = {name: eval_counter_basis(cfg) for name, cfg in ordered.items()}
    return {
        "configs": ordered,
        # Recorded per config AND summarised, so a reader of comparison.json
        # alone can tell whether its eval-derived columns are commensurable
        # without re-deriving the rule (#177).
        "eval_counter_bases": bases,
        "eval_counter_conflicts": basis_conflicts(ordered),
    }


def _format_seconds(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "—"
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 1e-3:
        return f"{t * 1e3:.1f}ms"
    return f"{t * 1e6:.0f}μs"


def _get_perf(cfg: dict, key: str) -> float:
    perf = cfg.get("performance", {})
    val = perf.get(key)
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    return float("nan")


def _conflict_banner(conflicts: dict[str, list[str]], cell_id: str) -> list[str]:
    """Loud, specific report of a cell whose rows use different eval counters.

    Names every file on both sides rather than saying "mixed" — the whole
    failure mode this guards is a number that looks fine, so the message has
    to make the split checkable by hand (#177).
    """
    lines = [
        f"!! {cell_id}: REFUSING to compare eval-derived metrics — "
        f"{len(conflicts)} different eval-counter bases in one cell.",
        "   `likelihood_evals` does not count the same thing in these rows, so "
        "`time_per_eval_ms` is not comparable between them:",
    ]
    for basis, names in conflicts.items():
        lines.append(f"     - {eval_basis_label(basis)} ({basis}), {len(names)} row(s):")
        for name in names[:_BANNER_MAX_NAMES]:
            lines.append(f"         {name}")
        if len(names) > _BANNER_MAX_NAMES:
            # Never a silent cap — the count is stated, and comparison.json's
            # `eval_counter_bases` carries the full per-row mapping.
            lines.append(
                f"         ... and {len(names) - _BANNER_MAX_NAMES} more "
                f"(full list in comparison.json -> eval_counter_bases)"
            )
    lines.append(
        "   Wall metrics, max_log_likelihood and log_evidence ARE still comparable "
        "and are reported in full below."
    )
    lines.append(
        "   A v1 MultiStart row recorded stored samples, not evaluations, and its "
        "step count was never written — its per-eval figure cannot be recovered, "
        "only re-run under schema v2."
    )
    return lines


def _render_table(comparison: dict, cell_id: str) -> str:
    conflicts = comparison.get("eval_counter_conflicts") or {}
    bases = comparison.get("eval_counter_bases") or {}
    lines = [f"=== {cell_id} ==="]
    if conflicts:
        lines.extend(_conflict_banner(conflicts, cell_id))
    rows = [("config", "backend", "total", "sampler", "viz", "per_eval", "log_evidence", "basis")]
    for name, cfg in comparison["configs"].items():
        backend = cfg.get("device", {}).get("backend", "?")
        log_evidence = cfg.get("results", {}).get("log_evidence")
        basis = bases.get(name, EVAL_BASIS_UNKNOWN)
        # In a mixed cell the per-eval column is withheld outright. Rendering
        # it "for reference" is exactly how the wrong comparison gets made.
        per_eval = "—" if conflicts else f"{_get_perf(cfg, 'time_per_eval_ms'):.2f}ms"
        rows.append(
            (
                name,
                str(backend),
                _format_seconds(_get_perf(cfg, "total_wall_s")),
                _format_seconds(_get_perf(cfg, "sampler_wall_s")),
                _format_seconds(_get_perf(cfg, "viz_wall_s")),
                per_eval,
                f"{log_evidence:.4f}" if isinstance(log_evidence, (int, float)) else "—",
                basis,
            )
        )
    col_w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        lines.append("  " + "  ".join(s.ljust(w) for s, w in zip(r, col_w)))
    return "\n".join(lines)


def _render_png(comparison: dict, cell_id: str, png_path: Path) -> None:
    """Grouped bar chart of the headline metrics.

    A cell holding one eval basis charts all four metrics as before. A MIXED
    cell charts the wall metrics only and says so in the title: the per-eval
    row is the one that would be read across incommensurable rows, and on the
    shared log x-axis a 257-eval row and a 247,808-eval row sit orders of
    magnitude apart for no physical reason (#177).
    """
    configs = comparison["configs"]
    if not configs:
        return

    conflicts = comparison.get("eval_counter_conflicts") or {}
    metrics = _WALL_METRICS_FOR_BAR_CHART if conflicts else _METRICS_FOR_BAR_CHART

    config_names = list(configs.keys())
    n_cfgs = len(config_names)
    n_metrics = len(metrics)

    # The mixed-cell title runs to two lines; without the extra height
    # tight_layout cannot fit it and matplotlib warns on every cell.
    title_lines = 2 if conflicts else 1
    fig_h = max(3.5, 0.35 * n_metrics + 1.5) + 0.4 * (title_lines - 1)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    cmap = plt.get_cmap("tab10")
    bar_height = 0.8 / n_cfgs

    y_metric = np.arange(n_metrics)
    for j, cname in enumerate(config_names):
        cfg = configs[cname]
        values = [_get_perf(cfg, key) for key, _label in metrics]
        offset = (j - (n_cfgs - 1) / 2) * bar_height
        ax.barh(
            y_metric + offset,
            values,
            height=bar_height,
            label=cname,
            color=cmap(j % cmap.N),
            edgecolor="white",
        )

    ax.set_yticks(y_metric)
    ax.set_yticklabels([label for _key, label in metrics], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Value (log scale; seconds or ms per the row)")
    title = f"{cell_id}  — search profiling comparison"
    if conflicts:
        title += "\nwall only — mixed eval-counter bases, per-eval withheld (#177)"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    if args.cell:
        cells: list[tuple[str, ...]] = []
        for spec in args.cell:
            parts = spec.split("/")
            if len(parts) != 4:
                sys.stderr.write(
                    f"bad --cell argument: {spec!r} (expected sampler/class/model/instrument)\n"
                )
                return 2
            cells.append(tuple(parts))
    else:
        cells = _discover_cells(args.output_root)

    if not cells:
        sys.stderr.write(f"no cells found under {args.output_root}\n")
        return 1

    mixed_cells: list[str] = []
    for cell_tuple in cells:
        cell_id = "/".join(cell_tuple)
        cell_dir = args.output_root.joinpath(*cell_tuple)
        if not cell_dir.exists():
            sys.stderr.write(f"  skipping {cell_id}: dir missing\n")
            continue

        comparison = _aggregate_cell(cell_dir)
        if not comparison["configs"]:
            sys.stderr.write(f"  skipping {cell_id}: no per-config JSONs found\n")
            continue

        conflicts = comparison.get("eval_counter_conflicts") or {}
        if conflicts:
            mixed_cells.append(cell_id)
            sys.stderr.write("\n".join(_conflict_banner(conflicts, cell_id)) + "\n")

        comparison_path = cell_dir / "comparison.json"
        png_path = cell_dir / "comparison.png"
        comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
        _render_png(comparison, cell_id, png_path)

        print(_render_table(comparison, cell_id))
        print(f"  -> {comparison_path}")
        print(f"  -> {png_path}\n")

    if mixed_cells:
        # Non-zero so a sweep or CI step cannot pass over a cell whose
        # eval-derived metrics were withheld. The artifacts are still written
        # — the wall numbers and evidences in them are correct and useful.
        sys.stderr.write(
            f"\naggregate: {len(mixed_cells)} cell(s) hold more than one eval-counter "
            f"basis; per-eval metrics withheld for: {', '.join(mixed_cells)}\n"
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
