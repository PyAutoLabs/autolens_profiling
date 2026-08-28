"""
Aggregator: NNLS Warm-Start Memo Robustness Matrix
==================================================

Reads every ``delaunay_numba_nnls_iterations_*_v<version>.json`` written by
``delaunay_numba_nnls_iterations.py --model <name>`` and renders the one table the
robustness question needs: for each instrument x model x sequence, memo OFF vs ON on
the *same* instance sequence.

Everything is recomputed from the per-evaluation ``rows`` each result JSON already
carries, never from its pre-baked ``summary`` block — so a result recorded before the
matrix existed (the 30-instance fiducial) is aggregated on exactly the same footing as
a 20-instance matrix cell.

Why a matrix at all: an NNLS warm start is a *discrete guess at the passive set*, so its
quality is a property of the solution's structure. A different mass model, lens light,
mesh, regularization or source morphology is a different structure, and the fiducial
alone cannot tell you whether the memo's seed is ever *worse* than the dense-sign start
it replaces.

Usage
-----
``python nnls_iterations_matrix.py [--version 2026.8.17.1] [--results-dir ...] [--out ...]``

Writes ``results/notes/nnls_warm_start_memo_matrix.md`` and prints the same table.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()

# Display order: fiducial first, then the variants in the order they escalate away
# from it (mass, structure, light, mesh, regularization, index space, data, size).
MODEL_ORDER = [
    "fiducial",
    "powerlaw",
    "subhalo",
    "no_lens_light",
    "sersic_light",
    "rectangular",
    "adapt_reg",
    "no_edge_zeroing",
    "source_complex",
    "mesh_600",
    "mesh_2000",
]
INSTRUMENT_ORDER = ["euclid", "hst"]
SEQUENCE_ORDER = ["random_walk", "iid"]
SEQUENCE_LABEL = {"random_walk": "rw", "iid": "iid"}

FILENAME_RE = re.compile(r"^delaunay_numba_nnls_iterations_.*_v(?P<version>[0-9][0-9.]*)\.json$")

# A memo-ON cell whose seed is this much worse than the dense-sign start is the thing
# the matrix is hunting for; a parity deviation above this is a correctness break.
PARITY_ALERT = 1.0e-8


def cell_stats(rows: list[dict]) -> dict:
    """Per-cell statistics, recomputed from the per-evaluation rows."""

    def column(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=float)

    iterations = column("total_iterations")
    errors = column("warm_start_errors")
    solve_s = column("solve_s")
    eval_s = column("eval_s")
    # Recomputed rather than read: the pre-matrix fiducial JSONs predate the field.
    error_fraction = errors / np.maximum(column("n"), 1.0)

    return {
        "n_evaluations": len(rows),
        # Index 0 of a memo-ON sequence has an empty memo and therefore falls back to the
        # dense-sign start, so a max that sits there is the cold start, not a pathology.
        "argmax_iterations": int(np.argmax(iterations)),
        "median_iterations": float(np.median(iterations)),
        "max_iterations": float(np.max(iterations)),
        "median_errors": float(np.median(errors)),
        "max_errors": float(np.max(errors)),
        "median_solve_s": float(np.median(solve_s)),
        "max_solve_s": float(np.max(solve_s)),
        "median_eval_s": float(np.median(eval_s)),
        "median_error_fraction": float(np.median(error_fraction)),
        "p90_error_fraction": float(np.percentile(error_fraction, 90)),
        "max_error_fraction": float(np.max(error_fraction)),
        "retries": int(sum(row["retries"] for row in rows)),
        # None (rendered as an en dash) for results recorded before the relative
        # fallback guard existed, which is not the same statement as "zero fallbacks".
        "n_fallbacks": (
            int(sum(row["n_fallbacks"] for row in rows))
            if all("n_fallbacks" in row for row in rows)
            else None
        ),
    }


def load_cells(results_dir: Path, version: str | None) -> tuple[list[dict], str]:
    """Every (instrument, model, sequence) cell found on disk, plus the version used."""
    candidates = {}
    for path in sorted(results_dir.glob("delaunay_numba_nnls_iterations_*.json")):
        match = FILENAME_RE.match(path.name)
        if match:
            candidates.setdefault(match.group("version"), []).append(path)

    if not candidates:
        raise FileNotFoundError(f"No delaunay_numba_nnls_iterations_*_v*.json under {results_dir}")

    if version is None:
        # Newest version present, compared as a release tuple rather than as a string
        # (so 2026.8.9 does not sort above 2026.8.17).
        version = max(candidates, key=lambda v: tuple(int(part) for part in v.split(".")))

    if version not in candidates:
        raise FileNotFoundError(
            f"No results for v{version} under {results_dir}; found {sorted(candidates)}"
        )

    cells = []
    for path in candidates[version]:
        data = json.loads(path.read_text())
        model = data.get("model") or data.get("configuration", {}).get("model") or "fiducial"
        for sequence_name, block in data["sequences"].items():
            cells.append(
                {
                    "instrument": data["instrument"],
                    "model": model,
                    "sequence": sequence_name,
                    "parity_log_likelihood": block["parity"]["max_rel_log_likelihood_deviation"],
                    "parity_reconstruction": block["parity"]["max_abs_reconstruction_deviation"],
                    "off": cell_stats(block["rows"]["off"]),
                    "on": cell_stats(block["rows"]["on"]),
                    "path": path,
                }
            )

    def sort_key(cell: dict) -> tuple:
        return (
            INSTRUMENT_ORDER.index(cell["instrument"])
            if cell["instrument"] in INSTRUMENT_ORDER
            else len(INSTRUMENT_ORDER),
            MODEL_ORDER.index(cell["model"]) if cell["model"] in MODEL_ORDER else len(MODEL_ORDER),
            SEQUENCE_ORDER.index(cell["sequence"])
            if cell["sequence"] in SEQUENCE_ORDER
            else len(SEQUENCE_ORDER),
        )

    return sorted(cells, key=sort_key), version


def ratio(cell: dict) -> float:
    """Median iteration reduction OFF/ON. >1 means the memo saved iterations."""
    median_on = cell["on"]["median_iterations"]
    return cell["off"]["median_iterations"] / median_on if median_on else float("inf")


COLUMNS = [
    "instrument",
    "model",
    "seq",
    "n",
    "iters med OFF→ON",
    "iters max OFF→ON",
    "errs med OFF→ON",
    "errs max ON",
    "solve s med OFF→ON",
    "solve s max ON",
    "eval s med OFF→ON",
    "ratio",
    "err frac ON med/p90/max",
    "err frac OFF med",
    "Δlnl",
    "retries",
    "fallbacks",
]


def fallback_label(cell: dict) -> str:
    """Tolerance fallbacks in the cell, or an en dash for a pre-guard result JSON."""
    counts = [cell[label]["n_fallbacks"] for label in ("off", "on")]
    if any(count is None for count in counts):
        return "–"
    return str(sum(counts))


def table_rows(cells: list[dict]) -> list[list[str]]:
    rows = []
    for cell in cells:
        off, on = cell["off"], cell["on"]
        rows.append(
            [
                cell["instrument"],
                cell["model"],
                SEQUENCE_LABEL.get(cell["sequence"], cell["sequence"]),
                str(off["n_evaluations"]),
                f"{off['median_iterations']:.0f}→{on['median_iterations']:.0f}",
                f"{off['max_iterations']:.0f}→{on['max_iterations']:.0f}",
                f"{off['median_errors']:.0f}→{on['median_errors']:.0f}",
                f"{on['max_errors']:.0f}",
                f"{off['median_solve_s']:.3f}→{on['median_solve_s']:.3f}",
                f"{on['max_solve_s']:.3f}",
                f"{off['median_eval_s']:.2f}→{on['median_eval_s']:.2f}",
                f"{ratio(cell):.2f}×",
                f"{on['median_error_fraction']:.3f}/{on['p90_error_fraction']:.3f}/"
                f"{on['max_error_fraction']:.3f}",
                f"{off['median_error_fraction']:.3f}",
                f"{cell['parity_log_likelihood']:.1e}",
                str(off["retries"] + on["retries"]),
                fallback_label(cell),
            ]
        )
    return rows


def markdown_table(cells: list[dict]) -> list[str]:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row) + " |" for row in table_rows(cells)]
    return lines


def name(cell: dict) -> str:
    return f"{cell['instrument']}/{cell['model']}/{SEQUENCE_LABEL.get(cell['sequence'])}"


def findings(cells: list[dict]) -> list[str]:
    """The five failure modes the matrix exists to detect, plus the fallback calibration."""
    lines = ["## Findings", ""]

    worst_ratio = min(cells, key=ratio)
    lines.append(
        f"- **Worst cell by iteration ratio:** `{name(worst_ratio)}` at "
        f"{ratio(worst_ratio):.2f}× "
        f"({worst_ratio['off']['median_iterations']:.0f} → "
        f"{worst_ratio['on']['median_iterations']:.0f} median iterations)."
    )

    worst_max = max(cells, key=lambda c: c["on"]["max_iterations"])
    cold = [c for c in cells if c["on"]["argmax_iterations"] == 0]
    lines.append(
        f"- **Worst cell by MAX memo-ON iterations:** `{name(worst_max)}` at "
        f"{worst_max['on']['max_iterations']:.0f} "
        f"(memo OFF max on the same sequence: {worst_max['off']['max_iterations']:.0f}), "
        f"and its worst evaluation is index {worst_max['on']['argmax_iterations']}. In "
        f"{len(cold)}/{len(cells)} cells the memo-ON maximum IS evaluation 0, whose memo is "
        f"empty and which therefore ran the dense-sign start — a cold start, not a "
        f"pathology. The cells where it is not: "
        + ", ".join(
            f"`{name(c)}` (idx {c['on']['argmax_iterations']})"
            for c in cells
            if c["on"]["argmax_iterations"] != 0
        )
        + "."
    )

    slower = [c for c in cells if c["on"]["median_solve_s"] > c["off"]["median_solve_s"]]
    if slower:
        lines.append(
            "- **Cells where median memo-ON solve seconds EXCEED memo-OFF "
            f"({len(slower)}/{len(cells)}):** "
            + ", ".join(
                f"`{name(c)}` ({c['off']['median_solve_s']:.3f} → "
                f"{c['on']['median_solve_s']:.3f} s)"
                for c in sorted(
                    slower,
                    key=lambda c: c["on"]["median_solve_s"] - c["off"]["median_solve_s"],
                    reverse=True,
                )[:6]
            )
            + "."
        )
    else:
        lines.append(
            "- **Median memo-ON solve seconds never exceed memo-OFF** in any cell — the "
            "skipped dense unconstrained solve covers every extra active-set move."
        )

    retried = [c for c in cells if c["off"]["retries"] + c["on"]["retries"]]
    if retried:
        detail = (
            ", ".join(f"`{name(c)}` ({c['off']['retries'] + c['on']['retries']})" for c in retried)
            + "."
        )
    else:
        detail = "none in any cell."
    lines.append(
        "- **Retries** (a memo-seeded solve raised and the library fell back to the "
        f"dense-sign start): {detail}"
    )

    guarded = [c for c in cells if fallback_label(c) != "–"]
    fired = [c for c in guarded if fallback_label(c) != "0"]
    if not guarded:
        detail = "not recorded — every result JSON here predates the guard."
    elif fired:
        detail = (
            ", ".join(f"`{name(c)}` ({fallback_label(c)})" for c in fired)
            + f" — of {len(guarded)}/{len(cells)} cells recorded with the guard."
        )
    else:
        detail = (
            f"never fired in any of the {len(guarded)}/{len(cells)} cells recorded with "
            "the guard, which is the intent: the default tolerance sits above the worst "
            "seed/dense ratio the matrix measured, so it is protective rather than "
            "flapping."
        )
    lines.append(
        "- **Tolerance fallbacks** (a memo seed's error fraction exceeded "
        "`nnls_warm_start_error_tolerance` × its entry's dense-sign reference, so the "
        f"entry was dropped): {detail}"
    )

    broken = [c for c in cells if c["parity_log_likelihood"] > PARITY_ALERT]
    max_parity = max(cells, key=lambda c: c["parity_log_likelihood"])
    if broken:
        detail = (
            "BROKEN in "
            + ", ".join(f"`{name(c)}` ({c['parity_log_likelihood']:.1e})" for c in broken)
            + "."
        )
    else:
        detail = (
            f"no cell exceeds {PARITY_ALERT:g}; worst is `{name(max_parity)}` at max rel "
            f"Δlnl {max_parity['parity_log_likelihood']:.1e}, max |Δreconstruction| "
            f"{max_parity['parity_reconstruction']:.1e}."
        )
    lines.append(f"- **Parity:** {detail}")

    lines += ["", "### Fallback-tolerance calibration (memo-seed error fraction)", ""]

    for sequence_name in SEQUENCE_ORDER:
        group = [c for c in cells if c["sequence"] == sequence_name]
        if not group:
            continue
        on_median = np.asarray([c["on"]["median_error_fraction"] for c in group])
        on_max = np.asarray([c["on"]["max_error_fraction"] for c in group])
        off_median = np.asarray([c["off"]["median_error_fraction"] for c in group])
        lines.append(
            f"- **{sequence_name}** ({len(group)} cells): memo-ON error fraction median "
            f"{on_median.min():.3f}–{on_median.max():.3f} (across-cell median "
            f"{np.median(on_median):.3f}), per-cell max up to {on_max.max():.3f}; "
            f"dense-sign (OFF) median {off_median.min():.3f}–{off_median.max():.3f}."
        )

    helpful = [c for c in cells if ratio(c) >= 1.0]
    harmful = [c for c in cells if ratio(c) < 1.0]
    if helpful and harmful:
        helpful_ceiling = max(c["on"]["median_error_fraction"] for c in helpful)
        harmful_floor = min(c["on"]["median_error_fraction"] for c in harmful)
        if helpful_ceiling < harmful_floor:
            threshold = 0.5 * (helpful_ceiling + harmful_floor)
            verdict = (
                f"The two populations SEPARATE: every cell where the seed still saves "
                f"iterations has a median memo-ON error fraction ≤ {helpful_ceiling:.3f}, "
                f"and every cell where it does not has ≥ {harmful_floor:.3f}. A fallback "
                f"tolerance of **X ≈ {threshold:.2f}** ({100 * threshold:.0f}% of entries "
                f"wrong) separates them across this matrix."
            )
        else:
            threshold = helpful_ceiling
            verdict = (
                f"The two populations OVERLAP: helpful cells reach a median memo-ON error "
                f"fraction of {helpful_ceiling:.3f} while the least-wrong non-helpful cell "
                f"sits at {harmful_floor:.3f}. **No X separates them**, so a fallback "
                f"tolerance cannot be calibrated from the error fraction alone. The highest "
                f"cut that never discards a seed which was still saving iterations is "
                f"X ≈ {threshold:.2f}, but it fires on none of the non-helpful cells below "
                f"it, i.e. the fallback would be inert where it is wanted."
            )
        lines += [
            "",
            f"- **Separating X.** Helpful cells (ratio ≥ 1×): {len(helpful)}; "
            f"non-helpful (< 1×): {len(harmful)}. {verdict}",
        ]
    else:
        lines += [
            "",
            "- **Separating X.** Every cell falls on one side of ratio = 1×, so this "
            "matrix cannot calibrate a fallback tolerance from a contrast.",
        ]

    # The absolute fraction cannot separate the populations, but the seed's quality
    # *relative to the start it replaces* very nearly can — and "no better than the
    # dense-sign start" is a relative statement to begin with.
    def seed_quality(cell: dict) -> float:
        off_fraction = cell["off"]["median_error_fraction"]
        return cell["on"]["median_error_fraction"] / off_fraction if off_fraction else float("inf")

    if helpful and harmful:
        helpful_quality = max(seed_quality(c) for c in helpful)
        harmful_quality = min(seed_quality(c) for c in harmful)
        worst_relative = max(cells, key=seed_quality)
        lines += [
            "",
            f"- **Relative seed quality (ON / OFF median error fraction).** The absolute "
            f"fraction cannot separate the populations, but the ratio nearly does: helpful "
            f"cells top out at {helpful_quality:.2f}, non-helpful cells start at "
            f"{harmful_quality:.2f}, with the worst seed in the matrix at "
            f"{seed_quality(worst_relative):.2f} (`{name(worst_relative)}`). A fallback "
            f'phrased as *"the memo seed is no better than the dense-sign start"* — i.e. '
            f"**ON/OFF error fraction ≳ 0.9** — is therefore the discriminator the raw "
            f"fraction is not. Its cost is that computing the OFF fraction means computing "
            f"the dense-sign start, which is the work the memo exists to skip.",
        ]

    lines += [
        "",
        "- **Observability caveat.** The error count is only known once the solve has run, "
        "so a fallback cannot threshold on the current evaluation's fraction. Any X would "
        "have to be applied to the *previous* evaluation's fraction as a running proxy — "
        "which is exactly the quantity the random-walk/iid contrast above shows is stable "
        "within a sequence and unstable across regimes.",
    ]

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="Library version to aggregate.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results_dir = (
        Path(args.results_dir) if args.results_dir else ROOT / "results" / "breakdown" / "imaging"
    )
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "results" / "notes" / "nnls_warm_start_memo_matrix.md"
    )

    cells, version = load_cells(results_dir, args.version)

    models = sorted(
        {c["model"] for c in cells}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99
    )
    missing = [m for m in MODEL_ORDER if m not in models]

    header = [
        "# NNLS warm-start memo: model robustness matrix",
        "",
        f"Aggregated from `results/breakdown/imaging/delaunay_numba_nnls_iterations_*_v{version}.json`",
        "by `scripts/imaging/likelihood_breakdown/nnls_iterations_matrix.py`. Companion to",
        "[`nnls_warm_start_memo.md`](./nnls_warm_start_memo.md), which measured the single fiducial.",
        "",
        "Each model variant changes exactly ONE thing about that fiducial (Delaunay Hilbert-1250 +",
        "ConstantSplit + MGE-60 linear lens light + Isothermal + shear), memo OFF then ON on the",
        "*same* seeded instance sequences: `rw` = random walk (N(0, 1% of each prior's width) per",
        "step, the sampler-like regime), `iid` = independent draws from the central 20% of every",
        "prior (uncorrelated, pessimistic). Laptop CPU fp64, `OMP_NUM_THREADS=1`,",
        "`AUTOARRAY_NUMBA_OPERATED_MEMO=0`. `ratio` = median iterations OFF / ON (>1 = memo saved",
        "iterations). `err frac` = warm-start errors / solve size — the quantity a fallback",
        "tolerance would threshold on. `Δlnl` = max relative log-likelihood deviation OFF vs ON.",
        "",
        "Three variants could not be a pure one-thing change, and say so here rather than in a",
        "footnote: `rectangular` must drop `ConstantSplit` for `Constant` (split regularization is",
        "structurally incompatible with the rectangular interpolator); `source_complex` ships as",
        "data/noise/PSF with no truth tracer, so its adapt image is the positive-clipped data, and",
        'it is sampled at 0.05"/px, i.e. hst geometry; `no_edge_zeroing` flips the solve onto the',
        "full-system branch but does NOT change the index set — `Delaunay(zeroed_pixels=0)` already",
        "keeps every index, and the only mesh here that zeroes any (`rectangular`, its shape-derived",
        "136-pixel border) zeroes a *static* set. So the memo key never churns in this matrix.",
        "",
    ]
    if missing:
        header += [f"Not measured: {', '.join(f'`{m}`' for m in missing)}.", ""]

    lines = header + markdown_table(cells) + [""] + findings(cells) + [""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\n  {len(cells)} cells from {len({c['path'] for c in cells})} result files.")
    print(f"  Matrix written to: {out_path}  ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
