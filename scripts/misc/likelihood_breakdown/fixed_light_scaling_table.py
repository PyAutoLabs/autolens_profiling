"""Aggregator for the fixed-lens-light SOURCE-PIXEL SCALING sweep — #257, phase 4.

Folds the per-leg result JSONs written by
``scripts/imaging/likelihood_breakdown/fixed_light.py`` into, per hardware:

* an **ms-vs-N** table per mesh (certified at its certifying budget, certified at
  phase 3's safe budget, unconstrained Cholesky, PDIP, the two log determinants,
  and the whole library likelihood call);
* a **fitted scaling exponent** per row — least squares of ``log(ms)`` on
  ``log(N)``, reported with its R² and the number of points it used;
* a **memory** table — peak device bytes per N, and the largest N that fits;
* the N at which the whole library call crosses **100 ms** and **1 s**, by
  interpolation *between measured points only*;
* a two-panel log-log **figure** (one panel per mesh).

Run from the repo root::

    python scripts/misc/likelihood_breakdown/fixed_light_scaling_table.py
    python scripts/misc/likelihood_breakdown/fixed_light_scaling_table.py --out /tmp/x.md

Why a leg with no JSON is still a row
-------------------------------------

This sweep's second question is memory: **at what N does each hardware stop
fitting?** The answer to that is a leg that died, so a leg that produced no
result JSON is not a gap in the table — it is the measurement. The script reads,
in order:

1. ``results/sweep/fixed_light_scaling_manifest.jsonl`` — one line per leg that
   was launched (``mesh``, ``n``, ``hardware``, and either ``status`` or the
   SLURM ``job``). This is the only record that distinguishes *launched and
   died* from *never launched*.
2. ``results/breakdown/imaging/*.json`` — the rows themselves.

Nothing here is interpolated or extrapolated except the two explicitly-labelled
threshold crossings, which are linear in log-log **between two measured points**
and are reported as "not measured" when the threshold lies outside the measured
range.

Filenames it reads
------------------

``resolve_output_paths`` builds ``<cell>_<config_name>``, and the cell appends
its own ``_n<N>`` suffix (the **built** pixel count — on the rectangular mesh
that is the nearest square to the request, so 500 -> 484), giving::

    fixed_light_<mesh>_n<built>_<config_name>.json

with ``<mesh>`` one of ``rectangular`` / ``delaunay`` and ``<config_name>`` one
of the four in ``HARDWARE`` below. Legs are keyed by the **requested** N so the
four hardware columns line up, and the built count is carried beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()

#: Requested source-pixel counts, in sweep order. The rectangular mesh rounds
#: each to the nearest square; the Delaunay mesh takes it exactly.
SIZES = (500, 1000, 1500, 2500, 4000)

MESHES = ("rectangular", "delaunay")

#: ``label -> (config_name, description)``. The label is the filename stem of
#: the rendered table and figure.
HARDWARE = {
    "hpc_a100_fp64": (
        "hpc_a100_fp64_fixed_light",
        "A100 80GB PCIe (RAL gpu-2), fp64, @vmap 16",
    ),
    "local_rtx2060_fp64": (
        "local_rtx2060_fp64_fixed_light",
        "RTX 2060 Max-Q 6 GB, fp64, single call",
    ),
    "local_rtx2060_mp": (
        "local_rtx2060_mp_fixed_light",
        "RTX 2060 Max-Q 6 GB, mixed precision, single call",
    ),
    "local_cpu_fp64": (
        "local_cpu_fp64_fixed_light",
        "JAX-CPU i9-10885H, NPROC 8 / BLAS 1, fp64, single call",
    ),
}

#: ``s3.rows`` keys, exactly as the cell writes them (seconds).
ROW_UNCONSTRAINED = "Cholesky solve (unconstrained)"
ROW_PDIP = "NNLS PDIP (cell-driven, max_iter 50)"
ROW_BUILD = "curvature_reg_matrix build (dense)"
ROW_LOGDET_CURV = "Log det Cholesky (F+λH reduced)"
ROW_LOGDET_REG = "Log det Cholesky (H reduced)"

#: The rows the ms-vs-N table prints, in order: ``(key, label)``.
TABLE_ROWS = (
    ("certified_certifying_ms", "certified active set (at its certifying budget)"),
    ("certified_safe_ms", "certified active set (phase 3 safe budget)"),
    ("unconstrained_ms", "Cholesky solve (unconstrained)"),
    ("pdip_ms", "NNLS PDIP (reference)"),
    ("build_ms", "F+λH build (dense)"),
    ("logdet_curv_ms", "log det Cholesky (F+λH)"),
    ("logdet_reg_ms", "log det Cholesky (H)"),
    ("library_s3_ms", "library likelihood call (S3)"),
    ("library_s0_ms", "library likelihood call (S0)"),
)

#: The rows the log-log figure draws.
FIGURE_ROWS = (
    ("certified_safe_ms", "certified (safe budget)", "#4C72B0"),
    ("unconstrained_ms", "unconstrained Cholesky", "#55A868"),
    ("pdip_ms", "PDIP", "#C44E52"),
    ("library_s3_ms", "library call (S3)", "#8172B3"),
)

#: Thresholds the crossover section reports, in ms.
CROSSINGS = (100.0, 1000.0)


# ---------------------------------------------------------------------------
# leg identity and file resolution
# ---------------------------------------------------------------------------


def built_pixels(mesh: str, n: int) -> int:
    """The pixel count *mesh* actually builds for a request of *n*.

    The cell's own rule (``fixed_light.py``): the rectangular mesh is
    ``round(sqrt(n))`` squared; the Delaunay family places ``n`` vertices.
    """
    if mesh == "rectangular":
        side = int(round(math.sqrt(n)))
        return side * side
    return int(n)


def json_candidates(mesh: str, n: int, config_name: str) -> list[str]:
    """Result-JSON basenames for one leg, most-preferred first."""
    return [f"fixed_light_{mesh}_n{built_pixels(mesh, n)}_{config_name}"]


def find_json(results_dir: Path, mesh: str, n: int, config_name: str) -> Path | None:
    for basename in json_candidates(mesh, n, config_name):
        candidate = results_dir / f"{basename}.json"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> list[dict]:
    """One dict per launched leg; an absent manifest is an empty list."""
    if not manifest_path.exists():
        return []
    rows = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def manifest_status(manifest: list[dict], mesh: str, n: int, hardware: str) -> tuple[str, str]:
    """``(status, job)`` the manifest records for this leg, else ``("missing", "")``."""
    for row in manifest:
        if (
            row.get("mesh") == mesh
            and int(row.get("n", -1)) == int(n)
            and row.get("hardware") == hardware
        ):
            return str(row.get("status", "pending")), str(row.get("job", "") or "")
    return "missing", ""


# ---------------------------------------------------------------------------
# reading one leg
# ---------------------------------------------------------------------------


def _dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class Leg:
    """One ``(mesh, n, hardware)`` cell of the sweep, measured or not."""

    def __init__(self, mesh: str, n: int, hardware: str):
        self.mesh = mesh
        self.n = n
        self.hardware = hardware
        self.status = "missing"
        self.job = ""
        self.json_path: Path | None = None
        self.payload: dict = {}

    # -- identity --------------------------------------------------------
    @property
    def fits(self) -> bool:
        return self.status == "ok"

    @property
    def built(self) -> int | None:
        value = _dig(self.payload, "configuration.source_pixels")
        return int(value) if isinstance(value, int) else None

    def fits_cell(self) -> str:
        if self.status != "ok":
            return {
                "oom": "OOM",
                "timeout": "timeout",
                "failed": "failed",
                "pending": "pending",
                "missing": "not run",
            }.get(self.status, self.status)
        gb = self.peak_gb()
        return "✓" if gb is None else f"✓ {gb:.2f} GB"

    # -- memory ----------------------------------------------------------
    def peak_gb(self) -> float | None:
        value = _dig(self.payload, "peak_bytes.after_all")
        if isinstance(value, (int, float)) and value > 0:
            return float(value) / 1024**3
        return None

    # -- rows ------------------------------------------------------------
    def _s3_row_ms(self, name: str) -> float | None:
        rows = _dig(self.payload, "s3.rows") or {}
        value = rows.get(name)
        return float(value) * 1e3 if isinstance(value, (int, float)) else None

    @property
    def certifying_budget(self) -> int | None:
        value = _dig(self.payload, "active_set.smallest_certifying_budget")
        return int(value) if isinstance(value, int) else None

    @property
    def safe_budget(self) -> int | None:
        value = _dig(self.payload, "active_set.safe_budget")
        return int(value) if isinstance(value, int) else None

    def budget_entry(self, budget: int | None) -> dict | None:
        if budget is None:
            return None
        for entry in _dig(self.payload, "active_set.budgets") or []:
            if isinstance(entry, dict) and entry.get("pass_budget") == budget:
                return entry
        return None

    def metrics(self) -> dict[str, float | None]:
        """Every value the tables and the figure read, in ms (or None)."""
        if self.status != "ok":
            return {key: None for key, _ in TABLE_ROWS}
        certifying = self.budget_entry(self.certifying_budget)
        safe = self.budget_entry(self.safe_budget)
        return {
            "certified_certifying_ms": (
                float(certifying["ms"]) if certifying and certifying.get("ms") is not None else None
            ),
            "certified_safe_ms": (
                float(safe["ms"]) if safe and safe.get("ms") is not None else None
            ),
            "unconstrained_ms": self._s3_row_ms(ROW_UNCONSTRAINED),
            "pdip_ms": self._s3_row_ms(ROW_PDIP),
            "build_ms": self._s3_row_ms(ROW_BUILD),
            "logdet_curv_ms": self._s3_row_ms(ROW_LOGDET_CURV),
            "logdet_reg_ms": self._s3_row_ms(ROW_LOGDET_REG),
            "library_s3_ms": _as_float(_dig(self.payload, "library_row.s3_ms")),
            "library_s0_ms": _as_float(_dig(self.payload, "library_row.s0_ms")),
        }

    @property
    def pdip_iterations(self) -> int | None:
        value = _dig(self.payload, "s3.nnls.iterations")
        return int(value) if isinstance(value, int) else None


def _as_float(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def collect(
    results_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[tuple[str, int, str], Leg]:
    results_dir = results_dir or (ROOT / "results" / "breakdown" / "imaging")
    manifest_path = manifest_path or (
        ROOT / "results" / "sweep" / "fixed_light_scaling_manifest.jsonl"
    )
    manifest = load_manifest(manifest_path)

    legs: dict[tuple[str, int, str], Leg] = {}
    for hardware, (config_name, _) in HARDWARE.items():
        for mesh in MESHES:
            for n in SIZES:
                leg = Leg(mesh, n, hardware)
                leg.status, leg.job = manifest_status(manifest, mesh, n, hardware)
                path = find_json(results_dir, mesh, n, config_name)
                if path is not None:
                    try:
                        leg.payload = json.loads(path.read_text())
                        leg.json_path = path
                        leg.status = "ok"
                    except (OSError, json.JSONDecodeError):
                        leg.status = "failed"
                elif leg.status == "ok":
                    # The manifest claims it finished but no JSON exists — do not
                    # believe the manifest over the absence of the artifact.
                    leg.status = "failed"
                legs[(mesh, n, hardware)] = leg
    return legs


# ---------------------------------------------------------------------------
# the exponent fit
# ---------------------------------------------------------------------------


def fit_exponent(points: list[tuple[float, float]]) -> dict:
    """Least-squares ``log(ms) = alpha * log(N) + c`` over measured points only.

    Returns ``{"alpha", "intercept", "r_squared", "n_points"}``; ``alpha`` is
    ``None`` when fewer than two distinct N were measured, because a power law
    through one point is not a measurement.
    """
    usable = [(x, y) for x, y in points if x > 0 and y is not None and y > 0]
    if len({x for x, _ in usable}) < 2:
        return {"alpha": None, "intercept": None, "r_squared": None, "n_points": len(usable)}

    xs = [math.log(x) for x, _ in usable]
    ys = [math.log(y) for _, y in usable]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    alpha = sxy / sxx
    intercept = mean_y - alpha * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (alpha * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "alpha": alpha,
        "intercept": intercept,
        "r_squared": r_squared,
        "n_points": n,
    }


def crossing_n(points: list[tuple[float, float]], threshold_ms: float) -> float | None:
    """The N at which a row crosses *threshold_ms*, log-log linear BETWEEN two
    measured points. ``None`` when the threshold is outside the measured range —
    never an extrapolation."""
    usable = sorted((x, y) for x, y in points if x > 0 and y is not None and y > 0)
    if len(usable) < 2:
        return None
    for (x0, y0), (x1, y1) in zip(usable, usable[1:]):
        lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
        if lo <= threshold_ms <= hi:
            if y0 == y1:
                return x0
            t = (math.log(threshold_ms) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))
    return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def render_hardware(legs: dict[tuple[str, int, str], Leg], hardware: str) -> list[str]:
    config_name, description = HARDWARE[hardware]
    out = [f"# Fixed lens light — source-pixel scaling — `{hardware}`", ""]
    out += [f"{description}. Config name `{config_name}`.", ""]
    out += [
        "Every cell is measured. `—` means the leg produced no value; `OOM` /",
        "`timeout` / `failed` / `not run` in the *fits* table say why. Nothing in",
        "this file is extrapolated.",
        "",
    ]

    for mesh in MESHES:
        row_legs = [legs[(mesh, n, hardware)] for n in SIZES]
        built = [leg.built for leg in row_legs]
        header_n = [f"{n}" if b is None or b == n else f"{n} ({b})" for n, b in zip(SIZES, built)]
        out += [f"## {mesh}", ""]
        out += ["Requested N (built N in brackets where they differ).", ""]

        out += ["### Fits, and at what peak", ""]
        out += ["| N | " + " | ".join(header_n) + " |"]
        out += ["|---|" + "---:|" * len(SIZES)]
        out += ["| outcome | " + " | ".join(leg.fits_cell() for leg in row_legs) + " |"]
        out += [
            "| certifying budget | "
            + " | ".join(
                str(leg.certifying_budget) if leg.certifying_budget is not None else "—"
                for leg in row_legs
            )
            + " |"
        ]
        out += [
            "| PDIP iterations | "
            + " | ".join(
                str(leg.pdip_iterations) if leg.pdip_iterations is not None else "—"
                for leg in row_legs
            )
            + " |"
        ]
        out += [""]

        metrics = [leg.metrics() for leg in row_legs]
        out += ["### Per call, ms", ""]
        out += ["| Row | " + " | ".join(header_n) + " | exponent α | R² |"]
        out += ["|---|" + "---:|" * (len(SIZES) + 2)]
        for key, label in TABLE_ROWS:
            values = [m[key] for m in metrics]
            points = [
                (float(leg.built or leg.n), value)
                for leg, value in zip(row_legs, values)
                if value is not None
            ]
            fit = fit_exponent(points)
            alpha = "—" if fit["alpha"] is None else f"{fit['alpha']:.2f}"
            r2 = "—" if fit["r_squared"] is None else f"{fit['r_squared']:.3f}"
            out += [
                f"| {label} | " + " | ".join(_fmt_ms(v) for v in values) + f" | {alpha} | {r2} |"
            ]
        out += [""]

        out += ["### Where the whole call crosses a threshold", ""]
        lib_points = [
            (float(leg.built or leg.n), m["library_s3_ms"])
            for leg, m in zip(row_legs, metrics)
            if m["library_s3_ms"] is not None
        ]
        for threshold in CROSSINGS:
            n_cross = crossing_n(lib_points, threshold)
            label = f"{threshold:.0f} ms" if threshold < 1000 else "1 s"
            if n_cross is None:
                out += [
                    f"- **{label}** — not measured: the threshold lies outside the "
                    f"measured range, and this table never extrapolates."
                ]
            else:
                out += [f"- **{label}** — N ≈ {n_cross:.0f} (library likelihood call, S3)"]
        out += [""]

    return out


def render(legs: dict[tuple[str, int, str], Leg], hardware: str) -> str:
    return "\n".join(render_hardware(legs, hardware)) + "\n"


def save_figure(legs: dict[tuple[str, int, str], Leg], hardware: str, path: Path) -> bool:
    """Two-panel log-log figure, one panel per mesh. ``False`` when nothing was
    measured on this hardware (no empty figure is written)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_point = False
    fig, axes = plt.subplots(1, len(MESHES), figsize=(12, 5), sharey=True)
    for ax, mesh in zip(axes, MESHES):
        for key, label, colour in FIGURE_ROWS:
            xs, ys = [], []
            for n in SIZES:
                leg = legs[(mesh, n, hardware)]
                value = leg.metrics()[key]
                if value is None:
                    continue
                xs.append(float(leg.built or n))
                ys.append(value)
            if len(xs) >= 1:
                any_point = True
                fit = fit_exponent(list(zip(xs, ys)))
                suffix = "" if fit["alpha"] is None else f"  (α={fit['alpha']:.2f})"
                ax.plot(xs, ys, "o-", color=colour, label=f"{label}{suffix}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("source pixels N")
        ax.set_title(mesh)
        ax.grid(True, which="both", alpha=0.25)
        # A panel with no measured leg has no artists, and matplotlib warns
        # rather than drawing an empty legend. Nothing was measured — say so in
        # the panel instead of emitting a warning no reader will see.
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no leg measured", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_ylabel("per call (ms)")
    fig.suptitle(
        f"Fixed lens light — source-pixel scaling — {HARDWARE[hardware][1]}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    if any_point:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return any_point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "breakdown" / "imaging",
        help="where the per-leg result JSONs live",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "sweep" / "fixed_light_scaling_manifest.jsonl",
        help="one line per launched leg (mesh, n, hardware, status/job)",
    )
    parser.add_argument(
        "--hardware",
        action="append",
        choices=sorted(HARDWARE),
        help="render only these (repeatable); default: all four",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "breakdown" / "imaging",
        help="where the rendered .md and .png are written",
    )
    parser.add_argument("--stdout", action="store_true", help="also print each table")
    args = parser.parse_args(argv)

    legs = collect(args.results_dir, args.manifest)
    targets = args.hardware or sorted(HARDWARE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for hardware in targets:
        text = render(legs, hardware)
        md_path = args.out_dir / f"fixed_light_scaling_{hardware}.md"
        md_path.write_text(text)
        png_path = args.out_dir / f"fixed_light_scaling_{hardware}.png"
        drew = save_figure(legs, hardware, png_path)
        measured = sum(
            1 for mesh in MESHES for n in SIZES if legs[(mesh, n, hardware)].status == "ok"
        )
        print(f"{hardware}: {measured}/{len(MESHES) * len(SIZES)} legs measured -> {md_path}")
        if drew:
            print(f"{' ' * len(hardware)}  figure -> {png_path}")
        if args.stdout:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
