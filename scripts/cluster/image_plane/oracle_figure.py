"""
Cluster PointSolver: Oracle Verification Figure (cluster-pointsolver phase 1)
============================================================================

Post-processing for ``robust_settings_sweep.py`` (autolens_profiling#436): draw what
the independent oracle (``_oracle.py``) says the images are, and what each solver
row found, so the oracle's roots can be checked by eye before any verdict is
trusted. A campaign measurement artefact; it stays in autolens_profiling and is
not an Eyes surface.

Usage::

    python scripts/cluster/image_plane/oracle_figure.py <sweep.json> \\
        [--config ROW ...] [--draw N ...] [--output-dir DIR] [--close-radius ARCSEC]

One PNG per (draw, source plane):
``oracle_figure_<dataset>[_pilot]_<config_name>_draw<N>_plane<J>.png`` under
``results/breakdown/cluster/`` by default. The PNGs are inspection aids; only the
ones a ledger cites are copied into ``results/breakdown/cluster/figures/`` and
committed.

What is reconstructed, and from what
------------------------------------

The sweep JSON stores every judged draw's full model vector (``draws.vectors``),
and the oracle block stores, per draw, the source-plane coordinates actually
solved (``oracle.per_draw[].betas``), each source's plane index and every oracle
root (position, residual, signed magnification, parity, accepted, singular
values, stability). The figure rebuilds the dataset ``ClusterCase`` exactly as the
sweep does (``_datasets.load_simple`` / ``_datasets.load_a2744``) and each draw's
tracer as ``CASE.tracer_from_vector(draws.vectors[N])`` -- the same call the sweep
used for the oracle -- so the critical curves and caustics belong to the judged
draw, not to the truth. The betas, roots and verdicts are read from the JSON and
never recomputed. The dataset file digests are compared with the JSON's
``dataset_sha256``; a mismatch is printed (the model structure, not the stored
positions, is what the tracer needs). For A2744 the case is loaded with the
``prior-median-smoke`` reference: only the model structure is used, the vector
always comes from the JSON.

Panels
------

(a) image plane, all sources on this plane: the oracle's accepted roots (circle =
positive parity, square = negative parity, marker area ~ log10|mu|, colour per
source as in ``autolens.plot``'s cluster helpers), rejected roots as grey crosses,
the plane's tangential (solid) and radial (dashed) critical curves
(``aplt.plot_critical_curves``) and the dataset's stored positions
(``aplt.plot_positions_overlay``; solver output, a comparison not truth). Other
planes' roots are faint grey dots for context.

source plane: the plane's caustics (``aplt.plot_caustics``) with each source's
beta as a star.

(b) one panel per sweep row (when rows carry ``per_draw[].sources[].roots``): the
row's solver extent (dashed box), each considered root coloured by verdict
(covered / missed / duplicated), the row's candidates (matched, duplicated,
spurious, ignored = attributed to a below-threshold root) and a count legend.
Rows default to ``control`` plus every row with a non-covered root or spurious
candidate on that draw (at most ``--max-rows``).

(c) zooms around every group of accepted roots on the plane whose
nearest-neighbour separation is below ``--close-radius`` (cusp / fold pairs and
triplets), at a half-width set by the group's span so the separation is visible,
with locally re-contoured critical curves and every shown row's candidates.

NumPy path only: the critical-curve contouring is not ``jax.vmap``-safe.

## Visualization defaults that fail at cluster scale

Every default below was overridden here (feeds the Mind prompt
``cluster_fit_visualizer``):

- ``plot_critical_curves`` / ``plot_caustics`` ``grid``: galaxy-scale scripts
  contour on a ~0.05"/pixel grid over a few arcsec. At cluster scale the grid
  must cover the image spread (tens of arcsec) and the tangential curve, which can
  sit outside the images; this figure contours on a square grid 1.6x the roots'
  field (+15 % margin) at ``--cc-pixels`` (default 160) per side, i.e.
  ~0.2-0.8"/pixel, then frames the panel on roots + curves. Cost is dominated by the Richardson finite-difference
  Hessian (``LensCalc._hessian_via_richardson``: 16 multi-plane ray-traces per
  call, and convergence and shear each compute the Hessian again), ~15-35 s per
  curve type per plane for the 13-component simple model on a laptop, roughly
  independent of grid size above ~150^2 -- so a finer full-field grid is not
  affordable; the zooms re-contour on a small local grid instead.
- ``plot_critical_curves`` / ``plot_caustics`` colour each plane's curves with
  ``WONG_PALETTE[plane_index - 1]`` while the positions helpers colour source
  ``i`` with ``WONG_PALETTE[i]``; the two only agree when source ``i`` sits on
  plane ``i + 1`` (true for ``simple``, not guaranteed for A2744 when sources
  share a plane). There is no colour argument: the curves are recoloured to
  neutral dark grey after drawing so source colour means one thing.
- every helper calls ``ax.legend(loc="upper right", fontsize=9)`` and sets its own
  title, which at cluster scale sits on top of images in the field's corners;
  legends are removed and rebuilt outside the axes.
- ``plot_positions_overlay`` draws fixed 140-pt open circles with no parity or
  magnification encoding, labelled "Source i images" by list index (colour also
  by list index): it is used only for the stored dataset positions, with the list
  padded so source ``k`` keeps colour ``k``, relabelled, and the oracle roots are
  drawn bespoke (parity marker, log|mu| size).
- ``plot_positions_overlay`` ``kpc_scale_bar=50``: ~8" at z=0.5, too short to read
  against a 60-160" field; the bar is chosen as ~15 % of the field width
  (50 / 100 / 200 / 500 kpc).
- ``plot_image_group_zooms``: needs a background image with positive pixels
  (``LogNorm``), draws one position per panel, fixes the width at
  ``zoom_arcsec=6`` and always creates and saves its own figure (no ``ax``). The
  sweep has no image, close roots must share a panel, and a cusp triplet can be
  ~0.1" or ~5" across, so zooms are drawn bespoke at a span-derived half-width.
- the ``plot_positions_overlay`` scale bar and its label are white (they assume a
  dark ``LogNorm`` image underneath); with no image they are invisible, so they
  are recoloured black.
- marker sizes (140-220 pt) are galaxy-scale "a handful of images" sizes; with
  ~5 configs x 3-6 roots per panel they hide neighbours, so roots are 25-165 pt
  by log10|mu| and candidates 20-60 pt.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_HERE = str(Path(__file__).resolve().parent)
for _p in (str(_ROOT / "scripts" / "misc"), str(_ROOT), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _judge import VERDICTS  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DEFAULT_OUTPUT_DIR = _ROOT / "results" / "breakdown" / "cluster"
CLOSE_RADIUS = 3.0  # arcsec: roots closer than this to a neighbour get a zoom panel
MAX_ZOOMS = 8
MAX_ROWS = 6
CC_PIXELS = 160
ZOOM_GRID_N = 81
FIELD_MARGIN = 0.15
GRID_FIELD_FACTOR = 1.6  # contour grid half-width / roots-field half-width
SOURCE_COLOURS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]  # == autolens.cluster.plot.cluster_plots.WONG_PALETTE (checked at render time)
VERDICT_COLOURS = {
    "covered": "#009E73",
    "missed": "#D55E00",
    "duplicated": "#E69F00",
    "spurious": "#CC79A7",
    "ignored": "#8C8C8C",
}
ROW_MARKERS = ["+", "x", "1", "2", "3", "4", "|", "_"]
CURVE_COLOUR = "0.25"


# ---------------------------------------------------------------------------
# Pure helpers (no autolens import; tested in scripts/misc/test)
# ---------------------------------------------------------------------------


def load_sweep(path) -> dict:
    return json.loads(Path(path).read_text())


def output_stem(summary: dict) -> str:
    """``oracle_figure_<dataset>[_pilot]_<config_name>``."""
    pilot = "_pilot" if summary.get("pilot") else ""
    return f"oracle_figure_{summary['dataset']}{pilot}_{summary['config_name']}"


def figure_path(out_dir, summary: dict, draw: int, plane: int) -> Path:
    return Path(out_dir) / f"{output_stem(summary)}_draw{draw}_plane{plane}.png"


def oracle_draw(summary: dict, draw: int) -> dict:
    for o in summary["oracle"]["per_draw"]:
        if o["draw"] == draw:
            return o
    raise KeyError(f"draw {draw} not in oracle.per_draw")


def plane_sources(o: dict) -> dict:
    """{plane index: [source indices]} for one oracle draw, planes ascending."""
    out = {}
    for k, j in enumerate(o["plane_indices"]):
        out.setdefault(int(j), []).append(k)
    return dict(sorted(out.items()))


def row_entry(row: dict, draw: int, source: int):
    for d in row.get("per_draw") or []:
        if d["draw"] == draw:
            return d["sources"][source]
    return None


def rows_with_positions(summary: dict) -> list:
    """Row names whose per-draw source entries carry candidate positions."""
    out = []
    for name, row in summary.get("rows", {}).items():
        ok = any(
            s is not None and "roots" in s for d in row.get("per_draw") or [] for s in d["sources"]
        )
        if ok:
            out.append(name)
    return out


def _has_problem(entry) -> bool:
    if entry is None:
        return False
    return bool(
        entry.get("n_missed")
        or entry.get("n_duplicated")
        or entry.get("n_spurious")
        or not entry.get("multiplicity_match", True)
    )


def select_rows(summary: dict, draw: int, sources, requested=None, max_rows=MAX_ROWS) -> list:
    """Rows to draw: the requested ones, else control + rows with a problem on this draw."""
    available = rows_with_positions(summary)
    if requested:
        unknown = [r for r in requested if r not in summary.get("rows", {})]
        if unknown:
            raise SystemExit(f"--config: unknown rows {unknown}; rows: {list(summary['rows'])}")
        return [r for r in requested if r in available]
    chosen = [r for r in available if r == "control"]
    for name in available:
        if name in chosen:
            continue
        if any(_has_problem(row_entry(summary["rows"][name], draw, k)) for k in sources):
            chosen.append(name)
    return chosen[:max_rows]


def close_groups(points, radius: float) -> list:
    """Index groups (size >= 2) linked by pairwise separation < ``radius`` (single linkage)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(p)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for k in range(i + 1, n):
            if np.hypot(*(p[i] - p[k])) < radius:
                parent[find(i)] = find(k)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def zoom_window(points, tol: float = 0.0) -> tuple:
    """(centre_y, centre_x, half_width): the group fills ~2/3 of the panel."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    centre = 0.5 * (p.min(axis=0) + p.max(axis=0))
    span = float(np.max(p.max(axis=0) - p.min(axis=0))) if len(p) > 1 else 0.0
    half = max(0.75 * span, 10.0 * tol, 0.05)
    return float(centre[0]), float(centre[1]), half


def root_marker_size(magnification) -> float:
    """Marker area in pt^2: 25 at |mu| <= 0.1 up to 165 at |mu| >= 1000."""
    m = abs(float(magnification)) if magnification is not None else 1.0
    level = np.clip(np.log10(max(m, 1e-12)) + 1.0, 0.0, 4.0)
    return float(25.0 + 35.0 * level)


def field_limits(points, margin: float = FIELD_MARGIN) -> tuple:
    """Square (y_min, y_max, x_min, x_max) around ``points`` + margin (at least 5")."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(p):
        return (-10.0, 10.0, -10.0, 10.0)
    centre = 0.5 * (p.min(axis=0) + p.max(axis=0))
    half = max(0.5 * float(np.max(p.max(axis=0) - p.min(axis=0))) * (1 + 2 * margin), 5.0)
    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


def kpc_bar(field_width_arcsec: float, arcsec_per_kpc: float) -> float:
    """A round physical bar (kpc) closest to ~15 % of the field width."""
    target = 0.15 * field_width_arcsec / max(arcsec_per_kpc, 1e-12)
    choices = (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
    return min(choices, key=lambda c: abs(np.log(c / target)))


def nearest_curve_distance(point_yx, curves):
    """Distance from a (y, x) point to the nearest vertex of harvested (x, y, ls) curves."""
    best = None
    for xs, ys, _ls in curves:
        if len(xs):
            d = float(np.min(np.hypot(np.asarray(ys) - point_yx[0], np.asarray(xs) - point_yx[1])))
            best = d if best is None else min(best, d)
    return best


def accepted_roots(o: dict, sources) -> list:
    """[(source index, root dict)] for every accepted oracle root of ``sources``."""
    return [(k, r) for k in sources for r in o["sources"][k]["roots"] if r["accepted"]]


# ---------------------------------------------------------------------------
# Dataset / tracer reconstruction
# ---------------------------------------------------------------------------


def rebuild_case(summary: dict):
    import _datasets

    if summary["dataset"] == "simple":
        case = _datasets.load_simple()
    else:
        case = _datasets.load_a2744("prior-median-smoke")
    want = summary.get("dataset_sha256") or {}
    have = case.provenance.get("files_sha256") or {}
    diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    if diff:
        print(f"WARNING dataset digests differ from the sweep JSON for {diff}")
    if case.paths != summary["reference"]["vector_paths"]:
        raise SystemExit("model parameter paths differ from the sweep JSON: cannot rebuild draws")
    return case


def draw_tracer(case, summary: dict, draw: int):
    vector = np.asarray(summary["draws"]["vectors"][draw], dtype=float)
    tracer = case.tracer_from_vector(vector)
    o = oracle_draw(summary, draw)
    rebuilt = case.plane_indices(tracer)
    if list(rebuilt) != list(o["plane_indices"]):
        raise SystemExit(f"draw {draw}: plane indices {rebuilt} != JSON {o['plane_indices']}")
    return tracer


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _harvest_lines(ax, start: int) -> list:
    """Line data drawn by an aplt helper since ``start``, recoloured neutral."""
    out = []
    for line in ax.lines[start:]:
        line.set_color(CURVE_COLOUR)
        line.set_label("_nolegend_")
        x, y = line.get_data()
        out.append((np.asarray(x), np.asarray(y), line.get_linestyle()))
    return out


def _strip(ax):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def _draw_curves(ax, curves, lw=1.0, alpha=0.8):
    for x, y, ls in curves:
        ax.plot(x, y, color=CURVE_COLOUR, lw=lw, ls=ls, alpha=alpha, zorder=2)


def _draw_roots(ax, roots, colour_of, edge="black", zorder=6, scale=1.0):
    for k, r in roots:
        y, x = r["position"]
        ax.scatter(
            [x],
            [y],
            marker="o" if r["parity"] > 0 else "s",
            s=scale * root_marker_size(r["magnification"]),
            facecolors=colour_of(k),
            edgecolors=edge,
            linewidths=0.8,
            zorder=zorder,
        )


def _row_panel(ax, summary, row_name, draw, sources, o, curves, lims):
    row = summary["rows"][row_name]
    ax.set_title(f"{row_name} [{row.get('status')}]", fontsize=9)
    _draw_curves(ax, curves, lw=0.8, alpha=0.5)
    box = (row.get("static") or {}).get("limits")
    if box:
        ax.add_patch(
            Rectangle(
                (box[2], box[0]),
                box[3] - box[2],
                box[1] - box[0],
                fill=False,
                ls="--",
                lw=0.9,
                ec="0.4",
                zorder=1,
            )
        )
    counts = dict.fromkeys(VERDICTS + ("spurious", "ignored"), 0)
    tol, max_err = None, []
    for k in sources:
        e = row_entry(row, draw, k)
        if e is None or "roots" not in e:
            continue
        tol = e.get("tol", tol)
        for r in e["roots"]:
            counts[r["verdict"]] += 1
            y, x = r["position"]
            c = VERDICT_COLOURS[r["verdict"]]
            ax.scatter(
                [x],
                [y],
                marker="o" if (r.get("parity") or 1) > 0 else "s",
                s=root_marker_size(r["magnification"]),
                facecolors=c if r["verdict"] == "covered" else "none",
                edgecolors=c,
                linewidths=1.6,
                zorder=5,
            )
            if r["verdict"] == "missed":
                ax.scatter([x], [y], marker="x", s=90, c=c, linewidths=2.0, zorder=6)
            max_err += [m["error"] for m in r["matched"]]
            for m in r["matched"]:
                my, mx = m["position"]
                ax.scatter([mx], [my], marker="+", s=40, c="black", linewidths=1.0, zorder=7)
        for p in e.get("duplicated_candidates", []):
            ax.scatter(
                [p[1]],
                [p[0]],
                marker="x",
                s=50,
                c=VERDICT_COLOURS["duplicated"],
                linewidths=1.4,
                zorder=8,
            )
        for s in e.get("spurious", []):
            counts["spurious"] += 1
            ax.scatter(
                [s["position"][1]],
                [s["position"][0]],
                marker="D",
                s=45,
                facecolors="none",
                edgecolors=VERDICT_COLOURS["spurious"],
                linewidths=1.5,
                zorder=8,
            )
        for s in e.get("ignored_candidates", []):
            counts["ignored"] += 1
            ax.scatter(
                [s["position"][1]],
                [s["position"][0]],
                marker=".",
                s=30,
                c=VERDICT_COLOURS["ignored"],
                zorder=8,
            )
        counts["spurious"] = max(counts["spurious"], int(e.get("n_spurious") or 0))
    ax.set_xlim(lims[2], lims[3])
    ax.set_ylim(lims[0], lims[1])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    handles = [
        Line2D(
            [], [], marker="o", ls="", mfc=VERDICT_COLOURS["covered"], mec="none", label="covered"
        ),
        Line2D([], [], marker="x", ls="", color=VERDICT_COLOURS["missed"], mew=2, label="missed"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec=VERDICT_COLOURS["duplicated"],
            label="duplicated",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc="none",
            mec=VERDICT_COLOURS["spurious"],
            label="spurious",
        ),
        Line2D([], [], marker=".", ls="", color=VERDICT_COLOURS["ignored"], label="ignored"),
        Line2D([], [], marker="+", ls="", color="black", label="candidate"),
    ]
    text = (
        f"cov {counts['covered']}  miss {counts['missed']}  dup {counts['duplicated']}\n"
        f"spur {counts['spurious']}  ign {counts['ignored']}"
        + (f'\ntol {tol:.1e}"' if tol is not None else "")
        + (f'  max err {max(max_err):.1e}"' if max_err else "")
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        family="monospace",
        bbox=dict(fc="white", ec="0.7", alpha=0.85),
    )
    return handles, counts


def render(
    summary,
    case,
    draw: int,
    plane: int,
    out_dir,
    rows=None,
    close_radius: float = CLOSE_RADIUS,
    cc_pixels: int = CC_PIXELS,
    max_rows: int = MAX_ROWS,
    tracer=None,
):
    """One PNG for (draw, plane). Returns (path, info dict)."""
    import autolens as al
    import autolens.plot as aplt
    from autolens.cluster.plot import cluster_plots

    if list(cluster_plots.WONG_PALETTE) != SOURCE_COLOURS:
        print("WARNING: autolens WONG_PALETTE changed; source colours may not match the helpers")

    o = oracle_draw(summary, draw)
    sources = plane_sources(o)[plane]
    tracer = tracer if tracer is not None else draw_tracer(case, summary, draw)
    roots = accepted_roots(o, sources)
    rejected = [(k, r) for k in sources for r in o["sources"][k]["roots"] if not r["accepted"]]
    others = [
        r["position"]
        for k in range(len(o["sources"]))
        if k not in sources
        for r in o["sources"][k]["roots"]
        if r["accepted"]
    ]
    stored = [np.atleast_2d(case.observed_positions[k]) for k in sources]
    pts = [r["position"] for _, r in roots] + [p for s in stored for p in s.tolist()]
    lims = field_limits(pts)
    half = 0.5 * (lims[1] - lims[0])
    rows = select_rows(summary, draw, sources, rows, max_rows=max_rows)

    def colour_of(k):
        return SOURCE_COLOURS[k % len(SOURCE_COLOURS)]

    groups = close_groups([r["position"] for _, r in roots], close_radius)[:MAX_ZOOMS]
    n_row_cols = min(3, max(1, len(rows)))
    n_row_rows = int(np.ceil(len(rows) / n_row_cols)) if rows else 0
    n_zoom_cols = min(4, max(1, len(groups)))
    n_zoom_rows = int(np.ceil(len(groups) / n_zoom_cols)) if groups else 0
    heights = [3.0] + [2.6] * n_row_rows + [2.2] * n_zoom_rows
    fig = plt.figure(figsize=(13.5, 3.0 * sum(heights) / 2.6 + 0.8), constrained_layout=True)
    subfigs = np.atleast_1d(fig.subfigures(len(heights), 1, height_ratios=heights))

    # (a) image plane + source plane ------------------------------------------------
    ax_img, ax_src = subfigs[0].subplots(1, 2, width_ratios=[1.6, 1.0])
    grid = al.Grid2D.uniform(
        shape_native=(cc_pixels, cc_pixels),
        pixel_scales=2.0 * GRID_FIELD_FACTOR * half / cc_pixels,
        origin=(0.5 * (lims[0] + lims[1]), 0.5 * (lims[2] + lims[3])),
        over_sample_size=1,
    )
    n0 = len(ax_img.lines)
    aplt.plot_critical_curves(
        tracer, grid=grid, plane_indices=[plane], include_radial=True, ax=ax_img, title=""
    )
    curves = _harvest_lines(ax_img, n0)
    # Display the roots AND the curves (the tangential curve can sit outside the roots).
    lims = field_limits(pts + [(y, x) for xs, ys, _ in curves for x, y in zip(xs, ys)])
    half = 0.5 * (lims[1] - lims[0])
    ax_img.set_xlim(lims[2], lims[3])
    ax_img.set_ylim(lims[0], lims[1])
    _strip(ax_img)
    n_c0, n_l0, n_t0 = len(ax_img.collections), len(ax_img.lines), len(ax_img.texts)
    padded = [
        stored[sources.index(k)] if k in sources else np.zeros((0, 2))
        for k in range(max(sources) + 1)
    ]
    cosmology = getattr(tracer, "cosmology", None)
    z_lens = float(tracer.planes[0].redshift)
    bar = None
    if cosmology is not None:
        bar = kpc_bar(2.0 * half, float(cosmology.arcsec_per_kpc_proper(z_lens)))
    aplt.plot_positions_overlay(
        padded,
        redshift=z_lens if cosmology is not None else None,
        cosmology=cosmology,
        kpc_scale_bar=bar,
        ax=ax_img,
        title="",
    )
    for coll in ax_img.collections[n_c0:]:
        coll.set_label("_nolegend_")
        coll.set_sizes([110])
        coll.set_linewidths([1.0])
        coll.set_linestyle("--")
    for artist in ax_img.lines[n_l0:] + ax_img.texts[n_t0:]:
        artist.set_color("black")  # the helper's scale bar is white (assumes an image)
    _strip(ax_img)
    if others:
        oth = np.asarray(others)
        ax_img.scatter(oth[:, 1], oth[:, 0], s=8, c="0.75", zorder=3)
    _draw_roots(ax_img, roots, colour_of)
    for _k, r in rejected:
        ax_img.scatter([r["position"][1]], [r["position"][0]], marker="x", s=30, c="0.5", zorder=4)
    for gi, g in enumerate(groups):
        cy, cx, hw = zoom_window([roots[i][1]["position"] for i in g])
        ax_img.add_patch(
            Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False, ec="0.2", lw=0.8, zorder=7)
        )
        ax_img.text(cx + hw, cy + hw, f"Z{gi + 1}", fontsize=8, va="bottom", ha="left")
    ax_img.set_xlim(lims[2], lims[3])
    ax_img.set_ylim(lims[0], lims[1])
    ax_img.set_aspect("equal")
    zp = float(tracer.planes[plane].redshift)
    ax_img.set_title(
        f"(a) image plane: oracle roots, plane {plane} (z={zp:.2f}) curves", fontsize=10
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc=colour_of(k),
            mec="black",
            label=f"{case.source_names[k]} (src {k}) roots",
        )
        for k in sources
    ] + [
        Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="parity +"),
        Line2D([], [], marker="s", ls="", mfc="white", mec="black", label="parity -"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", label="size ~ log|mu|"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="none",
            mec="0.3",
            markersize=9,
            label="stored position (--)",
        ),
        Line2D([], [], color=CURVE_COLOUR, label="tangential crit."),
        Line2D([], [], color=CURVE_COLOUR, ls="--", label="radial crit."),
        Line2D([], [], marker="x", ls="", color="0.5", label="rejected root"),
        Line2D([], [], marker=".", ls="", color="0.75", label="other planes' roots"),
    ]
    ax_img.legend(
        handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9
    )

    m0 = len(ax_src.lines)
    aplt.plot_caustics(tracer, grid=grid, plane_indices=[plane], ax=ax_src, title="")
    caustics = _harvest_lines(ax_src, m0)
    _strip(ax_src)
    betas = [o["betas"][k] for k in sources]
    for k, b in zip(sources, betas):
        ax_src.scatter(
            [b[1]], [b[0]], marker="*", s=110, c=colour_of(k), edgecolors="black", zorder=6
        )
    cpts = [(y, x) for x, y, _ in caustics for x, y in zip(x, y)] + [tuple(b) for b in betas]
    slims = field_limits(cpts, margin=0.1)
    ax_src.set_xlim(slims[2], slims[3])
    ax_src.set_ylim(slims[0], slims[1])
    ax_src.set_aspect("equal")
    ax_src.set_title("source plane: caustics (solid tang., dashed radial), beta *", fontsize=9)
    # Inset on the first source's beta: is it inside or outside the nearby caustic fold/cusp?
    d_caustic = [nearest_curve_distance(b, caustics) for b in betas]
    if caustics and d_caustic[0] is not None:
        b0 = betas[0]
        hw_s = max(4.0 * d_caustic[0], 0.1)
        ins = ax_src.inset_axes([0.6, 0.56, 0.38, 0.38])
        for xs, ys, ls in caustics:
            ins.plot(xs, ys, color=CURVE_COLOUR, lw=1.0, ls=ls)
        for k, b in zip(sources, betas):
            ins.scatter([b[1]], [b[0]], marker="*", s=90, c=colour_of(k), edgecolors="black")
        ins.set_xlim(b0[1] - hw_s, b0[1] + hw_s)
        ins.set_ylim(b0[0] - hw_s, b0[0] + hw_s)
        ins.set_aspect("equal")
        ins.tick_params(labelsize=5)
        ins.text(
            0.03,
            0.97,
            f'beta-caustic {d_caustic[0]:.2g}"',
            transform=ins.transAxes,
            va="top",
            fontsize=6,
            bbox=dict(fc="white", ec="none", alpha=0.8),
        )
    ax_src.text(
        0.02,
        0.02,
        "nearest caustic point: "
        + ", ".join(
            f'src {k} {d:.2g}"' if d is not None else f"src {k} -"
            for k, d in zip(sources, d_caustic)
        )
        + f'\n(contour grid {2.0 * GRID_FIELD_FACTOR * half / cc_pixels:.2g}"/px image plane)',
        transform=ax_src.transAxes,
        fontsize=6,
    )
    ax_src.tick_params(labelsize=7)
    ax_img.tick_params(labelsize=7)

    # (b) rows -----------------------------------------------------------------------
    info = {"rows": {}, "groups": [], "sources": sources, "n_roots": len(roots)}
    if rows:
        axes = np.atleast_1d(subfigs[1 : 1 + n_row_rows]).tolist()
        row_axes = []
        for sf in axes:
            row_axes += list(np.atleast_1d(sf.subplots(1, n_row_cols)))
        for ax, name in zip(row_axes, rows):
            row_handles, counts = _row_panel(ax, summary, name, draw, sources, o, curves, lims)
            info["rows"][name] = counts
        for ax in row_axes[len(rows) :]:
            ax.set_axis_off()
        row_axes[0].figure.legend(
            handles=row_handles, loc="outside right upper", fontsize=7, framealpha=0.9
        )
        row_axes[0].figure.suptitle(
            "(b) solver rows: verdict per considered oracle root; dashed box = row extent",
            fontsize=9,
        )
    elif not rows_with_positions(summary):
        ax_img.text(
            0.01,
            0.01,
            "rows carry no candidate positions (older JSON): panel (b) skipped",
            transform=ax_img.transAxes,
            fontsize=7,
        )

    # (c) zooms ----------------------------------------------------------------------
    if groups:
        zaxes = []
        for sf in subfigs[1 + n_row_rows :]:
            zaxes += list(np.atleast_1d(sf.subplots(1, n_zoom_cols)))
        tols = [
            (row_entry(summary["rows"][n], draw, k) or {}).get("tol") or 0.0
            for n in rows
            for k in sources
        ]
        for gi, (ax, g) in enumerate(zip(zaxes, groups)):
            gpos = [roots[i][1]["position"] for i in g]
            cy, cx, hw = zoom_window(gpos, max(tols, default=0.0))
            local = al.Grid2D.uniform(
                shape_native=(ZOOM_GRID_N, ZOOM_GRID_N),
                pixel_scales=2.2 * hw / ZOOM_GRID_N,
                origin=(cy, cx),
                over_sample_size=1,
            )
            z0 = len(ax.lines)
            try:
                aplt.plot_critical_curves(
                    tracer, grid=local, plane_indices=[plane], include_radial=True, ax=ax, title=""
                )
            except Exception as exc:  # noqa: BLE001 - a local contour can legitimately fail
                print(f"zoom Z{gi + 1}: local critical curve failed ({exc}); using full-field")
                _draw_curves(ax, curves)
            _harvest_lines(ax, z0)
            _strip(ax)
            _draw_roots(ax, [roots[i] for i in g], colour_of, scale=1.4)
            for ri, name in enumerate(rows):
                for k in sources:
                    e = row_entry(summary["rows"][name], draw, k)
                    if e is None or "roots" not in e:
                        continue
                    pts_v = [
                        (m["position"], "covered" if r["verdict"] == "covered" else r["verdict"])
                        for r in e["roots"]
                        for m in r["matched"]
                    ]
                    pts_v += [(s["position"], "spurious") for s in e.get("spurious", [])]
                    pts_v += [(s["position"], "ignored") for s in e.get("ignored_candidates", [])]
                    for p, v in pts_v:
                        ax.scatter(
                            [p[1]],
                            [p[0]],
                            marker=ROW_MARKERS[ri % len(ROW_MARKERS)],
                            s=60,
                            c=VERDICT_COLOURS[v],
                            linewidths=1.3,
                            zorder=8,
                        )
            ax.set_xlim(cx - hw, cx + hw)
            ax.set_ylim(cy - hw, cy + hw)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            seps = [
                np.hypot(*(np.asarray(gpos[a]) - np.asarray(gpos[b])))
                for a in range(len(gpos))
                for b in range(a + 1, len(gpos))
            ]
            ax.set_title(
                f'Z{gi + 1}: {len(g)} roots, min sep {min(seps):.3g}", half-width {hw:.3g}"',
                fontsize=8,
            )
            info["groups"].append({"n": len(g), "min_sep": float(min(seps)), "half_width": hw})
        for ax in zaxes[len(groups) :]:
            ax.set_axis_off()
        if rows:
            zaxes[0].legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        marker=ROW_MARKERS[i % len(ROW_MARKERS)],
                        ls="",
                        color="black",
                        label=n,
                    )
                    for i, n in enumerate(rows)
                ],
                fontsize=6,
                loc="lower left",
                framealpha=0.8,
                title="row (colour = verdict)",
                title_fontsize=6,
            )
        zaxes[0].figure.suptitle(
            f'(c) zooms: roots with a neighbour < {close_radius:g}"', fontsize=9
        )

    stab = [o["sources"][k]["stability"].get("stable") for k in sources]
    sets = summary["draws"].get("set_per_draw") or []
    fig.suptitle(
        f"Oracle check: {summary['dataset']} {summary['config_name']}"
        f"{' PILOT' if summary.get('pilot') else ''} | draw {draw}"
        f" ({sets[draw] if draw < len(sets) else '?'}) | plane {plane} z={zp:.2f} | "
        f"sources {[case.source_names[k] for k in sources]} | oracle stable {stab} | "
        f"{'quotable' if summary.get('quotable') else 'NOT quotable'}",
        fontsize=10,
    )
    path = figure_path(out_dir, summary, draw, plane)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sweep_json")
    parser.add_argument("--config", nargs="*", default=None, help="sweep rows to overlay")
    parser.add_argument("--draw", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--close-radius", type=float, default=CLOSE_RADIUS)
    parser.add_argument("--cc-pixels", type=int, default=CC_PIXELS)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    summary = load_sweep(args.sweep_json)
    case = rebuild_case(summary)
    draws = args.draw if args.draw else [o["draw"] for o in summary["oracle"]["per_draw"]]
    if not rows_with_positions(summary):
        print("NOTE: no row carries candidate positions (pre-position JSON): oracle panels only")
    written = []
    for draw in draws:
        o = oracle_draw(summary, draw)
        tracer = draw_tracer(case, summary, draw)
        for plane in plane_sources(o):
            path, info = render(
                summary,
                case,
                draw,
                plane,
                args.output_dir,
                rows=args.config,
                close_radius=args.close_radius,
                cc_pixels=args.cc_pixels,
                max_rows=args.max_rows,
                tracer=tracer,
            )
            written.append(path)
            print(f"draw {draw} plane {plane}: {path} rows {info['rows']} zooms {info['groups']}")
    return written


if __name__ == "__main__":
    main()
