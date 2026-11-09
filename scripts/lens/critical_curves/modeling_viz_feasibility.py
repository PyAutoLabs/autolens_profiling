"""
Critical Curves: Zero-Contour Feasibility for Modeling-Run Visualization
=======================================================================

Research for autolens_profiling#521 (critical-curves campaign). Asks whether the
JAX ``zero_contour`` engine could replace ``marching_squares`` when a modeling run
draws critical curves and caustics on its fit figures. Library behaviour is
observed, never patched. No default changes here.

Each lens model is a posterior draw from a completed phase-4 autolens_inference
leg (autolens_inference#59): the maximum-likelihood sample plus weighted draws.
Figures during a run show fits away from the truth, so the truth alone is not
enough.

``--run`` drives one subprocess per leg (bounded by ``LEG_TIMEOUT``). Each worker
writes its JSON after every draw, so a timeout keeps the partial evidence.
``--summarize`` tabulates a results directory without recomputing anything.

__Pre-registered rule (written before the first galaxy run, 2026-11-06)__

- Reference: marching squares at 0.01 arcsec over a +/-4 arcsec field (800^2,
  under the 1000-pixel evaluation-grid cap), five times finer than production.
  Production curves use the fit's own unmasked grid, which ``evaluation_grid``
  resamples to 0.05 arcsec.
- An engine passes a draw when every reference tangential component is matched
  one-to-one within 0.1 arcsec (two 0.05 arcsec pixels, as in phase 3a). For
  zero-contour, every path must also be closed: its largest segment, including
  the closing chord, is at most 3 x the 0.05 arcsec step.
- Radial: graded only where the reference radial component's area is above
  pi * 0.1^2 arcsec^2. Smaller ones are below the plotting resolution and are
  reported but not graded. Production marching squares itself skips radial
  curves whose area is no more than the grid pixel scale.
- Caustics: graded under the same rule against the reference caustics.
- Errors, timeouts and empty results never pass.
- A reference curve within two reference pixels of the field edge marks the
  draw ``reference_clipped``. That draw is excluded and reported.

__Cost__

Each draw times the exact production helper
(``autolens.lens.plot.critical_curves._compute_critical_curve_lines``) with the
method forced through the visualize config. Every production call builds a
fresh ``LensCalc``, so the cost is in-process call by call: draw 0 is cold, and
later draws show whether compilation recurs across visualizations.

__Fallback detector__

A coarse marching-squares tangential count (0.2 arcsec) and the zero-contour
closure check are recorded per draw. ``--summarize`` scores them as a detector
for zero-contour failures.

__Env__
ENV: jax full_datasets
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(SCRIPT.parent))

import dispatch  # noqa: E402  (phase-3a geometry and provenance helpers)

RESULTS = ROOT / "results/lens/critical_curves/viz_feasibility"
RAL_ROOT = "/mnt/ral/jnightin/inference/autolens_inference/"
LEG_TIMEOUT = 3600

# Production field per (dataset_class, instrument): the unmasked plotting grid
# the fit figures pass (``fit.mask.derive_grid.all_false`` for imaging,
# ``real_space_mask.derive_grid.all_false`` for interferometer).
PRODUCTION = {
    ("imaging", "hst"): {"shape": (140, 140), "pixel_scale": 0.05},
    ("interferometer", "sma"): {"shape": (256, 256), "pixel_scale": 0.1},
}
SCALE = {"imaging": "galaxy", "interferometer": "galaxy"}

REFERENCE = {"half_width": 4.0, "pixel_scale": 0.01}
COARSE = 0.2
TOLERANCE = 0.1
ZC_STEP = 0.05
CLOSED_MAX_SEGMENT = 3 * ZC_STEP
RADIAL_GRADE_AREA = 3.141592653589793 * TOLERANCE**2


def load_draws(leg_dir, n_draws, seed):
    """Max-likelihood sample plus ``n_draws`` posterior-weighted draws."""
    import autofit as af
    import numpy as np

    model = af.Model.from_json(str(leg_dir / "files/model.json"))
    with open(leg_dir / "files/samples.csv") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    paths = [".".join(p) for p in model.paths]
    idx = [head.index(p) for p in paths]
    weight = data[:, head.index("weight")]
    log_l = data[:, head.index("log_likelihood")]
    rng = np.random.default_rng(seed)
    picks = [int(np.argmax(log_l))]
    picks += rng.choice(len(weight), size=n_draws, p=weight / weight.sum()).tolist()
    labels = ["max_likelihood"] + [f"draw_{i}" for i in range(n_draws)]
    return model, [(lab, int(i), data[i, idx].tolist()) for lab, i in zip(labels, picks)]


def tracer_from(model, vector):
    """Mirror ``AnalysisLens.tracer_via_instance_from`` (no subhalo/cosmology)."""
    import autolens as al
    from autolens.lens import tracer_util

    inst = model.instance_from_vector(vector, ignore_assertions=True)
    galaxies = list(inst.galaxies)
    for name in ("extra_galaxies", "scaling_galaxies"):
        if getattr(inst, name, None) is not None:
            galaxies += list(getattr(inst, name))
    fields = tracer_util.fields_list_from(getattr(inst, "fields", None))
    return al.Tracer(galaxies=galaxies, fields=fields)


def production_lines(tracer, grid, method):
    """Time the production helper with ``critical_curves_method`` forced."""
    import numpy as np
    from autogalaxy.util import plot_utils
    from autolens.lens.plot.critical_curves import _compute_critical_curve_lines
    from autonerves import conf

    conf.instance["visualize"]["general"]["general"]["critical_curves_method"] = method
    if plot_utils._critical_curves_method() != method:
        raise RuntimeError(f"visualize config did not take critical_curves_method={method}")
    start = time.perf_counter()
    out = _compute_critical_curve_lines(tracer, grid)
    elapsed = time.perf_counter() - start
    image_lines, image_colors, source_lines, source_colors = out
    split = {}
    for prefix, lines, colors in (
        ("critical_curve", image_lines, image_colors),
        ("caustic", source_lines, source_colors),
    ):
        lines = lines or []
        for kind, colour in (("tangential", "white"), ("radial", "yellow")):
            split[f"{kind}_{prefix}"] = [
                np.asarray(c, dtype=float).tolist()
                for c, col in zip(lines, colors or [])
                if col == colour
            ]
    return {"seconds": elapsed, "none_returned": out[0] is None and out[2] is None}, split


def reference_curves(tracer):
    import autolens as al
    import numpy as np
    from autogalaxy.operate.lens_calc import LensCalc

    hw, ps = REFERENCE["half_width"], REFERENCE["pixel_scale"]
    n = round(2 * hw / ps)
    grid = al.Grid2D.uniform(shape_native=(n, n), pixel_scales=ps)
    calc = LensCalc.from_mass_obj(tracer)
    curves = {}
    for quantity in dispatch.QUANTITIES:
        fn = getattr(calc, quantity + "_list_from")
        try:
            curves[quantity] = [
                np.asarray(c, dtype=float).tolist() for c in fn(grid=grid, pixel_scale=ps)
            ]
        except ValueError:
            curves[quantity] = []
    edge = hw - 2 * ps
    clipped = any(
        np.abs(np.asarray(c)).max() >= edge
        for q in ("tangential_critical_curve", "radial_critical_curve")
        for c in curves[q]
        if len(c)
    )
    return curves, clipped


def reused_calc_seconds(tracer):
    """Cold then warm on one LensCalc: the cost if the compile cache were hit.

    Production builds a fresh LensCalc per figure, so it never sees the warm call.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    calc = LensCalc.from_mass_obj(tracer)
    seconds = []
    for _ in range(2):
        start = time.perf_counter()
        for quantity in dispatch.QUANTITIES:
            dispatch.synchronized_arrays(getattr(calc, quantity + "_list_via_zero_contour_from")())
        seconds.append(time.perf_counter() - start)
    return {"cold": seconds[0], "warm": seconds[1]}


def coarse_tangential_count(tracer, grid):
    from autogalaxy.operate.lens_calc import LensCalc

    try:
        calc = LensCalc.from_mass_obj(tracer)
        return len(calc.tangential_critical_curve_list_from(grid=grid, pixel_scale=COARSE))
    except ValueError:
        return 0


def grade(curves, refs, quantity):
    """Phase-3a comparison plus the radial resolution gate and closure check."""
    import numpy as np

    if quantity.startswith("radial"):
        refs_kept = []
        for r in refs:
            if dispatch.geometry(r)["area"] > RADIAL_GRADE_AREA:
                refs_kept.append(r)
        if not refs_kept:
            return {"graded": False, "reference_count": len(refs), "count": len(curves)}
        refs = refs_kept
        # Production drops small radial curves too; match only components that
        # sit near a graded reference so sub-resolution extras do not fail it.
        curves = [c for c in curves if min(dispatch.distance(c, r) for r in refs) <= 10 * TOLERANCE]
    try:
        result = dispatch.comparison(curves, refs, TOLERANCE)
    except ValueError as exc:
        return {"graded": True, "pass": False, "error": str(exc)}
    result.pop("metrics", None)
    result["graded"] = True
    result["closed"] = (
        bool(all(dispatch.geometry(c)["max_segment"] <= CLOSED_MAX_SEGMENT for c in curves))
        if curves
        else False
    )
    result["max_segment"] = max((dispatch.geometry(c)["max_segment"] for c in curves), default=None)
    result["reference_areas"] = [dispatch.geometry(r)["area"] for r in refs]
    if curves:
        result["centroid_spread"] = float(
            np.ptp([dispatch.geometry(c)["centroid"] for c in curves], axis=0).max()
        )
    return result


def worker(leg_json, out_path, n_draws, samples_root):
    import autolens as al

    leg = json.loads(leg_json)
    leg_dir = Path(samples_root) / leg["output_path"].replace(RAL_ROOT, "")
    spec = PRODUCTION[(leg["dataset_class"], leg["instrument"])]
    grid = al.Grid2D.uniform(shape_native=spec["shape"], pixel_scales=spec["pixel_scale"])
    seed = int(dispatch.hashlib.sha256(leg["record"].encode()).hexdigest()[:8], 16)
    model, draws = load_draws(leg_dir, n_draws, seed)
    row = {
        "leg": leg,
        "scale": SCALE[leg["dataset_class"]],
        "production_grid": spec,
        "reference_grid": REFERENCE,
        "rule": {
            "tolerance": TOLERANCE,
            "closed_max_segment": CLOSED_MAX_SEGMENT,
            "radial_grade_area": RADIAL_GRADE_AREA,
            "coarse_pixel_scale": COARSE,
        },
        "draw_seed": seed,
        "provenance": {**dispatch.provenance(), "script_sha256": dispatch.digest(SCRIPT)},
        "draws": [],
    }
    for label, sample_index, vector in draws:
        tracer = tracer_from(model, vector)
        entry = {"label": label, "sample_index": sample_index}
        refs, clipped = reference_curves(tracer)
        entry["reference_clipped"] = clipped
        entry["reference_counts"] = {q: len(v) for q, v in refs.items()}
        for method in ("marching_squares", "zero_contour"):
            try:
                timing, curves = production_lines(tracer, grid, method)
                entry[method] = {
                    "timing": timing,
                    "grades": {q: grade(curves[q], refs[q], q) for q in dispatch.QUANTITIES},
                }
            except Exception as exc:  # noqa: BLE001 - recorded as a failure, never a pass
                entry[method] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        if label == "max_likelihood":
            # A repeat on the same tracer: does a fresh LensCalc recompile?
            timing, _ = production_lines(tracer, grid, "zero_contour")
            entry["zero_contour_repeat_s"] = timing["seconds"]
            entry["zero_contour_reused_calc_s"] = reused_calc_seconds(tracer)
        entry["coarse_tangential_count"] = coarse_tangential_count(tracer, grid)
        row["draws"].append(entry)
        Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))
    row["complete"] = True
    Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))


def run(selected, samples_root, output, classes, n_draws, limit):
    output.mkdir(parents=True, exist_ok=True)
    legs = [
        r
        for r in json.loads(Path(selected).read_text())
        if r["dataset_class"] in classes and r["status"] == "complete"
    ]
    if limit:
        legs = legs[:limit]
    for i, leg in enumerate(legs):
        name = leg["record"].replace("results/", "").replace("/stages.json", "").replace("/", "__")
        out = output / f"{name}.json"
        if out.exists() and json.loads(out.read_text()).get("complete"):
            print(f"[{i + 1}/{len(legs)}] skip {name}", flush=True)
            continue
        leg_dir = Path(samples_root) / leg["output_path"].replace(RAL_ROOT, "")
        if not (leg_dir / "files/samples.csv").exists():
            print(f"[{i + 1}/{len(legs)}] MISSING samples {name}", flush=True)
            continue
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
        env.pop("PYAUTO_FAST_PLOTS", None)
        env.pop("PYAUTO_SMALL_DATASETS", None)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--worker",
            json.dumps(leg),
            str(out),
            "--draws",
            str(n_draws),
            "--samples-root",
            str(samples_root),
        ]
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = proc.wait(timeout=LEG_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                code = "timeout"
        print(
            f"[{i + 1}/{len(legs)}] {name} exit={code} {time.perf_counter() - start:.0f}s",
            flush=True,
        )


def summarize(output):
    import numpy as np

    rows = [json.loads(p.read_text()) for p in sorted(Path(output).glob("*.json"))]
    table = {}
    for row in rows:
        scale = row["scale"]
        for d in row["draws"]:
            if d["reference_clipped"]:
                table.setdefault((scale, "clipped"), 0)
                table[(scale, "clipped")] += 1
                continue
            for method in ("marching_squares", "zero_contour"):
                e = d[method]
                key = (scale, method)
                t = table.setdefault(
                    key,
                    {
                        "draws": 0,
                        "errors": 0,
                        "fail": {},
                        "graded": {},
                        "unclosed": 0,
                        "seconds": [],
                    },
                )
                t["draws"] += 1
                if "error" in e:
                    t["errors"] += 1
                    continue
                t["seconds"].append(e["timing"]["seconds"])
                for q, g in e["grades"].items():
                    if not g.get("graded"):
                        continue
                    t["graded"][q] = t["graded"].get(q, 0) + 1
                    if not g.get("pass"):
                        t["fail"][q] = t["fail"].get(q, 0) + 1
                    if method == "zero_contour" and not g.get("closed", True):
                        t["unclosed"] += 1
    for key, t in sorted(table.items()):
        if isinstance(t, int):
            print(f"{key}: {t} reference-clipped draws excluded")
            continue
        s = np.array(t.pop("seconds")) if t["seconds"] else np.array([np.nan])
        print(
            f"{key}: draws={t['draws']} errors={t['errors']} unclosed={t['unclosed']} "
            f"graded={t['graded']} fail={t['fail']} "
            f"s: first={s[0]:.3f} median={np.median(s):.3f} max={s.max():.3f}"
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--samples-root", type=Path)
    parser.add_argument("--classes", default="imaging,interferometer")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], args.draws, args.samples_root)
    elif args.run:
        run(
            args.selected,
            args.samples_root,
            args.run,
            args.classes.split(","),
            args.draws,
            args.limit,
        )
    elif args.summarize:
        summarize(args.summarize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
