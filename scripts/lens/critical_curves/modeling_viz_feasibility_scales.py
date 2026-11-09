"""
Critical Curves: Zero-Contour Feasibility at Group and Cluster Scale
===================================================================

Group and cluster legs of autolens_profiling#521. This extends
``modeling_viz_feasibility.py`` (galaxy scale), which it imports and does not
modify, under the same pre-registered rule. Library behaviour is observed, never
patched.

__Group (imaging/euclid)__

Same production path as galaxy: ``_compute_critical_curve_lines`` on the fit's
unmasked grid (250^2 at 0.1 arcsec, ``evaluation_grid`` resampled to 0.05
arcsec). Reference: marching squares at 0.02 arcsec over +/-10 arcsec (1000^2,
at the cap). Tolerance 0.1 arcsec, two production pixels, as for galaxy.

__Cluster (cluster/simple, point-source fits)__

Cluster figures do not use ``LensCalc`` or ``critical_curves_method``. Point fits
draw each source's own plane with ``point_sources.curves_for_planes``: deflection
fields contoured at ``CURVE_GRID_SHAPE`` (400^2) over the solver extent (100x100
at 1 arcsec, i.e. +/-50 arcsec). Production is that call, timed as is. The
zero-contour candidate is ``LensCalc.from_tracer(use_multi_plane=True,
plane_j=j)`` per source plane, with its automatic seeds (what routing cluster
figures to zero-contour would run). It is also run with explicit seeds from the
reference, as in phase 3a, to separate seed coverage from tracing.

- Reference: ``curves_for_planes`` at 1600^2 (0.0625 arcsec) on the same extent.
- Tolerance: 0.25 arcsec (the phase-3a cluster bound). Radial curves are graded
  only above pi * 0.25^2 arcsec^2.
- The positions-driven enlargement of the extent is not reproduced: every
  observed position lies inside the solver grid. A reference curve within two
  reference pixels of the extent edge marks the draw clipped, and it is excluded.

__Env__
ENV: jax full_datasets
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

import dispatch  # noqa: E402
import modeling_viz_feasibility as mvf  # noqa: E402

GROUP = {
    "production": {"shape": (250, 250), "pixel_scale": 0.1},
    "reference": {"half_width": 10.0, "pixel_scale": 0.02},
}
CLUSTER = {
    "extent": [-50.0, 50.0, -50.0, 50.0],
    "production_shape": 400,
    "reference_shape": 1600,
    "coarse_shape": 100,
    "tolerance": 0.25,
}
PLANE_QUANTITIES = {
    "tangential_critical_curve": "tangential",
    "radial_critical_curve": "radial",
    "tangential_caustic": "tangential_caustics",
    "radial_caustic": "radial_caustics",
}


def galaxy_worker(leg_json, out_path, n_draws, samples_root):
    """Galaxy legs with shared priors (MGE): ``mvf.worker`` with the all-paths loader.

    ``mvf.load_draws`` maps each prior by its first path only, which fails when a
    prior is shared (an MGE's common centre) and the CSV names another of its paths.
    """
    mvf.load_draws = load_factor_draws
    mvf.worker(leg_json, out_path, n_draws, samples_root)
    row = json.loads(Path(out_path).read_text())
    row["scales_script_sha256"] = dispatch.digest(SCRIPT)
    row["loader"] = "load_factor_draws"
    Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))


def group_worker(leg_json, out_path, n_draws, samples_root):
    mvf.load_draws = load_factor_draws
    mvf.PRODUCTION[("group", "euclid")] = GROUP["production"]
    mvf.SCALE["group"] = "group"
    mvf.REFERENCE = GROUP["reference"]
    mvf.worker(leg_json, out_path, n_draws, samples_root)
    row = json.loads(Path(out_path).read_text())
    row["scales_script_sha256"] = dispatch.digest(SCRIPT)
    Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))


def load_factor_draws(leg_dir, n_draws, seed):
    """``mvf.load_draws`` for factor-graph models (one sub-model per point dataset).

    Shared priors appear under one path per factor; ``samples.csv`` names each by
    one of them, so match a column through every path the prior has.
    """
    import csv

    import autofit as af
    import numpy as np

    model = af.Model.from_json(str(leg_dir / "files/model.json"))
    with open(leg_dir / "files/samples.csv") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    params = head[: head.index("log_likelihood")]
    idx = [None] * len(model.all_paths)
    for i, prior_paths in enumerate(model.all_paths):
        hits = {params.index(".".join(p)) for p in prior_paths if ".".join(p) in params}
        if len(hits) > 1:
            raise ValueError(f"prior {prior_paths[0]} maps to several columns")
        if hits:
            idx[i] = hits.pop()
    # A compound prior (e.g. a scaling relation's ``b0.left_``) is labelled by its
    # name in the CSV (``b0.b0_ref``): pair leftovers by path prefix, one-to-one only.
    unused = [c for c in range(len(params)) if c not in idx]
    for i, prior_paths in enumerate(model.all_paths):
        if idx[i] is not None:
            continue
        prefixes = {".".join(p[:-1]) for p in prior_paths}
        hits = [c for c in unused if params[c].rsplit(".", 1)[0] in prefixes]
        if len(hits) != 1:
            raise ValueError(f"cannot map prior {prior_paths[0]} to one samples.csv column")
        idx[i] = hits[0]
        unused.remove(hits[0])
    if unused or len(set(idx)) != len(idx):
        raise ValueError(f"unmapped samples.csv columns: {[params[c] for c in unused]}")
    weight = data[:, head.index("weight")]
    log_l = data[:, head.index("log_likelihood")]
    rng = np.random.default_rng(seed)
    picks = [int(np.argmax(log_l))]
    picks += rng.choice(len(weight), size=n_draws, p=weight / weight.sum()).tolist()
    labels = ["max_likelihood"] + [f"draw_{i}" for i in range(n_draws)]
    return model, [(lab, int(i), data[i, idx].tolist()) for lab, i in zip(labels, picks)]


def factor_tracer_from(model, vector):
    """Full tracer of a factor-graph instance: every factor shares one galaxy set."""
    import autolens as al

    inst = model.instance_from_vector(vector, ignore_assertions=True)
    if hasattr(inst, "galaxies"):
        return mvf.tracer_from(model, vector)
    factor = next(v for k, v in vars(inst).items() if k.isdigit() and hasattr(v, "galaxies"))
    return al.Tracer(galaxies=list(factor.galaxies))


def plane_curves(tracer, planes, shape):
    import numpy as np
    from autolens.point.plot.point_sources import curves_for_planes

    start = time.perf_counter()
    curves = curves_for_planes(tracer, planes, CLUSTER["extent"], shape=shape)
    elapsed = time.perf_counter() - start
    out = {}
    for j, pc in curves.items():
        for q, attr in PLANE_QUANTITIES.items():
            out[f"{q}@plane{j}"] = [np.asarray(c, dtype=float).tolist() for c in getattr(pc, attr)]
    return elapsed, out


def zero_contour_planes(tracer, planes, seeds=None):
    """Zero-contour per source plane. ``seeds`` maps plane -> reference tangential curves."""
    import numpy as np
    from autogalaxy.operate.lens_calc import LensCalc

    out, errors = {}, {}
    start = time.perf_counter()
    for j in planes:
        calc = LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=j)
        quantities = (
            dispatch.QUANTITIES
            if seeds is None
            else (
                "tangential_critical_curve",
                "tangential_caustic",
            )
        )
        for q in quantities:
            kwargs = {}
            if seeds is not None:
                refs = seeds.get(j, [])
                if not refs:
                    continue
                kwargs["init_guess"] = np.array([c[len(c) // 2] for c in refs])
            try:
                curves = getattr(calc, q + "_list_via_zero_contour_from")(**kwargs)
                out[f"{q}@plane{j}"] = dispatch.synchronized_arrays(curves)
            except Exception as exc:  # noqa: BLE001 - recorded, never a pass
                errors[f"{q}@plane{j}"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                out[f"{q}@plane{j}"] = []
    return time.perf_counter() - start, out, errors


def reused_calc_planes(tracer, planes):
    from autogalaxy.operate.lens_calc import LensCalc

    result = {}
    for j in planes:
        calc = LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=j)
        seconds = []
        for _ in range(2):
            start = time.perf_counter()
            try:
                for q in dispatch.QUANTITIES:
                    dispatch.synchronized_arrays(getattr(calc, q + "_list_via_zero_contour_from")())
            except Exception as exc:  # noqa: BLE001
                result[j] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                break
            seconds.append(time.perf_counter() - start)
        else:
            result[j] = {"cold": seconds[0], "warm": seconds[1]}
    return result


def clipped(refs):
    import numpy as np

    x_min, x_max, y_min, y_max = CLUSTER["extent"]
    pad = 2 * (x_max - x_min) / CLUSTER["reference_shape"]
    for key, curves in refs.items():
        if "critical_curve" not in key:
            continue
        for c in curves:
            c = np.asarray(c)
            if len(c) and (
                c[:, 0].min() <= y_min + pad
                or c[:, 0].max() >= y_max - pad
                or c[:, 1].min() <= x_min + pad
                or c[:, 1].max() >= x_max - pad
            ):
                return True
    return False


def cluster_worker(leg_json, out_path, n_draws, samples_root):
    mvf.TOLERANCE = CLUSTER["tolerance"]
    mvf.RADIAL_GRADE_AREA = 3.141592653589793 * CLUSTER["tolerance"] ** 2
    leg = json.loads(leg_json)
    leg_dir = Path(samples_root) / leg["output_path"].replace(mvf.RAL_ROOT, "")
    seed = int(dispatch.hashlib.sha256(leg["record"].encode()).hexdigest()[:8], 16)
    model, draws = load_factor_draws(leg_dir, n_draws, seed)
    row = {
        "leg": leg,
        "scale": "cluster",
        "production": "point_sources.curves_for_planes",
        "cluster": CLUSTER,
        "rule": {
            "tolerance": mvf.TOLERANCE,
            "closed_max_segment": mvf.CLOSED_MAX_SEGMENT,
            "radial_grade_area": mvf.RADIAL_GRADE_AREA,
        },
        "draw_seed": seed,
        "provenance": {
            **dispatch.provenance(),
            "script_sha256": dispatch.digest(mvf.SCRIPT),
            "scales_script_sha256": dispatch.digest(SCRIPT),
        },
        "draws": [],
    }
    for label, sample_index, vector in draws:
        tracer = factor_tracer_from(model, vector)
        planes = list(range(1, len(tracer.planes)))
        entry = {
            "label": label,
            "sample_index": sample_index,
            "plane_redshifts": [float(p.redshift) for p in tracer.planes],
        }
        ref_s, refs = plane_curves(tracer, planes, CLUSTER["reference_shape"])
        entry["reference_s"] = ref_s
        entry["reference_clipped"] = clipped(refs)
        entry["reference_counts"] = {q: len(v) for q, v in refs.items()}
        try:
            seconds, curves = plane_curves(tracer, planes, CLUSTER["production_shape"])
            entry["marching_squares"] = {
                "timing": {"seconds": seconds},
                "grades": {q: mvf.grade(curves[q], refs[q], q) for q in refs},
            }
        except Exception as exc:  # noqa: BLE001
            entry["marching_squares"] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        seconds, curves, errors = zero_contour_planes(tracer, planes)
        entry["zero_contour"] = {
            "timing": {"seconds": seconds},
            "errors": errors,
            "grades": {q: mvf.grade(curves.get(q, []), refs[q], q) for q in refs},
        }
        seeds = {j: refs[f"tangential_critical_curve@plane{j}"] for j in planes}
        seconds, curves, errors = zero_contour_planes(tracer, planes, seeds=seeds)
        entry["zero_contour_explicit"] = {
            "timing": {"seconds": seconds},
            "errors": errors,
            "grades": {q: mvf.grade(curves[q], refs[q], q) for q in curves},
        }
        if label == "max_likelihood":
            entry["zero_contour_reused_calc_s"] = reused_calc_planes(tracer, planes)
        _, coarse = plane_curves(tracer, planes, CLUSTER["coarse_shape"])
        entry["coarse_tangential_count"] = {
            q: len(v) for q, v in coarse.items() if q.startswith("tangential_critical")
        }
        row["draws"].append(entry)
        Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))
    row["complete"] = True
    Path(out_path).write_text(json.dumps(row, allow_nan=False, default=float))


def worker(leg_json, out_path, n_draws, samples_root):
    dataset_class = json.loads(leg_json)["dataset_class"]
    if dataset_class == "cluster":
        cluster_worker(leg_json, out_path, n_draws, samples_root)
    elif dataset_class == "group":
        group_worker(leg_json, out_path, n_draws, samples_root)
    else:
        galaxy_worker(leg_json, out_path, n_draws, samples_root)


def summarize_corpus(directory):
    """Per-family correctness, failure modes, cost and fallback-detector score.

    Failure modes for zero-contour tangential critical curves (galaxy/group, unclipped draws):
    ``none_found`` (no seed reached the curve), ``extra_dupes`` (the reference curve traced
    more than once, every copy within tolerance) and ``mismatch`` (anything else). The detector
    flags a draw when the zero-contour count differs from the coarse marching-squares count or
    a path is not closed; it is scored against the rule's pass/fail.
    """
    import collections
    import statistics

    rows = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*__seed_*.json"))]
    families, detector = {}, collections.Counter()
    for row in rows:
        leg = row["leg"]["record"].split("/")
        family = f"{leg[1]}/{leg[2]}"
        fam = families.setdefault(
            family,
            {
                "scale": row["scale"],
                "draws": 0,
                "clipped": 0,
                "ms_s": [],
                "zc_s": [],
                "zc_cold_s": [],
                "zc_warm_s": [],
                "modes": collections.Counter(),
                "ms_tangential_fail": 0,
                "graded": 0,
            },
        )
        reused = row["draws"][0].get("zero_contour_reused_calc_s", {})
        for value in [reused] if "warm" in reused else reused.values():
            if "warm" in value:
                fam["zc_cold_s"].append(value["cold"])
                fam["zc_warm_s"].append(value["warm"])
        for d in row["draws"]:
            fam["draws"] += 1
            fam["ms_s"].append(d["marching_squares"]["timing"]["seconds"])
            fam["zc_s"].append(d["zero_contour"]["timing"]["seconds"])
            if d["reference_clipped"]:
                fam["clipped"] += 1
                continue
            for key, g in d["zero_contour"]["grades"].items():
                if not key.startswith("tangential_critical_curve") or not g.get("graded"):
                    continue
                ms = d["marching_squares"]["grades"][key]
                fam["graded"] += 1
                fam["ms_tangential_fail"] += not ms.get("pass")
                n, ref = g.get("count", 0), g.get("reference_count", 0)
                if g.get("pass"):
                    mode = "pass"
                elif n == 0:
                    mode = "none_found"
                elif n > ref and all(p["distance"] <= mvf.TOLERANCE for p in g.get("pairs", [])):
                    mode = "extra_dupes"
                else:
                    mode = "mismatch"
                fam["modes"][mode] += 1
                if row["scale"] != "cluster":
                    flagged = n != d["coarse_tangential_count"] or (n > 0 and not g.get("closed"))
                    detector[(mode != "pass", flagged)] += 1
    summary = {}
    for family, fam in sorted(families.items()):
        med = {
            k: statistics.median(fam[k]) if fam[k] else None
            for k in ("ms_s", "zc_s", "zc_cold_s", "zc_warm_s")
        }
        summary[family] = {
            **{k: v for k, v in fam.items() if not k.endswith("_s")},
            "modes": dict(fam["modes"]),
            "median_s": med,
        }
        print(
            f"{family:28s} {fam['scale']:7s} draws={fam['draws']:3d} clipped={fam['clipped']:2d} "
            f"zc_tangential={dict(fam['modes'])} ms_tangential_fail={fam['ms_tangential_fail']}/"
            f"{fam['graded']} | MS {med['ms_s']:.2f}s ZC {med['zc_s']:.2f}s "
            f"cold {med['zc_cold_s'] or 0:.2f}s warm {med['zc_warm_s'] or 0:.2f}s"
        )
    score = {f"zc_fail={a},flagged={b}": v for (a, b), v in sorted(detector.items())}
    print("fallback detector:", score)
    return {"families": summary, "fallback_detector": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--summarize-corpus", type=Path, help="Results directory to summarise.")
    parser.add_argument("--summary-out", type=Path, help="Write the corpus summary JSON here.")
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--samples-root", type=Path)
    parser.add_argument("--classes", default="group,cluster")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only this leg of --selected (a SLURM array task id).",
    )
    args = parser.parse_args()
    if args.summarize_corpus:
        summary = summarize_corpus(args.summarize_corpus)
        if args.summary_out:
            args.summary_out.write_text(json.dumps(summary, indent=1, default=float))
    elif args.worker:
        worker(args.worker[0], args.worker[1], args.draws, args.samples_root)
    elif args.run:
        # Reuse the galaxy driver (subprocess per leg, LEG_TIMEOUT, resume) with
        # this script as the worker entry point.
        mvf.SCRIPT = SCRIPT
        mvf.LEG_TIMEOUT = 7200
        selected = args.selected
        if args.index is not None:
            legs = json.loads(Path(args.selected).read_text())
            args.run.mkdir(parents=True, exist_ok=True)
            selected = args.run / f"selected_{args.index}.json"
            selected.write_text(json.dumps([legs[args.index]]))
        mvf.run(
            selected,
            args.samples_root,
            args.run,
            args.classes.split(","),
            args.draws,
            args.limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    os.environ.pop("PYAUTO_FAST_PLOTS", None)
    main()
