"""
Critical Curves: Cluster Dispatch Audit
======================================

Opt-in CPU fp64 evidence for Source & Cluster phase 3a. Run ``--run`` from the
profiling repo root; ``--summarize`` validates the saved evidence without writing it.
Library behavior is observed, never patched. Each worker has a 120-second cap.

__Contents__
Synthetic controls; bounded engine workers; geometric comparisons; provenance.

__Env__
ENV: jax full_datasets real_plots
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = next(p for p in SCRIPT.parents if (p / "ruff.toml").exists())
RESULTS = ROOT / "results/lens/critical_curves"
KINDS = ("tangential", "radial")
QUANTITIES = tuple(f"{k}_{q}" for k in KINDS for q in ("critical_curve", "caustic"))
CASES = ("control", "cluster_z1", "cluster_z2")
TIMEOUT = 120


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    import autoarray
    import autofit
    import autogalaxy
    import autolens
    import autonerves
    import jax

    libraries = {}
    for module in (autoarray, autofit, autogalaxy, autolens, autonerves):
        package = Path(module.__file__).resolve().parent
        root = package.parent
        if not (root / ".git").exists():
            raise RuntimeError(
                f"This source-checkout campaign requires a Git checkout for {module.__name__}"
            )
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        content = hashlib.sha256()
        for path in sorted(
            p for p in package.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml")
        ):
            content.update(str(path.relative_to(package)).encode())
            content.update(bytes.fromhex(digest(path)))
        libraries[module.__name__] = {"sha": sha, "python_config_sha256": content.hexdigest()}
    return {
        "libraries": libraries,
        "script_sha256": digest(SCRIPT),
        "python": sys.version,
        "autolens_version": autolens.__version__,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "jax_zero_contour",
                "numpy",
                "scipy",
                "scikit-image",
            )
        },
        "devices": [str(d) for d in jax.devices()],
        "host": platform.node(),
        "platform": platform.platform(),
        "workspace_config_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "config").rglob("*.yaml"))
        },
        "x64": jax.config.x64_enabled,
    }


"""__Synthetic fixtures__

The control is a cored spherical isothermal lens: alpha(r)=b*r/(sqrt(r²+s²)+s).
Its two critical radii are analytic. The cluster uses the test workspace's
z_lens=0.5, z_source=(1,2) convention, but a deliberately small smooth host and
one offset member so the audit needs no simulator files or forward point solve.
"""


def fixture(case):
    import autolens as al
    from autogalaxy.operate.lens_calc import LensCalc

    if case == "control":
        profile = al.mp.IsothermalCoreSph(einstein_radius=2.0, core_radius=0.2)
        return LensCalc.from_mass_obj(profile), 4.0, 0.1, 0.05, {"b": 2.0, "s": 0.2}
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalCoreSph(einstein_radius=12.0, core_radius=1.2),
            ),
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.IsothermalCoreSph(
                    centre=(5.0, 3.0), einstein_radius=1.2, core_radius=0.2
                ),
            ),
            al.Galaxy(redshift=1.0),
            al.Galaxy(redshift=2.0),
        ]
    )
    j = 1 if case == "cluster_z1" else 2
    return (
        LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=j),
        20.0,
        0.25,
        0.125,
        {
            "plane_j": j,
            "redshifts": [0.5, 1.0, 2.0],
            "host": {"b": 12.0, "s": 1.2, "centre": [0.0, 0.0]},
            "member": {"b": 1.2, "s": 0.2, "centre": [5.0, 3.0]},
        },
    )


def synchronized_arrays(curves):
    import numpy as np

    # Device-to-host conversion waits for completion, including caustic deflections.
    return [np.asarray(c, dtype=float).tolist() for c in curves]


def worker(case, engine, mode, reference):
    import autolens as al
    import numpy as np
    from autogalaxy.operate.lens_calc import evaluation_grid

    calc, extent, coarse, fine, spec = fixture(case)
    spacing = coarse if mode == "coarse" else fine
    grid = al.Grid2D.uniform(shape_native=(round(2 * extent / spacing),) * 2, pixel_scales=spacing)

    @evaluation_grid
    def inspect_grid(_calc, grid, pixel_scale):
        return {
            "shape": list(grid.shape_native),
            "pixel_scales": list(grid.pixel_scales),
            "extent_yx": [
                np.min(np.asarray(grid), axis=0).tolist(),
                np.max(np.asarray(grid), axis=0).tolist(),
            ],
        }

    row = {
        "case": case,
        "engine": engine,
        "mode": mode,
        "fixture": spec,
        "requested_grid": {"half_width": extent, "pixel_scale": spacing},
        "effective_grid": inspect_grid(calc, grid, pixel_scale=spacing),
        "quantities": {},
    }
    row["zero_contour_settings"] = (
        {
            "delta": 0.05,
            "N": 500,
            "tol": 1e-6,
            "max_newton": 5,
            "automatic_seed_half_width": 3.0,
            "automatic_seed_shape": [25, 25],
        }
        if engine == "zero_contour"
        else None
    )
    row["loadavg_before"] = list(getattr(os, "getloadavg", lambda: ())())
    row["execution"] = {
        key: os.environ.get(key)
        for key in (
            "JAX_ENABLE_X64",
            "JAX_PLATFORMS",
            "JAX_DISABLE_JIT",
            "JAX_ENABLE_COMPILATION_CACHE",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "NPROC",
        )
    }
    before = provenance()
    refs = json.loads(Path(reference).read_text()) if reference else None
    for quantity in QUANTITIES:
        kind = quantity.split("_")[0]
        if engine == "marching_squares":
            fn = getattr(calc, quantity + "_list_from")
            kwargs = {"grid": grid, "pixel_scale": spacing}
        else:
            fn = getattr(calc, quantity + "_list_via_zero_contour_from")
            kwargs = {}
            if mode in ("explicit", "disabled_jit", "outer_jit"):
                curves = refs["quantities"][kind + "_critical_curve"]["curves"]
                kwargs["init_guess"] = np.array([c[len(c) // 2] for c in curves])
        times = []
        results = None
        try:
            if mode == "outer_jit":
                import jax
                import jax.numpy as jnp

                seed = jnp.asarray(kwargs.pop("init_guess"))
                results = synchronized_arrays(
                    jax.jit(lambda s, fn=fn, kwargs=kwargs: fn(init_guess=s, **kwargs))(seed)
                )
            else:
                for _ in range(3):
                    start = time.perf_counter()
                    results = synchronized_arrays(fn(**kwargs))
                    times.append(time.perf_counter() - start)
            row["quantities"][quantity] = {
                "status": "ok",
                "curves": results,
                "cold_s": times[0] if times else None,
                "warm_s": times[1:],
            }
        except Exception as exc:
            row["quantities"][quantity] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
            }
        # Preserve partial results if a later quantity exceeds the process cap.
        row["provenance_before"] = before
        Path(os.environ["AUDIT_RESULT"]).write_text(json.dumps(row, allow_nan=False))
    row["loadavg_after"] = list(getattr(os, "getloadavg", lambda: ())())
    row["provenance_after"] = provenance()
    row["provenance_stable"] = row["provenance_after"] == before
    return row


"""__Geometry__

Use polygon centroids/areas and symmetric point-to-segment Hausdorff distances,
not sample-count-weighted centroids or nearest-vertex distances. Match components
one-to-one. Empty sets are explicit failures when the reference has a component.
The predeclared distance bound is two fine-grid pixels; area/centroid differences
remain reported even when the distance bound passes. No tolerance tuning to JAX.
"""


def geometry(curve):
    import numpy as np

    pts = np.asarray(curve, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all():
        raise ValueError("Invalid curve")
    nxt = np.roll(pts, -1, axis=0)
    cross = pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]
    signed = cross.sum() / 2
    centroid = (
        ((pts + nxt) * cross[:, None]).sum(axis=0) / (6 * signed)
        if abs(signed) > 1e-12
        else pts.mean(axis=0)
    )
    segments = np.linalg.norm(nxt - pts, axis=1)
    return {
        "area": float(abs(signed)),
        "centroid": centroid.tolist(),
        "points": len(pts),
        "max_segment": float(segments.max()),
        "closure_gap": float(np.linalg.norm(pts[0] - pts[-1])),
    }


def distance(a, b):
    import numpy as np

    def directed(points, curve):
        points, curve = np.asarray(points), np.asarray(curve)
        ends = np.roll(curve, -1, axis=0)
        delta = ends - curve
        length2 = (delta * delta).sum(axis=1)
        # Chunk to bound memory for full cluster contours.
        maxima = []
        for chunk in np.array_split(points, max(1, len(points) // 128)):
            diff = chunk[:, None] - curve
            t = np.clip((diff * delta).sum(axis=2) / np.maximum(length2, 1e-30), 0, 1)
            d = np.linalg.norm(diff - t[:, :, None] * delta, axis=2)
            maxima.append(d.min(axis=1).max())
        return max(maxima)

    return float(max(directed(a, b), directed(b, a)))


def comparison(curves, refs, tolerance):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    metrics = [geometry(c) for c in curves]
    ref_metrics = [geometry(c) for c in refs]
    if not curves or not refs:
        return {
            "count": len(curves),
            "reference_count": len(refs),
            "metrics": metrics,
            "pairs": [],
            "pass": len(curves) == len(refs),
        }
    distances = np.array([[distance(c, r) for r in refs] for c in curves])
    ci, ri = linear_sum_assignment(distances)
    pairs = [
        {
            "component": int(i),
            "reference": int(j),
            "distance": float(distances[i, j]),
            "area_difference": metrics[i]["area"] - ref_metrics[j]["area"],
            "centroid_distance": float(
                np.linalg.norm(np.array(metrics[i]["centroid"]) - ref_metrics[j]["centroid"])
            ),
        }
        for i, j in zip(ci, ri)
    ]
    return {
        "count": len(curves),
        "reference_count": len(refs),
        "metrics": metrics,
        "tolerance": tolerance,
        "pairs": pairs,
        "pass": len(curves) == len(refs) and all(p["distance"] <= tolerance for p in pairs),
    }


def analyze(rows):
    import numpy as np

    summaries = []
    for case in CASES:
        ref = next(
            r
            for r in rows
            if r["case"] == case and r["engine"] == "marching_squares" and r["mode"] == "fine"
        )
        tolerance = 2 * ref["requested_grid"]["pixel_scale"]
        for row in [r for r in rows if r["case"] == case]:
            entry = {k: row[k] for k in ("case", "engine", "mode", "status")}
            entry["quantities"] = {}
            for q, value in row.get("quantities", {}).items():
                if value["status"] != "ok":
                    entry["quantities"][q] = {"status": value["status"]}
                    continue
                ref_value = ref["quantities"][q]
                assert ref_value["status"] == "ok", "No usable reference"
                comp = comparison(value["curves"], ref_value["curves"], tolerance)
                if case == "control":
                    b, s = 2.0, 0.2
                    t = (-s + np.sqrt(s * s + 4 * b * s)) / 2
                    radius = (
                        np.sqrt(b * b - 2 * b * s)
                        if q.startswith("tangential")
                        else np.sqrt(t * t - s * s)
                    )
                    if q.endswith("caustic"):
                        radius = abs(radius - b * radius / (np.sqrt(radius * radius + s * s) + s))
                    comp["analytic_radius"] = float(radius)
                    comp["analytic_max_radial_error"] = max(
                        (
                            float(np.max(np.abs(np.linalg.norm(c, axis=1) - radius)))
                            for c in value["curves"]
                        ),
                        default=None,
                    )
                    comp["analytic_pass"] = (
                        len(value["curves"]) == 1 and comp["analytic_max_radial_error"] <= tolerance
                    )
                if q.endswith("caustic"):
                    parent = q.split("_")[0] + "_critical_curve"
                    comp["parent_curve_pass"] = (
                        entry["quantities"].get(parent, {}).get("pass", False)
                    )
                    comp["admitted"] = comp["pass"] and comp["parent_curve_pass"]
                entry["quantities"][q] = comp
            summaries.append(entry)
    return summaries


def cap_probe():
    import autolens as al
    import numpy as np
    from autogalaxy.operate.lens_calc import evaluation_grid
    from autonerves import conf

    @evaluation_grid
    def capture(_, grid, pixel_scale):
        return {
            "shape": list(grid.shape_native),
            "pixel_scales": list(grid.pixel_scales),
            "extent_yx": [
                np.min(np.asarray(grid), axis=0).tolist(),
                np.max(np.asarray(grid), axis=0).tolist(),
            ],
        }

    grid = al.Grid2D.uniform(
        shape_native=(120, 120), pixel_scales=0.5, respect_small_datasets=False
    )
    return {
        "input_shape": [120, 120],
        "input_pixel_scale": 0.5,
        "requested_pixel_scale": 0.05,
        "configured_cap": conf.instance["general"]["grid"]["max_evaluation_grid_size"],
        "effective": capture(None, grid, pixel_scale=0.05),
    }


def run(output, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from _profile_cli import device_info_dict

    start = provenance()
    rows = []
    for case in CASES:
        tasks = [
            ("marching_squares", "coarse"),
            ("marching_squares", "fine"),
            ("zero_contour", "auto"),
            ("zero_contour", "explicit"),
        ]
        if case == "control":
            tasks += [("zero_contour", "disabled_jit"), ("zero_contour", "outer_jit")]
        for engine, mode in tasks:
            name = f"{case}-{engine}-{mode}"
            result_path = scratch / (name + ".json")
            result_path.unlink(missing_ok=True)
            env = dict(
                os.environ,
                AUDIT_RESULT=str(result_path),
                JAX_ENABLE_X64="1",
                JAX_PLATFORMS="cpu",
                JAX_DISABLE_JIT="1" if mode == "disabled_jit" else "0",
            )
            env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
            env.pop("PYAUTO_SMALL_DATASETS", None)
            env.pop("PYAUTO_FAST_PLOTS", None)
            ref = scratch / f"{case}-marching_squares-fine.json"
            cmd = [sys.executable, str(SCRIPT), "--worker", case, engine, mode]
            if mode in ("explicit", "disabled_jit", "outer_jit"):
                cmd += ["--reference", str(ref)]
            with (scratch / (name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=TIMEOUT)
                    status = "ok" if code == 0 else "error"
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    status = "timeout"
            row = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"case": case, "engine": engine, "mode": mode}
            )
            row["status"] = status
            row["timeout_s"] = TIMEOUT
            if status == "error":
                row["error_excerpt"] = (scratch / (name + ".log")).read_text()[-1800:]
            rows.append(row)
            print(name, status, flush=True)
    end = provenance()
    evidence = {
        "schema": 1,
        "date": date.today().isoformat(),
        "device": {
            **device_info_dict(),
            "jax_compilation_cache_dir": "machine-local path omitted; disabled in workers",
        },
        "dispatch_probe": dispatch_probe(),
        "provenance_before": start,
        "provenance_after": end,
        "provenance_stable": start == end,
        "rows": rows,
        "cap_probe": cap_probe(),
        "summary": analyze(rows),
    }
    assert start == end, "Provenance changed during run"
    output.parent.mkdir(parents=True, exist_ok=True)
    validate(evidence)
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    render(evidence, output.with_suffix(".png"))


def validate(evidence):
    assert evidence["schema"] == 1
    assert evidence["provenance_stable"]
    before, after = evidence["provenance_before"], evidence["provenance_after"]
    assert {k: v for k, v in before.items() if k != "script_sha256"} == {
        k: v for k, v in after.items() if k != "script_sha256"
    }
    if "collection" in evidence:
        assert digest(ROOT / evidence["collection"]["measured_source"]) == before["script_sha256"]
    else:
        assert before["script_sha256"] == after["script_sha256"]
    assert evidence["provenance_before"]["x64"]
    assert all("cpu" in d.lower() for d in evidence["provenance_before"]["devices"])
    assert len(evidence["rows"]) == 14
    expected = {
        (c, e, m)
        for c in CASES
        for e, m in (
            ("marching_squares", "coarse"),
            ("marching_squares", "fine"),
            ("zero_contour", "auto"),
            ("zero_contour", "explicit"),
        )
    }
    expected |= {("control", "zero_contour", m) for m in ("disabled_jit", "outer_jit")}
    assert {(r["case"], r["engine"], r["mode"]) for r in evidence["rows"]} == expected
    assert len(evidence["dispatch_probe"]) == 6
    for probe in evidence["dispatch_probe"]:
        expected_selector = (
            "zero_contour"
            if probe["requested"] == "zero_contour" and not probe["simulate_missing_dependency"]
            else "marching_squares"
        )
        assert probe["selector"] == expected_selector
        assert len(probe["cluster_calls"]) == 8
        assert {c["plane_j"] for c in probe["cluster_calls"]} == {1, 2}
    for row in evidence["rows"]:
        if "provenance_before" in row:
            assert row["provenance_before"] == evidence["provenance_before"]
        if row["status"] == "ok":
            assert set(row["quantities"]) == set(QUANTITIES)
            if row["engine"] == "marching_squares" and row["mode"] == "fine":
                assert all(
                    v["status"] == "ok" and v["curves"] for v in row["quantities"].values()
                ), "Empty/missing reference component"
            assert row["provenance_stable"]
            assert row["provenance_before"] == evidence["provenance_before"]
            for value in row["quantities"].values():
                if value["status"] == "ok" and row["mode"] != "outer_jit":
                    assert len(value["warm_s"]) == 2 and value["cold_s"] >= 0
                    assert all(t >= 0 for t in value["warm_s"])
    assert analyze(evidence["rows"]) == evidence["summary"], (
        "Geometry summary does not match raw curves"
    )
    for row in evidence["summary"]:
        if row["case"] == "control" and row["engine"] == "marching_squares":
            assert len(row["quantities"]) == 4
            assert all(q["analytic_pass"] for q in row["quantities"].values()), (
                "Analytic control failed"
            )
    print("Evidence/provenance/geometry/analytic controls PASS")
    for row in evidence["summary"]:
        checks = [q.get("pass", False) for q in row["quantities"].values()]
        print(
            row["case"],
            row["engine"],
            row["mode"],
            row["status"],
            f"geometry {sum(checks)}/{len(checks)}",
        )


def dispatch_probe():
    """Spy only on engine routing; real numerical geometry is tested separately."""
    import builtins
    from types import SimpleNamespace
    from unittest.mock import patch

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from autogalaxy.util import plot_utils
    from autolens.cluster.plot import cluster_plots
    from autonerves import conf

    config = conf.instance["visualize"]["general"]["general"]
    results = []
    real_import = builtins.__import__

    def without_zero(name, *args, **kwargs):
        if name == "jax_zero_contour":
            raise ImportError("audit simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    tracer = SimpleNamespace(planes=[SimpleNamespace(redshift=z) for z in (0.5, 1.0, 2.0)])
    for requested in ("marching_squares", "zero_contour", "invalid"):
        for missing in (False, True):
            calls = []

            class Spy:
                def __init__(self, plane, calls=calls):
                    self.plane = plane
                    self.calls = calls

                def __getattr__(self, name):
                    def call(**kwargs):
                        self.calls.append({"plane_j": self.plane, "method": name})
                        return []

                    return call

            with patch.dict(config, critical_curves_method=requested):
                with patch("builtins.__import__", without_zero if missing else real_import):
                    selected = plot_utils._critical_curves_method()
                    with patch.object(
                        cluster_plots,
                        "_lens_calc_for_plane",
                        side_effect=lambda tracer, plane_index: Spy(plane_index),
                    ):
                        fig, ax = plt.subplots()
                        try:
                            cluster_plots.plot_critical_curves(
                                tracer=tracer,
                                grid=None,
                                plane_indices=[1, 2],
                                include_radial=True,
                                ax=ax,
                            )
                            cluster_plots.plot_caustics(
                                tracer=tracer,
                                grid=None,
                                plane_indices=[1, 2],
                                include_radial=True,
                                ax=ax,
                            )
                        finally:
                            plt.close(fig)
            results.append(
                {
                    "requested": requested,
                    "simulate_missing_dependency": missing,
                    "selector": selected,
                    "cluster_calls": calls,
                }
            )
    return results


def render(evidence, path):
    """A standalone geometry-admission figure; red experiments are not speed wins."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    rows = evidence["summary"]
    values = [
        [
            (1 if r["quantities"][q].get("admitted", r["quantities"][q].get("pass")) else 0)
            if q in r["quantities"]
            else -1
            for q in QUANTITIES
        ]
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.imshow(
        np.array(values),
        cmap=ListedColormap(["#bdbdbd", "#d95f59", "#4e9e76"]),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(
        range(4),
        ["Tangential curve", "Tangential caustic", "Radial curve", "Radial caustic"],
        rotation=15,
    )
    ax.set_yticks(
        range(len(rows)),
        [f"{r['case']} / {r['engine']} / {r['mode']} [{r['status']}]" for r in rows],
    )
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(
                j,
                i,
                {1: "PASS", 0: "FAIL", -1: "NOT MEASURED"}[val],
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_title(
        "Critical-curve audit: component count + reference-distance gate\nCPU fp64; caustics also require their parent critical curve to pass"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect(directory, log, output):
    """Collect immutable measured workers after a post-processing-only repair."""
    import ast

    measured = ROOT / "results/lens/critical_curves/dispatch_measured_source.txt"
    old = ast.parse(measured.read_text())
    new = ast.parse(SCRIPT.read_text())
    numerical = (
        "fixture",
        "synchronized_arrays",
        "worker",
        "geometry",
        "distance",
        "comparison",
        "analyze",
        "cap_probe",
        "dispatch_probe",
    )

    def bodies(tree):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in numerical
        }

    assert bodies(old) == bodies(new), "Numerical code changed; rerun instead of collecting"
    statuses = {}
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("ok", "timeout", "error"):
            statuses[parts[0]] = parts[1]
    rows = []
    for case in CASES:
        tasks = [
            ("marching_squares", "coarse"),
            ("marching_squares", "fine"),
            ("zero_contour", "auto"),
            ("zero_contour", "explicit"),
        ]
        if case == "control":
            tasks += [("zero_contour", "disabled_jit"), ("zero_contour", "outer_jit")]
        for engine, mode in tasks:
            name = f"{case}-{engine}-{mode}"
            status = statuses[name]
            path = directory / (name + ".json")
            row = (
                json.loads(path.read_text())
                if path.exists()
                else {"case": case, "engine": engine, "mode": mode}
            )
            row.update(status=status, timeout_s=TIMEOUT)
            if status == "error":
                row["error_excerpt"] = (directory / (name + ".log")).read_text()[-1800:]
            rows.append(row)
    before = rows[0]["provenance_before"]
    assert before["script_sha256"] == digest(measured)
    after = provenance()
    assert {k: v for k, v in before.items() if k != "script_sha256"} == {
        k: v for k, v in after.items() if k != "script_sha256"
    }, "Runtime/config/library drift"
    sys.path.insert(0, str(ROOT))
    from _profile_cli import device_info_dict

    evidence = {
        "schema": 1,
        "date": date.today().isoformat(),
        "device": {
            **device_info_dict(),
            "jax_compilation_cache_dir": "machine-local path omitted; disabled in workers",
        },
        "dispatch_probe": dispatch_probe(),
        "provenance_before": before,
        "provenance_after": after,
        "provenance_stable": True,
        "rows": rows,
        "cap_probe": cap_probe(),
        "summary": analyze(rows),
        "collection": {
            "measured_source": str(measured.relative_to(ROOT)),
            "numerical_ast_identical": list(numerical),
            "reason": "Correct CPU validator: TFRT_CPU_0 is a CPU device; preserve completed measurements and partial timeouts.",
        },
    }
    validate(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    render(evidence, output.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("../scratch/critical-curves"))
    args = parser.parse_args()
    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        import autolens

        print("critical-curves imports OK", autolens.__version__)
        return
    if not (args.worker or args.run or args.summarize or args.collect):
        parser.print_help()
        return
    if args.output is None:
        import autolens

        args.output = (
            RESULTS
            / f"dispatch_summary_cpu_fp64_v{autolens.__version__}_{date.today().isoformat()}.json"
        )
    if args.worker:
        row = worker(*args.worker, args.reference)
        Path(os.environ["AUDIT_RESULT"]).write_text(json.dumps(row, allow_nan=False))
    elif args.collect:
        if args.output.exists() or args.run_log is None:
            parser.error("Collection needs --run-log and a new --output path")
        collect(args.collect, args.run_log, args.output)
    elif args.run:
        if args.output.exists():
            parser.error("Evidence exists; choose a new --output path to preserve the earlier run")
        run(args.output, args.scratch.resolve())
    elif args.summarize:
        validate(json.loads(args.output.read_text()))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
