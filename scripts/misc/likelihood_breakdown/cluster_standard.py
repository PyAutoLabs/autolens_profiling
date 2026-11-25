"""The cluster standard cell and its JAX runtime driver (likelihood-lifecycle 2e, #474).

Every setup of one cluster likelihood evaluates one model: the standard cluster of
``dataset/cluster/simple`` (2 main dPIE + 10 scaling-relation dPIE members + 1 NFW host at
z = 0.5; 2 point sources at z = 1.0 and 2.0, so ray-tracing is multi-plane), parameterised the
way ``scripts/cluster/image_plane/likelihood_breakdown.py`` parameterises its fused rows: tight
Gaussian priors on the truth (main-lens centres, dispersions and cut radii free, one shared scaling-relation normalisation,
a free NFW centre), evaluated at the prior-median instance. This module holds that model, the
PointSolver settings of the standard cell, and the runtime driver the two runtime scripts call,
so the runtime and breakdown of one likelihood cannot drift:

- ``scripts/cluster/image_plane/likelihood_runtime.py`` -- the image-plane chi-squared, which
  forward-solves the lens equation with the ``PointSolver`` for every source;
- ``scripts/cluster/source_plane/likelihood_runtime.py`` -- the source-plane chi-squared, which
  ray-traces the observed images back (no solver).

The PointSolver of the standard cell (``--standard-size``, decision D1 on the campaign page) is
the workspace cluster solver the ``cluster-pointsolver-speed`` epic decided to keep (human
2026-10-23): 100 x 100 @ 1.0", ``pixel_scale_precision`` 0.001, ``magnification_threshold`` 0.1,
``residual_threshold`` 10 (on; 10 x precision, PyAutoLens#800, in 2026.10.25.1). Every setting is
passed explicitly so a later library default cannot move the cell. Without ``--standard-size``
the breakdown keeps its historic 200 x 200 @ 0.7", precision 0.01 solver.

The headline of each runtime row is the solved fit class (``FitPositionsImagePairRepeatSolved`` /
``FitPositionsSourceSolved`` with ``PointSolved``: the source centre is solved analytically, so no
free centre is sampled), the class the workspace cluster search uses (decision D3, decided by the
human 2026-10-29: "the cluster default setup should use solved"). It is summed over the two
systems the way a ``FactorGraphModel`` sums its factors and JIT-compiled end to end from a model
instance whose free parameters are arguments. The plain class (``FitPositionsImagePairRepeat`` /
``FitPositionsSource``, free ``Point`` centres) is a sub-row. Every row records ``resources``,
writes ``parity.json`` for the headline class (the JIT value against the eager NumPy
``AnalysisPoint`` in the same process, tolerance 1e-4 x |reference| in fp64) and renders
``fit.png`` from the headline class's eager fits. The two breakdowns decompose the same headline
class.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

#: The two cluster likelihoods (``scripts/cluster/<likelihood>/``).
LIKELIHOODS = ("image_plane", "source_plane")
#: The instrument folder every cluster row is keyed by (``catalogue_lifecycle.DEFAULT_INSTRUMENTS``).
INSTRUMENT = "simple"

#: The standard cell's PointSolver (decision D1): the workspace cluster solver
#: (``autolens_workspace/scripts/cluster/modeling.py``), kept by the human 2026-10-23 with
#: ``residual_threshold`` on. Explicit, never re-tuned here.
STANDARD_SOLVER = {
    "shape_native": (100, 100),
    "pixel_scales": 1.0,
    "pixel_scale_precision": 0.001,
    "magnification_threshold": 0.1,
    "residual_threshold": 10.0,
}
#: The historic breakdown solver (the tutorial-scale configuration the committed rows used;
#: ``residual_threshold`` was never passed, so the library default applies).
HISTORIC_SOLVER = {
    "shape_native": (200, 200),
    "pixel_scales": 0.7,
    "pixel_scale_precision": 0.01,
    "magnification_threshold": 0.1,
    "residual_threshold": None,
}

#: The fit classes per likelihood: the solved headline and the plain sub-row (decision D3).
FIT_CLASS_NAMES = {
    "image_plane": {
        "plain": "FitPositionsImagePairRepeat",
        "solved": "FitPositionsImagePairRepeatSolved",
    },
    "source_plane": {"plain": "FitPositionsSource", "solved": "FitPositionsSourceSolved"},
}
#: The arm the row is headlined by (timed, vmapped, witnessed, drawn) and its sub-row (D3, human
#: 2026-10-29). The workspace cluster search fits ``FitPositionsSourceSolved``.
HEADLINE_ARM = "solved"
SUB_ARM = "plain"
#: Steady repeats of the JIT block: the image-plane call re-runs the multi-plane forward solve.
N_REPEATS = {"image_plane": 3, "source_plane": 10}
#: The vmap batch per backend: the A100 table has no cluster entry, and on CPU the batch is
#: capped at 3 like every other cell (``vram.config._NON_GPU_BATCH_CAP``).
VMAP_BATCH = {"cpu": 3, "gpu": 16}
#: The parity quantity both runtime rows compare.
PARITY_QUANTITY = (
    "log_likelihood_function summed over both systems (JIT AnalysisPoint, the headline solved fit "
    "class) at the prior-median instance, vs the eager NumPy AnalysisPoint in the same process"
)


def solver_settings(standard: bool) -> dict:
    """The PointSolver settings of the standard cell (``standard``) or the historic breakdown."""
    return dict(STANDARD_SOLVER if standard else HISTORIC_SOLVER)


def settings_record(settings: dict) -> dict:
    """The JSON form of :func:`solver_settings` (tuples as lists, the default spelled out)."""
    return {
        "grid_shape_native": list(settings["shape_native"]),
        "grid_pixel_scales": settings["pixel_scales"],
        "pixel_scale_precision": settings["pixel_scale_precision"],
        "magnification_threshold": settings["magnification_threshold"],
        "residual_threshold": settings["residual_threshold"],
        "residual_threshold_source": (
            "explicit" if settings["residual_threshold"] is not None else "library default"
        ),
    }


def build_solver(settings: dict, *, use_jax: bool = True):
    """``al.PointSolver.for_grid`` with every setting of ``settings`` passed explicitly."""
    import autolens as al

    kwargs = {
        "grid": al.Grid2D.uniform(
            shape_native=tuple(settings["shape_native"]), pixel_scales=settings["pixel_scales"]
        ),
        "pixel_scale_precision": settings["pixel_scale_precision"],
        "magnification_threshold": settings["magnification_threshold"],
        "use_jax": use_jax,
    }
    if settings["residual_threshold"] is not None:
        kwargs["residual_threshold"] = settings["residual_threshold"]
    return al.PointSolver.for_grid(**kwargs)


# --------------------------------------------------------------------------- the cell


def load_cell(profiling_root: Path) -> dict:
    """The dataset, scaling table and truth tracer of the standard cluster."""
    import autolens as al
    import numpy as np

    from _profile_cli import auto_simulate_if_missing
    from simple_cluster import GALLERY_REVISION

    dataset_path = Path(profiling_root) / "dataset" / "cluster" / INSTRUMENT
    auto_simulate_if_missing(
        dataset_path, dataset_type="cluster", instrument=INSTRUMENT, workspace_root=profiling_root
    )
    dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
    truth = al.from_json(file_path=dataset_path / "tracer.json")
    lens_plane, *source_planes = truth.planes
    # Only mass and point profiles enter the timed likelihood.
    lenses = [al.Galaxy(redshift=g.redshift, mass=g.mass) for g in lens_plane[:-1]]
    host = lens_plane[-1]
    lenses.append(al.Galaxy(redshift=host.redshift, dark=host.dark))
    centres = [plane[0].bulge.centre for plane in source_planes]
    sources = [
        al.Galaxy(redshift=d.redshift, **{d.name: al.ps.Point(centre=c)})
        for d, c in zip(dataset_list, centres, strict=True)
    ]
    return {
        "dataset_list": dataset_list,
        "gallery_revision": GALLERY_REVISION,
        "lenses": lenses,
        "source_centres": centres,
        "tracer": al.Tracer(galaxies=lenses + sources),
        "positions_list": [np.asarray(d.positions) for d in dataset_list],
        "n_mass_profiles": len(lenses),
    }


def backtraced_centres(cell: dict) -> list[tuple[float, float]]:
    """Per system, the centroid of its observed images traced to its source plane (the image-
    plane breakdown's step 1: what a fit derives from its ``Point`` centre)."""
    import autolens as al
    import numpy as np

    tracer = cell["tracer"]
    centres = []
    for d, positions in zip(cell["dataset_list"], cell["positions_list"]):
        index = tracer.plane_index_via_redshift_from(redshift=float(d.redshift))
        traced = np.asarray(
            tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(positions))[index]
        )
        centres.append(tuple(float(v) for v in traced.mean(axis=0)))
    return centres


def model(cell: dict, *, solved: bool, point_centres):
    """Build the shared gallery lens model and attach the fitted point profiles."""
    import autofit as af
    import autolens as al

    from simple_cluster import gaussian, lens_model

    model = lens_model(cell["lenses"])
    for i, (d, centre) in enumerate(zip(cell["dataset_list"], point_centres)):
        if solved:
            point = af.Model(al.ps.PointSolved)
        else:
            point = af.Model(al.ps.Point)
            point.centre.centre_0 = gaussian(centre[0])
            point.centre.centre_1 = gaussian(centre[1])
        model.galaxies[f"source_{i}"] = af.Model(
            al.Galaxy, redshift=float(d.redshift), **{d.name: point}
        )

    return model


def fit_classes(likelihood: str) -> dict:
    import autolens as al

    return {arm: getattr(al, name) for arm, name in FIT_CLASS_NAMES[likelihood].items()}


def analyses(cell: dict, likelihood: str, arm: str, solver, *, use_jax: bool) -> list:
    """One ``AnalysisPoint`` per system (the factors a cluster fit sums)."""
    import autolens as al

    cls = fit_classes(likelihood)[arm]
    return [
        al.AnalysisPoint(dataset=d, solver=solver, fit_positions_cls=cls, use_jax=use_jax)
        for d in cell["dataset_list"]
    ]


# --------------------------------------------------------------------------- the runtime


def summed_likelihood(model, analysis_list):
    """Rebuild the physical model from its free parameters inside the timed JIT."""
    import jax.numpy as jnp

    def full(params):
        instance = model.instance_from_vector(vector=params, xp=jnp)
        return sum(a.log_likelihood_function(instance=instance) for a in analysis_list)

    return full


def run_runtime(likelihood: str, cli, *, profiling_root: Path) -> Path:
    """The JAX runtime row of a cluster likelihood: eager NumPy reference, JIT full pipeline
    (lower / compile / first call / steady), the steady median and the vmap per call of the
    headline (solved) class, the plain sub-row, ``resources``, ``parity.json`` and ``fit.png``.
    Returns the row's path."""
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"unknown cluster likelihood {likelihood!r}; expected {LIKELIHOODS}")
    import time

    import autolens as al
    import jax
    import jax.numpy as jnp
    import numpy as np
    from autofit.jax import register_model
    from gradient import harness as gradient_harness

    from _profile_cli import (
        device_info_dict,
        render_fit_png,
        resolve_output_paths,
        resources_dict,
        write_parity_json,
    )

    from . import timing
    from .provenance import thread_environment

    if cli.instrument not in (None, INSTRUMENT):
        raise SystemExit(
            f"cluster cells are keyed by instrument {INSTRUMENT!r}, not {cli.instrument!r}"
        )

    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    n_repeats = N_REPEATS[likelihood]

    with timer.section("cell_load"):
        cell = load_cell(profiling_root)
    standard = bool(cli.standard_size)
    if likelihood == "image_plane":
        settings = solver_settings(standard)
        with timer.section("solver_build"):
            solver = build_solver(settings)
        point_centres = backtraced_centres(cell)
        point_centres_source = (
            "plain sub-row only: back-traced centroid of each system's observed images"
        )
    else:
        settings, solver = None, None
        point_centres = cell["source_centres"]
        point_centres_source = "plain sub-row only: the simulator's true source centres"

    arms: dict[str, Any] = {}
    headline_compiled = None
    eager_fits = None
    solved_centres = None
    # The headline first: its compiled pipeline is kept for the steady median.
    for arm in (HEADLINE_ARM, SUB_ARM):
        print(f"\n--- {likelihood} {arm}: {FIT_CLASS_NAMES[likelihood][arm]} ---")
        m = model(cell, solved=arm == "solved", point_centres=point_centres)
        instance = m.instance_from_vector(vector=m.physical_values_from_prior_medians)
        register_model(m)
        params = jnp.asarray(m.physical_values_from_prior_medians)

        eager = analyses(cell, likelihood, arm, solver, use_jax=False)
        t0 = time.perf_counter()
        eager_value = float(sum(a.log_likelihood_function(instance=instance) for a in eager))
        eager_s = time.perf_counter() - t0
        print(f"  eager NumPy log L = {eager_value!r} ({eager_s:.3f} s)")
        if arm == HEADLINE_ARM:
            eager_fits = [a.fit_from(instance=instance) for a in eager]
            # The analytically solved source centre (beta*) of each system, as the fit read it.
            solved_centres = [
                [float(v) for v in fit.positions.source_plane_coordinate] for fit in eager_fits
            ]

        full = summed_likelihood(m, analyses(cell, likelihood, arm, solver, use_jax=True))
        if arm == HEADLINE_ARM and gradient_harness.gradient_requested():
            # Phase 6 (#507): the headline (solved) class's own fused likelihood.
            device = "gpu" if jax.default_backend() == "gpu" else "cpu"
            return gradient_harness.run_gradient(
                full,
                params,
                point=gradient_harness.POINT_PRIOR_MEDIANS,
                dataset_type="cluster",
                variant=likelihood,
                instrument=INSTRUMENT,
                setup_id=f"jax_{device}_na_fp64",
                path=getattr(cli, "output_path", None),
                configuration={"arm": arm, "fit_positions_cls": FIT_CLASS_NAMES[likelihood][arm]},
            )
        compiled, result = timing.jit_profile(
            full,
            f"full_pipeline_{arm}",
            params,
            n_repeats=n_repeats,
            timer=timer,
            jit_records=jit_records,
        )
        jit_value = float(result)
        print(f"  JIT log L = {jit_value!r}")
        arms[arm] = {
            "fit_positions_cls": FIT_CLASS_NAMES[likelihood][arm],
            "free_parameters": int(m.prior_count),
            "eager_log_likelihood": eager_value,
            "eager_single_call_s": eager_s,
            "log_likelihood_jit": jit_value,
            "single_jit_block_s": jit_records[f"full_pipeline_{arm}"]["steady_per_call_s"],
            "jit_phases": jit_records[f"full_pipeline_{arm}"],
        }
        if arm == HEADLINE_ARM:
            headline_compiled, headline_params, full_headline = compiled, params, full
        else:
            del compiled
        gc.collect()

    headline_cls = FIT_CLASS_NAMES[likelihood][HEADLINE_ARM]
    full_per_call = arms[HEADLINE_ARM]["single_jit_block_s"]
    full_median = timing.headline_steady_median(
        headline_compiled, headline_params, block_mean_s=full_per_call
    )
    del headline_compiled
    jax.clear_caches()
    gc.collect()

    # vmap: the batched cost a sampler pays per point (the catalogue's JAX per-call headline).
    backend = jax.default_backend()
    batch_size = VMAP_BATCH["gpu" if backend == "gpu" else "cpu"]
    print(f"\n--- vmap {headline_cls} (batch {batch_size}, cluster_standard.VMAP_BATCH) ---")
    vmap: dict[str, Any] | None = None
    attempts = []
    for nb in sorted({batch_size, max(1, batch_size // 2), 1}, reverse=True):
        batched = jax.tree_util.tree_map(
            lambda leaf, n=nb: jnp.broadcast_to(leaf, (n, *jnp.shape(leaf))), headline_params
        )
        try:
            per_call = timing.vmap_profile(
                full_headline,
                "full_pipeline",
                batched,
                nb,
                n_repeats=n_repeats,
                timer=timer,
                jit_records=jit_records,
            )
        except Exception as exc:  # noqa: BLE001 — a failed vmap must not cost the row
            attempts.append(
                {"batch": nb, "status": type(exc).__name__, "error": str(exc).splitlines()[0][:300]}
            )
            print(f"  vmap batch {nb}: {type(exc).__name__}, trying smaller")
            jax.clear_caches()
            gc.collect()
            continue
        vmap = {
            "batch_size": nb,
            "batch_size_resolved": batch_size,
            "batch_size_source": "declared (cluster_standard.VMAP_BATCH)",
            "batch_time": per_call * nb,
            "per_call": per_call,
            "speedup_vs_single_jit": round(full_per_call / per_call, 2),
            "instances": "the prior-median instance broadcast to the batch",
            "attempts": attempts,
        }
        break
    jax.clear_caches()
    gc.collect()

    al_version = al.__version__
    configuration = {
        "gallery_revision": cell["gallery_revision"],
        "n_images_per_family": {d.name: len(d.positions) for d in cell["dataset_list"]},
        "mass_parameterization": "dPIEMassSph (sigma, r_core, r_cut)",
        "likelihood": f"{likelihood} ({headline_cls})",
        "headline_arm": HEADLINE_ARM,
        "sub_row_arm": SUB_ARM,
        "n_systems": len(cell["dataset_list"]),
        "n_images_total": int(sum(len(p) for p in cell["positions_list"])),
        "n_planes": len(cell["tracer"].planes),
        "n_mass_profiles": cell["n_mass_profiles"],
        "free_parameters": arms[HEADLINE_ARM]["free_parameters"],
        "solved_source_centres": solved_centres,
        "solved_source_centres_source": (
            "the headline fit's analytic beta* (source_plane_coordinate) per system, eager"
        ),
        "point_centres": [list(c) for c in point_centres],
        "point_centres_source": point_centres_source,
        "instance": "prior medians of tight Gaussians on the truth (cluster_standard.model)",
        "solver": settings_record(settings) if settings is not None else None,
        "standard_size": bool(cli.standard_size),
        "n_repeats": n_repeats,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("jax"),
        "thread_environment": thread_environment(),
        "instrument": INSTRUMENT,
        "model": likelihood,
        "fit_positions_cls": headline_cls,
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "configuration": configuration,
        "eager_log_likelihood": arms[HEADLINE_ARM]["eager_log_likelihood"],
        "log_likelihood_jit": arms[HEADLINE_ARM]["log_likelihood_jit"],
        "full_pipeline_single_jit": full_per_call,
        **full_median,
        "vmap": vmap
        if vmap is not None
        else {"status": "failed", "attempts": attempts, "note": "every vmap batch failed"},
        "jit_phases": jit_records,
        # Both arms in full; ``headline_arm`` names the one the top-level fields report.
        "headline_arm": HEADLINE_ARM,
        "solved": arms["solved"],
        "plain": arms["plain"],
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root / "results" / "runtime" / "cluster" / likelihood,
        default_basename=f"{likelihood}_likelihood_summary_{INSTRUMENT}_v{al_version}",
        cell=likelihood,
        hints=summary,
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Row written: {dict_path}")
    timer.summary()

    # Parity witness: the JIT value against the eager NumPy AnalysisPoint of the same model,
    # instance and solver in this process, after every timed region (1e-4 x |reference|, fp64).
    write_parity_json(
        dict_path.parent,
        log_likelihood=arms[HEADLINE_ARM]["log_likelihood_jit"],
        reference_log_likelihood=arms[HEADLINE_ARM]["eager_log_likelihood"],
        use_mixed_precision=cli.use_mixed_precision,
        reference_source=(
            f"eager NumPy AnalysisPoint(use_jax=False, {headline_cls}) "
            f"summed over {len(cell['dataset_list'])} systems in the same process: same dataset, "
            "model, instance and solver"
        ),
        autolens_version=al_version,
        quantity=PARITY_QUANTITY,
        dataset_type="cluster",
    )
    # Fit figure (likelihood-lifecycle 1B): every system's eager headline (solved) fit at the
    # headline instance.
    if not np.isfinite(arms[HEADLINE_ARM]["eager_log_likelihood"]):
        print("  WARNING: the eager reference is not finite; fit.png still drawn")
    render_fit_png(eager_fits, dict_path.parent, dataset_type="cluster", cli=cli, solver=solver)
    return dict_path
