"""The cluster standard cell and its JAX runtime driver (likelihood-lifecycle 2e, #474).

Every setup of one cluster likelihood evaluates one model: the standard cluster of
``dataset/cluster/simple`` (2 main dPIE + 10 scaling-relation dPIE members + 1 NFW host at
z = 0.5; 2 point sources at z = 1.0 and 2.0, so ray-tracing is multi-plane), parameterised the
way ``scripts/cluster/image_plane/likelihood_breakdown.py`` parameterises its fused rows: tight
Gaussian priors on the truth (main lenses fully free, one shared scaling-relation normalisation,
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

The headline of each runtime row is the plain fit class the cluster breakdowns decompose
(``FitPositionsImagePairRepeat`` / ``FitPositionsSource``; decision D3), summed over the two
systems the way a ``FactorGraphModel`` sums its factors and JIT-compiled end to end from a model
instance whose free parameters are arguments. The solved class (``...Solved`` with
``PointSolved``) is a sub-row. Every row records ``resources``, writes ``parity.json`` (the JIT
value against the eager NumPy ``AnalysisPoint`` in the same process, tolerance 1e-4 x |reference|
in fp64) and renders ``fit.png``.
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

#: Constants of the standard cluster (``simulators/cluster.py``).
REDSHIFT_LENS = 0.5
MAIN_LENS_PARAMS = (
    ((0.0, 0.0), 8.0, 20.0, 3.0),
    ((10.0, 8.0), 5.0, 12.0, 1.2),
)
SCALING_B0_REF, SCALING_RS_REF, SCALING_RA, SCALING_EXPONENT = 0.12, 10.0, 0.1, 0.5
MASS_AT_200 = 10**15.3
#: The simulator's true source centres, the source-plane cell's ``Point`` centres.
TRUTH_SOURCE_CENTRES = ((0.3, 0.5), (-0.8, 1.2))

#: The fit classes per likelihood: the plain headline and the solved sub-row (decision D3).
FIT_CLASS_NAMES = {
    "image_plane": {
        "plain": "FitPositionsImagePairRepeat",
        "solved": "FitPositionsImagePairRepeatSolved",
    },
    "source_plane": {"plain": "FitPositionsSource", "solved": "FitPositionsSourceSolved"},
}
#: Steady repeats of the JIT block: the image-plane call re-runs the multi-plane forward solve.
N_REPEATS = {"image_plane": 3, "source_plane": 10}
#: The vmap batch per backend: the A100 table has no cluster entry, and on CPU the batch is
#: capped at 3 like every other cell (``vram.config._NON_GPU_BATCH_CAP``).
VMAP_BATCH = {"cpu": 3, "gpu": 16}
#: The parity quantity both runtime rows compare.
PARITY_QUANTITY = (
    "log_likelihood_function summed over both systems (JIT AnalysisPoint, plain fit class) at "
    "the prior-median instance, vs the eager NumPy AnalysisPoint in the same process"
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

    dataset_path = Path(profiling_root) / "dataset" / "cluster" / INSTRUMENT
    auto_simulate_if_missing(
        dataset_path, dataset_type="cluster", instrument=INSTRUMENT, workspace_root=profiling_root
    )
    dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    lum_ref = max(scaling_table.luminosities)
    z_source_max = max(float(d.redshift) for d in dataset_list)

    galaxies = [
        al.Galaxy(
            redshift=REDSHIFT_LENS,
            mass=al.mp.dPIEMassB0Sph(centre=centre, ra=ra, rs=rs, b0=b0),
        )
        for centre, ra, rs, b0 in MAIN_LENS_PARAMS
    ]
    galaxies += [
        al.Galaxy(
            redshift=REDSHIFT_LENS,
            mass=al.mp.dPIEMassB0Sph(
                centre=tuple(centre),
                ra=SCALING_RA,
                rs=SCALING_RS_REF * (lum / lum_ref) ** SCALING_EXPONENT,
                b0=SCALING_B0_REF * (lum / lum_ref) ** SCALING_EXPONENT,
            ),
        )
        for centre, lum in zip(scaling_table.centres, scaling_table.luminosities)
    ]
    galaxies.append(
        al.Galaxy(
            redshift=REDSHIFT_LENS,
            dark=al.mp.NFWMCRLudlowSph(
                centre=(0.0, 0.0),
                mass_at_200=MASS_AT_200,
                redshift_object=REDSHIFT_LENS,
                redshift_source=z_source_max,
            ),
        )
    )
    lens_tracer = al.Tracer(
        galaxies=galaxies
        + [
            al.Galaxy(redshift=float(d.redshift), **{d.name: al.ps.Point(centre=c)})
            for d, c in zip(dataset_list, TRUTH_SOURCE_CENTRES)
        ]
    )
    positions_list = [np.atleast_2d(np.asarray(d.positions)) for d in dataset_list]
    return {
        "dataset_path": dataset_path,
        "dataset_list": dataset_list,
        "scaling_table": scaling_table,
        "lum_ref": lum_ref,
        "z_source_max": z_source_max,
        "tracer": lens_tracer,
        "positions_list": positions_list,
        "n_mass_profiles": len(galaxies),
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


def _gaussian(mean, fraction=0.01, floor=0.01):
    import autofit as af

    return af.GaussianPrior(mean=float(mean), sigma=max(abs(float(mean)) * fraction, floor))


def model(cell: dict, *, solved: bool, point_centres):
    """The cell's model: tight Gaussians on the truth, the prior-median instance is the truth
    lens with each ``Point`` at ``point_centres`` (``PointSolved`` when ``solved``)."""
    import autofit as af
    import autolens as al

    galaxies = {}
    for i, (centre, ra, rs, b0) in enumerate(MAIN_LENS_PARAMS):
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre.centre_0 = _gaussian(centre[0])
        mass.centre.centre_1 = _gaussian(centre[1])
        mass.ra = _gaussian(ra)
        mass.rs = _gaussian(rs)
        mass.b0 = _gaussian(b0)
        galaxies[f"main_{i}"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, mass=mass)

    table = cell["scaling_table"]
    b0_ref = _gaussian(SCALING_B0_REF, floor=0.001)
    for i, (centre, lum) in enumerate(zip(table.centres, table.luminosities)):
        ratio = (lum / cell["lum_ref"]) ** SCALING_EXPONENT
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre = tuple(centre)
        mass.ra = SCALING_RA
        mass.rs = SCALING_RS_REF * ratio
        mass.b0 = b0_ref * ratio
        galaxies[f"scaling_{i}"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, mass=mass)

    dark = af.Model(
        al.mp.NFWMCRLudlowSph,
        mass_at_200=MASS_AT_200,
        redshift_object=REDSHIFT_LENS,
        redshift_source=cell["z_source_max"],
    )
    dark.centre.centre_0 = _gaussian(0.0)
    dark.centre.centre_1 = _gaussian(0.0)
    galaxies["host_halo"] = af.Model(al.Galaxy, redshift=REDSHIFT_LENS, dark=dark)

    for i, (d, centre) in enumerate(zip(cell["dataset_list"], point_centres)):
        if solved:
            point = af.Model(al.ps.PointSolved)
        else:
            point = af.Model(al.ps.Point)
            point.centre.centre_0 = _gaussian(centre[0])
            point.centre.centre_1 = _gaussian(centre[1])
        galaxies[f"source_{i}"] = af.Model(al.Galaxy, redshift=float(d.redshift), **{d.name: point})

    return af.Collection(galaxies=af.Collection(**galaxies))


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


def _summed(analysis_list):
    def full(params):
        return sum(a.log_likelihood_function(instance=params) for a in analysis_list)

    return full


def run_runtime(likelihood: str, cli, *, profiling_root: Path) -> Path:
    """The JAX runtime row of a cluster likelihood: eager NumPy reference, JIT full pipeline
    (lower / compile / first call / steady), the steady median, the vmap per call, the solved
    sub-row, ``resources``, ``parity.json`` and ``fit.png``. Returns the row's path."""
    if likelihood not in LIKELIHOODS:
        raise ValueError(f"unknown cluster likelihood {likelihood!r}; expected {LIKELIHOODS}")
    import time

    import autolens as al
    import jax
    import jax.numpy as jnp
    import numpy as np
    from autofit.jax import register_model

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
        point_centres_source = "back-traced centroid of each system's observed images"
    else:
        settings, solver = None, None
        point_centres = list(TRUTH_SOURCE_CENTRES)
        point_centres_source = "the simulator's true source centres"

    arms: dict[str, Any] = {}
    plain_compiled = None
    eager_fits = None
    for arm in ("plain", "solved"):
        print(f"\n--- {likelihood} {arm}: {FIT_CLASS_NAMES[likelihood][arm]} ---")
        m = model(cell, solved=arm == "solved", point_centres=point_centres)
        instance = m.instance_from_vector(vector=m.physical_values_from_prior_medians)
        register_model(m)
        params = jax.tree_util.tree_map(jnp.asarray, instance)

        eager = analyses(cell, likelihood, arm, solver, use_jax=False)
        t0 = time.perf_counter()
        eager_value = float(sum(a.log_likelihood_function(instance=instance) for a in eager))
        eager_s = time.perf_counter() - t0
        print(f"  eager NumPy log L = {eager_value!r} ({eager_s:.3f} s)")
        if arm == "plain":
            eager_fits = [a.fit_from(instance=instance) for a in eager]

        full = _summed(analyses(cell, likelihood, arm, solver, use_jax=True))
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
        if arm == "plain":
            plain_compiled, plain_params = compiled, params
        else:
            del compiled
        gc.collect()

    full_per_call = arms["plain"]["single_jit_block_s"]
    full_median = timing.headline_steady_median(
        plain_compiled, plain_params, block_mean_s=full_per_call
    )
    del plain_compiled
    jax.clear_caches()
    gc.collect()

    # vmap: the batched cost a sampler pays per point (the catalogue's JAX per-call headline).
    backend = jax.default_backend()
    batch_size = VMAP_BATCH["gpu" if backend == "gpu" else "cpu"]
    print(f"\n--- vmap (batch {batch_size}, declared cluster_standard.VMAP_BATCH) ---")
    full_plain = _summed(analyses(cell, likelihood, "plain", solver, use_jax=True))
    vmap: dict[str, Any] | None = None
    attempts = []
    for nb in sorted({batch_size, max(1, batch_size // 2), 1}, reverse=True):
        batched = jax.tree_util.tree_map(
            lambda leaf, n=nb: jnp.broadcast_to(leaf, (n, *jnp.shape(leaf))), plain_params
        )
        try:
            per_call = timing.vmap_profile(
                full_plain,
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
        "likelihood": f"{likelihood} ({FIT_CLASS_NAMES[likelihood]['plain']})",
        "n_systems": len(cell["dataset_list"]),
        "n_images_total": int(sum(len(p) for p in cell["positions_list"])),
        "n_planes": len(cell["tracer"].planes),
        "n_mass_profiles": cell["n_mass_profiles"],
        "free_parameters": arms["plain"]["free_parameters"],
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
        "fit_positions_cls": FIT_CLASS_NAMES[likelihood]["plain"],
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "configuration": configuration,
        "eager_log_likelihood": arms["plain"]["eager_log_likelihood"],
        "log_likelihood_jit": arms["plain"]["log_likelihood_jit"],
        "full_pipeline_single_jit": full_per_call,
        **full_median,
        "vmap": vmap
        if vmap is not None
        else {"status": "failed", "attempts": attempts, "note": "every vmap batch failed"},
        "jit_phases": jit_records,
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
        log_likelihood=arms["plain"]["log_likelihood_jit"],
        reference_log_likelihood=arms["plain"]["eager_log_likelihood"],
        use_mixed_precision=cli.use_mixed_precision,
        reference_source=(
            f"eager NumPy AnalysisPoint(use_jax=False, {FIT_CLASS_NAMES[likelihood]['plain']}) "
            f"summed over {len(cell['dataset_list'])} systems in the same process: same dataset, "
            "model, instance and solver"
        ),
        autolens_version=al_version,
        quantity=PARITY_QUANTITY,
        dataset_type="cluster",
    )
    # Fit figure (likelihood-lifecycle 1B): every system's eager fit at the headline instance.
    if not np.isfinite(arms["plain"]["eager_log_likelihood"]):
        print("  WARNING: the eager reference is not finite; fit.png still drawn")
    render_fit_png(eager_fits, dict_path.parent, dataset_type="cluster", cli=cli, solver=solver)
    return dict_path
