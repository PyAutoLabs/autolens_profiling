"""Shared driver for a single first-class search profiling cell.

Every leaf script under ``searches/<sampler>/<dataset_class>/<model>.py``
calls :func:`run_search` with its cell identity; this module handles
everything else — CLI parsing, smoke short-circuit, dataset/model/analysis
build, viz-time instrumentation, ``search.fit()``, metric collection, and
JSON+PNG output.

The split between this runner and the per-leaf scripts is deliberate: every
sampler × cell combination shares the same plumbing, so the leaf script is
two lines (import + call) and adding a new sampler is one entry in
``_samplers.SAMPLER_BUILDERS``.
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


import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_WORKSPACE_ROOT = _profiling_root()
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from searches._metrics import attach_viz_timer, collect_metrics  # noqa: E402
from searches._samplers import (  # noqa: E402
    _MULTI_START_AUTOCONV,
    _MULTI_START_CLASSES,
    SAMPLER_BUILDERS,
    multi_start_settings,
    n_live_for,
    nss_settings,
    vmap_batch_for_cell,
)

from _profile_cli import (  # noqa: E402
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

# Samplers that have an ``n_live`` (nested sampling). MAP optimizers such as
# ``multi_start_adam`` do not, and record ``null`` rather than a misleading value.
_SAMPLERS_WITH_N_LIVE = frozenset({"nautilus", "nss"})
from searches._recovery import load_truth, recovery_report  # noqa: E402
from searches._setup import (  # noqa: E402
    build_for_cell,
    cluster_point_fit_cls_for,
    format_best_fit,
    point_source_fit_cls_for,
)

# point_source model_types whose truth source profile is al.ps.PointSolved
# (parameter-free — the *Solved fit classes solve the source centre
# analytically); everything else uses al.ps.PointFlux at the truth centre.
_POINT_SOURCE_SOLVED_MODEL_TYPES = frozenset(
    {"image_plane_solved", "source_plane_solved", "image_plane_repeat_solved"}
)

# cluster model_types whose truth source profiles are al.ps.PointSolved
# (#678 phase B chunk 2); everything else (source_plane / source_plane_tensor)
# keeps al.ps.Point at the truth centre, mirroring
# likelihood_breakdown/{image_plane,source_plane}.py's tracer_solved swap.
_CLUSTER_SOLVED_MODEL_TYPES = frozenset({"source_plane_solved", "image_plane_solved"})


def _recovery_for_cell(dataset_class: str, instrument: str, best_instance) -> dict | None:
    """Truth-recovery report for cells that ship a ``truth.json`` (group only).

    Returns ``None`` for cells without a truth file (every non-group cell) or
    when the best instance is unavailable, so the summary simply omits the block.
    """
    if dataset_class != "group" or best_instance is None:
        return None
    dataset_path = _WORKSPACE_ROOT / "dataset" / "imaging" / "group4_mge" / instrument
    truth = load_truth(dataset_path)
    if truth is None:
        return None
    try:
        return recovery_report(best_instance, truth)
    except Exception as exc:  # never let scoring kill a completed profile
        return {"error": repr(exc)}


def _truth_anchor_for_cell(
    dataset_class: str,
    instrument: str,
    model_type: str,
    dataset: Any,
    max_log_likelihood: float,
) -> dict | None:
    """Truth-anchored log likelihood for point_source cells (#678 phase B).

    Builds the truth tracer directly from ``dataset/point_source/<instrument>/
    truth.json`` — never via ``model.instance_from_prior_medians()`` (that's
    the wrong tool: it reads prior means, not the simulator's actual input, and
    is fragile for solved/free-mix models) — then evaluates it through
    ``al.FitPointDataset`` using the SAME ``fit_positions_cls`` the cell itself
    searches with (``_setup.point_source_fit_cls_for``), so
    ``delta_max_ll_vs_truth`` is directly comparable to the search's own
    ``max_log_likelihood``.

    Returns ``None`` for non point_source / cluster cells. Never raises: on
    any exception the failure is printed loudly and the exception string is
    stored in place of the numeric fields instead of crashing the run.

    Extended for cluster cells (#678 phase B chunk 2) — see
    ``_cluster_truth_anchor`` below; dispatched here so both dataset_classes
    share one entry point from ``run_search``.
    """
    if dataset_class == "cluster":
        return _cluster_truth_anchor(instrument, model_type, dataset, max_log_likelihood)
    if dataset_class != "point_source":
        return None
    try:
        dataset_path = _WORKSPACE_ROOT / "dataset" / "point_source" / instrument
        truth = json.loads((dataset_path / "truth.json").read_text())

        lens = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=truth["lens_centre"],
                ell_comps=truth["lens_ell_comps"],
                einstein_radius=truth["lens_einstein_radius"],
            ),
        )
        if model_type in _POINT_SOURCE_SOLVED_MODEL_TYPES:
            point_0 = al.ps.PointSolved()
        else:
            # Includes "source_plane_tensor": free-centre tensor weighting,
            # truth profile is still PointFlux at the truth centre.
            point_0 = al.ps.PointFlux(centre=truth["source_centre"], flux=truth["source_flux"])
        source = al.Galaxy(redshift=1.0, point_0=point_0)
        truth_tracer = al.Tracer(galaxies=[lens, source])

        solver_kwargs = dataset._profiling_solver_kwargs
        grid = al.Grid2D.uniform(
            shape_native=solver_kwargs["grid_shape"],
            pixel_scales=solver_kwargs["pixel_scale"],
        )
        solver = al.PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=solver_kwargs["pixel_scale_precision"],
            magnification_threshold=solver_kwargs["magnification_threshold"],
        )
        fit_positions_cls = point_source_fit_cls_for(model_type)
        fit = al.FitPointDataset(
            dataset=dataset,
            tracer=truth_tracer,
            solver=solver,
            fit_positions_cls=fit_positions_cls,
        )
        truth_log_likelihood = float(fit.log_likelihood)
        return {
            "truth_log_likelihood": truth_log_likelihood,
            "delta_max_ll_vs_truth": float(max_log_likelihood) - truth_log_likelihood,
        }
    except Exception as exc:  # never let the anchor step kill a completed search
        print(f"  WARNING: truth-anchor step failed [{dataset_class}/{model_type}]: {exc!r}")
        return {"truth_log_likelihood": repr(exc), "delta_max_ll_vs_truth": repr(exc)}


def _cluster_truth_anchor(
    instrument: str,
    model_type: str,
    dataset_list: Any,
    max_log_likelihood: float,
) -> dict | None:
    """Truth-anchored log likelihood for cluster cells (#678 phase B chunk 2).

    Loads the true ``Tracer`` from ``dataset/cluster/<instrument>/
    tracer.json`` (written by ``scripts/misc/simulators/cluster.py`` via
    ``al.output_to_json``) for its lens-plane galaxies (2 main dPIE lenses +
    10 scaling members + 1 host halo, all at the lens redshift) — unlike the
    point_source anchor above, which rebuilds the truth tracer field-by-field
    from a small ``truth.json``, because the cluster tracer carries too many
    mass components to hand-derive.

    **Deviation (verified library gap)**: ``Tracer`` JSON round-tripping
    silently DROPS a ``Galaxy``'s point-source profile (``al.ps.Point`` /
    ``PointSolved``) — confirmed by a minimal ``output_to_json`` +
    ``from_json`` reproduction; only ``bulge``/mass attributes survive. So
    the spec's literal "swap Point for PointSolved inside the loaded tracer"
    cannot work: there is no ``Point`` there to swap. Instead, the source
    galaxies are rebuilt fresh from ``source_centres.json`` (written by the
    same simulator, index-aligned with ``dataset_list``'s ``point_0``,
    ``point_1``, ... order — both derive from the same simulator loop over
    ``source_centres``) — parameter-free ``al.ps.PointSolved`` for
    ``*_solved`` model types, or ``al.ps.Point(centre=<truth centre>)``
    otherwise — dropping the tracer.json copies of those galaxies (only
    their now-useless ``bulge`` survived anyway) and keeping every other
    (lens-plane) galaxy unchanged.

    ``truth_log_likelihood`` sums, over every system in ``dataset_list``,
    ``al.FitPointDataset(dataset=<system>, tracer=<truth>, solver=<the cell's
    solver>, fit_positions_cls=<the cell's fit class>).log_likelihood`` — the
    same per-system fit-total pattern those breakdown scripts use (their
    steps 3/4 for image-plane, 5/6 for source-plane). Never raises: on any
    exception the failure is printed loudly and the exception string is
    stored in place of the numeric fields instead of crashing the run.
    """
    try:
        dataset_path = _WORKSPACE_ROOT / "dataset" / "cluster" / instrument
        truth_tracer = al.from_json(file_path=str(dataset_path / "tracer.json"))
        source_centres = al.from_json(file_path=str(dataset_path / "source_centres.json"))

        source_redshifts = {float(d.redshift) for d in dataset_list}
        lens_plane_galaxies = [
            galaxy
            for galaxy in truth_tracer.galaxies
            if float(galaxy.redshift) not in source_redshifts
        ]
        source_galaxies = []
        for i, dataset in enumerate(dataset_list):
            if model_type in _CLUSTER_SOLVED_MODEL_TYPES:
                point = al.ps.PointSolved()
            else:
                point = al.ps.Point(centre=tuple(float(v) for v in source_centres[i]))
            source_galaxies.append(
                al.Galaxy(redshift=float(dataset.redshift), **{dataset.name: point})
            )
        truth_tracer = al.Tracer(galaxies=lens_plane_galaxies + source_galaxies)

        solver_kwargs = dataset_list._profiling_solver_kwargs
        grid = al.Grid2D.uniform(
            shape_native=solver_kwargs["grid_shape"],
            pixel_scales=solver_kwargs["pixel_scale"],
        )
        solver = al.PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=solver_kwargs["pixel_scale_precision"],
            magnification_threshold=solver_kwargs["magnification_threshold"],
        )
        fit_positions_cls = cluster_point_fit_cls_for(model_type)
        truth_log_likelihood = 0.0
        for dataset in dataset_list:
            fit = al.FitPointDataset(
                dataset=dataset,
                tracer=truth_tracer,
                solver=solver,
                fit_positions_cls=fit_positions_cls,
            )
            truth_log_likelihood += float(fit.log_likelihood)
        return {
            "truth_log_likelihood": truth_log_likelihood,
            "delta_max_ll_vs_truth": float(max_log_likelihood) - truth_log_likelihood,
        }
    except Exception as exc:  # never let the anchor step kill a completed search
        print(f"  WARNING: truth-anchor step failed [cluster/{model_type}]: {exc!r}")
        return {"truth_log_likelihood": repr(exc), "delta_max_ll_vs_truth": repr(exc)}


def _posterior_stats(result: Any, uses_n_live: bool) -> dict[str, dict[str, float]] | None:
    """Per-free-parameter ``{name: {"mean": .., "std": ..}}`` from the search
    posterior (#678 phase B).

    Only meaningful for samplers with an actual posterior (``nautilus`` /
    nested sampling, gated by ``uses_n_live``); MAP optimizers (MultiStart*)
    return a single best point, so this is ``None`` for them, matching the
    existing ``n_live: null`` convention for MAP rows.

    Reads the same ``samples.parameter_lists`` / ``samples.weight_list`` the
    framework's own ``SamplesPDF.summary()`` derives its statistics from,
    keyed by ``result.model.parameter_names`` (the flat free-parameter names,
    in the same order as ``parameter_lists`` columns — the free-parameter
    analogue of the named attributes ``format_best_fit`` reads off the best-fit
    instance). Key order follows ``parameter_names`` and is therefore stable
    across runs of the same cell.
    """
    if not uses_n_live:
        return None
    try:
        samples = result.samples
        names = result.model.parameter_names
        parameters = np.asarray(samples.parameter_lists)
        weights = np.asarray(samples.weight_list)
        stats: dict[str, dict[str, float]] = {}
        for i, name in enumerate(names):
            values = parameters[:, i]
            mean = float(np.average(values, weights=weights))
            variance = float(np.average((values - mean) ** 2.0, weights=weights))
            stats[name] = {"mean": mean, "std": float(variance**0.5)}
        return stats
    except Exception as exc:  # never let the stats step kill a completed search
        print(f"  WARNING: posterior_stats step failed: {exc!r}")
        return {"_error": repr(exc)}


_DEFAULT_INSTRUMENTS: dict[str, str] = {
    "imaging": "hst",
    "interferometer": "sma",
    "point_source": "simple",
    "datacube": "sma",
    "cluster": "simple",
}


def run_search(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    default_instrument: str | None = None,
) -> None:
    """Run one (sampler, dataset_class, model_type, instrument, config) cell.

    Designed to be called from a leaf script with no extra plumbing. All
    behavioural toggles come from CLI flags parsed by ``parse_profile_cli``.
    """
    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        # Phase-5 lint smoke: confirm imports + module setup succeed
        # without paying for dataset + sampling.
        print(
            f"[smoke] searches/{sampler}/{dataset_class}/{model_type}.py: "
            f"imports + module setup OK; exiting."
        )
        return

    cli = parse_profile_cli()
    instrument = cli.instrument or default_instrument or _DEFAULT_INSTRUMENTS[dataset_class]
    config_name = cli.config_name or "default"
    use_jax = _decide_use_jax()

    uses_n_live = sampler in _SAMPLERS_WITH_N_LIVE
    n_live = n_live_for(dataset_class, model_type) if uses_n_live else None
    if sampler == "nss":
        # The NSS builder honours SEARCHES_NSS_N_LIVE (Phase 2 n_live scan);
        # keep the recorded value in lockstep with what was actually built.
        n_live = int(os.environ.get("SEARCHES_NSS_N_LIVE", n_live))

    print(
        f"\n--- searches/{sampler}/{dataset_class}/{model_type}"
        f" [{instrument}, {config_name}, use_jax={use_jax},"
        f" mp={cli.use_mixed_precision}] ---"
    )
    print(f"  n_live: {n_live if n_live is not None else 'n/a (MAP optimizer)'}")

    print("  Building dataset / model / analysis...")
    dataset, model, analysis = build_for_cell(
        dataset_class=dataset_class,
        model_type=model_type,
        instrument=instrument,
        use_jax=use_jax,
        use_mixed_precision=cli.use_mixed_precision,
    )
    print(f"  Model free parameters: {model.total_free_parameters}")

    builder = SAMPLER_BUILDERS[sampler]
    search = builder(
        sampler=sampler,
        dataset_class=dataset_class,
        model_type=model_type,
        instrument=instrument,
        config_name=config_name,
        use_jax=use_jax,
    )

    # Capture visualization wall-time across the full fit (pre-fit + every
    # update + search-side plot_results). SEARCHES_DISABLE_VIZ=1 replaces the
    # hooks with no-ops instead — see attach_viz_timer's docstring (the group
    # cell's 8-galaxy pre-fit visualization costs ~1h before step 0).
    disable_viz = os.environ.get("SEARCHES_DISABLE_VIZ") == "1"
    if disable_viz:
        print("  Visualization DISABLED (SEARCHES_DISABLE_VIZ=1) — viz_wall_s not measured.")
    viz_timer = attach_viz_timer(analysis, search, disable=disable_viz)

    print("  Running search.fit() ...")
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    total_wall_s = time.time() - t0

    # FactorGraphModel fits (datacube) return a list of per-factor Result
    # objects, all backed by the same global posterior — take the first
    # for sample stats, then summarise the per-channel best fit from the
    # global instance.
    primary_result = result[0] if isinstance(result, list) else result

    metrics = collect_metrics(
        result=primary_result,
        total_wall_s=total_wall_s,
        viz_wall_s=viz_timer.total_s,
    )

    best_instance = None
    try:
        best_instance = primary_result.max_log_likelihood_instance
        best_fit = format_best_fit(best_instance)
    except Exception as exc:
        best_fit = f"(unavailable: {exc!r})"

    recovery = _recovery_for_cell(dataset_class, instrument, best_instance)
    if recovery is not None:
        print(
            f"  Truth recovery:     overall_pass={recovery['overall_pass']} "
            f"(max ER frac err {recovery['max_einstein_radius_frac_error']:.3f}, "
            f'max centre err {recovery["max_centre_error_arcsec"]:.3f}")'
        )

    truth_anchor = _truth_anchor_for_cell(
        dataset_class, instrument, model_type, dataset, metrics.max_log_likelihood
    )
    if truth_anchor is not None:
        print(
            f"  Truth log L:        {truth_anchor['truth_log_likelihood']!r} "
            f"(delta vs max: {truth_anchor['delta_max_ll_vs_truth']!r})"
        )

    posterior_stats = _posterior_stats(primary_result, uses_n_live)

    summary = _build_summary(
        sampler=sampler,
        dataset_class=dataset_class,
        model_type=model_type,
        instrument=instrument,
        config_name=config_name,
        cli=cli,
        use_jax=use_jax,
        n_free_params=int(model.total_free_parameters),
        n_live=n_live,
        metrics=metrics,
        viz_n_calls=viz_timer.n_calls,
        best_fit=best_fit,
        recovery=recovery,
        viz_disabled=disable_viz,
        truth_anchor=truth_anchor,
        posterior_stats=posterior_stats,
    )

    _print_summary(summary, metrics)

    default_dir = (
        _WORKSPACE_ROOT / "results" / "searches" / sampler / dataset_class / model_type / instrument
    )
    json_path, png_path = resolve_output_paths(
        cli, default_dir=default_dir, default_basename=config_name
    )
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results JSON saved to: {json_path}")

    _render_png(metrics, summary, png_path)
    print(f"  Bar chart saved to:    {png_path}")


def _sampler_config_dict(
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    n_live: int | None,
    use_jax: bool,
) -> dict:
    """Return the JSON-friendly sampler config block for the metric write.

    Per-sampler shape matches the kwargs the factory in ``_samplers.py``
    actually constructs the search with — so the JSON faithfully
    records what was run, including the per-cell vmap batch cap.
    """
    if sampler == "nautilus":
        batch = vmap_batch_for_cell(dataset_class, model_type, instrument)
        return {
            "n_live": n_live,
            "n_batch": batch,
            "number_of_cores": 1,
            "use_jax_vmap": use_jax,
            "force_x1_cpu": use_jax,
            "iterations_per_update": 3 * n_live,
        }
    if sampler == "nss":
        # Mirrors the fork-era JSON shape (n_live, num_mcmc_steps, num_delete,
        # chunk_size, termination, seed, jax_native) so mainline rows diff
        # cleanly against the recorded v2026.5.21.1 fork rows.
        return {
            "n_live": n_live,
            **nss_settings(),
            "jax_native": True,
        }
    if sampler in _MULTI_START_CLASSES:
        # MAP optimizer: no n_live; records its own multi-start knobs, plus the
        # auto-convergence early-stop criterion for the ``*_autoconv`` variants.
        cfg = {
            **multi_start_settings(sampler, dataset_class, model_type),
            "number_of_cores": 1,
        }
        # Recorded for BOTH arms: the fixed-step cells explicitly disable
        # checking (leaving it unset would silently enable it, see _samplers).
        if sampler in _MULTI_START_AUTOCONV:
            cfg["convergence"] = {
                "check_for_convergence": True,
                "window": 50,
                "rtol": 1e-4,
                "atol": 1e-3,
                "min_steps": 100,
            }
        else:
            cfg["convergence"] = {"check_for_convergence": False}
        return cfg
    return {"n_live": n_live, "_note": f"unknown sampler {sampler!r}"}


def _decide_use_jax() -> bool:
    """JAX is used unless the user has explicitly disabled it.

    Mirrors the gate already in PyAutoFit (`PYAUTO_DISABLE_JAX=1`). The
    search-profiling sweep usually wants JAX on for every config except a
    pure-NumPy CPU baseline, which can be driven by setting the env var
    in the sweep config (not currently default-on).
    """
    return os.environ.get("PYAUTO_DISABLE_JAX") != "1"


def _build_summary(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    cli: Any,
    use_jax: bool,
    n_free_params: int,
    n_live: int | None,
    metrics: Any,
    viz_n_calls: int,
    best_fit: str,
    recovery: dict | None = None,
    viz_disabled: bool = False,
    truth_anchor: dict | None = None,
    posterior_stats: dict | None = None,
) -> dict:
    results_block: dict = {
        "log_evidence": metrics.log_evidence,
        "max_log_likelihood": metrics.max_log_likelihood,
        "posterior_samples": metrics.posterior_samples,
    }
    if truth_anchor is not None:
        results_block.update(truth_anchor)
    # Always present (null for MAP optimizers without a posterior) so the key
    # order + shape is stable across every cell, not just point_source ones.
    results_block["posterior_stats"] = posterior_stats

    summary = {
        "sampler": sampler,
        "dataset_class": dataset_class,
        "model": model_type,
        "instrument": instrument,
        "config_name": config_name,
        "version": al.__version__,
        "device": device_info_dict(),
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "sampler_config": _sampler_config_dict(
            sampler, dataset_class, model_type, instrument, n_live, use_jax
        ),
        "model_summary": {
            "free_parameters": n_free_params,
            "best_fit": best_fit,
        },
        "results": results_block,
        "performance": {
            "total_wall_s": metrics.total_wall_s,
            "viz_wall_s": metrics.viz_wall_s,
            "viz_n_calls": viz_n_calls,
            "viz_disabled": viz_disabled,
            "sampler_wall_s": metrics.sampler_wall_s,
            "likelihood_evals": metrics.likelihood_evals,
            "time_per_eval_ms": metrics.time_per_eval_ms,
        },
    }
    if recovery is not None:
        summary["recovery"] = recovery
    return summary


def _print_summary(summary: dict, metrics: Any) -> None:
    print("\n" + "=" * 70)
    print(
        f"SEARCH SUMMARY — {summary['sampler']}/{summary['dataset_class']}/"
        f"{summary['model']} [{summary['instrument']}, {summary['config_name']}]"
    )
    print("=" * 70)
    print(f"  Best fit:           {summary['model_summary']['best_fit']}")
    print(f"  Log evidence:       {metrics.log_evidence:.4f}")
    print(f"  Max log L:          {metrics.max_log_likelihood:.4f}")
    print(f"  Posterior samples:  {metrics.posterior_samples}")
    print(f"  Likelihood evals:   {metrics.likelihood_evals}")
    print(f"  Total wall:         {metrics.total_wall_s:.2f} s")
    print(f"  Viz wall:           {metrics.viz_wall_s:.2f} s")
    print(f"  Sampler wall:       {metrics.sampler_wall_s:.2f} s")
    print(f"  Time per eval:      {metrics.time_per_eval_ms:.3f} ms")


def _render_png(metrics: Any, summary: dict, png_path: Path) -> None:
    labels = ["total_wall (s)", "sampler_wall (s)", "viz_wall (s)", "time_per_eval (ms)"]
    values = [
        metrics.total_wall_s,
        metrics.sampler_wall_s,
        metrics.viz_wall_s,
        metrics.time_per_eval_ms,
    ]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title(
        f"{summary['sampler']} {summary['dataset_class']}/{summary['model']} "
        f"[{summary['instrument']}, {summary['config_name']}] — v{summary['version']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
