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


import dataclasses
import json
import os
import sys
import time
from contextlib import nullcontext
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

from searches import _targets  # noqa: E402
from searches._metrics import attach_viz_timer, collect_metrics  # noqa: E402
from searches._per_lane import capture_search_internal, per_lane_block  # noqa: E402
from searches._samplers import (  # noqa: E402
    _MULTI_START_AUTOCONV,
    _MULTI_START_CLASSES,
    SAMPLER_BUILDERS,
    multi_start_settings,
    n_live_for,
    nautilus_seed,
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
    _LOG_DET_METHOD_DATASET_CLASSES,
    _PIX_MODEL_TYPES,
    _adapt_images_for,
    apply_diagnostic_prior_overrides,
    build_for_cell,
    cluster_point_fit_cls_for,
    format_best_fit,
    point_source_fit_cls_for,
    positions_arm_tag,
    positions_enabled,
    positions_settings,
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
    *,
    use_mixed_precision: bool = False,
    log_det_method: str | None = None,
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

    Returns ``None`` for non point_source / cluster / imaging cells. Never
    raises: on any exception the failure is printed loudly and the exception
    string is stored in place of the numeric fields instead of crashing the
    run.

    Extended for cluster cells (#678 phase B chunk 2) — see
    ``_cluster_truth_anchor`` below — and for imaging cells (W4 / issue #161,
    Phase 1) — see ``_imaging_truth_anchor``; dispatched here so every
    dataset_class shares one entry point from ``run_search``.
    ``use_mixed_precision`` / ``log_det_method`` are only consumed by the
    imaging branch (its ``al.Settings`` must match the search's own).
    """
    if dataset_class == "cluster":
        return _cluster_truth_anchor(instrument, model_type, dataset, max_log_likelihood)
    if dataset_class == "imaging":
        return _imaging_truth_anchor(
            instrument,
            model_type,
            dataset,
            max_log_likelihood,
            use_mixed_precision=use_mixed_precision,
            log_det_method=log_det_method,
        )
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
            "bar_source": "truth_tracer",
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
            "bar_source": "truth_tracer",
        }
    except Exception as exc:  # never let the anchor step kill a completed search
        print(f"  WARNING: truth-anchor step failed [cluster/{model_type}]: {exc!r}")
        return {"truth_log_likelihood": repr(exc), "delta_max_ll_vs_truth": repr(exc)}


def _imaging_truth_anchor(
    instrument: str,
    model_type: str,
    dataset: Any,
    max_log_likelihood: float,
    *,
    use_mixed_precision: bool = False,
    log_det_method: str | None = None,
) -> dict:
    """Truth-anchored log likelihood for imaging cells (W4 / issue #161, Phase 1).

    Extends the truth-anchor mechanism (previously point_source/cluster only)
    to imaging: builds ``al.FitImaging`` directly from the simulator's own
    truth tracer (``dataset/imaging/<instrument>/tracer.json``) using the
    SAME adapt images + ``al.Settings`` the search's own analysis was built
    with (``use_border_relocator`` gated on ``model_type in _PIX_MODEL_TYPES``,
    matching ``_setup._build_analysis`` exactly), so ``delta_max_ll_vs_truth``
    is directly comparable to the search's own ``max_log_likelihood``.

    ``bar_source`` is always ``"truth_tracer"`` here — there is no completed
    reference run to anchor against instead (the other value the schema
    allows, ``"reference_run"``, is reserved for a future baseline-anchored
    mode). **The Δ<=2 nats tolerance (``Tolerances.delta_max_ll_nats``,
    ``searches._targets``) applies to the ``mge`` target only**: for
    pixelized targets (``pixelization``/``delaunay*``/``knn``/
    ``slam_source_pix*``) the truth tracer's PSF-convolved analytic image and
    the mesh reconstruction's best fit are not expected to agree to within a
    couple of nats even at the true lens parameters — the inversion smooths
    and regularizes the source differently from the noiseless analytic
    truth — so a large delta on those targets reflects that structural
    mismatch, not a search failure.

    Never raises: on any exception the failure is printed loudly and the
    exception string is stored in place of the numeric fields instead of
    crashing a completed run.
    """
    try:
        dataset_path = _WORKSPACE_ROOT / "dataset" / "imaging" / instrument
        tracer = al.from_json(file_path=dataset_path / "tracer.json")
        adapt_images = _adapt_images_for(
            "imaging", model_type, dataset_path=dataset_path, dataset=dataset
        )
        settings = al.Settings(
            use_border_relocator=model_type in _PIX_MODEL_TYPES,
            use_mixed_precision=use_mixed_precision,
            log_det_method=log_det_method,
        )
        fit = al.FitImaging(
            dataset=dataset, tracer=tracer, adapt_images=adapt_images, settings=settings
        )
        truth_log_likelihood = float(fit.log_likelihood)
        return {
            "truth_log_likelihood": truth_log_likelihood,
            "delta_max_ll_vs_truth": float(max_log_likelihood) - truth_log_likelihood,
            "bar_source": "truth_tracer",
        }
    except Exception as exc:  # never let the anchor step kill a completed search
        print(f"  WARNING: truth-anchor step failed [imaging/{model_type}]: {exc!r}")
        return {
            "truth_log_likelihood": repr(exc),
            "delta_max_ll_vs_truth": repr(exc),
            "bar_source": "truth_tracer",
        }


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


def _penalty_at_best(
    analysis: Any, best_instance: Any, max_log_likelihood: float | None
) -> dict | None:
    """Decompose the best point's recorded log-likelihood into fit and penalty.

    Phase 4 / issue #182. Every ``PositionsLH``-attached analysis returns
    ``fit.figure_of_merit - log_likelihood_penalty`` from its
    ``log_likelihood_function`` (``autolens/imaging/model/analysis.py``), and
    PyAutoFit records THAT as the sample's ``log_likelihood``. So a
    positions-on row's ``results.max_log_likelihood`` is the PENALISED value,
    and how much of it is penalty was, until now, unrecoverable from the
    artifact — the question every Phase-4 arm's write-up had to answer with
    "not recorded".

    This costs **no likelihood evaluation**: ``AnalysisDataset.
    log_likelihood_penalty_from`` builds the tracer and traces the (four)
    truth positions to the source plane. No imaging fit, no inversion — it is
    the penalty term alone, evaluated at the best instance the search already
    returned.

    ``None`` when positions are off, when the analysis carries no positions
    list, or when no best instance was recoverable — never a fabricated 0.0,
    which would be indistinguishable from "positions on and the model traced
    inside the threshold".
    """
    positions_list = getattr(analysis, "positions_likelihood_list", None)
    if best_instance is None or not positions_list:
        return None
    try:
        penalty = float(np.asarray(analysis.log_likelihood_penalty_from(instance=best_instance)))
    except Exception as exc:  # never let the readout kill a completed search
        print(f"  WARNING: penalty_at_best step failed: {exc!r}")
        return {"error": repr(exc)}
    penalised = None if max_log_likelihood is None else float(max_log_likelihood)
    return {
        "positions_penalty": penalty,
        "log_likelihood_penalised": penalised,
        "log_likelihood_unpenalised": None if penalised is None else penalised + penalty,
        # with-penalty MINUS without-penalty, i.e. what the penalty cost the
        # objective at this point. Zero means the best model traced inside the
        # threshold, so the penalty was inert AT THE BEST POINT — it says
        # nothing about the rest of the trajectory.
        "delta_log_likelihood": -penalty,
        "source": (
            "analysis.log_likelihood_penalty_from(max_log_likelihood_instance) — "
            "the penalty term alone (tracer + truth-position trace), not a "
            "re-evaluated likelihood"
        ),
    }


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

    # Opt-in positions-penalty arm (Phase 4 Stage 1, issue #159). Off unless
    # SEARCHES_POSITIONS=on. Appended to config_name (not just unique_tag) so
    # the JSON/PNG output filename itself makes a positions-on run impossible
    # to mistake for the positions-off run of the "same" config_name — a
    # positions-on run is a different objective (figure_of_merit minus a
    # penalty term), so it must never silently overwrite the positions-off
    # artifact on disk.
    pos_tag = positions_arm_tag()
    if pos_tag is not None:
        config_name = f"{config_name}_{pos_tag}"
        print("  " + "!" * 66)
        print(f"  !! POSITIONS ARM (target_class 3): SEARCHES_POSITIONS=on -> {pos_tag!r}")
        print(f"  !!   config_name -> {config_name!r}")
        print("  !! Idealised: truth-derived positions, not re-solved from a completed search.")
        print("  " + "!" * 66)

    uses_n_live = sampler in _SAMPLERS_WITH_N_LIVE
    n_live = n_live_for(dataset_class, model_type) if uses_n_live else None
    if sampler == "nss":
        # The NSS builder honours SEARCHES_NSS_N_LIVE (Phase 2 n_live scan);
        # keep the recorded value in lockstep with what was actually built.
        n_live = int(os.environ.get("SEARCHES_NSS_N_LIVE", n_live))
    if sampler == "nautilus":
        # The Nautilus builder honours SEARCHES_NAUTILUS_N_LIVE (W4 / issue
        # #161 reference-baseline submits); keep the recorded value in
        # lockstep with what was actually built.
        n_live = int(os.environ.get("SEARCHES_NAUTILUS_N_LIVE", n_live))

    print(
        f"\n--- searches/{sampler}/{dataset_class}/{model_type}"
        f" [{instrument}, {config_name}, use_jax={use_jax},"
        f" mp={cli.use_mixed_precision}] ---"
    )
    print(f"  n_live: {n_live if n_live is not None else 'n/a (MAP optimizer)'}")

    print("  Building dataset / model / analysis...")
    log_det_method = resolve_log_det_method(
        sampler=sampler, dataset_class=dataset_class, model_type=model_type, use_jax=use_jax
    )
    if log_det_method is not None:
        print(
            f"  log_det_method: {log_det_method} (W8 GPU gradient-cell default / SEARCHES_LOG_DET_METHOD)"
        )
    dataset, model, analysis = build_for_cell(
        dataset_class=dataset_class,
        model_type=model_type,
        instrument=instrument,
        use_jax=use_jax,
        use_mixed_precision=cli.use_mixed_precision,
        log_det_method=log_det_method,
    )
    print(f"  Model free parameters: {model.total_free_parameters}")

    # Opt-in, target-CHANGING prior override for the Phase-3 diagnostic arm.
    # Off unless SEARCHES_DIAGNOSTIC_THETA_E_PRIOR is set, and loud when on:
    # the one failure mode that must not exist is a target change nobody
    # noticed in the artifact.
    model, target_override = apply_diagnostic_prior_overrides(model)
    if target_override is not None:
        print("  " + "!" * 66)
        print(f"  !! DIAGNOSTIC ARM (target_class 3): {target_override['parameter']}")
        print(f"  !!   {target_override['prior_before']} -> {target_override['prior_after']}")
        print("  !! NOT comparable to the campaign's other arms.")
        print("  " + "!" * 66)

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

    # Per-lane preservation (PROGRAMME.md §3): the MultiStart* searches write
    # every lane's final position and — since PyAutoFit PR#1515 — every lane's
    # own best into ``search_internal``, which is DELETED on successful
    # completion. It has to be captured as it is written, so the fit runs inside
    # the capture for those samplers and untouched (``nullcontext``) for the
    # rest, keeping the nested-sampler paths bit-identical.
    is_multi_start = sampler in _MULTI_START_CLASSES
    capture = capture_search_internal() if is_multi_start else nullcontext({})

    print("  Running search.fit() ...")
    with capture as captured:
        t0 = time.time()
        result = search.fit(model=model, analysis=analysis)
        total_wall_s = time.time() - t0

    # FactorGraphModel fits (datacube) return a list of per-factor Result
    # objects, all backed by the same global posterior — take the first
    # for sample stats, then summarise the per-channel best fit from the
    # global instance.
    primary_result = result[0] if isinstance(result, list) else result

    # W4 / issue #161 (Phase 1): MultiStart's likelihood_evals correction needs
    # total_steps from the captured search_internal, read here (before
    # per_lane_block, which also reads `captured`) so both derive from the
    # same capture.
    _captured_total_steps = captured.get("total_steps") if is_multi_start else None
    _multi_start_total_steps = (
        int(_captured_total_steps) if _captured_total_steps is not None else None
    )
    _multi_start_n_starts = int(search.n_starts) if is_multi_start else None
    metrics = collect_metrics(
        result=primary_result,
        total_wall_s=total_wall_s,
        viz_wall_s=viz_timer.total_s,
        is_multi_start=is_multi_start,
        n_starts=_multi_start_n_starts,
        multi_start_total_steps=_multi_start_total_steps,
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
        dataset_class,
        instrument,
        model_type,
        dataset,
        metrics.max_log_likelihood,
        use_mixed_precision=cli.use_mixed_precision,
        log_det_method=log_det_method,
    )
    if truth_anchor is not None:
        print(
            f"  Truth log L:        {truth_anchor['truth_log_likelihood']!r} "
            f"(delta vs max: {truth_anchor['delta_max_ll_vs_truth']!r})"
        )

    posterior_stats = _posterior_stats(primary_result, uses_n_live)

    penalty_at_best = _penalty_at_best(analysis, best_instance, metrics.max_log_likelihood)
    if penalty_at_best is not None and "positions_penalty" in penalty_at_best:
        print(
            f"  Positions penalty:  {penalty_at_best['positions_penalty']!r} at the best point "
            f"(unpenalised logL {penalty_at_best['log_likelihood_unpenalised']!r})"
        )

    diagnostics = None
    if is_multi_start:
        diagnostics = per_lane_block(captured=captured, model=model, n_starts=int(search.n_starts))
        print(
            f"  Per-lane:           {diagnostics['n_lanes_recorded']} lanes recorded, "
            f"stop_reason={diagnostics['counters']['stop_reason']!r}, "
            f"total_steps={diagnostics['counters']['total_steps']!r}"
        )
        if not diagnostics["valid"]:
            # Recorded on the artifact AND shouted, but never raised: a
            # completed multi-hour GPU fit is not thrown away because its
            # diagnostics block is suspect.
            print("  !! PER-LANE BLOCK INVALID — this run cannot be interpreted:")
            for reason in diagnostics["invalid_reasons"]:
                print(f"     - {reason}")

    summary = _build_summary(
        sampler=sampler,
        dataset_class=dataset_class,
        model_type=model_type,
        instrument=instrument,
        config_name=config_name,
        cli=cli,
        use_jax=use_jax,
        model=model,
        n_live=n_live,
        metrics=metrics,
        viz_n_calls=viz_timer.n_calls,
        best_fit=best_fit,
        recovery=recovery,
        viz_disabled=disable_viz,
        truth_anchor=truth_anchor,
        posterior_stats=posterior_stats,
        diagnostics=diagnostics,
        target_override=target_override,
        penalty_at_best=penalty_at_best,
    )

    _print_summary(summary, metrics)

    default_dir = (
        _WORKSPACE_ROOT / "results" / "searches" / sampler / dataset_class / model_type / instrument
    )
    # resolve_output_paths derives its basename from cli.config_name (the RAW
    # CLI flag), not the positions-tagged `config_name` local above -- so
    # without this, a positions-on and positions-off run of the same
    # --config-name silently overwrite the SAME results/searches/.../*.json
    # (even though their output/searches/... PyAutoFit directories are
    # correctly disambiguated, see positions_arm_tag's docstring). Passing a
    # cli with the tagged config_name fixes the results-JSON path too.
    json_path, png_path = resolve_output_paths(
        dataclasses.replace(cli, config_name=config_name),
        default_dir=default_dir,
        default_basename=config_name,
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
            "seed": nautilus_seed(),
            "positions": positions_settings(),
        }
    if sampler == "nss":
        # Mirrors the fork-era JSON shape (n_live, num_mcmc_steps, num_delete,
        # chunk_size, termination, seed, jax_native) so mainline rows diff
        # cleanly against the recorded v2026.5.21.1 fork rows.
        return {
            "n_live": n_live,
            **nss_settings(),
            "jax_native": True,
            "positions": positions_settings(),
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


# W8 (autolens_profiling#165, DECISIONS.md 2026-08-24 CP-4 human call): slogdet
# is the default evidence log-det for GRADIENT-WORK cells on GPU tiers only.
# CP-4 measured zero regressions and 64-73 % NaN-wall rescue at 1.03x GPU cost,
# but 3.7x on CPU (fails the 2x ceiling) — so CPU keeps cholesky. Nested
# samplers (nautilus / nss) keep cholesky everywhere so the truth bars do not
# move. ``SEARCHES_LOG_DET_METHOD`` overrides everything (A/B rows). The
# PyAutoArray library default is untouched (W9, #166).
_GRADIENT_SAMPLERS: frozenset[str] = frozenset({*_MULTI_START_CLASSES, "nuts"})


def _jax_backend_is_gpu() -> bool:
    try:
        import jax

        return jax.default_backend() in ("gpu", "cuda", "rocm")
    except Exception:
        return False


def resolve_log_det_method(
    *, sampler: str, dataset_class: str, model_type: str, use_jax: bool
) -> str | None:
    """The ``log_det_method`` a cell runs with, or ``None`` for the packaged default.

    Precedence: ``SEARCHES_LOG_DET_METHOD`` env (any value the library accepts)
    > W8 default (``"slogdet"`` iff gradient sampler AND pixelized model AND a
    dataset class whose analysis is built through ``al.Settings`` AND JAX on a
    GPU backend) > ``None``.
    """
    override = os.environ.get("SEARCHES_LOG_DET_METHOD")
    if override:
        return override.strip().lower()
    if (
        use_jax
        and sampler in _GRADIENT_SAMPLERS
        and model_type in _PIX_MODEL_TYPES
        and dataset_class in _LOG_DET_METHOD_DATASET_CLASSES
        and _jax_backend_is_gpu()
    ):
        return "slogdet"
    return None


def _decide_use_jax() -> bool:
    """JAX is used unless the user has explicitly disabled it.

    Mirrors the gate already in PyAutoFit (`PYAUTO_DISABLE_JAX=1`). The
    search-profiling sweep usually wants JAX on for every config except a
    pure-NumPy CPU baseline, which can be driven by setting the env var
    in the sweep config (not currently default-on).
    """
    return os.environ.get("PYAUTO_DISABLE_JAX") != "1"


def _target_for_cell(
    dataset_class: str, model_type: str, instrument: str, use_mixed_precision: bool
) -> Any | None:
    """Resolve the ``searches._targets.Target`` a cell's arm corresponds to, if any.

    W4 / issue #161 (Phase 1): the ``TARGETS`` registry covers
    ``dataset_class="imaging"``, ``instrument="hst"`` only today (every other
    dataset_class / instrument returns ``None`` here, not a mismatched
    lookup) — a positions-on/off + fp64/mp cell maps to a registry key via
    the same ``<model_type>[_pos]_<precision>`` convention
    ``searches._targets._target_key`` uses, so a leaf script needs no changes
    to pick up the schema-v2 ``target`` block automatically.
    """
    if dataset_class != "imaging" or instrument != "hst":
        return None
    key = _targets._target_key(
        model_type, "on" if positions_enabled() else "off", "mp" if use_mixed_precision else "fp64"
    )
    target = _targets.TARGETS.get(key)
    if target is not None and target.instrument != instrument:
        return None
    return target


def _build_summary(
    *,
    sampler: str,
    dataset_class: str,
    model_type: str,
    instrument: str,
    config_name: str,
    cli: Any,
    use_jax: bool,
    model: Any,
    n_live: int | None,
    metrics: Any,
    viz_n_calls: int,
    best_fit: str,
    recovery: dict | None = None,
    viz_disabled: bool = False,
    truth_anchor: dict | None = None,
    posterior_stats: dict | None = None,
    diagnostics: dict | None = None,
    target_override: dict | None = None,
    penalty_at_best: dict | None = None,
) -> dict:
    n_free_params = int(model.total_free_parameters)
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
    # Always present (null when positions are off, or when the best instance
    # was unavailable) — same absent-vs-null discipline. See
    # ``_penalty_at_best`` for what the block decomposes.
    results_block["penalty_at_best"] = penalty_at_best

    sampler_config = _sampler_config_dict(
        sampler, dataset_class, model_type, instrument, n_live, use_jax
    )

    summary = {
        "sampler": sampler,
        "dataset_class": dataset_class,
        "model": model_type,
        "instrument": instrument,
        "config_name": config_name,
        "version": al.__version__,
        "device": device_info_dict(),
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "sampler_config": sampler_config,
        # Always present (Phase 4 Stage 1, issue #159): {"enabled": False} when
        # SEARCHES_POSITIONS is off, so a positions-on and positions-off run
        # of the "same" cell/config_name are never ambiguous in the artifact.
        "positions": positions_settings(),
        # W8: the evidence log-det the cell ran with (None = packaged default).
        "log_det_method": resolve_log_det_method(
            sampler=sampler, dataset_class=dataset_class, model_type=model_type, use_jax=use_jax
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
            # W4 / issue #161 (Phase 1) schema-v2 additions — additive.
            "stored_samples": metrics.stored_samples,
            "gradient_evals": metrics.gradient_evals,
            "kish_ess": metrics.kish_ess,
            "evals_per_ess": metrics.evals_per_ess,
            "ess_per_min": metrics.ess_per_min,
        },
    }
    if recovery is not None:
        summary["recovery"] = recovery
    # Both keys are added ONLY when populated, so every non-MultiStart cell's
    # JSON keeps the exact shape its recorded rows have.
    if target_override is not None:
        summary["target_override"] = target_override
    if diagnostics is not None:
        summary["diagnostics"] = diagnostics

    # ---- Schema v2 (W4 / issue #161, Phase 1) — added BESIDE every v1 key
    # above, never replacing one, so a v1-shaped reader (build_readme.py's
    # dashboard scan) keeps working unchanged.
    summary["schema_version"] = 2
    target = _target_for_cell(dataset_class, model_type, instrument, bool(cli.use_mixed_precision))
    if target is not None:
        dataset_path = _WORKSPACE_ROOT / "dataset" / dataset_class / instrument
        # The RESOLVED positions arm (issue #182), not the registry defaults:
        # `Target.positions` is only "off"/"on", so without this the three
        # Phase-4 arms (t0.3/f1e8, t0.3/f1e5, tauto0.2/f1e8) — three different
        # objectives with three different output directories — all hashed to
        # one target_id. It is the SAME dict recorded above as
        # ``summary["positions"]``, so any later reader can re-derive this id
        # from the artifact alone (see _targets._positions_block).
        summary["target"] = _targets.target_block(target, model, dataset_path, summary["positions"])
    else:
        # Not covered by the Phase 1 TARGETS registry (imaging/hst only
        # today) — shape stays stable (the key is always present) with an
        # honest null id rather than a fabricated one.
        summary["target"] = {
            "target_id": None,
            "cell": f"{dataset_class}/{model_type}/{instrument}",
            "model_dim": n_free_params,
            "priors_ref": None,
            "note": "not covered by the Phase 1 TARGETS registry (imaging/hst only today)",
        }
    summary["algorithm"] = {
        "name": sampler,
        "config_id": config_name,
        "settings": sampler_config,
        "seed": sampler_config.get("seed"),
    }
    summary["hardware"] = {
        "tier": config_name,
        "precision": "mp" if bool(cli.use_mixed_precision) else "fp64",
        "device": device_info_dict(),
    }
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
