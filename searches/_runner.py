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

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import autolens as al  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from _profile_cli import (  # noqa: E402
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)
from searches._metrics import attach_viz_timer, collect_metrics  # noqa: E402
from searches._samplers import (  # noqa: E402
    _MULTI_START_AUTOCONV,
    _MULTI_START_CLASSES,
    SAMPLER_BUILDERS,
    multi_start_settings,
    n_live_for,
    vmap_batch_for_cell,
)

# Samplers that have an ``n_live`` (nested sampling). MAP optimizers such as
# ``multi_start_adam`` do not, and record ``null`` rather than a misleading value.
_SAMPLERS_WITH_N_LIVE = frozenset({"nautilus"})
from searches._recovery import load_truth, recovery_report  # noqa: E402
from searches._setup import build_for_cell, format_best_fit  # noqa: E402


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


_DEFAULT_INSTRUMENTS: dict[str, str] = {
    "imaging": "hst",
    "interferometer": "sma",
    "point_source": "simple",
    "datacube": "sma",
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
    if sampler in _MULTI_START_CLASSES:
        # MAP optimizer: no n_live; records its own multi-start knobs, plus the
        # auto-convergence early-stop criterion for the ``*_autoconv`` variants.
        cfg = {**multi_start_settings(sampler, dataset_class), "number_of_cores": 1}
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
) -> dict:
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
        "results": {
            "log_evidence": metrics.log_evidence,
            "max_log_likelihood": metrics.max_log_likelihood,
            "posterior_samples": metrics.posterior_samples,
        },
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
