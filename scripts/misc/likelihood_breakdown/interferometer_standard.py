"""Shared drivers for the interferometer standard-size cells (likelihood-lifecycle 2b, #469).

Every setup of an interferometer pixelized likelihood x instrument evaluates one likelihood: the
standard-size production cell of ruling R3 (``_production_config.standard_interferometer_cell``:
Delaunay Hilbert 1250 + 30 zeroed edge points with a free ``AdaptSplit``, or
``RectangularBilinearAdaptImage`` 35 x 35 with a free ``Adapt``; no lens light; the truth
lensed-source adapt image, uncapped) at the preset's seeded iid draw 0. This module holds the
code the cell scripts call under ``--standard-size``, so the JAX runtime, the numba runtime and
the parity reference task cannot drift:

- :func:`run_jax` -- ``scripts/interferometer/{rectangular,delaunay}/likelihood_runtime.py
  --standard-size [--sparse]``: the JAX full pipeline (``AnalysisInterferometer(use_jax=True)``)
  on the sparse W~ operator (``--sparse``, setup ``jax_*_sparse_fp64_wtilde``) or the dense
  NUFFT route (setup ``jax_*_dense_fp64_nufft``; planned at sma only, ruling R1(a)): lower /
  compile / first call / steady state, the steady median, and the vmap per call the catalogue
  reads as the headline;
- :func:`run_numba` -- ``.../likelihood_runtime_numba.py --standard-size``: the NumPy / numba
  likelihood over the preset's iid stream, headline = the packaged cost-model route (R4), the
  forced numba ``direct_conv`` and NumPy FFT routes as sub-rows;
- :func:`reference_task` / :func:`finalise_parity` -- ``scripts/interferometer/pixelized/
  parity_reference.py``: the visibility-chunked eager NumPy dense reference (R6), cached per
  (variant, instrument), and the pass that rewrites each cell's ``parity.json`` against it.

Every row records ``resources``, renders ``fit.png`` and writes ``parity.json`` (tolerance
1e-8 x |reference| in fp64, R7), and carries a ``parity_candidate`` block (the value its witness
compares and the instance it was taken at) so the reference task can finalise a witness written
before the reference existed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

#: The numba runtime's arms (ruling R4): the packaged cost-model route is the headline, the two
#: routes it chooses between are forced as sub-rows.
NUMBA_ARMS = {
    "default": {},
    "numba": {"interferometer_route": "numba"},
    "numpy_fft": {"interferometer_route": "fft"},
}
#: The inversion class each forced arm must build.
NUMBA_ARM_CLASSES = {
    "numba": "InversionInterferometerSparseNumba",
    "numpy_fft": "InversionInterferometerSparse",
}
#: Above this visibility count a JAX CPU leg uses 3 steady repeats instead of 10.
LARGE_VIS = 200_000
#: The quantity every interferometer pixelized witness compares (the figure of merit).
PARITY_QUANTITY = (
    "log_likelihood_function (the cell's figure of merit, the log evidence) at the standard "
    "cell's iid draw 0, vs the visibility-chunked eager NumPy dense reference"
)


def _variant_model_name(variant: str) -> str:
    """The historic ``model`` field of the interferometer rows (rectangular rows say pixelization)."""
    return "pixelization" if variant == "rectangular" else variant


def _require_standard_size(cli, script: str) -> None:
    if not cli.standard_size:
        raise SystemExit(
            f"{script}: this driver runs only the standard-size cell; pass --standard-size"
        )


def load_standard_inputs(variant: str, instrument: str, profiling_root: Path, timer) -> dict:
    """The masked dataset, its adapt image and the standard cell, built the one shared way."""
    from _adapt_image_util import adapt_image_for_dataset

    from . import interferometer_pixelized as shared

    ds = shared.load_dataset(instrument, None, profiling_root, timer)
    with timer.section("adapt_image_build"):
        adapt_image = adapt_image_for_dataset(dataset_path=ds.dataset_path, dataset=ds.dataset)
    return {"ds": ds, "adapt_image": adapt_image}


def parity_candidate(
    *, log_likelihood, cell: dict, use_mixed_precision: bool, variant: str, instrument: str
) -> dict:
    """The ``parity_candidate`` block a row carries (what :func:`finalise_parity` reads)."""
    return {
        "log_likelihood": None if log_likelihood is None else float(log_likelihood),
        "quantity": PARITY_QUANTITY,
        "variant": variant,
        "instrument": instrument,
        "instance_unit_vector": cell["unit_vector"],
        "source_pixels": int(cell["n_source_pixels"]),
        "source_pixels_requested": int(cell["source_pixels_requested"]),
        "use_mixed_precision": bool(use_mixed_precision),
    }


def _write_parity(
    cell_dir: Path,
    *,
    candidate: dict,
    ds,
    profiling_root: Path,
    autolens_version: str,
    compute_reference=None,
):
    import _production_config as pc

    return pc.write_interferometer_parity(
        cell_dir,
        log_likelihood=candidate["log_likelihood"],
        variant=candidate["variant"],
        instrument=candidate["instrument"],
        dataset_path=ds.dataset_path,
        root=profiling_root,
        unit_vector=candidate["instance_unit_vector"],
        use_mixed_precision=candidate["use_mixed_precision"],
        autolens_version=autolens_version,
        source_pixels=candidate["source_pixels"],
        source_pixels_requested=candidate["source_pixels_requested"],
        quantity=candidate["quantity"],
        compute_reference=compute_reference,
    )


def _in_process_reference(ds, cell: dict, adapt_images, *, memo: str):
    """A ``compute_reference`` callable for sma (the only instrument cheap enough in-process)."""
    import _production_config as pc

    if ds.instrument != "sma":
        return None
    fp64_settings, _ = pc.analysis_settings(cell["preset"], memo)

    def compute():
        return pc.interferometer_dense_reference(
            ds.dataset, adapt_images, cell["instance"], fp64_settings
        )

    return compute


def _configuration(ds, cell: dict, **extra) -> dict:
    return {
        "pixel_scale_arcsec": ds.pixel_scale,
        "mask_radius_arcsec": ds.mask_radius,
        "real_space_shape": list(ds.real_space_shape),
        "image_pixels_masked": int(ds.n_image_pixels),
        "visibilities": int(ds.n_vis),
        "source_pixels": int(cell["n_source_pixels"]),
        # What was asked for (Delaunay 1250 Hilbert points, rectangular 1225); built is above.
        "source_pixels_requested": int(cell["source_pixels_requested"]),
        "standard_size": True,
        "production_preset": cell["record"],
        "regularization": cell["reg_provenance"],
        "memo_provenance": cell["memo_provenance"],
        "transformer_chunk_size": ds.transformer_chunk,
        "transformer_chunk_size_preset": ds.preset_chunk,
        "lens_light": None,
        **extra,
    }


def _render_fit(fit_factory, cell_dir: Path, cli) -> None:
    """Render ``fit.png`` from an eager NumPy fit; a fit that fails to build costs only the figure."""
    from _profile_cli import render_fit_png

    try:
        fit = fit_factory()
    except Exception as exc:  # noqa: BLE001 — a figure must not cost the row
        print(f"  WARNING: [fit.png] eager fit failed: {type(exc).__name__}: {exc}")
        return
    render_fit_png(fit, cell_dir, dataset_type="interferometer", cli=cli)


# --------------------------------------------------------------------------- JAX runtime


def run_jax(variant: str, cli, *, script_file: str, profiling_root: Path) -> Path:
    """The JAX runtime row of the standard-size interferometer cell (sparse W~ or dense NUFFT)."""
    import autolens as al
    import jax
    import jax.numpy as jnp
    import numpy as np
    from autofit.jax import register_model
    from vram import resolve_vmap_batch

    import _production_config as pc
    from _profile_cli import (
        device_info_dict,
        legacy_landing_dir,
        resolve_output_paths,
        resources_dict,
    )

    from . import timing

    _require_standard_size(cli, script_file)
    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True (fp32 truncation).")
    instrument = cli.instrument or "sma"
    sparse = bool(cli.use_sparse_operator)
    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    load_start = list(os.getloadavg())
    print(
        f"--- {script_file} [{variant} / {instrument}] standard size, {'sparse W~' if sparse else 'dense NUFFT'} ---"
    )

    inputs = load_standard_inputs(variant, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    dataset = ds.dataset
    operator_build_s = None
    if sparse:
        with timer.section("apply_sparse_operator"):
            dataset = dataset.apply_sparse_operator(
                method="nufft",
                batch_size=cli.sparse_batch_size,
                nufft_chunk_size=ds.transformer_chunk,
                use_jax=True,
            )
        operator_build_s = timer.records[-1][1]

    cell = pc.standard_interferometer_cell(
        variant,
        instrument,
        dataset=ds.dataset,
        adapt_image=adapt_image,
        mask_radius=ds.mask_radius,
        memo=cli.memo,
        use_mixed_precision=cli.use_mixed_precision,
    )
    model, instance, settings = cell["model"], cell["instance"], cell["settings"]
    print(
        f"  preset {cell['preset'].name}: {cell['n_source_pixels']} source pixels, "
        f"{model.total_free_parameters} free parameters, {ds.n_vis} visibilities"
    )
    register_model(model)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
    adapt_images = pc.adapt_images_for(instance, cell["image_plane_mesh_grid"], adapt_image)
    analysis = al.AnalysisInterferometer(
        dataset=dataset, adapt_images=adapt_images, settings=settings, use_jax=True
    )

    # The route the traced pipeline takes, read off a shape-only trace (no compute).
    trace_info: dict[str, Any] = {}

    def traced_fit(params):
        fit = al.FitInterferometer(
            dataset=dataset,
            tracer=al.Tracer(galaxies=list(params.galaxies), fields=[params.fields]),
            # Keyed on the galaxy object, so rebuilt for the traced instance.
            adapt_images=pc.adapt_images_for(params, cell["image_plane_mesh_grid"], adapt_image),
            settings=settings,
            xp=jnp,
        )
        trace_info["inversion_class"] = type(fit.inversion).__name__
        return fit.figure_of_merit

    jax.eval_shape(traced_fit, params_tree)
    expected = "InversionInterferometerSparse" if sparse else "InversionInterferometerMapping"
    print(f"  inversion class (traced): {trace_info.get('inversion_class')}")
    if trace_info.get("inversion_class") != expected:
        raise SystemExit(f"expected {expected} on this setup, traced {trace_info}")

    on_cpu = jax.default_backend() == "cpu"
    n_repeats = 3 if (on_cpu and ds.n_vis > LARGE_VIS) else 10

    def full_pipeline(params):
        return analysis.log_likelihood_function(instance=params)

    print("\n--- Full pipeline: jax.jit(AnalysisInterferometer.log_likelihood_function) ---")
    compiled, full_result = timing.jit_profile(
        full_pipeline,
        "full_pipeline",
        params_tree,
        n_repeats=n_repeats,
        timer=timer,
        jit_records=jit_records,
    )
    full_per_call = jit_records["full_pipeline"]["steady_per_call_s"]
    full_median = timing.headline_steady_median(
        compiled,
        params_tree,
        block_mean_s=full_per_call,
        block_mean_is=timing.single_jit_block_is(n_repeats),
    )
    log_evidence_jit = float(full_result)
    print(f"  full log evidence = {log_evidence_jit!r}")
    del compiled
    jax.clear_caches()
    gc.collect()

    # vmap: the batched cost a sampler pays per point (the catalogue's JAX per-call headline).
    batch_resolved, batch_source = resolve_vmap_batch(
        "interferometer",
        _variant_model_name(variant),
        instrument,
        output_dir=cli.output_dir
        or legacy_landing_dir(
            profiling_root / "results" / "runtime" / "interferometer" / _variant_model_name(variant)
        ),
        path="sparse" if sparse else "dense",
        backend=jax.default_backend(),
    )
    batch_size = batch_resolved or 3
    print(f"\n--- vmap (batch {batch_size}, source: {batch_source}) ---")
    vmap: Any = None
    vmap_attempts = []
    for nb in sorted({batch_size, max(1, batch_size // 2), 1}, reverse=True):
        batched = jax.tree_util.tree_map(
            lambda leaf, n=nb: jnp.broadcast_to(leaf, (n, *leaf.shape)), params_tree
        )
        try:
            per_call = timing.vmap_profile(
                full_pipeline,
                "full_pipeline",
                batched,
                nb,
                n_repeats=n_repeats,
                timer=timer,
                jit_records=jit_records,
            )
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "RESOURCE_EXHAUSTED" not in text and "out of memory" not in text.lower():
                raise
            vmap_attempts.append(
                {"batch": nb, "status": "oom", "error": text.splitlines()[0][:300]}
            )
            print(f"  vmap batch {nb}: out of memory, trying smaller")
            jax.clear_caches()
            gc.collect()
            continue
        vmap = {
            "batch_size": nb,
            "batch_size_resolved": batch_size,
            "batch_size_source": batch_source,
            "batch_time": per_call * nb,
            "per_call": per_call,
            "speedup_vs_single_jit": round(full_per_call / per_call, 2),
            "instances": "draw 0 broadcast to the batch (the runtime cells' convention)",
            "attempts": vmap_attempts,
        }
        break
    jax.clear_caches()
    gc.collect()

    al_version = al.__version__
    candidate = parity_candidate(
        log_likelihood=log_evidence_jit,
        cell=cell,
        use_mixed_precision=cli.use_mixed_precision,
        variant=variant,
        instrument=instrument,
    )
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("jax"),
        "instrument": instrument,
        "model": _variant_model_name(variant),
        "transformer": "TransformerNUFFT",
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "configuration": _configuration(
            ds,
            cell,
            inversion_path="sparse" if sparse else "dense",
            inversion_class=trace_info["inversion_class"],
            sparse_operator_method="nufft" if sparse else None,
            sparse_batch_size=int(cli.sparse_batch_size) if sparse else None,
            operator_build_s=operator_build_s,
            n_repeats=n_repeats,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        "log_evidence_jit": log_evidence_jit,
        "full_pipeline_single_jit": full_per_call,
        **full_median,
        "vmap": vmap
        if vmap is not None
        else {"status": "oom", "attempts": vmap_attempts, "note": "every vmap batch OOM'd"},
        "jit_phases": jit_records,
        "parity_candidate": candidate,
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root
        / "results"
        / "runtime"
        / "interferometer"
        / _variant_model_name(variant),
        default_basename=f"{_variant_model_name(variant)}_likelihood_summary_{instrument}_v{al_version}",
        hints={**summary, "transformer": "TransformerNUFFT"},
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Row written: {dict_path}")

    # After every timed region: the eager NumPy fit (log likelihood + fit.png) and the witness.
    def eager_fit():
        return al.FitInterferometer(
            dataset=dataset,
            tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
            adapt_images=adapt_images,
            settings=settings,
            xp=np,
        )

    try:
        fit = eager_fit()
        summary["log_likelihood_eager"] = float(fit.log_likelihood)
        summary["figure_of_merit_eager"] = float(fit.figure_of_merit)
        summary["eager_inversion_class"] = type(fit.inversion).__name__
        summary["eager_vs_jit_abs_diff"] = abs(summary["figure_of_merit_eager"] - log_evidence_jit)
        print(
            f"  eager figure of merit {summary['figure_of_merit_eager']!r} ({summary['eager_inversion_class']})"
        )
    except Exception as exc:  # noqa: BLE001
        fit = None
        summary["eager"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
    reference = _write_parity(
        dict_path.parent,
        candidate=candidate,
        ds=ds,
        profiling_root=profiling_root,
        autolens_version=al_version,
        compute_reference=_in_process_reference(ds, cell, adapt_images, memo=cli.memo),
    )
    if reference is not None:
        summary["parity_reference"] = {
            k: reference.get(k) for k in ("log_evidence", "method", "n_vis_chunks", "total_s")
        }
    summary["resources"] = resources_dict("jax")
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    if fit is not None:
        _render_fit(lambda: fit, dict_path.parent, cli)
    return dict_path


# --------------------------------------------------------------------------- numba runtime


def _numba_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--arms",
        default="default,numba,numpy_fft",
        help="Comma-separated arms: default (the packaged cost-model route, the headline), "
        "numba and numpy_fft (each route forced, sub-rows).",
    )
    return parser


def run_numba(
    variant: str, cli, thread_env: dict, *, script_file: str, profiling_root: Path
) -> Path:
    """The numba / NumPy runtime row of the standard-size interferometer cell (ruling R4)."""
    import autolens as al
    import numba
    import numpy as np
    from autoarray.inversion.inversion.interferometer.sparse import (
        InversionInterferometerSparse,
    )
    from autoarray.inversion.inversion.interferometer_numba.sparse import (
        InversionInterferometerSparseNumba,
    )

    import _production_config as pc
    from _profile_cli import device_info_dict, resolve_output_paths, resources_dict

    from . import interferometer_pixelized_numpy as numpy_harness
    from . import timing

    _require_standard_size(cli, script_file)
    args, _ = _numba_parser().parse_known_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms or any(a not in NUMBA_ARMS for a in arms):
        raise SystemExit(f"--arms {args.arms!r}: choose from {tuple(NUMBA_ARMS)}")
    headline = "default" if "default" in arms else arms[0]
    numba.set_num_threads(1)
    instrument = cli.instrument or "sma"
    timer = timing.Timer()
    load_start = list(os.getloadavg())
    print(f"--- {script_file} [{variant} / {instrument}] standard size, arms {arms} ---")

    inputs = load_standard_inputs(variant, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    dataset_sparse, preload = numpy_harness.preload_sparse_dataset(
        ds, cli.sparse_batch_size, True, timer
    )
    n_instances = max(int(cli.n_instances), 2 + int(cli.cold_evals))
    cell = pc.standard_interferometer_cell(
        variant,
        instrument,
        dataset=ds.dataset,
        adapt_image=adapt_image,
        mask_radius=ds.mask_radius,
        memo=cli.memo,
        n_instances=n_instances,
    )
    instances = cell["instances"]
    grid = cell["image_plane_mesh_grid"]
    settings = {
        arm: pc.analysis_settings(cell["preset"], cli.memo, **NUMBA_ARMS[arm])[0] for arm in arms
    }

    def analysis_for(arm, instance):
        return al.AnalysisInterferometer(
            dataset=dataset_sparse,
            adapt_images=pc.adapt_images_for(instance, grid, adapt_image),
            settings=settings[arm],
            use_jax=False,
        )

    # Which route each evaluation took: the numba class overrides ``curvature_matrix_diag``, so
    # counting its computations (a counter of the descriptor's own type) names the route.
    numba_target = [(InversionInterferometerSparseNumba, "curvature_matrix_diag")]
    assert "curvature_matrix_diag" in InversionInterferometerSparse.__dict__

    per_eval: dict[str, list[float]] = {arm: [] for arm in arms}
    log_likelihoods: dict[str, list[float]] = {arm: [] for arm in arms}
    routes: dict[str, list[str]] = {arm: [] for arm in arms}
    print(f"\n--- iid stream: {n_instances} instances (0 = warm-up), arms {arms} ---")
    for i, instance in enumerate(instances):
        order = arms if i % 2 == 0 else list(reversed(arms))
        line = []
        for arm in order:
            analysis = analysis_for(arm, instance)

            def call(analysis=analysis, instance=instance):
                t0 = time.perf_counter()
                value = analysis.log_likelihood_function(instance=instance)
                return time.perf_counter() - t0, float(value)

            (dt, value), counts = numpy_harness.count_evaluations(numba_target, call)
            route = "numba" if counts["curvature_matrix_diag"] else "numpy_fft"
            per_eval[arm].append(dt)
            log_likelihoods[arm].append(value)
            routes[arm].append(route)
            line.append(f"{arm} {dt * 1e3:.1f} ms ({route})")
            del analysis
        gc.collect()
        print(f"  [{i:>2}] " + "  ".join(line))

    for arm, forced in NUMBA_ARM_CLASSES.items():
        if arm in arms:
            want = "numba" if forced.endswith("Numba") else "numpy_fft"
            if any(r != want for r in routes[arm]):
                raise SystemExit(f"arm {arm!r} took routes {routes[arm]}, expected {want}")

    arm_records = {}
    for arm in arms:
        summary_arm = pc.timing_summary(per_eval[arm], n_cold=int(cli.cold_evals))
        arm_records[arm] = {
            **summary_arm,
            "settings": {k: v for k, v in NUMBA_ARMS[arm].items()} or "packaged (cost model)",
            "route_per_instance": routes[arm],
            "route_counts": {r: routes[arm].count(r) for r in sorted(set(routes[arm]))},
            "log_likelihood_per_instance": log_likelihoods[arm],
        }
    head = arm_records[headline]
    agreement = {}
    if "numba" in arms and "numpy_fft" in arms:
        gaps = [abs(a - b) for a, b in zip(log_likelihoods["numba"], log_likelihoods["numpy_fft"])]
        agreement["numba_vs_numpy_fft_max_abs_nats"] = float(max(gaps))

    al_version = al.__version__
    candidate = parity_candidate(
        log_likelihood=log_likelihoods[headline][0],
        cell=cell,
        use_mixed_precision=False,
        variant=variant,
        instrument=instrument,
    )
    operator = dataset_sparse.sparse_operator
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "instrument": instrument,
        "model": f"{_variant_model_name(variant)}_numba",
        "transformer": "TransformerNUFFT",
        "configuration": _configuration(
            ds,
            cell,
            inversion_path="sparse_numba",
            headline_arm=headline,
            arms=arms,
            headline_route=(
                "packaged cost model (Settings.interferometer_numba_nnz_per_source_max=None)"
                if headline == "default"
                else f"forced {headline}"
            ),
            use_jax=False,
            n_instances=n_instances,
            thread_env=thread_env,
            numba_version=numba.__version__,
            numba_threads=1,
            sparse_operator_method="nufft",
            sparse_batch_size=int(operator.batch_size),
            operator_extent_shape=[int(operator.y_shape), int(operator.x_shape)],
            preload=preload,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        # The catalogue's numba per-call headline: the headline arm's warm iid median.
        "full_pipeline_single_jit": head["warm_iid_median_s"],
        **{k: v for k, v in head.items() if k not in ("settings",)},
        "arms": arm_records,
        "agreement": agreement,
        "log_likelihood": log_likelihoods[headline][-1],
        "parity_candidate": candidate,
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root
        / "results"
        / "runtime"
        / "interferometer"
        / f"{_variant_model_name(variant)}_numba",
        default_basename=(
            f"{_variant_model_name(variant)}_numba_likelihood_summary_{instrument}_v{al_version}"
        ),
        hints=summary,
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(
        f"\n  headline ({headline}) warm iid median {head['warm_iid_median_s']:.6f} s; row {dict_path}"
    )

    reference = _write_parity(
        dict_path.parent,
        candidate=candidate,
        ds=ds,
        profiling_root=profiling_root,
        autolens_version=al_version,
        compute_reference=_in_process_reference(
            ds, cell, pc.adapt_images_for(cell["instance"], grid, adapt_image), memo=cli.memo
        ),
    )
    if reference is not None:
        summary["parity_reference"] = {
            k: reference.get(k) for k in ("log_evidence", "method", "n_vis_chunks", "total_s")
        }
    summary["resources"] = resources_dict("numba")
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    _render_fit(
        lambda: analysis_for(headline, cell["instance"]).fit_from(instance=cell["instance"]),
        dict_path.parent,
        cli,
    )
    return dict_path


# --------------------------------------------------------------------------- parity reference


def reference_task(
    variant: str,
    instrument: str,
    *,
    profiling_root: Path,
    vis_chunk: int | None = None,
    column_chunk: int | None = None,
    force: bool = False,
    compute: bool = True,
) -> dict | None:
    """Compute (or reuse) and cache the chunked dense reference of one (variant, instrument)."""
    import _production_config as pc
    from _profile_cli import device_info_dict, provenance_dict, resources_dict

    from . import timing

    timer = timing.Timer()
    inputs = load_standard_inputs(variant, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    cell = pc.standard_interferometer_cell(
        variant, instrument, dataset=ds.dataset, adapt_image=adapt_image, mask_radius=ds.mask_radius
    )
    revisions = provenance_dict().get("library_revisions")
    cached, why = pc.load_interferometer_reference(
        profiling_root,
        variant,
        instrument,
        dataset_path=ds.dataset_path,
        library_revisions=revisions,
        unit_vector=cell["unit_vector"],
    )
    path = pc.interferometer_reference_path(profiling_root, variant, instrument)
    if cached is not None and not force:
        print(f"  reference cached and valid: {path} (log evidence {cached['log_evidence']!r})")
        return cached
    if not compute:
        print(f"  no valid reference ({why}) and --finalise-only: nothing to finalise against")
        return None
    adapt_images = pc.adapt_images_for(cell["instance"], cell["image_plane_mesh_grid"], adapt_image)
    print(
        f"\n--- Chunked dense reference: {variant} / {instrument}, {ds.n_vis} visibilities, "
        f"{cell['n_source_pixels']} source pixels ---"
    )
    result = pc.interferometer_dense_reference(
        ds.dataset,
        adapt_images,
        cell["instance"],
        cell["settings"],
        vis_chunk=vis_chunk or pc.INTERFEROMETER_REFERENCE_VIS_CHUNK,
        column_chunk=column_chunk or pc.INTERFEROMETER_REFERENCE_COLUMN_CHUNK,
    )
    doc = {
        "schema": "p2b-interferometer-reference",
        "version": 1,
        "variant": variant,
        "instrument": instrument,
        **result,
        "method_short": "visibility-chunked eager NumPy dense (R6)",
        "dataset_fingerprint": pc.dataset_fingerprint(ds.dataset_path),
        "library_revisions": revisions,
        "instance_unit_vector": cell["unit_vector"],
        "production_preset": cell["record"],
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(doc, indent=2, default=float) + "\n")
    os.replace(staged, path)
    print(f"  reference log evidence {doc['log_evidence']!r} in {doc['total_s']:.1f} s -> {path}")
    return doc


def finalise_parity(variant: str, instrument: str, *, results_root: Path, reference) -> dict:
    """Rewrite the ``parity.json`` of every cell row of (variant, instrument) under ``results_root``.

    A row is finalised only when its ``parity_candidate`` was taken at the reference's instance and
    its library revisions are the reference's; the others are listed, never touched.
    """
    from _profile_cli import write_parity_json

    base = Path(results_root) / "results" / "interferometer" / variant / instrument
    out: dict[str, Any] = {"finalised": [], "skipped": [], "failed": [], "root": str(base)}
    if reference is None:
        out["failed"].append({"reason": "no reference"})
        return out
    for row_path in sorted(base.glob("*/*/*.json")):
        if row_path.stem not in ("runtime", "breakdown"):
            continue
        rel = str(row_path)
        try:
            row = json.loads(row_path.read_text())
        except (OSError, ValueError) as exc:
            out["failed"].append({"row": rel, "reason": f"unreadable: {exc}"})
            continue
        candidate = row.get("parity_candidate")
        if not candidate:
            out["skipped"].append({"row": rel, "reason": "no parity_candidate"})
            continue
        revisions = ((row.get("device") or {}).get("provenance") or {}).get("library_revisions")
        if candidate.get("instance_unit_vector") != reference.get("instance_unit_vector"):
            out["failed"].append({"row": rel, "reason": "different instance"})
            continue
        if revisions != reference.get("library_revisions"):
            out["failed"].append({"row": rel, "reason": "different library revisions"})
            continue
        written = write_parity_json(
            row_path.parent,
            log_likelihood=candidate["log_likelihood"],
            reference_log_likelihood=reference["log_evidence"],
            use_mixed_precision=bool(candidate.get("use_mixed_precision")),
            reference_source=(
                f"cached reference task interferometer_{variant}_{instrument}.json "
                f"({reference.get('method_short')})"
            ),
            autolens_version=str(row.get("autolens_version")),
            source_pixels=int(candidate["source_pixels"]),
            source_pixels_requested=candidate.get("source_pixels_requested"),
            quantity=candidate["quantity"],
            dataset_type="interferometer",
        )
        (out["finalised"] if written else out["failed"]).append({"row": rel})
    return out
