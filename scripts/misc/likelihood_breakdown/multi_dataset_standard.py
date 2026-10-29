"""Shared drivers for the multi-dataset standard-size cells (likelihood-lifecycle 2f, #475).

A ``multi_dataset/delaunay/hst`` cell is phase 2a's imaging standard cell of hst
(``_production_config.standard_jax_cell("delaunay", "hst")``: the hst ``source_pix[2]`` production
preset at Delaunay 1250 + 30 zeroed edge points, MGE lens light, free ``AdaptSplit``, production
over-sampling and adapt image, iid draw 0) fitted to :data:`N_DATASETS` imaging datasets at once
(decision D6):

- every dataset shares the mask, PSF, over-sampling, adapt image, image-plane mesh and lens model;
- dataset ``e`` holds hst's data AND noise map scaled by ``s_e`` (:data:`DATASET_SCALES`:
  1, 0.5): a second exposure with a different flux calibration and the same signal-to-noise, so
  every data-derived production input (over-sampling, adapt image, mesh) is the first dataset's,
  while each dataset's curvature, solve and evidence differ and none is a copy of another's.
  Dataset 0 IS the 2a hst cell;
- the likelihood is the production one: ``af.FactorGraphModel`` over one ``AnalysisImaging`` per
  dataset with the default ``shared_preloads=False`` (the workspace ``multi_dataset/modeling.py``
  configuration), so every dataset computes its own inversion.

Drivers (the cell scripts call them under ``--standard-size``):

- :func:`run_jax` -- ``scripts/multi_dataset/delaunay/likelihood_runtime.py --standard-size
  [--sparse]``: ``jax.jit`` of the summed log likelihood (JAX w-tilde with ``--sparse``, the
  dense mapping route without), lower / compile / first call / steady, its steady median and the
  vmap per call;
- :func:`run_numba` -- ``.../likelihood_runtime_numba.py --standard-size``: the NumPy / numba
  sum (``apply_sparse_operator_cpu``) over the preset's iid stream, its warm median;
- :func:`run_breakdown` -- ``.../likelihood_breakdown{,_numba}.py --standard-size``: phase 2a's
  imaging breakdown of dataset 0 (which is the 2a cell), scaled to the sum (no step is shared, so
  the multi-dataset total is ``N_DATASETS`` x the per-dataset total; decision D7), and an eager
  evaluation of the sum for the parity witness.

Parity (decision D8): 2a's reference per dataset -- the eager NumPy dense-mapping
``AnalysisImaging(use_jax=False)`` without a sparse operator (``numpy_dense_reference``) -- summed
over the datasets, computed in-process (hst is small); tolerance the imaging default 1e-4 x
|reference| in fp64.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

VARIANT = "delaunay"
INSTRUMENT = "hst"
MASK_RADIUS = 3.5
#: Datasets per fit and their flux-calibration scales (decision D6).
N_DATASETS = 2
DATASET_SCALES = (1.0, 0.5)
PARITY_QUANTITY = (
    "the summed log likelihood (sum over datasets of log_likelihood_function, the figure of "
    "merit, FactorGraphModel) at the standard cell's iid draw 0, vs the sum of 2a's eager NumPy "
    "dense-mapping reference per dataset (decision D8)"
)


def dataset_scales(n_datasets: int) -> list[float]:
    """``s_e`` for ``n_datasets`` datasets: :data:`DATASET_SCALES`, halving beyond them."""
    n = max(1, int(n_datasets))
    return [DATASET_SCALES[e] if e < len(DATASET_SCALES) else 0.5**e for e in range(n)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=N_DATASETS,
        help=f"Datasets in the fit (default {N_DATASETS}, decision D6).",
    )
    return parser


def multi_args() -> argparse.Namespace:
    args, _ = _parser().parse_known_args()
    if args.n_datasets < 1:
        raise SystemExit("--n-datasets must be at least 1")
    return args


def _require_standard_size(cli, script: str) -> None:
    if not cli.standard_size:
        raise SystemExit(
            f"{script}: this driver runs only the standard-size cell; pass --standard-size"
        )
    instrument = cli.instrument or INSTRUMENT
    if instrument != INSTRUMENT:
        raise SystemExit(f"{script}: the multi-dataset cell is planned at {INSTRUMENT} only")


# --------------------------------------------------------------------------- the datasets


def load_base(profiling_root: Path, timer, *, memo: str = "off", use_mixed_precision=False):
    """2a's hst standard cell: the production dataset (no operator), adapt image and cell."""
    import autolens as al
    from simulators.imaging import INSTRUMENTS

    import _production_config as pc
    from _profile_cli import auto_simulate_if_missing

    dataset_path = Path("dataset") / "imaging" / INSTRUMENT
    auto_simulate_if_missing(
        dataset_path, dataset_type="imaging", instrument=INSTRUMENT, workspace_root=profiling_root
    )
    with timer.section("dataset_load"):
        dataset = al.Imaging.from_fits(
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            pixel_scales=INSTRUMENTS[INSTRUMENT]["pixel_scale"],
        )
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=MASK_RADIUS
        )
        dataset = dataset.apply_mask(mask=mask)
    preset = pc.standard_size_preset(pc.STANDARD_FAMILY[VARIANT], INSTRUMENT)
    with timer.section("production_data"):
        dataset, adapt_image, data_record = pc.production_standard_data(
            dataset, dataset_path, preset, sparse=None
        )
    cell = pc.standard_jax_cell(
        VARIANT,
        INSTRUMENT,
        dataset=dataset,
        adapt_image=adapt_image,
        dataset_path=dataset_path,
        mask_radius=MASK_RADIUS,
        memo=memo,
        use_mixed_precision=use_mixed_precision,
    )
    return {
        "dataset": dataset,
        "dataset_path": dataset_path,
        "adapt_image": adapt_image,
        "data_record": data_record,
        "cell": cell,
    }


def scaled_datasets(base, scales: list[float], *, sparse: str | None, batch_size: int, timer):
    """One dataset per scale: ``base`` with its data and noise map times ``s_e`` (D6).

    ``sparse``: ``"jax"`` (``apply_sparse_operator``), ``"cpu"`` (the numba
    ``apply_sparse_operator_cpu``) or None (dense mapping). Each dataset builds its own operator
    (it depends on the noise map). Returns ``(datasets, operator build seconds)``.
    """
    import autolens as al
    import numpy as np

    raw = [base]
    for s in scales[1:]:
        raw.append(
            al.Imaging(
                data=al.Array2D(values=np.asarray(base.data) * s, mask=base.mask),
                noise_map=al.Array2D(values=np.asarray(base.noise_map) * s, mask=base.mask),
                psf=base.psf,
                noise_covariance_matrix=base.noise_covariance_matrix,
                over_sample_size_lp=base.over_sample_size_lp,
                over_sample_size_pixelization=base.over_sample_size_pixelization,
                convolve_over_sample_size_lp=base.convolve_over_sample_size_lp,
                convolve_over_sample_size_pixelization=base.convolve_over_sample_size_pixelization,
                check_noise_map=False,
            )
        )
    if sparse is None:
        return raw, None
    with timer.section("apply_sparse_operator"):
        if sparse == "cpu":
            out = [d.apply_sparse_operator_cpu() for d in raw]
        else:
            out = [d.apply_sparse_operator(batch_size=int(batch_size)) for d in raw]
    return out, timer.records[-1][1]


def factor_graph(datasets, *, cell, adapt_images, settings, use_jax: bool):
    """The production multi-dataset fit: ``FactorGraphModel`` of one analysis per dataset."""
    import autofit as af
    import autolens as al

    analyses = [
        al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=settings,
            positions_likelihood_list=cell["positions_likelihood_list"],
            use_jax=use_jax,
        )
        for dataset in datasets
    ]
    factors = [af.AnalysisFactor(prior_model=cell["model"], analysis=a) for a in analyses]
    return af.FactorGraphModel(*factors, use_jax=use_jax), analyses


def summed_log_likelihood(graph, n_datasets: int, instance):
    return graph.log_likelihood_function([instance] * n_datasets)


def reference(datasets, *, cell, adapt_images) -> dict:
    """D8: 2a's eager NumPy dense-mapping reference of each dataset, summed."""
    import _production_config as pc

    t0 = time.perf_counter()
    per = [
        pc.numpy_dense_reference(
            d,
            adapt_images,
            cell["instance"],
            settings=cell["settings"],
            positions_likelihood_list=cell["positions_likelihood_list"],
        )
        for d in datasets
    ]
    return {
        "log_likelihood": float(sum(per)),
        "log_likelihood_per_dataset": per,
        "total_s": time.perf_counter() - t0,
        "method": "sum over datasets of the eager NumPy dense-mapping AnalysisImaging(use_jax=False) "
        "without a sparse operator (2a's reference, decision D8)",
    }


def configuration(base, scales, **extra) -> dict:
    cell, dataset = base["cell"], base["dataset"]
    return {
        "pixel_scale_arcsec": float(dataset.pixel_scales[0]),
        "mask_radius_arcsec": MASK_RADIUS,
        "image_pixels_masked": int(dataset.data.shape[0]),
        "over_sampled_pixels": int(dataset.grids.lp.over_sampled.shape[0]),
        "source_pixels": int(cell["n_source_pixels"]),
        "source_pixels_requested": 1250,
        "standard_size": True,
        "production_preset": cell["record"],
        "production_data_record": base["data_record"],
        "regularization": cell["reg_provenance"],
        "memo_provenance": cell["memo_provenance"],
        "n_datasets": len(scales),
        "dataset_scales": [float(s) for s in scales],
        "shared_preloads": False,
        "multi_dataset_definition": (
            "decision D6: 2a's hst imaging standard cell in every dataset (shared mask, PSF, "
            "over-sampling, adapt image, mesh and lens model), dataset e's data and noise map "
            "times s_e (same S/N); FactorGraphModel of AnalysisImaging (shared_preloads=False)"
        ),
        **extra,
    }


def _write_witness(cell_dir, *, value, ref, autolens_version, use_mixed_precision, quantity):
    from _profile_cli import write_parity_json

    return write_parity_json(
        Path(cell_dir),
        log_likelihood=value,
        reference_log_likelihood=float("nan") if ref is None else ref["log_likelihood"],
        use_mixed_precision=bool(use_mixed_precision),
        reference_source=(ref or {}).get("method", "2a reference per dataset (failed)"),
        autolens_version=autolens_version,
        source_pixels=1280,
        source_pixels_requested=1250,
        quantity=quantity,
        dataset_type="imaging",
    )


def _finish(summary, dict_path, *, cli, base, datasets, adapt_images, value, engine):
    """After every timed region: dataset 0's eager fit (fit.png), the witness, resources."""
    import autolens as al
    import numpy as np

    from _profile_cli import render_fit_png, resources_dict

    cell = base["cell"]
    instance = cell["instance"]
    try:
        ref = reference(datasets, cell=cell, adapt_images=adapt_images)
        summary["parity_reference"] = ref
    except Exception as exc:  # noqa: BLE001 — a failed reference must not cost the row
        ref = None
        summary["parity_reference"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_witness(
        dict_path.parent,
        value=value,
        ref=ref,
        autolens_version=summary["autolens_version"],
        use_mixed_precision=summary.get("use_mixed_precision", False),
        quantity=PARITY_QUANTITY,
    )
    summary["resources"] = resources_dict(engine)
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    try:
        fit = al.FitImaging(
            dataset=datasets[0],
            tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
            adapt_images=adapt_images,
            settings=cell["settings"],
            xp=np,
        )
    except Exception as exc:  # noqa: BLE001 — a figure must not cost the row
        print(f"  WARNING: [fit.png] eager fit failed: {type(exc).__name__}: {exc}")
        return
    render_fit_png(fit, dict_path.parent, dataset_type="imaging", cli=cli)


# --------------------------------------------------------------------------- JAX runtime


def run_jax(cli, *, script_file: str, profiling_root: Path) -> Path:
    """The JAX runtime row of the standard-size multi-dataset cell (w-tilde or dense mapping)."""
    import autolens as al
    import jax
    import jax.numpy as jnp
    from autofit.jax import register_model

    import _production_config as pc
    from _profile_cli import device_info_dict, resolve_output_paths, resources_dict

    from . import timing

    _require_standard_size(cli, script_file)
    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True (fp32 truncation).")
    sparse = bool(cli.use_sparse_operator)
    scales = dataset_scales(multi_args().n_datasets)
    n = len(scales)
    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    load_start = list(os.getloadavg())
    print(
        f"--- {script_file} [multi_dataset {VARIANT} / {INSTRUMENT}] standard size, {n} datasets, "
        f"{'w-tilde' if sparse else 'dense mapping'} ---"
    )
    base = load_base(
        profiling_root, timer, memo=cli.memo, use_mixed_precision=cli.use_mixed_precision
    )
    cell = base["cell"]
    datasets, operator_s = scaled_datasets(
        base["dataset"],
        scales,
        sparse="jax" if sparse else None,
        batch_size=cli.sparse_batch_size,
        timer=timer,
    )
    model, instance = cell["model"], cell["instance"]
    register_model(model)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
    adapt_images = pc.adapt_images_for(instance, cell["image_plane_mesh_grid"], base["adapt_image"])
    graph, _ = factor_graph(
        datasets, cell=cell, adapt_images=adapt_images, settings=cell["settings"], use_jax=True
    )
    print(
        f"  preset {cell['preset'].name}: {cell['n_source_pixels']} source pixels, "
        f"{model.total_free_parameters} free parameters, {n} datasets"
    )

    def full(params):
        return summed_log_likelihood(graph, n, params)

    print("\n--- Full pipeline: jax.jit(FactorGraphModel.log_likelihood_function) ---")
    compiled, result = timing.jit_profile(
        full, "full_pipeline", params_tree, n_repeats=10, timer=timer, jit_records=jit_records
    )
    per_call = jit_records["full_pipeline"]["steady_per_call_s"]
    median = timing.headline_steady_median(
        compiled,
        params_tree,
        block_mean_s=per_call,
        block_mean_is=timing.single_jit_block_is(10),
    )
    value = float(result)
    print(f"  summed log likelihood (JIT) = {value!r}")
    del compiled
    jax.clear_caches()
    gc.collect()

    vmap = None
    attempts = []
    for nb in (3, 1):
        batched = jax.tree_util.tree_map(
            lambda leaf, k=nb: jnp.broadcast_to(leaf, (k, *leaf.shape)), params_tree
        )
        try:
            vper = timing.vmap_profile(
                full,
                "full_pipeline",
                batched,
                nb,
                n_repeats=10,
                timer=timer,
                jit_records=jit_records,
            )
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "RESOURCE_EXHAUSTED" not in text and "out of memory" not in text.lower():
                raise
            attempts.append({"batch": nb, "status": "oom", "error": text.splitlines()[0][:300]})
            jax.clear_caches()
            gc.collect()
            continue
        vmap = {
            "batch_size": nb,
            "batch_size_source": "fixed (3, OOM fallback 1)",
            "batch_time": vper * nb,
            "per_call": vper,
            "instances": "draw 0 broadcast to the batch (the runtime cells' convention)",
            "attempts": attempts,
        }
        break
    jax.clear_caches()
    gc.collect()

    al_version = al.__version__
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("jax"),
        "instrument": INSTRUMENT,
        "model": VARIANT,
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "n_datasets": n,
        "configuration": configuration(
            base,
            scales,
            inversion_path="sparse" if sparse else "dense",
            sparse_batch_size=int(cli.sparse_batch_size) if sparse else None,
            operator_build_s=operator_s,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        "log_likelihood_jit": value,
        "full_pipeline_single_jit": per_call,
        **median,
        "per_dataset_equivalent_s": per_call / n,
        "vmap": vmap
        if vmap is not None
        else {"status": "oom", "attempts": attempts, "note": "every vmap batch OOM'd"},
        "jit_phases": jit_records,
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root / "results" / "runtime" / "multi_dataset" / VARIANT,
        default_basename=f"{VARIANT}_likelihood_summary_{INSTRUMENT}_v{al_version}",
        hints=summary,
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Row written: {dict_path}")
    _finish(
        summary,
        dict_path,
        cli=cli,
        base=base,
        datasets=datasets,
        adapt_images=adapt_images,
        value=value,
        engine="jax",
    )
    return dict_path


# --------------------------------------------------------------------------- numba runtime


def run_numba(cli, thread_env: dict, *, script_file: str, profiling_root: Path) -> Path:
    """The numba / NumPy runtime row of the standard-size multi-dataset cell."""
    import autolens as al
    import numba

    import _production_config as pc
    from _profile_cli import device_info_dict, resolve_output_paths, resources_dict

    from . import timing

    _require_standard_size(cli, script_file)
    numba.set_num_threads(1)
    scales = dataset_scales(multi_args().n_datasets)
    n = len(scales)
    timer = timing.Timer()
    load_start = list(os.getloadavg())
    print(
        f"--- {script_file} [multi_dataset {VARIANT} / {INSTRUMENT}] standard size, {n} datasets, "
        "numba ---"
    )
    base = load_base(profiling_root, timer, memo=cli.memo)
    cell = base["cell"]
    datasets, operator_s = scaled_datasets(
        base["dataset"], scales, sparse="cpu", batch_size=cli.sparse_batch_size, timer=timer
    )
    n_instances = max(int(cli.n_instances), 2 + int(cli.cold_evals))
    instances = pc.iid_instances(cell["model"], n_instances, cell["preset"])
    grid = cell["image_plane_mesh_grid"]
    per_eval, values = [], []
    print(f"\n--- iid stream: {n_instances} instances (0 = warm-up) ---")
    for i, instance in enumerate(instances):
        graph, _ = factor_graph(
            datasets,
            cell=cell,
            adapt_images=pc.adapt_images_for(instance, grid, base["adapt_image"]),
            settings=cell["settings"],
            use_jax=False,
        )
        t0 = time.perf_counter()
        value = float(summed_log_likelihood(graph, n, instance))
        dt = time.perf_counter() - t0
        per_eval.append(dt)
        values.append(value)
        del graph
        gc.collect()
        print(f"  [{i:>2}] {dt * 1e3:.1f} ms  log L {value!r}")
    stats = pc.timing_summary(per_eval, n_cold=int(cli.cold_evals))
    al_version = al.__version__
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "instrument": INSTRUMENT,
        "model": f"{VARIANT}_numba",
        "n_datasets": n,
        "configuration": configuration(
            base,
            scales,
            inversion_path="sparse_numba",
            use_jax=False,
            n_instances=n_instances,
            thread_env=thread_env,
            numba_version=numba.__version__,
            numba_threads=1,
            operator_build_s=operator_s,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        # The numba headline: the warm iid median of the summed likelihood (2a's convention).
        "full_pipeline_single_jit": stats["warm_iid_median_s"],
        "per_dataset_equivalent_s": stats["warm_iid_median_s"] / n,
        **stats,
        "log_likelihood_per_instance": values,
        "log_likelihood": values[0],
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root / "results" / "runtime" / "multi_dataset" / f"{VARIANT}_numba",
        default_basename=f"{VARIANT}_numba_likelihood_summary_{INSTRUMENT}_v{al_version}",
        hints=summary,
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(
        f"\n  warm iid median {stats['warm_iid_median_s']:.6f} s per summed call; row {dict_path}"
    )
    _finish(
        summary,
        dict_path,
        cli=cli,
        base=base,
        datasets=datasets,
        adapt_images=pc.adapt_images_for(cell["instance"], grid, base["adapt_image"]),
        value=values[0],
        engine="numba",
    )
    return dict_path


# --------------------------------------------------------------------------- breakdown


def run_breakdown(cli, *, numba: bool, script_file: str, profiling_root: Path) -> Path:
    """The multi-dataset breakdown row: 2a's imaging breakdown of dataset 0 (the 2a cell), its
    total scaled to the sum (D7), and an eager evaluation of the sum for the witness."""
    import autolens as al

    import _production_config as pc
    from _profile_cli import resources_dict

    from . import timing

    _require_standard_size(cli, script_file)
    if not cli.output_path:
        raise SystemExit(f"{script_file}: pass --output-path (the mirrored multi_dataset cell)")
    scales = dataset_scales(multi_args().n_datasets)
    n = len(scales)
    imaging = Path(profiling_root) / "scripts" / "imaging" / VARIANT
    target = imaging / ("likelihood_breakdown_numba.py" if numba else "likelihood_breakdown.py")
    argv = [target.name]
    skip = False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg == "--n-datasets":
            skip = True
            continue
        if not arg.startswith("--n-datasets="):
            argv.append(arg)
    print(f"--- {script_file}: 2a's {target.name} on dataset 0 (the hst imaging cell) ---")
    saved = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = saved
    dict_path = Path(cli.output_path)
    row = json.loads(dict_path.read_text())
    lead = row.pop("total_step_by_step", None)
    row["total_step_by_step_lead_dataset"] = lead
    row["total_step_by_step"] = None if lead is None else float(lead) * n
    row["multi_dataset"] = {
        "n_datasets": n,
        "dataset_scales": [float(s) for s in scales],
        "per_dataset_total_s": lead,
        "total_step_by_step": row["total_step_by_step"],
        "rule": (
            "decision D7: shared_preloads=False, so every step runs once per dataset and the "
            "datasets share one cost structure (same mask, over-sampling, mesh and operator "
            "shape); the total is n_datasets x dataset 0's total. 'steps' are dataset 0's."
        ),
    }
    row["n_datasets"] = n

    timer = timing.Timer()
    base = load_base(profiling_root, timer, use_mixed_precision=bool(cli.use_mixed_precision))
    cell = base["cell"]
    sparse = "cpu" if numba else ("jax" if cli.use_sparse_operator else None)
    datasets, _ = scaled_datasets(
        base["dataset"], scales, sparse=sparse, batch_size=cli.sparse_batch_size, timer=timer
    )
    adapt_images = pc.adapt_images_for(
        cell["instance"], cell["image_plane_mesh_grid"], base["adapt_image"]
    )
    graph, _ = factor_graph(
        datasets, cell=cell, adapt_images=adapt_images, settings=cell["settings"], use_jax=False
    )
    with timer.section("eager_sum"):
        value = float(summed_log_likelihood(graph, n, cell["instance"]))
    row["summed_log_likelihood_eager"] = value
    row["configuration"] = {**row.get("configuration", {}), **configuration(base, scales)}
    try:
        ref = reference(datasets, cell=cell, adapt_images=adapt_images)
        row["parity_reference"] = ref
    except Exception as exc:  # noqa: BLE001
        ref = None
        row["parity_reference"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_witness(
        dict_path.parent,
        value=value,
        ref=ref,
        autolens_version=str(row.get("autolens_version", al.__version__)),
        use_mixed_precision=bool(cli.use_mixed_precision),
        quantity=PARITY_QUANTITY + "; the breakdown's value is an eager NumPy evaluation of the "
        "sum on the cell's route, after dataset 0's timed steps",
    )
    row["resources"] = resources_dict("numba" if numba else "jax")
    dict_path.write_text(json.dumps(row, indent=2, default=float))
    print(
        f"\n  multi-dataset total: {n} x {lead!r} s = {row['total_step_by_step']!r} s; "
        f"eager sum {value!r}"
    )
    return dict_path
