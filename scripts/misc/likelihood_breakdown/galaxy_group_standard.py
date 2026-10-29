"""Shared drivers for the multi_galaxy and group standard cells (likelihood-lifecycle 2d, #473).

Every setup of a ``multi_galaxy`` / ``group`` likelihood x instrument evaluates one likelihood:
the standard cell of ``scripts/misc/cell_builders/galaxy_group.py`` (the workspace modeling
model, MGE lens light, with an MGE source or the standard-size production pixelization) at its
seeded iid draw 0. The cell scripts under ``scripts/{multi_galaxy,group}/<variant>/`` are thin
wrappers over :func:`main`, so the setups cannot drift:

- ``likelihood_runtime.py [--sparse]`` -- :func:`run_jax_runtime`: ``AnalysisImaging(use_jax=
  True)`` on the dense mapping route (setup ``jax_*_dense_fp64``) or the JAX w-tilde sparse
  operator (``--sparse``, ``jax_*_sparse_fp64``): lower / compile / first call / steady block,
  the steady median, and the vmap per call the catalogue reads as the JAX headline; then, after
  every timed region, ``parity.json`` against the eager NumPy dense reference and ``fit.png``.
- ``likelihood_breakdown.py [--sparse]`` -- :func:`run_jax_breakdown`: a nested prefix walk of
  ``AnalysisImaging.fit_from`` (traced grids, blurred image, data vector, curvature, F + H,
  reconstruction, figure of merit), each prefix jitted and timed, rows by successive differences
  (``timing.split_by_successive_differences``; the XLA fusion caveat applies).
- ``likelihood_runtime_numba.py`` -- :func:`run_numba_runtime`: ``AnalysisImaging(use_jax=False)``
  on the numba CPU sparse operator over the iid stream (instance 0 discarded, ``--cold-evals``
  cold, the rest warm; headline the warm iid median), threads pinned to 1; parity and fit.png.
- ``likelihood_breakdown_numba.py`` -- :func:`run_numba_breakdown`: the eager stage walk of the
  same fit on the iid stream (each stage's lazily-evaluated property timed in the library's own
  order), the per-stage median over the warm instances.

As in the imaging cells, the runtime row writes the cell's ``parity.json`` and ``fit.png``; the
breakdown row leaves both to it (they share the cell folder). Without ``--output-path`` a row
goes to the gitignored scratch tree ``output/p2d_local/results/...`` (never ``results/``).
``--fast`` (a local check, never a campaign leg) cuts repeats, the median and the stream.

Stdlib only at import: the numba scripts pin their threads before numpy loads.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: Provisional A100 vmap batch per variant (decision D8): by analogy with the imaging hst rows of
#: ``vram/config.py`` (pixelized 16, MGE 64; these datasets have 11k / 18k masked pixels against
#: imaging hst's 15k). An out-of-memory batch halves until one fits. Non-GPU backends cap at 3.
VMAP_BATCH_GPU = {"mge": 64, "rectangular": 16, "delaunay": 16}
NON_GPU_VMAP_BATCH = 3

#: The JAX breakdown's prefix stages, in order (key -> row name).
JAX_STAGES = {
    1: "Ray-trace grids (tracer)",
    2: "Blurred image (non-linear light)",
    3: "Data vector D (mapping + PSF)",
    4: "Curvature matrix F",
    5: "Regularization (F + H)",
    6: "Reconstruction (NNLS)",
    7: "Mapped reconstruction, chi-squared & log evidence",
}
#: The numba breakdown's eager stages, in the library's lazy-evaluation order.
NUMBA_STAGES = (
    "FitImaging construct (tracer)",
    "Blurred image (non-linear light)",
    "Inversion build (trace + mesh + mappers)",
    "Data vector D [numba]",
    "Curvature matrix F [numba]",
    "Regularization (F + H)",
    "Reconstruction (NNLS)",
    "Mapped reconstructed image",
    "Chi-squared & log evidence",
)
STAGE_NOTE = (
    "Each row is one stage of AnalysisImaging.fit_from in the library's own lazy order; the "
    "rows sum to total_step_by_step. JAX rows are successive differences of jitted prefixes "
    "(params -> stage k), so XLA fusion can move work across a boundary and a small negative "
    "row is noise; numba rows time each cached property on first access, per instance, and "
    "report the median over the warm instances."
)
#: XLA:CPU deadlock workaround (autolens_profiling#473, decision D12). On jaxlib 0.11.2 a compiled
#: 2d likelihood on the CPU backend deadlocks after a few calls: execution blocks in
#: ``CpuPjRtRawLoadedExecutable::Execute`` / ``BlockUntilReady`` with every thread idle (0 % CPU,
#: futex). It reproduces from the exported StableHLO alone in a process that imports only jax
#: (no PyAuto library, no numba), and goes away with either ``--xla_cpu_multi_thread_eigen=false``
#: or ``--xla_cpu_enable_concurrency_optimized_scheduler=false``. The first is the cheaper one
#: (laptop, per call: the second is about 2x slower than it on the MGE cells and within ~15 % on
#: the pixelized cells), so every JAX leg appends it to ``XLA_FLAGS`` (one flag for every cell,
#: human 2026-10-29); the row records it in ``device.xla_flags`` and
#: ``configuration.xla_cpu_workaround``. It only changes XLA:CPU code (no effect on a GPU
#: executable), and it is a library-free change to how the cell is compiled, not to what it computes.
XLA_CPU_WORKAROUND_FLAG = "--xla_cpu_multi_thread_eigen=false"
XLA_CPU_WORKAROUND_NOTE = (
    f"{XLA_CPU_WORKAROUND_FLAG} appended to XLA_FLAGS: jaxlib 0.11.2 XLA:CPU deadlocks on these "
    "executables after a few calls with the default (autolens_profiling#473, D12)"
)
PARITY_QUANTITY = (
    "figure_of_merit (the cell's log_likelihood_function value: the log evidence) at iid draw 0"
)


def _avoid_xla_cpu_deadlock() -> None:
    """Append :data:`XLA_CPU_WORKAROUND_FLAG` to ``XLA_FLAGS`` before the JAX backend starts."""
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {XLA_CPU_WORKAROUND_FLAG}".strip()


def _driver_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Local check only: 2 steady repeats, a 5-call median, vmap batch 2, 4 instances.",
    )
    return parser


def _setup_id(engine: str, cli) -> str:
    sparse = engine == "numba" or bool(cli.use_sparse_operator)
    if engine == "numba":
        return "numba_cpu_sparse_fp64"
    import jax

    device = "gpu" if jax.default_backend() == "gpu" else "cpu"
    precision = "mixed" if cli.use_mixed_precision else "fp64"
    return f"jax_{device}_{'sparse' if sparse else 'dense'}_{precision}"


def _host_class(cli) -> str:
    name = cli.config_name or ""
    for host in ("hpc_ral_cpu", "hpc_a100"):
        if name.startswith(host):
            return host
    return "local_gpu" if "gpu" in name else "local_laptop"


def output_path(cli, *, dataset_type, variant, instrument, engine, measurement, root) -> Path:
    """``--output-path`` when given (the campaign submits always pass one), else the scratch
    mirror ``output/p2d_local/results/<dt>/<variant>/<inst>/<setup>/<host>/<measurement>.json``."""
    if getattr(cli, "output_path", None) is not None:
        path = Path(cli.output_path)
    else:
        path = (
            Path(root)
            / "output"
            / "p2d_local"
            / "results"
            / dataset_type
            / variant
            / instrument
            / _setup_id(engine, cli)
            / _host_class(cli)
            / f"{measurement}.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _configuration(cell: dict, **extra) -> dict:
    spec, dataset = cell["spec"], cell["dataset"]
    return {
        "dataset_type": spec.dataset_type,
        "pixel_scale_arcsec": spec.pixel_scale,
        "mask_radius_arcsec": spec.mask_radius,
        "image_pixels_masked": int(dataset.data.shape[0]),
        "over_sampled_pixels": int(dataset.grids.lp.over_sampled.shape[0]),
        "source_pixels": cell["n_source_pixels"],
        # What was asked for (Delaunay 1250 Hilbert points, rectangular 1225); built is above.
        "source_pixels_requested": cell["source_pixels_requested"],
        "standard_size": True,
        "galaxy_group_cell": cell["record"],
        "memo_provenance": cell["memo_provenance"],
        "instance": "iid draw 0 of the seeded sequence",
        "instance_unit_vector": cell["unit_vectors"][0],
        **extra,
    }


def _analysis(cell: dict, instance, *, use_jax: bool, dataset=None):
    import autolens as al
    from cell_builders import galaxy_group as gg

    return al.AnalysisImaging(
        dataset=cell["dataset"] if dataset is None else dataset,
        adapt_images=gg.adapt_images_for(cell, instance),
        settings=cell["settings"],
        use_jax=use_jax,
    )


def _reference_and_fit(cell: dict, instance):
    """``(reference log evidence, eager NumPy dense fit)`` at ``instance``; a failure costs only
    the witness and the figure, never the row."""
    import _production_config as pc

    try:
        analysis = _analysis(
            cell, instance, use_jax=False, dataset=pc.numpy_dense_twin(cell["dataset"])
        )
        fit = analysis.fit_from(instance=instance)
        return float(fit.figure_of_merit), fit
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: eager NumPy dense reference failed: {type(exc).__name__}: {exc}")
        return float("nan"), None


def _write_witness_and_figure(
    dict_path: Path, cell: dict, *, value, use_mixed_precision, autolens_version, cli, note
):
    """``parity.json`` then ``fit.png`` beside the row, after every timed region."""
    from _profile_cli import render_fit_png, write_parity_json

    reference, fit = _reference_and_fit(cell, cell["instances"][0])
    spec = cell["spec"]
    write_parity_json(
        dict_path.parent,
        log_likelihood=value,
        reference_log_likelihood=reference,
        use_mixed_precision=use_mixed_precision,
        reference_source=(
            "eager NumPy dense-mapping al.AnalysisImaging(use_jax=False, no sparse operator) in "
            f"the same process: {spec.dataset_type}/{spec.instrument}, same dataset, model and "
            f"instance ({note})"
        ),
        autolens_version=autolens_version,
        source_pixels=cell["n_source_pixels"],
        source_pixels_requested=cell["source_pixels_requested"],
        quantity=PARITY_QUANTITY,
        dataset_type=spec.dataset_type,
    )
    if fit is not None:
        render_fit_png(fit, dict_path.parent, dataset_type="imaging", cli=cli)
    return reference


def _write(dict_path: Path, summary: dict) -> None:
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Row written: {dict_path}")


# --------------------------------------------------------------------------- JAX


def _jax_cell(dataset_type, variant, cli, root, n_instances=1):
    import jax
    from cell_builders import galaxy_group as gg

    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True (fp32 truncation).")
    sparse = int(cli.sparse_batch_size) if cli.use_sparse_operator else None
    return gg.standard_cell(
        dataset_type,
        variant,
        root=root,
        sparse=sparse,
        memo=cli.memo,
        n_instances=n_instances,
        use_mixed_precision=cli.use_mixed_precision,
    )


def _flat_params(instance):
    """``(leaves, treedef)`` of the instance pytree. The jitted functions take the flat leaves and
    unflatten inside the trace: ``jax.jit(f).lower(tree)`` rebuilds the tree from abstract
    ``ArgInfo`` leaves, and a profile whose constructor computes on its parameters
    (``dPIEMassSph`` derives ``b0`` from ``sigma``) cannot be built from those (the group cells).
    """
    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(jax.tree_util.tree_map(jnp.asarray, instance))
    return leaves, treedef


def _check_route(analysis, params_tree, sparse: bool, treedef) -> str:
    """The inversion class the traced pipeline builds (a shape-only trace, no compute)."""
    import jax

    info: dict[str, str] = {}

    def traced(params):
        fit = analysis.fit_from(instance=jax.tree_util.tree_unflatten(treedef, params))
        info["inversion_class"] = type(fit.inversion).__name__
        return fit.figure_of_merit

    jax.eval_shape(traced, params_tree)
    want = "InversionImagingSparse" if sparse else "InversionImagingMapping"
    if info.get("inversion_class") != want:
        raise SystemExit(f"expected {want} on this setup, traced {info}")
    print(f"  inversion class (traced): {info['inversion_class']}")
    return info["inversion_class"]


def run_jax_runtime(dataset_type, variant, cli, args, *, script_file, root) -> Path:
    import autolens as al
    import jax
    import jax.numpy as jnp
    from autofit.jax import register_model

    from _profile_cli import device_info_dict, resources_dict

    from . import timing

    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    load_start = list(os.getloadavg())
    sparse = bool(cli.use_sparse_operator)
    print(f"--- {script_file} [{dataset_type}/{variant}] {'sparse' if sparse else 'dense'} ---")
    with timer.section("cell_build"):
        cell = _jax_cell(dataset_type, variant, cli, root)
    instance = cell["instances"][0]
    model = cell["model"]
    register_model(model)
    params_tree, treedef = _flat_params(instance)
    analysis = _analysis(cell, instance, use_jax=True)
    inversion_class = _check_route(analysis, params_tree, sparse, treedef)
    n_repeats = 2 if args.fast else 10

    def full_pipeline(params):
        return analysis.log_likelihood_function(
            instance=jax.tree_util.tree_unflatten(treedef, params)
        )

    print("\n--- Full pipeline: jax.jit(AnalysisImaging.log_likelihood_function) ---")
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
        n_timed=5 if args.fast else None,
        block_mean_is=timing.single_jit_block_is(n_repeats),
    )
    value = float(full_result)
    print(f"  full log evidence = {value!r}")
    del compiled
    jax.clear_caches()
    gc.collect()

    on_gpu = jax.default_backend() == "gpu"
    batch = 2 if args.fast else (VMAP_BATCH_GPU[variant] if on_gpu else NON_GPU_VMAP_BATCH)
    vmap, attempts = None, []
    for nb in sorted({batch, max(1, batch // 2), max(1, batch // 4), 1}, reverse=True):
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
            attempts.append({"batch": nb, "status": "oom", "error": text.splitlines()[0][:300]})
            print(f"  vmap batch {nb}: out of memory, trying smaller")
            jax.clear_caches()
            gc.collect()
            continue
        vmap = {
            "batch_size": nb,
            "batch_size_requested": batch,
            "batch_size_source": "galaxy_group_standard.VMAP_BATCH_GPU (provisional, D8)"
            if on_gpu
            else f"non-GPU cap {NON_GPU_VMAP_BATCH}",
            "batch_time": per_call * nb,
            "per_call": per_call,
            "speedup_vs_single_jit": round(full_per_call / per_call, 2),
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
        "instrument": cell["spec"].instrument,
        "model": variant,
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "configuration": _configuration(
            cell,
            inversion_path="sparse" if sparse else "dense",
            inversion_class=inversion_class,
            sparse_batch_size=int(cli.sparse_batch_size) if sparse else None,
            free_parameters=int(model.total_free_parameters),
            n_repeats=n_repeats,
            fast=bool(args.fast),
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
            xla_cpu_workaround=XLA_CPU_WORKAROUND_NOTE,
        ),
        "log_evidence_jit": value,
        "full_pipeline_single_jit": full_per_call,
        **full_median,
        "vmap": vmap
        if vmap is not None
        else {"status": "oom", "attempts": attempts, "note": "every vmap batch OOM'd"},
        "jit_phases": jit_records,
        "timer": dict(timer.records),
    }
    dict_path = output_path(
        cli,
        dataset_type=dataset_type,
        variant=variant,
        instrument=cell["spec"].instrument,
        engine="jax",
        measurement="runtime",
        root=root,
    )
    _write(dict_path, summary)
    reference = _write_witness_and_figure(
        dict_path,
        cell,
        value=value,
        use_mixed_precision=cli.use_mixed_precision,
        autolens_version=al_version,
        cli=cli,
        note="the JIT log_likelihood_function value",
    )
    summary["parity_reference_log_evidence"] = reference
    summary["resources"] = resources_dict("jax")
    _write(dict_path, summary)
    return dict_path


def _stage_value(fit, stage: int):
    """The sum of every quantity up to ``stage`` (forces XLA to compute them all)."""
    import jax.numpy as jnp

    def raw(x):
        return jnp.asarray(getattr(x, "array", x))

    total = jnp.sum(raw(fit.tracer.traced_grid_2d_list_from(grid=fit.grids.lp, xp=jnp)[-1]))
    if stage >= 2:
        total = total + jnp.sum(raw(fit.blurred_image))
    inversion = fit.inversion
    if stage >= 3:
        total = total + jnp.sum(inversion.data_vector)
    if stage >= 4:
        total = total + jnp.sum(inversion.curvature_matrix)
    if stage >= 5:
        total = total + jnp.sum(inversion.curvature_reg_matrix)
    if stage >= 6:
        total = total + jnp.sum(inversion.reconstruction)
    if stage >= 7:
        total = total + fit.figure_of_merit
    return total


def run_jax_breakdown(dataset_type, variant, cli, args, *, script_file, root) -> Path:
    import autolens as al
    import jax
    import jax.numpy as jnp
    from autofit.jax import register_model

    from _profile_cli import device_info_dict, resources_dict

    from . import timing

    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    sparse = bool(cli.use_sparse_operator)
    print(f"--- {script_file} [{dataset_type}/{variant}] {'sparse' if sparse else 'dense'} ---")
    with timer.section("cell_build"):
        cell = _jax_cell(dataset_type, variant, cli, root)
    instance = cell["instances"][0]
    register_model(cell["model"])
    params_tree, treedef = _flat_params(instance)
    analysis = _analysis(cell, instance, use_jax=True)
    inversion_class = _check_route(analysis, params_tree, sparse, treedef)
    n_repeats = 2 if args.fast else 10

    prefix_per_call: dict[int, float] = {}
    for stage, label in JAX_STAGES.items():
        print(f"\n--- prefix {stage}: {label} ---")

        def prefix(params, stage=stage):
            fit = analysis.fit_from(instance=jax.tree_util.tree_unflatten(treedef, params))
            return _stage_value(fit, stage)

        compiled, _ = timing.jit_profile(
            prefix,
            f"prefix_{stage}",
            params_tree,
            n_repeats=n_repeats,
            timer=timer,
            jit_records=jit_records,
        )
        prefix_per_call[stage] = jit_records[f"prefix_{stage}"]["steady_per_call_s"]
        del compiled
        jax.clear_caches()
        gc.collect()
    steps = timing.split_by_successive_differences(prefix_per_call, JAX_STAGES)

    al_version = al.__version__
    summary = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("jax"),
        "instrument": cell["spec"].instrument,
        "model": variant,
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "configuration": _configuration(
            cell,
            inversion_path="sparse" if sparse else "dense",
            inversion_class=inversion_class,
            sparse_batch_size=int(cli.sparse_batch_size) if sparse else None,
            n_repeats=n_repeats,
            fast=bool(args.fast),
            xla_cpu_workaround=XLA_CPU_WORKAROUND_NOTE,
        ),
        "steps": steps,
        "total_step_by_step": float(sum(steps.values())),
        "prefix_per_call_s": {JAX_STAGES[k]: v for k, v in prefix_per_call.items()},
        "steps_note": STAGE_NOTE,
        "jit_phases": jit_records,
    }
    dict_path = output_path(
        cli,
        dataset_type=dataset_type,
        variant=variant,
        instrument=cell["spec"].instrument,
        engine="jax",
        measurement="breakdown",
        root=root,
    )
    _write(dict_path, summary)
    return dict_path


# --------------------------------------------------------------------------- numba


def _numba_cell(dataset_type, variant, cli, root, args):
    from cell_builders import galaxy_group as gg

    n_cold = 1 if args.fast else int(cli.cold_evals)
    n_instances = (2 + n_cold + 1) if args.fast else max(int(cli.n_instances), 2 + n_cold)
    cell = gg.standard_cell(
        dataset_type, variant, root=root, sparse="cpu", memo=cli.memo, n_instances=n_instances
    )
    return cell, n_cold


def _guard_numba_route(cell) -> str:
    from autoarray.inversion.inversion.imaging_numba.sparse import InversionImagingSparseNumba

    instance = cell["instances"][0]
    fit = _analysis(cell, instance, use_jax=False).fit_from(instance=instance)
    if not isinstance(fit.inversion, InversionImagingSparseNumba):
        raise SystemExit(
            f"expected InversionImagingSparseNumba, got {type(fit.inversion).__name__}: the numba "
            "sparse-operator path is not engaged"
        )
    return type(fit.inversion).__name__


def run_numba_runtime(dataset_type, variant, cli, args, thread_env, *, script_file, root) -> Path:
    import autolens as al
    import numpy as np

    import _production_config as pc
    from _profile_cli import device_info_dict, resources_dict

    print(f"--- {script_file} [{dataset_type}/{variant}] numba sparse ---")
    cell, n_cold = _numba_cell(dataset_type, variant, cli, root, args)
    inversion_class = _guard_numba_route(cell)
    per_eval, values = [], []
    for i, instance in enumerate(cell["instances"]):
        analysis = _analysis(cell, instance, use_jax=False)
        t0 = time.perf_counter()
        value = analysis.log_likelihood_function(instance=instance)
        per_eval.append(time.perf_counter() - t0)
        values.append(float(value))
        print(f"  [{i:>2}] {per_eval[-1] * 1e3:9.2f} ms  log L = {values[-1]:.6f}")
    timing = pc.timing_summary(per_eval, n_cold=n_cold)
    print(
        f"  cold median {timing['cold_eval_median_s']:.6f} s, warm iid median "
        f"{timing['warm_iid_median_s']:.6f} s (n = {timing['n_warm']})"
    )
    al_version = al.__version__
    summary = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "instrument": cell["spec"].instrument,
        "model": variant,
        "configuration": _configuration(
            cell,
            inversion_path="sparse_numba",
            inversion_class=inversion_class,
            use_jax=False,
            n_instances=len(cell["instances"]),
            thread_env=thread_env,
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            fast=bool(args.fast),
        ),
        # Headline key aggregate.py reads: the warm iid MEDIAN (the imaging numba cells' rule).
        "full_pipeline_single_jit": timing["warm_iid_median_s"],
        **timing,
        "first_call_incl_numba_compile_s": per_eval[0],
        "first_call_finite": bool(np.isfinite(values[0])),
        "log_likelihood": values[0] if len(values) == 1 else values[-1],
        "log_likelihood_draw0": values[0],
        "log_likelihood_sequence": values,
    }
    dict_path = output_path(
        cli,
        dataset_type=dataset_type,
        variant=variant,
        instrument=cell["spec"].instrument,
        engine="numba",
        measurement="runtime",
        root=root,
    )
    _write(dict_path, summary)
    # Draw 0 again, after the timed stream: the warm-up value may be the cold-cache hazard.
    instance = cell["instances"][0]
    value = float(
        _analysis(cell, instance, use_jax=False).log_likelihood_function(instance=instance)
    )
    reference = _write_witness_and_figure(
        dict_path,
        cell,
        value=value,
        use_mixed_precision=False,
        autolens_version=al_version,
        cli=cli,
        note="the numba sparse log_likelihood_function value",
    )
    summary["parity_reference_log_evidence"] = reference
    summary["resources"] = resources_dict("numba")
    _write(dict_path, summary)
    return dict_path


def _eager_stage_times(analysis, instance) -> tuple[list[float], float]:
    """Per-stage seconds of one eager fit, each lazily-evaluated property timed on first access."""
    steps = []

    def timed(fn):
        t0 = time.perf_counter()
        out = fn()
        steps.append(time.perf_counter() - t0)
        return out

    fit = timed(lambda: analysis.fit_from(instance=instance))
    timed(lambda: fit.blurred_image)
    inversion = timed(lambda: fit.inversion)
    timed(lambda: inversion.data_vector)
    timed(lambda: inversion.curvature_matrix)
    timed(lambda: inversion.curvature_reg_matrix)
    timed(lambda: inversion.reconstruction)
    timed(lambda: inversion.mapped_reconstructed_data)
    value = timed(lambda: float(fit.figure_of_merit))
    return steps, value


def run_numba_breakdown(dataset_type, variant, cli, args, thread_env, *, script_file, root):
    import autolens as al
    import numpy as np

    from _profile_cli import device_info_dict, resources_dict

    print(f"--- {script_file} [{dataset_type}/{variant}] numba sparse, stage walk ---")
    cell, n_cold = _numba_cell(dataset_type, variant, cli, root, args)
    inversion_class = _guard_numba_route(cell)
    per_instance = []
    for i, instance in enumerate(cell["instances"]):
        steps, value = _eager_stage_times(_analysis(cell, instance, use_jax=False), instance)
        per_instance.append(steps)
        print(f"  [{i:>2}] {sum(steps) * 1e3:9.2f} ms  log L = {value:.6f}")
    warm = np.array(per_instance[1 + n_cold :])
    medians = {label: float(np.median(warm[:, k])) for k, label in enumerate(NUMBA_STAGES)}
    al_version = al.__version__
    summary = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "instrument": cell["spec"].instrument,
        "model": variant,
        "configuration": _configuration(
            cell,
            inversion_path="sparse_numba",
            inversion_class=inversion_class,
            use_jax=False,
            n_instances=len(cell["instances"]),
            n_cold=n_cold,
            thread_env=thread_env,
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            fast=bool(args.fast),
        ),
        "steps": medians,
        "total_step_by_step": float(sum(medians.values())),
        "steps_note": STAGE_NOTE,
        "steps_per_instance_s": per_instance,
        "protocol": (
            "instance 0 discarded (numba compile), the next n_cold excluded, per-stage median "
            "over the remaining warm iid instances"
        ),
    }
    dict_path = output_path(
        cli,
        dataset_type=dataset_type,
        variant=variant,
        instrument=cell["spec"].instrument,
        engine="numba",
        measurement="breakdown",
        root=root,
    )
    _write(dict_path, summary)
    return dict_path


# --------------------------------------------------------------------------- entry point


def main(dataset_type: str, variant: str, measurement: str, engine: str, *, script_file: str):
    """Run one cell script. ``engine`` ``jax`` or ``numba``; numba scripts pin their threads
    (``_production_config.pin_thread_env(1)``) before calling this."""
    root = next(p for p in Path(script_file).resolve().parents if (p / "ruff.toml").exists())
    for path in (root, root / "scripts" / "misc"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from cell_builders import galaxy_group as gg

    from _production_config import observe_thread_env, pin_thread_env
    from _profile_cli import parse_profile_cli

    cli = parse_profile_cli()
    args, _ = _driver_parser().parse_known_args()
    gg.spec_for(dataset_type, cli.instrument)
    thread_env = pin_thread_env(1) if engine == "numba" else observe_thread_env()
    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        print(f"[smoke] {script_file}: imports + module setup OK; exiting.")
        return None
    if engine == "numba":
        if cli.use_sparse_operator:
            raise SystemExit("the numba cells are sparse by construction; drop --sparse")
        import numba

        numba.set_num_threads(1)
        run = run_numba_runtime if measurement == "runtime" else run_numba_breakdown
        return run(dataset_type, variant, cli, args, thread_env, script_file=script_file, root=root)
    _avoid_xla_cpu_deadlock()
    run = run_jax_runtime if measurement == "runtime" else run_jax_breakdown
    return run(dataset_type, variant, cli, args, script_file=script_file, root=root)
