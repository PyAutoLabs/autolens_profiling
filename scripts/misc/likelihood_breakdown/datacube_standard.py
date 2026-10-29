"""Shared drivers for the datacube standard-size cells (likelihood-lifecycle 2f, #475).

A ``datacube/delaunay/<instrument>`` cell is the 2b interferometer standard cell of that
instrument (``interferometer_standard``, ruling R3: Delaunay Hilbert 1250 + 30 zeroed edge points
with a free ``AdaptSplit``, the truth lensed-source adapt image, iid draw 0) fitted to a cube of
:data:`N_CHANNELS` channels (decision D2):

- every channel shares the instrument's ``uv_wavelengths``, noise map and real-space mask, and
  the lens model (the narrow-emission-line regime the workspace datacube example and the
  library's ``shared_preloads`` assume);
- channel ``c`` holds the instrument's visibilities scaled by an emission-line amplitude
  ``a_c`` (:func:`channel_scales`): ``a_0 = 1``, so channel 0 IS the 2b interferometer cell, and
  the data differ per channel, so no channel's work is a copy of another's;
- the likelihood is the production one: ``af.FactorGraphModel`` over one
  ``AnalysisInterferometer(shared_preloads=True)`` per channel, so the lead channel builds the
  mapper and the curvature matrix ``F`` once and every channel reuses them
  (``AnalysisInterferometer.shared_state_from``).

Drivers (the cell scripts call them under ``--standard-size``):

- :func:`run_jax` -- ``scripts/datacube/delaunay/likelihood_runtime.py --standard-size
  [--sparse]``: ``jax.jit`` of the cube log likelihood (sparse W~ or dense NUFFT), lower /
  compile / first call / steady state, its steady median and the vmap per call;
- :func:`run_numba` -- ``.../likelihood_runtime_numba.py --standard-size``: the NumPy / numba
  cube over the preset's iid stream, headline = the packaged cost-model route (R4);
- :func:`run_breakdown` -- ``.../likelihood_breakdown{,_numba}.py --standard-size``: the 2b
  breakdown of the lead channel, projected to the cube by step (channel-invariant steps once,
  the rest ``N`` times), and an eager cube evaluation for the parity witness;
- :func:`cube_reference` / :func:`finalise_parity` -- ``scripts/datacube/delaunay/
  parity_reference.py``: the cube reference from the 2b chunked dense reference (decision D3).

The parity reference (D3). Scaling a channel's visibilities by ``a`` with the noise map, the
adapt image and the regularization unchanged scales the positive-only solution by ``a`` (the
problem ``(F + H) s = a D`` with ``s >= 0`` is solved by ``a s``), so ``chi^2`` and ``s^T H s``
scale by ``a^2`` and the two log determinants and the noise normalisation do not change. The
cube's reference log evidence is therefore, from the terms of the 2b reference of the same
(variant, instrument, instance),

    sum_c -0.5 (a_c^2 (chi^2 + s^T H s) + log det(F + H) - log det H + noise normalisation)

so the visibility-chunked eager NumPy dense reference (2b ruling R6, independent of W~, numba
and JAX) is computed once per instrument, never per channel. At sma it is computed in-process;
above sma it is 2b's cached reference task output (``output/p2b_references/``), and a row
written before it exists carries a ``parity_candidate`` that :func:`finalise_parity` rewrites.
The tolerance is 2b's R7 (1e-8 x |reference| in fp64).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from . import interferometer_standard as istd

VARIANT = "delaunay"
#: Channels per cube (decision D2): the fiducial of the earlier datacube cells (the 34-channel
#: ALMA cube they were sized on). ``--n-channels`` overrides it for a smoke run; a row records
#: the count it ran.
N_CHANNELS = 34
#: The emission line's width in channels, as a fraction of ``N_CHANNELS`` (``a_0 = 1`` at the line
#: peak, ``a_{N-1} ~ exp(-2)``).
LINE_SIGMA_FRACTION = 0.5
#: The numba runtime's default arm (decision D5): the packaged cost-model route only; the forced
#: routes of 2b's R4 are sub-rows there and each costs a full cube here.
DEFAULT_NUMBA_ARMS = "default"
#: What every datacube witness compares.
PARITY_QUANTITY = (
    "the cube log likelihood (sum over channels of log_likelihood_function, the log evidence, "
    "FactorGraphModel with shared_preloads) at the standard cell's iid draw 0, vs the cube "
    "reference from the visibility-chunked eager NumPy dense reference (decision D3)"
)
#: Decision D4: the steps of the 2b breakdown the shared path computes once per cube -- the lead
#: channel's ray-tracing, mapper, mapping matrix ``L`` (the W~ route's sparse triplets of it) and
#: curvature matrix ``F`` (``AnalysisInterferometer.shared_state_from`` hands the mapper and ``F``
#: to every channel). Every other step runs once per channel: ``H``, ``D`` (on the dense route
#: with its transformed mapping matrix ``T``, which ``D = T^T N^-1 d`` needs per channel), the
#: solve, the log determinants and the figure of merit. Matched as lower-case substrings of the
#: step names of both 2b harnesses.
CHANNEL_INVARIANT_STEPS = (
    "ray-trac",
    "ray trac",
    "border relocation",
    "mapper",
    "inversion build",
    "mapping matrix l",
    "sparse triplets",
    "curvature matrix f",
    # The numba harness: "Curvature F + H" is F by the packaged route plus the O(S^2) add of H
    # (H itself is its own per-channel step), and "F marshalling" builds F's index arrays.
    "curvature f",
    "f marshalling",
)


def channel_scales(n_channels: int) -> list[float]:
    """The per-channel visibility scale ``a_c``: a Gaussian emission line peaking at channel 0."""
    n = max(1, int(n_channels))
    sigma = LINE_SIGMA_FRACTION * n
    return [math.exp(-0.5 * (c / sigma) ** 2) for c in range(n)]


def _cube_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument(
        "--n-channels",
        type=int,
        default=N_CHANNELS,
        help=f"Channels in the cube (default {N_CHANNELS}, decision D2).",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="numba runtime: comma-separated arms (default: the packaged route only, D5).",
    )
    return parser


def cube_args() -> argparse.Namespace:
    args, _ = _cube_parser().parse_known_args()
    if args.n_channels < 1:
        raise SystemExit("--n-channels must be at least 1")
    return args


# --------------------------------------------------------------------------- the cube


def channel_datasets(ds, scales: list[float], *, sparse: str | None, batch_size: int, timer):
    """The cube's channel datasets: ``ds.dataset`` with its visibilities scaled by each ``a_c``.

    ``sparse``: ``"jax"`` (the JAX W~ operator), ``"numpy"`` (the NumPy / numba one, W~ from the
    2b preload cache) or None (the dense NUFFT route). The NUFFT precision operator ``W~`` depends
    only on the uv coverage and the noise map, which every channel shares, so it is built once
    and handed to every channel; each channel computes its own dirty image. Channel 0 is
    ``ds.dataset`` itself (``a_0 = 1``). Returns ``(datasets, preload provenance)``.
    """
    import autolens as al
    import numpy as np

    from . import interferometer_pixelized_numpy as numpy_harness

    base = ds.dataset

    def transformer(uv_wavelengths, real_space_mask):
        return al.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            chunk_size=ds.transformer_chunk,
        )

    raw = [base]
    for a in scales[1:]:
        raw.append(
            al.Interferometer(
                data=al.Visibilities(np.asarray(base.data) * a),
                noise_map=base.noise_map,
                uv_wavelengths=base.uv_wavelengths,
                real_space_mask=base.real_space_mask,
                transformer_class=transformer,
            )
        )
    if sparse is None:
        return raw, None
    kwargs = dict(
        method="nufft",
        batch_size=batch_size,
        nufft_chunk_size=ds.transformer_chunk,
        use_jax=sparse == "jax",
    )
    if sparse == "numpy":
        lead, preload = numpy_harness.preload_sparse_dataset(ds, batch_size, True, timer)
    else:
        with timer.section("apply_sparse_operator"):
            lead = base.apply_sparse_operator(**kwargs)
        preload = {"operator_build_s": timer.records[-1][1], "cache": "off"}
    precision = np.asarray(lead.sparse_operator.nufft_precision_operator)
    out = [lead]
    with timer.section("apply_sparse_operator_channels"):
        for dataset in raw[1:]:
            out.append(dataset.apply_sparse_operator(nufft_precision_operator=precision, **kwargs))
    preload = {
        **preload,
        "channels_s": timer.records[-1][1],
        "note": "W~ built once (shared uv coverage and noise map) and handed to every channel; "
        "each channel computes its own dirty image. One-off, never a per-evaluation step.",
    }
    return out, preload


def factor_graph(datasets, *, cell, adapt_images, settings, use_jax: bool):
    """The production cube: ``FactorGraphModel`` of one shared-preloads analysis per channel."""
    import autofit as af
    import autolens as al

    analyses = [
        al.AnalysisInterferometer(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=settings,
            use_jax=use_jax,
            shared_preloads=True,
        )
        for dataset in datasets
    ]
    factors = [af.AnalysisFactor(prior_model=cell["model"], analysis=a) for a in analyses]
    return af.FactorGraphModel(*factors, use_jax=use_jax), analyses


def cube_log_likelihood(graph, n_channels: int, instance):
    """The cube log likelihood of one shared instance: every factor fits the same lens model."""
    return graph.log_likelihood_function([instance] * n_channels)


def parity_candidate(*, log_likelihood, cell, scales, use_mixed_precision, instrument) -> dict:
    return {
        **istd.parity_candidate(
            log_likelihood=log_likelihood,
            cell=cell,
            use_mixed_precision=use_mixed_precision,
            variant=VARIANT,
            instrument=instrument,
        ),
        "quantity": PARITY_QUANTITY,
        "n_channels": len(scales),
        "channel_scales": [float(a) for a in scales],
    }


def cube_configuration(ds, cell, scales, **extra) -> dict:
    return istd._configuration(
        ds,
        cell,
        n_channels=len(scales),
        channel_scales=[float(a) for a in scales],
        channel_line_sigma_fraction=LINE_SIGMA_FRACTION,
        shared_preloads=True,
        cube_definition=(
            "decision D2: the 2b interferometer standard cell's dataset in every channel (shared "
            "uv_wavelengths, noise map, mask and lens model), channel c's visibilities scaled by "
            "a_c (Gaussian emission line, a_0 = 1); FactorGraphModel of "
            "AnalysisInterferometer(shared_preloads=True)"
        ),
        **extra,
    )


# --------------------------------------------------------------------------- the reference


def cube_reference_from(reference: dict, scales: list[float]) -> dict:
    """The cube's reference log evidence from one channel's reference terms (decision D3)."""
    chi_reg = float(reference["chi_squared"]) + float(reference["regularization_term"])
    fixed = (
        float(reference["log_det_curvature_reg"])
        - float(reference["log_det_regularization"])
        + float(reference["noise_normalization"])
    )
    per_channel = [-0.5 * (a * a * chi_reg + fixed) for a in scales]
    return {
        "log_evidence": float(sum(per_channel)),
        "log_evidence_per_channel": per_channel,
        "n_channels": len(scales),
        "channel_scales": [float(a) for a in scales],
        "channel_reference_log_evidence": float(reference["log_evidence"]),
        "method": (
            "sum over channels of -0.5 (a_c^2 (chi^2 + s^T H s) + log det(F + H) - log det H + "
            "noise normalisation), the terms from the channel reference (decision D3)"
        ),
    }


def cube_reference(ds, cell, adapt_images, scales, *, profiling_root: Path):
    """``(cube reference dict or None, source text, why)`` for this cube.

    The 2b reference task's cache when it describes this (variant, instrument, instance); at sma
    the chunked dense reference computed in-process; otherwise None (``not_assessed`` until
    :func:`finalise_parity`).
    """
    import _production_config as pc
    from _profile_cli import provenance_dict

    revisions = provenance_dict().get("library_revisions")
    doc, why = pc.load_interferometer_reference(
        profiling_root,
        VARIANT,
        ds.instrument,
        dataset_path=ds.dataset_path,
        library_revisions=revisions,
        unit_vector=cell["unit_vector"],
    )
    if doc is not None:
        path = pc.interferometer_reference_path(profiling_root, VARIANT, ds.instrument)
        source = f"cube reference from the cached 2b reference task {path.name} (D3)"
        return cube_reference_from(doc, scales), source, "ok"
    compute = istd._in_process_reference(ds, cell, adapt_images, memo="off")
    if compute is None:
        return None, None, why
    try:
        computed = compute()
    except Exception as exc:  # noqa: BLE001 — a failed reference must not cost the row
        return None, None, f"in-process reference failed: {type(exc).__name__}: {exc}"
    source = (
        "cube reference from the visibility-chunked eager NumPy dense reference computed in the "
        f"same process ({computed['n_vis_chunks']} chunk(s)), D3"
    )
    return cube_reference_from(computed, scales), source, "ok"


def write_parity(cell_dir: Path, *, candidate: dict, reference, source, why, autolens_version):
    from _profile_cli import write_parity_json

    return write_parity_json(
        Path(cell_dir),
        log_likelihood=candidate["log_likelihood"],
        reference_log_likelihood=float("nan") if reference is None else reference["log_evidence"],
        use_mixed_precision=bool(candidate["use_mixed_precision"]),
        reference_source=source or "cube reference from the 2b chunked dense reference (pending)",
        autolens_version=autolens_version,
        source_pixels=int(candidate["source_pixels"]),
        source_pixels_requested=candidate.get("source_pixels_requested"),
        quantity=candidate["quantity"],
        # 2b's R7 tolerance (1e-8 x |reference| in fp64), decision D3.
        dataset_type="interferometer",
        pending_reason=None if reference is not None else f"reference pending: {why}",
    )


def finalise_parity(instrument: str, *, results_root: Path, reference: dict | None) -> dict:
    """Rewrite ``parity.json`` of every datacube row of ``instrument`` under ``results_root``.

    ``reference`` is the 2b channel reference document (``interferometer_standard.
    reference_task``). A row is finalised only when its ``parity_candidate`` was taken at the
    reference's instance on the reference's library revisions.
    """
    from _profile_cli import write_parity_json

    base = Path(results_root) / "results" / "datacube" / VARIANT / instrument
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
        if not candidate or "channel_scales" not in candidate:
            out["skipped"].append({"row": rel, "reason": "no cube parity_candidate"})
            continue
        revisions = ((row.get("device") or {}).get("provenance") or {}).get("library_revisions")
        if candidate.get("instance_unit_vector") != reference.get("instance_unit_vector"):
            out["failed"].append({"row": rel, "reason": "different instance"})
            continue
        if revisions != reference.get("library_revisions"):
            out["failed"].append({"row": rel, "reason": "different library revisions"})
            continue
        cube = cube_reference_from(reference, candidate["channel_scales"])
        written = write_parity_json(
            row_path.parent,
            log_likelihood=candidate["log_likelihood"],
            reference_log_likelihood=cube["log_evidence"],
            use_mixed_precision=bool(candidate.get("use_mixed_precision")),
            reference_source=(
                f"cube reference (D3) from the cached reference task "
                f"interferometer_{VARIANT}_{instrument}.json ({reference.get('method_short')})"
            ),
            autolens_version=str(row.get("autolens_version")),
            source_pixels=int(candidate["source_pixels"]),
            source_pixels_requested=candidate.get("source_pixels_requested"),
            quantity=candidate["quantity"],
            dataset_type="interferometer",
        )
        (out["finalised"] if written else out["failed"]).append({"row": rel})
    return out


# --------------------------------------------------------------------------- JAX runtime


def run_jax(cli, *, script_file: str, profiling_root: Path) -> Path:
    """The JAX runtime row of the standard-size datacube cell (sparse W~ or dense NUFFT)."""
    import autolens as al
    import jax
    import jax.numpy as jnp
    from autofit.jax import register_model

    from _profile_cli import device_info_dict, resolve_output_paths, resources_dict

    from . import timing

    istd._require_standard_size(cli, script_file)
    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True (fp32 truncation).")
    args = cube_args()
    instrument = cli.instrument or "sma"
    sparse = bool(cli.use_sparse_operator)
    scales = channel_scales(args.n_channels)
    n = len(scales)
    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    load_start = list(os.getloadavg())
    print(
        f"--- {script_file} [datacube {VARIANT} / {instrument}] standard size, {n} channels, "
        f"{'sparse W~' if sparse else 'dense NUFFT'} ---"
    )
    inputs = istd.load_standard_inputs(VARIANT, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    import _production_config as pc

    cell = pc.standard_interferometer_cell(
        VARIANT,
        instrument,
        dataset=ds.dataset,
        adapt_image=adapt_image,
        mask_radius=ds.mask_radius,
        memo=cli.memo,
        use_mixed_precision=cli.use_mixed_precision,
    )
    model, instance, settings = cell["model"], cell["instance"], cell["settings"]
    datasets, preload = channel_datasets(
        ds, scales, sparse="jax" if sparse else None, batch_size=cli.sparse_batch_size, timer=timer
    )
    register_model(model)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
    adapt_images = pc.adapt_images_for(instance, cell["image_plane_mesh_grid"], adapt_image)
    graph, _analyses = factor_graph(
        datasets, cell=cell, adapt_images=adapt_images, settings=settings, use_jax=True
    )
    print(
        f"  preset {cell['preset'].name}: {cell['n_source_pixels']} source pixels, "
        f"{model.total_free_parameters} free parameters, {ds.n_vis} visibilities x {n} channels"
    )

    trace_info: dict[str, Any] = {}

    def traced_fit(params):
        fit = al.FitInterferometer(
            dataset=datasets[0],
            tracer=al.Tracer(galaxies=list(params.galaxies), fields=[params.fields]),
            adapt_images=pc.adapt_images_for(params, cell["image_plane_mesh_grid"], adapt_image),
            settings=settings,
            xp=jnp,
        )
        trace_info["inversion_class"] = type(fit.inversion).__name__
        return fit.figure_of_merit

    jax.eval_shape(traced_fit, params_tree)
    expected = "InversionInterferometerSparse" if sparse else "InversionInterferometerMapping"
    print(f"  inversion class (traced, lead channel): {trace_info.get('inversion_class')}")
    if trace_info.get("inversion_class") != expected:
        raise SystemExit(f"expected {expected} on this setup, traced {trace_info}")

    on_cpu = jax.default_backend() == "cpu"
    n_repeats = 3 if (on_cpu and ds.n_vis * n > istd.LARGE_VIS) else 10

    def full_cube(params):
        return cube_log_likelihood(graph, n, params)

    print("\n--- Full cube: jax.jit(FactorGraphModel.log_likelihood_function) ---")
    compiled, result = timing.jit_profile(
        full_cube,
        "full_pipeline_cube",
        params_tree,
        n_repeats=n_repeats,
        timer=timer,
        jit_records=jit_records,
    )
    per_call = jit_records["full_pipeline_cube"]["steady_per_call_s"]
    median = timing.headline_steady_median(
        compiled,
        params_tree,
        block_mean_s=per_call,
        prefix="full_pipeline_cube_single_jit",
        block_mean_is=timing.single_jit_block_is(n_repeats),
    )
    cube_jit = float(result)
    print(f"  cube log likelihood (JIT) = {cube_jit!r}")
    del compiled
    jax.clear_caches()
    gc.collect()

    vmap, vmap_attempts = _vmap_cube(
        full_cube, params_tree, n_repeats=n_repeats, timer=timer, jit_records=jit_records
    )
    jax.clear_caches()
    gc.collect()

    al_version = al.__version__
    candidate = parity_candidate(
        log_likelihood=cube_jit,
        cell=cell,
        scales=scales,
        use_mixed_precision=cli.use_mixed_precision,
        instrument=instrument,
    )
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("jax"),
        "instrument": instrument,
        "model": VARIANT,
        "transformer": "TransformerNUFFT",
        "use_mixed_precision": bool(cli.use_mixed_precision),
        "n_channels": n,
        "configuration": cube_configuration(
            ds,
            cell,
            scales,
            inversion_path="sparse" if sparse else "dense",
            inversion_class=trace_info["inversion_class"],
            sparse_operator_method="nufft" if sparse else None,
            sparse_batch_size=int(cli.sparse_batch_size) if sparse else None,
            preload=preload,
            n_repeats=n_repeats,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        "cube_log_evidence_jit": cube_jit,
        "full_pipeline_cube_single_jit": per_call,
        **median,
        "per_channel_equivalent_s": per_call / n,
        "vmap": vmap
        if vmap is not None
        else {"status": "oom", "attempts": vmap_attempts, "note": "every vmap batch OOM'd"},
        "jit_phases": jit_records,
        "parity_candidate": candidate,
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root / "results" / "runtime" / "datacube" / VARIANT,
        default_basename=f"{VARIANT}_likelihood_summary_{instrument}_v{al_version}",
        hints={**summary, "transformer": "TransformerNUFFT"},
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  Row written: {dict_path}")

    _finish(
        summary,
        dict_path,
        cli=cli,
        ds=ds,
        cell=cell,
        datasets=datasets,
        adapt_images=adapt_images,
        scales=scales,
        candidate=candidate,
        profiling_root=profiling_root,
        engine="jax",
    )
    return dict_path


def _vmap_cube(full_cube, params_tree, *, n_repeats, timer, jit_records):
    """2b's vmap with its OOM fallback (batch 3, then 1): the batched cost a sampler pays."""
    import jax
    import jax.numpy as jnp

    from . import timing

    attempts = []
    for nb in (3, 1):
        batched = jax.tree_util.tree_map(
            lambda leaf, n=nb: jnp.broadcast_to(leaf, (n, *leaf.shape)), params_tree
        )
        try:
            per_call = timing.vmap_profile(
                full_cube,
                "full_pipeline_cube",
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
        return {
            "batch_size": nb,
            "batch_size_source": "fixed (3, OOM fallback 1)",
            "batch_time": per_call * nb,
            "per_call": per_call,
            "instances": "draw 0 broadcast to the batch (the runtime cells' convention)",
            "attempts": attempts,
        }, attempts
    return None, attempts


def _finish(
    summary,
    dict_path: Path,
    *,
    cli,
    ds,
    cell,
    datasets,
    adapt_images,
    scales,
    candidate,
    profiling_root,
    engine,
):
    """After every timed region: the eager lead-channel fit (fit.png), the witness, resources."""
    import autolens as al
    import numpy as np

    from _profile_cli import resources_dict

    instance = cell["instance"]

    def eager_fit():
        return al.FitInterferometer(
            dataset=datasets[0],
            tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
            adapt_images=adapt_images,
            settings=cell["settings"],
            xp=np,
        )

    try:
        fit = eager_fit()
        summary["lead_channel_figure_of_merit_eager"] = float(fit.figure_of_merit)
        summary["lead_channel_eager_inversion_class"] = type(fit.inversion).__name__
    except Exception as exc:  # noqa: BLE001
        fit = None
        summary["eager"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
    reference, source, why = cube_reference(
        ds, cell, adapt_images, scales, profiling_root=profiling_root
    )
    write_parity(
        dict_path.parent,
        candidate=candidate,
        reference=reference,
        source=source,
        why=why,
        autolens_version=summary["autolens_version"],
    )
    if reference is not None:
        summary["parity_reference"] = {
            k: reference.get(k)
            for k in ("log_evidence", "channel_reference_log_evidence", "method")
        }
    summary["resources"] = resources_dict(engine)
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    if fit is not None:
        istd._render_fit(lambda: fit, dict_path.parent, cli)


# --------------------------------------------------------------------------- numba runtime


def run_numba(cli, thread_env: dict, *, script_file: str, profiling_root: Path) -> Path:
    """The numba / NumPy runtime row of the standard-size datacube cell (R4, decision D5)."""
    import autolens as al
    import numba
    from autoarray.inversion.inversion.interferometer_numba.sparse import (
        InversionInterferometerSparseNumba,
    )

    import _production_config as pc
    from _profile_cli import device_info_dict, resolve_output_paths, resources_dict

    from . import interferometer_pixelized_numpy as numpy_harness
    from . import timing

    istd._require_standard_size(cli, script_file)
    args = cube_args()
    arms = [a.strip() for a in (args.arms or DEFAULT_NUMBA_ARMS).split(",") if a.strip()]
    if not arms or any(a not in istd.NUMBA_ARMS for a in arms):
        raise SystemExit(f"--arms {args.arms!r}: choose from {tuple(istd.NUMBA_ARMS)}")
    headline = "default" if "default" in arms else arms[0]
    numba.set_num_threads(1)
    instrument = cli.instrument or "sma"
    scales = channel_scales(args.n_channels)
    n = len(scales)
    timer = timing.Timer()
    load_start = list(os.getloadavg())
    print(
        f"--- {script_file} [datacube {VARIANT} / {instrument}] standard size, {n} channels, "
        f"arms {arms} ---"
    )
    inputs = istd.load_standard_inputs(VARIANT, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    datasets, preload = channel_datasets(
        ds, scales, sparse="numpy", batch_size=cli.sparse_batch_size, timer=timer
    )
    n_instances = max(int(cli.n_instances), 2 + int(cli.cold_evals))
    cell = pc.standard_interferometer_cell(
        VARIANT,
        instrument,
        dataset=ds.dataset,
        adapt_image=adapt_image,
        mask_radius=ds.mask_radius,
        memo=cli.memo,
        n_instances=n_instances,
    )
    grid = cell["image_plane_mesh_grid"]
    settings = {
        arm: pc.analysis_settings(cell["preset"], cli.memo, **istd.NUMBA_ARMS[arm])[0]
        for arm in arms
    }
    numba_target = [(InversionInterferometerSparseNumba, "curvature_matrix_diag")]

    per_eval: dict[str, list[float]] = {arm: [] for arm in arms}
    log_likelihoods: dict[str, list[float]] = {arm: [] for arm in arms}
    routes: dict[str, list[str]] = {arm: [] for arm in arms}
    print(f"\n--- iid stream: {n_instances} instances (0 = warm-up), arms {arms} ---")
    for i, instance in enumerate(cell["instances"]):
        order = arms if i % 2 == 0 else list(reversed(arms))
        line = []
        for arm in order:
            graph, _ = factor_graph(
                datasets,
                cell=cell,
                adapt_images=pc.adapt_images_for(instance, grid, adapt_image),
                settings=settings[arm],
                use_jax=False,
            )

            def call(graph=graph, instance=instance):
                t0 = time.perf_counter()
                value = cube_log_likelihood(graph, n, instance)
                return time.perf_counter() - t0, float(value)

            (dt, value), counts = numpy_harness.count_evaluations(numba_target, call)
            route = "numba" if counts["curvature_matrix_diag"] else "numpy_fft"
            per_eval[arm].append(dt)
            log_likelihoods[arm].append(value)
            routes[arm].append(route)
            line.append(f"{arm} {dt * 1e3:.1f} ms ({route})")
            del graph
        gc.collect()
        print(f"  [{i:>2}] " + "  ".join(line))

    for arm, forced in istd.NUMBA_ARM_CLASSES.items():
        if arm in arms:
            want = "numba" if forced.endswith("Numba") else "numpy_fft"
            if any(r != want for r in routes[arm]):
                raise SystemExit(f"arm {arm!r} took routes {routes[arm]}, expected {want}")

    arm_records = {}
    for arm in arms:
        summary_arm = pc.timing_summary(per_eval[arm], n_cold=int(cli.cold_evals))
        arm_records[arm] = {
            **summary_arm,
            "settings": dict(istd.NUMBA_ARMS[arm]) or "packaged (cost model)",
            "route_per_instance": routes[arm],
            "route_counts": {r: routes[arm].count(r) for r in sorted(set(routes[arm]))},
            "log_likelihood_per_instance": log_likelihoods[arm],
        }
    head = arm_records[headline]
    al_version = al.__version__
    candidate = parity_candidate(
        log_likelihood=log_likelihoods[headline][0],
        cell=cell,
        scales=scales,
        use_mixed_precision=False,
        instrument=instrument,
    )
    operator = datasets[0].sparse_operator
    summary: dict[str, Any] = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "resources": resources_dict("numba"),
        "instrument": instrument,
        "model": f"{VARIANT}_numba",
        "transformer": "TransformerNUFFT",
        "n_channels": n,
        "configuration": cube_configuration(
            ds,
            cell,
            scales,
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
            preload=preload,
            host_load_avg_start=load_start,
            host_load_avg_end=list(os.getloadavg()),
        ),
        # The cube headline: the headline arm's warm iid median (the catalogue's
        # cube_single_jit_median; numba has no JIT block, the name is the schema's).
        "full_pipeline_cube_single_jit": head["warm_iid_median_s"],
        "full_pipeline_cube_single_jit_median": head["warm_iid_median_s"],
        "per_channel_equivalent_s": head["warm_iid_median_s"] / n,
        **{k: v for k, v in head.items() if k != "settings"},
        "arms": arm_records,
        "cube_log_likelihood": log_likelihoods[headline][-1],
        "parity_candidate": candidate,
    }
    dict_path, _chart = resolve_output_paths(
        cli,
        default_dir=profiling_root / "results" / "runtime" / "datacube" / f"{VARIANT}_numba",
        default_basename=f"{VARIANT}_numba_likelihood_summary_{instrument}_v{al_version}",
        hints=summary,
    )
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    print(
        f"\n  headline ({headline}) warm iid median {head['warm_iid_median_s']:.6f} s per cube; "
        f"row {dict_path}"
    )
    _finish(
        summary,
        dict_path,
        cli=cli,
        ds=ds,
        cell=cell,
        datasets=datasets,
        adapt_images=pc.adapt_images_for(cell["instance"], grid, adapt_image),
        scales=scales,
        candidate=candidate,
        profiling_root=profiling_root,
        engine="numba",
    )
    return dict_path


# --------------------------------------------------------------------------- breakdown


def step_is_invariant(name: str) -> bool:
    """Whether a 2b breakdown step is computed once per cube under ``shared_preloads``."""
    low = name.lower()
    return any(key in low for key in CHANNEL_INVARIANT_STEPS)


def cube_projection(row: dict, n_channels: int) -> dict:
    """The cube block of a breakdown row: invariant steps once, the rest ``n_channels`` times."""
    steps = row.get("steps_median") or row.get("steps") or {}
    invariant = {
        k: v for k, v in steps.items() if isinstance(v, (int, float)) and step_is_invariant(k)
    }
    variant = {
        k: v for k, v in steps.items() if isinstance(v, (int, float)) and not step_is_invariant(k)
    }
    inv_s, var_s = float(sum(invariant.values())), float(sum(variant.values()))
    return {
        "n_channels": int(n_channels),
        "steps_source": "steps_median" if row.get("steps_median") else "steps",
        "channel_invariant_steps": sorted(invariant),
        "channel_variant_steps": sorted(variant),
        "channel_invariant_s": inv_s,
        "channel_variant_s": var_s,
        "total_step_by_step_cube": inv_s + n_channels * var_s,
        "rule": (
            "cube = sum(channel-invariant steps) + n_channels x sum(per-channel steps); the "
            "invariant steps are those the shared_preloads path builds once on the lead channel "
            "(mapper, mapping matrix, curvature matrix F), every other step runs per channel"
        ),
    }


def run_breakdown(cli, cell_args, spec, *, numba: bool, thread_env, script_file, profiling_root):
    """The datacube breakdown row: the 2b breakdown of the lead channel (which IS the 2b cell),
    projected to the cube, then an eager cube evaluation for the parity witness."""
    import autolens as al

    import _production_config as pc
    from _profile_cli import resources_dict

    from . import interferometer_pixelized as jax_harness
    from . import interferometer_pixelized_numpy as numpy_harness
    from . import timing

    istd._require_standard_size(cli, script_file)
    args = cube_args()
    scales = channel_scales(args.n_channels)
    n = len(scales)
    if numba:
        numpy_harness.run(spec, cli, cell_args, script_file, profiling_root, thread_env)
    else:
        jax_harness.run(spec, cli, cell_args, script_file, profiling_root)
    dict_path = Path(cli.output_path)
    row = json.loads(dict_path.read_text())
    projection = cube_projection(row, n)
    # The cube total is the headline (the catalogue's cube_component_total): the lead channel's
    # own total moves aside so it is not read as the cube's.
    if "total_step_by_step" in row:
        row["total_step_by_step_lead_channel"] = row.pop("total_step_by_step")
    row["total_step_by_step_cube"] = projection["total_step_by_step_cube"]
    row["datacube"] = projection
    row["n_channels"] = n

    # The cube log likelihood, eagerly (NumPy, the cell's route), for the witness.
    timer = timing.Timer()
    instrument = cli.instrument or "sma"
    inputs = istd.load_standard_inputs(VARIANT, instrument, profiling_root, timer)
    ds, adapt_image = inputs["ds"], inputs["adapt_image"]
    cell = pc.standard_interferometer_cell(
        VARIANT, instrument, dataset=ds.dataset, adapt_image=adapt_image, mask_radius=ds.mask_radius
    )
    datasets, _preload = channel_datasets(
        ds,
        scales,
        sparse="numpy"
        if numba or row.get("configuration", {}).get("inversion_path") != "dense"
        else None,
        batch_size=cli.sparse_batch_size,
        timer=timer,
    )
    adapt_images = pc.adapt_images_for(cell["instance"], cell["image_plane_mesh_grid"], adapt_image)
    graph, _ = factor_graph(
        datasets, cell=cell, adapt_images=adapt_images, settings=cell["settings"], use_jax=False
    )
    with timer.section("eager_cube"):
        cube_eager = float(cube_log_likelihood(graph, n, cell["instance"]))
    row["cube_log_likelihood_eager"] = cube_eager
    row["cube_eager_s"] = timer.records[-1][1]
    row["configuration"] = {
        **row.get("configuration", {}),
        **{
            k: v
            for k, v in cube_configuration(ds, cell, scales).items()
            if k
            in (
                "n_channels",
                "channel_scales",
                "channel_line_sigma_fraction",
                "shared_preloads",
                "cube_definition",
            )
        },
    }
    candidate = parity_candidate(
        log_likelihood=cube_eager,
        cell=cell,
        scales=scales,
        use_mixed_precision=bool(cli.use_mixed_precision),
        instrument=instrument,
    )
    candidate["quantity"] = (
        PARITY_QUANTITY + "; the breakdown's cube value is an eager NumPy evaluation of the cube "
        "on the cell's route, after the lead channel's timed steps"
    )
    row["parity_candidate"] = candidate
    reference, source, why = cube_reference(
        ds, cell, adapt_images, scales, profiling_root=profiling_root
    )
    write_parity(
        dict_path.parent,
        candidate=candidate,
        reference=reference,
        source=source,
        why=why,
        autolens_version=str(row.get("autolens_version", al.__version__)),
    )
    if reference is not None:
        row["parity_reference"] = {
            k: reference.get(k)
            for k in ("log_evidence", "channel_reference_log_evidence", "method")
        }
    row["resources"] = resources_dict("numba" if numba else "jax")
    dict_path.write_text(json.dumps(row, indent=2, default=float))
    print(
        f"\n  cube projection: invariant {projection['channel_invariant_s']:.6f} s + {n} x "
        f"{projection['channel_variant_s']:.6f} s = {projection['total_step_by_step_cube']:.6f} s"
    )
    return dict_path
