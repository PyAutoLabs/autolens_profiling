"""
JAX Profiling: MGE Interferometer Likelihood — Per-Step Breakdown
==================================================================

Decomposes the JAX likelihood of an interferometer dataset whose source is a
20-Gaussian multi-Gaussian expansion (MGE) into its pipeline steps, and
JIT-profiles each one. It is the **breakdown** counterpart to
``likelihood_runtime/mge.py`` (same dataset, lens and source), which measures
only the fused full-pipeline cost (autolens_profiling#308, interferometer
likelihood campaign phase 1/3).

Steps, in library order
-----------------------

1. Ray-trace the real-space grid to the source plane.
2. Mapping matrix: the 20 un-operated Gaussian images on the traced grid
   (``LightProfileLinearObjFuncList.mapping_matrix``), shape ``(M_pix, 20)``.
3. Transformed mapping matrix: the NUFFT (or DFT) of those 20 columns,
   ``transform_mapping_matrix``, shape ``(N_vis, 20)`` complex.
4. Data vector ``D`` (visibility space, real + imaginary).
5. Curvature matrix ``F`` (dense ``Tᵀ N⁻¹ T``, real + imaginary, plus the
   no-regularization diagonal floor).
6. Positive-only reconstruction: the library's JAX PDIP NNLS, on the
   preconditioning the inversion picks for a mapper-less system (``raw`` since
   PyAutoArray#571). The iteration count and convergence flag come back through
   the solver's ``stats`` out-dict.
7. Log-det terms: **structurally zero here, not timed.** An MGE-only inversion
   carries no regularization, so ``regularization_term``,
   ``log_det_curvature_reg_matrix_term`` and ``log_det_regularization_matrix_term``
   all return 0.0 and the figure of merit is the log likelihood.
8. Fast chi-squared and figure of merit — ``inversion.fast_chi_squared``
   (``sᵀ F s - 2 sᵀ D + dᵀ N⁻¹ d``), which is what ``FitInterferometer.
   figure_of_merit`` / ``AnalysisInterferometer`` return: the production path
   never forms the mapped visibilities. Because ``F`` there carries the
   no-regularization diagonal floor, the figure of merit sits
   ``0.5 * floor * |s|²`` nats below ``FitInterferometer.log_likelihood``
   (recorded as ``diagonal_floor_bias_nats``). The residual-visibility form is
   timed as an overlapping sub-row and checked against ``log_likelihood``.

Steps 1-3 are the params-driven inversion setup. They are timed as nested
prefixes (``params -> traced grid``, ``params -> mapping matrix``,
``params -> transformed mapping matrix``) and attributed by successive
differences (``timing.split_by_successive_differences``; recorded as
``setup_split``). Standalone ray-trace and transform rows, which isolate the
same work from the params plumbing, are recorded as overlapping sub-rows.
Steps 4-8 are timed standalone on the arrays the library produced.

The model and why it has no lens light
--------------------------------------

Isothermal + ExternalShear with ``GaussianPrior(mean=truth, sigma=small)``, so
the prior median — the point every step is evaluated at — is the simulator
truth. The mass is kept free rather than hard-fixed: with every mass parameter
a Python constant, ``params -> traced grid`` is a closed-over constant that XLA
may fold away, and the ray-trace prefix would time nothing. The source is
``al.model_util.mge_model_from(total_gaussians=20)`` centred on the simulator's
(0.1, 0.1), matching the imaging MGE-20 cell and the runtime cell. There is no
lens light: ``simulators/interferometer.py`` puts no lens emission in the
visibilities, so a lens-light MGE would fit nothing.

The W~ arm (``--w-tilde``)
--------------------------

An MGE-only fit always takes the dense path: ``inversion/factory.py`` switches
the sparse operator off when every linear object is an
``AbstractLinearObjFuncList``. The W~ curvature helpers that the mixed
mapper + MGE inversion already uses (``InterferometerSparseOperator.
curvature_matrix_func_list_from`` / ``operated_matrix_slim_from``,
``interferometer/sparse.py``) can nevertheless be driven directly on this
cell's ``(M_pix, 20)`` mapping matrix. That is what this arm does, as a
measurement only, with no library change:

- ``F~ = Bᵀ W~ B`` through the FFT operator, checked against the dense ``F``;
- ``D~ = Bᵀ d~`` from the cached dirty image;
- the figure of merit from the same ``fast_chi_squared`` identity the library
  uses (``chi² = dᵀ N⁻¹ d - 2 sᵀ D + sᵀ F s``) with the per-dataset constants
  hoisted, so no transformed mapping matrix is formed at all; checked against
  ``FitInterferometer.figure_of_merit`` (floored ``F~``) and ``log_likelihood``
  (raw ``F~``).

None of the W~ per-call steps touches the visibilities, so their cost should
not grow with N_vis. The dense-vs-W~ gap is recorded per instrument as
``w_tilde_steps`` / ``w_tilde``. The one-off operator build (precision operator
+ dirty image, two eager type-1 NUFFTs) is chunked over visibilities above the
eager threshold, and its wall time is recorded as ``w_tilde.operator_build_s``.

Memory and repeats at alma scale
--------------------------------

Above ``EAGER_REFERENCE_MAX_VIS`` (200k) visibilities the eager ``xp=np``
reference is replaced by a ``jax.jit`` of the same ``FitInterferometer``: the
op-by-op nufftax type-2 gather would materialise ``(20, N_vis, 169)``
complex128 (54 GB at alma), whereas under ``jax.jit`` XLA fuses it (compiled
temp 2.1 GB at alma, 3.1 GB at alma_high; peak RSS ~4 GB at alma). On CPU at
that scale the steady-state repeat count defaults to 2 (one NUFFT call is
~40 s at alma) and ``--vmap-batch`` is skipped, both recorded in the JSON.

Output
------

JSON + PNG under ``results/breakdown/interferometer/``:

- NUFFT, single config: ``mge_breakdown_{instrument}_v{al_version}``.
- DFT (``--use-dft``): ``mge_dft_breakdown_{instrument}_v{al_version}``; the
  transformer rides in the script token, as the ``*_numba_*`` cells' variants
  do, so ``build_readme.py``'s ``ARTIFACT_RE`` reads the instrument correctly.
- Config-tagged (``--config-name hpc_a100_fp64`` etc.): ``mge_<config>.json``.
  ``CONFIG_TAGGED_RE`` accepts no suffix after the config name, and the
  instrument lives only in the payload, so instruments are kept apart by
  directory: alma writes to ``interferometer/`` itself (the phase-B witness,
  ``mge_hpc_a100_fp64.json``), every other instrument to
  ``interferometer/<instrument>/``. The dashboard reads the instrument from the
  payload either way.
"""

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


import argparse
import json
import os
import resource
import sys
import traceback
from pathlib import Path

import autoarray as aa
import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(_profiling_root()))
# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import sys as _smoke_sys

# Shared breakdown harness. Imported *before* the smoke short-circuit so the CI
# import smoke covers the package too.
from likelihood_breakdown import timing  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from autoarray.inversion.inversion import inversion_util  # noqa: E402
from autoarray.inversion.inversion.interferometer import (  # noqa: E402
    inversion_interferometer_util,
)
from simulators.interferometer import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    resolve_output_paths,
)
from instruments.interferometer import transformer_chunk_size_for  # noqa: E402

_cli = parse_profile_cli()

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument(
    "--use-dft",
    action="store_true",
    help="Use TransformerDFT instead of TransformerNUFFT (sma-scale only).",
)
_cell_parser.add_argument(
    "--w-tilde",
    action="store_true",
    help="Also time the func-list W~ curvature path (measurement-only prototype).",
)
_cell_parser.add_argument(
    "--vmap-batch",
    type=int,
    default=None,
    help="Re-time the setup prefixes and the full pipeline under jax.vmap at this batch.",
)
_cell_parser.add_argument(
    "--n-repeats",
    type=int,
    default=None,
    help=(
        "Steady-state repeats per timed JIT. Default: 10, or 2 on CPU above "
        "EAGER_REFERENCE_MAX_VIS visibilities (one NUFFT call is ~1 min there)."
    ),
)
_cell_args = _cli.parse_cell_args(_cell_parser)

USE_DFT = bool(_cell_args.use_dft)
USE_W_TILDE = bool(_cell_args.w_tilde)
_vmap_batch = _cell_args.vmap_batch

instrument = _cli.instrument or "sma"  # --instrument overrides (default sma)
total_gaussians = 20  # matches the imaging MGE-20 cell and likelihood_runtime/mge.py

N_REPEATS = None  # resolved once the visibility count is known

# ---------------------------------------------------------------------------
# Harness binding
# ---------------------------------------------------------------------------

# Host load at start/end, recorded in the JSON: a shared laptop's timings are
# only comparable when the load says the core was not contended.
_load_avg_start = list(os.getloadavg())

timer = timing.Timer()
block = timing.block
likelihood_steps = []  # (label, per_call_seconds) — the summed rows
sub_rows: dict[str, float] = {}  # overlapping rows, never summed
jit_records: dict[str, dict] = {}  # {label: {lower_s, compile_s, first_call_s, ...}}


def jit_profile(func, label, *args, n_repeats=None):
    """Cell-local binding of ``timing.jit_profile`` (this cell's timer/records)."""
    return timing.jit_profile(
        func,
        label,
        *args,
        n_repeats=n_repeats if n_repeats is not None else N_REPEATS,
        timer=timer,
        jit_records=jit_records,
    )


def _per_call(label):
    return jit_records[label]["steady_per_call_s"]


def _peak_rss_mb() -> float:
    """Process high-water RSS (Linux reports ``ru_maxrss`` in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ===================================================================
# PART A — Setup (not JIT-compiled)
# ===================================================================

print(f"\n--- Dataset loading [{instrument}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
mask_radius = INSTRUMENTS[instrument]["mask_radius"]
dataset_path = Path("dataset") / "interferometer" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="interferometer",
    instrument=instrument,
    workspace_root=_workspace_root,
)

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

transformer_chunk_size = None if USE_DFT else transformer_chunk_size_for(instrument)


def _build_transformer(uv_wavelengths, real_space_mask):
    """Inject the per-instrument ``chunk_size`` into ``TransformerNUFFT``.

    ``Interferometer.from_fits`` has no transformer-kwargs API, so the preset is
    injected through ``transformer_class`` (the delaunay breakdown's pattern).
    Note ``TransformerNUFFT.transform_mapping_matrix`` does not read
    ``chunk_size`` — it runs one ``nufft2d2`` over every column and every
    visibility — so the chunking reaches the forward/adjoint image transforms
    and the W~ precision-operator build, not step 3.
    """
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=transformer_chunk_size,
    )


_transformer_class = al.TransformerDFT if USE_DFT else _build_transformer
transformer_name = "TransformerDFT" if USE_DFT else "TransformerNUFFT"
print(f"  Transformer:             {transformer_name} (chunk_size={transformer_chunk_size})")

with timer.section("dataset_load"):
    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=_transformer_class,
    )

n_visibilities = int(dataset.uv_wavelengths.shape[0])
n_image_pixels = int(dataset.grids.lp.shape[0])
print(f"  Visibilities:            {n_visibilities}")
print(f"  Real-space masked pixels: {n_image_pixels}")

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

print("\n--- Model construction ---")

with timer.section("model_build"):
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    _lens_mass_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_lens_mass_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_lens_mass_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    field = af.Model(al.MassField, redshift=0.5, shear=shear)

    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=total_gaussians,
        centre_prior_is_uniform=False,
        centre=(0.1, 0.1),
        centre_sigma=0.005,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

print(f"  Total free parameters: {model.total_free_parameters}")

with timer.section("instance_from_vector"):
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

with timer.section("register_pytrees"):
    _register_model_pytrees(model)

params_tree = jax.tree_util.tree_map(jnp.asarray, instance)
tracer = al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields])

settings = al.Settings(use_mixed_precision=_cli.use_mixed_precision)


def _tracer_from(params):
    return al.Tracer(galaxies=list(params.galaxies), fields=[params.fields])


# ---------------------------------------------------------------------------
# Reference (FitInterferometer)
# ---------------------------------------------------------------------------
#
# Below EAGER_REFERENCE_MAX_VIS the reference is the eager ``xp=np`` fit. Above
# it, the eager path is not measurable: its NUFFT runs nufftax op by op, and the
# type-2 interpolation materialises a ``(n_trans, N_vis, nspread²)`` complex128
# gather — 20 x 1e6 x 169 x 16 B = 54 GB at alma (nspread 13 at eps 1e-12) —
# that XLA only fuses away under ``jax.jit``. (First alma attempt: killed at
# 5.9 GB RSS inside the eager fit on a 16 GB laptop.) So at alma scale and above
# the reference is the same ``FitInterferometer`` traced with ``xp=jnp`` under
# one ``jax.jit`` — still the library's own code path, and the one
# ``AnalysisInterferometer`` runs.

EAGER_REFERENCE_MAX_VIS = 200_000

_large_on_cpu = jax.default_backend() == "cpu" and n_visibilities > EAGER_REFERENCE_MAX_VIS
if _cell_args.n_repeats is not None:
    N_REPEATS = int(_cell_args.n_repeats)
else:
    N_REPEATS = 2 if _large_on_cpu else 10
print(f"  Steady-state repeats per step: {N_REPEATS}")

# A batched re-timing multiplies an already ~1 min NUFFT call by the batch on
# one CPU core and says nothing the GPU run will not; skipped there, with the
# reason recorded.
vmap_skipped_reason = None
if _vmap_batch is not None and _large_on_cpu:
    vmap_skipped_reason = (
        f"--vmap-batch {_vmap_batch} skipped on CPU above {EAGER_REFERENCE_MAX_VIS} "
        "visibilities: each unbatched NUFFT call is already ~1 min on one core."
    )
    print(f"  {vmap_skipped_reason}")
    _vmap_batch = None


def _nufftax_version():
    try:
        from importlib.metadata import version

        return version("nufftax")
    except Exception:  # noqa: BLE001
        return None


def _write_reference_oom(exc):
    """Record a device OOM of the JIT FitInterferometer reference as the result.

    The reference is the library's own dense MGE likelihood, whose
    ``transform_mapping_matrix`` is one ``nufft2d2`` over every column and every
    visibility (it ignores ``chunk_size``). Under pure-JAX fp64 nufftax (>= 0.6)
    on GPU the interpolation materialises an O(N_vis x N_gauss x nspread^2)
    intermediate, so at alma scale and above this allocation exceeds the device.
    Every dense step 3 would hit the same allocation, so the run stops here and
    writes this JSON (``stage='oom_reference'``, ``steps`` null) rather than a
    traceback-only failure. A RESOURCE_EXHAUSTED is a result, not a failure.
    """
    msg = str(exc).splitlines()[0][:500]
    print(f"\n  JIT reference OOM (recorded as the result): {msg}")
    default_dir = _workspace_root / "results" / "breakdown" / "interferometer"
    if _cli.config_name is not None and instrument != "alma":
        default_dir = default_dir / instrument
    cell_name = "mge_dft" if USE_DFT else "mge"
    out_path, _ = resolve_output_paths(
        _cli,
        default_dir=default_dir,
        default_basename=f"{cell_name}_breakdown_{instrument}_v{al.__version__}",
        cell=cell_name,
    )
    summary = {
        "stage": "oom_reference",
        "autolens_version": al.__version__,
        "device": device_info_dict(),
        "instrument": instrument,
        "model": "mge",
        "transformer": transformer_name,
        "use_mixed_precision": bool(_cli.use_mixed_precision),
        "configuration": {
            "pixel_scale_arcsec": pixel_scale,
            "mask_radius_arcsec": mask_radius,
            "real_space_shape": list(real_space_shape),
            "image_pixels_masked": n_image_pixels,
            "visibilities": n_visibilities,
            "inversion_path": "dense",
            "transformer": transformer_name,
            "chunk_size": transformer_chunk_size,
            "nufftax_version": _nufftax_version(),
            "w_tilde_arm": USE_W_TILDE,
        },
        "reference_mode": reference_mode,
        "oom": {
            "where": "jax.jit(FitInterferometer) reference (dense transform_mapping_matrix)",
            "error": msg,
            "note": (
                "TransformerNUFFT.transform_mapping_matrix runs one nufft2d2 over every "
                "Gaussian column and every visibility; pure-JAX fp64 nufftax interpolation "
                "on GPU materialises O(N_vis x N_gauss x nspread^2). No dense step and no W~ "
                "row was measured (both compare against this reference)."
            ),
        },
        "steps": None,
        "total_step_by_step": None,
        "full_pipeline_single_jit": None,
        "peak_rss_mb": _peak_rss_mb(),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Results dict saved to: {out_path}")


fit = al.FitInterferometer(dataset=dataset, tracer=tracer, settings=settings, xp=np)
reference_mode = "eager_numpy" if n_visibilities <= EAGER_REFERENCE_MAX_VIS else "jit_jax"

if reference_mode == "eager_numpy":
    print("\n--- Full FitInterferometer (eager baseline) ---")
    with timer.section("fit_interferometer_eager"):
        log_likelihood_ref = float(fit.log_likelihood)
        figure_of_merit_ref = float(fit.figure_of_merit)
else:
    print("\n--- Full FitInterferometer (JIT reference; eager path would OOM) ---")

    def _reference_fn(params):
        fit_jax = al.FitInterferometer(
            dataset=dataset, tracer=_tracer_from(params), settings=settings, xp=jnp
        )
        return fit_jax.log_likelihood, fit_jax.figure_of_merit

    try:
        with timer.section("fit_interferometer_jit_reference"):
            _ll, _fom = jax.jit(_reference_fn)(params_tree)
            log_likelihood_ref, figure_of_merit_ref = float(_ll), float(_fom)
    except Exception as exc:  # noqa: BLE001 — only a device OOM is recorded, see below
        if "RESOURCE_EXHAUSTED" not in str(exc):
            raise
        _write_reference_oom(exc)
        sys.exit(0)

# Structural properties only (class, parameter count, solver, preconditioning,
# no-regularization indices): none evaluates a mapping matrix. ``fit.inversion``
# itself is avoided above the eager threshold, because building it forms
# ``profile_subtracted_visibilities`` — an eager NUFFT of the (zero) standard
# light image whose op-by-op gather alone is ~8 GB RSS at alma.
if reference_mode == "eager_numpy":
    inversion = fit.inversion
else:
    inversion = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=dataset.data,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            transformer=dataset.transformer,
            sparse_operator=None,
        ),
        tracer=tracer,
        settings=settings,
        xp=np,
    ).inversion
n_linear = int(inversion.total_params)
preconditioning = inversion.positive_only_preconditioning_used
solver = inversion.positive_only_solver_used

print(f"  log_likelihood  = {log_likelihood_ref}")
print(f"  figure_of_merit = {figure_of_merit_ref}")
print(f"  inversion class = {type(inversion).__name__}")
print(f"  linear params   = {n_linear}; NNLS solver={solver}, preconditioning={preconditioning}")
print(
    "  log-det / regularization terms = "
    f"{inversion.log_det_curvature_reg_matrix_term}, "
    f"{inversion.log_det_regularization_matrix_term}, {inversion.regularization_term}"
)

# ===================================================================
# PART B — Per-step JIT profiling
# ===================================================================

print("\n" + "=" * 70)
print("PER-STEP JIT PROFILING")
print("=" * 70)

grid_lp_raw = jnp.asarray(dataset.grids.lp.array)
data_jnp = jnp.asarray(dataset.data.array)
noise_jnp = jnp.asarray(dataset.noise_map.array)
data_real_jnp = jnp.real(data_jnp)
data_imag_jnp = jnp.imag(data_jnp)
noise_real_jnp = jnp.real(noise_jnp)
noise_imag_jnp = jnp.imag(noise_jnp)
no_reg_index_list = list(inversion.no_regularization_index_list)


def _mapping_matrix_from(params):
    """``params -> (M_pix, 20)`` mapping matrix, exactly as TracerToInversion builds it."""
    tti = al.TracerToInversion(
        dataset=aa.DatasetInterface(
            data=dataset.data,
            noise_map=dataset.noise_map,
            grids=dataset.grids,
            transformer=dataset.transformer,
            sparse_operator=None,
        ),
        tracer=_tracer_from(params),
        settings=settings,
        xp=jnp,
    )
    funcs = list(tti.lp_linear_func_list_galaxy_dict.keys())
    matrices = [f.mapping_matrix for f in funcs]
    return jnp.hstack(matrices) if len(matrices) > 1 else matrices[0]


def _transform(mapping_matrix):
    return dataset.transformer.transform_mapping_matrix(mapping_matrix=mapping_matrix, xp=jnp)


def setup_prefix_fn(upto: int):
    """Nested ``params -> stage`` prefix for the steps 1-3 successive-difference split."""

    def fn(params):
        if upto == 1:
            traced = _tracer_from(params).traced_grid_2d_list_from(grid=dataset.grids.lp, xp=jnp)
            return jnp.stack([jnp.asarray(g.array) for g in traced])
        mapping_matrix = _mapping_matrix_from(params)
        if upto == 2:
            return mapping_matrix
        return _transform(mapping_matrix)

    return fn


prefix_labels = {
    1: "Ray-trace grids",
    2: f"Mapping matrix ({total_gaussians} Gaussians)",
    3: f"Transformed mapping matrix ({'DFT' if USE_DFT else 'NUFFT'})",
}

prefix_per_call: dict[int, float] = {}
prefix_outputs: dict[int, object] = {}
for _upto in (1, 2, 3):
    print(f"\n--- Setup prefix 1..{_upto}: {prefix_labels[_upto]} ---")
    _, prefix_outputs[_upto] = jit_profile(
        setup_prefix_fn(_upto), f"setup_prefix_{_upto}", params_tree
    )
    prefix_per_call[_upto] = _per_call(f"setup_prefix_{_upto}")
    print(f"  peak RSS so far: {_peak_rss_mb():.0f} MB")

setup_split = timing.split_by_successive_differences(prefix_per_call, prefix_labels)
for _key in (1, 2, 3):
    likelihood_steps.append((prefix_labels[_key], setup_split[prefix_labels[_key]]))

mapping_matrix_jit = prefix_outputs[2]
transformed_mm_jit = prefix_outputs[3]
print(f"  mapping_matrix shape: {mapping_matrix_jit.shape}")
print(f"  transformed_mapping_matrix shape: {transformed_mm_jit.shape}")

# Cross-check the JIT setup against the eager library inversion (eager mode
# only — the eager transformed matrix is exactly what OOMs at alma scale).
mapping_matrix_max_rel_diff = None
transformed_mm_max_rel_diff = None
if reference_mode == "eager_numpy":
    mapping_matrix_ref = np.asarray(inversion.mapping_matrix)
    transformed_mm_ref = np.asarray(inversion.operated_mapping_matrix)
    mapping_matrix_max_rel_diff = float(
        np.max(np.abs(np.asarray(mapping_matrix_jit) - mapping_matrix_ref))
        / max(np.max(np.abs(mapping_matrix_ref)), 1e-300)
    )
    transformed_mm_max_rel_diff = float(
        np.max(np.abs(np.asarray(transformed_mm_jit) - transformed_mm_ref))
        / max(np.max(np.abs(transformed_mm_ref)), 1e-300)
    )
    print(f"  mapping matrix     JIT vs eager max rel diff: {mapping_matrix_max_rel_diff:.3e}")
    print(f"  transformed matrix JIT vs eager max rel diff: {transformed_mm_max_rel_diff:.3e}")

# Standalone overlapping sub-rows: the same work without the params plumbing.
print("\n--- Standalone sub-rows (overlap steps 1 and 3) ---")


def ray_trace_raw(grid_raw):
    grid = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    traced = tracer.traced_grid_2d_list_from(grid=grid, xp=jnp)
    return jnp.stack([tg.array for tg in traced])


jit_profile(ray_trace_raw, "ray_trace_standalone", grid_lp_raw)
sub_rows["Ray-trace (standalone, eager tracer)"] = _per_call("ray_trace_standalone")

mapping_matrix_jnp = jnp.asarray(mapping_matrix_jit)
jit_profile(_transform, "transform_mapping_matrix_standalone", mapping_matrix_jnp)
sub_rows["transform_mapping_matrix (standalone)"] = _per_call("transform_mapping_matrix_standalone")
print(f"  peak RSS so far: {_peak_rss_mb():.0f} MB")

transformed_mm_jnp = jnp.asarray(transformed_mm_jit)

# ---------------------------------------------------------------------------
# Step 4: Data vector (D)
# ---------------------------------------------------------------------------

print("\n--- Step 4: Data vector (D) ---")


def compute_data_vector(transformed_mapping_matrix, visibilities, noise_map):
    return inversion_interferometer_util.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=visibilities,
        noise_map=noise_map,
    )


_, data_vector = jit_profile(
    compute_data_vector, "data_vector", transformed_mm_jnp, data_jnp, noise_jnp
)
likelihood_steps.append(("Data vector (D)", _per_call("data_vector")))

# ---------------------------------------------------------------------------
# Step 5: Curvature matrix (F)
# ---------------------------------------------------------------------------

print("\n--- Step 5: Curvature matrix (F) ---")


def compute_curvature_matrix_raw(transformed_mapping_matrix, noise_map):
    """Dense ``Re(T)ᵀ N⁻¹ Re(T) + Im(T)ᵀ N⁻¹ Im(T)``, before the diagonal floor."""
    real_curv = inversion_util.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrix.real,
        noise_map=noise_map.real,
        settings=settings,
        xp=jnp,
    )
    imag_curv = inversion_util.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=transformed_mapping_matrix.imag,
        noise_map=noise_map.imag,
        settings=settings,
        xp=jnp,
    )
    return real_curv + imag_curv


def add_no_reg_floor(curvature_matrix):
    """The library's no-regularization diagonal floor (every MGE column is unregularized)."""
    return inversion_util.curvature_matrix_with_added_to_diag_from(
        curvature_matrix=curvature_matrix,
        value=settings.no_regularization_add_to_curvature_diag_value,
        no_regularization_index_list=no_reg_index_list,
        xp=jnp,
    )


def compute_curvature_matrix(transformed_mapping_matrix, noise_map):
    return add_no_reg_floor(compute_curvature_matrix_raw(transformed_mapping_matrix, noise_map))


_, curvature_matrix = jit_profile(
    compute_curvature_matrix, "curvature_matrix", transformed_mm_jnp, noise_jnp
)
likelihood_steps.append(("Curvature matrix (F)", _per_call("curvature_matrix")))
curvature_matrix_raw = jax.jit(compute_curvature_matrix_raw)(transformed_mm_jnp, noise_jnp)

# ---------------------------------------------------------------------------
# Step 6: Positive-only reconstruction (PDIP NNLS)
# ---------------------------------------------------------------------------

print("\n--- Step 6: Reconstruction (PDIP NNLS) ---")


def compute_reconstruction(data_vector, curvature_reg_matrix):
    stats: dict = {}
    reconstruction = inversion_util.reconstruction_positive_only_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        settings=settings,
        xp=jnp,
        solver=solver,
        stats=stats,
        preconditioning=preconditioning,
    )
    return reconstruction, stats.get("converged", -1), stats.get("iterations", -1)


_, (reconstruction, _nnls_converged, _nnls_iterations) = jit_profile(
    compute_reconstruction, "reconstruction", data_vector, curvature_matrix
)
likelihood_steps.append(("Reconstruction (PDIP NNLS)", _per_call("reconstruction")))
nnls_iterations = int(_nnls_iterations)
nnls_converged = bool(int(_nnls_converged) == 1)
print(f"  PDIP iterations: {nnls_iterations} (cap 50), converged={nnls_converged}")

# Step 7 (log-det terms) is structurally zero for an MGE-only inversion — see
# the module docstring. Recorded, not timed.

# ---------------------------------------------------------------------------
# Step 8: Mapped visibilities + chi-squared + log likelihood
# ---------------------------------------------------------------------------

# What the sampler receives is ``FitInterferometer.figure_of_merit`` =
# ``log_evidence``, whose chi-squared is ``inversion.fast_chi_squared``
# (``sᵀ F s - 2 sᵀ D + dᵀ N⁻¹ d``, interferometer/abstract.py) — the mapped
# visibilities are never formed on the production path. ``F`` there is the
# curvature matrix *with* the no-regularization diagonal floor, so the figure of
# merit sits ``0.5 * floor * |s|²`` nats below ``FitInterferometer.log_likelihood``
# (which forms the residual visibilities). Step 8 times the production form; the
# residual-visibility form is an overlapping sub-row checked against
# ``log_likelihood``.

print("\n--- Step 8: Fast chi-squared + figure of merit ---")


def noise_normalization(noise_map):
    return jnp.sum(jnp.log(2 * jnp.pi * noise_map.real**2)) + jnp.sum(
        jnp.log(2 * jnp.pi * noise_map.imag**2)
    )


def data_weighted_norm_from(visibilities, noise_map):
    """``dᵀ N⁻¹ d`` over real and imaginary parts (``fast_chi_squared``'s term 3)."""
    return jnp.sum(visibilities.real**2 / noise_map.real**2) + jnp.sum(
        visibilities.imag**2 / noise_map.imag**2
    )


def compute_figure_of_merit(reconstruction, data_vector, curvature_matrix, visibilities, noise_map):
    """``FitInterferometer.log_evidence`` for a regularization-free inversion."""
    chi_squared = (
        reconstruction @ curvature_matrix @ reconstruction
        - 2.0 * reconstruction @ data_vector
        + data_weighted_norm_from(visibilities, noise_map)
    )
    return -0.5 * (chi_squared + noise_normalization(noise_map))


def compute_log_likelihood(transformed_mapping_matrix, reconstruction, visibilities, noise_map):
    """``FitInterferometer.log_likelihood``: residual visibilities, no floor bias."""
    mapped = transformed_mapping_matrix @ reconstruction
    residual = visibilities - mapped
    chi_squared = jnp.sum((residual.real / noise_map.real) ** 2) + jnp.sum(
        (residual.imag / noise_map.imag) ** 2
    )
    return -0.5 * (chi_squared + noise_normalization(noise_map))


_, figure_of_merit_steps = jit_profile(
    compute_figure_of_merit,
    "figure_of_merit",
    reconstruction,
    data_vector,
    curvature_matrix,
    data_jnp,
    noise_jnp,
)
likelihood_steps.append(("Fast chi-squared + figure of merit", _per_call("figure_of_merit")))
figure_of_merit_steps = float(figure_of_merit_steps)

_, log_likelihood_steps = jit_profile(
    compute_log_likelihood,
    "log_likelihood_residual_visibilities",
    transformed_mm_jnp,
    reconstruction,
    data_jnp,
    noise_jnp,
)
sub_rows["Mapped visibilities + residual chi-squared (log_likelihood)"] = _per_call(
    "log_likelihood_residual_visibilities"
)
log_likelihood_steps = float(log_likelihood_steps)

# Correctness, in nats (an rtol on a log L whose noise normalisation grows with
# N_vis says nothing at alma scale).
LOG_L_ATOL_NATS = 1.0 if _cli.use_mixed_precision else 1e-3
figure_of_merit_abs_diff = abs(figure_of_merit_steps - figure_of_merit_ref)
log_likelihood_abs_diff = abs(log_likelihood_steps - log_likelihood_ref)
floor_bias_nats = log_likelihood_ref - figure_of_merit_ref
print(f"  figure_of_merit (step-by-step)     = {figure_of_merit_steps}")
print(f"  figure_of_merit (FitInterferometer) = {figure_of_merit_ref}")
print(f"  |diff| = {figure_of_merit_abs_diff:.3e} nats (atol {LOG_L_ATOL_NATS:g})")
print(f"  log_likelihood (residual form) |diff vs Fit| = {log_likelihood_abs_diff:.3e} nats")
print(f"  log_likelihood - figure_of_merit (diagonal-floor bias) = {floor_bias_nats:.6e} nats")
np.testing.assert_allclose(
    figure_of_merit_steps,
    figure_of_merit_ref,
    rtol=0.0,
    atol=LOG_L_ATOL_NATS,
    err_msg="interferometer/mge: step-by-step figure of merit does not match FitInterferometer",
)
np.testing.assert_allclose(
    log_likelihood_steps,
    log_likelihood_ref,
    rtol=0.0,
    atol=LOG_L_ATOL_NATS,
    err_msg="interferometer/mge: residual-form log L does not match FitInterferometer",
)
print("  Assertions PASSED: step-by-step figure of merit and log L match FitInterferometer")

full_pipeline_per_call = None
full_pipeline_logl = None
full_pipeline_abs_diff = None

w_tilde_steps: dict[str, float] | None = None
w_tilde: dict | None = None
vmap_steps: dict[str, float] | None = None
vmap_split: dict[str, float] | None = None
vmap_error: str | None = None


# ===================================================================
# Summary + JSON + PNG (written after the dense steps, then again at the end)
# ===================================================================

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__


def write_results(stage: str):
    """Write the result JSON + PNG.

    Called once right after the dense steps (``stage='dense'``) so an OOM kill
    in the W~ arm or the vmap block — which the OS delivers as SIGKILL, not an
    exception — cannot lose the dense measurement, and again at the end
    (``stage='complete'``).
    """
    global dict_path, chart_path

    step_total = float(sum(t for _, t in likelihood_steps))
    peak_rss_mb = _peak_rss_mb()

    print("\n" + "=" * 70)
    print(f"PER-STEP BREAKDOWN — {instrument.upper()} — {transformer_name} — v{al_version}")
    print("=" * 70)
    print(
        f"  Visibilities: {n_visibilities}   masked pixels: {n_image_pixels}   Gaussians: {n_linear}"
    )
    _w = max(len(label) for label, _ in likelihood_steps)
    for i, (label, per_call) in enumerate(likelihood_steps, 1):
        print(f"  {i:>2}. {label:<{_w}}  {per_call * 1e3:12.3f} ms")
    print("-" * 70)
    print(f"      {'TOTAL (step-by-step)':<{_w}}  {step_total * 1e3:12.3f} ms")
    if full_pipeline_per_call is not None:
        print(
            f"      {'Full pipeline (single JIT)':<{_w}}  {full_pipeline_per_call * 1e3:12.3f} ms"
        )
    print("  Overlapping sub-rows (not summed):")
    for label, per_call in sub_rows.items():
        print(f"      {label:<{_w}}  {per_call * 1e3:12.3f} ms")
    if w_tilde_steps:
        print("  W~ arm rows:")
        for label, per_call in w_tilde_steps.items():
            print(f"      {label:<{_w}}  {per_call * 1e3:12.3f} ms")
    print(f"  Peak RSS: {peak_rss_mb:.0f} MB")
    print("=" * 70)

    breakdown_summary = {
        "stage": stage,
        "autolens_version": al_version,
        "device": device_info_dict(),
        "instrument": instrument,
        "model": "mge",
        "transformer": transformer_name,
        "use_mixed_precision": bool(_cli.use_mixed_precision),
        "configuration": {
            "pixel_scale_arcsec": pixel_scale,
            "mask_radius_arcsec": mask_radius,
            "real_space_shape": list(real_space_shape),
            "image_pixels_masked": n_image_pixels,
            "visibilities": n_visibilities,
            "linear_gaussians": n_linear,
            "lens_light": None,
            "inversion_path": "dense",
            "inversion_class": type(inversion).__name__,
            "transformer": transformer_name,
            "chunk_size": transformer_chunk_size,
            "chunk_size_note": (
                "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one "
                "nufft2d2 over every column and every visibility."
            ),
            "nnls_solver": solver,
            "nnls_preconditioning": preconditioning,
            "nufftax_version": _nufftax_version(),
            "w_tilde_arm": USE_W_TILDE,
            "n_repeats": N_REPEATS,
            "thread_env": _observe_thread_env(),
            "host_load_avg_start": _load_avg_start,
            "host_load_avg_end": list(os.getloadavg()),
        },
        "reference_mode": reference_mode,
        "log_likelihood_reference": log_likelihood_ref,
        "figure_of_merit_reference": figure_of_merit_ref,
        "figure_of_merit_step_by_step": figure_of_merit_steps,
        "log_likelihood_step_by_step": log_likelihood_steps,
        "figure_of_merit_full_pipeline": full_pipeline_logl,
        "figure_of_merit_abs_diff_nats": float(figure_of_merit_abs_diff),
        "log_likelihood_abs_diff_nats": float(log_likelihood_abs_diff),
        "full_pipeline_abs_diff_nats": (
            float(full_pipeline_abs_diff) if full_pipeline_abs_diff is not None else None
        ),
        "log_likelihood_atol_nats": LOG_L_ATOL_NATS,
        "diagonal_floor_bias_nats": float(floor_bias_nats),
        "diagonal_floor_note": (
            "log_likelihood - figure_of_merit: fast_chi_squared uses F with the "
            "no-regularization diagonal floor "
            f"({settings.no_regularization_add_to_curvature_diag_value}), so the figure of "
            "merit the sampler sees sits 0.5 * floor * |s|^2 below the residual-form log L."
        ),
        "setup_cross_check": {
            "mapping_matrix_max_rel_diff_vs_eager": mapping_matrix_max_rel_diff,
            "transformed_mapping_matrix_max_rel_diff_vs_eager": transformed_mm_max_rel_diff,
        },
        "log_det_terms": {
            "value": float(inversion.log_det_curvature_reg_matrix_term)
            - float(inversion.log_det_regularization_matrix_term),
            "timed": False,
            "note": "Structurally zero: an MGE-only inversion has no regularization.",
        },
        "nnls": {
            "iterations": nnls_iterations,
            "converged": nnls_converged,
            "max_iter": 50,
            "solver": solver,
            "preconditioning": preconditioning,
        },
        "steps": {label: float(per_call) for label, per_call in likelihood_steps},
        "total_step_by_step": step_total,
        "full_pipeline_single_jit": (
            float(full_pipeline_per_call) if full_pipeline_per_call is not None else None
        ),
        "steps_sub_rows": {k: float(v) for k, v in sub_rows.items()},
        "setup_split": {k: float(v) for k, v in setup_split.items()},
        "setup_prefix_per_call_s": {str(k): float(v) for k, v in prefix_per_call.items()},
        "jit_phases": jit_records,
        "peak_rss_mb": float(peak_rss_mb),
    }

    if USE_W_TILDE:
        breakdown_summary["w_tilde"] = w_tilde
        if w_tilde_steps is not None:
            breakdown_summary["w_tilde_steps"] = {k: float(v) for k, v in w_tilde_steps.items()}

    if vmap_skipped_reason is not None:
        breakdown_summary["vmap_skipped_reason"] = vmap_skipped_reason

    if _vmap_batch is not None:
        breakdown_summary["vmap_batch"] = int(_vmap_batch)
        if vmap_error is not None:
            breakdown_summary["vmap_error"] = vmap_error
        elif vmap_steps is not None:
            breakdown_summary["steps_vmap_per_call"] = {k: float(v) for k, v in vmap_steps.items()}
            breakdown_summary["setup_split_vmap"] = {k: float(v) for k, v in vmap_split.items()}

    _cell_name = "mge_dft" if USE_DFT else "mge"
    _default_dir = _workspace_root / "results" / "breakdown" / "interferometer"
    if _cli.config_name is not None and instrument != "alma":
        # Config-tagged names carry no instrument (CONFIG_TAGGED_RE); keep the
        # instruments apart by directory. See the module docstring, "Output".
        _default_dir = _default_dir / instrument

    dict_path, chart_path = resolve_output_paths(
        _cli,
        default_dir=_default_dir,
        default_basename=f"{_cell_name}_breakdown_{instrument}_v{al_version}",
        cell=_cell_name,
    )
    dict_path.write_text(json.dumps(breakdown_summary, indent=2))
    print(f"\n  Results dict saved to: {dict_path}")

    labels = [label for label, _ in likelihood_steps]
    times = [per_call for _, per_call in likelihood_steps]
    if w_tilde_steps:
        labels += [f"W~: {label}" for label in w_tilde_steps]
        times += list(w_tilde_steps.values())
    colors = ["#4C72B0"] * len(likelihood_steps) + ["#55A868"] * (
        len(labels) - len(likelihood_steps)
    )

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 2))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t * 1e3:.3f} ms",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Time per call (s)", fontsize=11)
    fig.suptitle(
        f"MGE Interferometer Likelihood — Per-Step Breakdown — {instrument.upper()} ({transformer_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{al_version}  |  {n_visibilities} visibilities  |  {n_image_pixels} pixels  |  "
        f"{n_linear} Gaussians  |  step total: {step_total * 1e3:.3f} ms  |  "
        + (
            f"full JIT: {full_pipeline_per_call * 1e3:.3f} ms"
            if full_pipeline_per_call is not None
            else "full JIT: not reached"
        ),
        fontsize=8,
    )
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to:    {chart_path}")


# Checkpoint: the per-step rows are on disk before the fused comparators, whose
# extra resident buffers are what an OOM kill lands on at alma_high scale on CPU.
write_results("steps")


# ---------------------------------------------------------------------------
# Comparators: the dense chain from the mapping matrix, and the fused pipeline
# ---------------------------------------------------------------------------

print("\n--- Dense chain from mapping matrix (steps 3-8 in one JIT) ---")


def dense_chain_from_mapping_matrix(mapping_matrix):
    tmm = _transform(mapping_matrix)
    dv = compute_data_vector(tmm, data_jnp, noise_jnp)
    cm = compute_curvature_matrix(tmm, noise_jnp)
    recon, _, _ = compute_reconstruction(dv, cm)
    return compute_figure_of_merit(recon, dv, cm, data_jnp, noise_jnp)


_, _dense_chain_logl = jit_profile(
    dense_chain_from_mapping_matrix, "dense_chain_from_mapping_matrix", mapping_matrix_jnp
)
sub_rows["Dense chain from mapping matrix (steps 3-8 fused)"] = _per_call(
    "dense_chain_from_mapping_matrix"
)

print("\n--- Full pipeline (AnalysisInterferometer.log_likelihood_function) ---")

analysis = al.AnalysisInterferometer(dataset=dataset, settings=settings, use_jax=True)


def full_pipeline(params):
    return analysis.log_likelihood_function(instance=params)


_, full_pipeline_logl = jit_profile(full_pipeline, "full_pipeline", params_tree)
full_pipeline_per_call = _per_call("full_pipeline")
full_pipeline_logl = float(full_pipeline_logl)
full_pipeline_abs_diff = abs(full_pipeline_logl - figure_of_merit_ref)
print(
    f"  full-pipeline value = {full_pipeline_logl} "
    f"(|diff vs figure_of_merit| = {full_pipeline_abs_diff:.3e} nats)"
)
np.testing.assert_allclose(
    full_pipeline_logl,
    figure_of_merit_ref,
    rtol=0.0,
    atol=LOG_L_ATOL_NATS,
    err_msg="interferometer/mge: full-pipeline JIT does not match FitInterferometer.figure_of_merit",
)
print(f"  peak RSS so far: {_peak_rss_mb():.0f} MB")

write_results("dense")

# ===================================================================
# PART C — W~ arm (--w-tilde): measurement-only func-list W~ path
# ===================================================================


w_tilde_nufft_chunk_size = (
    transformer_chunk_size
    if transformer_chunk_size is not None
    else (100_000 if n_visibilities > EAGER_REFERENCE_MAX_VIS else None)
)

if USE_W_TILDE:
    print("\n" + "=" * 70)
    print("W~ ARM (func-list W~ curvature, measurement only)")
    print("=" * 70)
    try:
        with timer.section("w_tilde_apply_sparse_operator"):
            # ``apply_sparse_operator`` runs two eager (op-by-op) type-1 NUFFTs:
            # the precision-operator build and the dirty image
            # (``transformer.image_from``). Both honour a visibility chunk size —
            # the builder through ``nufft_chunk_size``, the dirty image through
            # the transformer's own ``chunk_size`` — so above the eager threshold
            # the operator is built from a copy of the dataset whose transformer
            # is chunked. Unchunked, the build was killed at 9.2-9.6 GB RSS at
            # alma on the laptop. The operator and dirty image are the same
            # quantities either way (chunking only bounds the spread buffer), and
            # the F~ / D~ checks below verify it against the dense path.
            if w_tilde_nufft_chunk_size is not None and not USE_DFT:
                _chunk = w_tilde_nufft_chunk_size
                dataset_for_operator = al.Interferometer(
                    data=dataset.data,
                    noise_map=dataset.noise_map,
                    uv_wavelengths=dataset.uv_wavelengths,
                    real_space_mask=real_space_mask,
                    transformer_class=lambda uv_wavelengths, real_space_mask: al.TransformerNUFFT(
                        uv_wavelengths=uv_wavelengths,
                        real_space_mask=real_space_mask,
                        chunk_size=_chunk,
                    ),
                )
            else:
                dataset_for_operator = dataset
            dataset_w = dataset_for_operator.apply_sparse_operator(
                use_jax=True, nufft_chunk_size=w_tilde_nufft_chunk_size
            )
        w_tilde_build_s = timer.records[-1][1]
        sparse_operator = dataset_w.sparse_operator
        extent_index = jnp.asarray(
            np.asarray(dataset.real_space_mask.extent_index_for_masked_pixel), dtype=jnp.int32
        )
        dirty_image_jnp = jnp.asarray(sparse_operator.dirty_image)
        # dᵀ N⁻¹ d and the noise normalisation are per-dataset constants: the W~
        # path never needs the visibilities per call.
        data_weighted_norm = float(data_weighted_norm_from(data_jnp, noise_jnp))
        noise_norm_const = float(noise_normalization(noise_jnp))

        def w_tilde_curvature_raw(mapping_matrix):
            return sparse_operator.curvature_matrix_func_list_from(
                curvature_weights_0=mapping_matrix,
                curvature_weights_1=mapping_matrix,
                extent_index_for_masked_pixel=extent_index,
                xp=jnp,
            )

        def w_tilde_curvature(mapping_matrix):
            return add_no_reg_floor(w_tilde_curvature_raw(mapping_matrix))

        def w_tilde_data_vector(mapping_matrix):
            return mapping_matrix.T @ dirty_image_jnp

        def w_tilde_figure_of_merit(reconstruction, data_vector, curvature_matrix):
            """``fast_chi_squared`` with the per-dataset constants hoisted.

            Passed the floored ``F~`` this is the production figure of merit;
            passed the raw ``F~`` it is the exact (unbiased) log likelihood.
            """
            chi_squared = (
                data_weighted_norm
                - 2.0 * jnp.dot(reconstruction, data_vector)
                + jnp.dot(reconstruction, curvature_matrix @ reconstruction)
            )
            return -0.5 * (chi_squared + noise_norm_const)

        def w_tilde_chain_from_mapping_matrix(mapping_matrix):
            f_floored = w_tilde_curvature(mapping_matrix)
            dv = w_tilde_data_vector(mapping_matrix)
            recon, _, _ = compute_reconstruction(dv, f_floored)
            return w_tilde_figure_of_merit(recon, dv, f_floored)

        w_tilde_steps = {}
        _, curvature_w = jit_profile(w_tilde_curvature, "w_tilde_curvature", mapping_matrix_jnp)
        w_tilde_steps["Curvature matrix F~ = Bᵀ W~ B"] = _per_call("w_tilde_curvature")
        _, data_vector_w = jit_profile(
            w_tilde_data_vector, "w_tilde_data_vector", mapping_matrix_jnp
        )
        w_tilde_steps["Data vector D~ = Bᵀ d~"] = _per_call("w_tilde_data_vector")
        _, (recon_w, _, _) = jit_profile(
            compute_reconstruction, "w_tilde_reconstruction", data_vector_w, curvature_w
        )
        w_tilde_steps["Reconstruction (PDIP NNLS)"] = _per_call("w_tilde_reconstruction")
        curvature_w_raw = jax.jit(w_tilde_curvature_raw)(mapping_matrix_jnp)
        _, figure_of_merit_w = jit_profile(
            w_tilde_figure_of_merit,
            "w_tilde_figure_of_merit",
            recon_w,
            data_vector_w,
            curvature_w,
        )
        w_tilde_steps["Fast chi-squared + figure of merit (constants hoisted)"] = _per_call(
            "w_tilde_figure_of_merit"
        )
        log_likelihood_w = jax.jit(w_tilde_figure_of_merit)(recon_w, data_vector_w, curvature_w_raw)
        _, figure_of_merit_w_chain = jit_profile(
            w_tilde_chain_from_mapping_matrix,
            "w_tilde_chain_from_mapping_matrix",
            mapping_matrix_jnp,
        )

        f_dense = np.asarray(curvature_matrix_raw)
        f_w = np.asarray(curvature_w_raw)
        f_max_rel_diff = float(np.max(np.abs(f_w - f_dense)) / np.max(np.abs(f_dense)))
        d_max_rel_diff = float(
            np.max(np.abs(np.asarray(data_vector_w) - np.asarray(data_vector)))
            / np.max(np.abs(np.asarray(data_vector)))
        )
        log_likelihood_w = float(log_likelihood_w)
        figure_of_merit_w = float(figure_of_merit_w)
        figure_of_merit_w_chain = float(figure_of_merit_w_chain)
        dense_curvature_s = _per_call("curvature_matrix")
        dense_setup_to_f_s = sub_rows["transform_mapping_matrix (standalone)"] + dense_curvature_s
        w_tilde = {
            "operator_build_s": float(w_tilde_build_s),
            "operator_build_nufft_chunk_size": w_tilde_nufft_chunk_size,
            "operator_shape": [int(sparse_operator.y_shape), int(sparse_operator.x_shape)],
            "curvature_max_rel_diff_vs_dense": f_max_rel_diff,
            "data_vector_max_rel_diff_vs_dense": d_max_rel_diff,
            "figure_of_merit": figure_of_merit_w,
            "figure_of_merit_chain": figure_of_merit_w_chain,
            "figure_of_merit_abs_diff_vs_fit_nats": abs(figure_of_merit_w - figure_of_merit_ref),
            "log_likelihood_unfloored": log_likelihood_w,
            "log_likelihood_abs_diff_vs_fit_nats": abs(log_likelihood_w - log_likelihood_ref),
            "total_step_by_step": float(sum(w_tilde_steps.values())),
            "chain_from_mapping_matrix_s": _per_call("w_tilde_chain_from_mapping_matrix"),
            "dense_chain_from_mapping_matrix_s": _per_call("dense_chain_from_mapping_matrix"),
            "dense_transform_plus_curvature_s": float(dense_setup_to_f_s),
            "w_tilde_curvature_s": _per_call("w_tilde_curvature"),
            "speedup_curvature_vs_dense_transform_plus_curvature": float(
                dense_setup_to_f_s / _per_call("w_tilde_curvature")
            ),
            "speedup_chain_vs_dense_chain": float(
                _per_call("dense_chain_from_mapping_matrix")
                / _per_call("w_tilde_chain_from_mapping_matrix")
            ),
            "note": (
                "Measurement-only: an MGE-only fit never takes this path in the library "
                "(inversion/factory.py switches the sparse operator off when every linear "
                "object is an AbstractLinearObjFuncList). The chain row replaces steps 3-8 "
                "from the same mapping matrix; its per-call cost has no N_vis term."
            ),
        }
        print(f"  F~ vs dense F: max rel diff {f_max_rel_diff:.3e}")
        print(f"  D~ vs dense D: max rel diff {d_max_rel_diff:.3e}")
        print(
            f"  W~ figure of merit = {figure_of_merit_w} "
            f"(|diff vs Fit| = {w_tilde['figure_of_merit_abs_diff_vs_fit_nats']:.3e} nats); "
            f"unfloored log L |diff vs Fit| = "
            f"{w_tilde['log_likelihood_abs_diff_vs_fit_nats']:.3e} nats"
        )
        print(
            f"  W~ chain {w_tilde['chain_from_mapping_matrix_s'] * 1e3:.3f} ms vs dense chain "
            f"{w_tilde['dense_chain_from_mapping_matrix_s'] * 1e3:.3f} ms"
        )
    except Exception:  # noqa: BLE001 — a W~ failure must not lose the dense run
        w_tilde = {"error": traceback.format_exc()}
        print("  W~ ARM FAILED — dense results are unaffected. Traceback:")
        print(w_tilde["error"])

# ===================================================================
# PART D — Optional batched re-timing (--vmap-batch N)
# ===================================================================

if _vmap_batch is not None:
    print(f"\n--- Batched re-timing (--vmap-batch {_vmap_batch}) ---")
    _params_batched = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (_vmap_batch, *leaf.shape)), params_tree
    )
    try:
        _vmap_prefix: dict[int, float] = {}
        for _upto in (1, 2, 3):
            _vmap_prefix[_upto] = timing.vmap_profile(
                setup_prefix_fn(_upto),
                f"setup_prefix_{_upto}",
                _params_batched,
                _vmap_batch,
                n_repeats=N_REPEATS,
                timer=timer,
                jit_records=jit_records,
            )
        vmap_split = timing.split_by_successive_differences(_vmap_prefix, prefix_labels)
        vmap_steps = {
            "Full pipeline": timing.vmap_profile(
                full_pipeline,
                "full_pipeline",
                _params_batched,
                _vmap_batch,
                n_repeats=N_REPEATS,
                timer=timer,
                jit_records=jit_records,
            )
        }
    except Exception:  # noqa: BLE001 — a vmap failure must not lose the unbatched run
        vmap_error = traceback.format_exc()
        print("  VMAP FAILED — unbatched results are unaffected. Traceback:")
        print(vmap_error)

# ===================================================================
write_results("complete")


# ===================================================================
# Pinned reference figure of merit — recorded, never asserted
# ===================================================================
#
# Profiling records drift; it does not adjudicate library correctness
# (results/notes/design_lock_in.md). The pin is the figure of merit (what
# AnalysisInterferometer returns to the sampler), per (instrument, transformer),
# from one fp64 JAX-CPU run on this host (2026-09-25, autolens_profiling#308);
# ``None`` means "no pin yet". sma is the eager NumPy fit, alma the JIT
# reference (the eager path does not fit in memory there).

EXPECTED_FIGURE_OF_MERIT = {
    ("sma", "TransformerNUFFT"): -3153.948230379729,
    ("sma", "TransformerDFT"): -3153.948230379729,
    ("alma", "TransformerNUFFT"): -12047193.68761333,
    ("alma_high", "TransformerNUFFT"): -60242552.89876968,
    ("jvla", "TransformerNUFFT"): None,
}

_pinned_expected = EXPECTED_FIGURE_OF_MERIT.get((instrument, transformer_name))
_pinned_drift: list = []
if _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for ({instrument}, {transformer_name}) — "
        f"reference figure_of_merit = {figure_of_merit_ref!r}"
    )
else:
    _rtol = 1e-6 if _cli.use_mixed_precision else 1e-9
    for _label, _value in (
        ("reference", figure_of_merit_ref),
        ("step_by_step", figure_of_merit_steps),
        ("full_pipeline", full_pipeline_logl),
    ):
        _rec = check_pinned(_value, _pinned_expected, label=_label, rtol=_rtol)
        if _rec is not None:
            _pinned_drift.append(_rec)
record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
