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

Before PyAutoArray#575 an MGE-only fit always took the dense path:
``inversion/factory.py`` switched the sparse operator off when every linear
object is an ``AbstractLinearObjFuncList``. The W~ curvature helpers that the mixed
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

The library W~ route (``--sparse``)
-----------------------------------

Since PyAutoArray#575 the factory keeps the sparse operator for an all-func-list
interferometer inversion, so ``FitInterferometer`` on a dataset with
``apply_sparse_operator()`` applied takes ``InversionInterferometerSparse``.
The shared ``--sparse`` flag adds that arm (PART C2) after the dense arms, which
still run, so one JSON holds before and after:

- the inversion class the library picks (asserted to be the sparse class);
- steps as nested ``params -> stage`` prefixes on the library path (ray-trace,
  mapping matrix, ``D~`` + ``F~``, NNLS, ``FitInterferometer.figure_of_merit``),
  attributed by successive differences, in ``steps`` / ``setup_split`` /
  ``jit_phases`` — the dense rows move to ``dense_steps`` /
  ``dense_total_step_by_step`` / ``dense_setup_split``, and
  ``configuration.inversion_path`` becomes ``"sparse"``;
- the jitted ``FitInterferometer.log_likelihood`` and the
  ``AnalysisInterferometer`` full pipeline on the sparse dataset, checked
  against the dense reference (``library_sparse.witness_pass``: ``|Δ| <= 1e-6``
  nats where the dense reference runs on the device; against the laptop CPU pins
  otherwise, recorded only);
- where the dense reference runs, an MGE + lens Sersic variant, sparse vs dense,
  with an uncorrected-dirty-image control (``library_sparse.lens_sersic_variant``)
  that exercises the #575 ``d~ - W~ i_p`` correction;
- ``--sparse-vmap-batch 64,16,4``: the sparse full pipeline under ``jax.vmap``,
  largest batch first, stopping at the first that fits (the VRAM rows).

The sparse dataset is the one the W~ arm builds (chunked transformer above the
eager threshold), so ``library_sparse.fit_transformer_chunk_size`` records the
visibility chunk the fit's own NUFFTs (mapped visibilities, profile
visibilities) ran with. A library error, or a gap beyond ``LOG_L_ATOL_NATS``,
exits non-zero after the JSON is written.

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
- ``--sparse`` appends ``_sparse`` (``_profile_cli.resolve_output_paths``):
  ``mge_breakdown_{instrument}_v{al_version}_sparse`` /
  ``mge_<config>_sparse.json``, so the #308 dense files are kept.
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
import re
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
from likelihood_breakdown.provenance import source_revisions  # noqa: E402

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
_cell_parser.add_argument(
    "--transform-chunk",
    type=int,
    default=None,
    help=(
        "Measurement-only chunked-transform arm: compute the transformed mapping matrix "
        "column-by-column (jax.lax.map, this many columns per vmapped batch) through the "
        "transformer's own visibility-chunked forward NUFFT, instead of the library's "
        "one-shot transform_mapping_matrix. Steps 3-8 then run on it."
    ),
)
_cell_parser.add_argument(
    "--transform-vis-chunk",
    type=int,
    default=None,
    help=(
        "Visibility chunk for the --transform-chunk arm's per-column forward NUFFT. "
        "Default: the instrument's transformer_chunk_size preset (None = one shot)."
    ),
)
_cell_parser.add_argument(
    "--sparse-vmap-batch",
    type=str,
    default=None,
    help=(
        "Library W~ route arm (--sparse) only: comma-separated vmap batches to try for the "
        "sparse full pipeline, largest first; the first that runs without a device OOM is "
        "kept (a small VRAM probe). Default: --vmap-batch."
    ),
)
_cell_args = _cli.parse_cell_args(_cell_parser)

USE_DFT = bool(_cell_args.use_dft)
USE_W_TILDE = bool(_cell_args.w_tilde)
# The library W~ route (PyAutoArray#575): the shared ``--sparse`` flag applies
# ``dataset.apply_sparse_operator()`` and fits through the library, which since
# #575 routes an MGE-only inversion to ``InversionInterferometerSparse``.
USE_LIBRARY_SPARSE = bool(_cli.use_sparse_operator)
TRANSFORM_COLUMN_BATCH = _cell_args.transform_chunk
_vmap_batch = _cell_args.vmap_batch

# Import provenance: which library checkouts actually ran (a branch run on RAL
# prepends a private copy to PYTHONPATH; the log and the JSON must show it).
_source_revisions = source_revisions(_profiling_root())
print("--- Import provenance ---")
for _module in (aa, al, af):
    print(f"  {_module.__name__}.__file__ = {_module.__file__}")
import autogalaxy as _ag  # noqa: E402

print(f"  autogalaxy.__file__ = {_ag.__file__}")
for _repo, _rev in _source_revisions.items():
    print(f"  {_repo:<18} {_rev}")

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


def _oom_record(exc, where):
    """A device RESOURCE_EXHAUSTED recorded as a result: requested bytes + where."""
    msg = str(exc).splitlines()[0][:500]
    requested = None
    m = re.search(r"allocate ([0-9.]+)\s*(B|KiB|MiB|GiB|TiB)", msg)
    if m:
        scale = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}[m.group(2)]
        requested = int(float(m.group(1)) * scale)
    print(f"  RESOURCE_EXHAUSTED in {where} (recorded as a result): {msg}")
    return {"status": "oom", "requested_bytes": requested, "where": where, "error": msg}


def _is_oom(exc) -> bool:
    return "RESOURCE_EXHAUSTED" in str(exc)


# Pinned reference values from one fp64 JAX-CPU run on the laptop (2026-09-25,
# autolens_profiling#308): the figure of merit (what AnalysisInterferometer
# returns) and the residual-form log likelihood, per (instrument, transformer).
# They are recorded against, never asserted at the end; when the library's own
# JIT reference cannot run on this device (the A100 OOM, see "library path"
# below) they also stand in as the step-by-step reference
# (``reference_mode = "cpu_pinned"``). ``None`` means "no pin yet".

EXPECTED_FIGURE_OF_MERIT = {
    ("sma", "TransformerNUFFT"): -3153.948230379729,
    ("sma", "TransformerDFT"): -3153.948230379729,
    ("alma", "TransformerNUFFT"): -12047193.68761333,
    ("alma_high", "TransformerNUFFT"): -60242552.89876968,
    ("jvla", "TransformerNUFFT"): None,
}
EXPECTED_LOG_LIKELIHOOD = {
    ("sma", "TransformerNUFFT"): -3153.942384509246,
    ("alma", "TransformerNUFFT"): -12047193.687220689,
    ("alma_high", "TransformerNUFFT"): -60242552.89874968,
}


fit = al.FitInterferometer(dataset=dataset, tracer=tracer, settings=settings, xp=np)
reference_mode = "eager_numpy" if n_visibilities <= EAGER_REFERENCE_MAX_VIS else "jit_jax"
library_path = {"status": "ok", "where": "eager FitInterferometer reference"}

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

    # LIBRARY PATH as-is. TransformerNUFFT.transform_mapping_matrix is one
    # nufft2d2 over every column and every visibility; under pure-JAX fp64
    # nufftax (>= 0.6) on GPU its interpolation materialises
    # O(N_vis x N_gauss x nspread^2) — 61 GiB at alma, 300 GiB at alma_high,
    # 1.46 TiB at jvla on the A100 (#308 phase B). A device OOM here is the
    # VRAM re-test result: it is recorded as ``library_path`` and the run
    # continues on the --transform-chunk arm, with the laptop CPU pins as the
    # reference.
    try:
        with timer.section("fit_interferometer_jit_reference"):
            _ll, _fom = jax.jit(_reference_fn)(params_tree)
            log_likelihood_ref, figure_of_merit_ref = float(_ll), float(_fom)
        library_path = {"status": "ok", "where": "jax.jit(FitInterferometer) reference"}
    except Exception as exc:  # noqa: BLE001 — only a device OOM is recorded
        if not _is_oom(exc):
            raise
        library_path = _oom_record(exc, "jax.jit(FitInterferometer) reference")
        _pin_key = (instrument, transformer_name)
        figure_of_merit_ref = EXPECTED_FIGURE_OF_MERIT.get(_pin_key)
        log_likelihood_ref = EXPECTED_LOG_LIKELIHOOD.get(_pin_key)
        reference_mode = "cpu_pinned" if figure_of_merit_ref is not None else "none"
        if TRANSFORM_COLUMN_BATCH is None:
            raise RuntimeError(
                "library path OOMed and no --transform-chunk arm was requested"
            ) from exc

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
if reference_mode in ("eager_numpy", "jit_jax"):
    pass
elif reference_mode == "cpu_pinned":
    print("  reference = laptop fp64 JAX-CPU pins (library path OOMed on this device)")
else:
    print("  reference = none (library path OOMed and no CPU pin for this instrument)")
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


def _transform_library(mapping_matrix):
    return dataset.transformer.transform_mapping_matrix(mapping_matrix=mapping_matrix, xp=jnp)


TRANSFORM_VIS_CHUNK = (
    _cell_args.transform_vis_chunk
    if _cell_args.transform_vis_chunk is not None
    else transformer_chunk_size
)
if TRANSFORM_COLUMN_BATCH is not None:
    if USE_DFT:
        raise ValueError("--transform-chunk is a TransformerNUFFT arm")
    # The arm's own transformer: same uv / mask, visibility-chunked forward NUFFT.
    _chunk_transformer = al.TransformerNUFFT(
        uv_wavelengths=np.asarray(dataset.uv_wavelengths),
        real_space_mask=real_space_mask,
        chunk_size=TRANSFORM_VIS_CHUNK,
    )
    _slim_rows, _slim_cols = real_space_mask.slim_to_native_tuple
    _slim_rows = jnp.asarray(_slim_rows)
    _slim_cols = jnp.asarray(_slim_cols)
    _n_y, _n_x = real_space_mask.shape_native


def _transform_script_chunked(mapping_matrix):
    """Measurement-only prototype of a chunked ``transform_mapping_matrix``.

    Same arithmetic as the library (scatter each column into the native image,
    row-flip, ``nufft2d2``, phase shift — ``TransformerNUFFT._forward_native``
    does the flip and shift), but over columns with ``jax.lax.map``
    (``TRANSFORM_COLUMN_BATCH`` columns vmapped per step) and, inside each
    column, over visibility chunks of ``TRANSFORM_VIS_CHUNK`` (the transformer's
    own ``lax.scan``). Peak interpolation buffer ~ batch x chunk x nspread² x 16 B
    instead of N_gauss x N_vis x nspread² x 16 B. No library edit.
    """

    def one_column(column):
        image = jnp.zeros((_n_y, _n_x), dtype=jnp.complex128)
        image = image.at[_slim_rows, _slim_cols].set(column.astype(jnp.complex128))
        return _chunk_transformer._forward_native(image, xp=jnp)

    vis = jax.lax.map(one_column, mapping_matrix.T, batch_size=TRANSFORM_COLUMN_BATCH)
    return vis.T


_transform = _transform_script_chunked if TRANSFORM_COLUMN_BATCH is not None else _transform_library
transform_config = {
    "transform": "script_chunked" if TRANSFORM_COLUMN_BATCH is not None else "library",
    "transform_column_batch": TRANSFORM_COLUMN_BATCH,
    "transform_vis_chunk": TRANSFORM_VIS_CHUNK if TRANSFORM_COLUMN_BATCH is not None else None,
}
print(f"  step-3 transform: {transform_config}")


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

# With the chunked arm on and the library path alive, the library's one-shot
# transform is timed as a guarded sub-row too (the lever's before/after).
transformed_mm_library_max_rel_diff = None
if TRANSFORM_COLUMN_BATCH is not None and library_path["status"] == "ok":
    try:
        _, _tmm_lib = jit_profile(
            _transform_library, "transform_mapping_matrix_library", mapping_matrix_jnp
        )
        sub_rows["transform_mapping_matrix (library one-shot)"] = _per_call(
            "transform_mapping_matrix_library"
        )
        transformed_mm_library_max_rel_diff = float(
            jnp.max(jnp.abs(_tmm_lib - transformed_mm_jit)) / jnp.max(jnp.abs(_tmm_lib))
        )
        print(
            f"  chunked vs library transform max rel diff: {transformed_mm_library_max_rel_diff:.3e}"
        )
        del _tmm_lib
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        library_path["standalone_transform"] = _oom_record(exc, "library transform_mapping_matrix")

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


def _abs_diff(a, b):
    return None if a is None or b is None else float(abs(a - b))


figure_of_merit_abs_diff = _abs_diff(figure_of_merit_steps, figure_of_merit_ref)
log_likelihood_abs_diff = _abs_diff(log_likelihood_steps, log_likelihood_ref)
floor_bias_nats = (
    log_likelihood_ref - figure_of_merit_ref
    if log_likelihood_ref is not None and figure_of_merit_ref is not None
    else log_likelihood_steps - figure_of_merit_steps
)
print(f"  figure_of_merit (step-by-step)     = {figure_of_merit_steps}")
print(f"  figure_of_merit (FitInterferometer) = {figure_of_merit_ref}")
print(f"  |diff| = {figure_of_merit_abs_diff} nats (atol {LOG_L_ATOL_NATS:g}; {reference_mode})")
print(f"  log_likelihood (residual form) |diff vs reference| = {log_likelihood_abs_diff} nats")
print(f"  log_likelihood - figure_of_merit (diagonal-floor bias) = {floor_bias_nats:.6e} nats")
if figure_of_merit_ref is not None:
    np.testing.assert_allclose(
        figure_of_merit_steps,
        figure_of_merit_ref,
        rtol=0.0,
        atol=LOG_L_ATOL_NATS,
        err_msg="interferometer/mge: step-by-step figure of merit does not match the reference",
    )
if log_likelihood_ref is not None:
    np.testing.assert_allclose(
        log_likelihood_steps,
        log_likelihood_ref,
        rtol=0.0,
        atol=LOG_L_ATOL_NATS,
        err_msg="interferometer/mge: residual-form log L does not match the reference",
    )
print(f"  Assertions vs reference ({reference_mode}): PASSED or skipped where no reference exists")

full_pipeline_per_call = None
full_pipeline_logl = None
full_pipeline_abs_diff = None
full_pipeline_status: dict = {"status": "not_run"}

w_tilde_steps: dict[str, float] | None = None
w_tilde: dict | None = None
vmap_steps: dict[str, float] | None = None
vmap_split: dict[str, float] | None = None
vmap_error: str | None = None

# Library W~ route arm (--sparse), filled in PART C2.
library_sparse: dict | None = None
library_sparse_steps: dict[str, float] | None = None


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
    if library_sparse_steps:
        print(f"  Library W~ route rows ({(library_sparse or {}).get('inversion_class')}):")
        for label, per_call in library_sparse_steps.items():
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
            # With --sparse the top-level rows are the library W~ route; the dense
            # rows of the same run move to ``dense_steps`` (before/after in one JSON).
            "inversion_path": "sparse" if USE_LIBRARY_SPARSE else "dense",
            "inversion_class": (
                (library_sparse or {}).get("inversion_class")
                if USE_LIBRARY_SPARSE
                else type(inversion).__name__
            ),
            "dense_inversion_class": type(inversion).__name__,
            "transformer": transformer_name,
            "chunk_size": transformer_chunk_size,
            "chunk_size_note": (
                "TransformerNUFFT.transform_mapping_matrix ignores chunk_size: step 3 is one "
                "nufft2d2 over every column and every visibility."
            ),
            "nnls_solver": solver,
            "nnls_preconditioning": preconditioning,
            "nufftax_version": _nufftax_version(),
            **transform_config,
            "w_tilde_arm": USE_W_TILDE,
            "library_sparse_arm": USE_LIBRARY_SPARSE,
            "n_repeats": N_REPEATS,
            "thread_env": _observe_thread_env(),
            "host_load_avg_start": _load_avg_start,
            "host_load_avg_end": list(os.getloadavg()),
        },
        "reference_mode": reference_mode,
        "library_path": library_path,
        "full_pipeline_status": full_pipeline_status,
        "transformed_mapping_matrix_chunked_vs_library_max_rel_diff": (
            transformed_mm_library_max_rel_diff
        ),
        "log_likelihood_reference": log_likelihood_ref,
        "figure_of_merit_reference": figure_of_merit_ref,
        "figure_of_merit_step_by_step": figure_of_merit_steps,
        "log_likelihood_step_by_step": log_likelihood_steps,
        "figure_of_merit_full_pipeline": full_pipeline_logl,
        "figure_of_merit_abs_diff_nats": figure_of_merit_abs_diff,
        "log_likelihood_abs_diff_nats": log_likelihood_abs_diff,
        "full_pipeline_abs_diff_nats": full_pipeline_abs_diff,
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
        "source_revisions": _source_revisions,
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

    if USE_LIBRARY_SPARSE:
        # The dashboard reads ``steps`` / ``total_step_by_step`` and labels the
        # row by ``configuration.inversion_path``: they are the library W~ route
        # here, and the dense rows of this same run are kept alongside.
        breakdown_summary["dense_steps"] = breakdown_summary["steps"]
        breakdown_summary["dense_total_step_by_step"] = step_total
        breakdown_summary["dense_setup_split"] = breakdown_summary["setup_split"]
        breakdown_summary["steps"] = {k: float(v) for k, v in (library_sparse_steps or {}).items()}
        breakdown_summary["total_step_by_step"] = (
            float(sum(library_sparse_steps.values())) if library_sparse_steps else None
        )
        breakdown_summary["setup_split"] = (library_sparse or {}).get("setup_split")
        breakdown_summary["library_sparse"] = library_sparse

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
    if library_sparse_steps:
        labels += [f"Library W~: {label}" for label in library_sparse_steps]
        times += list(library_sparse_steps.values())
        colors += ["#C44E52"] * len(library_sparse_steps)

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


# The library's fused pipeline (it runs the library transform, never the arm's).
if library_path["status"] == "ok":
    try:
        _, full_pipeline_logl = jit_profile(full_pipeline, "full_pipeline", params_tree)
        full_pipeline_per_call = _per_call("full_pipeline")
        full_pipeline_logl = float(full_pipeline_logl)
        full_pipeline_status = {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        full_pipeline_status = _oom_record(exc, "AnalysisInterferometer full pipeline")
else:
    full_pipeline_status = {"status": "skipped", "reason": "library path OOMed (same transform)"}
if full_pipeline_logl is not None:
    full_pipeline_abs_diff = _abs_diff(full_pipeline_logl, figure_of_merit_ref)
    print(
        f"  full-pipeline value = {full_pipeline_logl} "
        f"(|diff vs figure_of_merit| = {full_pipeline_abs_diff} nats)"
    )
    if figure_of_merit_ref is not None:
        np.testing.assert_allclose(
            full_pipeline_logl,
            figure_of_merit_ref,
            rtol=0.0,
            atol=LOG_L_ATOL_NATS,
            err_msg="interferometer/mge: full-pipeline JIT does not match the reference",
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

_sparse_dataset_cache: dict = {}


def sparse_dataset():
    """``apply_sparse_operator()`` once, shared by the W~ arm and the library arm.

    ``apply_sparse_operator`` runs two eager (op-by-op) type-1 NUFFTs: the
    precision-operator build and the dirty image (``transformer.image_from``).
    Both honour a visibility chunk size — the builder through
    ``nufft_chunk_size``, the dirty image through the transformer's own
    ``chunk_size`` — so above the eager threshold the operator is built from a
    copy of the dataset whose transformer is chunked. Unchunked, the build was
    killed at 9.2-9.6 GB RSS at alma on the laptop. The operator and dirty image
    are the same quantities either way (chunking only bounds the spread buffer),
    and the F~ / D~ checks and the library-vs-dense witness verify it.

    Returns ``(dataset_w, build_s, build_peak_rss_mb)``.
    """
    if "dataset_w" not in _sparse_dataset_cache:
        with timer.section("w_tilde_apply_sparse_operator"):
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
            _sparse_dataset_cache["dataset_w"] = dataset_for_operator.apply_sparse_operator(
                use_jax=True, nufft_chunk_size=w_tilde_nufft_chunk_size
            )
        _sparse_dataset_cache["build_s"] = timer.records[-1][1]
        _sparse_dataset_cache["build_peak_rss_mb"] = _peak_rss_mb()
    return (
        _sparse_dataset_cache["dataset_w"],
        _sparse_dataset_cache["build_s"],
        _sparse_dataset_cache["build_peak_rss_mb"],
    )


if USE_W_TILDE:
    print("\n" + "=" * 70)
    print("W~ ARM (func-list W~ curvature, measurement only)")
    print("=" * 70)
    try:
        dataset_w, w_tilde_build_s, w_tilde_build_peak_rss_mb = sparse_dataset()
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
            "figure_of_merit_abs_diff_vs_fit_nats": _abs_diff(
                figure_of_merit_w, figure_of_merit_ref
            ),
            "figure_of_merit_abs_diff_vs_step_by_step_nats": _abs_diff(
                figure_of_merit_w, figure_of_merit_steps
            ),
            "log_likelihood_unfloored": log_likelihood_w,
            "log_likelihood_abs_diff_vs_fit_nats": _abs_diff(log_likelihood_w, log_likelihood_ref),
            "operator_build_peak_rss_mb": float(w_tilde_build_peak_rss_mb),
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
            f"(|diff vs reference| = {w_tilde['figure_of_merit_abs_diff_vs_fit_nats']} nats, "
            f"vs step-by-step {w_tilde['figure_of_merit_abs_diff_vs_step_by_step_nats']:.3e}); "
            f"unfloored log L |diff vs reference| = "
            f"{w_tilde['log_likelihood_abs_diff_vs_fit_nats']} nats"
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
# PART C2 — Library W~ route (--sparse): the MGE-only fit on the library path
# ===================================================================
#
# Since PyAutoArray#575 the inversion factory no longer switches the sparse
# operator off for an all-func-list inversion, so ``FitInterferometer`` on a
# dataset with ``apply_sparse_operator()`` applied takes
# ``InversionInterferometerSparse`` for this MGE-only model. This arm times that
# library path in the dense arm's schema (nested ``params -> stage`` prefixes,
# successive differences, ``jit_phases``) and checks its jitted
# ``log_likelihood`` / ``figure_of_merit`` against the dense reference.
#
# Witness: ``|Δ log L| <= 1e-6`` nats against the dense library reference where
# the dense reference runs on this device, and no device OOM.

# Structural dense-vs-sparse witness threshold (fp64); mixed precision is recorded only.
LIBRARY_SPARSE_WITNESS_NATS = 1e-6
library_sparse_failed = False


def _sparse_vmap_batches() -> list[int]:
    if _cell_args.sparse_vmap_batch is not None:
        return sorted({int(b) for b in _cell_args.sparse_vmap_batch.split(",")}, reverse=True)
    return [int(_vmap_batch)] if _vmap_batch is not None else []


def _tracer_with_lens_sersic(params):
    """The cell's tracer plus an ordinary (non-linear) Sersic on the lens galaxy.

    Exercises the sparse dirty-image correction (``d~ - W~ i_p``,
    ``autogalaxy.interferometer.fit_interferometer.sparse_dirty_image_from``):
    with a regular light profile the inversion fits the profile-subtracted
    visibilities, and the cached dirty image alone would be the wrong D.
    """
    galaxies = list(params.galaxies)
    lens_with_light = al.Galaxy(
        redshift=0.5,
        mass=galaxies[0].mass,
        light=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(0.05, 0.0),
            intensity=0.1,
            effective_radius=0.6,
            sersic_index=2.5,
        ),
    )
    return al.Tracer(galaxies=[lens_with_light, *galaxies[1:]], fields=[params.fields])


if USE_LIBRARY_SPARSE:
    print("\n" + "=" * 70)
    print("LIBRARY W~ ROUTE (--sparse: apply_sparse_operator + FitInterferometer)")
    print("=" * 70)
    library_sparse = {"status": "running"}
    try:
        dataset_sparse, _build_s, _build_rss = sparse_dataset()
        library_sparse["operator_build_s"] = float(_build_s)
        library_sparse["operator_build_peak_rss_mb"] = float(_build_rss)
        library_sparse["operator_build_nufft_chunk_size"] = w_tilde_nufft_chunk_size
        library_sparse["fit_transformer_chunk_size"] = getattr(
            dataset_sparse.transformer, "chunk_size", None
        )

        # Structural: which inversion class the library picks (no mapping matrix evaluated).
        _inversion_sparse = al.TracerToInversion(
            dataset=aa.DatasetInterface(
                data=dataset.data,
                noise_map=dataset.noise_map,
                grids=dataset.grids,
                transformer=dataset_sparse.transformer,
                sparse_operator=dataset_sparse.sparse_operator,
            ),
            tracer=tracer,
            settings=settings,
            xp=np,
        ).inversion
        library_sparse["inversion_class"] = type(_inversion_sparse).__name__
        library_sparse["nnls_solver"] = _inversion_sparse.positive_only_solver_used
        library_sparse["nnls_preconditioning"] = (
            _inversion_sparse.positive_only_preconditioning_used
        )
        print(f"  inversion class = {library_sparse['inversion_class']}")
        if library_sparse["inversion_class"] != "InversionInterferometerSparse":
            raise RuntimeError(
                "MGE-only fit on a sparse dataset did not take InversionInterferometerSparse "
                f"(got {library_sparse['inversion_class']}): are the #575 libraries imported?"
            )

        def _fit_sparse(params):
            return al.FitInterferometer(
                dataset=dataset_sparse, tracer=_tracer_from(params), settings=settings, xp=jnp
            )

        def sparse_prefix_fn(upto: int):
            """Nested ``params -> stage`` prefix on the library W~ route."""

            def fn(params):
                if upto <= 2:
                    return setup_prefix_fn(upto)(params)
                fit_w = _fit_sparse(params)
                if upto == 3:
                    return fit_w.inversion.data_vector, fit_w.inversion.curvature_reg_matrix
                if upto == 4:
                    return fit_w.inversion.reconstruction
                return fit_w.figure_of_merit

            return fn

        sparse_prefix_labels = {
            1: "Ray-trace grids",
            2: f"Mapping matrix ({total_gaussians} Gaussians)",
            3: "Data vector + curvature matrix (library W~: Bᵀ d~, Bᵀ W~ B)",
            4: "Reconstruction (PDIP NNLS)",
            5: "Fast chi-squared + figure of merit (FitInterferometer)",
        }
        sparse_prefix_per_call: dict[int, float] = {}
        _sparse_prefix_outputs: dict[int, object] = {}
        for _upto in sorted(sparse_prefix_labels):
            print(f"\n--- Library W~ prefix 1..{_upto}: {sparse_prefix_labels[_upto]} ---")
            _, _sparse_prefix_outputs[_upto] = jit_profile(
                sparse_prefix_fn(_upto), f"library_sparse_prefix_{_upto}", params_tree
            )
            sparse_prefix_per_call[_upto] = _per_call(f"library_sparse_prefix_{_upto}")
            print(f"  peak RSS so far: {_peak_rss_mb():.0f} MB")
        library_sparse_setup_split = timing.split_by_successive_differences(
            sparse_prefix_per_call, sparse_prefix_labels
        )
        library_sparse_steps = dict(library_sparse_setup_split)
        figure_of_merit_sparse = float(_sparse_prefix_outputs[5])

        _dv_w, _f_w = _sparse_prefix_outputs[3]
        library_sparse["data_vector_max_rel_diff_vs_dense"] = float(
            jnp.max(jnp.abs(_dv_w - data_vector)) / jnp.max(jnp.abs(data_vector))
        )
        library_sparse["curvature_reg_matrix_max_rel_diff_vs_dense"] = float(
            jnp.max(jnp.abs(_f_w - curvature_matrix)) / jnp.max(jnp.abs(curvature_matrix))
        )

        print("\n--- Library W~: jax.jit(FitInterferometer.log_likelihood) ---")
        _, _ll_sparse = jit_profile(
            lambda params: _fit_sparse(params).log_likelihood,
            "library_sparse_fit_log_likelihood",
            params_tree,
        )
        log_likelihood_sparse = float(_ll_sparse)

        print("\n--- Library W~: full pipeline (AnalysisInterferometer on the sparse dataset) ---")
        analysis_sparse = al.AnalysisInterferometer(
            dataset=dataset_sparse, settings=settings, use_jax=True
        )

        def full_pipeline_sparse(params):
            return analysis_sparse.log_likelihood_function(instance=params)

        _, _fp_sparse = jit_profile(
            full_pipeline_sparse, "library_sparse_full_pipeline", params_tree
        )
        full_pipeline_sparse_logl = float(_fp_sparse)

        _ll_diff = _abs_diff(log_likelihood_sparse, log_likelihood_ref)
        _fom_diff = _abs_diff(figure_of_merit_sparse, figure_of_merit_ref)
        _reference_is_dense_here = reference_mode in ("eager_numpy", "jit_jax")
        library_sparse.update(
            {
                "status": "ok",
                "setup_split": {k: float(v) for k, v in library_sparse_setup_split.items()},
                "setup_prefix_per_call_s": {
                    str(k): float(v) for k, v in sparse_prefix_per_call.items()
                },
                "total_step_by_step": float(sum(library_sparse_steps.values())),
                "fit_log_likelihood_s": _per_call("library_sparse_fit_log_likelihood"),
                "full_pipeline_single_jit": _per_call("library_sparse_full_pipeline"),
                "log_likelihood": log_likelihood_sparse,
                "figure_of_merit": figure_of_merit_sparse,
                "figure_of_merit_full_pipeline": full_pipeline_sparse_logl,
                "reference_mode": reference_mode,
                "log_likelihood_abs_diff_vs_reference_nats": _ll_diff,
                "figure_of_merit_abs_diff_vs_reference_nats": _fom_diff,
                "full_pipeline_abs_diff_vs_reference_nats": _abs_diff(
                    full_pipeline_sparse_logl, figure_of_merit_ref
                ),
                "witness_threshold_nats": LIBRARY_SPARSE_WITNESS_NATS,
                "witness_pass": (
                    bool(
                        _ll_diff <= LIBRARY_SPARSE_WITNESS_NATS
                        and _fom_diff <= LIBRARY_SPARSE_WITNESS_NATS
                    )
                    if _reference_is_dense_here and not _cli.use_mixed_precision
                    else None
                ),
                "witness_note": (
                    "dense library reference ran on this device"
                    if _reference_is_dense_here
                    else f"no dense reference on this device ({reference_mode}); diffs are "
                    "against the laptop fp64 CPU pins where they exist"
                ),
                "dense_full_pipeline_single_jit": full_pipeline_per_call,
                "dense_total_step_by_step": float(sum(t for _, t in likelihood_steps)),
            }
        )
        if w_tilde is not None and "figure_of_merit" in w_tilde:
            library_sparse["figure_of_merit_abs_diff_vs_w_tilde_arm_nats"] = _abs_diff(
                figure_of_merit_sparse, w_tilde["figure_of_merit"]
            )
            library_sparse["log_likelihood_abs_diff_vs_w_tilde_arm_nats"] = _abs_diff(
                log_likelihood_sparse, w_tilde["log_likelihood_unfloored"]
            )
        print(
            f"  library W~ log_likelihood  = {log_likelihood_sparse} "
            f"(|Δ vs {reference_mode}| = {_ll_diff} nats)"
        )
        print(
            f"  library W~ figure_of_merit = {figure_of_merit_sparse} "
            f"(|Δ vs {reference_mode}| = {_fom_diff} nats); witness_pass = "
            f"{library_sparse['witness_pass']}"
        )
        write_results("library_sparse")

        # MGE + lens Sersic: sparse vs dense, where the dense library fit runs here.
        if _reference_is_dense_here:
            print("\n--- Library W~: MGE + lens Sersic variant (dirty-image correction) ---")

            def _sersic_values(ds):
                def fn(params):
                    fit_s = al.FitInterferometer(
                        dataset=ds,
                        tracer=_tracer_with_lens_sersic(params),
                        settings=settings,
                        xp=jnp,
                    )
                    return fit_s.log_likelihood, fit_s.figure_of_merit

                return fn

            with timer.section("lens_sersic_sparse"):
                _ll_s, _fom_s = jax.jit(_sersic_values(dataset_sparse))(params_tree)
                _ll_s, _fom_s = float(_ll_s), float(_fom_s)
            with timer.section("lens_sersic_dense"):
                _ll_d, _fom_d = jax.jit(_sersic_values(dataset))(params_tree)
                _ll_d, _fom_d = float(_ll_d), float(_fom_d)
            # Control: the same sparse fit with the dirty-image correction switched
            # off (the pre-#575 behaviour), so the check is shown to discriminate.
            import autolens.interferometer.fit_interferometer as _lens_fit_module

            _correction = _lens_fit_module.sparse_dirty_image_from
            _lens_fit_module.sparse_dirty_image_from = lambda **_kwargs: None
            try:
                with timer.section("lens_sersic_sparse_uncorrected"):
                    _ll_u, _ = jax.jit(_sersic_values(dataset_sparse))(params_tree)
                    _ll_u = float(_ll_u)
            finally:
                _lens_fit_module.sparse_dirty_image_from = _correction
            library_sparse["lens_sersic_variant"] = {
                "sparse_log_likelihood": _ll_s,
                "dense_log_likelihood": _ll_d,
                "sparse_figure_of_merit": _fom_s,
                "dense_figure_of_merit": _fom_d,
                "log_likelihood_abs_diff_nats": _abs_diff(_ll_s, _ll_d),
                "figure_of_merit_abs_diff_nats": _abs_diff(_fom_s, _fom_d),
                "control_sparse_uncorrected_log_likelihood": _ll_u,
                "control_uncorrected_abs_diff_nats": _abs_diff(_ll_u, _ll_d),
                "note": (
                    "Lens Sersic(ell_comps=(0.05,0), I=0.1, R_eff=0.6, n=2.5) added to the "
                    "cell's MGE-only model; single jitted calls (first call, compile included)."
                ),
            }
            print(
                f"  sparse vs dense: |Δ log L| = "
                f"{library_sparse['lens_sersic_variant']['log_likelihood_abs_diff_nats']} nats, "
                f"|Δ FoM| = {library_sparse['lens_sersic_variant']['figure_of_merit_abs_diff_nats']}"
            )
        else:
            library_sparse["lens_sersic_variant"] = {
                "status": "skipped",
                "reason": f"no dense library reference on this device ({reference_mode})",
            }

        # vmap probe of the sparse full pipeline (VRAM rows): largest batch first.
        library_sparse["vmap"] = []
        for _batch in _sparse_vmap_batches():
            print(f"\n--- Library W~ full pipeline under vmap, batch {_batch} ---")
            _params_b = jax.tree_util.tree_map(
                lambda leaf, n=_batch: jnp.broadcast_to(leaf, (n, *leaf.shape)), params_tree
            )
            try:
                _per = timing.vmap_profile(
                    full_pipeline_sparse,
                    "library_sparse_full_pipeline",
                    _params_b,
                    _batch,
                    n_repeats=N_REPEATS,
                    timer=timer,
                    jit_records=jit_records,
                )
                library_sparse["vmap"].append(
                    {"batch": _batch, "status": "ok", "per_call_s": float(_per)}
                )
                break
            except Exception as exc:  # noqa: BLE001 — only a device OOM is recorded
                if not _is_oom(exc):
                    raise
                library_sparse["vmap"].append(
                    {"batch": _batch, **_oom_record(exc, f"library W~ vmap batch {_batch}")}
                )
    except Exception as exc:  # noqa: BLE001 — recorded, then the run exits non-zero
        if _is_oom(exc):
            library_sparse.update(_oom_record(exc, "library W~ route"))
        else:
            library_sparse_failed = True
            library_sparse.update({"status": "error", "error": traceback.format_exc()})
            print("  LIBRARY W~ ROUTE FAILED. Traceback:")
            print(library_sparse["error"])

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
        vmap_steps = {}
        if full_pipeline_status.get("status") == "ok":
            vmap_steps["Full pipeline"] = timing.vmap_profile(
                full_pipeline,
                "full_pipeline",
                _params_batched,
                _vmap_batch,
                n_repeats=N_REPEATS,
                timer=timer,
                jit_records=jit_records,
            )
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

# (EXPECTED_FIGURE_OF_MERIT is defined next to the reference, above.)

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
        if _value is None or (_label == "reference" and reference_mode == "cpu_pinned"):
            continue
        _rec = check_pinned(_value, _pinned_expected, label=_label, rtol=_rtol)
        if _rec is not None:
            _pinned_drift.append(_rec)
record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")

# ===================================================================
# Library W~ route — fail loudly (after every result is on disk)
# ===================================================================
#
# A library error, or a library-vs-reference gap beyond the cell's own
# ``LOG_L_ATOL_NATS``, exits non-zero so a SLURM "COMPLETED" cannot hide it. A
# device OOM is a recorded result (``library_sparse.status = "oom"``), as in the
# dense arms.

if USE_LIBRARY_SPARSE:
    if library_sparse_failed:
        raise SystemExit("interferometer/mge: library W~ route raised (see library_sparse.error)")
    for _key in (
        "log_likelihood_abs_diff_vs_reference_nats",
        "figure_of_merit_abs_diff_vs_reference_nats",
    ):
        _gap = (library_sparse or {}).get(_key)
        if _gap is not None and _gap > LOG_L_ATOL_NATS:
            raise SystemExit(
                f"interferometer/mge: library W~ route {_key} = {_gap} > {LOG_L_ATOL_NATS} nats"
            )
    print(
        f"  Library W~ route: status={library_sparse.get('status')}, "
        f"witness_pass={library_sparse.get('witness_pass')}"
    )
