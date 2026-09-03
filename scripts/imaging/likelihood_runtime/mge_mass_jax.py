"""
JAX Profiling: fixed-MGE lens *mass* imaging likelihood — the trace-time deflection fold
=======================================================================================

The JAX sibling of ``pixelization_numba_mge_mass.py``, for the second phase of the
fixed-geometry deflection memo (PyAutoGalaxy#604).

The model is the SLaM ``mass_light_dark`` shape: a 30-component ``lmp_linear.Gaussian``
basis whose centre, ``ell_comps``, ``intensity`` and ``sigma`` are all fixed and which
shares **one** free ``mass_to_light_ratio``, plus a dark ``NFWSph``, an
``ExternalShear`` and a rectangular pixelized source. Every likelihood evaluation
ray-traces 30 Gaussians whose geometry never moves.

On the numpy path phase 1 memoised that field across evaluations. On the JAX path there
is nothing to memoise *across* calls — there is only one call, the compiled one — so the
memo instead folds the field out of the trace: it evaluates the unit-ratio field once,
at trace time, with numpy and ``scipy.special.wofz``, and hands the trace a constant that
the traced ratio multiplies. Two facts make that necessary rather than redundant:
``jax.jit`` stages every ``jax.numpy`` call whatever its operands, and this stack
disables XLA's constant folding (``--xla_disable_hlo_passes=constant_folding``, set by
``autonerves/jax_wrapper.py``).

What is measured
----------------

Both legs run in one process — memo off (``deflections_memo.memo_disabled()``) then memo
on — and the script reports:

1. **timings**: ``vmap_first_call`` (trace + compile, which is where the fold is paid)
   and ``vmap_steady_x10``, three repeats each, memo off vs memo on.
2. **witness**: ``mge._wofz`` call counts split by backend. Memo on, compiling call:
   N numpy-backend calls and **zero** jnp-backend calls — the Faddeeva evaluation
   happened in scipy, at trace time. Memo off: zero numpy and N jnp. Steady state: zero
   on both legs, because a compiled program calls no Python.
3. **jaxpr size**: equations in the vmapped fitness's jaxpr, counted recursively,
   memo off vs memo on. The delta is the Faddeeva block leaving the graph.
4. **agreement**: the log likelihood memo-off vs memo-on. The two differ only in
   ``scipy.special.wofz`` versus the Weideman-32 series, so they must agree to ~1e-13.
5. **controls**: a model with a free ``grid_offset`` makes the grid itself a tracer, so
   the fold cannot fire and the jnp-backend counter is non-zero again with the value
   unchanged; and the ``AUTOGALAXY_DEFLECTIONS_MEMO=0`` kill switch reproduces the
   memo-off leg exactly.

Output
------
``results/runtime/imaging/mge_mass_jax/mge_mass_jax_likelihood_summary_<instrument>_v<version>.json``
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

_sys.path.insert(0, str(_profiling_root()))

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os  # noqa: E402
import sys as _smoke_sys  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402
from autogalaxy.profiles.mass.abstract import deflections_memo as _memo  # noqa: E402
from autogalaxy.profiles.mass.abstract import mge as _mge  # noqa: E402
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "hst"

MASK_RADIUS = 3.5
TOTAL_GAUSSIANS = 30
SIGMA_MIN = 0.01
MESH_PIXELS_YX = 28
BATCH_SIZE = 3
N_STEADY = 10
N_REPEATS = 3

# The log-likelihood this model produces under the JAX likelihood, per (rect mesh,
# instrument). Filled from the first run of a new key (the run prints the value and says
# so); checked at rtol 1e-6 thereafter.
EXPECTED_LOG_LIKELIHOOD: dict[str, dict[str, float]] = {
    "bilinear": {"hst": -56107.56407588643},
}


# ---------------------------------------------------------------------------
# `_wofz` call-count witness
# ---------------------------------------------------------------------------
#
# `mge.zeta_from` reaches the Faddeeva function through the module-global `_wofz` (both
# directly and via `_wofz_masked`), so wrapping that global counts every call on either
# backend. The split is the whole point: a numpy-backend call under a JAX fit is the
# fold happening, a jnp-backend call is the subgraph still being traced.

_wofz_counts = {"numpy": 0, "jnp": 0}
_wofz_original = _mge._wofz


def _wofz_counted(z, xp=np):
    _wofz_counts["numpy" if xp is np else "jnp"] += 1
    return _wofz_original(z, xp=xp)


_mge._wofz = _wofz_counted


def _reset_wofz_counts():
    _wofz_counts["numpy"] = 0
    _wofz_counts["jnp"] = 0


def _wofz_snapshot() -> dict:
    return dict(_wofz_counts)


# ---------------------------------------------------------------------------
# jaxpr size
# ---------------------------------------------------------------------------


def _count_equations(jaxpr) -> int:
    """
    Equations in ``jaxpr``, counting the sub-jaxprs of every higher-order primitive
    (``pjit``, ``scan``, ``cond``, ``custom_jvp_call`` ...) recursively.
    """
    inner = getattr(jaxpr, "jaxpr", jaxpr)

    total = 0

    for equation in inner.eqns:
        total += 1
        for value in equation.params.values():
            for candidate in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(candidate, "eqns") or hasattr(candidate, "jaxpr"):
                    total += _count_equations(candidate)

    return total


def block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


# ===================================================================
# PART A — dataset
# ===================================================================

print(f"\n--- Dataset loading & masking [{instrument}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=instrument,
    workspace_root=_workspace_root,
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=pixel_scale,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=MASK_RADIUS,
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

# ===================================================================
# PART B — model (the SLaM mass_light_dark shape)
# ===================================================================

print("\n--- Model construction (fixed MGE light-and-mass + free ratio) ---")

log10_sigma_list = np.linspace(np.log10(SIGMA_MIN), np.log10(MASK_RADIUS), TOTAL_GAUSSIANS)

_bulge_ell = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)


def basis_model_from():
    gaussian_list = af.Collection(af.Model(al.lmp_linear.Gaussian) for _ in range(TOTAL_GAUSSIANS))

    mass_to_light_ratio = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)

    for index, gaussian in enumerate(gaussian_list):
        gaussian.centre = (0.0, 0.0)
        gaussian.ell_comps = (float(_bulge_ell[0]), float(_bulge_ell[1]))
        gaussian.sigma = float(10 ** log10_sigma_list[index])
        gaussian.intensity = 1.0
        gaussian.mass_to_light_ratio = mass_to_light_ratio

    return af.Model(al.lp_basis.Basis, profile_list=gaussian_list)


def model_from(free_grid_offset: bool = False):
    dark = af.Model(al.mp.NFWSph)
    dark.centre = (0.0, 0.0)
    dark.kappa_s = af.GaussianPrior(mean=0.2, sigma=0.01)
    dark.scale_radius = 10.0

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=basis_model_from(), dark=dark, shear=shear)

    pixelization = al.Pixelization(
        mesh=rect_mesh_classes(_cli)[0](shape=(MESH_PIXELS_YX, MESH_PIXELS_YX)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    galaxies = af.Collection(lens=lens, source=source)

    if not free_grid_offset:
        return af.Collection(galaxies=galaxies)

    dataset_model = af.Model(al.DatasetModel)
    dataset_model.grid_offset.grid_offset_0 = af.GaussianPrior(mean=0.0, sigma=0.01)
    dataset_model.grid_offset.grid_offset_1 = af.GaussianPrior(mean=0.0, sigma=0.01)
    dataset_model.grid_rotation_angle = 0.0
    dataset_model.background_sky_level = 0.0

    return af.Collection(galaxies=galaxies, dataset_model=dataset_model)


model = model_from()

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  MGE components:        {TOTAL_GAUSSIANS}")
print(f"  OMP_NUM_THREADS:       {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"  XLA_FLAGS:             {os.environ.get('XLA_FLAGS', '(unset)')}")


def batched_instance_from(model):
    """
    The pytree-native ``ModelInstance`` at the prior medians, broadcast to a batch.

    ``autofit.jax.register_model`` registers the model's class graph with
    ``jax.tree_util``, so the instance crosses the ``jit`` / ``vmap`` boundary directly
    and free parameters arrive at the profiles as tracers — which is the whole point:
    ``mass_to_light_ratio`` must be traced, or the measurement would be a
    jit-on-concrete that folds itself.
    """
    _register_model_pytrees(model)

    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (BATCH_SIZE, *jnp.shape(leaf))),
        params_tree,
    )


parameters = batched_instance_from(model)


# ===================================================================
# PART C — one measured leg
# ===================================================================


def vmapped_from():
    """
    A *fresh* ``jax.jit(jax.vmap(...))`` of the likelihood.

    Fresh matters: JAX caches compilation on the function object, so re-timing a
    ``vmap_first_call`` needs a new closure or it would time a cache hit.
    """
    analysis = al.AnalysisImaging(
        dataset=dataset, settings=al.Settings(use_border_relocator=True), use_jax=True
    )

    def full_pipeline_from_params(params_tree):
        return analysis.log_likelihood_function(instance=params_tree)

    return jax.jit(jax.vmap(full_pipeline_from_params)), full_pipeline_from_params


def measure(label: str, memo_on: bool, params=parameters) -> dict:
    """
    One leg: ``N_REPEATS`` (fresh trace + compile, then ``N_STEADY`` steady calls), with
    the ``_wofz`` witness and the memo counters read from the first repeat.
    """
    first_calls, steadies = [], []
    witness_compile = witness_steady = None
    memo_stats = None
    log_likelihood = None

    for repeat in range(N_REPEATS):
        if memo_on:
            _memo.memo_clear()

        vmapped, _ = vmapped_from()

        _reset_wofz_counts()

        start = time.perf_counter()
        result = block(vmapped(params))
        first_calls.append(time.perf_counter() - start)

        if repeat == 0:
            witness_compile = _wofz_snapshot()
            memo_stats = _memo.memo_stats()
            log_likelihood = float(np.asarray(result)[0])

        _reset_wofz_counts()

        start = time.perf_counter()
        for _ in range(N_STEADY):
            block(vmapped(params))
        steadies.append((time.perf_counter() - start) / N_STEADY)

        if repeat == 0:
            witness_steady = _wofz_snapshot()

        print(
            f"  [{label}] repeat {repeat + 1}: first_call {first_calls[-1]:.3f} s, "
            f"steady {steadies[-1]:.4f} s/call"
        )

    return {
        "first_call_s": first_calls,
        "first_call_median_s": float(statistics.median(first_calls)),
        "first_call_min_s": float(min(first_calls)),
        "steady_s": steadies,
        "steady_median_s": float(statistics.median(steadies)),
        "steady_min_s": float(min(steadies)),
        "wofz_compile": witness_compile,
        "wofz_steady": witness_steady,
        "memo_stats": memo_stats,
        "log_likelihood": log_likelihood,
    }


def jaxpr_equations(params=parameters) -> int:
    analysis = al.AnalysisImaging(
        dataset=dataset, settings=al.Settings(use_border_relocator=True), use_jax=True
    )

    def full_pipeline_from_params(params_tree):
        return analysis.log_likelihood_function(instance=params_tree)

    return _count_equations(jax.make_jaxpr(jax.vmap(full_pipeline_from_params))(params))


# ===================================================================
# PART D — memo off, then memo on
# ===================================================================

print("\n--- Leg 1: memo OFF ---")

with _memo.memo_disabled():
    off = measure("memo off", memo_on=False)
    off_jaxpr_equations = jaxpr_equations()

print("\n--- Leg 2: memo ON ---")

_memo.memo_clear()
on = measure("memo on", memo_on=True)

_memo.memo_clear()
on_jaxpr_equations = jaxpr_equations()

max_relative = abs(on["log_likelihood"] - off["log_likelihood"]) / max(
    abs(off["log_likelihood"]), 1e-300
)

# ===================================================================
# PART E — controls
# ===================================================================

# ===================================================================
# Summary
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 78)
print(f"FIXED-MGE-MASS JAX LIKELIHOOD — {instrument.upper()} — v{al_version}")
print("=" * 78)
print(f"  vmap batch size:                       {BATCH_SIZE}")
print(f"  memo OFF  first call (median of {N_REPEATS}):  {off['first_call_median_s']:>10.4f} s")
print(f"  memo ON   first call (median of {N_REPEATS}):  {on['first_call_median_s']:>10.4f} s")
print(f"  memo OFF  steady     (median of {N_REPEATS}):  {off['steady_median_s']:>10.4f} s")
print(f"  memo ON   steady     (median of {N_REPEATS}):  {on['steady_median_s']:>10.4f} s")
print(f"  memo OFF  steady     (min of {N_REPEATS}):     {off['steady_min_s']:>10.4f} s")
print(f"  memo ON   steady     (min of {N_REPEATS}):     {on['steady_min_s']:>10.4f} s")
print(
    f"  first-call speed-up (median):          "
    f"{off['first_call_median_s'] / on['first_call_median_s']:>10.2f}x"
)
print(
    f"  steady speed-up (median / min):        "
    f"{off['steady_median_s'] / on['steady_median_s']:>10.2f}x / "
    f"{off['steady_min_s'] / on['steady_min_s']:.2f}x"
)
print("-" * 78)
print(f"  _wofz  memo OFF compile:               {off['wofz_compile']}")
print(f"  _wofz  memo ON  compile:               {on['wofz_compile']}")
print(f"  _wofz  memo OFF steady:                {off['wofz_steady']}")
print(f"  _wofz  memo ON  steady:                {on['wofz_steady']}")
print(f"  memo stats (ON, first repeat):         {on['memo_stats']}")
print("-" * 78)
print(f"  jaxpr equations memo OFF:              {off_jaxpr_equations}")
print(f"  jaxpr equations memo ON:               {on_jaxpr_equations}")
print(f"  delta:                                 {on_jaxpr_equations - off_jaxpr_equations}")
print("-" * 78)
print(f"  log_likelihood memo OFF:               {off['log_likelihood']!r}")
print(f"  log_likelihood memo ON:                {on['log_likelihood']!r}")
print(f"  max relative difference:               {max_relative:.3e}")
print("=" * 78)

print("\n--- Control 1: kill switch (AUTOGALAXY_DEFLECTIONS_MEMO=0) ---")

os.environ["AUTOGALAXY_DEFLECTIONS_MEMO"] = "0"
_memo.memo_clear()

kill_switch_vmapped, _ = vmapped_from()
_reset_wofz_counts()
kill_switch_result = float(np.asarray(block(kill_switch_vmapped(parameters)))[0])
kill_switch_witness = _wofz_snapshot()
kill_switch_memo_stats = _memo.memo_stats()

os.environ.pop("AUTOGALAXY_DEFLECTIONS_MEMO")

print(f"  log_likelihood {kill_switch_result!r}, _wofz {kill_switch_witness}")

print("\n--- Control 2: free grid_offset (the grid is a tracer) ---")

# The flat-vector production path (`Fitness.call`): `instance_from_vector(..., xp=jnp)`
# inside the trace. It is used here rather than the pytree-native instance because a
# `DatasetModel` with a free `grid_offset` cannot round-trip through
# `autofit.jax.register_model`'s tree_unflatten (`TypeError: Only scalar arrays can be
# converted to Python scalars` from `Prior.tree_unflatten`) — a pre-existing limitation
# of the pytree registration, unrelated to the memo, and not this task's to fix.
offset_model = model_from(free_grid_offset=True)

offset_analysis = al.AnalysisImaging(
    dataset=dataset, settings=al.Settings(use_border_relocator=True), use_jax=True
)


def offset_pipeline_from_vector(vector):
    return offset_analysis.log_likelihood_function(
        instance=offset_model.instance_from_vector(vector=vector, xp=jnp)
    )


offset_vector = jnp.asarray(offset_model.physical_values_from_prior_medians)
offset_vectors = jnp.broadcast_to(offset_vector, (BATCH_SIZE, offset_vector.shape[0]))

_memo.memo_clear()
_reset_wofz_counts()
offset_result = float(
    np.asarray(block(jax.jit(jax.vmap(offset_pipeline_from_vector))(offset_vectors)))[0]
)
offset_witness = _wofz_snapshot()
offset_memo_stats = _memo.memo_stats()

print(
    f"  free params {offset_model.total_free_parameters}, "
    f"log_likelihood {offset_result!r}, _wofz {offset_witness}"
)

assert max_relative < 1e-9, (
    f"the memo changed the JAX log likelihood by {max_relative:.3e} relative "
    f"({on['log_likelihood']!r} vs {off['log_likelihood']!r})"
)

assert on["wofz_compile"]["jnp"] == 0, (
    "memo ON still evaluated the Faddeeva function on the JAX backend: "
    f"{on['wofz_compile']!r} — the fold did not fire"
)
assert on["wofz_compile"]["numpy"] > 0, (
    f"memo ON performed no numpy Faddeeva evaluation, so nothing was folded: {on['wofz_compile']!r}"
)
assert off["wofz_compile"]["numpy"] == 0, (
    f"memo OFF evaluated the Faddeeva function on numpy: {off['wofz_compile']!r}"
)
assert on["wofz_steady"] == {"numpy": 0, "jnp": 0}, (
    f"a compiled program called back into Python: {on['wofz_steady']!r}"
)
# One fold per (fixed Gaussian, grid): the likelihood evaluates deflections on the
# light-profile, pixelization and blurring grids, so the count is a whole multiple of the
# number of Gaussians, and every fold stores an entry.
assert on["memo_stats"]["jax_folds"] == on["memo_stats"]["entries"], (
    f"every fold must store an entry: {on['memo_stats']!r}"
)
assert on["memo_stats"]["jax_folds"] >= TOTAL_GAUSSIANS, (
    f"expected at least one fold per fixed Gaussian ({TOTAL_GAUSSIANS}), got "
    f"{on['memo_stats']['jax_folds']}"
)
assert on["memo_stats"]["jax_folds"] % TOTAL_GAUSSIANS == 0, (
    f"folds should be one per (Gaussian, grid), got {on['memo_stats']['jax_folds']} for "
    f"{TOTAL_GAUSSIANS} Gaussians"
)

assert kill_switch_witness["numpy"] == 0, (
    f"the kill switch did not disable the fold: {kill_switch_witness!r}"
)
assert kill_switch_memo_stats["jax_folds"] == 0
assert abs(kill_switch_result - off["log_likelihood"]) / abs(off["log_likelihood"]) < 1e-9

assert offset_witness["jnp"] > 0, (
    "a free grid_offset makes the grid a tracer, so the Faddeeva subgraph must still be "
    f"traced on the JAX backend: {offset_witness!r}"
)
assert offset_memo_stats["jax_folds"] == 0, (
    f"a traced grid must not be folded: {offset_memo_stats!r}"
)

print("\n  All witness and control assertions PASSED")

# ===================================================================
# Write + pinned-value drift record
# ===================================================================

summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": MASK_RADIUS,
        "image_pixels_masked": int(dataset.data.shape[0]),
        "over_sampled_pixels": int(dataset.grids.lp.over_sampled.shape[0]),
        "mesh_shape": [MESH_PIXELS_YX, MESH_PIXELS_YX],
        "rect_mesh": _cli.rect_mesh,
        "source_pixels": MESH_PIXELS_YX**2,
        "use_jax": True,
        "vmap_batch_size": BATCH_SIZE,
        "lens_light": f"mge_{TOTAL_GAUSSIANS}_lmp_linear_fixed_geometry",
        "lens_mass": "mge_basis + NFWSph + ExternalShear",
        "free_parameters": int(model.total_free_parameters),
        "n_repeats": N_REPEATS,
        "n_steady": N_STEADY,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
        "xla_flags": os.environ.get("XLA_FLAGS", None),
    },
    "full_pipeline_single_jit": on["steady_median_s"],
    "deflections_memo_off": off,
    "deflections_memo_on": on,
    "deflections_memo_jaxpr_equations_off": off_jaxpr_equations,
    "deflections_memo_jaxpr_equations_on": on_jaxpr_equations,
    "deflections_memo_jaxpr_equations_delta": on_jaxpr_equations - off_jaxpr_equations,
    "deflections_memo_max_relative_likelihood_difference": max_relative,
    "controls": {
        "kill_switch": {
            "log_likelihood": kill_switch_result,
            "wofz": kill_switch_witness,
            "memo_stats": kill_switch_memo_stats,
        },
        "free_grid_offset": {
            "log_likelihood": offset_result,
            "wofz": offset_witness,
            "memo_stats": offset_memo_stats,
            "free_parameters": int(offset_model.total_free_parameters),
        },
    },
    "log_likelihood": on["log_likelihood"],
    "log_likelihood_memo_off": off["log_likelihood"],
}

dict_path, _chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "runtime" / "imaging" / "mge_mass_jax",
    default_basename=f"mge_mass_jax_likelihood_summary_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

_pinned_drift: list = []
_pinned_expected = EXPECTED_LOG_LIKELIHOOD.get(_cli.rect_mesh, {}).get(instrument)

if _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for {_cli.rect_mesh}/{instrument} (no pinned value). "
        f"log_likelihood = {on['log_likelihood']!r} — paste it into "
        f"EXPECTED_LOG_LIKELIHOOD to pin it."
    )
else:
    _record = check_pinned(on["log_likelihood"], _pinned_expected, label="jax_mge_mass", rtol=1e-6)
    if _record is not None:
        _pinned_drift.append(_record)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
