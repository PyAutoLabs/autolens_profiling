"""
JAX Profiling: Weak-Lensing Likelihood Runtime (isothermal, standard catalogue)
===============================================================================

Likelihood-lifecycle phase 2g (autolens_profiling#476). Profiles
``AnalysisWeak.log_likelihood_function`` on the standard weak cell
(``scripts/misc/likelihood_breakdown/weak_standard.py``): the ``simple`` shear catalogue of 1500
galaxies (3000 shear components) fitted with a free elliptical isothermal lens at the simulator
truth. One likelihood evaluation is the tracer from the instance, the model shear at every
catalogue position (``LensCalc.shear_yx_2d_via_hessian_from``: JAX ``jacfwd`` of the deflections),
then the residuals, chi-squared and noise normalization. No inversion.

The row (``--output-path``, else a scratch cell under ``output/weak_local/``) carries the JIT full
pipeline (lower, compile, first call, 10-call block mean and the steady median headline), the
vmap per-call cost at batch ``VMAP_BATCH`` (what a sampler pays per point), the eager NumPy
reference, ``device`` with provenance and ``resources``. ``parity.json`` (JIT vs the eager NumPy
``FitWeak`` in the same process) and ``fit.png`` (the weak producer's ``subplot_fit_weak``) are
written beside it after every timed region.

Which setup the row is (``jax_cpu_dense_fp64`` or ``jax_gpu_dense_fp64``) is the JAX backend the
process runs on: ``JAX_PLATFORMS=cpu`` measures the CPU setup even on an A100 host (decision D4).
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

import json
import os as _smoke_os
import sys

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

sys.path.insert(0, str(_profiling_root()))
from likelihood_breakdown import weak_standard as ws  # noqa: E402
from likelihood_breakdown.timing import (  # noqa: E402
    Timer,
    headline_steady_median,
    jit_profile,
    vmap_profile,
)

from _profile_cli import (  # noqa: E402
    device_info_dict,
    parse_profile_cli,
    render_fit_png,
    resources_dict,
    write_parity_json,
)

_cli = parse_profile_cli()

matplotlib.use("Agg")
import autolens as al  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402

timer = Timer()
jit_records: dict = {}
instrument = _cli.instrument or ws.DEFAULT_INSTRUMENT
_workspace_root = _profiling_root()

# ===================================================================
# Setup: the standard weak cell
# ===================================================================

print(f"\n--- Dataset + model [{instrument}] ---")
with timer.section("dataset_and_model"):
    dataset, model, instance = ws.load_cell(instrument, _workspace_root)
configuration = ws.configuration(dataset, model, instrument)
print(f"  {configuration['n_galaxies']} galaxies, {model.total_free_parameters} free parameters")

# ===================================================================
# Eager NumPy reference (the parity reference, and the fit.png fit)
# ===================================================================

print("\n--- Eager NumPy FitWeak ---")
analysis_eager = al.AnalysisWeak(dataset=dataset, use_jax=False)
with timer.section("fit_eager"):
    fit_eager = analysis_eager.fit_from(instance=instance)
    log_likelihood_ref = float(fit_eager.log_likelihood)
n_eager_repeats = 10
with timer.section(f"eager_log_likelihood_x{n_eager_repeats}"):
    for _ in range(n_eager_repeats):
        analysis_eager.log_likelihood_function(instance=instance)
eager_per_call = timer.records[-1][1] / n_eager_repeats
print(f"  log_likelihood = {log_likelihood_ref}  eager per call = {eager_per_call:.6f} s")

# ===================================================================
# Full-pipeline JIT
# ===================================================================

print("\n--- Full-pipeline JIT ---")
analysis_jax = al.AnalysisWeak(dataset=dataset, use_jax=True)
_register_model_pytrees(model)
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)


def full_pipeline_from_params(params):
    return analysis_jax.log_likelihood_function(instance=params)


full_compiled, full_result = jit_profile(
    full_pipeline_from_params, "full_pipeline", params_tree, timer=timer, jit_records=jit_records
)
full_pipeline_per_call = timer.records[-1][1] / 10
full_pipeline_median = headline_steady_median(
    full_compiled, params_tree, block_mean_s=full_pipeline_per_call
)
log_likelihood_jit = float(full_result)
print(f"  JIT log_likelihood = {log_likelihood_jit}")

# ===================================================================
# vmap: the per-point cost a batched sampler pays
# ===================================================================

print(f"\n--- vmap batch {ws.VMAP_BATCH} ---")
batched_params = jax.tree_util.tree_map(
    lambda leaf: jnp.broadcast_to(leaf, (ws.VMAP_BATCH, *jnp.shape(leaf))), params_tree
)
vmap_per_call = vmap_profile(
    full_pipeline_from_params,
    "full_pipeline",
    batched_params,
    ws.VMAP_BATCH,
    timer=timer,
    jit_records=jit_records,
)
vmap_batch_time = vmap_per_call * ws.VMAP_BATCH
vmap_values = np.asarray(jax.jit(jax.vmap(full_pipeline_from_params))(batched_params))
np.testing.assert_allclose(vmap_values, log_likelihood_jit, rtol=1e-10)

memory = jax.jit(jax.vmap(full_pipeline_from_params)).lower(batched_params).compile()
memory = memory.memory_analysis()

timer.summary()

# ===================================================================
# The row
# ===================================================================

al_version = al.__version__
backend = jax.default_backend()
full = jit_records["full_pipeline"]
likelihood_summary = {
    "autolens_version": al_version,
    "dataset": instrument,
    "instrument": instrument,
    "likelihood": f"weak/{ws.VARIANT}",
    "setup_id": ws.setup_id(backend, _cli.use_mixed_precision),
    "use_mixed_precision": bool(_cli.use_mixed_precision),
    "configuration": configuration,
    "eager_per_call": eager_per_call,
    "log_likelihood_eager": log_likelihood_ref,
    "log_likelihood_jit": log_likelihood_jit,
    "full_pipeline_lower_s": full["lower_s"],
    "full_pipeline_compile_s": full["compile_s"],
    "full_pipeline_first_call_s": full["first_call_s"],
    "full_pipeline_single_jit": full_pipeline_per_call,
    **full_pipeline_median,
    "vmap": {
        "batch_size": ws.VMAP_BATCH,
        "batch_time": vmap_batch_time,
        "per_call": vmap_per_call,
        "speedup_vs_single_jit": round(full_pipeline_per_call / vmap_per_call, 2),
    },
    "memory_mb": {
        "output": memory.output_size_in_bytes / 1024**2,
        "temp": memory.temp_size_in_bytes / 1024**2,
    },
    "jit_phases": jit_records,
    "timer": {label: seconds for label, seconds in timer.records},
    "device": device_info_dict(),
    "resources": resources_dict("jax"),
}

dict_path = _cli.output_path or ws.default_output_path(
    "runtime", instrument, backend, _cli.use_mixed_precision
)
dict_path.parent.mkdir(parents=True, exist_ok=True)
dict_path.write_text(json.dumps(likelihood_summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# Parity witness: the JIT full pipeline against the eager NumPy FitWeak computed above on the same
# dataset, model and instance (decision D6: the default fp64 tolerance, 1e-4 x |reference|).
write_parity_json(
    dict_path.parent,
    log_likelihood=log_likelihood_jit,
    reference_log_likelihood=log_likelihood_ref,
    use_mixed_precision=_cli.use_mixed_precision,
    reference_source=(
        "eager NumPy AnalysisWeak(use_jax=False).fit_from(instance).log_likelihood in the same "
        f"process (adaptive Richardson finite-difference Hessian): {instrument}, "
        f"{configuration['n_galaxies']} galaxies, same dataset, model and instance"
    ),
    autolens_version=al_version,
    quantity="log_likelihood_function (JIT jacfwd Hessian vs eager NumPy finite differences)",
    dataset_type="weak",
)

# Fit figure (likelihood-lifecycle 1B): the eager fit, drawn after every timed region and kept as
# fit.png beside this row. Off under AUTOLENS_PROFILING_SMOKE and with --no-fit-png.
render_fit_png(fit_eager, dict_path.parent, dataset_type="weak", cli=_cli)
