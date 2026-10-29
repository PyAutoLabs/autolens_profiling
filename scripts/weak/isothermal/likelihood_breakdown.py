"""
JAX Profiling: Weak-Lensing Likelihood Breakdown (isothermal, standard catalogue)
=================================================================================

Likelihood-lifecycle phase 2g (autolens_profiling#476). Splits one evaluation of
``AnalysisWeak.log_likelihood_function`` on the standard weak cell
(``scripts/misc/likelihood_breakdown/weak_standard.py``: 1500 galaxies, free elliptical isothermal
lens at the simulator truth) into its two stages (decision D8), each JIT-compiled on its own:

1. ``model_shear``: tracer from the instance, then ``FitWeak(xp=jnp).model_shear`` — the
   ``LensCalc.shear_yx_2d_via_hessian_from`` Hessian (``jacfwd`` of the deflections) at every
   catalogue position. Takes the instance, returns the ``(N, 2)`` model shear.
2. ``likelihood_from_shear``: residuals, chi-squared over the 2N components and the noise
   normalization, from a model shear array (FitWeak's own formulas, asserted equal to the full
   pipeline).

``total_step_by_step`` (the breakdown headline) is the sum of the steps' steady medians; the full
JIT pipeline is timed beside it. ``reference_steps.deflections_yx`` (one deflection evaluation at
the N positions) is recorded for scale only: the Hessian differentiates it, so it is not a step.

The row also carries ``device`` with provenance and ``resources``; ``parity.json`` and ``fit.png``
are written beside it, as in the runtime script (the same cell folder).
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
import math
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
from likelihood_breakdown.timing import Timer, jit_profile  # noqa: E402

from _profile_cli import (  # noqa: E402
    device_info_dict,
    parse_profile_cli,
    render_fit_png,
    resources_dict,
    write_parity_json,
)

_cli = parse_profile_cli()

matplotlib.use("Agg")
import autoarray as aa  # noqa: E402
import autolens as al  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402

timer = Timer()
jit_records: dict = {}
instrument = _cli.instrument or ws.DEFAULT_INSTRUMENT
_workspace_root = _profiling_root()
MEDIAN_N_WARM = 5
MEDIAN_N_TIMED = 100

print(f"\n--- Dataset + model [{instrument}] ---")
with timer.section("dataset_and_model"):
    dataset, model, instance = ws.load_cell(instrument, _workspace_root)
configuration = ws.configuration(dataset, model, instrument)

analysis_eager = al.AnalysisWeak(dataset=dataset, use_jax=False)
with timer.section("fit_eager"):
    fit_eager = analysis_eager.fit_from(instance=instance)
    log_likelihood_ref = float(fit_eager.log_likelihood)

analysis_jax = al.AnalysisWeak(dataset=dataset, use_jax=True)
_register_model_pytrees(model)
params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

data_shear = jnp.asarray(np.asarray(dataset.shear_yx))
noise = jnp.asarray(np.asarray(dataset.noise_map))[:, None]
noise_normalization = float(
    2.0 * np.sum(np.log(2.0 * math.pi * np.asarray(dataset.noise_map) ** 2))
)
positions_raw = jnp.asarray(np.asarray(dataset.positions))


def full_pipeline(params):
    return analysis_jax.log_likelihood_function(instance=params)


def model_shear(params):
    tracer = analysis_jax.tracer_via_instance_from(instance=params)
    return al.FitWeak(dataset=dataset, tracer=tracer, xp=jnp).model_shear


def likelihood_from_shear(shear):
    chi_squared = jnp.sum(((data_shear - shear) / noise) ** 2)
    return -0.5 * (chi_squared + noise_normalization)


def deflections_yx(params):
    tracer = analysis_jax.tracer_via_instance_from(instance=params)
    grid = aa.Grid2DIrregular(values=positions_raw, xp=jnp)
    return tracer.deflections_yx_2d_from(grid=grid, xp=jnp).array


profile = dict(timer=timer, jit_records=jit_records)
median = dict(median_n_warm=MEDIAN_N_WARM, median_n_timed=MEDIAN_N_TIMED)

print("\n--- Full pipeline (for comparison) ---")
_, full_result = jit_profile(full_pipeline, "full_pipeline", params_tree, **profile, **median)
full_pipeline_per_call = jit_records["full_pipeline"]["steady_per_call_s"]

print("\n--- Step 1: model shear (tracer + Hessian at every position) ---")
_, shear = jit_profile(model_shear, "model_shear", params_tree, **profile, **median)

print("\n--- Step 2: likelihood from the model shear ---")
_, step_ll = jit_profile(likelihood_from_shear, "likelihood_from_shear", shear, **profile, **median)

print("\n--- Reference: one deflection evaluation at the N positions ---")
jit_profile(deflections_yx, "deflections_yx", params_tree, **profile, **median)

np.testing.assert_allclose(float(step_ll), float(full_result), rtol=1e-12)
np.testing.assert_allclose(
    np.asarray(shear), np.asarray(fit_eager.model_shear), rtol=1e-5, atol=1e-9
)
print("  steps reproduce the full pipeline; model shear matches eager NumPy")

STEPS = ("model_shear", "likelihood_from_shear")
steps = {name: jit_records[name]["steady_per_call_s"] for name in STEPS}
steps_median = {name: jit_records[name]["steady_median"]["median_s"] for name in STEPS}
total_step_by_step = sum(steps_median.values())

timer.summary()
print(
    f"  step-by-step total (medians) {total_step_by_step * 1e3:.3f} ms vs full pipeline "
    f"median {jit_records['full_pipeline']['steady_median']['median_s'] * 1e3:.3f} ms"
)

al_version = al.__version__
backend = jax.default_backend()
breakdown = {
    "autolens_version": al_version,
    "dataset": instrument,
    "instrument": instrument,
    "likelihood": f"weak/{ws.VARIANT}",
    "setup_id": ws.setup_id(backend, _cli.use_mixed_precision),
    "use_mixed_precision": bool(_cli.use_mixed_precision),
    "configuration": configuration,
    "log_likelihood_eager": log_likelihood_ref,
    "log_likelihood_jit": float(full_result),
    "steps": steps,
    "steps_median": steps_median,
    "total_step_by_step": total_step_by_step,
    "total_step_by_step_is": "sum of the steps' steady medians",
    "full_pipeline": {
        "per_call_s": full_pipeline_per_call,
        "median_s": jit_records["full_pipeline"]["steady_median"]["median_s"],
    },
    "reference_steps": {
        "deflections_yx": jit_records["deflections_yx"]["steady_median"]["median_s"],
        "note": "one deflection evaluation at the N positions, for scale; the Hessian "
        "differentiates it, so it is not a step",
    },
    "jit_phases": jit_records,
    "timer": {label: seconds for label, seconds in timer.records},
    "device": device_info_dict(),
    "resources": resources_dict("jax"),
}

dict_path = _cli.output_path or ws.default_output_path(
    "breakdown", instrument, backend, _cli.use_mixed_precision
)
dict_path.parent.mkdir(parents=True, exist_ok=True)
dict_path.write_text(json.dumps(breakdown, indent=2, default=float))
print(f"\n  Results dict saved to: {dict_path}")

write_parity_json(
    dict_path.parent,
    log_likelihood=float(full_result),
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
