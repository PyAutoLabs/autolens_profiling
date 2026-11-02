"""
Gradient cost and correctness: MGE imaging likelihood
=====================================================

Likelihood lifecycle phase 6 (autolens_profiling#505), the pilot ``gradient_breakdown.py``.

Builds the same fused, jit-able MGE imaging likelihood as ``likelihood_runtime.py`` (shared
``cell_builders.imaging_mge``: dataset, mask, over-sampling, model, analysis), at the same
point (the prior medians), and hands it to ``scripts/misc/gradient/harness.py``, which:

* times the forward call and ``jax.value_and_grad`` (first call with compile, steady
  single-call median, vmap per call);
* runs the pre-registered finite-difference check (3 seeded directions, Richardson
  extrapolation of steps h and h/2, relative error <= 1e-4);
* flags a hazard when value_and_grad costs more than 4x the forward call or the FD check
  fails;
* writes ``results/imaging/mge/<instrument>/jax_<cpu|gpu>_dense_fp64/<host_class>/gradient.json``,
  or the ``--output-path`` a campaign submit gives (``scripts/misc/tooling/gradient_witness.py``), which ``build_catalogue.py`` reads as a
  ``measurement: gradient`` record.

Gradients are taken with respect to the flattened free parameters (``ravel_pytree`` of the
model instance). fp64 only; ``--use-mixed-precision`` is refused.

Run::

    python scripts/imaging/mge/gradient_breakdown.py --instrument hst
"""

from __future__ import annotations

import os as _os
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

import autolens as al  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402
from cell_builders.imaging_mge import build_analysis, build_model, prepare_dataset  # noqa: E402
from gradient import harness  # noqa: E402
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _profile_cli import auto_simulate_if_missing, parse_profile_cli  # noqa: E402

if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

_cli = parse_profile_cli()
if _cli.use_mixed_precision:
    raise SystemExit(
        "gradient_breakdown.py measures fp64 only (phase 6); drop --use-mixed-precision"
    )
jax.config.update("jax_enable_x64", True)

instrument = _cli.instrument or "hst"
_root = _profiling_root()

# --- the cell, as likelihood_runtime.py builds it --------------------------------------------
dataset_path = _Path("dataset") / "imaging" / instrument
auto_simulate_if_missing(
    dataset_path, dataset_type="imaging", instrument=instrument, workspace_root=_root
)
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=INSTRUMENTS[instrument]["pixel_scale"],
)
dataset = prepare_dataset(dataset, use_sparse_operator=_cli.use_sparse_operator)
model = build_model()
instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
_register_model_pytrees(model)
params = jax.tree_util.tree_map(jnp.asarray, instance)
analysis = build_analysis(dataset)


def log_likelihood(params_tree):
    return analysis.log_likelihood_function(instance=params_tree)


# --- measure and write ------------------------------------------------------------------------
# ``harness.run_gradient`` as at the phase-6b construction sites, so a campaign submit's
# ``--output-path`` is honoured (#509: the pilot used to write into ``results/`` regardless).
operator = "sparse" if _cli.use_sparse_operator else "dense"
harness.run_gradient(
    log_likelihood,
    params,
    point=harness.POINT_PRIOR_MEDIANS,
    dataset_type="imaging",
    variant="mge",
    instrument=instrument,
    setup_id=f"jax_{'gpu' if jax.default_backend() in ('gpu', 'cuda') else 'cpu'}_{operator}_fp64",
    path=getattr(_cli, "output_path", None),
)
