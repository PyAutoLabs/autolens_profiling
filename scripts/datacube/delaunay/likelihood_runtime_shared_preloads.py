"""
Datacube Runtime: shared_preloads speed-up
==========================================

Measures the realized end-to-end speed-up of the `FactorGraphModel` shared-state path for the
interferometer datacube: the same identical-channel cube is timed with `shared_preloads=False`
(every channel rebuilds its inversion-setup quantities) and with `shared_preloads=True` (the
channel-invariant mapper + curvature matrix `F = LᵀW̃L` are computed once and reused), and the
ratio is reported.

Timing uses `Fitness._vmap` over a batch rather than a single jit on concrete parameters, because a
single jit over fixed parameters can constant-fold the work and report a misleadingly fast time;
the vmapped per-evaluation time is honest (see PyAutoLabs notes on jit const-folding).

Correctness of `shared_preloads` is asserted separately in
`autolens_workspace_test/scripts/jax_likelihood_functions/datacube/shared_preloads.py`; this script
is purely a runtime measurement, and the companion to the per-step decomposition in
`likelihood_breakdown/datacube/delaunay.py`.

Provisional vs authoritative
----------------------------
SMA-scale numbers are *provisional* — the per-channel inversion-setup cost (and therefore the
realized speed-up) grows with visibility count and mask resolution. Pin the authoritative figure by
running at ALMA scale on a quiet A100 (`--instrument alma_high`, raise `N_CHANNELS`). The ratio is
the robust deliverable; absolute seconds are environment-bound.

Run from the repo root:

    python likelihood_runtime/datacube/shared_preloads.py [--instrument sma|alma|alma_high]
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


import os
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax.numpy as jnp
import numpy as np
from autofit.non_linear.fitness import Fitness

sys.path.insert(0, str(_profiling_root()))

# CI lint smoke: verify the import graph + module setup without the heavy timing run.
if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    from simulators.interferometer import INSTRUMENTS  # noqa: F401

    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from simulators.interferometer import INSTRUMENTS  # noqa: E402

from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
instrument = _cli.instrument or "sma"

N_CHANNELS = 4
BATCH_SIZE = 2
N_REPEATS = 5

pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
mask_radius = INSTRUMENTS[instrument]["mask_radius"]

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)

dataset_path = Path("dataset") / "interferometer" / instrument


def _model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.05, upper_limit=0.15)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(8, 8)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _factor_graph(shared_preloads):
    dataset_list = [
        al.Interferometer.from_fits(
            data_path=str(dataset_path / "data.fits"),
            noise_map_path=str(dataset_path / "noise_map.fits"),
            uv_wavelengths_path=str(dataset_path / "uv_wavelengths.fits"),
            real_space_mask=real_space_mask,
            transformer_class=al.TransformerDFT,
        ).apply_sparse_operator(use_jax=True)
        for _ in range(N_CHANNELS)
    ]

    analysis_list = [
        al.AnalysisInterferometer(
            dataset=dataset,
            use_jax=True,
            shared_preloads=shared_preloads,
            raise_inversion_positions_likelihood_exception=False,
        )
        for dataset in dataset_list
    ]

    analysis_factor_list = [
        af.AnalysisFactor(prior_model=_model().copy(), analysis=analysis)
        for analysis in analysis_list
    ]

    return af.FactorGraphModel(*analysis_factor_list, use_jax=True)


def _time_vmap(shared_preloads):
    factor_graph = _factor_graph(shared_preloads)

    fitness = Fitness(
        model=factor_graph.global_prior_model,
        analysis=factor_graph,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    medians = factor_graph.global_prior_model.physical_values_from_prior_medians
    parameters = jnp.array(np.tile(np.asarray(medians), (BATCH_SIZE, 1)))

    fitness._vmap(parameters).block_until_ready()  # warm-up (compile excluded)

    start = time.perf_counter()
    for _ in range(N_REPEATS):
        fitness._vmap(parameters).block_until_ready()
    elapsed = time.perf_counter() - start

    return elapsed / (N_REPEATS * BATCH_SIZE)


if __name__ == "__main__":
    print(
        f"datacube shared_preloads runtime  "
        f"(instrument={instrument}, N_CHANNELS={N_CHANNELS}, sparse route)"
    )

    per_eval_unshared = _time_vmap(shared_preloads=False)
    per_eval_shared = _time_vmap(shared_preloads=True)

    print(f"  per-likelihood (unshared): {per_eval_unshared * 1e3:.1f} ms")
    print(f"  per-likelihood (shared):   {per_eval_shared * 1e3:.1f} ms")
    print(f"  speed-up:                  {per_eval_unshared / per_eval_shared:.2f}x")
    print(
        "  (provisional SMA numbers — re-run --instrument alma_high on a quiet A100 "
        "for the authoritative figure; the ratio is the robust deliverable)"
    )
