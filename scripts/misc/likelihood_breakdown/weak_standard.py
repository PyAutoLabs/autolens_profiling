"""The weak-lensing standard cell, shared by the weak runtime and breakdown scripts (2g, #476).

One likelihood evaluation of a weak cell is ``AnalysisWeak.log_likelihood_function(instance)``:
build the tracer from the instance, evaluate the model shear at every catalogue position through
``LensCalc.shear_yx_2d_via_hessian_from`` (JAX: an exact ``jacfwd`` Hessian of the deflections per
position; NumPy: adaptive Richardson finite differences), then the residuals, chi-squared over
the 2N shear components and the noise normalization. There is no linear algebra (decision D1 on
``wiki/campaigns/weak/likelihood_lifecycle_p2g.md``).

The standard cell is the ``simple`` catalogue of ``scripts/misc/simulators/weak.py`` (1500
galaxies, decision D3) fitted with the free elliptical isothermal model at the simulator truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
if str(ROOT / "scripts" / "misc") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))

from simulators import weak as weak_sim  # noqa: E402

#: The standard number of catalogue galaxies (2N shear components), decision D3.
STANDARD_N_GALAXIES = 1500
DEFAULT_INSTRUMENT = "simple"
VARIANT = weak_sim.VARIANT
#: The vmap batch every weak runtime row reports (decision D7): one sampler batch of points.
VMAP_BATCH = 32
#: Where a run without ``--output-path`` writes: a mirrored-tree scratch cell under the
#: gitignored ``output/``, never in place over ``results/``.
LOCAL_RESULTS = ROOT / "output" / "weak_local"


def setup_id(backend: str, use_mixed_precision: bool) -> str:
    """The §3.3 setup the running process measures (``jax_<cpu|gpu>_dense_<fp64|mixed>``)."""
    device = "gpu" if backend == "gpu" else "cpu"
    return f"jax_{device}_dense_{'mixed' if use_mixed_precision else 'fp64'}"


def default_output_path(measurement: str, instrument: str, backend: str, mixed: bool) -> Path:
    host = "local_gpu" if backend == "gpu" else "local_laptop"
    return (
        LOCAL_RESULTS
        / "results"
        / "weak"
        / VARIANT
        / instrument
        / setup_id(backend, mixed)
        / host
        / f"{measurement}.json"
    )


def load_cell(instrument: str, root: Path = ROOT):
    """(dataset, model, instance) of the standard weak cell, simulating a missing dataset."""
    import autolens as al

    from _profile_cli import auto_simulate_if_missing

    path = weak_sim.dataset_path(instrument, root)
    auto_simulate_if_missing(path, dataset_type="weak", instrument=instrument, workspace_root=root)
    dataset = al.from_json(file_path=path / "dataset.json")
    model = weak_sim.standard_model(instrument)
    instance = model.instance_from_vector(vector=weak_sim.truth_vector(instrument))
    return dataset, model, instance


def configuration(dataset, model, instrument: str) -> dict:
    preset = weak_sim.INSTRUMENTS[instrument]
    n = int(dataset.n_galaxies)
    return {
        "variant": VARIANT,
        "instrument": instrument,
        "n_galaxies": n,
        "shear_components": 2 * n,
        "standard_n_galaxies": STANDARD_N_GALAXIES,
        "standard_size": n == STANDARD_N_GALAXIES,
        "noise_sigma": float(preset["noise_sigma"]),
        "annulus_arcsec": [float(preset["radius_inner"]), float(preset["radius_outer"])],
        "mass_profile": "Isothermal",
        "is_reduced": bool(dataset.is_reduced),
        "redshifts": dataset.redshifts is not None,
        "free_parameters": int(model.total_free_parameters),
        "instance": "simulator truth (truth_vector)",
    }
