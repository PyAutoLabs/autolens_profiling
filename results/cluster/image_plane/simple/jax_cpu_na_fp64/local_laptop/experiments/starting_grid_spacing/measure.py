"""Run from the profiling checkout with a grid spacing in arcseconds and output JSON path."""

import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "scripts/misc")]
import autolens
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model
from likelihood_breakdown import cluster_standard as cluster
from likelihood_breakdown import timing
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from _profile_cli import device_info_dict

scale = float(sys.argv[1])
size = round(100 / scale)
settings = dict(cluster.STANDARD_SOLVER, shape_native=(size, size), pixel_scales=scale)
cell = cluster.load_cell(root)
model = cluster.model(cell, solved=True, point_centres=cell["source_centres"])
register_model(model)
params = jnp.asarray(model.physical_values_from_prior_medians)
instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
assert jax.default_backend() == "cpu" and params.dtype == jnp.float64
solver = cluster.build_solver(settings)
fits = cluster.analyses(cell, "image_plane", "solved", solver, use_jax=True)
full = cluster.summed_likelihood(model, fits)
records = {}
compiled, value = timing.jit_profile(
    full, "solved", params, timer=timing.Timer(), jit_records=records, n_repeats=3
)
steady = timing.steady_median_profile(compiled, params, n_warm=5, n_timed=200)
print(f"{scale} arcsec warm median: {steady['median_s'] * 1000:.3f} ms", flush=True)
numpy_solver = cluster.build_solver(settings, use_jax=False)
references = cluster.analyses(cell, "image_plane", "solved", numpy_solver, use_jax=False)
eager = sum(float(a.log_likelihood_function(instance=instance)) for a in references)
np.testing.assert_allclose(float(value), eager, rtol=0, atol=1e-6)
families = {}
for dataset, analysis in zip(cell["dataset_list"], references, strict=True):
    fit = analysis.fit_from(instance=instance).positions
    predictions = np.asarray(fit.model_data)
    predictions = predictions[np.isfinite(predictions).all(axis=1)]
    distances = cdist(np.asarray(dataset.positions), predictions)
    observed, predicted = linear_sum_assignment(distances)
    families[dataset.name] = dict(
        observed=len(distances),
        predicted=len(predictions),
        nearest_arcsec=distances.min(axis=1).tolist(),
        matched_distinct_within_0p01_arcsec=int(np.sum(distances[observed, predicted] <= 0.01)),
        source_centre_yx_arcsec=np.asarray(fit.source_plane_coordinate).tolist(),
    )
result = dict(
    settings=cluster.settings_record(settings),
    families=families,
    gallery_revision=cell["gallery_revision"],
    fit_class=cluster.FIT_CLASS_NAMES["image_plane"]["solved"],
    parameter_vector=np.asarray(params).tolist(),
    numpy_log_likelihood=eager,
    jax_log_likelihood=float(value),
    phases=records["solved"],
    steady=steady,
    device=device_info_dict(),
)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(families, flush=True)
