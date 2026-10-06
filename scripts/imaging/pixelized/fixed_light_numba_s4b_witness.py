"""Seeded-draw correctness witness for the permuted CPU NNLS kernel (#276).

The S3 construction matches the phase-4 curvature witness. Every lane traverses
seed-263 central-20-percent iid draws with its own production memo history.
Exact passive-set agreement is required; disagreements retain both KKT records.
Timing promotion also requires a separate >=5% whole-call win.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "misc")]
from _production_config import pin_thread_env
from _profile_cli import parse_profile_cli

_cli = parse_profile_cli()
parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--mesh", choices=("delaunay",), default="delaunay")
parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--n-draws", type=int, default=8)
parser.add_argument("--kernel", choices=("library", "permuted"), default="library")
args = _cli.parse_cell_args(parser)
if args.n_draws < 1 or args.threads < 1:
    parser.error("n-draws and threads must be positive")
MESH, DATASET = args.mesh, args.dataset
thread_env = pin_thread_env(args.threads)
os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
os.environ["AUTOARRAY_NNLS_WARM_START"] = "1"

import autoarray
import autofit as af
import autolens as al
import numpy as np
import scipy
from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
from autoarray.inversion.inversion.imaging_numba.sparse import InversionImagingSparseNumba
from likelihood_breakdown import fixed_light_numpy_solvers as solvers
from likelihood_breakdown import fixed_light_system
from likelihood_breakdown.fixed_light_s4b_checks import compare, evaluate_fit
from simulators.imaging import INSTRUMENTS

from _adapt_image_util import adapt_image_for_dataset
from _profile_cli import auto_simulate_if_missing, delaunay_regularization, machine_info_dict

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print("[smoke] s4b witness imports OK")
    raise SystemExit(0)

_t_start = time.time()
_workspace_root = ROOT

instrument = DATASET

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

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

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4,
    over_sample_size_pixelization=1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

n_mesh_vertices = 1500 if _cli.source_pixels is None else int(_cli.source_pixels)

print("\n--- Adapt image (lensed source) ---")
adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

print("\n--- Image mesh construction (Hilbert) ---")
image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)
print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

print("\n--- Model construction ---")

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=60,
    centre_prior_is_uniform=True,
)

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

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass)
field = af.Model(al.MassField, redshift=0.5, shear=shear)


mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization:        {reg_scheme} ({reg_provenance})")

# Instance 0 of `fixed_light_numba.py`'s `--instances iid` stream: the same seed
# and the same central-20 % draw, so this witness stands on the very first
# instance every timed row of the A/B legs starts from.
_IID_SEED = 263
_rng = np.random.default_rng(_IID_SEED)
instance = model.instance_from_unit_vector(
    unit_vector=list(_rng.uniform(0.4, 0.6, size=model.prior_count))
)


def adapt_images_of(one_instance):
    """``AdaptImages`` for one instance — the dicts are keyed on its own galaxy."""
    return al.AdaptImages(
        galaxy_image_dict={one_instance.galaxies.source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        galaxy_image_plane_mesh_grid_dict={one_instance.galaxies.source: image_plane_mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )


adapt_images = adapt_images_of(instance)

_settings = al.Settings(use_border_relocator=True)

print("\n--- S0: full FitImaging with linear MGE lens light (eager, dense) ---")
fit_s0 = al.FitImaging(
    dataset=dataset,
    tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
    adapt_images=adapt_images,
    settings=_settings,
    xp=np,
)
assert isinstance(fit_s0.inversion, InversionImagingMapping), (
    f"S0 dispatched {type(fit_s0.inversion).__name__}, not InversionImagingMapping"
)
print(f"  inversion class: {type(fit_s0.inversion).__name__}")

print("\n--- S3: lens light converted to regular profiles + subtracted (sparse numba) ---")
system_s3 = fixed_light_system.fixed_light_system_from(
    fit_s0,
    dataset,
    adapt_images=adapt_images,
    settings=_settings,
    name="S3_sparse_numba",
    scaling="numpy",
    sparse_operator="rebake_cpu",
)
dataset_s3 = system_s3.dataset
source_only_tracer = system_s3.source_only_tracer
print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)
print(f"  S3 operator:     {type(dataset_s3.sparse_operator).__name__}")
print(f"  subtracted light flux: {system_s3.subtracted_light_flux}")


def _s3_instance(one_instance):
    """``one_instance`` with the lens light stripped, for the S3 fit."""
    import copy

    stripped = copy.deepcopy(one_instance)
    galaxies = list(
        fixed_light_system._light_stripped_tracer(
            al.Tracer(galaxies=list(one_instance.galaxies), fields=[one_instance.fields])
        ).galaxies
    )
    stripped.galaxies.lens = galaxies[0]
    stripped.galaxies.source = galaxies[1]
    return stripped


adapt_images_s3 = adapt_images_of(_s3_instance(instance))


# Three independent lanes: unmodified library, identity injection, candidate.
# The baseline and candidate must never warm-start from one another's answer.
kernel = solvers.fnnls_cholesky if args.kernel == "library" else solvers.fnnls_cholesky_permuted
memos = [{}, {}, {}]
rows = []
for index in range(args.n_draws):
    one = (
        instance
        if index == 0
        else model.instance_from_unit_vector(
            unit_vector=list(_rng.uniform(0.4, 0.6, size=model.prior_count))
        )
    )
    stripped = _s3_instance(one)

    def make_fit(stripped=stripped):
        fit = al.FitImaging(
            dataset=dataset_s3,
            tracer=al.Tracer(galaxies=list(stripped.galaxies), fields=[stripped.fields]),
            adapt_images=adapt_images_of(stripped),
            settings=_settings,
            xp=np,
        )
        assert isinstance(fit.inversion, InversionImagingSparseNumba)
        return fit

    outcomes = []
    for lane, candidate in enumerate((solvers.fnnls_cholesky, solvers.fnnls_cholesky, kernel)):
        outcome, memos[lane] = evaluate_fit(make_fit, candidate, memos[lane])
        outcomes.append(outcome)
    identity = compare(outcomes[0], outcomes[1], identity=True)
    candidate = compare(outcomes[0], outcomes[2], identity=args.kernel == "library")
    rows.append({"draw": index, "identity": identity, "candidate": candidate})
    print(
        f"draw {index}: identity {identity['verdict']}, {args.kernel} {candidate['verdict']}, "
        f"Jaccard {candidate['active_set']['jaccard']}, "
        f"delta evidence {candidate['evidence']['delta_nats']:+.3e}",
        flush=True,
    )

passed = all(row[leg]["verdict"] == "PASS" for row in rows for leg in ("identity", "candidate"))
config = _cli.config_name or "local"
record = {
    "cell": "fixed_light_numba_s4b_witness",
    "issue": "autolens_profiling#276",
    "verdict": "PASS" if passed else "FAIL",
    "draws": rows,
    "configuration": {
        "config_name": config,
        "mesh": MESH,
        "dataset": DATASET,
        "source_pixels": n_mesh_vertices,
        "kernel": args.kernel,
        "n_draws": args.n_draws,
        "instance_seed": _IID_SEED,
        "instance_mode": "iid_central_20_percent",
        "threads": args.threads,
        "thread_env": thread_env,
        "numba_thread_env": {"NUMBA_NUM_THREADS": os.environ["NUMBA_NUM_THREADS"]},
        "nnls_warm_start_env": os.environ["AUTOARRAY_NNLS_WARM_START"],
        "regularization": reg_provenance,
        "use_jax": False,
        "w2_jaccard_required": 1.0,
        "w3_rtol": solvers.WITNESS_EVIDENCE_RTOL,
        "w6_atol_nats": solvers.WITNESS_LOG_DET_ATOL,
        "w6_allowed_spacings": solvers.WITNESS_LOG_DET_ULPS,
        "w6_factor_rtol": solvers.WITNESS_FACTOR_RTOL,
        "fallback_policy": "same seed_source, memo guard flag, kernel errors and attempt count",
        "pdip_fallback": "not applicable: numpy production entrypoint uses fnnls only",
        "memo_policy": "independent identical-initial-state streams, never cross-seeded",
    },
    "provenance": {
        "autoarray_file": autoarray.__file__,
        "autoarray_version": autoarray.__version__,
        "autolens_version": al.__version__,
        "scipy_version": scipy.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_s": time.time() - _t_start,
    },
    "machine": machine_info_dict(),
}
out_dir = _cli.output_dir or ROOT / "results" / "breakdown" / "imaging"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"fixed_light_numba_s4b_witness_{config}.json"
out_path.write_text(json.dumps(record, indent=2, default=str))
print(f"VERDICT: {record['verdict']} — {out_path}")
if not passed:
    raise SystemExit(1)
