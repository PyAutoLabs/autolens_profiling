"""
Numba CPU Profiling: Delaunay NNLS Active-Set Iterations (Warm-Start Memo A/B)
=============================================================================

Measures what the cross-evaluation NNLS warm-start memo
(``aa.Settings(nnls_warm_start_memo=...)``, PyAutoArray#498) actually buys on the
production Delaunay numba-CPU fiducial, so the default can be chosen from numbers
rather than from the plausibility of the idea.

The positive-only reconstruction is ~70% of a numba CPU Delaunay likelihood
evaluation at Euclid resolution. Its Bro & De Jong active-set loop is warm-started
from a guess at the passive set; production guesses it from the sign of the
unconstrained dense solve, and the memo instead reuses the *previous evaluation's*
final passive set.

Method
------
The memo only pays off across *different* parameter points — re-fitting one fixed
instance would hand the memo a 100%-correct seed and fake the answer. So two
30-instance sequences are built and each is run **twice**, memo OFF then memo ON:

* ``random_walk`` — a seeded Gaussian random walk from the prior medians, each step
  perturbing every parameter by N(0, 1% of that parameter's prior width). This is
  the sampler-like regime the memo is designed for.
* ``iid`` — 30 independent draws with unit values uniform in [0.4, 0.6] (the central
  20% of every prior). Neighbouring evaluations are unrelated: the pessimistic case.

``autoarray.util.fnnls.fnnls_cholesky`` is wrapped at module level (the caller
imports it at call time) so every solve's ``stats`` dict — outer/inner active-set
iterations, final passive-set size, and how many entries the warm start got wrong —
is captured from inside a real ``FitImaging``. A memo-seeded solve that raises is
retried by the library from the dense-sign start; both attempts are recorded and the
retries are counted and reported.

Parity is asserted per instance: the NNLS optimum is unique, so the memo must not
move the log likelihood (rtol 1e-6) or the reconstruction.

Output
------
``results/breakdown/imaging/delaunay_numba_nnls_iterations_<instrument>_v<version>.{json,png}``
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

import json
import os

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os
import sys as _smoke_sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)

_cli = parse_profile_cli()

instrument = _cli.instrument or "euclid"  # default; override via --instrument

N_INSTANCES = 30
RANDOM_WALK_STEP_FRACTION = 0.01  # step sigma as a fraction of each prior's width
IID_UNIT_LOW, IID_UNIT_HIGH = 0.4, 0.6  # central 20% of every prior
SEED = 498  # PyAutoArray#498
PARITY_RTOL = 1.0e-6

# ===================================================================
# Setup — identical fiducial to delaunay_numba.py (the breakdown sibling)
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

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
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

dataset = dataset.apply_sparse_operator_cpu()

print("\n--- Adapt image + Hilbert image mesh (one-off) ---")

n_mesh_vertices = 1250

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_mesh = al.image_mesh.Hilbert(pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0)
image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)

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

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0),
    regularization=al.reg.ConstantSplit(coefficient=1.0),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# ===================================================================
# Instance sequences
# ===================================================================


def prior_step_sigmas() -> np.ndarray:
    """
    Per-parameter random-walk step sigma: ``RANDOM_WALK_STEP_FRACTION`` of each
    prior's width, where "width" is a Gaussian prior's sigma and a uniform prior's
    ``upper - lower``.
    """
    sigmas = []
    for prior in model.priors_ordered_by_id:
        sigma = getattr(prior, "sigma", None)
        if sigma is None:
            sigma = float(prior.upper_limit) - float(prior.lower_limit)
        sigmas.append(RANDOM_WALK_STEP_FRACTION * float(sigma))
    return np.asarray(sigmas)


def prior_support() -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter (lower, upper) physical limits, used to keep the walk in support."""
    lower = np.asarray([float(p.lower_limit) for p in model.priors_ordered_by_id])
    upper = np.asarray([float(p.upper_limit) for p in model.priors_ordered_by_id])
    # Inset finite limits so a step that lands exactly on a bound is still interior.
    span = np.where(np.isfinite(upper - lower), (upper - lower), 0.0)
    return lower + 1.0e-6 * span, upper - 1.0e-6 * span


def random_walk_instances(n: int) -> list:
    """
    ``n`` instances along a seeded Gaussian random walk from the prior medians.

    This is the regime the memo targets: successive sampler evaluations sit close
    together, so their passive sets overlap almost entirely.
    """
    rng = np.random.default_rng(SEED)
    step_sigmas = prior_step_sigmas()
    lower, upper = prior_support()

    vector = np.asarray(model.physical_values_from_prior_medians, dtype=float)

    instances = []
    for _ in range(n):
        vector = np.clip(vector + rng.normal(0.0, step_sigmas), lower, upper)
        instances.append(model.instance_from_vector(vector=list(vector)))
    return instances


def iid_instances(n: int) -> list:
    """
    ``n`` independent instances with unit values uniform in the central 20% of every
    prior — the pessimistic case, where neighbouring evaluations are unrelated.
    """
    rng = np.random.default_rng(SEED + 1)
    return [
        model.instance_from_unit_vector(
            unit_vector=list(rng.uniform(IID_UNIT_LOW, IID_UNIT_HIGH, size=model.prior_count))
        )
        for _ in range(n)
    ]


def adapt_images_for(instance) -> al.AdaptImages:
    """
    Rebuild the adapt images for ``instance``: the dict is keyed on the instance's own
    source galaxy object, so it cannot be shared between instances.
    """
    return al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={instance.galaxies.source: image_plane_mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )


# ===================================================================
# Solver-stat capture
# ===================================================================

import autoarray.util.fnnls as fnnls_mod  # noqa: E402
from autoarray.inversion.inversion import nnls_memo  # noqa: E402

_orig_fnnls_cholesky = fnnls_mod.fnnls_cholesky

_solve_records: list[dict] = []
_recording = [False]


def _wrapped_fnnls_cholesky(ZTZ, ZTx, P_initial=np.zeros(0, dtype=int), stats=None):
    """
    Record every fnnls solve's diagnostics. ``reconstruction_positive_only_from``
    resolves ``fnnls_cholesky`` by ``from ... import`` at call time, so replacing the
    module attribute is enough to see solves from inside a real fit.

    A raising solve is recorded too — that is exactly the memo-seed retry the library
    catches, and it must be counted, not silently dropped.
    """
    stats = {} if stats is None else stats
    start = time.perf_counter()
    try:
        reconstruction = _orig_fnnls_cholesky(ZTZ, ZTx, P_initial, stats=stats)
    except BaseException as e:
        if _recording[0]:
            _solve_records.append(
                {
                    "n": int(np.shape(ZTZ)[0]),
                    "solve_s": time.perf_counter() - start,
                    "failed": type(e).__name__,
                }
            )
        raise
    elapsed = time.perf_counter() - start
    if _recording[0]:
        _solve_records.append(
            {
                "n": int(np.shape(ZTZ)[0]),
                "solve_s": elapsed,
                "failed": None,
                "outer_iterations": int(stats["outer_iterations"]),
                "inner_iterations": int(stats["inner_iterations"]),
                "total_iterations": int(stats["outer_iterations"] + stats["inner_iterations"]),
                "n_passive": int(stats["n_passive"]),
                "warm_start_errors": int(stats["warm_start_errors"]),
            }
        )
    return reconstruction


fnnls_mod.fnnls_cholesky = _wrapped_fnnls_cholesky


def one_evaluation(instance, memo_on: bool) -> dict:
    """One full likelihood evaluation, with its solve diagnostics and timings."""
    _solve_records.clear()
    _recording[0] = True

    start = time.perf_counter()
    fit = al.FitImaging(
        dataset=dataset,
        tracer=al.Tracer(galaxies=list(instance.galaxies)),
        adapt_images=adapt_images_for(instance),
        settings=al.Settings(use_border_relocator=True, nnls_warm_start_memo=memo_on),
        xp=np,
    )
    recon_start = time.perf_counter()
    reconstruction = np.asarray(fit.inversion.reconstruction)
    reconstruction_s = time.perf_counter() - recon_start
    figure_of_merit = float(fit.figure_of_merit)
    eval_s = time.perf_counter() - start

    _recording[0] = False
    solves = list(_solve_records)

    successful = [s for s in solves if s["failed"] is None]
    if not successful:
        raise RuntimeError("No fnnls solve was recorded for this evaluation.")

    # The final successful solve is the one that produced the reconstruction; any
    # earlier one is a memo-seed attempt the library retried.
    primary = successful[-1]

    return {
        "reconstruction_s": reconstruction_s,
        "eval_s": eval_s,
        "log_likelihood": figure_of_merit,
        "n_solves": len(solves),
        "retries": len(solves) - 1,
        "solve_s_total": float(sum(s["solve_s"] for s in solves)),
        "reconstruction": reconstruction,
        **{
            k: primary[k]
            for k in (
                "n",
                "solve_s",
                "outer_iterations",
                "inner_iterations",
                "total_iterations",
                "n_passive",
                "warm_start_errors",
            )
        },
    }


def run_sequence(instances: list, memo_on: bool) -> list[dict]:
    """Run one instance sequence end-to-end with the memo in one state."""
    nnls_memo._nnls_passive_set_memo.clear()
    rows = []
    for index, instance in enumerate(instances):
        row = one_evaluation(instance, memo_on=memo_on)
        row["index"] = index
        rows.append(row)
    nnls_memo._nnls_passive_set_memo.clear()
    return rows


# ===================================================================
# Warm-up + configuration report
# ===================================================================

from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)

_median_instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Pixel scale:             {pixel_scale} arcsec/pixel")
print(f"  Mask radius:             {mask_radius} arcsec")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Delaunay vertices:       {n_mesh_vertices}")
print(f"  Free parameters:         {model.prior_count}")
print(f"  Instances per sequence:  {N_INSTANCES}")
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"  Operated-matrix memo:    {os.environ.get('AUTOARRAY_NUMBA_OPERATED_MEMO', '(unset)')}")

print("\n--- Warm-up evaluation (numba compile) ---")
_warm_start = time.perf_counter()
_warm = one_evaluation(_median_instance, memo_on=False)
warmup_s = time.perf_counter() - _warm_start
nnls_memo._nnls_passive_set_memo.clear()
print(f"  warm-up total: {warmup_s:.4f} s (log_likelihood = {_warm['log_likelihood']})")
print(
    f"  solve: n = {_warm['n']}, outer = {_warm['outer_iterations']}, "
    f"inner = {_warm['inner_iterations']}, n_passive = {_warm['n_passive']}, "
    f"warm_start_errors = {_warm['warm_start_errors']}"
)

_check_fit = al.FitImaging(
    dataset=dataset,
    tracer=al.Tracer(galaxies=list(_median_instance.galaxies)),
    adapt_images=adapt_images_for(_median_instance),
    settings=al.Settings(use_border_relocator=True),
    xp=np,
)
assert isinstance(_check_fit.inversion, InversionImagingSparseNumba), (
    f"Expected InversionImagingSparseNumba, got {type(_check_fit.inversion).__name__}"
)
del _check_fit

# ===================================================================
# The A/B runs
# ===================================================================

SEQUENCES = {
    "random_walk": random_walk_instances(N_INSTANCES),
    "iid": iid_instances(N_INSTANCES),
}

results: dict[str, dict] = {}

for sequence_name, instances in SEQUENCES.items():
    print(f"\n--- Sequence '{sequence_name}' ({N_INSTANCES} instances) ---")

    cells = {}
    for memo_on in (False, True):
        label = "on" if memo_on else "off"
        start = time.perf_counter()
        cells[label] = run_sequence(instances, memo_on=memo_on)
        print(f"  memo {label.upper():<3} done in {time.perf_counter() - start:.1f} s")

    # --- parity: the NNLS optimum is unique, so the memo must change nothing ---
    max_rel_log_likelihood = 0.0
    max_abs_reconstruction = 0.0
    for row_off, row_on in zip(cells["off"], cells["on"]):
        denominator = max(abs(row_off["log_likelihood"]), 1e-300)
        max_rel_log_likelihood = max(
            max_rel_log_likelihood,
            abs(row_on["log_likelihood"] - row_off["log_likelihood"]) / denominator,
        )
        max_abs_reconstruction = max(
            max_abs_reconstruction,
            float(np.max(np.abs(row_on["reconstruction"] - row_off["reconstruction"]))),
        )
        assert np.isclose(
            row_on["log_likelihood"], row_off["log_likelihood"], rtol=PARITY_RTOL, atol=0.0
        ), (
            f"[{sequence_name}] memo parity FAILED at instance {row_off['index']}: "
            f"off = {row_off['log_likelihood']!r}, on = {row_on['log_likelihood']!r}"
        )

    results[sequence_name] = {
        "cells": cells,
        "parity": {
            "rtol": PARITY_RTOL,
            "max_rel_log_likelihood_deviation": max_rel_log_likelihood,
            "max_abs_reconstruction_deviation": max_abs_reconstruction,
        },
    }
    print(
        f"  parity: max rel Δlog_likelihood = {max_rel_log_likelihood:.3e}, "
        f"max |Δreconstruction| = {max_abs_reconstruction:.3e}"
    )

# ===================================================================
# Summary
# ===================================================================

METRICS = (
    "outer_iterations",
    "inner_iterations",
    "total_iterations",
    "n_passive",
    "warm_start_errors",
    "solve_s",
    "reconstruction_s",
    "eval_s",
)


def summarize(rows: list[dict]) -> dict:
    summary = {"n_evaluations": len(rows), "retries": int(sum(r["retries"] for r in rows))}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in rows], dtype=float)
        summary[f"median_{metric}"] = float(np.median(values))
        summary[f"mean_{metric}"] = float(np.mean(values))
    return summary


for sequence_name, block in results.items():
    block["summary"] = {label: summarize(rows) for label, rows in block["cells"].items()}
    median_off = block["summary"]["off"]["median_total_iterations"]
    median_on = block["summary"]["on"]["median_total_iterations"]
    block["iteration_reduction_ratio"] = (
        float(median_off / median_on) if median_on else float("inf")
    )
    block["solve_speedup_ratio"] = float(
        block["summary"]["off"]["median_solve_s"] / block["summary"]["on"]["median_solve_s"]
    )
    block["eval_speedup_ratio"] = float(
        block["summary"]["off"]["median_eval_s"] / block["summary"]["on"]["median_eval_s"]
    )

al_version = al.__version__
total_retries = int(
    sum(b["summary"][k]["retries"] for b in results.values() for k in ("off", "on"))
)

print("\n" + "=" * 96)
print(f"NNLS WARM-START MEMO A/B — {instrument.upper()} — v{al_version}")
print("=" * 96)
header = (
    f"  {'sequence':<12} {'memo':<5} {'outer':>7} {'inner':>7} {'total':>7} "
    f"{'ws_err':>7} {'n_pass':>7} {'solve s':>10} {'recon s':>10} {'eval s':>10} {'retry':>6}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for sequence_name, block in results.items():
    for label in ("off", "on"):
        s = block["summary"][label]
        print(
            f"  {sequence_name:<12} {label.upper():<5} "
            f"{s['median_outer_iterations']:>7.0f} {s['median_inner_iterations']:>7.0f} "
            f"{s['median_total_iterations']:>7.0f} {s['median_warm_start_errors']:>7.0f} "
            f"{s['median_n_passive']:>7.0f} {s['median_solve_s']:>10.4f} "
            f"{s['median_reconstruction_s']:>10.4f} {s['median_eval_s']:>10.4f} "
            f"{s['retries']:>6d}"
        )
    print(
        f"  {'':<12} {'->':<5} iteration reduction "
        f"{block['iteration_reduction_ratio']:.2f}x  |  solve "
        f"{block['solve_speedup_ratio']:.2f}x  |  eval "
        f"{block['eval_speedup_ratio']:.2f}x  |  parity max rel Δlnl "
        f"{block['parity']['max_rel_log_likelihood_deviation']:.2e}"
    )
print("=" * 96)
print(
    f"  Median values over {N_INSTANCES} evaluations per cell. Total memo-seed retries: {total_retries}"
)

summary_dict = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "configuration": {
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "image_pixels_masked": int(n_image_pixels),
        "over_sampled_pixels": int(n_over_sampled_pixels),
        "source_pixels": int(n_mesh_vertices),
        "free_parameters": int(model.prior_count),
        "inversion_path": "sparse_numba",
        "use_jax": False,
        "mesh": "delaunay_hilbert_1250",
        "regularization": "constant_split",
        "lens_light": "mge_60_linear",
        "n_instances_per_sequence": N_INSTANCES,
        "random_walk_step_fraction_of_prior_width": RANDOM_WALK_STEP_FRACTION,
        "iid_unit_range": [IID_UNIT_LOW, IID_UNIT_HIGH],
        "seed": SEED,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", None),
        "autoarray_numba_operated_memo": os.environ.get("AUTOARRAY_NUMBA_OPERATED_MEMO", None),
    },
    "warmup_incl_numba_compile_s": warmup_s,
    "total_memo_seed_retries": total_retries,
    "sequences": {
        sequence_name: {
            "summary": block["summary"],
            "parity": block["parity"],
            "iteration_reduction_ratio": block["iteration_reduction_ratio"],
            "solve_speedup_ratio": block["solve_speedup_ratio"],
            "eval_speedup_ratio": block["eval_speedup_ratio"],
            "rows": {
                label: [{k: v for k, v in row.items() if k != "reconstruction"} for row in rows]
                for label, rows in block["cells"].items()
            },
        }
        for sequence_name, block in results.items()
    },
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"delaunay_numba_nnls_iterations_{instrument}_v{al_version}",
)
dict_path.write_text(json.dumps(summary_dict, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# ===================================================================
# Chart
# ===================================================================

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sequence_names = list(results.keys())
fig, axes = plt.subplots(2, len(sequence_names), figsize=(12, 7), sharex="col")

COLORS = {"off": "#C44E52", "on": "#4C72B0"}

for column, sequence_name in enumerate(sequence_names):
    block = results[sequence_name]
    for row_index, (metric, ylabel) in enumerate(
        (("total_iterations", "Active-set iterations"), ("solve_s", "fnnls solve (s)"))
    ):
        ax = axes[row_index, column]
        for label in ("off", "on"):
            rows = block["cells"][label]
            ax.plot(
                [r["index"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                markersize=3,
                linewidth=1.3,
                color=COLORS[label],
                label=f"memo {label.upper()}",
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3)
        if row_index == 0:
            ax.set_title(
                f"{sequence_name}  —  median iteration ratio OFF/ON = "
                f"{block['iteration_reduction_ratio']:.2f}x",
                fontsize=10,
            )
            ax.legend(fontsize=9)
        else:
            ax.set_xlabel("Evaluation index", fontsize=10)

fig.suptitle(
    f"NNLS Warm-Start Memo A/B — Delaunay numba CPU — {instrument.upper()}  (v{al_version})",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Chart saved to:        {chart_path}")
