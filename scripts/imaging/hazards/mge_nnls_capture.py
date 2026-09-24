"""
Hazard capture: JAX PDIP NNLS non-convergence on the SLaM MGE model
====================================================================

Captures the exact positive-only linear systems ``(curvature_reg_matrix,
data_vector)`` that the JAX likelihood hands to
``autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from``
for the SLaM ``source_lp[1]`` model (PyAutoArray#571), so the solver failure can
be reproduced as a small library regression fixture without rebuilding a lens
model.

Model (mirrors ``autolens_workspace/scripts/imaging/features/pixelization/slam.py``
``source_lp``): lens light = 2 bases x 20 Gaussians with
``sigma_min = pixel_scale / 10``, source = 20 Gaussians, free Isothermal +
ExternalShear (60 linear columns). The free mass/shear priors are centred on the
simulator truth of the profiling HST dataset (``dataset/imaging/hst``), and 48
parameter vectors are drawn from a seeded +/-0.005 box around a generic
near-truth point (the 2026-09-24 MGE audit recipe).

For each vector the script records the system on the JAX path (the solver
function is wrapped so its inputs are returned from the jitted likelihood --
post-regularisation, pre-Jacobi), then off-line:

- the Jacobi-preconditioned PDIP solve (``autoarray.util.jax_nnls.solve_nnls``,
  exactly as ``inversion_util`` scales it) at iteration caps 50 and 200:
  ``converged`` and ``pdip_iter``;
- the NumPy ``fnnls_cholesky`` solution;
- the NNLS objective ``0.5 x^T Q x - q^T x`` of each solution;
- the JAX and NumPy log-likelihoods of the full fit.

Output
------

``results/hazards/component/mge/nnls_capture_slam_hst_v{al_version}.json`` -- the
per-vector summary (no matrices), and a compressed ``.npz`` of the same basename
holding 8 systems (5 never converging at cap 200, 2 hitting cap 50 but
converging by 200, 1 healthy) as ``Q_<i>`` / ``q_<i>`` plus a ``meta`` JSON
string. The ``.npz`` is the fixture copied to PyAutoArray
``test_autoarray/inversion/inversion/files/mge_slam_nnls_systems.npz``.

Run from the repo root on CPU fp64::

    python scripts/imaging/hazards/mge_nnls_capture.py
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
MISC_ROOT = REPO_ROOT / "scripts" / "misc"
if str(MISC_ROOT) not in sys.path:
    sys.path.insert(0, str(MISC_ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from autoarray.inversion.inversion import inversion_util  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

N_VECTORS = 48
SEED = 1
BOX = 0.01  # full width: +/-0.005 around the near-truth point
CAPS = (50, 200)
N_NEVER, N_SLOW, N_HEALTHY = 5, 2, 1

OUT_DIR = REPO_ROOT / "results" / "hazards" / "component" / "mge"


def _dataset():
    dataset_path = REPO_ROOT / "dataset" / "imaging" / "hst"
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=0.05,
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
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size), mask_radius


def _slam_source_lp_model(dataset, mask_radius):
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.1)
    ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell_comps[0], sigma=0.05)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell_comps[1], sigma=0.05)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.02)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.02)
    # Created last on purpose: autofit orders the parameter vector by prior creation, so this
    # order fixes which parameter each entry of the sampled vectors (below) perturbs.
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
        ),
        fields=af.Model(al.MassField, redshift=0.5, shear=shear),
    )


def _vectors(model):
    medians = np.asarray(model.physical_values_from_prior_medians, dtype=float)
    # A generic point: ell_comps = (0, 0) exactly has a NaN d(angle)/d(e).
    centre = medians + 0.013 * np.sin(np.arange(medians.size) + 1.0)
    rng = np.random.default_rng(SEED)
    return np.stack([centre + BOX * (rng.random(centre.size) - 0.5) for _ in range(N_VECTORS)])


_CAPTURED: list = []
_ORIGINAL = inversion_util.reconstruction_positive_only_from


def _recording_solver(data_vector, curvature_reg_matrix, *args, **kwargs):
    xp = kwargs.get("xp", args[1] if len(args) > 1 else np)
    if xp.__name__.startswith("jax"):
        _CAPTURED.append((curvature_reg_matrix, data_vector))
    return _ORIGINAL(data_vector, curvature_reg_matrix, *args, **kwargs)


def _objective(Q, q, x):
    return float(0.5 * x @ Q @ x - q @ x)


def _git_sha(module) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=Path(module.__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    import autoarray as aa
    from autoarray.util.fnnls import fnnls_cholesky
    from autoarray.util.jax_nnls import solve_nnls

    dataset, mask_radius = _dataset()
    model = _slam_source_lp_model(dataset, mask_radius)
    vectors = _vectors(model)

    analysis_jax = al.AnalysisImaging(dataset=dataset, use_jax=True)
    analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False)

    def likelihood_and_system(vector):
        _CAPTURED.clear()
        instance = model.instance_from_vector(vector, xp=jnp)
        log_likelihood = analysis_jax.log_likelihood_function(instance=instance)
        calls = len(_CAPTURED)
        Q, q = _CAPTURED[0]
        return log_likelihood, jnp.asarray(Q), jnp.asarray(q), calls

    inversion_util.reconstruction_positive_only_from = _recording_solver
    try:
        traced = jax.jit(lambda v: likelihood_and_system(v)[:3])
        calls_per_trace = None
        rows, systems = [], []
        solvers = {
            cap: jax.jit(lambda Q, q, cap=cap: solve_nnls(Q, q, max_iter=cap)) for cap in CAPS
        }
        for i, vector in enumerate(vectors):
            log_likelihood_jax, Q, q = traced(jnp.asarray(vector))
            if calls_per_trace is None:
                calls_per_trace = len(_CAPTURED)
            log_likelihood_np = float(
                analysis_np.log_likelihood_function(instance=model.instance_from_vector(vector))
            )
            Q = np.asarray(Q, dtype=float)
            q = np.asarray(q, dtype=float)
            D = 1.0 / np.sqrt(np.diag(Q))
            Q_pc = Q * D[:, None] * D[None, :]
            q_pc = q * D

            x_fnnls = np.asarray(
                fnnls_cholesky(Q, q, P_initial=np.linalg.solve(Q, q) > 0), dtype=float
            )
            row = {
                "i": i,
                "log_likelihood_jax": float(log_likelihood_jax),
                "log_likelihood_numpy": log_likelihood_np,
                "objective_fnnls": _objective(Q, q, x_fnnls),
                "fnnls_n_positive": int(np.sum(x_fnnls > 0)),
            }
            for cap, solver in solvers.items():
                x, _, _, converged, iterations = solver(jnp.asarray(Q_pc), jnp.asarray(q_pc))
                x = np.asarray(x) * D
                row[f"converged_{cap}"] = int(converged)
                row[f"pdip_iter_{cap}"] = int(iterations)
                row[f"objective_pdip_{cap}"] = (
                    _objective(Q, q, x) if np.all(np.isfinite(x)) else None
                )
            rows.append(row)
            systems.append((Q, q))
            print(
                i,
                f"ll jax/np {row['log_likelihood_jax']:.4g} {log_likelihood_np:.4g}",
                "cap50",
                row["converged_50"],
                row["pdip_iter_50"],
                "cap200",
                row["converged_200"],
                row["pdip_iter_200"],
                flush=True,
            )
    finally:
        inversion_util.reconstruction_positive_only_from = _ORIGINAL

    never = [r["i"] for r in rows if r["converged_200"] == 0]
    slow = sorted(
        (r for r in rows if r["converged_50"] == 0 and r["converged_200"] == 1),
        key=lambda r: -r["pdip_iter_200"],
    )
    slow = [r["i"] for r in slow]
    healthy = sorted((r for r in rows if r["converged_50"] == 1), key=lambda r: r["pdip_iter_50"])
    healthy = [healthy[len(healthy) // 2]["i"]] if healthy else []
    chosen = (
        [(i, "never_converges_cap200") for i in never[:N_NEVER]]
        + [(i, "cap50_hit_converges_by_200") for i in slow[:N_SLOW]]
        + [(i, "healthy") for i in healthy[:N_HEALTHY]]
    )

    import autofit
    import autogalaxy

    versions = {
        "autolens": al.__version__,
        "autoarray": aa.__version__,
        "jax": jax.__version__,
        "shas": {
            "PyAutoArray": _git_sha(aa),
            "PyAutoGalaxy": _git_sha(autogalaxy),
            "PyAutoLens": _git_sha(al),
            "PyAutoFit": _git_sha(autofit),
        },
    }
    generator = "autolens_profiling/scripts/imaging/hazards/mge_nnls_capture.py"
    date = datetime.date.today().isoformat()

    npz_meta = {
        "generator": generator,
        "date": date,
        "issue": "PyAutoLabs/PyAutoArray#571",
        "versions": versions,
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
        "note": "Q = curvature_reg_matrix, q = data_vector as passed to "
        "reconstruction_positive_only_from on the JAX path (pre-Jacobi).",
        "systems": [],
    }
    arrays = {}
    for k, (i, category) in enumerate(chosen):
        Q, q = systems[i]
        arrays[f"Q_{k}"] = Q
        arrays[f"q_{k}"] = q
        r = rows[i]
        npz_meta["systems"].append(
            {
                "key": k,
                "vector_index": i,
                "category": category,
                "converged_50": r["converged_50"],
                "pdip_iter_50": r["pdip_iter_50"],
                "converged_200": r["converged_200"],
                "pdip_iter_200": r["pdip_iter_200"],
                "objective_fnnls": r["objective_fnnls"],
            }
        )
    arrays["meta"] = np.asarray(json.dumps(npz_meta))

    summary = {
        "hazard": "jax_pdip_nnls_nonconvergence",
        "issue": "PyAutoLabs/PyAutoArray#571",
        "generator": generator,
        "date": date,
        "model": "SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + "
        "source 20 MGE, free Isothermal + ExternalShear",
        "dataset": "dataset/imaging/hst (mask 3.5 arcsec, radial over-sampling 4/2/2)",
        "n_vectors": N_VECTORS,
        "seed": SEED,
        "box_half_width": BOX / 2,
        "linear_columns": int(systems[0][0].shape[0]),
        "solver_calls_per_likelihood": calls_per_trace,
        "versions": versions,
        "device": npz_meta["device"],
        "x64": npz_meta["x64"],
        "counts": {
            "unconverged_cap50": sum(1 - r["converged_50"] for r in rows),
            "unconverged_cap200": sum(1 - r["converged_200"] for r in rows),
            "log_likelihood_mismatch_gt_0p1": sum(
                1
                for r in rows
                if not np.isfinite(r["log_likelihood_jax"])
                or abs(r["log_likelihood_jax"] - r["log_likelihood_numpy"]) > 0.1
            ),
        },
        "fixture_systems": npz_meta["systems"],
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"nnls_capture_slam_hst_v{al.__version__}"
    (OUT_DIR / f"{basename}.json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez_compressed(OUT_DIR / f"{basename}.npz", **arrays)
    print(json.dumps(summary["counts"]), "fixture", [c for _, c in chosen])
    print("WROTE", OUT_DIR / f"{basename}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
