"""
Linear-solver corpus capture: add a group of real positive-only systems to the corpus
=====================================================================================

Captures ``(curvature_reg_matrix, data_vector)`` exactly as the JAX likelihood hands them to
``autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from`` (the function
is wrapped with ``mge_nnls_capture._recording_solver``, so the systems are post-regularisation,
pre-Jacobi) and adds them to the corpus as one group through ``_corpus.add_group`` — which also
computes and stores the fnnls reference. One ``--source`` per group:

``slam48`` -> group ``slam48_hst``
    All 48 SLaM ``source_lp[1]`` near-truth systems of the PyAutoArray#571 capture
    (``scripts/imaging/hazards/mge_nnls_capture.py``, which keeps only 8). The dataset, model and
    the 48 vectors come from that script's own functions (``_dataset``,
    ``_slam_source_lp_model``, ``_vectors``) so they are the same 48; the 8 fixture systems
    (group ``slam_fixture_571``) are compared against their matching vectors and the result is
    recorded on the group source as ``fixture_check``.
``slam_spread`` -> group ``slam_spread_hst``
    A conditioning spread of the same recipe at 6 of the 48 vectors (0, 8, 16, 24, 32, 40) and
    four variants: the lens-light MGE ``sigma_min`` scaled by 0.5 and 2 (the model otherwise
    identical, same prior creation order, so the same vectors), and the dataset noise map scaled
    by 0.3 and 3 (``noise_map.fits`` multiplied by the factor, then loaded, masked and
    over-sampled exactly as ``_dataset`` does). 24 systems. A noise-map factor ``f`` scales the
    curvature matrix and ``q`` by ``1/f**2`` but *not* the unregularised-column diagonal shift
    (``settings.no_regularization_add_to_curvature_diag_value`` = 1e-3, added to every column
    here), so ``Q = C/f**2 + 1e-3 I``: ``max|q|`` moves by ``1/f**2`` and ``cond(Q)`` roughly by
    ``1/f**2`` too (the shift sets the smallest eigenvalue), and the reference solution changes
    (verified at v00: off-diagonal and ``q`` match ``1/f**2`` scaling to 1e-14, the diagonal
    differs by exactly ``(1/f**2 - 1) * 1e-3`` after rescaling).
``euclid_vis_lp`` -> group ``euclid_vis_lp``
    The system(s) the euclid pipeline's
    ``tests/test_compute_latent_variable.py::test_latent_euclid_variables_traces_under_jax_jit``
    evaluates: the ``vis_lp`` model at ``_ordered_median_vector`` on the committed simulated
    ``euclid_dr1_like`` dataset (``PYAUTO_SMALL_DATASETS`` unset). Every positive-only solve made
    by the jitted ``LatentEuclid.variables`` call and by the jitted log likelihood at that vector
    is captured (identical systems are stored once, ``euclid_vis_lp_k<i>``); the eager NumPy and
    jitted ``total_source_flux`` latents are recorded as the group's ``latent_reference``.

Every system records ``source`` (script, args, git SHA of this repo, and the
``_profile_cli._source_revisions()`` library SHAs), the model, the column layout
(``source_column_index_list`` / ``no_regularization_index_list``, read from the NumPy fit's
inversion at a captured vector, not inferred) and ``library_versions``.

Run from the repo root on CPU fp64::

    python scripts/lens/solver/capture.py --source slam48
    python scripts/lens/solver/capture.py --source slam_spread
    python scripts/lens/solver/capture.py --source euclid_vis_lp

``--label`` suffixes the group name (a re-capture against another library lands beside the
original group instead of replacing it).

Output
------
``results/lens/solver/corpus/<group>.npz`` plus its entry in ``corpus/manifest.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
for _p in (REPO_ROOT / "scripts" / "misc", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from autoarray.inversion.inversion import inversion_util  # noqa: E402

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke) -- before the capture module import,
# which carries its own smoke exit.
if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

sys.path.insert(0, str(REPO_ROOT / "scripts" / "imaging" / "hazards"))
import _corpus  # noqa: E402
import mge_nnls_capture as mnc  # noqa: E402

SCRIPT = "scripts/lens/solver/capture.py"
SLAM_MODEL = (
    "SLaM source_lp[1]: lens 2x20 MGE (sigma_min=pixel_scale/10) + source 20 MGE, "
    "free Isothermal + ExternalShear"
)
SLAM_DATASET = "dataset/imaging/hst (mask 3.5 arcsec, radial over-sampling 4/2/2)"
SPREAD_VECTORS = (0, 8, 16, 24, 32, 40)
SPREAD_VARIANTS = (
    ("sigma_min_x0p5", {"sigma_min_factor": 0.5}),
    ("sigma_min_x2", {"sigma_min_factor": 2.0}),
    ("noise_x0p3", {"noise_factor": 0.3}),
    ("noise_x3", {"noise_factor": 3.0}),
)
EUCLID_ROOT = REPO_ROOT.parent / "euclid_strong_lens_modeling_pipeline"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _source_block(args: argparse.Namespace, **extras) -> dict:
    from _profile_cli import _library_versions, _source_revisions

    try:
        revisions = _source_revisions()
    except Exception as exc:  # noqa: BLE001 -- provenance must never kill a capture
        revisions = {"error": f"unavailable: {type(exc).__name__}"}
    return {
        "script": SCRIPT,
        "args": ["--source", args.source] + (["--label", args.label] if args.label else []),
        "git_sha": revisions.get("autolens_profiling"),
        "library_revisions": revisions,
        "library_versions": _library_versions(),
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
        "note": "Q = curvature_reg_matrix, q = data_vector as passed to "
        "reconstruction_positive_only_from on the JAX path (pre-Jacobi).",
        **extras,
    }


def _column_layout(fit) -> tuple[list[int], list[int], list[dict]]:
    """Source columns (the highest-redshift galaxy's linear objects) from a NumPy fit's inversion."""
    inversion = fit.inversion
    galaxy_of = inversion.linear_obj_galaxy_dict
    blocks, start = [], 0
    for linear_obj in inversion.linear_obj_list:
        stop = start + int(linear_obj.params)
        galaxy = galaxy_of[linear_obj]
        blocks.append(
            {
                "range": [start, stop],
                "type": type(linear_obj).__name__,
                "redshift": float(galaxy.redshift),
            }
        )
        start = stop
    z_source = max(b["redshift"] for b in blocks)
    source_cols = [i for b in blocks if b["redshift"] == z_source for i in range(*b["range"])]
    return source_cols, [int(i) for i in inversion.no_regularization_index_list], blocks


class _Recorder:
    """Swap ``reconstruction_positive_only_from`` for ``mnc._recording_solver`` in a ``with``."""

    def __enter__(self):
        mnc._CAPTURED.clear()
        inversion_util.reconstruction_positive_only_from = mnc._recording_solver
        return mnc._CAPTURED

    def __exit__(self, *exc):
        inversion_util.reconstruction_positive_only_from = mnc._ORIGINAL
        return False


def _jit_likelihood_systems(analysis, model, ignore_assertions: bool = False):
    """A jitted ``vector -> (log_likelihood, [(Q, q), ...])`` over every solve the trace makes.

    ``ignore_assertions=True`` is the production ``Fitness`` JAX path for models with assertions
    (they are applied to the merit with ``where``; a traced instance cannot check them in Python).
    """

    def fn(vector):
        mnc._CAPTURED.clear()
        kwargs = {"ignore_assertions": True} if ignore_assertions else {}
        instance = model.instance_from_vector(vector, xp=jnp, **kwargs)
        log_likelihood = analysis.log_likelihood_function(instance=instance)
        return log_likelihood, [(jnp.asarray(c[0]), jnp.asarray(c[1])) for c in mnc._CAPTURED]

    return jax.jit(fn)


def _as_np(Q, q) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(Q, dtype=np.float64), np.asarray(q, dtype=np.float64)


# ---------------------------------------------------------------------------
# SLaM (PyAutoArray#571 recipe)
# ---------------------------------------------------------------------------


def _slam_model(dataset, mask_radius, sigma_min_factor: float = 1.0):
    """``mnc._slam_source_lp_model`` with the lens MGE ``sigma_min`` scaled by ``sigma_min_factor``.

    At factor 1 the base function itself is used. Otherwise its body is repeated with only
    ``sigma_min`` changed, in the same prior creation order (checked by the caller: the sampled
    vectors must match the base model's).
    """
    if sigma_min_factor == 1.0:
        return mnc._slam_source_lp_model(dataset, mask_radius)
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        sigma_min=sigma_min_factor * dataset.pixel_scales[0] / 10.0,
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


def _slam_dataset(noise_factor: float = 1.0):
    """``mnc._dataset`` with ``noise_map.fits`` multiplied by ``noise_factor`` before loading."""
    if noise_factor == 1.0:
        return mnc._dataset()
    from astropy.io import fits

    src = mnc.REPO_ROOT / "dataset" / "imaging" / "hst"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with fits.open(src / "noise_map.fits") as hdul:
            hdul[0].data = np.asarray(hdul[0].data, dtype=np.float64) * noise_factor
            hdul.writeto(tmp / "noise_map.fits")
        dataset = al.Imaging.from_fits(
            data_path=src / "data.fits",
            psf_path=src / "psf.fits",
            noise_map_path=tmp / "noise_map.fits",
            pixel_scales=0.05,
        )
    mask_radius = 3.5
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=mask_radius
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size), mask_radius


def _capture_slam(vector_indices, *, sigma_min_factor=1.0, noise_factor=1.0, base_vectors=None):
    """``[(vector_index, Q, q)]`` plus the column layout for one SLaM configuration."""
    dataset, mask_radius = _slam_dataset(noise_factor)
    model = _slam_model(dataset, mask_radius, sigma_min_factor)
    vectors = mnc._vectors(model)
    if base_vectors is not None and not np.array_equal(vectors, base_vectors):
        raise RuntimeError("variant model draws different vectors: prior creation order changed")
    analysis_jax = al.AnalysisImaging(dataset=dataset, use_jax=True)
    analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False)
    out = []
    with _Recorder():
        traced = _jit_likelihood_systems(analysis_jax, model)
        for i in vector_indices:
            log_likelihood, systems = traced(jnp.asarray(vectors[i]))
            if len(systems) != 1:
                raise RuntimeError(f"expected 1 solve per SLaM likelihood, got {len(systems)}")
            Q, q = _as_np(*systems[0])
            out.append((i, Q, q, float(log_likelihood)))
    fit = analysis_np.fit_from(instance=model.instance_from_vector(vectors[vector_indices[0]]))
    layout = _column_layout(fit)
    return out, layout, vectors


def _fixture_check(captured: dict[int, tuple]) -> dict:
    """Compare group ``slam_fixture_571`` against the matching freshly captured vectors."""
    try:
        fixture = _corpus.load_corpus([_corpus.FIXTURE_571_GROUP])
    except KeyError:
        return {"reproduces_fixture": None, "reason": "fixture group not in corpus"}
    per = []
    for s in fixture:
        i = s.meta["vector_index"]
        Q, q = captured[i][0], captured[i][1]
        rel_Q = float(np.max(np.abs(Q - s.Q)) / np.max(np.abs(s.Q)))
        rel_q = float(np.max(np.abs(q - s.q)) / np.max(np.abs(s.q)))
        per.append(
            {
                "fixture_system": s.name,
                "vector_index": i,
                "bit_identical": bool(np.array_equal(Q, s.Q) and np.array_equal(q, s.q)),
                "max_rel_diff_Q": rel_Q,
                "max_rel_diff_q": rel_q,
            }
        )
    max_rel = max(max(p["max_rel_diff_Q"], p["max_rel_diff_q"]) for p in per)
    return {
        "reproduces_fixture": bool(max_rel <= 1.0e-12),
        "tolerance": 1.0e-12,
        "max_rel_diff": max_rel,
        "definition": "max|X_new - X_fixture| / max|X_fixture| over X in (Q, q), per system",
        "all_bit_identical": all(p["bit_identical"] for p in per),
        "systems": per,
    }


def run_slam48(args) -> tuple[str, list[dict], dict]:
    captured, (src_cols, no_reg, blocks), _ = _capture_slam(list(range(mnc.N_VECTORS)))
    by_index = {i: (Q, q) for i, Q, q, _ in captured}
    check = _fixture_check(by_index)
    # Per-system detail goes on the system entries; the group source (copied onto every system
    # by add_group) keeps only the summary.
    per_fixture = {p["vector_index"]: p for p in check.pop("systems", [])}
    systems = [
        {
            "name": f"v{i:02d}",
            "Q": Q,
            "q": q,
            "model": SLAM_MODEL,
            "source_column_index_list": src_cols,
            "no_regularization_index_list": no_reg,
            "vector_index": i,
            "log_likelihood_jax": ll,
            "fixture_571_check": per_fixture.get(i),
        }
        for i, Q, q, ll in captured
    ]
    source = _source_block(
        args,
        dataset=SLAM_DATASET,
        recipe="mge_nnls_capture._dataset / _slam_source_lp_model / _vectors "
        f"(seed {mnc.SEED}, box +/-{mnc.BOX / 2}, {mnc.N_VECTORS} vectors)",
        column_blocks=blocks,
        fixture_check=check,
    )
    print(
        f"fixture check: reproduces={check.get('reproduces_fixture')} "
        f"max_rel_diff={check.get('max_rel_diff')} bit_identical={check.get('all_bit_identical')}"
    )
    return "slam48_hst", systems, source


def run_slam_spread(args) -> tuple[str, list[dict], dict]:
    dataset, mask_radius = mnc._dataset()
    base_vectors = mnc._vectors(mnc._slam_source_lp_model(dataset, mask_radius))
    systems, blocks_by_variant = [], {}
    for variant, kwargs in SPREAD_VARIANTS:
        captured, (src_cols, no_reg, blocks), _ = _capture_slam(
            list(SPREAD_VECTORS), base_vectors=base_vectors, **kwargs
        )
        blocks_by_variant[variant] = blocks
        for i, Q, q, ll in captured:
            systems.append(
                {
                    "name": f"{variant}_v{i:02d}",
                    "Q": Q,
                    "q": q,
                    "model": SLAM_MODEL,
                    "source_column_index_list": src_cols,
                    "no_regularization_index_list": no_reg,
                    "vector_index": i,
                    "variant": variant,
                    "sigma_min_factor": kwargs.get("sigma_min_factor", 1.0),
                    "noise_factor": kwargs.get("noise_factor", 1.0),
                    "log_likelihood_jax": ll,
                }
            )
        print(f"  variant {variant}: {len(captured)} systems", flush=True)
    source = _source_block(
        args,
        dataset=SLAM_DATASET,
        vector_indices=list(SPREAD_VECTORS),
        variants={v: k for v, k in SPREAD_VARIANTS},
        variant_definitions={
            "sigma_min_factor": "lens-light MGE sigma_min = factor * pixel_scale / 10 (both "
            "bases); every other model choice and the prior creation order unchanged, so the "
            "vectors are the base model's (asserted).",
            "noise_factor": "dataset/imaging/hst/noise_map.fits multiplied by the factor, then "
            "loaded / masked (3.5 arcsec) / over-sampled (4/2/2) exactly as "
            "mge_nnls_capture._dataset. q and the curvature matrix scale by 1/factor**2 but "
            "the unregularised-column diagonal shift (no_regularization_add_to_curvature_"
            "diag_value = 1e-3, all 60 columns) does not: Q = C/factor**2 + 1e-3 I, so max|q| "
            "and cond(Q) both move by ~1/factor**2 and the reference solution changes.",
        },
        column_blocks=blocks_by_variant,
    )
    return "slam_spread_hst", systems, source


# ---------------------------------------------------------------------------
# Euclid vis_lp (the failing latent test's system)
# ---------------------------------------------------------------------------


def run_euclid_vis_lp(args) -> tuple[str, list[dict], dict]:
    os.environ.pop("PYAUTO_SMALL_DATASETS", None)
    for p in (EUCLID_ROOT, EUCLID_ROOT / "tests"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import test_compute_latent_variable as tl
    import util
    from likelihood_breakdown.provenance import git_revision

    euclid_dataset = util.load_vis_dataset(tl.SIMULATED_DATASET, sample_name=tl.SIMULATED_SAMPLE)
    analysis_np = tl._analysis_from(euclid_dataset)
    analysis_jax = tl._analysis_from(euclid_dataset, use_jax=True)
    model = tl._vis_lp_model(euclid_dataset)
    vector = tl._ordered_median_vector(model)
    keys = list(util.LatentEuclid.keys(analysis_np))
    k_flux = keys.index("total_source_flux")

    eager = [
        float(v)
        for v in util.LatentEuclid.variables(analysis=analysis_np, parameters=vector, model=model)
    ]

    def latent_fn(parameters):
        mnc._CAPTURED.clear()
        values = util.LatentEuclid.variables(
            analysis=analysis_jax, parameters=parameters, model=model
        )
        return values, [(jnp.asarray(c[0]), jnp.asarray(c[1])) for c in mnc._CAPTURED]

    with _Recorder():
        values, latent_systems = jax.jit(latent_fn)(jnp.asarray(vector))
        traced = [float(v) for v in values]
        log_likelihood, ll_systems = _jit_likelihood_systems(
            analysis_jax, model, ignore_assertions=True
        )(jnp.asarray(vector))

    fit = analysis_np.fit_from(instance=model.instance_from_vector(vector))
    src_cols, no_reg, blocks = _column_layout(fit)

    unique, calls = [], []
    for context, sys_list in (
        ("latent_variables_jit", latent_systems),
        ("log_likelihood_jit", ll_systems),
    ):
        for j, (Q, q) in enumerate(sys_list):
            Q, q = _as_np(Q, q)
            k = next(
                (
                    k
                    for k, (Qu, qu) in enumerate(unique)
                    if Qu.shape == Q.shape and np.array_equal(Qu, Q) and np.array_equal(qu, q)
                ),
                None,
            )
            if k is None:
                unique.append((Q, q))
                k = len(unique) - 1
            calls.append({"context": context, "call_index": j, "system": f"euclid_vis_lp_k{k}"})
    systems = []
    for k, (Q, q) in enumerate(unique):
        name = f"euclid_vis_lp_k{k}"
        systems.append(
            {
                "name": name,
                "Q": Q,
                "q": q,
                "model": "euclid vis_lp (initial_lens_model.vis_lp_model_from) at "
                "_ordered_median_vector, dataset simulated/euclid_dr1_like",
                "source_column_index_list": src_cols
                if Q.shape[0] == fit.inversion.total_params
                else None,
                "no_regularization_index_list": no_reg
                if Q.shape[0] == fit.inversion.total_params
                else None,
                "produced_by": [c for c in calls if c["system"] == name],
            }
        )
    latent_reference = {
        "key": "total_source_flux",
        "eager_numpy": eager[k_flux],
        "jit": traced[k_flux],
        "rel_diff": (traced[k_flux] - eager[k_flux]) / eager[k_flux],
        "all_keys": keys,
        "eager_numpy_all": eager,
        "jit_all": traced,
        "log_likelihood_jit": float(log_likelihood),
        "log_likelihood_numpy": float(fit.log_likelihood),
    }
    source = _source_block(
        args,
        test="euclid_strong_lens_modeling_pipeline/tests/test_compute_latent_variable.py::"
        "test_latent_euclid_variables_traces_under_jax_jit",
        euclid_pipeline_git_sha=git_revision(EUCLID_ROOT.resolve()),
        dataset=f"euclid_strong_lens_modeling_pipeline/dataset/{tl.SIMULATED_SAMPLE}/"
        f"{tl.SIMULATED_DATASET} (util.load_vis_dataset, PYAUTO_SMALL_DATASETS unset)",
        parameter_vector=[float(v) for v in vector],
        column_blocks=blocks,
        solve_calls=calls,
        latent_reference=latent_reference,
    )
    print(
        f"euclid: {len(calls)} solve calls -> {len(unique)} unique systems; "
        f"total_source_flux eager={eager[k_flux]:.6g} jit={traced[k_flux]:.6g}"
    )
    return "euclid_vis_lp", systems, source


RUNNERS = {"slam48": run_slam48, "slam_spread": run_slam_spread, "euclid_vis_lp": run_euclid_vis_lp}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--source", choices=sorted(RUNNERS), required=True)
    parser.add_argument(
        "--label", default="", help="Suffix for the group name (group becomes <group>_<label>)."
    )
    args = parser.parse_args()
    group, systems, source = RUNNERS[args.source](args)
    if args.label:
        group = f"{group}_{args.label}"
    entry = _corpus.add_group(group, systems, source)
    for s in entry["systems"]:
        print(f"{group}/{s['name']}: n={s['n']} cond={s['cond_Q']:.3e} max|q|={s['max_abs_q']:.3e}")
    size = (_corpus.CORPUS_DIR / entry["npz"]).stat().st_size
    print(
        f"wrote {len(systems)} systems -> {_corpus.CORPUS_DIR / entry['npz']} ({size / 1e3:.1f} kB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
