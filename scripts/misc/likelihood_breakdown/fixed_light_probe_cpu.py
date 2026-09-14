"""CPU probe: what does the source-only inversion cost once the lens light is fixed?

The measurement behind ``fixed-lens-light-source-only`` (issue #248). It is a
*CPU anatomy* run, not a timing cell — nothing here claims a millisecond on any
device. What it measures is structure:

1. **Conditioning.** ``cond(F + λH)`` with the 60 unregularised linear-MGE
   columns in the system (S0) and without them (S3).
2. **Active-set anatomy.** How big the NNLS active set ``A*`` actually is, how
   much of it the unconstrained solve already identifies (``N = {x_unc < 0}``),
   and how deep an ε-band around zero has to go to cover it.
3. **The certified active-set scheme.** How many passes and factorisations
   :func:`active_set_steps.active_set_certified` needs to *certify* the KKT
   point, against the library's PDIP iteration count on the same matrices, and
   what the log-evidence error of a fixed pass budget is.
4. **Stability under the mass model.** The same anatomy under ±1 % perturbations
   of ``einstein_radius`` and of ``ell_comps_0``, each rebuilt end to end (the
   S0 fit is redone with the perturbed mass, its MGE intensities re-converted,
   and the light re-subtracted), plus a warm start seeded from the fiducial's
   active set.

Systems
-------

======  ===========================================================
S0      MGE-60 **linear** lens light + source mapper — the control:
        exactly what the production breakdown cells fit today.
S1      ``data − blurred(true simulator Sersic)``, source mapper only
        (``--with-s1``) — the unreachable best case, since the true
        light is never known.
S3      ``data − blurred(MGE-60 basis at the S0-solved intensities)``,
        source mapper only. The production-relevant system: what SLaM
        has after ``light[1]``.
======  ===========================================================

Edge zeroing, reported both ways
--------------------------------

The library's positive-only solver does not solve the full system on a
rectangular mesh: ``Inversion.solve_ids_to_keep`` drops the border ring (152 of
1521 source pixels on the fiducial) and scatters zeros back. Every scheme is
therefore run twice — ``edge_zeroed`` (the library's problem: that index set is
in the fixed set from pass 0 and never released, and the reference is the
library's own ``inversion.reconstruction``) and ``pure`` (the full QP, reference
``nnls_pdip``). The two differ by ~215 nats of log-evidence on the rectangular
fiducial; on Delaunay there is no edge zeroing and they coincide. Only the
``edge_zeroed`` numbers describe what a production solver would have to do.

Dense only
----------

``sparse_steps.sparse_context_from`` raises for a source-only inversion (no
linear func list), and ``InversionImagingSparse`` would read a ``weight_map``
baked from the *unsubtracted* image, so the sparse data vector never sees the
subtraction. See ``active_set_steps`` for the full argument. Dense and sparse
totals agree to <10 % on the matrix-free sweep and every quantity here is the
same code on both legs.

Usage
-----

::

    cd autolens_profiling
    python scripts/misc/likelihood_breakdown/fixed_light_probe_cpu.py --mesh rectangular --source-pixels 400
    python scripts/misc/likelihood_breakdown/fixed_light_probe_cpu.py --mesh rectangular --draws
    python scripts/misc/likelihood_breakdown/fixed_light_probe_cpu.py --mesh delaunay --draws

Writes ``results/misc/fixed_light_probe/<mesh>_<n_source_pixels>.{json,md}``.
The filename deliberately matches neither ``ARTIFACT_RE`` nor
``CONFIG_TAGGED_RE`` in ``scripts/misc/tooling/build_readme.py``, so this probe
is README-invisible: it is an anatomy artifact, not a profiling row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


PROFILING_ROOT = _profiling_root()
for _path in (PROFILING_ROOT, PROFILING_ROOT / "scripts" / "misc"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from instruments.imaging import INSTRUMENTS  # noqa: E402
from likelihood_breakdown import active_set_steps as ass  # noqa: E402

OUT_ROOT = PROFILING_ROOT / "results" / "misc" / "fixed_light_probe"

# The lens light the dataset was simulated with, read off
# ``scripts/misc/simulators/imaging.py`` (``setup_galaxies``). Only ``--with-s1``
# uses it; it is the *unknowable* reference, not a system anyone can fit.
TRUE_LENS_BULGE = dict(
    centre=(0.0, 0.0),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    intensity=2.0,
    effective_radius=0.6,
    sersic_index=3.0,
)

BUDGET_PASSES = (1, 2, 3, 4, 6, 8, 12)
ACTIVE_SET_TOLS = (1e-8, 1e-6, 1e-4)
EPS_BANDS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Setup — the per-mesh model construction, copied from the breakdown cells
# ---------------------------------------------------------------------------


def build_shared(mesh: str, source_pixels: int | None, instrument: str = "hst") -> dict:
    """Dataset, mask, over-sampling, adapt image and the prior-median instance.

    Copied from ``scripts/imaging/likelihood_breakdown/`` rather than imported:
    the pins in those cells are statements about an exact object graph, and a
    shared helper that later drifts would silently change what is being
    measured here. Same reason the cells themselves do not share one.
    """
    pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
    dataset_path = Path("dataset") / "imaging" / instrument
    mask_radius = 3.5

    dataset = al.Imaging.from_fits(
        data_path=PROFILING_ROOT / dataset_path / "data.fits",
        psf_path=PROFILING_ROOT / dataset_path / "psf.fits",
        noise_map_path=PROFILING_ROOT / dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=4, over_sample_size_pixelization=1)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size, over_sample_size_pixelization=1
    )

    if mesh == "rectangular":
        side = 39 if source_pixels is None else int(round(math.sqrt(source_pixels)))
        mesh_shape = (side, side)
        n_source_pixels = side * side
        n_mesh_vertices = None
    else:
        mesh_shape = None
        n_mesh_vertices = 1500 if source_pixels is None else int(source_pixels)
        n_source_pixels = n_mesh_vertices

    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

    image_plane_mesh_grid = None
    if mesh != "rectangular":
        image_mesh = al.image_mesh.Hilbert(
            pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=dataset.mask, adapt_data=adapt_image
        )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=60, centre_prior_is_uniform=True
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)

    if mesh == "rectangular":
        mesh_obj = al.mesh.RectangularBilinearAdaptImage(
            shape=mesh_shape, weight_power=1.0, weight_floor=0.0
        )
        regularization = al.reg.Constant(coefficient=1.0)
        reg_provenance = {"scheme": "constant", "coefficient": 1.0}
    else:
        mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        regularization = al.reg.AdaptSplit(
            inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1
        )
        reg_provenance = {
            "scheme": "adapt_split",
            "inner_coefficient": 0.1,
            "outer_coefficient": 10.0,
            "signal_scale": 0.1,
        }

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

    adapt_kwargs = {
        "galaxy_image_dict": {instance.galaxies.source: adapt_image},
        "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
    }
    if image_plane_mesh_grid is not None:
        adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {
            instance.galaxies.source: image_plane_mesh_grid
        }
        adapt_kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
            "('galaxies', 'source')": image_plane_mesh_grid
        }

    return {
        "instrument": instrument,
        "mesh": mesh,
        "dataset": dataset,
        "mask_radius": mask_radius,
        "instance": instance,
        "adapt_images": al.AdaptImages(**adapt_kwargs),
        "n_source_pixels": n_source_pixels,
        "reg_provenance": reg_provenance,
        "settings": al.Settings(use_border_relocator=True),
    }


def lens_galaxy(instance, *, einstein_scale: float = 1.0, ell_delta: float = 0.0, with_light=True):
    """The probe's lens, optionally with a perturbed ``Isothermal`` mass."""
    lens = instance.galaxies.lens
    mass = lens.mass
    kwargs = dict(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(float(mass.centre[0]), float(mass.centre[1])),
            ell_comps=(float(mass.ell_comps[0]) + ell_delta, float(mass.ell_comps[1])),
            einstein_radius=float(mass.einstein_radius) * einstein_scale,
        ),
        shear=lens.shear,
    )
    if with_light:
        kwargs["bulge"] = lens.bulge
    return al.Galaxy(**kwargs)


def fit_from(shared, tracer, dataset=None):
    return al.FitImaging(
        dataset=shared["dataset"] if dataset is None else dataset,
        tracer=tracer,
        adapt_images=shared["adapt_images"],
        settings=shared["settings"],
        xp=np,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def set_stats(a: np.ndarray, b: np.ndarray) -> dict:
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return {
        "n_a": int(a.sum()),
        "n_b": int(b.sum()),
        "n_intersection": intersection,
        "n_a_not_b": int(np.sum(a & ~b)),
        "n_b_not_a": int(np.sum(b & ~a)),
        "jaccard": (intersection / union) if union else 1.0,
    }


def spectrum_summary(matrix) -> dict:
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    lam_min, lam_max = float(eigenvalues[0]), float(eigenvalues[-1])
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": (lam_max / lam_min) if lam_min > 0 else float("inf"),
        "n_below_1e-6_lambda_max": int(np.sum(eigenvalues < 1e-6 * lam_max)),
        "n_nonpositive": int(np.sum(eigenvalues <= 0.0)),
    }


def _scheme_entry(
    system,
    result,
    *,
    x_reference,
    reference_log_evidence,
    budget_passes,
    with_evidence,
) -> dict:
    scale = max(float(np.max(np.abs(x_reference))), 1e-300)
    x_final = result.x * system.d_scale

    entry = result.as_dict()
    entry["max_abs_dev_vs_reference_rel"] = float(np.max(np.abs(x_final - x_reference))) / scale
    entry["final_fixed_vs_active_set_jaccard"] = None

    if with_evidence:
        entry["d_log_evidence_final"] = ass.active_set_evidence_error(
            system, x_final, reference_log_evidence=reference_log_evidence
        )
        budget = {}
        for k in budget_passes:
            if k >= len(result.iterates):
                continue
            x_k = result.iterates[k] * system.d_scale
            budget[str(k)] = {
                "n_factorisations": result.factorisations_after_pass[k],
                "n_negative_entries": int(np.sum(x_k < 0)),
                "min_entry": float(np.min(x_k)),
                "d_log_evidence_raw": ass.active_set_evidence_error(
                    system, x_k, reference_log_evidence=reference_log_evidence
                ),
                "d_log_evidence_clipped": ass.active_set_evidence_error(
                    system, x_k, reference_log_evidence=reference_log_evidence, clip=True
                ),
                "max_abs_dev_vs_reference_rel": float(np.max(np.abs(x_k - x_reference))) / scale,
            }
        entry["fixed_budget"] = budget
    return entry


def measure_system(
    name: str,
    system,
    *,
    max_passes: int,
    tau_rel: float,
    budget_passes=BUDGET_PASSES,
    with_spectrum: bool = True,
    with_evidence: bool = True,
    warm_start_mask: np.ndarray | None = None,
) -> dict:
    """Anatomy + both active-set schemes, both edge-zeroing conventions."""
    t_start = time.perf_counter()

    result: dict = {
        "system": name,
        "n_params": system.n_params,
        "n_mapper": system.n_mapper,
        "n_funcs": system.n_funcs,
        "n_edge_zeroed": int(system.edge_zero_mask.sum()),
        "edge_zeroing_active": system.ids_to_keep is not None,
        "subtracted_light_flux": system.subtracted_light_flux,
    }
    if with_spectrum:
        result["spectrum_full"] = spectrum_summary(system.curvature_reg_matrix)

    # --- the two references -------------------------------------------------
    x_pdip, converged, iterations = system.pdip()
    result["pdip_iterations"] = int(iterations)
    result["pdip_converged"] = bool(converged)

    x_library = np.asarray(system.inversion.reconstruction, dtype=float)
    result["library_log_evidence"] = float(system.fit.log_evidence)
    result["max_abs_diff_pdip_vs_library_reconstruction"] = float(
        np.max(np.abs(x_pdip - x_library))
    )

    log_ev_pdip = system.log_evidence(x_pdip) if with_evidence else None
    log_ev_library = system.log_evidence(x_library) if with_evidence else None
    result["log_evidence_pdip"] = log_ev_pdip
    result["log_evidence_library_edge_zeroed"] = log_ev_library
    if with_evidence:
        result["d_log_evidence_pdip_minus_library"] = log_ev_pdip - log_ev_library

    # --- anatomy ------------------------------------------------------------
    x_max = float(np.max(x_pdip))
    active_sets = {f"{tol:g}": x_pdip <= tol * x_max for tol in ACTIVE_SET_TOLS}
    reference_set = active_sets["1e-06"]

    x_unconstrained = system.unconstrained(use_edge_zero=False)
    negatives = x_unconstrained < 0.0
    result["d_log_evidence_unconstrained"] = (
        (system.log_evidence(x_unconstrained) - log_ev_pdip) if with_evidence else None
    )

    anatomy = {
        "max_x_pdip": x_max,
        "min_x_pdip": float(np.min(x_pdip)),
        "active_set_sizes": {k: int(v.sum()) for k, v in active_sets.items()},
        "n_negative_unconstrained": int(negatives.sum()),
        "overlap_negatives_vs_active_set": {
            k: set_stats(negatives, v) for k, v in active_sets.items()
        },
        "eps_bands": {},
    }
    unconstrained_max = float(np.max(x_unconstrained))
    for eps in EPS_BANDS:
        band = x_unconstrained < eps * unconstrained_max
        anatomy["eps_bands"][f"{eps:g}"] = {
            "n_below": int(band.sum()),
            **set_stats(band, reference_set),
        }
    result["anatomy"] = anatomy
    result["active_set_1e-6_indices"] = np.where(reference_set)[0].tolist()

    # --- the schemes, both conventions --------------------------------------
    conventions = {
        # The library's own problem: the edge-zero ring is fixed from pass 0 and
        # never released, and the answer to reproduce is inversion.reconstruction.
        "edge_zeroed": (system.edge_zero_mask, x_library, log_ev_library),
        # The pure QP, for comparison with the scratch probe and with PDIP.
        "pure": (None, x_pdip, log_ev_pdip),
    }

    schemes: dict[str, dict] = {}
    for convention, (fixed0, x_reference, reference_log_evidence) in conventions.items():
        if convention == "pure" and system.ids_to_keep is None:
            # No edge zeroing on this mesh: the two conventions are the same run.
            continue
        x_unc = ass.unconstrained_solve(system.Q, system.q, fixed0=fixed0)
        for policy in ("free_all", "free_one"):
            run = ass.active_set_certified(
                system.Q,
                system.q,
                x_unc,
                policy=policy,
                max_passes=max_passes,
                tau_rel=tau_rel,
                fixed0=fixed0,
            )
            entry = _scheme_entry(
                system,
                run,
                x_reference=x_reference,
                reference_log_evidence=reference_log_evidence,
                budget_passes=budget_passes,
                with_evidence=with_evidence,
            )
            entry["final_fixed_vs_active_set_jaccard"] = set_stats(run.fixed_set, reference_set)[
                "jaccard"
            ]
            schemes[f"{convention}__{policy}"] = entry
            log(
                f"    {name} [{convention}/{policy}] certified={run.certified} "
                f"passes={run.passes_to_certification} facts={run.n_factorisations} "
                f"dev={entry['max_abs_dev_vs_reference_rel']:.2e} "
                f"dlogev={entry.get('d_log_evidence_final')}"
            )

    if warm_start_mask is not None:
        run = ass.active_set_certified(
            system.Q,
            system.q,
            None,
            policy="free_all",
            max_passes=max_passes,
            tau_rel=tau_rel,
            fixed0=system.edge_zero_mask,
            # Releasable: the seed is the *fiducial's* active set, a guess about
            # a neighbouring parameter point. Held permanently it would certify
            # the restricted problem instead of the QP.
            initial_fixed=np.asarray(warm_start_mask, dtype=bool),
        )
        entry = _scheme_entry(
            system,
            run,
            x_reference=x_library,
            reference_log_evidence=log_ev_library,
            budget_passes=budget_passes,
            with_evidence=with_evidence,
        )
        entry["seed_vs_active_set_jaccard"] = set_stats(
            np.asarray(warm_start_mask, dtype=bool), reference_set
        )["jaccard"]
        schemes["edge_zeroed__free_all_warm_start"] = entry
        log(
            f"    {name} [warm_start] certified={run.certified} "
            f"passes={run.passes_to_certification}"
        )

    result["schemes"] = schemes

    # --- the jit-able twin, on the same system ------------------------------
    certified_run = schemes.get("edge_zeroed__free_all", {})
    budget = certified_run.get("passes_to_certification") or min(max_passes, 6)
    masked = jax.jit(ass.active_set_masked_jax, static_argnums=(3,))(
        jnp.asarray(system.Q),
        jnp.asarray(system.q),
        jnp.asarray(system.edge_zero_mask),
        int(budget),
    )
    certified_flags = np.asarray(masked["certified"])
    x_masked = np.asarray(masked["x"], dtype=float) * system.d_scale
    result["masked_jax"] = {
        "n_passes": int(budget),
        "certified_at_pass": (int(np.argmax(certified_flags)) + 1)
        if certified_flags.any()
        else None,
        "n_fixed_per_pass": [int(v) for v in np.asarray(masked["n_fixed"])],
        "n_primal_violations_per_pass": [int(v) for v in np.asarray(masked["n_primal_violations"])],
        "n_dual_violations_per_pass": [int(v) for v in np.asarray(masked["n_dual_violations"])],
        "max_abs_dev_vs_numpy_reference_rel": float(
            np.max(np.abs(x_masked - x_library)) / max(float(np.max(np.abs(x_library))), 1e-300)
        ),
        "d_log_evidence": (
            ass.active_set_evidence_error(system, x_masked, reference_log_evidence=log_ev_library)
            if with_evidence
            else None
        ),
    }

    result["wall_s"] = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        return f"{value:.4g}"
    return str(value)


def render_markdown(payload: dict) -> str:
    systems = [s for s in payload["systems"] if "schemes" in s]
    names = [s["system"] for s in systems]

    lines = [
        f"# Fixed-lens-light source-only probe — mesh {payload['mesh']}, "
        f"{payload['source_pixels']} source pixels",
        "",
        f"`{payload['artifact']}` — autolens {payload['autolens_version']}, instrument "
        f"{payload['instrument']}, reg {payload['regularization']['scheme']}, "
        f"tau_rel = {payload['tau_rel']:g}, pass cap {payload['max_passes']}. CPU, fp64, dense.",
        "",
        "`d log_ev` is nats against the reference for that convention — the library's own "
        "edge-zeroed reconstruction for `edge_zeroed`, the PDIP solution for `pure`. A "
        "**positive** value means an infeasible early iterate scoring above the constrained "
        "optimum, which is not a usable evidence.",
        "",
        "| quantity | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(label, getter):
        lines.append(f"| {label} | " + " | ".join(_fmt(getter(s)) for s in systems) + " |")

    row("n_params", lambda s: s["n_params"])
    row("n_funcs (MGE columns)", lambda s: s["n_funcs"])
    row("edge-zeroed pixels", lambda s: s["n_edge_zeroed"])
    row("cond(F+lH)", lambda s: (s.get("spectrum_full") or {}).get("cond"))
    row("PDIP iterations", lambda s: s["pdip_iterations"])
    for tol in ("1e-08", "1e-06", "0.0001"):
        row(f"n(A*) tol {tol}", lambda s, t=tol: s["anatomy"]["active_set_sizes"][t])
    row("n(N) = #(x_unc < 0)", lambda s: s["anatomy"]["n_negative_unconstrained"])
    row(
        "Jaccard(N, A* 1e-6)",
        lambda s: s["anatomy"]["overlap_negatives_vs_active_set"]["1e-06"]["jaccard"],
    )
    row("log_ev(PDIP)", lambda s: s["log_evidence_pdip"])
    row("log_ev(library, edge-zeroed)", lambda s: s["log_evidence_library_edge_zeroed"])
    row("d log_ev unconstrained - PDIP", lambda s: s["d_log_evidence_unconstrained"])

    scheme_keys = sorted({k for s in systems for k in s["schemes"]})
    for key in scheme_keys:
        lines.append(f"| **{key}** |" + " |" * len(names))
        row("  certified", lambda s, k=key: (s["schemes"].get(k) or {}).get("certified"))
        row(
            "  passes to certification",
            lambda s, k=key: (s["schemes"].get(k) or {}).get("passes_to_certification"),
        )
        row(
            "  factorisations",
            lambda s, k=key: (s["schemes"].get(k) or {}).get("n_factorisations"),
        )
        row(
            "  max dev / max x_ref",
            lambda s, k=key: (s["schemes"].get(k) or {}).get("max_abs_dev_vs_reference_rel"),
        )
        row(
            "  d log_ev final",
            lambda s, k=key: (s["schemes"].get(k) or {}).get("d_log_evidence_final"),
        )
        row(
            "  Jaccard(Z_final, A* 1e-6)",
            lambda s, k=key: (s["schemes"].get(k) or {}).get("final_fixed_vs_active_set_jaccard"),
        )
        for budget in BUDGET_PASSES:
            row(
                f"  d log_ev at {budget} pass(es) [raw]",
                lambda s, k=key, b=budget: (
                    ((s["schemes"].get(k) or {}).get("fixed_budget") or {}).get(str(b)) or {}
                ).get("d_log_evidence_raw"),
            )
            row(
                f"  d log_ev at {budget} pass(es) [clipped]",
                lambda s, k=key, b=budget: (
                    ((s["schemes"].get(k) or {}).get("fixed_budget") or {}).get(str(b)) or {}
                ).get("d_log_evidence_clipped"),
            )

    lines.append("| **masked JAX (edge-zeroed, free_all)** |" + " |" * len(names))
    row("  pass budget", lambda s: s["masked_jax"]["n_passes"])
    row("  certified at pass", lambda s: s["masked_jax"]["certified_at_pass"])
    row(
        "  max dev vs library / max x",
        lambda s: s["masked_jax"]["max_abs_dev_vs_numpy_reference_rel"],
    )
    row("  d log_ev", lambda s: s["masked_jax"]["d_log_evidence"])

    lines += [
        "",
        "Per-pass fixed-set history, `edge_zeroed__free_all` (pass: |Z|, primal viol., dual viol.):",
        "",
    ]
    for system in systems:
        entry = system["schemes"].get("edge_zeroed__free_all")
        if not entry:
            continue
        history = ", ".join(
            f"{h['pass']}:({h['n_fixed']},{h['n_primal_violations']},{h['n_dual_violations']})"
            for h in entry["history"][:20]
        )
        tail = " ..." if len(entry["history"]) > 20 else ""
        lines.append(f"- `{system['system']}` — {history}{tail}")

    cross = next(
        (s for s in payload["systems"] if s.get("system") == "_cross_draw_active_set_jaccard"),
        None,
    )
    if cross:
        lines += ["", "Cross-draw Jaccard of A* (tol 1e-6):", "", "| pair | Jaccard |", "|---|---|"]
        for pair, value in cross["jaccard"].items():
            # The JSON key joins the two draws with "|", which would close the
            # markdown cell early.
            lines.append(f"| {pair.replace('|', ' vs ')} | {value:.3f} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def jaccard_indices(a, b) -> float:
    set_a, set_b = set(a), set(b)
    union = len(set_a | set_b)
    return (len(set_a & set_b) / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh", choices=("rectangular", "delaunay"), default="rectangular")
    parser.add_argument("--source-pixels", type=int, default=None)
    parser.add_argument("--instrument", default="hst")
    parser.add_argument("--max-passes", type=int, default=40)
    parser.add_argument("--tau-rel", type=float, default=ass.TAU_REL_DEFAULT)
    parser.add_argument(
        "--draws", action="store_true", help="also run the three +-1 %% mass perturbations on S3"
    )
    parser.add_argument(
        "--with-s1", action="store_true", help="also run S1 (subtract the TRUE simulator Sersic)"
    )
    parser.add_argument("--no-spectrum", action="store_true", help="skip the eigenvalue spectra")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    shared = build_shared(args.mesh, args.source_pixels, instrument=args.instrument)
    tag = args.tag or f"{args.mesh}_{shared['n_source_pixels']}"
    log(f"setup done: mesh={args.mesh} n_source_pixels={shared['n_source_pixels']}")

    instance = shared["instance"]
    source = instance.galaxies.source
    measure_kwargs = dict(
        max_passes=args.max_passes,
        tau_rel=args.tau_rel,
        with_spectrum=not args.no_spectrum,
    )

    systems: list[dict] = []

    # --- S0: the control, MGE-60 linear lens light + source mapper ----------
    fit_s0 = fit_from(shared, al.Tracer(galaxies=[lens_galaxy(instance), source]))
    system_s0 = ass.linear_system_from(fit_s0, shared["dataset"], name="S0_current_mge60")
    result_s0 = measure_system("S0_current_mge60", system_s0, **measure_kwargs)
    result_s0["description"] = "MGE-60 linear lens light + source mapper (the cell's model)"
    result_s0["draw"] = "fiducial"
    systems.append(result_s0)
    log(f"S0 done ({result_s0['wall_s']:.0f}s) pdip={result_s0['pdip_iterations']}")

    # --- S1 (optional): subtract the TRUE simulator Sersic -------------------
    if args.with_s1:
        truth_light = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(**TRUE_LENS_BULGE))
        blurred = al.Galaxies(galaxies=[truth_light]).blurred_image_2d_from(
            grid=shared["dataset"].grids.lp,
            psf=shared["dataset"].psf,
            blurring_grid=shared["dataset"].grids.blurring,
            xp=np,
        )
        dataset_s1 = shared["dataset"].__class__(
            data=shared["dataset"].data - blurred,
            noise_map=shared["dataset"].noise_map,
            psf=shared["dataset"].psf,
            over_sample_size_lp=shared["dataset"].over_sample_size_lp,
            over_sample_size_pixelization=shared["dataset"].over_sample_size_pixelization,
        )
        tracer_source_only = al.Tracer(galaxies=[lens_galaxy(instance, with_light=False), source])
        system_s1 = ass.linear_system_from(
            fit_from(shared, tracer_source_only, dataset=dataset_s1),
            dataset_s1,
            name="S1_clean_true_sersic",
        )
        result_s1 = measure_system("S1_clean_true_sersic", system_s1, **measure_kwargs)
        result_s1["description"] = "data - blurred(TRUE simulator Sersic); source mapper only"
        result_s1["draw"] = "fiducial"
        systems.append(result_s1)
        log(f"S1 done ({result_s1['wall_s']:.0f}s) pdip={result_s1['pdip_iterations']}")

    # --- S3: the MGE converted to regular at the S0-solved intensities -------
    system_s3 = ass.fixed_light_system_from(
        fit_s0, shared["dataset"], name="S3_mge_converted_to_regular"
    )
    result_s3 = measure_system("S3_mge_converted_to_regular", system_s3, **measure_kwargs)
    result_s3["description"] = (
        "data - blurred(MGE-60 basis at the S0-solved intensities, via "
        "fit.tracer_linear_light_profiles_to_light_profiles); source mapper only"
    )
    result_s3["draw"] = "fiducial"
    systems.append(result_s3)
    log(f"S3 done ({result_s3['wall_s']:.0f}s) pdip={result_s3['pdip_iterations']}")

    # --- the mass-model draws ------------------------------------------------
    draws: list[dict] = []
    if args.draws:
        seed_mask = np.zeros(result_s3["n_params"], dtype=bool)
        seed_mask[np.asarray(result_s3["active_set_1e-6_indices"], dtype=int)] = True

        draw_defs = [
            ("er_plus_1pct", dict(einstein_scale=1.01), "einstein_radius x 1.01"),
            ("er_minus_1pct", dict(einstein_scale=0.99), "einstein_radius x 0.99"),
            ("ell0_plus_0.01", dict(ell_delta=0.01), "ell_comps_0 + 0.01"),
        ]
        for draw_name, perturbation, description in draw_defs:
            fit_draw = fit_from(
                shared,
                al.Tracer(galaxies=[lens_galaxy(instance, **perturbation), source]),
            )
            system_draw = ass.fixed_light_system_from(
                fit_draw,
                shared["dataset"],
                name=f"S3_mge_converted_to_regular__{draw_name}",
            )
            result_draw = measure_system(
                f"S3_mge_converted_to_regular__{draw_name}",
                system_draw,
                max_passes=args.max_passes,
                tau_rel=args.tau_rel,
                with_spectrum=False,
                warm_start_mask=seed_mask,
            )
            result_draw["description"] = f"S3 with mass perturbation: {description}"
            result_draw["draw"] = draw_name
            result_draw["perturbation"] = description
            systems.append(result_draw)
            draws.append(result_draw)
            log(f"draw {draw_name} done ({result_draw['wall_s']:.0f}s)")

        cross = {}
        for a in [result_s3] + draws:
            for b in [result_s3] + draws:
                if a["draw"] < b["draw"]:
                    cross[f"{a['draw']}|{b['draw']}"] = jaccard_indices(
                        a["active_set_1e-6_indices"], b["active_set_1e-6_indices"]
                    )
        systems.append({"system": "_cross_draw_active_set_jaccard", "jaccard": cross})

    for entry in systems:
        entry.pop("active_set_1e-6_indices", None)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "probe": "fixed_light_probe_cpu",
        "artifact": f"{tag}.json",
        "mesh": args.mesh,
        "source_pixels": shared["n_source_pixels"],
        "source_pixels_requested": args.source_pixels,
        "instrument": shared["instrument"],
        "autolens_version": al.__version__,
        "regularization": shared["reg_provenance"],
        "max_passes": args.max_passes,
        "tau_rel": args.tau_rel,
        "dense_only": True,
        "systems": systems,
    }

    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    (out_dir / f"{tag}.md").write_text(render_markdown(payload))
    log(f"wrote {out_dir / f'{tag}.json'} and {out_dir / f'{tag}.md'}")


if __name__ == "__main__":
    main()
