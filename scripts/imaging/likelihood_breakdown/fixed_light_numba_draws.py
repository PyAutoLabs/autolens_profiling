"""CPU memo robustness on the frozen 41-model HST graded draw set (#278).

Run from the profiling root. The lens light is re-solved and subtracted for each
model before timing. Headline samples call the unchanged production Analysis
likelihood, once per model per traversal; diagnostics run separately. Every
traversal begins with an empty NNLS memo, including after compilation warm-up.
This is a sequence robustness experiment, not a repeated-fiducial benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "misc")]

from _production_config import pin_thread_env  # noqa: E402

THREAD_ENV = pin_thread_env(1)
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
from likelihood_breakdown import fixed_light_numba_draws_steps as steps  # noqa: E402

MANIFEST = (
    "results/breakdown/imaging/fixed_light_draws_delaunay_hpc_a100_fp64_fixed_light_draws.json"
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(samples):
    values = np.asarray(samples, dtype=float)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "n": len(values),
    }


def prepare(rows, source_pixels):
    """Rebuild the established graded-draw model using NumPy and CPU operators."""
    import autofit as af
    import autolens as al
    from likelihood_breakdown import fixed_light_draws_steps as draws
    from likelihood_breakdown import fixed_light_system
    from simulators.imaging import INSTRUMENTS

    from _adapt_image_util import adapt_image_for_dataset
    from _profile_cli import auto_simulate_if_missing

    dataset_path = ROOT / "dataset" / "imaging" / "hst"
    auto_simulate_if_missing(
        dataset_path, dataset_type="imaging", instrument="hst", workspace_root=ROOT
    )
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        pixel_scales=INSTRUMENTS["hst"]["pixel_scale"],
    )
    dataset = dataset.apply_mask(
        al.Mask2D.circular(
            shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
        )
    ).apply_over_sampling(over_sample_size_lp=4, over_sample_size_pixelization=1)
    oversampling = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=oversampling, over_sample_size_pixelization=1
    )
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    mesh_grid = al.image_mesh.Hilbert(
        pixels=source_pixels, weight_power=1.0, weight_floor=0.0
    ).image_plane_mesh_grid_from(mask=dataset.mask, adapt_data=adapt_image)
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
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=al.model_util.mge_model_from(
            mask_radius=3.5, total_gaussians=60, centre_prior_is_uniform=True
        ),
        mass=mass,
    )
    field = af.Model(al.MassField, redshift=0.5, shear=shear)

    regularization = al.reg.AdaptSplit(
        inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1
    )
    model = af.Collection(
        galaxies=af.Collection(
            lens=lens,
            source=af.Model(
                al.Galaxy,
                redshift=1.0,
                pixelization=al.Pixelization(
                    mesh=al.mesh.Delaunay(pixels=source_pixels, zeroed_pixels=0),
                    regularization=regularization,
                ),
            ),
        ),
        fields=field,
    )
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    source = instance.galaxies.source
    base_mass = draws.fiducial_mass_values(instance)
    adapt = al.AdaptImages(
        galaxy_image_dict={source: adapt_image},
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        galaxy_image_plane_mesh_grid_dict={source: mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": mesh_grid},
    )
    settings = al.Settings(use_border_relocator=True, nnls_warm_start_memo=True)

    def fit_s0(offsets, family):
        return al.FitImaging(
            dataset=dataset,
            tracer=al.Tracer(
                galaxies=[
                    draws.lens_galaxy_from(
                        instance, draws.mass_from(base_mass, offsets, mass_cls=family)
                    ),
                    source,
                ],
                fields=[instance.fields],
            ),
            adapt_images=adapt,
            settings=settings,
            xp=np,
        )

    with steps.memo_scope(False):
        references = {
            family: float(fit_s0({}, family).figure_of_merit)
            for family in ("Isothermal", "PowerLaw")
        }
    promotion_rel = abs(references["Isothermal"] - references["PowerLaw"]) / max(
        abs(references["Isothermal"]), 1e-300
    )
    promotion = {
        "references": references,
        "relative_difference": promotion_rel,
        "tolerance": 1e-4,
        "passed": bool(np.isfinite(promotion_rel) and promotion_rel <= 1e-4),
    }
    cells, metadata = [], []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        with steps.memo_scope(False):
            fit = fit_s0(row["offsets"], row["mass_cls"])
            evidence = float(fit.figure_of_merit)
            system = fixed_light_system.fixed_light_system_from(
                fit,
                dataset,
                adapt_images=adapt,
                settings=settings,
                scaling="numpy",
                sparse_operator="rebake_cpu",
            )
        stripped = af.ModelInstance()
        stripped.galaxies = af.ModelInstance()
        stripped.galaxies.lens, stripped.galaxies.source = system.source_only_tracer.galaxies
        analysis = al.AnalysisImaging(
            dataset=system.dataset, adapt_images=adapt, settings=settings, use_jax=False
        )
        cells.append((analysis, stripped))
        metadata.append(
            {
                "manifest_draw": row,
                "name": row["name"],
                "kind": row["kind"],
                "s0_evidence_cpu": evidence,
                "achieved_d_log_l_cpu": evidence - references[row["mass_cls"]],
                "subtracted_light_flux_cpu": system.subtracted_light_flux,
                "n_params_cpu": system.n_params,
                "preparation_seconds": time.perf_counter() - started,
            }
        )
        log(f"prepared {index + 1}/{len(rows)} {row['name']}")
        del fit, system
        gc.collect()
    return cells, metadata, promotion


def diagnostic_traversal(cells, indices, enabled):
    from autoarray.inversion.inversion.imaging_numba.sparse import InversionImagingSparseNumba

    results = {}
    with steps.memo_scope(enabled), steps.observe_solver() as calls:
        for index in indices:
            analysis, instance = cells[index]
            before = len(calls)
            fit = analysis.fit_from(instance=instance)
            evidence = float(fit.figure_of_merit)
            inversion = fit.inversion
            if not isinstance(inversion, InversionImagingSparseNumba):
                raise AssertionError(f"Wrong inversion dispatch: {type(inversion).__name__}")
            current = calls[before:]
            if len(current) != 1:
                raise AssertionError(f"Expected one source-only solve, observed {len(current)}")
            results[index] = {
                "evidence": evidence,
                "reconstruction": np.asarray(inversion.reconstruction).copy(),
                "passive_set": current[0]["passive_set"],
                "solver": current[0],
            }
    return results


def plot_result(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for order_name, order in result["orders"].items():
        indices = order["indices"]
        warm = [order["per_draw"][str(i)]["warm"]["median_ms"] for i in indices]
        cold = [order["per_draw"][str(i)]["cold"]["median_ms"] for i in indices]
        axes[0].plot(np.arange(len(indices)), np.asarray(cold) / warm, ".-", label=order_name)
        axes[1].scatter(
            [-result["draws"][i]["achieved_d_log_l_cpu"] for i in indices],
            np.asarray(cold) / warm,
            label=order_name,
        )
    for ax in axes:
        ax.axhline(1, color="black", linewidth=0.8)
        ax.set_ylabel("Cold / warm median likelihood time")
        ax.legend()
    axes[0].set_xlabel("Position in traversal (first solve is cold)")
    axes[1].set_xlabel("Loss of S0 evidence relative to own mass-family fiducial (nats)")
    axes[1].set_xscale("symlog", linthresh=10)
    fig.suptitle(f"CPU NNLS memo robustness — {result['status']}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--permutation-seed", type=int, default=278)
    parser.add_argument("--source-pixels", type=int, default=1500)
    parser.add_argument("--max-draws", type=int, default=41, help="Smoke only when less than 41")
    parser.add_argument("--config-name", default="local_cpu_fp64_fixed_light_numba_s5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/breakdown/imaging")
    args = parser.parse_args(argv)
    if args.n_repeats < 1 or not 1 <= args.max_draws <= 41 or args.source_pixels < 10:
        parser.error("positive repeats, 1..41 draws and >=10 source pixels required")
    rows, manifest = steps.load_manifest(args.manifest)
    rows = rows[: args.max_draws]
    started = time.perf_counter()
    cells, metadata, promotion = prepare(rows, args.source_pixels)
    import autoarray
    import autofit
    import autogalaxy
    import autolens
    import numba
    from threadpoolctl import threadpool_info

    observed_threads = threadpool_info()
    for pool in observed_threads:
        if "filepath" in pool:
            pool["filepath"] = Path(pool["filepath"]).name
    thread_ok = numba.get_num_threads() == 1 and all(
        pool["num_threads"] == 1 for pool in observed_threads
    )
    result = {
        "study": "fixed-light-numba-s5",
        "autolens_version": autolens.__version__,
        "device": "cpu",
        "machine": platform.node(),
        "instrument": "hst",
        "configuration": {
            "source_pixels": args.source_pixels,
            "mesh": "delaunay",
            "formalism": "sparse_numba",
            "precision": "fp64",
            "n_repeats": args.n_repeats,
            "n_draws": len(rows),
            "permutation_seed": args.permutation_seed,
            "thread_env": THREAD_ENV,
            "numba_threads": numba.get_num_threads(),
            "threadpools": observed_threads,
            "timing_scope": "unchanged AnalysisImaging.log_likelihood_function; use_jax=False",
            "memo_reset": "empty at each traversal boundary; no self-seeded timing",
            "lens_light": "S0 re-solved and subtracted per model before timing",
            "regularization": {"scheme": "adapt_split", "inner": 0.1, "outer": 10.0, "signal": 0.1},
        },
        "manifest": manifest,
        "slope_promotion": promotion,
        "draws": metadata,
        "provenance": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "repo_revision": revision(ROOT),
            "libraries": {
                module.__name__: {
                    "version": module.__version__,
                    "revision": revision(Path(module.__file__).parent.parent),
                }
                for module in (autoarray, autofit, autogalaxy, autolens)
            },
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(steps.__file__),
                    ROOT / "scripts/misc/likelihood_breakdown/fixed_light_draws_steps.py",
                    ROOT / "scripts/misc/likelihood_breakdown/fixed_light_system.py",
                    ROOT / "_adapt_image_util.py",
                    ROOT / "_production_config.py",
                )
            },
        },
        "dataset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "dataset/imaging/hst").glob("*.fits"))
        },
        "load_average_before": list(os.getloadavg()),
        "orders": {},
        "gates": {"single_thread": bool(thread_ok), "slope_promotion": promotion["passed"]},
    }

    def evaluate(index):
        analysis, instance = cells[index]
        return float(analysis.log_likelihood_function(instance=instance))

    # Compile both family code paths and initialize analysis datasets; discarded.
    with steps.memo_scope(False):
        for index in range(len(rows)):
            evaluate(index)
    orders = {
        "graded": list(range(len(rows))),
        "permuted": np.random.default_rng(args.permutation_seed).permutation(len(rows)).tolist(),
    }
    all_passed = True
    for order_name, indices in orders.items():
        log(f"diagnostics {order_name}")
        cold = diagnostic_traversal(cells, indices, False)
        warm = diagnostic_traversal(cells, indices, True)
        gates = {str(i): steps.compare_solutions(warm[i], cold[i]) for i in indices}
        samples = {"warm": [], "cold": []}
        for repeat in range(args.n_repeats):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                lane = "warm" if enabled else "cold"
                log(f"timing {order_name} repeat={repeat + 1} memo={lane}")
                samples[lane].append(steps.replay_sequence(evaluate, indices, enabled))
        per_draw = {}
        for i in indices:
            row = {
                "equivalence": gates[str(i)],
                "warm_solver": warm[i]["solver"],
                "cold_solver": cold[i]["solver"],
            }
            row["observer_agreement"] = {}
            for lane, observed in (("warm", warm), ("cold", cold)):
                entries = [
                    entry
                    for traversal in samples[lane]
                    for entry in traversal
                    if entry["index"] == i
                ]
                row[lane] = summarize([entry["elapsed_ms"] for entry in entries])
                row["observer_agreement"][lane] = all(
                    np.isfinite(entry["evidence"])
                    and abs(entry["evidence"] - observed[i]["evidence"])
                    <= 1e-9 * max(abs(observed[i]["evidence"]), 1e-300)
                    for entry in entries
                )
            all_passed &= gates[str(i)]["passed"] and all(row["observer_agreement"].values())
            per_draw[str(i)] = row
        sequence_totals = {
            lane: [sum(entry["elapsed_ms"] for entry in traversal) for traversal in samples[lane]]
            for lane in ("warm", "cold")
        }
        result["orders"][order_name] = {
            "indices": indices,
            "samples": samples,
            "per_draw": per_draw,
            "sequence_totals_ms": sequence_totals,
            "sequence_speedup": float(
                np.median(sequence_totals["cold"]) / np.median(sequence_totals["warm"])
            ),
            "warm_memo_hits": sum(warm[i]["solver"]["seed_source"] == "memo" for i in indices),
            "warm_invalidations": sum(warm[i]["solver"]["warm_start_fallback"] for i in indices),
            "warm_exception_retries": sum(
                len(warm[i]["solver"]["kernel_attempts"]) > 1 for i in indices
            ),
        }
    result["gates"]["numerical_equivalence"] = bool(all_passed)
    result["gates"]["complete_campaign_cell"] = (
        len(rows) == 41 and args.source_pixels == 1500 and args.n_repeats >= 5
    )
    result["status"] = "PASS" if all(result["gates"].values()) else "INCOMPLETE_OR_FAIL"
    result["wall_seconds"] = time.perf_counter() - started
    result["load_average_after"] = list(os.getloadavg())
    result["jax_imported_by_library"] = "jax" in sys.modules
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"fixed_light_numba_draws_delaunay_{args.config_name}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_result(result, stem.with_suffix(".png"))
    log(f"{result['status']}: {stem.name}")
    if not all_passed or not thread_ok or not promotion["passed"]:
        return 1
    return 0


if __name__ == "__main__" and os.environ.get("AUTOLENS_PROFILING_SMOKE") != "1":
    raise SystemExit(main())
