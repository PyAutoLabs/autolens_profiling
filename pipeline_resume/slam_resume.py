"""
SLaM pipeline resume-overhead profiler.

A user commonly sets a SLaM pipeline running, gets part way, and later resumes
it. Every completed stage then skips its non-linear search, but the resume
still pays for: library imports, dataset setup, per-stage completed-fit loading
(samples summary JSON, full samples CSV, result rebuild), the inter-stage
science that is re-run from each loaded result (adapt-image construction,
position-likelihood solving, model composition — each of which can trigger a
fresh JAX JIT compile), and any re-visualization. This script measures where
that time goes.

It runs the **full 5-stage SLaM chain** mirrored from
``autolens_workspace/scripts/guides/modeling/slam_start_here.py``:

    source_lp[1] -> source_pix[1] -> source_pix[2] -> light[1] -> mass_total[1]

The first invocation is the **cold** run (the searches actually sample); every
later invocation of the same command is a **resume** run (all stages complete,
so only the resume overhead is paid). Each invocation appends one run record to
the versioned summary JSON; once a cold and at least one resume run exist, a
comparison PNG is rendered.

Usage (from the ``autolens_profiling`` repo root)::

    python3 pipeline_resume/slam_resume.py                     # cold, then re-run to profile resume
    python3 pipeline_resume/slam_resume.py --fast              # scaled-down n_live harness run
    python3 pipeline_resume/slam_resume.py --reset             # delete pipeline output, force cold
    python3 pipeline_resume/slam_resume.py --instrument euclid

Timing decomposition:

- Script-level spans wrap each stage's ``search.fit`` and each inter-stage
  block (``adapt_images``, ``positions``, ``model_compose``).
- Library-level timing wrappers (installed at runtime, no library edits) time
  ``NonLinearSearch.result_via_completed_fit`` and the ``DirectoryPaths``
  samples loads inside it, so a resume run's ``search_fit`` span decomposes
  into completed-fit total / samples-summary load / full-samples load, with
  the remainder attributable to pre/post-fit output handling.

Results land at ``results/pipeline_resume/slam_resume_summary_<instrument>_
v<version>[_fast].{json,png}`` (versioned-summary shape, ``results/README.md``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_T0 = time.perf_counter()

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# jax_wrapper must import before autofit/autolens (it sets the JAX env).
# isort: off
from autoconf import jax_wrapper  # noqa: F401, E402
from autoconf.test_mode import test_mode_level, with_test_mode_segment  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

# isort: on

_IMPORT_S = time.perf_counter() - _T0

from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    device_info_dict,
    parse_profile_cli,
    resolve_output_paths,
)
from instruments.imaging import INSTRUMENTS  # noqa: E402

STAGE_NAMES = ["source_lp", "source_pix_1", "source_pix_2", "light_lp", "mass_total"]

# n_live per stage, mirroring slam_start_here.py; --fast scales these down so a
# full harness-validation chain completes quickly (the resume overhead being
# profiled is dominated by loading/compile costs, not sampling quality).
_N_LIVE = {
    "source_lp": 200,
    "source_pix_1": 150,
    "source_pix_2": 75,
    "light_lp": 150,
    "mass_total": 150,
}
_N_LIVE_FAST = {name: 50 for name in STAGE_NAMES}

# n_batch caps the JAX vmap width and with it peak memory. slam_start_here.py
# passes 50 (source_lp) / 20 (later stages); the Nautilus default of 100 OOMs
# a 16GB machine on the source_lp MGE model.
_N_BATCH = {
    "source_lp": 50,
    "source_pix_1": 20,
    "source_pix_2": 20,
    "light_lp": 20,
    "mass_total": 20,
}
# 4 not 10: n_batch is NOT a Nautilus identifier field, so this can be tuned
# to the machine without invalidating completed stages; the profiling laptop
# shares 16GB across sessions and the stage-2 inversion OOMs at batch 10.
_N_BATCH_FAST = {name: 4 for name in STAGE_NAMES}

_MESH_PIXELS_YX = 28  # slam_start_here.py fiducial
_MESH_PIXELS_YX_FAST = 20


# -----------------------------------------------------------------------------
# Timing spans
# -----------------------------------------------------------------------------


class Spans:
    """Named wall-clock spans, summed per name across a single process run."""

    def __init__(self):
        self.seconds: dict[str, float] = {}

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed


SPANS = Spans()

# Set before each stage runs so the library-level timing wrappers below can
# attribute their measurements to the stage in flight.
_CURRENT_STAGE = ["startup"]


def _install_resume_timers():
    """Wrap the completed-fit resume path with timing spans (runtime only —
    no library source is modified). ``search_fit`` minus ``completed_fit_total``
    is the pre/post-fit output overhead; ``completed_fit_total`` minus the two
    load spans is result rebuild + optional re-visualization."""
    from autofit.non_linear.paths.directory import DirectoryPaths
    from autofit.non_linear.search import abstract_search

    def _wrap(cls, attr: str, label: str):
        orig = getattr(cls, attr)

        def timed(self, *args, **kwargs):
            with SPANS.span(f"{_CURRENT_STAGE[0]}/{label}"):
                return orig(self, *args, **kwargs)

        timed.__name__ = orig.__name__
        timed.__doc__ = orig.__doc__
        setattr(cls, attr, timed)

    _wrap(
        abstract_search.NonLinearSearch,
        "result_via_completed_fit",
        "completed_fit_total",
    )
    _wrap(DirectoryPaths, "load_samples_summary", "samples_summary_load")
    _wrap(DirectoryPaths, "load_samples", "samples_load")


# -----------------------------------------------------------------------------
# The 5 SLaM stages (mirrored from slam_start_here.py, instrumented)
# -----------------------------------------------------------------------------


def source_lp(settings_search, dataset, mask_radius, n_live, n_batch) -> af.Result:
    _CURRENT_STAGE[0] = "source_lp"

    with SPANS.span("source_lp/model_compose"):
        analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

        lens_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
        )
        source_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=0.5,
                    bulge=lens_bulge,
                    disk=None,
                    mass=af.Model(al.mp.Isothermal),
                    shear=af.Model(al.mp.ExternalShear),
                ),
                source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
            ),
        )

        search = af.Nautilus(
            name="source_lp[1]", **settings_search.search_dict, n_live=n_live, n_batch=n_batch
        )

    with SPANS.span("source_lp/search_fit"):
        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_1(settings_search, dataset, source_lp_result, mesh_shape, n_live, n_batch):
    _CURRENT_STAGE[0] = "source_pix_1"

    with SPANS.span("source_pix_1/adapt_images"):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(result=source_lp_result)
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    with SPANS.span("source_pix_1/positions"):
        positions_likelihood = source_lp_result.positions_likelihood_from(
            factor=3.0, minimum_threshold=0.2
        )

    with SPANS.span("source_pix_1/model_compose"):
        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[positions_likelihood],
        )

        mass = al.util.chaining.mass_from(
            mass=source_lp_result.model.galaxies.lens.mass,
            mass_result=source_lp_result.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=mass,
                    shear=source_lp_result.model.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=mesh_shape),
                        regularization=al.reg.Adapt,
                    ),
                ),
            ),
        )

        search = af.Nautilus(
            name="source_pix[1]", **settings_search.search_dict, n_live=n_live, n_batch=n_batch
        )

    with SPANS.span("source_pix_1/search_fit"):
        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def source_pix_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh_shape,
    n_live,
    n_batch,
):
    _CURRENT_STAGE[0] = "source_pix_2"

    with SPANS.span("source_pix_2/adapt_images"):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_pix_result_1
        )
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    with SPANS.span("source_pix_2/model_compose"):
        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            use_jax=True,
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.lens.redshift,
                    bulge=source_lp_result.instance.galaxies.lens.bulge,
                    disk=source_lp_result.instance.galaxies.lens.disk,
                    mass=source_pix_result_1.instance.galaxies.lens.mass,
                    shear=source_pix_result_1.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=source_lp_result.instance.galaxies.source.redshift,
                    pixelization=af.Model(
                        al.Pixelization,
                        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
                        regularization=al.reg.Adapt,
                    ),
                ),
            ),
        )

        search = af.Nautilus(
            name="source_pix[2]", **settings_search.search_dict, n_live=n_live, n_batch=n_batch
        )

    with SPANS.span("source_pix_2/search_fit"):
        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def light_lp(
    settings_search,
    dataset,
    mask_radius,
    source_result_for_lens,
    source_result_for_source,
    n_live,
    n_batch,
):
    _CURRENT_STAGE[0] = "light_lp"

    with SPANS.span("light_lp/adapt_images"):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_result_for_lens
        )
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    with SPANS.span("light_lp/model_compose"):
        analysis = al.AnalysisImaging(dataset=dataset, adapt_images=adapt_images)

        lens_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
        )

        source = al.util.chaining.source_custom_model_from(
            result=source_result_for_source, source_is_model=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=None,
                    mass=source_result_for_lens.instance.galaxies.lens.mass,
                    shear=source_result_for_lens.instance.galaxies.lens.shear,
                ),
                source=source,
            ),
        )

        search = af.Nautilus(
            name="light[1]", **settings_search.search_dict, n_live=n_live, n_batch=n_batch
        )

    with SPANS.span("light_lp/search_fit"):
        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def mass_total(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    n_live,
    n_batch,
):
    _CURRENT_STAGE[0] = "mass_total"

    with SPANS.span("mass_total/adapt_images"):
        galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
            result=source_result_for_lens
        )
        adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    with SPANS.span("mass_total/positions"):
        positions_likelihood = source_result_for_source.positions_likelihood_from(
            factor=3.0, minimum_threshold=0.2
        )

    with SPANS.span("mass_total/model_compose"):
        analysis = al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            positions_likelihood_list=[positions_likelihood],
        )

        mass = al.util.chaining.mass_from(
            mass=af.Model(al.mp.PowerLaw),
            mass_result=source_result_for_lens.model.galaxies.lens.mass,
            unfix_mass_centre=True,
        )

        source = al.util.chaining.source_from(result=source_result_for_source)

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                    bulge=light_result.instance.galaxies.lens.bulge,
                    disk=light_result.instance.galaxies.lens.disk,
                    mass=mass,
                    shear=source_result_for_lens.model.galaxies.lens.shear,
                ),
                source=source,
            ),
        )

        search = af.Nautilus(
            name="mass_total[1]", **settings_search.search_dict, n_live=n_live, n_batch=n_batch
        )

    with SPANS.span("mass_total/search_fit"):
        return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


# -----------------------------------------------------------------------------
# Dataset setup (mirrors slam_start_here.py / searches/_setup.py conventions)
# -----------------------------------------------------------------------------


def build_dataset(instrument: str):
    cfg = INSTRUMENTS[instrument]
    dataset_path = Path("dataset") / "imaging" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="imaging",
        instrument=instrument,
        workspace_root=_WORKSPACE_ROOT,
    )
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=cfg["pixel_scale"],
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=cfg["mask_radius"],
    )
    dataset = dataset.apply_mask(mask=mask)
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
    return dataset, float(cfg["mask_radius"])


# -----------------------------------------------------------------------------
# Result JSON + PNG
# -----------------------------------------------------------------------------


def _stage_component(spans: dict[str, float], stage: str, component: str) -> float:
    return round(spans.get(f"{stage}/{component}", 0.0), 3)


def write_summary(json_path: Path, run_record: dict, instrument: str, fast: bool):
    if json_path.exists():
        data = json.loads(json_path.read_text())
    else:
        data = {
            "tier": "pipeline_resume",
            "instrument": instrument,
            "fast": fast,
            "al_version": al.__version__,
            "af_version": af.__version__,
            "stages": STAGE_NAMES,
            "runs": [],
        }
    data["device"] = device_info_dict()
    data["runs"].append(run_record)
    json_path.write_text(json.dumps(data, indent=2))
    return data


def render_png(png_path: Path, data: dict):
    """Render cold-vs-resume per-stage totals + the resume component breakdown.

    Skipped (with a note) until the JSON holds both a cold run and at least
    one resume run.
    """
    cold = next((r for r in data["runs"] if r["mode"] == "cold"), None)
    resume = next((r for r in reversed(data["runs"]) if r["mode"] == "resume"), None)
    if cold is None or resume is None:
        print(f"  PNG deferred: need a cold and a resume run (have {len(data['runs'])}).")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stages = data["stages"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — per-stage wall time, cold vs resume.
    y = np.arange(len(stages))
    cold_totals = [_stage_component(cold["spans"], s, "search_fit") for s in stages]
    resume_totals = [_stage_component(resume["spans"], s, "search_fit") for s in stages]
    ax0.barh(y - 0.2, cold_totals, height=0.4, label="cold (search runs)")
    ax0.barh(y + 0.2, resume_totals, height=0.4, label="resume (overhead only)")
    ax0.set_yticks(y, stages)
    ax0.invert_yaxis()
    ax0.set_xlabel("search_fit wall time [s]")
    ax0.set_xscale("log")
    ax0.set_title(
        f"SLaM stage wall time — cold vs resume\n[{data['instrument']}, v{data['al_version']}]"
    )
    ax0.legend()

    # Panel 2 — resume run decomposition per stage (stacked).
    components = [
        ("samples_summary_load", "samples summary load"),
        ("samples_load", "full samples load"),
        ("completed_fit_other", "result rebuild + output"),
        ("fit_wrapper_other", "pre/post-fit handling"),
        ("adapt_images", "adapt images"),
        ("positions", "positions likelihood"),
        ("model_compose", "model compose"),
    ]
    spans = resume["spans"]
    stacked = {}
    for stage in stages:
        completed_total = _stage_component(spans, stage, "completed_fit_total")
        loads = _stage_component(spans, stage, "samples_summary_load") + _stage_component(
            spans, stage, "samples_load"
        )
        fit_total = _stage_component(spans, stage, "search_fit")
        stacked[stage] = {
            "samples_summary_load": _stage_component(spans, stage, "samples_summary_load"),
            "samples_load": _stage_component(spans, stage, "samples_load"),
            "completed_fit_other": max(completed_total - loads, 0.0),
            "fit_wrapper_other": max(fit_total - completed_total, 0.0),
            "adapt_images": _stage_component(spans, stage, "adapt_images"),
            "positions": _stage_component(spans, stage, "positions"),
            "model_compose": _stage_component(spans, stage, "model_compose"),
        }
    left = np.zeros(len(stages))
    for key, label in components:
        vals = np.array([stacked[s][key] for s in stages])
        ax1.barh(y, vals, left=left, height=0.6, label=label)
        left += vals
    ax1.set_yticks(y, stages)
    ax1.invert_yaxis()
    ax1.set_xlabel("resume wall time [s]")
    ax1.set_title(
        f"Resume overhead decomposition per stage\n"
        f"(imports {resume['import_s']:.1f}s + dataset {resume['dataset_setup_s']:.1f}s "
        f"paid once per process)"
    )
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  PNG written: {png_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    cli = parse_profile_cli(default_config_name=None)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="Scaled-down n_live harness run.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this pipeline's search output first, forcing a cold run.",
    )
    extra, _unknown = parser.parse_known_args()

    instrument = cli.instrument or "hst"
    fast = extra.fast
    n_live = _N_LIVE_FAST if fast else _N_LIVE
    n_batch = _N_BATCH_FAST if fast else _N_BATCH
    mesh_pixels_yx = _MESH_PIXELS_YX_FAST if fast else _MESH_PIXELS_YX
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    # Test-mode runs get their own artifact (and the versioned-artifact regex
    # in scripts/build_readme.py ignores suffixed files), so instant
    # PYAUTO_TEST_MODE chains never mix with real measurement records.
    suffix = ("_fast" if fast else "") + ("_testmode" if test_mode_level() > 0 else "")
    unique_tag = f"{instrument}{suffix}"
    # Under PYAUTO_TEST_MODE the search outputs are namespaced into
    # output/test_mode/ (PyAutoFit's _test_mode_segment); the reset/mode
    # detection must look at the same tree. The resume invocation must then
    # ALSO run with PYAUTO_TEST_MODE set, or it will see a cold output dir.
    pipeline_output_dir = with_test_mode_segment(Path("output")) / "pipeline_resume" / unique_tag

    if extra.reset and pipeline_output_dir.exists():
        print(f"  --reset: removing {pipeline_output_dir}")
        shutil.rmtree(pipeline_output_dir)

    # A stage is complete when its search output carries a `.completed` marker.
    # 0 complete = cold; all complete = resume; anything else is a partial run
    # (real sampling in some stages), excluded from cold/resume comparisons.
    n_complete = sum(
        1
        for stage_glob in (
            "source_lp[1]",
            "source_pix[1]",
            "source_pix[2]",
            "light[1]",
            "mass_total[1]",
        )
        for _ in (pipeline_output_dir / stage_glob).glob("*/.completed")
    )
    if n_complete == 0:
        mode = "cold"
    elif n_complete >= len(STAGE_NAMES):
        mode = "resume"
    else:
        mode = "partial"
    print(f"\n--- SLaM resume profiler [{unique_tag}] — {mode} run ---")

    _install_resume_timers()

    with SPANS.span("dataset_setup"):
        dataset, mask_radius = build_dataset(instrument)

    settings_search = af.SettingsSearch(
        path_prefix=Path("pipeline_resume"),
        unique_tag=unique_tag,
        info=None,
        session=None,
    )

    t_pipeline = time.perf_counter()

    source_lp_result = source_lp(
        settings_search, dataset, mask_radius, n_live["source_lp"], n_batch["source_lp"]
    )
    source_pix_result_1 = source_pix_1(
        settings_search,
        dataset,
        source_lp_result,
        mesh_shape,
        n_live["source_pix_1"],
        n_batch["source_pix_1"],
    )
    source_pix_result_2 = source_pix_2(
        settings_search,
        dataset,
        source_lp_result,
        source_pix_result_1,
        mesh_shape,
        n_live["source_pix_2"],
        n_batch["source_pix_2"],
    )
    light_result = light_lp(
        settings_search,
        dataset,
        mask_radius,
        source_pix_result_1,
        source_pix_result_2,
        n_live["light_lp"],
        n_batch["light_lp"],
    )
    mass_total(
        settings_search,
        dataset,
        source_pix_result_1,
        source_pix_result_2,
        light_result,
        n_live["mass_total"],
        n_batch["mass_total"],
    )

    pipeline_s = time.perf_counter() - t_pipeline
    total_s = time.perf_counter() - _T0

    run_record = {
        "mode": mode,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_s": round(total_s, 3),
        "import_s": round(_IMPORT_S, 3),
        "dataset_setup_s": round(SPANS.seconds.get("dataset_setup", 0.0), 3),
        "pipeline_s": round(pipeline_s, 3),
        "n_live": n_live,
        "test_mode": test_mode_level(),
        "spans": {k: round(v, 3) for k, v in sorted(SPANS.seconds.items())},
    }

    json_path, png_path = resolve_output_paths(
        cli,
        default_dir=_WORKSPACE_ROOT / "results" / "pipeline_resume",
        default_basename=f"slam_resume_summary_{instrument}_v{al.__version__}{suffix}",
    )
    data = write_summary(json_path, run_record, instrument, fast)
    print(f"  JSON updated: {json_path} ({mode} run, total {total_s:.1f}s)")

    print(f"\n  {mode} run breakdown:")
    print(f"    imports        {_IMPORT_S:8.1f}s")
    print(f"    dataset setup  {run_record['dataset_setup_s']:8.1f}s")
    for stage in STAGE_NAMES:
        fit_s = run_record["spans"].get(f"{stage}/search_fit", 0.0)
        inter = sum(
            run_record["spans"].get(f"{stage}/{c}", 0.0)
            for c in ("adapt_images", "positions", "model_compose")
        )
        print(f"    {stage:<14} fit {fit_s:8.1f}s   inter-stage {inter:6.1f}s")

    render_png(png_path, data)


if __name__ == "__main__":
    main()
