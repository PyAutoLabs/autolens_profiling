"""
JAX Profiling: Fixed Lens Light — CAPTURE real Nautilus proposal batches
=======================================================================

Certified-solver phase C1 (autolens_profiling#304), step 1 of 2.

Phase B (autolens_profiling#300, ``results/notes/certified_solver_policy_phase_b_2026_09.md``)
measured the library's certified solver under the production composition
``jax.jit(jax.vmap(fn))`` with ``certified_fallback="none"`` as the fastest
batched row, with **zero** uncertified lanes. Its lanes, though, were one seeded
draw family near the fiducial (``fixed_light_draws_steps.random_draws``). Whether
an uncertified-lane guard (phase C2) is worth building depends on how often a lane
fails its certificate on the batches a real search actually proposes. This cell
records those batches; ``fixed_light_trace.py --lanes captured`` replays them.

What it does
------------

1. Builds the phase-B cell's S3 system — the HST dataset, the fixed-light
   subtraction, the Delaunay (Hilbert, N=1500, AdaptSplit) or rectangular
   (39x39, Constant) source — with PART A of ``fixed_light_trace.py`` copied
   line for line (the house rule of this family: "if a sibling cell's model
   changes, this cell must change with it"). The replay does not trust the copy:
   it re-derives the S3 fingerprint recorded here (dataset sha256s, mesh, source
   pixels, regularization, border relocator, the eager S3 figure of merit) and
   refuses a file that does not match.
2. Fits that S3 system with ``af.Nautilus`` at the settings of the production
   ``source_pix[1]`` stage of
   ``euclid_strong_lens_modeling_pipeline/scripts/full_model.py``:
   ``n_live=150``, ``n_batch=20``, ``use_jax_vmap=True`` (the ``SettingsSearch``
   default), one core, every other Nautilus argument at its default. The
   likelihood is the library's own — ``AnalysisImaging(use_jax=True)`` with the
   packaged positive-only solver (library PDIP): the batches must be the ones
   production proposes, so nothing about the solver is changed here.
3. Records every batched parameter array that reaches ``Fitness.call_wrap`` with
   its call index, the figure of merit it returned, its wall time and the
   Nautilus phase it was proposed in (``likelihood_breakdown.nautilus_batches``:
   a script-scoped patch of ``Fitness.call_wrap`` and
   ``nautilus.Sampler.evaluate_likelihood``, restored on exit — no library is
   edited).

The model, and where it departs from ``source_pix[1]``
------------------------------------------------------

Free: the lens ``Isothermal`` (centre, ell_comps, einstein_radius) and the
``ExternalShear`` — 7 parameters, with the phase-B cell's Gaussian priors (the
same priors ``fixed_light_draws_steps.PRIOR_SIGMA`` scales the phase-B lanes
from). Fixed: the lens light (subtracted, S3) and the source pixelization.

``source_pix[1]`` also frees the regularization (``AdaptSplit``) and chains its
priors from ``source_lp``; this cell keeps the regularization FIXED at the cell's
coefficients, because the replay's lane trees carry the pixelization in their
static aux data (``fixed_light_trace._lane_tree``) and every lane must share it.
The captured batches therefore vary the mass and shear only — the same axes as
the phase-B lanes, now proposed by the sampler instead of drawn by a seed.
Nautilus is seeded (``--seed``, default 1) so a capture is reproducible;
production passes no seed. Quick updates are disabled (they only render the
current best fit and never change what Nautilus proposes), and the search's
post-fit ``perform_update`` is replaced by a stop: the samples and the per-sample
latent-variable pass it would build cost more than the sampling on the laptop and
record nothing.

Flags (beyond the shared ``_profile_cli`` set)
----------------------------------------------

``--mesh {delaunay,rectangular}`` (required) — the phase-B meshes.
``--n-like-max N`` (default 40000) — Nautilus's own likelihood cap. On the A100
this is roughly an hour of sampling (phase B: ~39 ms per Delaunay lane under
library PDIP ``jit(vmap)``, plus Nautilus's CPU-side bound training); a laptop
witness passes a few batches' worth.
``--capture-seconds S`` (default: no limit) — a hard wall-clock cap. The batch
that crosses it is recorded, then Nautilus is unwound and the file written.
``--n-live`` (default 150), ``--n-batch`` (default 20), ``--seed`` (default 1).

Output
------

``<output-dir>/nautilus_batches_<mesh>[_n<N>].npz`` and ``.json`` beside it
(default output dir ``results/breakdown/imaging``). The npz keys are
``nautilus_batches.SCHEMA_VERSION``'s: ``parameters`` ``(n_lanes, 7)`` physical
values in ``meta.parameter_paths`` order, ``figure_of_merit``, ``call_index``,
``lane_in_call``, and per call ``call_size``, ``call_phase``, ``call_n_bounds``,
``call_explored``, ``call_n_like_before``, ``call_wall_s``; ``meta_json`` repeats
the provenance JSON's ``meta``. The JSON adds library source revisions, device,
the Nautilus settings and the batch-shape census.

What may not be quoted from this file
-------------------------------------

- ``call_wall_s`` as a likelihood cost. It is the wall of ``call_wrap`` under a
  running sampler, including the first call's compile; the timed pass of
  ``fixed_light_trace.py`` is where milliseconds come from.
- Any rate at all. This cell only records batches; the certified rate is the
  replay's (``fixed_light_trace.py --lanes captured --captured-pass rate``).
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
    _sys.path.insert(0, str(_misc_dir))


import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autofit.jax import register_model as _register_model_pytrees

sys.path.insert(0, str(_profiling_root()))

import os as _smoke_os
import sys as _smoke_sys

from likelihood_breakdown import (  # noqa: E402
    active_set_steps,
    nautilus_batches,
    timing,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke): imports + module
# setup, before any argparse, so a smoke run needs no flags.
if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from simulators.imaging import INSTRUMENTS  # noqa: E402

from _production_config import observe_thread_env as _observe_thread_env  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    device_info_dict,
    machine_info_dict,
    parse_profile_cli,
    rect_mesh_classes,
)

_cli = parse_profile_cli()

#: ``source_pix[1]`` of euclid_strong_lens_modeling_pipeline/scripts/full_model.py.
PRODUCTION_N_LIVE = 150
PRODUCTION_N_BATCH = 20

_cell_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--mesh", choices=("rectangular", "delaunay"), required=True)
_cell_parser.add_argument("--source-pixels", type=int, default=None)
_cell_parser.add_argument("--n-like-max", type=int, default=40000)
_cell_parser.add_argument("--capture-seconds", type=float, default=None)
_cell_parser.add_argument("--n-live", type=int, default=PRODUCTION_N_LIVE)
_cell_parser.add_argument("--n-batch", type=int, default=PRODUCTION_N_BATCH)
_cell_parser.add_argument("--seed", type=int, default=1)
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
N_LIKE_MAX = int(_cell_args.n_like_max)
CAPTURE_SECONDS = _cell_args.capture_seconds
N_LIVE = int(_cell_args.n_live)
N_BATCH = int(_cell_args.n_batch)
SEED = int(_cell_args.seed)
for _flag, _value in (("--n-like-max", N_LIKE_MAX), ("--n-live", N_LIVE), ("--n-batch", N_BATCH)):
    if _value < 1:
        raise ValueError(f"{_flag} must be >= 1 (got {_value})")
if CAPTURE_SECONDS is not None and CAPTURE_SECONDS <= 0:
    raise ValueError(f"--capture-seconds must be > 0 (got {CAPTURE_SECONDS})")

SOURCE_PIXELS_REQUESTED = (
    _cell_args.source_pixels if _cell_args.source_pixels is not None else _cli.source_pixels
)

DATASET = "hst"
if _cli.instrument is not None and _cli.instrument != DATASET:
    raise ValueError(f"--instrument {_cli.instrument!r} is not this cell's dataset (HST only).")

timer = timing.Timer()

# ===================================================================
# PART A — Setup: fixed_light_trace.py PART A, copied line for line
# ===================================================================

print("=" * 70)
print(f"NAUTILUS BATCH CAPTURE — {MESH} — n_live {N_LIVE}, n_batch {N_BATCH}")
print("=" * 70)

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[DATASET]["pixel_scale"]
dataset_path = Path("dataset") / "imaging" / DATASET


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


auto_simulate_if_missing(
    dataset_path,
    dataset_type="imaging",
    instrument=DATASET,
    workspace_root=_workspace_root,
)

dataset_sha256 = {
    name: _sha256(dataset_path / name) for name in ("data.fits", "noise_map.fits", "psf.fits")
}
cell_source_sha256 = _sha256(Path(__file__).resolve())

with timer.section("dataset_load"):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )

with timer.section("mask_and_oversample"):
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

if MESH == "rectangular":
    mesh_pixels_yx = (
        39 if SOURCE_PIXELS_REQUESTED is None else int(round(math.sqrt(SOURCE_PIXELS_REQUESTED)))
    )
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = 1500 if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    n_source_pixels = n_mesh_vertices

print(f"  Source pixels: {n_source_pixels}")

with timer.section("adapt_image_build"):
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_plane_mesh_grid = None
if MESH != "rectangular":
    with timer.section("image_mesh_hilbert"):
        image_mesh = al.image_mesh.Hilbert(
            pixels=n_mesh_vertices, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=dataset.mask, adapt_data=adapt_image
        )
    print(f"  Mesh vertices placed: {image_plane_mesh_grid.shape[0]}")

with timer.section("model_build"):
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

    if MESH == "rectangular":
        reg_scheme = "constant"
        mesh_obj = rect_mesh_classes(_cli)[1](shape=mesh_shape, weight_power=1.0, weight_floor=0.0)
        regularization = al.reg.Constant(coefficient=1.0)
        reg_provenance = {"scheme": reg_scheme, "coefficient": 1.0}
    else:
        mesh_obj = al.mesh.Delaunay(pixels=n_mesh_vertices, zeroed_pixels=0)
        reg_scheme, regularization, reg_provenance = delaunay_regularization(_cli)

    pixelization = al.Pixelization(mesh=mesh_obj, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

param_vector = model.physical_values_from_prior_medians
instance = model.instance_from_vector(vector=param_vector)
tracer = al.Tracer(galaxies=list(instance.galaxies))

_adapt_kwargs = {
    "galaxy_image_dict": {instance.galaxies.source: adapt_image},
    "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
}
if image_plane_mesh_grid is not None:
    _adapt_kwargs["galaxy_image_plane_mesh_grid_dict"] = {
        instance.galaxies.source: image_plane_mesh_grid
    }
    _adapt_kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
        "('galaxies', 'source')": image_plane_mesh_grid
    }
adapt_images = al.AdaptImages(**_adapt_kwargs)

# The cell's border relocator mode ``cell`` (True), as phase B ran it.
_settings = al.Settings(
    use_border_relocator=True,
    use_mixed_precision=_cli.use_mixed_precision,
)
BORDER_RELOCATOR_RESOLVED = bool(_settings.use_border_relocator)

print("\n--- S0 -> S3: lens light converted to regular profiles + subtracted (eager) ---")
with timer.section("fit_imaging_eager"):
    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=_settings,
        xp=np,
    )
with timer.section("s3_system_build"):
    system_s3 = active_set_steps.fixed_light_system_from(
        fit,
        dataset,
        adapt_images=adapt_images,
        settings=_settings,
        name="S3_mge_converted_to_regular",
    )
log_evidence_s3_library = float(system_s3.fit.figure_of_merit)
print(f"  S3 figure_of_merit (library) = {log_evidence_s3_library}")

# ===================================================================
# PART N — the Nautilus fit of the S3 system, recorded
# ===================================================================
# The fixed-light model: the S3 lens has no light (it was subtracted from the
# dataset), so the lens is mass + shear with the cell's priors, and the source is
# the SAME pixelization object the S3 instance carries.

_s3_lens, _s3_source = list(system_s3.source_only_tracer.galaxies)
_s3_lens_attrs = sorted(
    _k for _k in vars(_s3_lens) if not _k.startswith("_") and _k not in {"id", "redshift"}
)
if _s3_lens_attrs != ["mass", "shear"]:
    raise AssertionError(
        f"the S3 source-only lens carries {_s3_lens_attrs}, not ['mass', 'shear'] — the "
        f"captured lanes would not be the phase-B lane family"
    )

capture_model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear),
        source=af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization),
    )
)
parameter_paths = [".".join(_p) for _p in capture_model.paths]
print(f"\n  Capture model: {capture_model.total_free_parameters} free parameters")
for _p in parameter_paths:
    print(f"    {_p}")
_register_model_pytrees(capture_model)

analysis = al.AnalysisImaging(
    dataset=system_s3.dataset,
    adapt_images=adapt_images,
    settings=_settings,
    use_jax=True,
)

# The fiducial through the SAME analysis: the prior medians are the S3 fiducial,
# so this is the S3 system's own likelihood (recorded, and a sanity print).
_fiducial_instance = capture_model.instance_from_vector(
    vector=capture_model.physical_values_from_prior_medians
)
with timer.section("fiducial_jit"):
    _fiducial_ll = float(
        jax.jit(lambda _t: analysis.log_likelihood_function(instance=_t))(
            jax.tree_util.tree_map(jnp.asarray, _fiducial_instance)
        )
    )
print(f"  fiducial log likelihood (jit, library PDIP): {_fiducial_ll:.6f}")

search = af.Nautilus(
    n_live=N_LIVE,
    n_batch=N_BATCH,
    n_like_max=N_LIKE_MAX,
    seed=SEED,
    number_of_cores=1,
    use_jax_vmap=True,
    iterations_per_quick_update=1e12,
    silence=False,
)


class _SamplingComplete(Exception):
    """Raised in place of the search's post-fit ``perform_update``.

    With ``NullPaths`` Nautilus runs ONE ``Sampler.run`` to ``n_like_max`` and
    autofit then calls ``perform_update`` once, which builds the samples and
    evaluates a per-sample latent-variable jit over every accepted sample — on the
    laptop witness that cost more than the sampling itself, and none of it changes
    a recorded batch. The capture ends at the last batch instead.
    """


def _skip_post_fit(*_args, **_kwargs):
    raise _SamplingComplete()


search.perform_update = _skip_post_fit

recorder = nautilus_batches.BatchRecorder(
    deadline=(time.monotonic() + CAPTURE_SECONDS) if CAPTURE_SECONDS is not None else None
)
fit_status = "search.fit returned (post-fit hook not reached)"
fit_error = None
_t_start = time.perf_counter()
try:
    with nautilus_batches.recording(recorder):
        _result = search.fit(model=capture_model, analysis=analysis)
except _SamplingComplete:
    fit_status = "sampling complete (n_like_max or convergence); post-fit result skipped"
except nautilus_batches.CaptureBudgetReached as _exc:
    fit_status = "capture-seconds reached"
    fit_error = str(_exc)
except Exception as _exc:  # noqa: BLE001 — record what was captured before re-raising below
    fit_status = "error"
    fit_error = f"{type(_exc).__name__}: {_exc}"
capture_wall_s = time.perf_counter() - _t_start
print(
    f"\n  capture: {recorder.n_calls} calls, {recorder.n_lanes} lanes, "
    f"{capture_wall_s:.1f} s — {fit_status}"
)

# ===================================================================
# Output
# ===================================================================


def _library_revision(module) -> dict:
    _repo = Path(module.__file__).resolve().parents[1]
    try:
        _sha = subprocess.run(
            ["git", "-C", str(_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        _sha = None
    return {"head": _sha, "path": str(_repo)}


import autoarray as _autoarray  # noqa: E402
import autogalaxy as _autogalaxy  # noqa: E402
import autonerves as _autonerves  # noqa: E402
import nautilus as _nautilus  # noqa: E402

meta = {
    "schema_version": nautilus_batches.SCHEMA_VERSION,
    "mesh": MESH,
    "dataset": DATASET,
    "parameter_paths": parameter_paths,
    "s3_fingerprint": {
        "dataset_sha256": dataset_sha256,
        "mesh": MESH,
        "source_pixels": int(n_source_pixels),
        "regularization": reg_provenance,
        "border_relocator": BORDER_RELOCATOR_RESOLVED,
        "use_mixed_precision": bool(_cli.use_mixed_precision),
        "log_evidence_s3_library": log_evidence_s3_library,
    },
    "fiducial_log_likelihood_jit": _fiducial_ll,
    "nautilus": {
        "n_live": N_LIVE,
        "n_batch": N_BATCH,
        "n_like_max": N_LIKE_MAX,
        "seed": SEED,
        "use_jax_vmap": True,
        "number_of_cores": 1,
        "version": getattr(_nautilus, "__version__", None),
        "production_reference": (
            "euclid_strong_lens_modeling_pipeline/scripts/full_model.py source_pix_1: "
            "af.Nautilus(n_live=150, n_batch=20, **SettingsSearch.search_dict) with "
            "use_jax_vmap=True and number_of_cores=1 (SettingsSearch defaults)"
        ),
        "departures_from_production": [
            "regularization fixed at the cell's coefficients (source_pix_1 frees AdaptSplit)",
            "priors are the phase-B cell's Gaussians, not chained from source_lp",
            "seeded (production passes no seed)",
            "quick updates disabled (iterations_per_quick_update=1e12; render-only)",
            "NullPaths: no output directory, so no checkpoint and one Sampler.run call",
            "post-fit perform_update skipped (samples / latent variables never built)",
        ],
    },
    "solver": "library default positive-only solver (Settings() as packaged; PDIP)",
    "capture_seconds": CAPTURE_SECONDS,
    "fit_status": fit_status,
    "fit_error": fit_error,
    "capture_wall_s": capture_wall_s,
    "cell_source_sha256": cell_source_sha256,
    "library_revisions": {
        "PyAutoNerves": _library_revision(_autonerves),
        "PyAutoFit": _library_revision(af),
        "PyAutoArray": _library_revision(_autoarray),
        "PyAutoGalaxy": _library_revision(_autogalaxy),
        "PyAutoLens": _library_revision(al),
    },
}

_output_dir = (
    _cli.output_dir
    if _cli.output_dir is not None
    else _workspace_root / "results" / "breakdown" / "imaging"
)
_stem = f"nautilus_batches_{MESH}"
if SOURCE_PIXELS_REQUESTED is not None:
    _stem = f"{_stem}_n{int(n_source_pixels)}"
npz_path = Path(_output_dir) / f"{_stem}.npz"
json_path = Path(_output_dir) / f"{_stem}.json"

file_summary = nautilus_batches.save(npz_path, recorder, meta) if recorder.n_calls else None
_call_sizes = [int(_p.shape[0]) for _p in recorder.parameters]
provenance = {
    "meta": meta,
    "file": {
        "npz": str(npz_path),
        "npz_sha256": _sha256(npz_path) if file_summary else None,
        **(file_summary or {"n_calls": 0, "n_lanes": 0}),
    },
    "batch_shape_census": {
        "n_batch": N_BATCH,
        "all_calls_are_n_batch": bool(_call_sizes) and all(_s == N_BATCH for _s in _call_sizes),
        "sizes_seen": sorted(set(_call_sizes)),
        "note": (
            "Nautilus.fit_x1_cpu hands call_wrap to nautilus.Sampler with vectorized="
            "use_jax_vmap; Sampler.evaluate_likelihood passes every proposal batch whole. A "
            "size other than n_batch is a Nautilus batch of another shape, recorded as it came."
        ),
    },
    "first_call_wall_s": recorder.call_wall_s[0] if recorder.call_wall_s else None,
    "median_call_wall_s": (
        float(np.median(recorder.call_wall_s[1:])) if len(recorder.call_wall_s) > 1 else None
    ),
    "device": device_info_dict(),
    "machine": machine_info_dict(),
    "thread_env": _observe_thread_env(),
    "timer": {_label: _secs for _label, _secs in timer.records},
    "note": (
        "Captured proposal batches only. No rate and no millisecond in this file is a result: "
        "the certified rate and the timed table come from fixed_light_trace.py --lanes captured."
    ),
}
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(provenance, indent=2, default=str))
print(f"  batches: {npz_path}")
print(f"  provenance: {json_path}")
print(f"  batch sizes seen: {sorted(set(_call_sizes))} (n_batch {N_BATCH})")

if fit_status == "error":
    raise RuntimeError(f"the Nautilus fit raised after {recorder.n_calls} calls: {fit_error}")
if recorder.n_calls == 0:
    raise RuntimeError("no batch reached Fitness.call_wrap — nothing was captured")
