"""Numba CPU Profiling: the S3 positivity solve, kernel by kernel
================================================================

Phase 2 of the numba CPU profiling campaign (autolens_profiling#265).

``fixed_light_numba.py`` measures the **whole** likelihood call and decomposes
it. This cell measures the one step that decomposition points at: the positivity
solve of the source-only (S3) system, on the numba CPU path, at one thread —
kernel against kernel, on the same matrices, scored against the library's own
answer.

What the library actually spends
--------------------------------

On a source-only Delaunay ``N = 1500`` HST system ``fnnls_cholesky`` converges in
**zero** outer iterations: almost every entry is already free and only a handful
are negative. Its cost is therefore not the active-set scheme — it is **two
back-to-back full factorisations**. ``reconstruction_positive_only_from`` seeds
the passive set from the sign of ``np.linalg.solve(F + lambda H, D)``, an LU
factorisation of the whole matrix whose factor is thrown away
(``inversion_util.py:435``), and ``fnnls_cholesky`` then factorises the passive
block again (``fnnls.py:105``), which at ``k ~ n`` is essentially the same matrix.

So the lever this cell probes is the **factorisation count**, and the candidate
(``K3``) is the same Bro & de Jong iteration started from a single factorisation
that is kept.

The rows
--------

======  ===============================================================  ============
Row     Kernel                                                           Verdict
======  ===============================================================  ============
``K0``  library ``fnnls_cholesky``, cold (dense-sign seed, memo off)      the reference
``K1``  library entry point, memo warm, **random-walk** instance stream   candidate
``K1i`` library entry point, memo warm, **i.i.d.** instance stream        candidate
``K3``  ``nnls_factor_reuse`` — one Cholesky, downdates, same outer loop  candidate
``K4``  ``cho_factor``/``cho_solve``, positivity dropped                  RECORDED floor
======  ===============================================================  ============

**There is no ``K2``.** The numpy certified active set lives in
``likelihood_breakdown.active_set_steps``, which imports JAX — and this cell
asserts ``"jax" not in sys.modules``, because a JAX runtime in the process
changes the thread pools and the BLAS every row here is measured through. It is
not a gap in the row set: GPU phase 2 already measured the certified scheme
tying ``fnnls`` on this system, so it is a row whose answer is known and whose
cost of measuring it here is a different process. Run
``fixed_light_cpu_kernels.numpy_certified_row`` in a JAX process if it is wanted
again.

Why the two memo rows differ
----------------------------

The cross-evaluation memo seeds each solve from the **previous** solve's passive
set, so it is only worth anything between *different* evaluations — and how much
it is worth depends entirely on how far apart they sit. ``K1`` walks (successive
sampler evaluations, small correlated steps: the regime the memo was built for);
``K1i`` draws independently from the central 20 % of every prior (its worst
case). Both are timed one call per stream member, never ``n_repeats`` calls on
one system: re-solving one matrix would warm-start it from its own answer and
report a solve production never gets. The first call of each stream is cold and
is reported separately.

Scoring
-------

No row is a bare millisecond. Every kernel's reconstruction is scored for log
evidence **in numpy** (``fixed_light_cpu_common.log_evidence_terms_np`` — the
term-for-term twin of the JAX scorer the earlier phases used, because this
process may not import JAX), and every positivity row carries an equivalence pin
against the library's own reconstruction at ``rtol = 1e-9``. ``K4`` carries
``"recorded"`` instead: it is a different, infeasible minimiser, so there is no
sense in which it passes or fails an equivalence, and its Δ log-evidence and
negative count are the numbers to read.

Threads
-------

One. Production parallelises across likelihood evaluations with multiprocessing —
one single-threaded numba likelihood per process — so per-call multi-core gains
are not a target of this campaign. ``NUMBA_NUM_THREADS`` and the BLAS/OpenMP
family are pinned before ``import numpy`` and recorded verbatim.

Output
------
``results/breakdown/imaging/fixed_light_numba_solvers_<mesh>[_<dataset>][_n<N>]_<config>.{json,png}``
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

# ---------------------------------------------------------------------------
# HEADER ORDER IS LOAD-BEARING — read before moving anything below this line
# ---------------------------------------------------------------------------
# OpenBLAS / MKL / OpenMP read their thread-count variables ONCE, when the shared
# library loads, and numba reads NUMBA_NUM_THREADS once, when it initialises its
# threading layer. Both therefore have to be set before `import numpy` (which
# pulls in the BLAS) and before anything imports numba. `_production_config` and
# `_profile_cli` are stdlib-only at import time, which is what makes this
# possible. Same reasoning, verbatim, as `fixed_light_numba.py`.
import argparse as _argparse  # noqa: E402
import os as _os  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli as _parse_profile_cli  # noqa: E402

_cli = _parse_profile_cli()

#: Every kernel row this cell can run, in the order the summary prints them.
#: There is no K2 — see the module docstring.
ALL_KERNEL_KEYS = ("K0", "K1", "K1i", "K3", "K4")

#: The two instance streams. ``walk`` is production's regime (successive sampler
#: evaluations sit close together); ``iid`` is its worst case.
ALL_STREAM_KEYS = ("walk", "iid")

_cell_parser = _argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_cell_parser.add_argument("--mesh", choices=("delaunay", "rectangular"), default="delaunay")
_cell_parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")
_cell_parser.add_argument("--threads", type=int, default=1)
_cell_parser.add_argument("--n-repeats", type=int, default=10)
_cell_parser.add_argument("--n-stream", type=int, default=9)
_cell_parser.add_argument("--instances", choices=ALL_STREAM_KEYS, default="iid")
_cell_parser.add_argument("--pins", choices=("fp64", "none"), default="fp64")
_cell_args = _cli.parse_cell_args(_cell_parser)

MESH = _cell_args.mesh
DATASET = _cell_args.dataset
N_THREADS = max(1, int(_cell_args.threads))
N_REPEATS = max(1, int(_cell_args.n_repeats))
#: One cold call plus the warm remainder. Two is the minimum that means anything.
N_STREAM = max(2, int(_cell_args.n_stream))
PRIMARY_STREAM = _cell_args.instances
PINS_MODE = _cell_args.pins
PINS_ASSERT = PINS_MODE == "fp64"

# The BLAS/OpenMP family, through the shared helper so the recorded block is the
# same shape every other production cell records...
thread_env = _pin_thread_env(N_THREADS)

# ...and numba's own knob, cell-locally (see `fixed_light_numba.py`'s note: the
# tuple in `_production_config.THREAD_ENV_VARS` is recorded verbatim into every
# existing numba cell's `configuration.thread_env`, and widening it would change
# those blocks and the pins that describe them).
_numba_threads_before = _os.environ.get("NUMBA_NUM_THREADS")
_os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)
numba_thread_env = {
    "NUMBA_NUM_THREADS": _os.environ["NUMBA_NUM_THREADS"],
    "preexisting": _numba_threads_before,
    "overridden": (
        _numba_threads_before
        if _numba_threads_before is not None and _numba_threads_before != str(N_THREADS)
        else None
    ),
    "note": (
        "Set cell-locally before numpy/numba import. Deliberately NOT added to "
        "_production_config.THREAD_ENV_VARS: that tuple is recorded verbatim in "
        "every existing numba cell's configuration.thread_env and widening it "
        "would change those blocks and the pins that describe them."
    ),
}

# The cross-evaluation NNLS memo is OFF for the cell as a whole. The two memo
# rows turn it on for their own duration and restore it afterwards
# (`fixed_light_numpy_solvers.memo_warm_row`), so K0/K3/K4 cannot be served a
# seed left behind by K1.
_nnls_warm_start_before = _os.environ.get("AUTOARRAY_NNLS_WARM_START")
_os.environ["AUTOARRAY_NNLS_WARM_START"] = "0"

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    sys.exit(0)

from likelihood_breakdown import (  # noqa: E402
    fixed_light_cpu_common,
    fixed_light_numpy_solvers,
    fixed_light_system,
)
from simulators.imaging import INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    delaunay_regularization,
    machine_info_dict,
    record_pinned_check,
    rect_mesh_classes,
    resolve_output_paths,
)

# NO `device_info_dict()` ANYWHERE IN THIS CELL. It imports jax unconditionally
# (`_profile_cli.py:392`), which would put a JAX runtime in a process whose whole
# purpose is to measure a CPU kernel without one.
assert "jax" not in sys.modules, (
    "this cell must not import JAX at any level: a JAX runtime in the process "
    "changes the thread pools and the BLAS the numba rows are measured through. "
    f"Imported by: {sorted(m for m in sys.modules if m.startswith('jax'))}"
)

from autoarray.inversion.inversion.imaging_numba.sparse import (  # noqa: E402
    InversionImagingSparseNumba,
)

#: Every positivity row must reach the library's own reconstruction to this
#: relative tolerance. A kernel that does not is not faster, it is different.
EQUIVALENCE_RTOL = fixed_light_numpy_solvers.EQUIVALENCE_RTOL

#: Seeds. 263 is the i.i.d. seed ``fixed_light_numba.py`` draws its instance
#: stream with, reused so the two cells walk the same points; the walk uses
#: 263 + 1 so the two streams are not the same numbers in a different order.
IID_SEED = 263
WALK_SEED = IID_SEED + 1

#: The random walk's per-parameter step, as a fraction of each prior's width
#: (a Gaussian prior's sigma, a uniform prior's ``upper - lower``). The same
#: construction as ``scripts/misc/nnls_warm_start/delaunay_numba_nnls_iterations.py``,
#: which is where the memo's regime was first measured.
RANDOM_WALK_STEP_FRACTION = 0.05

#: The i.i.d. stream's unit-cube window — the central 20 % of every prior.
IID_UNIT_LOW = 0.4
IID_UNIT_HIGH = 0.6


# ===================================================================
# PART A — Setup
# ===================================================================
# Deliberately the same construction as ``fixed_light_numba.py`` PART A (which is
# itself ``fixed_light.py``'s, which is the pixelized breakdown cells'), so a
# kernel row here and a whole-call row there describe the same model rather than
# two experiments. The repo's convention is that each cell builds its own PART A
# — ``fixed_light_library.py`` says so in as many words — because the setup runs
# at module level and cannot be imported without running the cell that owns it.

instrument = DATASET

print(f"\n--- Dataset loading & masking [{instrument}, mesh={MESH}] ---")

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

SOURCE_PIXELS_REQUESTED = _cli.source_pixels
_default_source_pixels = 39 * 39 if MESH == "rectangular" else 1500

if MESH == "rectangular":
    _requested = (
        _default_source_pixels if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    )
    mesh_pixels_yx = int(round(math.sqrt(_requested)))
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)
    n_source_pixels = mesh_pixels_yx * mesh_pixels_yx
    n_mesh_vertices = None
else:
    mesh_pixels_yx = None
    mesh_shape = None
    n_mesh_vertices = (
        _default_source_pixels if SOURCE_PIXELS_REQUESTED is None else int(SOURCE_PIXELS_REQUESTED)
    )
    n_source_pixels = n_mesh_vertices

print("\n--- Adapt image (lensed source) ---")

adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

image_plane_mesh_grid = None
if MESH != "rectangular":
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
model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

print(f"  Total free parameters: {model.total_free_parameters}")
print(f"  Regularization: {reg_scheme} ({reg_provenance})")


# ===================================================================
# The two instance streams
# ===================================================================


def _prior_step_sigmas():
    """Per-parameter walk step: ``RANDOM_WALK_STEP_FRACTION`` of each prior's width."""
    sigmas = []
    for prior in model.priors_ordered_by_id:
        sigma = getattr(prior, "sigma", None)
        if sigma is None:
            sigma = float(prior.upper_limit) - float(prior.lower_limit)
        sigmas.append(RANDOM_WALK_STEP_FRACTION * float(sigma))
    return np.asarray(sigmas)


def _prior_support():
    """Per-parameter (lower, upper) physical limits, inset so a step stays interior."""
    lower = np.asarray([float(p.lower_limit) for p in model.priors_ordered_by_id])
    upper = np.asarray([float(p.upper_limit) for p in model.priors_ordered_by_id])
    span = np.where(np.isfinite(upper - lower), (upper - lower), 0.0)
    return lower + 1.0e-6 * span, upper - 1.0e-6 * span


def walk_instances(n):
    """``n`` instances along a seeded Gaussian random walk from the prior medians.

    The regime the cross-evaluation memo was built for: successive sampler
    evaluations sit close together, so their passive sets overlap almost
    entirely. The construction is
    ``scripts/misc/nnls_warm_start/delaunay_numba_nnls_iterations.py``'s, copied
    rather than imported — that cell runs its whole profile at module level.
    """
    rng = np.random.default_rng(WALK_SEED)
    step_sigmas = _prior_step_sigmas()
    lower, upper = _prior_support()

    vector = np.asarray(model.physical_values_from_prior_medians, dtype=float)
    out = []
    for _ in range(n):
        vector = np.clip(vector + rng.normal(0.0, step_sigmas), lower, upper)
        out.append(model.instance_from_vector(vector=list(vector)))
    return out


def iid_instances(n):
    """``n`` independent draws from the central 20 % of every prior — the worst case."""
    rng = np.random.default_rng(IID_SEED)
    return [
        model.instance_from_unit_vector(
            unit_vector=list(rng.uniform(IID_UNIT_LOW, IID_UNIT_HIGH, size=model.prior_count))
        )
        for _ in range(n)
    ]


instance_streams = {
    "walk": walk_instances(N_STREAM),
    "iid": iid_instances(N_STREAM),
}

#: The reference system K0 / K3 / K4 are measured on: the head of the primary
#: stream. Recorded, because which point in parameter space a kernel millisecond
#: was taken at is part of the number.
reference_instance = instance_streams[PRIMARY_STREAM][0]


def adapt_images_of(one_instance):
    """``AdaptImages`` for one instance — the dicts are keyed on its own galaxy."""
    kwargs = {
        "galaxy_image_dict": {one_instance.galaxies.source: adapt_image},
        "galaxy_name_image_dict": {"('galaxies', 'source')": adapt_image},
    }
    if image_plane_mesh_grid is not None:
        kwargs["galaxy_image_plane_mesh_grid_dict"] = {
            one_instance.galaxies.source: image_plane_mesh_grid
        }
        kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
            "('galaxies', 'source')": image_plane_mesh_grid
        }
    return al.AdaptImages(**kwargs)


n_image_pixels = dataset.data.shape[0]
n_over_sampled_pixels = dataset.grids.lp.over_sampled.shape[0]

print("\n--- Configuration (determines run time) ---")
print(f"  Instrument:              {instrument}")
print(f"  Mesh:                    {MESH}")
print(f"  Image pixels (masked):   {n_image_pixels}")
print(f"  Over-sampled pixels:     {n_over_sampled_pixels}")
print(f"  Source pixels:           {n_source_pixels}")
print(f"  Threads:                 {N_THREADS}")
print(f"  n-repeats:               {N_REPEATS}")
print(f"  n-stream:                {N_STREAM}")
print(f"  Primary stream:          {PRIMARY_STREAM}")
print(f"  Pins mode:               {PINS_MODE}")
print(f"  OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"  NUMBA_NUM_THREADS:       {os.environ.get('NUMBA_NUM_THREADS', '(unset)')}")

_settings = al.Settings(use_border_relocator=True)

#: The memo rows run through the library entry point, which only consults the
#: memo when the settings ask for it AND the env allows it AND a fingerprint is
#: given (``inversion_util.py:388-393``). All three are supplied here.
_settings_memo = al.Settings(use_border_relocator=True, nnls_warm_start_memo=True)

print("\n--- S0: full FitImaging with linear MGE lens light (eager, dense) ---")

_adapt_images_s0 = adapt_images_of(reference_instance)
fit_s0 = al.FitImaging(
    dataset=dataset,
    tracer=al.Tracer(
        galaxies=list(reference_instance.galaxies), fields=[reference_instance.fields]
    ),
    adapt_images=_adapt_images_s0,
    settings=_settings,
    xp=np,
)

print("\n--- S3: lens light fixed + subtracted, numba CPU sparse operator ---")

system_s3 = fixed_light_system.fixed_light_system_from(
    fit_s0,
    dataset,
    adapt_images=_adapt_images_s0,
    settings=_settings,
    name="S3_sparse_numba",
    scaling="numpy",
    sparse_operator="rebake_cpu",
)
dataset_s3 = system_s3.dataset

if not isinstance(system_s3.inversion, InversionImagingSparseNumba):
    raise AssertionError(
        f"the S3 system dispatched {type(system_s3.inversion).__name__}, not "
        f"InversionImagingSparseNumba — every row below would be measuring the dense "
        f"formalism's matrices wearing the sparse one's label."
    )
if system_s3.fit._xp is not np:
    raise AssertionError(f"the S3 fit's _xp is {system_s3.fit._xp!r}, not numpy.")

print(f"  inversion class: {type(system_s3.inversion).__name__}")
print(
    f"  S3: n={system_s3.n_params} (mapper {system_s3.n_mapper} + funcs {system_s3.n_funcs}), "
    f"edge-zeroed {int(system_s3.edge_zero_mask.sum())}"
)
print(f"  subtracted light flux: {system_s3.subtracted_light_flux}")


def instance_s3_of(one_instance):
    """``one_instance`` with the lens light stripped, for the S3 system."""
    stripped = copy.deepcopy(one_instance)
    galaxies = list(
        fixed_light_system._light_stripped_tracer(
            al.Tracer(galaxies=list(one_instance.galaxies), fields=[one_instance.fields])
        ).galaxies
    )
    stripped.galaxies.lens = galaxies[0]
    stripped.galaxies.source = galaxies[1]
    return stripped


def system_for(one_instance, name):
    """The S3 linear system of one instance, on the **fixed** subtracted dataset.

    The lens light is fixed at the reference instance by construction — that is
    the whole premise of S3 — so the subtracted dataset and its baked CPU
    operator are built once and only the mass and source parameters vary. Every
    system in a stream therefore differs from its neighbour exactly as two
    successive sampler evaluations do.
    """
    stripped = instance_s3_of(one_instance)
    # The adapt-image dicts are keyed on the instance's OWN source galaxy object
    # (`_light_stripped_tracer` passes the source through by identity), so they
    # are rebuilt per instance. Sharing one instance's dicts across the stream
    # would miss on every other member and silently rebuild the Hilbert mesh.
    fit = al.FitImaging(
        dataset=dataset_s3,
        tracer=al.Tracer(galaxies=list(stripped.galaxies), fields=[stripped.fields]),
        adapt_images=adapt_images_of(stripped),
        settings=_settings,
        xp=np,
    )
    if not isinstance(fit.inversion, InversionImagingSparseNumba):
        raise AssertionError(
            f"{name}: dispatched {type(fit.inversion).__name__}, not InversionImagingSparseNumba."
        )
    return fixed_light_system.linear_system_from(fit, dataset_s3, name=name, scaling="numpy")


print(f"\n--- Building the two instance streams ({N_STREAM} systems each) ---")
systems = {}
for _stream in ALL_STREAM_KEYS:
    systems[_stream] = [
        system_for(one, f"S3_{_stream}_{index}")
        for index, one in enumerate(instance_streams[_stream])
    ]
    print(f"  {_stream}: {len(systems[_stream])} systems, n={systems[_stream][0].n_params}")

reference_system = systems[PRIMARY_STREAM][0]
view = fixed_light_cpu_common.solver_view_of(reference_system)

#: The library's own answer on the reference system: what every positivity row is
#: pinned against, read off the inversion rather than recomputed.
reference_x = np.asarray(reference_system.inversion.reconstruction, dtype=float)
reference_log_evidence = float(
    fixed_light_cpu_common.log_evidence_terms_np(reference_system, reference_x)["log_evidence"]
)

#: The memo key's index space. Any stable string does — the library keys on
#: ``f"{n}:{fingerprint}"`` (``nnls_memo.memo_key``) and every system in a stream
#: is the same size, which is exactly the "nearby evaluations share a key"
#: property the memo exists to exploit.
MEMO_FINGERPRINT = f"fixed_light_numba_solvers:{MESH}:{instrument}:n{int(view['n_seen_by_solver'])}"

print(
    f"\n  solver sees n={view['n_seen_by_solver']} of {view['n_full']} "
    f"({view['n_full'] - view['n_seen_by_solver']} edge-zeroed)"
)
print(f"  reference log evidence: {reference_log_evidence}")


# ===================================================================
# PART B — The kernel rows
# ===================================================================


def _load_average():
    """``/proc/loadavg``'s 1/5/15-minute figures, or ``None`` where unavailable.

    Recorded at both ends of the timed rows, as first-class data: every timing
    here is wall clock on a shared machine, and a load average above the core
    count means the row was competing for the cores it was measured on.
    """
    try:
        return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        return None


load_average_at_start = _load_average()

print("\n" + "=" * 70)
print("KERNEL ROWS — the S3 positivity solve")
print("=" * 70)
if load_average_at_start is not None:
    print(f"  load average at start: {load_average_at_start} (cores: {os.cpu_count()})")

rows: dict[str, dict] = {}

rows["K0"] = fixed_light_numpy_solvers.library_cold_row(
    reference_system, view, reference_log_evidence, reference_x, n_repeats=N_REPEATS
)


def _stream_reference(stream_key):
    """The library's own answer on a stream's head, and its log evidence.

    Each memo row solves ITS OWN stream, so its Δ log-evidence has to be measured
    against that stream's head — not against the primary stream's. Scoring a walk
    row against an i.i.d. reference would report the distance between two points
    in parameter space as if it were a solver difference.
    """
    head = systems[stream_key][0]
    x_head = np.asarray(head.inversion.reconstruction, dtype=float)
    return (
        float(fixed_light_cpu_common.log_evidence_terms_np(head, x_head)["log_evidence"]),
        x_head,
    )


_walk_log_evidence, _walk_x = _stream_reference("walk")
_iid_log_evidence, _iid_x = _stream_reference("iid")

rows["K1"] = fixed_light_numpy_solvers.memo_warm_row(
    systems["walk"],
    _walk_log_evidence,
    _walk_x,
    label="library entry point, memo warm (random-walk stream)",
    fingerprint=f"{MEMO_FINGERPRINT}:walk",
    settings=_settings_memo,
)
rows["K1i"] = fixed_light_numpy_solvers.memo_warm_row(
    systems["iid"],
    _iid_log_evidence,
    _iid_x,
    label="library entry point, memo warm (i.i.d. stream — the worst case)",
    fingerprint=f"{MEMO_FINGERPRINT}:iid",
    settings=_settings_memo,
)
rows["K3"] = fixed_light_numpy_solvers.factor_reuse_row(
    reference_system, view, reference_log_evidence, reference_x, n_repeats=N_REPEATS
)
rows["K4"] = fixed_light_numpy_solvers.unconstrained_row(
    reference_system, view, reference_log_evidence, reference_x, n_repeats=N_REPEATS
)

# K1's own reference is its stream's head, not the primary stream's, so its
# Δ log-evidence is against the right problem. Say so in the row rather than
# leaving a reader to infer it from a seed.
rows["K1"]["scored_against"] = "the walk stream's own head (systems['walk'][0])"
rows["K1i"]["scored_against"] = "the i.i.d. stream's own head (systems['iid'][0])"
for _key in ("K0", "K3", "K4"):
    rows[_key]["scored_against"] = f"the reference system (the {PRIMARY_STREAM} stream's head)"
    rows[_key]["instance_stream"] = PRIMARY_STREAM
rows["K1"]["instance_stream"] = "walk"
rows["K1i"]["instance_stream"] = "iid"

for _key, _row in rows.items():
    _pin = _row.get("equivalence_pin")
    _verdict = "recorded" if _pin == "recorded" else ("PASS" if _pin["passed"] else "FAIL")
    print(
        f"  {_key:<4} {_row['ms']:>9.3f} ms  [{_verdict:>8}]  "
        f"Δlog-ev {_row['d_log_evidence_vs_library']:+.3e} nats  {_row['label']}"
    )


# ===================================================================
# PART C — Gates
# ===================================================================

gates: dict[str, dict] = {}


def _record_gate(name, status, detail):
    gates[name] = {"status": status, **detail}
    print(f"  [{status:>8}] {name}: {detail.get('summary', '')}")
    return gates[name]


print("\n" + "=" * 70)
print("GATES")
print("=" * 70)

for _key in ("K0", "K1", "K1i", "K3"):
    _row = rows[_key]
    _pin = _row["equivalence_pin"]
    _passed = bool(_pin["passed"])
    _status = ("PASS" if _passed else "FAIL") if PINS_ASSERT else "RECORDED"
    _record_gate(
        f"equivalence_{_key}_reconstruction_equals_library",
        _status,
        {
            "max_abs_diff_reconstruction_vs_library": _row[
                "max_abs_diff_reconstruction_vs_library"
            ],
            "d_log_evidence_nats": _row["d_log_evidence_vs_library"],
            "rtol": _pin["rtol"],
            "summary": (
                f"max |Δx| {_row['max_abs_diff_reconstruction_vs_library']:.3e}, "
                f"Δlog-ev {_row['d_log_evidence_vs_library']:+.3e} nats"
            ),
        },
    )
    if PINS_ASSERT and not _passed:
        raise AssertionError(
            f"{_key} does not reconstruct the library's answer: max |Δx| "
            f"{_row['max_abs_diff_reconstruction_vs_library']:.3e} at rtol "
            f"{_pin['rtol']:g}. Its millisecond is for a different problem."
        )

_record_gate(
    "K4_is_recorded_never_a_headline",
    "RECORDED",
    {
        "equivalence_pin": rows["K4"]["equivalence_pin"],
        "d_log_evidence_nats": rows["K4"]["d_log_evidence_vs_library"],
        "n_negative_entries": rows["K4"]["n_negative_entries"],
        "summary": (
            f"{rows['K4']['n_negative_entries']} negative entries, "
            f"Δlog-ev {rows['K4']['d_log_evidence_vs_library']:+.3e} nats"
        ),
    },
)

_record_gate(
    "K3_uses_one_factorisation",
    "PASS" if rows["K3"]["n_factorisations"] == 1 else "FAIL",
    {
        "n_factorisations": rows["K3"]["n_factorisations"],
        "n_downdates": rows["K3"]["n_downdates"],
        "n_seed_negatives": rows["K3"]["n_seed_negatives"],
        "outer_iterations": rows["K3"]["outer_iterations"],
        "summary": (
            f"{rows['K3']['n_factorisations']} factorisation(s), "
            f"{rows['K3']['n_downdates']} downdate(s), "
            f"{rows['K3']['n_seed_negatives']} seed negative(s)"
        ),
    },
)

# --- What happened to the JAX-free premise, measured rather than asserted -----
#
# This cell asserts `"jax" not in sys.modules` at IMPORT time, and that assert
# holds: nothing this cell imports pulls JAX in. It deliberately does NOT assert
# the same thing after the rows, because **the library imports JAX for itself**
# and the assert would fail on every run for a reason the cell cannot fix:
#
#     autoarray/abstract_ndarray.py:422-428 — `AbstractNDArray.__getitem__` runs
#     `import jax.numpy as jnp` inside a `try/except ImportError` on EVERY index,
#     to decide whether the result needs re-wrapping. The first `FitImaging` here
#     reaches it through `BorderRelocator` -> `derive_indexes.border_slim` ->
#     `mask_2d_util.border_slim_indexes_from`, which slices the mask.
#
# So in an environment where JAX is installed, no process that builds a
# `FitImaging` can stay JAX-free, and a cell that claimed otherwise would be
# claiming something about its environment rather than about itself. What the
# import-time assert buys is still real and is the thing worth having: no XLA
# device is created, no JAX computation runs, and `NPROC` (XLA's CPU intra-op
# pool) is never consulted — every row above is numpy/scipy/numba on the BLAS
# this cell pinned. The post-row state is therefore RECORDED, with its cause, so
# a reader is told the premise's exact shape instead of inferring it from an
# assert that is not there.
jax_after_rows = {
    "jax_in_sys_modules": "jax" in sys.modules,
    "jax_modules": sorted(m for m in sys.modules if m.split(".")[0] in ("jax", "jaxlib")),
    "asserted_at_import": True,
    "asserted_after_rows": False,
    "why_not_asserted_after_rows": (
        "autoarray/abstract_ndarray.py:422-428 imports jax.numpy inside "
        "AbstractNDArray.__getitem__ on every index (guarded only by ImportError), and "
        "the first FitImaging reaches it via BorderRelocator -> border_slim -> "
        "mask_2d_util.border_slim_indexes_from slicing the mask. In an environment "
        "where JAX is installed this is unavoidable, so the post-row state is recorded "
        "rather than asserted. No XLA device is created and no JAX computation runs: "
        "every row is numpy/scipy/numba on the pinned BLAS."
    ),
}
print(
    f"\n  jax in sys.modules after the rows: {jax_after_rows['jax_in_sys_modules']} "
    f"({len(jax_after_rows['jax_modules'])} modules) — imported by the library, not by "
    f"this cell (abstract_ndarray.py:422-428). Recorded, not asserted."
)

load_average_at_end = _load_average()


# ===================================================================
# PART D — Summary, JSON, PNG
# ===================================================================

al_version = al.__version__

print("\n" + "=" * 70)
print(f"NUMBA CPU S3 SOLVER KERNELS — {instrument.upper()} / {MESH} — v{al_version}")
print("=" * 70)

_k0_ms = rows["K0"]["ms"]
for _key, _row in rows.items():
    print(
        f"  {_key:<4} {_row['ms']:>9.3f} ms   x{_k0_ms / max(_row['ms'], 1e-300):>5.2f} vs K0   "
        f"{_row['label']}"
    )

_cores = os.cpu_count() or 1
_load_peak = max(
    [la[0] for la in (load_average_at_start, load_average_at_end) if la is not None] or [0.0]
)
_contended = _load_peak > _cores

#: A config name containing "smoke" declares the run's own purpose.
_is_smoke = bool(_cli.config_name and "smoke" in _cli.config_name.lower())

if _is_smoke:
    _timing_status = "wiring_only_not_measured"
    _timing_status_reason = (
        "This leg ran under a --config-name containing 'smoke'. Its purpose is to prove "
        "the cell end to end and to make the gates fire, NOT to produce timings. The ms "
        "values here are wiring evidence: do not quote them, do not put them in a note, "
        "do not compare them against another leg. The gate outcomes under `gates` and "
        "the equivalence pins ARE valid — they are exact comparisons, not wall-clock "
        "measurements."
    )
elif _contended:
    _timing_status = "measured_under_contention"
    _timing_status_reason = (
        f"Load average reached {_load_peak:.2f} on {_cores} cores during the timed "
        f"rows, so these kernels competed for the cores they were measured on. The "
        f"numbers are real but carry the host's queueing."
    )
else:
    _timing_status = "measured"
    _timing_status_reason = (
        f"Load average stayed at or below the core count ({_load_peak:.2f} on "
        f"{_cores} cores) across the timed rows."
    )

print(f"\n  timing_status: {_timing_status}")
if _timing_status != "measured":
    print(f"    {_timing_status_reason}")

device_block = {
    "use_jax": False,
    "backend": "numba_cpu",
    "inversion_path": type(system_s3.inversion).__name__,
    "hostname": socket.gethostname(),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS") or None,
    "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS") or None,
    "cpu_count": os.cpu_count(),
    "note": (
        "Cell-local. _profile_cli.device_info_dict() is deliberately NOT called: it "
        "imports jax unconditionally (_profile_cli.py:392), and a JAX runtime in this "
        "process changes the thread pools the numba rows are measured through."
    ),
}

configuration = {
    "mesh": MESH,
    "dataset": DATASET,
    "pixel_scale_arcsec": pixel_scale,
    "mask_radius_arcsec": mask_radius,
    "image_pixels_masked": int(n_image_pixels),
    "over_sampled_pixels": int(n_over_sampled_pixels),
    "source_pixels": int(n_source_pixels),
    "mesh_shape": list(mesh_shape) if mesh_shape is not None else None,
    "n_threads": N_THREADS,
    "n_repeats": N_REPEATS,
    "n_stream": N_STREAM,
    "primary_stream": PRIMARY_STREAM,
    "iid_seed": IID_SEED,
    "walk_seed": WALK_SEED,
    "random_walk_step_fraction_of_prior_width": RANDOM_WALK_STEP_FRACTION,
    "iid_unit_window": [IID_UNIT_LOW, IID_UNIT_HIGH],
    "pins_mode": PINS_MODE,
    "use_jax": False,
    "thread_env": thread_env,
    "numba_thread_env": numba_thread_env,
    "threads": fixed_light_cpu_common.thread_block(),
    "nnls_warm_start_env_cell_default": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
    "nnls_warm_start_env_preexisting": _nnls_warm_start_before,
    "memo_fingerprint": MEMO_FINGERPRINT,
    "over_sample_size_lp_rule": {
        "sub_size_list": [4, 2, 2],
        "radial_list": [0.3, 0.6],
        "centre": [0.0, 0.0],
    },
}

summary = {
    "autolens_version": al_version,
    "timing_status": _timing_status,
    "timing_status_reason": _timing_status_reason,
    "contention": {
        "load_average_at_start": load_average_at_start,
        "load_average_at_end": load_average_at_end,
        "cpu_count": _cores,
        "load_average_peak_1min": _load_peak,
        "contention_warning": _contended,
    },
    "device": device_block,
    "machine": machine_info_dict(),
    "instrument": instrument,
    "configuration": configuration,
    "regularization": reg_provenance,
    "all_kernel_keys": list(ALL_KERNEL_KEYS),
    "all_stream_keys": list(ALL_STREAM_KEYS),
    "system": {
        "n_params": int(reference_system.n_params),
        "n_mapper": int(reference_system.n_mapper),
        "n_funcs": int(reference_system.n_funcs),
        "n_seen_by_solver": int(view["n_seen_by_solver"]),
        "n_edge_zeroed": int(view["n_full"] - view["n_seen_by_solver"]),
        "edge_zeroed_pixels": int(reference_system.edge_zero_mask.sum()),
        "subtracted_light_flux": float(system_s3.subtracted_light_flux),
        "reference_log_evidence": reference_log_evidence,
        "inversion_class": type(reference_system.inversion).__name__,
    },
    "rows": rows,
    "gates": gates,
    "gate_thresholds": {"equivalence_rtol": EQUIVALENCE_RTOL},
    "jax_after_rows": jax_after_rows,
    "absent_row_K2": (
        "The numpy certified active set is deliberately ABSENT. It lives in "
        "likelihood_breakdown.active_set_steps, which imports JAX, and this cell asserts "
        "'jax' not in sys.modules because a JAX runtime changes the thread pools and the "
        "BLAS every row here is measured through. GPU phase 2 already measured the "
        "certified scheme tying fnnls on this system, so this is a row whose answer is "
        "known — not a gap. Run fixed_light_cpu_kernels.numpy_certified_row in a JAX "
        "process if it is wanted again."
    ),
    "method": (
        "Each row solves the SAME edge-zeroed subset the library hands its positive-only "
        "solver (Inversion.reconstruction subsets F + lambda H and D to solve_ids_to_keep "
        "before the call and scatters exact zeros back afterwards, abstract.py:607-618). "
        "K0/K3/K4 are timed as the median of n_repeats calls after two warm-ups; the two "
        "memo rows are timed ONE call per stream member, because the memo seeds from the "
        "previous evaluation and repeating one solve would warm-start it from its own "
        "answer."
    ),
    "no_pin_note": (
        "NO PIN WAS EVER CALIBRATED HERE. No numba fixed-light configuration has a pinned "
        "log likelihood, log evidence or log determinant anywhere in this repo, so every "
        "numerical quantity in this JSON is RECORDED and pinned_expected is null. The "
        "hard verdicts are the equivalence gates under `gates`, which are exact "
        "comparisons of reconstructions and not wall-clock measurements."
    ),
}

_cell_name = f"fixed_light_numba_solvers_{MESH}"
if DATASET != "hst":
    _cell_name = f"{_cell_name}_{DATASET}"
if SOURCE_PIXELS_REQUESTED is not None:
    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"

# `cell=` is passed EXPLICITLY. Without it `resolve_output_paths` derives the cell
# name from the first underscore-separated token of the basename — here `fixed` —
# which would collide with every other `fixed_light*` artifact under a shared
# --config-name (`_profile_cli.py:558-562`, autolens_profiling#219).
dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "imaging",
    default_basename=f"{_cell_name}_breakdown_{instrument}_v{al_version}",
    cell=_cell_name,
)
dict_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_labels = []
_times_ms = []
_colors = []
for _key in ALL_KERNEL_KEYS:
    _row = rows[_key]
    _pin = _row["equivalence_pin"]
    _verdict = "RECORDED" if _pin == "recorded" else ("PASS" if _pin["passed"] else "FAIL")
    _labels.append(f"{_key} [{_verdict}]")
    _times_ms.append(_row["ms"])
    if _key == "K0":
        _colors.append("#DD8452")
    elif _key == "K4":
        _colors.append("#8C8C8C")
    else:
        _colors.append("#4C72B0")

fig, ax = plt.subplots(figsize=(11, max(4.0, 0.6 * len(_labels))))
_y = np.arange(len(_labels), dtype=float)
_bars = ax.barh(_y, _times_ms, color=_colors, edgecolor="white", height=0.6)
for _bar, _t in zip(_bars, _times_ms):
    ax.text(
        _bar.get_width() + (max(_times_ms) if _times_ms else 1.0) * 0.01,
        _bar.get_y() + _bar.get_height() / 2,
        f"{_t:.2f} ms",
        va="center",
        fontsize=8,
    )
ax.set_yticks(_y)
ax.set_yticklabels(_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("S3 positivity solve (ms)", fontsize=11)
fig.suptitle(
    f"Numba CPU S3 solver kernels — {MESH} — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f'AutoLens v{al_version}  |  {pixel_scale}"/px  |  n={view["n_seen_by_solver"]} seen by '
    f"solver  |  {N_THREADS} thread(s)  |  {_timing_status}",
    fontsize=9,
)
ax.margins(x=0.20)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")

# ===================================================================
# Pinned-value record — there is no pin
# ===================================================================
# `expected=None` is the honest state: no numba fixed-light configuration has
# ever been calibrated, so there is nothing to drift from. Recording it
# explicitly (rather than omitting the block) is what stops a reader — or
# PyAutoHeart's profiling-drift scan — reading an absent block as a silent pass.

record_pinned_check(dict_path, None, [])
print(
    "\n  Pinned check: NO PIN WAS EVER CALIBRATED HERE "
    "(pinned_expected: null; every numerical quantity above is RECORDED)."
)

print("\nFinished.")
