"""Interferometer curvature route constants on one host: FFT vs ``direct_conv`` arms on real data.

PyAutoArray#632 routes a NumPy single-mapper interferometer inversion between the FFT route
(``InversionInterferometerSparse``) and the numba ``direct_conv`` kernel
(``InversionInterferometerSparseNumba``) with a two-constant cost model,
``inversion_interferometer_numba_util.curvature_route_seconds_from``:

    t_fft = c_fft * S * W_fft(2ny, 2nx)        (fft_route_work_from)
    t_dc  = c_dc  * nnz * ny * nx * min(1, kappa f)   (direct_conv_work_from, PyAutoArray#636)

``c_fft`` = 7.62e-10 and ``c_dc`` = 1.77e-09 were fitted on a laptop (i9-10885H). The
PyAutoArray#636 witness found FFT / full ``direct_conv`` = 0.68 on RAL at the real-Aris 512
1,459-vertex cell against 0.89 predicted (autolens_profiling#429). This script measures the
three arms on the host it runs on, so the constants can be refitted per host:

- ``fft``: the FFT route (``interferometer_route="fft"``);
- ``dc_full``: ``direct_conv`` with the symmetric kernel off
  (``interferometer_direct_conv_symmetric_max_work_fraction=0``);
- ``dc_sym``: ``direct_conv`` with the packaged gate (the symmetric kernel when f <= 0.7).

Each arm builds a fresh ``FitInterferometer`` per repeat and times only ``inv.curvature_matrix``
after the mapper (and, for numba, the kernel index arrays and the symmetric restriction) are built,
so all three arms time the curvature step alone. The first numba repeat includes compilation
unless the cache is warm; the reported figure is the minimum over repeats. Also recorded per
cell: S, nnz, extent, the work fraction f, the model's work units, the shipped constants'
predictions and the class the factory picks with default settings.

Data: Aris's SPT2319-55 (2018.1.00285.S) visibilities and helper ``.py`` files in ``--data-dir``
(not in git), the runner's maximum-likelihood fixed lens, the regular-overlay Delaunay mesh
(``[::step, ::step]`` of the masked grid plus 30 circle-edge points), ConstantSplit, the same
setup as the #635 witness (``results/notes/interferometer_direct_conv_symmetric_2026_10.md``).
The helpers are imported from a temporary copy; nothing in ``--data-dir`` is written.

    python scripts/interferometer/delaunay/route_constants_ral.py --data-dir DIR \\
        --n-pixels 512 --steps 12,8,6,4 --out OUT.json [--cache-dir DIR] [--repeats 3]

Writes one JSON (rewritten after every cell).
"""

import os

# One thread per process (the runner's setting; Nautilus parallelises over processes). Set
# before numpy / numba / JAX import.
for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import sys  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: real-data host script (needs --data-dir); exiting.")
    sys.exit(0)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--data-dir", required=True, type=Path)
parser.add_argument("--n-pixels", type=int, default=512)
parser.add_argument("--steps", type=str, default="8", help="comma-separated mesh steps")
parser.add_argument("--arms", type=str, default="fft,dc_full,dc_sym")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--cache-dir", type=Path, default=None)
parser.add_argument("--edge-points", type=int, default=30)
parser.add_argument("--nufft-chunk-size", type=int, default=1_000_000)
args = parser.parse_args()

try:
    from autonerves import jax_wrapper  # noqa: F401  Configure JAX before PyAutoLens.
except ImportError:  # pragma: no cover
    from autoconf import jax_wrapper  # noqa: F401

import autoarray  # noqa: E402
import autofit  # noqa: E402
import autogalaxy  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402
from autoarray.inversion.inversion.interferometer_numba import (  # noqa: E402
    inversion_interferometer_numba_util as nutil,
)
from autoarray.inversion.inversion.interferometer_numba import sparse as numba_sparse  # noqa: E402
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402

T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def git_info(module):
    d = Path(module.__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()
        return {"path": str(d), "sha": sha}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(d), "sha": None, "error": str(exc)[:200]}


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


data_dir = args.data_dir.resolve()
helper_dir = Path(tempfile.mkdtemp(prefix="aris_helpers_"))
for py in data_dir.glob("*.py"):
    shutil.copy2(py, helper_dir / py.name)
sys.path.insert(0, str(helper_dir))
from image_plane_mesh_grid_utils import append_with_mask_edge_points  # noqa: E402
from pipelines_utils import interferometer_preprocessing  # noqa: E402

# Runner setup (runner_SPT2319-55_2018.1.00285.S.py).
size, mask_radius, mask_centre = 3.0, 1.0, (0.025, -0.05)
n_pixels = args.n_pixels
pixel_scale = size / n_pixels
real_space_mask = al.Mask2D.circular(
    shape_native=(n_pixels, n_pixels),
    pixel_scales=pixel_scale,
    radius=mask_radius,
    centre=mask_centre,
)
log(
    f"mask {n_pixels}x{n_pixels}, pixel_scale {pixel_scale:.6f}, {real_space_mask.pixels_in_mask} px"
)


def transformer_class(uv_wavelengths, real_space_mask):
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=args.nufft_chunk_size,
    )


t = time.perf_counter()
interferometer = al.Interferometer.from_fits(
    data_path=data_dir / "visibilities.fits",
    noise_map_path=data_dir / "sigma.fits",
    uv_wavelengths_path=data_dir / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=transformer_class,
)
t_load = time.perf_counter() - t
n_vis = int(interferometer.uv_wavelengths.shape[0])
log(f"loaded {n_vis} visibilities in {t_load:.1f}s")

cache_dir = args.cache_dir or (
    args.out.resolve().parent / "preprocessing" / f"grid_{n_pixels}x{n_pixels}"
)
cache_hit = (Path(cache_dir) / "nufft_precision_operator.npy").is_file()
t = time.perf_counter()
interferometer = interferometer_preprocessing(
    dataset=interferometer, preprocessing_directory=cache_dir, use_jax=True, show_progress=False
)
t_pre = time.perf_counter() - t
operator = interferometer.sparse_operator
extent = (int(operator.y_shape), int(operator.x_shape))
log(
    f"preprocessing {t_pre:.1f}s (cache {'hit' if cache_hit else 'built'}: {cache_dir}); extent {extent}, "
    f"batch_size {operator.batch_size}; peak RSS {peak_rss_gb():.1f} GB"
)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(-0.09523, -0.10610), ell_comps=(-0.05719, -0.11315), einstein_radius=0.42708
    ),
    shear=al.mp.ExternalShear(gamma_1=-0.07346, gamma_2=-0.03464),
)
image_plane_grid = al.Grid2D.from_mask(real_space_mask)


def model_for(step):
    sampled_grid = image_plane_grid.native.array[::step, ::step]
    sampled_mask = real_space_mask.array[::step, ::step]
    mesh_grid = al.Grid2DIrregular(values=sampled_grid[~sampled_mask])
    n_active = len(mesh_grid)
    mesh_grid = append_with_mask_edge_points(
        image_plane_mesh_grid=mesh_grid, mask=real_space_mask, n_points=args.edge_points
    )
    source = al.Galaxy(
        redshift=2.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.Delaunay(
                pixels=len(mesh_grid) - args.edge_points, zeroed_pixels=args.edge_points
            ),
            regularization=al.reg.ConstantSplit(coefficient=1.0),
        ),
    )
    tracer = al.Tracer(galaxies=[lens, source])
    adapt = al.AdaptImages(galaxy_image_plane_mesh_grid_dict={source: mesh_grid})
    return tracer, adapt, n_active, len(mesh_grid)


ARM_SETTINGS = {
    "fft": dict(interferometer_route="fft"),
    "dc_full": dict(
        interferometer_route="numba", interferometer_direct_conv_symmetric_max_work_fraction=0.0
    ),
    "dc_sym": dict(interferometer_route="numba"),
}


def settings_for(arm):
    return al.Settings(use_positive_only_solver=False, **(ARM_SETTINGS[arm] if arm else {}))


def fit_for(tracer, adapt, settings):
    return al.FitInterferometer(
        dataset=interferometer, tracer=tracer, adapt_images=adapt, settings=settings, xp=np
    )


def time_arm(arm, tracer, adapt):
    reps = []
    for rep in range(args.repeats):
        fit = fit_for(tracer, adapt, settings_for(arm))
        t0 = time.perf_counter()
        inv = fit.inversion
        mapper = inv.cls_list_from(cls=Mapper)[0]
        _ = mapper.pix_indexes_for_sub_slim_index, mapper.pix_weights_for_sub_slim_index
        rec = {"rep": rep, "inversion_class": type(inv).__name__}
        if isinstance(inv, numba_sparse.InversionInterferometerSparseNumba):
            _ = inv.kernel_index_arrays
            rec["use_symmetric_kernel"] = bool(inv.use_symmetric_kernel)
            if rec["use_symmetric_kernel"]:
                _ = inv.symmetric_restriction_inputs
        t1 = time.perf_counter()
        curvature = inv.curvature_matrix
        t2 = time.perf_counter()
        rec.update(t_setup_s=t1 - t0, t_curvature_s=t2 - t1, load_avg=os.getloadavg()[0])
        log(
            f"  {arm} rep {rep}: {rec['inversion_class']} sym={rec.get('use_symmetric_kernel')} "
            f"curvature {rec['t_curvature_s']:.3f}s (setup {rec['t_setup_s']:.2f}s)"
        )
        reps.append(rec)
        last = np.asarray(curvature)
        del fit, inv, curvature
    return reps, last


meta = {
    "issue": "autolens_profiling#429",
    "purpose": "per-host route constants (FFT vs direct_conv curvature arms) on real Aris data",
    "host": socket.gethostname(),
    "platform": platform.platform(),
    "cpu_model": next(
        (
            ln.split(":", 1)[1].strip()
            for ln in open("/proc/cpuinfo")
            if ln.startswith("model name")
        ),
        None,
    )
    if Path("/proc/cpuinfo").exists()
    else None,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "threads": {
        v: os.environ.get(v) for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    },
    "numba_parallel": bool(numba_sparse._numba_parallel()),
    "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
        "scipy": __import__("scipy").__version__,
    },
    "revisions": {m.__name__: git_info(m) for m in (autoarray, autofit, autogalaxy, al)},
    "data": {"dir": str(data_dir), "n_visibilities": n_vis, "t_load_s": t_load},
    "grid": {
        "n_pixels": n_pixels,
        "pixel_scale": pixel_scale,
        "mask_radius": mask_radius,
        "mask_centre": list(mask_centre),
        "unmasked_pixels": int(real_space_mask.pixels_in_mask),
        "extent_ny_nx": list(extent),
        "sparse_batch_size": int(operator.batch_size),
    },
    "preprocessing": {"t_s": t_pre, "cache_hit": cache_hit, "cache_dir": str(cache_dir)},
    "shipped_constants": {
        "c_fft": nutil.FFT_ROUTE_SECONDS_PER_UNIT,
        "c_dc": nutil.DIRECT_CONV_SECONDS_PER_UNIT,
        "kappa": nutil.DIRECT_CONV_SYMMETRIC_WORK_FACTOR,
    },
    "timing": "min over repeats of inv.curvature_matrix with the mapper (and numba index arrays) built",
    "repeats": args.repeats,
    "load_avg_start": list(os.getloadavg()),
    "cells": [],
}
args.out.parent.mkdir(parents=True, exist_ok=True)


def write():
    meta["peak_rss_gb"] = peak_rss_gb()
    meta["load_avg_end"] = list(os.getloadavg())
    args.out.write_text(json.dumps(meta, indent=2))


arms = [a.strip() for a in args.arms.split(",") if a.strip()]
for step in [int(s) for s in args.steps.split(",")]:
    tracer, adapt, n_active, n_vertices = model_for(step)
    log(f"mesh step {step}: {n_active} active + {args.edge_points} edge = {n_vertices} vertices")
    # Geometry + the factory's default route (no curvature evaluated).
    fit = fit_for(tracer, adapt, settings_for(None))
    t0 = time.perf_counter()
    inv_default = fit.inversion
    t_route = time.perf_counter() - t0
    mapper = inv_default.cls_list_from(cls=Mapper)[0]
    S = int(mapper.params)
    nnz = int(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum())
    fit_dc = fit_for(tracer, adapt, settings_for("dc_sym"))
    sri = fit_dc.inversion.symmetric_restriction_inputs
    f = float(sri["work_fraction"])
    W_fft = float(nutil.fft_route_work_from(source_pixels=S, extent_shape=extent))
    W_dc = float(nutil.direct_conv_work_from(nnz=nnz, extent_shape=extent))
    gate = float(settings_for("dc_sym").interferometer_direct_conv_symmetric_max_work_fraction)
    p_fft, p_dc_full = nutil.curvature_route_seconds_from(
        source_pixels=S, nnz=nnz, extent_shape=extent
    )
    _, p_dc_sym = nutil.curvature_route_seconds_from(
        source_pixels=S, nnz=nnz, extent_shape=extent, work_fraction=f if f <= gate else None
    )
    cell = {
        "mesh_step": step,
        "vertices": n_vertices,
        "S": S,
        "nnz": nnz,
        "nnz_per_source": nnz / S,
        "work_fraction": f,
        "symmetric_gate": gate,
        "W_fft": W_fft,
        "W_dc": W_dc,
        "predicted_shipped_s": {"fft": p_fft, "dc_full": p_dc_full, "dc_sym": p_dc_sym},
        "default_route": {
            "inversion_class": type(inv_default).__name__,
            "use_symmetric_kernel": bool(getattr(inv_default, "use_symmetric_kernel", False)),
            "t_route_decision_s": t_route,
        },
        "arms": {},
    }
    del fit, fit_dc, inv_default
    log(
        f"  S={S} nnz={nnz} ({nnz / S:.1f}/col) f={f:.3f} W_fft={W_fft:.3e} W_dc={W_dc:.3e} "
        f"default={cell['default_route']['inversion_class']}"
    )
    curvatures = {}
    for arm in arms:
        reps, curvatures[arm] = time_arm(arm, tracer, adapt)
        cell["arms"][arm] = {
            "reps": reps,
            "t_min_s": min(r["t_curvature_s"] for r in reps),
            "t_max_s": max(r["t_curvature_s"] for r in reps),
        }
        meta["cells"] = [c for c in meta["cells"] if c["mesh_step"] != step] + [cell]
        write()
    ref = curvatures.get("fft")
    if ref is not None:
        scale = float(np.max(np.abs(ref)))
        cell["curvature_max_rel_diff_vs_fft"] = {
            arm: float(np.max(np.abs(c - ref)) / scale)
            for arm, c in curvatures.items()
            if arm != "fft"
        }
    del curvatures
    write()
    log(
        f"  cell done: { {a: round(v['t_min_s'], 3) for a, v in cell['arms'].items()} }; "
        f"peak RSS {peak_rss_gb():.1f} GB"
    )

write()
log(f"wrote {args.out}")
