"""Interferometer W~ curvature on the A100: pruned padded FFT and real-space pixel scale.

PyAutoPulse task ``interferometer_w_tilde_fft_size_levers`` (autolens_profiling#429, campaign
PyAutoPulse#62). On the A100 the sparse-path curvature matrix ``F = Aᵀ W~ A`` is FFT-bound on the
``(B, 2y, 2x)`` ``rfft2`` / ``irfft2`` pair of ``InterferometerSparseOperator.apply_operator``
(PyAutoArray ``inversion/inversion/interferometer/inversion_interferometer_util.py``), 91-92 % of
the jvla likelihood (#320). This measurement-only script (no library change) has two sections:

``prune`` (alma / alma_high / jvla)
    A harness copy of ``curvature_matrix_diag_from`` whose apply is separable and skips the
    padding it does not need. Only the top-left ``y x x`` quadrant of each padded block is
    non-zero and only the top-left ``y x x`` of the output is kept, so:

    1. ``rfft`` along x with ``n = 2x`` on the ``y`` non-zero rows only (the library transforms
       ``2y`` rows, ``y`` of them zero);
    2. ``fft`` along y with ``n = 2y`` (zero-padded);
    3. multiply by ``Khat``;
    4. ``ifft`` along y and keep the first ``y`` rows;
    5. ``irfft`` along x with ``n = 2x`` on those ``y`` rows only, keep the first ``x`` columns.

    Both F builds are jitted separately (lower, compile and steady state timed apart), then
    re-timed as interleaved steady medians (A-B-B-A). Each F also feeds the same jitted
    downstream (positive-only solve, the two log-dets, the figure of merit), and that fused
    pipeline is timed per variant. Recorded: per-call times, the max relative difference of the
    pruned F against the library F and the figure-of-merit shift. Bar (the Pulse task): F max rel
    diff <= 1e-12 and figure of merit <= 1e-6 nats; worth a library change if it saves >= 10 % of
    F at jvla.

``pixel_scale`` (jvla)
    The jvla Delaunay cell re-masked at real-space pixel scales 0.01 / 0.015 / 0.02 / 0.025″
    (mask radius 3.5″ fixed, field 8″ so the grid shape is ``round(8 / scale)``). Per scale:
    the extent, F per call, the library figure of merit as a jitted full call (compile apart) and
    its shift against 0.01″. The adapt image is the truth tracer's lensed source evaluated on
    each scale's grid (not the cached 0.01″ ``lensed_source.fits``), the 0.01″ reference
    included, so every row is built the same way; the Hilbert-1500 mesh is rebuilt on it. The
    science bar for a coarser scale is a 0.5-nat log-evidence shift.

Setup shared with the #320 cells (``scripts/misc/likelihood_breakdown/interferometer_pixelized.py``):
Hilbert-1500 Delaunay with ``AdaptSplit(0.1, 10, 0.1)``, Isothermal + ExternalShear at the prior
medians, sparse operator built with ``method="nufft"``, fp64 (the script refuses x64 off).

    python scripts/interferometer/delaunay/fft_size_levers.py \\
        [--instruments alma,alma_high,jvla] [--pixel-scales 0.01,0.015,0.02,0.025] \\
        [--sections prune,pixel_scale] [--out PATH]

Output (default): ``results/misc/likelihood_breakdown/interferometer/jvla/delaunay_hpc_a100_fp64_fft_prune.json``,
rewritten after every row.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
for _p in (str(ROOT / "scripts" / "misc"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: A100 measurement script; exiting.")
    sys.exit(0)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--sections", default="prune,pixel_scale")
parser.add_argument("--instruments", default="alma,alma_high,jvla")
parser.add_argument("--pixel-scales", default="0.01,0.015,0.02,0.025")
parser.add_argument("--pixel-scale-instrument", default="jvla")
parser.add_argument("--field-arcsec", type=float, default=8.0)
parser.add_argument("--n-repeats", type=int, default=10)
parser.add_argument("--n-timed", type=int, default=30, help="calls per interleaved steady median")
parser.add_argument(
    "--out",
    type=Path,
    default=ROOT
    / "results/misc/likelihood_breakdown/interferometer/jvla/delaunay_hpc_a100_fp64_fft_prune.json",
)
args = parser.parse_args()

try:
    from autonerves import jax_wrapper  # noqa: F401  Configure JAX before PyAutoLens.
except ImportError:  # pragma: no cover
    from autoconf import jax_wrapper  # noqa: F401

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from autoarray.inversion.inversion import inversion_util  # noqa: E402
from autoarray.inversion.mappers import mapper_util  # noqa: E402
from autoarray.inversion.mappers.abstract import Mapper  # noqa: E402
from autofit.jax import register_model as _register_model_pytrees  # noqa: E402
from jax import lax  # noqa: E402
from jax.ops import segment_sum  # noqa: E402
from likelihood_breakdown import interferometer_pixelized as harness  # noqa: E402
from likelihood_breakdown import timing  # noqa: E402
from likelihood_breakdown.provenance import source_revisions  # noqa: E402

from _profile_cli import device_info_dict  # noqa: E402

if not jax.config.x64_enabled:
    raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True.")

CLI = SimpleNamespace(regularization=None)  # AdaptSplit(0.1, 10, 0.1), the #320 default
F_MATCH_BAR = 1e-12
FOM_MATCH_BAR_NATS = 1e-6
SCIENCE_BAR_NATS = 0.5
T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def free():
    jax.clear_caches()
    gc.collect()


def arr(x):
    return jnp.asarray(getattr(x, "array", x))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


RESULT: dict = {
    "issue": "autolens_profiling#429",
    "pulse_task": "interferometer_w_tilde_fft_size_levers",
    "campaign": "PyAutoPulse#62",
    "precision": "fp64",
    "revisions": source_revisions(ROOT),
    # A git-archive checkout has no .git: its REVISION files (tag + SHA) are the provenance.
    "stack_revisions": {
        r: (Path(os.environ["FFTLEV_STACK"]) / r / "REVISION").read_text().strip()
        for r in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens")
        if os.environ.get("FFTLEV_STACK")
        and (Path(os.environ["FFTLEV_STACK"]) / r / "REVISION").is_file()
    },
    "autolens_profiling_revision": (ROOT / "REVISION").read_text().strip()
    if (ROOT / "REVISION").is_file()
    else None,
    "jax_version": jax.__version__,
    "jaxlib_version": __import__("jaxlib").__version__,
    "nufftax_version": harness._nufftax_version(),
    "device": device_info_dict(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "bars": {
        "f_max_rel_diff": F_MATCH_BAR,
        "figure_of_merit_nats": FOM_MATCH_BAR_NATS,
        "science_log_evidence_nats": SCIENCE_BAR_NATS,
        "library_change_if_f_saving_at_jvla_ge": 0.10,
    },
    "prune": {},
    "pixel_scale": {},
}


def write():
    RESULT["peak_rss_mb"] = harness._peak_rss_mb()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(RESULT, indent=2, default=float))


def load(instrument, pixel_scale=None):
    """The instrument's dataset, optionally re-masked at ``pixel_scale`` (radius kept)."""
    from simulators.interferometer import INSTRUMENTS

    from instruments.interferometer import transformer_chunk_size_for

    preset = INSTRUMENTS[instrument]
    scale = preset["pixel_scale"] if pixel_scale is None else float(pixel_scale)
    if pixel_scale is None:
        shape = tuple(preset["real_space_shape"])
    else:
        n = int(round(args.field_arcsec / scale))
        shape = (n + n % 2, n + n % 2)
    mask = al.Mask2D.circular(shape_native=shape, pixel_scales=scale, radius=preset["mask_radius"])
    chunk = transformer_chunk_size_for(instrument) or (
        100_000 if preset["n_visibilities"] > harness.LARGE_VIS else None
    )
    path = ROOT / "dataset" / "interferometer" / instrument

    def transformer(uv_wavelengths, real_space_mask):
        return al.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask, chunk_size=chunk
        )

    t = time.perf_counter()
    dataset = al.Interferometer.from_fits(
        data_path=path / "data.fits",
        noise_map_path=path / "noise_map.fits",
        uv_wavelengths_path=path / "uv_wavelengths.fits",
        real_space_mask=mask,
        transformer_class=transformer,
    )
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    sparse = dataset.apply_sparse_operator(method="nufft", nufft_chunk_size=chunk, use_jax=True)
    t_op = time.perf_counter() - t
    op = sparse.sparse_operator
    info = {
        "dataset": str(path.relative_to(ROOT)),
        "pixel_scale_arcsec": scale,
        "real_space_shape": list(shape),
        "mask_radius_arcsec": preset["mask_radius"],
        "visibilities": int(dataset.uv_wavelengths.shape[0]),
        "image_pixels_masked": int(mask.pixels_in_mask),
        "operator_extent_shape": [int(op.y_shape), int(op.x_shape)],
        "operator_fft_shape": [2 * int(op.y_shape), 2 * int(op.x_shape)],
        "sparse_batch_size": int(op.batch_size),
        "transformer_chunk_size": chunk,
        "t_load_s": t_load,
        "operator_build_s": t_op,
    }
    log(
        f'{instrument} @ {scale}": extent {op.y_shape}x{op.x_shape}, {info["visibilities"]} vis, '
        f"load {t_load:.1f}s, operator {t_op:.1f}s"
    )
    return path, dataset, sparse, info


def model_and_fit(path, dataset, sparse, adapt_image):
    mesh = harness.build_delaunay_mesh(CLI, dataset, adapt_image, None)
    model = harness.build_model(mesh)
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    _register_model_pytrees(model)
    params = jax.tree_util.tree_map(jnp.asarray, instance)
    adapt_for = harness.adapt_images_builder(adapt_image, mesh)
    settings = al.Settings(use_mixed_precision=False, positive_only_solver="pdip")

    def fit_from(p):
        return al.FitInterferometer(
            dataset=sparse,
            tracer=al.Tracer(galaxies=list(p.galaxies), fields=[p.fields]),
            adapt_images=adapt_for(p.galaxies.source),
            settings=settings,
            xp=jnp,
        )

    return mesh, params, fit_from, settings


def reference(fit_from, params):
    info = {}

    def quantities(p):
        fit = fit_from(p)
        inv = fit.inversion
        mapper = inv.cls_list_from(cls=Mapper)[0]
        info["inversion_class"] = type(inv).__name__
        return {
            "pix_indexes": arr(mapper.pix_indexes_for_sub_slim_index),
            "pix_weights": arr(mapper.pix_weights_for_sub_slim_index),
            "slim_index_for_sub": arr(mapper.slim_index_for_sub_slim_index),
            "sub_fraction": arr(mapper.over_sampler.sub_fraction),
            "mapping_matrix": arr(inv.mapping_matrix),
            "data_vector": arr(inv.data_vector),
            "curvature_matrix": arr(inv.curvature_matrix),
            "regularization_matrix": arr(inv.regularization_matrix),
            "log_det_curvature_reg": inv.log_det_curvature_reg_matrix_term,
            "log_det_regularization": inv.log_det_regularization_matrix_term,
            "figure_of_merit": fit.figure_of_merit,
        }

    ref = jax.jit(quantities)(params)
    ref = jax.tree_util.tree_map(lambda x: x.block_until_ready(), ref)
    if info["inversion_class"] != "InversionInterferometerSparse":
        raise SystemExit(f"expected InversionInterferometerSparse, got {info}")
    return ref


def triplets(sparse, ref):
    extent_index = jnp.asarray(
        np.asarray(sparse.transformer.real_space_mask.extent_index_for_masked_pixel),
        dtype=jnp.int32,
    )
    rows, cols, vals = mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=ref["pix_indexes"],
        pix_weights_for_sub=ref["pix_weights"],
        slim_index_for_sub=ref["slim_index_for_sub"],
        fft_index_for_masked_pixel=extent_index,
        sub_fraction_slim=ref["sub_fraction"],
        return_rows_slim=False,
        xp=jnp,
    )
    return rows, cols, vals


def curvature_library(op, S):
    def fn(rows, cols, vals):
        return op.curvature_matrix_diag_from(rows=rows, cols=cols, vals=vals, S=S, xp=jnp)

    return fn


def apply_pruned(op, F_flat):
    """``W~ F`` by the separable pruned transform (see the module docstring)."""
    y, x = op.y_shape, op.x_shape
    B = F_flat.shape[1]
    F_img = F_flat.T.reshape((B, y, x))
    X = jnp.fft.rfft(F_img, n=2 * x, axis=-1)  # (B, y, x + 1): y rows only
    X = jnp.fft.fft(X, n=2 * y, axis=-2)  # (B, 2y, x + 1)
    X = X * op.Khat[None, :, :]
    Y = jnp.fft.ifft(X, axis=-2)[:, :y, :]  # keep y rows
    G = jnp.fft.irfft(Y, n=2 * x, axis=-1)[:, :, :x]  # y rows only, keep x columns
    return G.reshape((B, y * x)).T


def curvature_pruned(op, S):
    """``curvature_matrix_diag_from`` (JAX branch) with ``apply_pruned`` as the apply."""
    M, B = op.M, op.batch_size
    n_blocks = (S + B - 1) // B

    def fn(rows, cols, vals):
        rows = jnp.asarray(rows, dtype=jnp.int32)
        cols = jnp.asarray(cols, dtype=jnp.int32)
        vals = jnp.asarray(vals, dtype=jnp.float64)
        C0 = jnp.zeros((S, n_blocks * B), dtype=jnp.float64)

        def body(block_i, C):
            start = block_i * B
            in_block = (cols >= start) & (cols < (start + B))
            bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals, 0.0)
            F = jnp.zeros((M, B), dtype=jnp.float64).at[rows, bc].add(v)
            G = apply_pruned(op, F)
            Cblock = segment_sum(vals[:, None] * G[rows, :], cols, num_segments=S)
            width = jnp.minimum(B, jnp.maximum(0, S - start))
            Cblock = Cblock * (op.col_offsets < width)[None, :]
            return lax.dynamic_update_slice(C, Cblock, (0, start))

        C = lax.fori_loop(0, n_blocks, body, C0)[:, :S]
        return 0.5 * (C + C.T)

    return fn


def downstream(settings, S):
    """F -> positive-only solve -> log-dets -> figure of merit (the #320 harness steps 8-10)."""

    def fn(F, D, H, vis, noise):
        creg = F + H
        recon = inversion_util.reconstruction_positive_only_from(
            data_vector=D,
            curvature_reg_matrix=creg,
            settings=settings,
            xp=jnp,
            solver="pdip",
            stats={},
            preconditioning="jacobi",
        )
        ld_creg = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(creg))))
        ld_reg = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(H))))
        chi2 = (
            recon @ F @ recon
            - 2.0 * recon @ D
            + jnp.sum(vis.real**2 / noise.real**2)
            + jnp.sum(vis.imag**2 / noise.imag**2)
        )
        norm = jnp.sum(jnp.log(2 * jnp.pi * noise.real**2)) + jnp.sum(
            jnp.log(2 * jnp.pi * noise.imag**2)
        )
        return -0.5 * (chi2 + recon @ H @ recon + ld_creg - ld_reg + norm)

    return fn


def jit_row(func, label, *fargs):
    timer = timing.Timer()
    records: dict = {}
    _, out = timing.jit_profile(
        func, label, *fargs, n_repeats=args.n_repeats, timer=timer, jit_records=records
    )
    return out, records[label]


def interleaved(funcs: dict, *fargs):
    """Steady medians in A-B-B-A order over already-compiled functions."""
    compiled = {k: jax.jit(f).lower(*fargs).compile() for k, f in funcs.items()}
    keys = list(compiled)
    order = keys + keys[::-1]
    out = {k: [] for k in keys}
    for k in order:
        out[k].append(
            timing.steady_median_profile(compiled[k], *fargs, n_warm=5, n_timed=args.n_timed)
        )
    return {
        k: {"median_s": float(np.median([r["median_s"] for r in v])), "rounds": v}
        for k, v in out.items()
    }


def prune_row(instrument):
    from _adapt_image_util import adapt_image_for_dataset

    path, dataset, sparse, info = load(instrument)
    op = sparse.sparse_operator
    adapt_image = adapt_image_for_dataset(dataset_path=path, dataset=dataset)
    mesh, params, fit_from, settings = model_and_fit(path, dataset, sparse, adapt_image)
    S = int(mesh.n_source_pixels)
    ref = reference(fit_from, params)
    free()
    rows, cols, vals = triplets(sparse, ref)
    lib, prn = curvature_library(op, S), curvature_pruned(op, S)
    F_lib, rec_lib = jit_row(lib, "F_library", rows, cols, vals)
    free()
    F_prn, rec_prn = jit_row(prn, "F_pruned", rows, cols, vals)
    free()
    inter = interleaved({"F_library": lib, "F_pruned": prn}, rows, cols, vals)
    free()
    D, H = ref["data_vector"], ref["regularization_matrix"]
    vis, noise = jnp.asarray(dataset.data.array), jnp.asarray(dataset.noise_map.array)
    post = downstream(settings, S)

    def fused_with(curv):
        def fn(rows, cols, vals, D, H, vis, noise):
            return post(curv(rows, cols, vals), D, H, vis, noise)

        return fn

    fom_lib, rec_fused_lib = jit_row(
        fused_with(lib), "fused_library", rows, cols, vals, D, H, vis, noise
    )
    free()
    fom_prn, rec_fused_prn = jit_row(
        fused_with(prn), "fused_pruned", rows, cols, vals, D, H, vis, noise
    )
    free()
    inter_fused = interleaved(
        {"fused_library": fused_with(lib), "fused_pruned": fused_with(prn)},
        rows,
        cols,
        vals,
        D,
        H,
        vis,
        noise,
    )
    free()
    f_lib_med, f_prn_med = inter["F_library"]["median_s"], inter["F_pruned"]["median_s"]
    row = {
        **info,
        "source_pixels": S,
        "mesh": mesh.configuration,
        "regularization": mesh.regularization,
        "nnz_triplets": int(rows.shape[0]),
        "curvature_blocks": int(-(-S // op.batch_size)),
        "F_library": rec_lib,
        "F_pruned": rec_prn,
        "F_interleaved": inter,
        "F_saving_fraction": 1.0 - f_prn_med / f_lib_med,
        "F_speedup": f_lib_med / f_prn_med,
        "F_max_rel_diff": rel(F_prn, F_lib),
        "F_library_vs_reference_max_rel_diff": rel(F_lib, ref["curvature_matrix"]),
        "fused_library": rec_fused_lib,
        "fused_pruned": rec_fused_prn,
        "fused_interleaved": inter_fused,
        "fused_speedup": inter_fused["fused_library"]["median_s"]
        / inter_fused["fused_pruned"]["median_s"],
        "figure_of_merit_library_fit": float(ref["figure_of_merit"]),
        "figure_of_merit_fused_library_F": float(fom_lib),
        "figure_of_merit_fused_pruned_F": float(fom_prn),
        "figure_of_merit_shift_nats": abs(float(fom_prn) - float(fom_lib)),
        "figure_of_merit_steps_vs_library_fit_nats": abs(
            float(fom_lib) - float(ref["figure_of_merit"])
        ),
    }
    row["passes_match_bars"] = bool(
        row["F_max_rel_diff"] <= F_MATCH_BAR
        and row["figure_of_merit_shift_nats"] <= FOM_MATCH_BAR_NATS
    )
    log(
        f"{instrument}: F lib {f_lib_med * 1e3:.2f} ms, pruned {f_prn_med * 1e3:.2f} ms "
        f"(saving {row['F_saving_fraction']:.1%}); F rel {row['F_max_rel_diff']:.2e}; "
        f"dFoM {row['figure_of_merit_shift_nats']:.2e}; fused x{row['fused_speedup']:.3f}"
    )
    del dataset, sparse, ref, F_lib, F_prn
    free()
    return row


def pixel_scale_row(instrument, scale):
    path, dataset, sparse, info = load(instrument, pixel_scale=scale)
    op = sparse.sparse_operator
    tracer = al.from_json(file_path=path / "tracer.json")
    adapt_image = tracer.image_2d_list_from(grid=dataset.grid)[-1]
    mesh, params, fit_from, _ = model_and_fit(path, dataset, sparse, adapt_image)
    S = int(mesh.n_source_pixels)

    def full(p):
        return fit_from(p).figure_of_merit

    fom, rec_full = jit_row(full, "full_figure_of_merit", params)
    free()
    ref = reference(fit_from, params)
    free()
    rows, cols, vals = triplets(sparse, ref)
    _, rec_F = jit_row(curvature_library(op, S), "F_library", rows, cols, vals)
    free()

    def terms(p):
        fit = fit_from(p)
        inv = fit.inversion
        return {
            "figure_of_merit": fit.figure_of_merit,
            "log_det_curvature_reg": inv.log_det_curvature_reg_matrix_term,
            "log_det_regularization": inv.log_det_regularization_matrix_term,
            "regularization_term": inv.regularization_term,
            "fast_chi_squared": inv.fast_chi_squared,
        }

    t = jax.jit(terms)(params)
    t = {k: float(v) for k, v in t.items()}
    row = {
        **info,
        "source_pixels": S,
        "mesh": mesh.configuration,
        "adapt_image": "truth tracer lensed source on this scale's grid",
        "full_figure_of_merit": rec_full,
        "F_library": rec_F,
        "F_share_of_full": rec_F["steady_per_call_s"] / rec_full["steady_per_call_s"],
        "figure_of_merit": float(fom),
        "terms": t,
    }
    log(
        f'{instrument} @ {scale}": F {rec_F["steady_per_call_s"] * 1e3:.1f} ms, full '
        f"{rec_full['steady_per_call_s'] * 1e3:.1f} ms, FoM {float(fom):.4f}"
    )
    del dataset, sparse, ref
    free()
    return row


sections = [s.strip() for s in args.sections.split(",") if s.strip()]
if "prune" in sections:
    for instrument in [i.strip() for i in args.instruments.split(",") if i.strip()]:
        try:
            RESULT["prune"][instrument] = prune_row(instrument)
        except Exception as exc:  # noqa: BLE001  An OOM or failure is a recorded result.
            RESULT["prune"][instrument] = {"status": "error", "error": str(exc)[:500]}
            log(f"{instrument} prune failed: {exc!r}"[:600])
            free()
        write()

if "pixel_scale" in sections:
    instrument = args.pixel_scale_instrument
    rows_ps: dict = {}
    for scale in [float(s) for s in args.pixel_scales.split(",")]:
        try:
            rows_ps[f"{scale:g}"] = pixel_scale_row(instrument, scale)
        except Exception as exc:  # noqa: BLE001
            rows_ps[f"{scale:g}"] = {"status": "error", "error": str(exc)[:500]}
            log(f"{instrument} @ {scale} failed: {exc!r}"[:600])
            free()
        ref_row = rows_ps.get(f"{min(float(s) for s in args.pixel_scales.split(',')):g}", {})
        for r in rows_ps.values():
            if "figure_of_merit" in r and "figure_of_merit" in ref_row:
                r["log_evidence_shift_vs_reference_nats"] = (
                    r["figure_of_merit"] - ref_row["figure_of_merit"]
                )
                r["within_science_bar"] = (
                    abs(r["log_evidence_shift_vs_reference_nats"]) <= SCIENCE_BAR_NATS
                )
                r["F_speedup_vs_reference"] = (
                    ref_row["F_library"]["steady_per_call_s"] / r["F_library"]["steady_per_call_s"]
                )
        RESULT["pixel_scale"] = {
            "instrument": instrument,
            "reference_scale": ref_row.get("pixel_scale_arcsec"),
            "rows": rows_ps,
        }
        write()

RESULT["device_end"] = device_info_dict()
write()
log(f"wrote {args.out}")
