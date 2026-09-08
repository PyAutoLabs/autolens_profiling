"""
Synthetic curvature-kernel bake-off — numba w-tilde vs FFT, interferometer
==========================================================================

Phase 2, step 1 of the ``numba-interferometer-revisit`` epic. Times every way of
assembling ``F = Mᵀ W~ M`` for an interferometer at three real geometries, with no
PyAutoLens stack in the loop: the only library call is ``aa.Mask2D.circular``, used to
get the same masked-pixel set and unmasked extent an ``sma`` / ``alma`` / ``alma_high``
dataset has. The ``W~`` preload and the mapper triplets are synthetic, seeded, and
**shared by every kernel**, so a timing difference is a kernel difference.

Why a synthetic bake-off is the right instrument
------------------------------------------------
``W~ = Re(Fᴴ W F)`` has no compact support for an interferometer, so the recovered
scatter kernel is ``O(N² P²)`` and the FFT convolution is ``O(S · M log M)`` with
``M = Ny·Nx`` the unmasked extent. Which wins is a function of the *geometry* alone —
``N``, ``M``, ``S`` and the mapper's ``P`` — not of the data values. Building the real
datasets to answer it would cost the ``O(N·K)`` preload (10–15 minutes at alma) per arm
for information the geometry already determines. The in-situ runs (step 3) then confirm
the synthetic ordering on the real likelihood.

Parity before timing
--------------------
Every kernel is pinned to the phase-1 reference kernel on ``F`` at ``rtol=1e-10`` with
``atol = 1e-10 · max|F_ref|`` before any of its timings are recorded; an unpinned
kernel's number is reported as ``null`` with its pin failure. A control multiplies one
kernel's output by 1.01 and must fail the same pin.

Run
---
    python scripts/misc/numba_interferometer/bakeoff.py                 # full bake-off
    python scripts/misc/numba_interferometer/bakeoff.py --mode threads  # step 2

Single-thread arms need ``OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1``; the recorded environment is written into the JSON.
"""

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_misc_dir = str(_ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
_sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import autoarray as aa  # noqa: E402
import numpy as np  # noqa: E402

from instruments.interferometer import INSTRUMENTS  # noqa: E402
from numba_interferometer import inversion_interferometer_numba_util as ref_util  # noqa: E402
from numba_interferometer import kernels as K  # noqa: E402

GEOMETRIES = ("sma", "alma", "alma_high")

MESHES = {
    # Delaunay-like: three vertices per image pixel, 1500-tier production fiducial.
    "delaunay": {"pix_pixels": 1500, "p_per_pixel": 3},
    # Rectangular bilinear: four corners per image pixel, 28x28 production fiducial.
    "rect": {"pix_pixels": 784, "p_per_pixel": 4},
}

KERNEL_ORDER = (
    "reference",
    "hoisted",
    "symmetric",
    "two_stage",
    "direct_conv",
    "source_loop",
    "rfft2_numpy",
    "fft2_numpy",
    "fft2_jax",
)

NUMBA_KERNELS = ("reference", "hoisted", "symmetric", "two_stage", "direct_conv", "source_loop")

PIN_RTOL = 1e-10

# At alma_high the O(N^2 P^2) pair-loop kernels are extrapolated from alma first and
# only run (once) if the estimate is under this many seconds; otherwise the cell is
# recorded as `skipped: extrapolated` with the estimate.
ALMA_HIGH_PAIR_LOOP_BUDGET_S = 180.0

# Source-column block width. 128 is `apply_sparse_operator`'s default `batch_size`, so
# every FFT arm assembles F in the same block structure the library does.
BATCH_SIZE = 128


# ===================================================================
# Synthetic inputs
# ===================================================================


def geometry_from(instrument: str) -> dict:
    """The masked-pixel set and unmasked extent of one instrument's real-space mask."""
    preset = INSTRUMENTS[instrument]

    mask = aa.Mask2D.circular(
        shape_native=preset["real_space_shape"],
        pixel_scales=preset["pixel_scale"],
        radius=preset["mask_radius"],
    )

    native = np.asarray(mask.derive_indexes.native_for_slim).astype(np.int64)

    return {
        "instrument": instrument,
        "native_index_for_slim_index": native,
        "n_pix": int(native.shape[0]),
        "extent_shape": tuple(int(s) for s in mask.shape_native_masked_pixels),
        "pixel_scale": preset["pixel_scale"],
        "mask_radius": preset["mask_radius"],
        "real_space_shape": tuple(int(s) for s in preset["real_space_shape"]),
        "n_visibilities": int(preset["n_visibilities"]),
    }


def synthetic_preload(ny: int, nx: int, seed: int = 20260907) -> np.ndarray:
    """A seeded ``(2Ny, 2Nx)`` offset array with the two properties ``W~`` has.

    ``W~[i,j] = Σ_k σ_k⁻² cos(2π(Δx·u_k + Δy·v_k))`` is (i) even in the offset and
    (ii) the real inverse transform of a non-negative spectrum, hence positive
    semi-definite as a circulant operator. Both are reproduced exactly here by taking
    the autocorrelation of a smoothed seeded random field: ``spec = |FFT(g)|² ≥ 0`` and
    ``k = IFFT(spec)`` is real and even. The values themselves are irrelevant to the
    timings — every kernel does the same arithmetic whatever they are — but getting the
    symmetry right matters, because the ``symmetric`` kernel exploits it.
    """
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal((2 * ny, 2 * nx))

    fy = np.fft.fftfreq(2 * ny)[:, None]
    fx = np.fft.fftfreq(2 * nx)[None, :]
    envelope = np.exp(-0.5 * (fy**2 + fx**2) / (0.05**2))

    field = np.fft.ifft2(np.fft.fft2(noise) * envelope).real
    spectrum = np.abs(np.fft.fft2(field)) ** 2
    preload = np.fft.ifft2(spectrum).real

    # Enforce evenness exactly (the transform above is even to ~1e-17 already).
    preload = 0.5 * (preload + np.roll(preload[::-1, ::-1], shift=(1, 1), axis=(0, 1)))

    return np.ascontiguousarray(preload)


def synthetic_mapper(n_pix: int, pix_pixels: int, p_per_pixel: int, seed: int = 314159):
    """Seeded COO triplets: ``P`` source pixels per image pixel, weights summing to 1.

    Every source pixel is guaranteed at least one mapping (the first ``S`` image pixels
    are assigned round-robin) so no column of ``A`` is empty — an empty column would
    make ``F`` singular and would silently shrink the source-loop kernel's work.
    """
    rng = np.random.default_rng(seed)

    indexes = rng.integers(0, pix_pixels, size=(n_pix, p_per_pixel)).astype(np.int64)
    indexes[:pix_pixels, 0] = np.arange(pix_pixels, dtype=np.int64)

    weights = rng.random((n_pix, p_per_pixel)) + 0.05
    weights /= weights.sum(axis=1, keepdims=True)

    sizes = np.full(n_pix, p_per_pixel, dtype=np.int64)

    return indexes, sizes, weights


# ===================================================================
# Kernel callables
# ===================================================================


def kernel_callables(preload, inputs, indexes, sizes, weights, native, pix_pixels, *, workers=1):
    """One zero-argument callable per kernel, all closing over the same inputs."""
    iy, ix = inputs["iy"], inputs["ix"]
    flat, rows_nnz = inputs["flat"], inputs["rows_nnz"]
    indptr, col, val = inputs["indptr"], inputs["col"], inputs["val"]
    cscptr, csc_row, csc_val = inputs["cscptr"], inputs["csc_row"], inputs["csc_val"]
    ny, nx = inputs["ny"], inputs["nx"]

    sub_indexes, sub_sizes, sub_weights = K.source_loop_inputs_from(
        indexes, sizes, weights, pix_pixels
    )

    sparse_operator = K.sparse_operator_from(rows_nnz, col, val, ny * nx, pix_pixels)

    calls = {
        "reference": lambda: (
            ref_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
                curvature_preload=preload,
                pix_indexes_for_sub_slim_index=indexes,
                pix_size_for_sub_slim_index=sizes,
                pix_weights_for_sub_slim_index=weights,
                native_index_for_slim_index=native,
                pix_pixels=pix_pixels,
            )
        ),
        "hoisted": lambda: K.curvature_hoisted(preload, iy, ix, indptr, col, val, pix_pixels),
        "symmetric": lambda: K.curvature_symmetric(preload, iy, ix, indptr, col, val, pix_pixels),
        "two_stage": lambda: K.curvature_two_stage(preload, iy, ix, indptr, col, val, pix_pixels),
        "direct_conv": lambda: K.curvature_direct_conv(
            preload, iy, ix, flat, indptr, col, val, cscptr, csc_row, csc_val, ny, nx, pix_pixels
        ),
        "source_loop": lambda: (
            ref_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from_2(
                curvature_preload=preload,
                native_index_for_slim_index=native,
                pix_pixels=pix_pixels,
                sub_slim_indexes_for_pix_index=sub_indexes,
                sub_slim_sizes_for_pix_index=sub_sizes,
                sub_slim_weights_for_pix_index=sub_weights,
            )
        ),
        "rfft2_numpy": lambda: K.curvature_fft_numpy(
            preload,
            rows_nnz,
            col,
            val,
            ny,
            nx,
            pix_pixels,
            real_fft=True,
            batch_size=BATCH_SIZE,
            workers=workers,
            sparse_operator=sparse_operator,
        ),
        "fft2_numpy": lambda: K.curvature_fft_numpy(
            preload,
            rows_nnz,
            col,
            val,
            ny,
            nx,
            pix_pixels,
            real_fft=False,
            batch_size=BATCH_SIZE,
            workers=workers,
            sparse_operator=sparse_operator,
        ),
        "fft2_jax": K.jax_curvature_callable(
            preload, rows_nnz, col, val, ny, nx, pix_pixels, batch_size=BATCH_SIZE
        ),
    }

    return calls


def operation_counts(inputs, pix_pixels, p_per_pixel) -> dict:
    """Analytic per-call operation-count estimates (multiply-accumulates)."""
    n_pix = inputs["n_pix"]
    nnz = inputs["nnz"]
    m_cells = inputs["ny"] * inputs["nx"]
    s = pix_pixels
    p = p_per_pixel

    grid = 4 * m_cells  # the padded (2Ny, 2Nx) FFT grid
    n_blocks = math.ceil(s / BATCH_SIZE)
    fft_butterflies = n_blocks * BATCH_SIZE * grid * math.log2(grid)

    return {
        "reference": n_pix * n_pix * p * p,
        "hoisted": n_pix * n_pix * p * p,
        "symmetric": n_pix * n_pix * p * p / 2,
        "two_stage": n_pix * n_pix * p + n_pix * p * s,
        "direct_conv": nnz * m_cells + s * nnz,
        "source_loop": nnz * nnz / 2,
        # Forward + inverse transform per block, plus the A^T projection.
        "rfft2_numpy": fft_butterflies + n_blocks * nnz * BATCH_SIZE,
        "fft2_numpy": 2 * fft_butterflies + n_blocks * nnz * BATCH_SIZE,
        "fft2_jax": 2 * fft_butterflies + n_blocks * nnz * BATCH_SIZE,
    }


# ===================================================================
# Pin + timing
# ===================================================================


def pin_result(candidate, anchor) -> dict:
    """``rtol=1e-10`` with ``atol`` scaled by ``max|F_anchor|`` — the brief's pin."""
    candidate = np.asarray(candidate, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)

    scale = float(np.abs(anchor).max())
    atol = PIN_RTOL * scale

    diff = float(np.abs(candidate - anchor).max())
    passed = bool(np.allclose(candidate, anchor, rtol=PIN_RTOL, atol=atol))

    return {
        "passed": passed,
        "max_abs_diff": diff,
        "max_abs_diff_over_scale": diff / scale if scale > 0 else float("inf"),
        "rtol": PIN_RTOL,
        "atol": atol,
    }


def dgemm_control_s(n: int = 1500, repeats: int = 3) -> float:
    """A fixed 1500x1500 BLAS ``dgemm``, timed at the head and tail of every cell.

    This laptop (i9-10885H) drops from turbo to something near base clock after a few
    minutes of saturated single-core work, and the bake-off runs for the best part of an
    hour: absolute seconds early in the run are not comparable with absolute seconds late
    in it. Round-robin interleaving already protects the *ratios* — every kernel in a
    round sees the same clock — and this control measures the drift the ratios are immune
    to, so the JSON says how much of it there was rather than leaving the reader to guess.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))

    a @ b

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        samples.append(time.perf_counter() - start)

    return float(statistics.median(samples))


def env_record() -> dict:
    keys = (
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
    )
    record = {k: os.environ.get(k) for k in keys}
    record["numba_threads_effective"] = None
    try:
        import numba

        record["numba_threads_effective"] = int(numba.get_num_threads())
        record["numba_version"] = numba.__version__
    except Exception:  # pragma: no cover - diagnostics only
        pass
    try:
        import scipy

        record["scipy_version"] = scipy.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import jax

        record["jax_version"] = jax.__version__
        record["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:  # pragma: no cover
        pass
    record["numpy_version"] = np.__version__
    record["platform"] = platform.platform()
    record["processor"] = platform.processor()
    record["cpu_count"] = os.cpu_count()
    return record


# ===================================================================
# The bake-off
# ===================================================================


def run_cell(instrument, mesh_name, kernel_names, reps, *, alma_estimates, workers=1, verbose=True):
    """One (geometry x mesh) cell: pin every kernel, then time them round-robin."""
    geometry = geometry_from(instrument)
    mesh = MESHES[mesh_name]
    pix_pixels = mesh["pix_pixels"]
    p_per_pixel = mesh["p_per_pixel"]

    native = geometry["native_index_for_slim_index"]
    n_pix = geometry["n_pix"]

    indexes, sizes, weights = synthetic_mapper(n_pix, pix_pixels, p_per_pixel)
    inputs = K.kernel_inputs_from(indexes, sizes, weights, native, pix_pixels)

    ny, nx = inputs["ny"], inputs["nx"]
    preload = synthetic_preload(ny, nx)

    calls = kernel_callables(
        preload, inputs, indexes, sizes, weights, native, pix_pixels, workers=workers
    )
    ops = operation_counts(inputs, pix_pixels, p_per_pixel)

    # --- which kernels can be afforded here -------------------------------
    #
    # The O(N^2 P^2) pair-loop kernels scale as N^2 between geometries. At alma_high
    # they are extrapolated from the alma measurement first and only run (once) when
    # the estimate fits the budget; otherwise the cell records the estimate and says
    # so, rather than reporting a number nobody waited for.
    scheduled, skipped = [], {}
    for name in kernel_names:
        estimate = None
        if instrument == "alma_high" and name in (
            "reference",
            "hoisted",
            "symmetric",
            "source_loop",
        ):
            base = alma_estimates.get((mesh_name, name))
            if base is None:
                skipped[name] = {
                    "status": "skipped: no alma measurement to extrapolate from",
                    "extrapolated_seconds": None,
                    "scaling": "O(N^2) in masked pixels",
                    "budget_seconds": ALMA_HIGH_PAIR_LOOP_BUDGET_S,
                }
                continue
            estimate = base * (n_pix / alma_estimates["n_pix"]) ** 2
            if estimate > ALMA_HIGH_PAIR_LOOP_BUDGET_S:
                skipped[name] = {
                    "status": "skipped: extrapolated",
                    "extrapolated_seconds": estimate,
                    "scaling": "O(N^2) in masked pixels, from the alma measurement",
                    "budget_seconds": ALMA_HIGH_PAIR_LOOP_BUDGET_S,
                }
                continue
        scheduled.append(name)

    # Fewer rounds where a single call is minutes long.
    rounds = {}
    for name in scheduled:
        if instrument == "alma_high":
            rounds[name] = 1 if name in ("reference", "hoisted", "symmetric", "source_loop") else 4
        else:
            rounds[name] = reps

    if verbose:
        print(f"\n=== {instrument} / {mesh_name} ===")
        print(
            f"  masked pixels N = {n_pix}, extent {ny}x{nx} (M = {ny * nx}), "
            f"S = {pix_pixels}, P = {p_per_pixel}, nnz = {inputs['nnz']}"
        )
        if skipped:
            print(f"  skipped (extrapolated): {sorted(skipped)}")

    control_dgemm_head_s = dgemm_control_s()

    # --- pin every kernel against the reference (or the anchor) -----------
    results = {}
    anchor_name = "reference" if "reference" in scheduled else "direct_conv"

    if verbose:
        print(f"  pin anchor: {anchor_name}")

    anchor_start = time.perf_counter()
    anchor = np.asarray(calls[anchor_name]())
    anchor_first_call_s = time.perf_counter() - anchor_start

    pins = {
        anchor_name: {
            "passed": True,
            "max_abs_diff": 0.0,
            "max_abs_diff_over_scale": 0.0,
            "rtol": PIN_RTOL,
            "atol": 0.0,
            "note": "anchor",
        }
    }

    for name in scheduled:
        if name == anchor_name:
            continue
        if verbose:
            print(f"  pinning {name} ...", flush=True)
        pins[name] = pin_result(np.asarray(calls[name]()), anchor)
        if verbose:
            print(
                f"    {'PASS' if pins[name]['passed'] else 'FAIL'} "
                f"(max|d|/scale = {pins[name]['max_abs_diff_over_scale']:.2e})"
            )

    control = pin_result(1.01 * anchor, anchor)
    control_record = {
        "kernel": anchor_name,
        "scale": 1.01,
        "pin_passed": control["passed"],
        "max_abs_diff_over_scale": control["max_abs_diff_over_scale"],
        "non_vacuous": not control["passed"],
    }

    # --- timing, round-robin over kernels (laptop throttling) -------------
    timings = {name: [] for name in scheduled}
    max_rounds = max(rounds.values()) if rounds else 0

    for round_index in range(max_rounds):
        for name in scheduled:
            if round_index >= rounds[name]:
                continue
            if not pins[name]["passed"]:
                continue
            start = time.perf_counter()
            calls[name]()
            timings[name].append(time.perf_counter() - start)
        if verbose:
            done = {n: (f"{timings[n][-1]:.4f}" if timings[n] else "-") for n in scheduled}
            print(f"  round {round_index}: {done}", flush=True)

    for name in scheduled:
        samples = timings[name]
        if not pins[name]["passed"]:
            results[name] = {
                "status": "unpinned",
                "median_s": None,
                "pin": pins[name],
                "operations": ops[name],
            }
            continue
        # Round 0 is discarded (numba compile / cache warm / first-touch pages), unless
        # it is the only round there was.
        timed = samples[1:] if len(samples) > 1 else samples
        results[name] = {
            "status": "measured" if len(samples) > 1 else "measured: single round",
            "median_s": statistics.median(timed) if timed else None,
            "min_s": min(timed) if timed else None,
            "all_rounds_s": samples,
            "rounds_timed": len(timed),
            "rounds_discarded": 1 if len(samples) > 1 else 0,
            "pin": pins[name],
            "operations": ops[name],
        }

    control_dgemm_tail_s = dgemm_control_s()

    for name, record in skipped.items():
        record["median_s"] = None
        record["operations"] = ops[name]
        results[name] = record

    return {
        "geometry": {
            "instrument": instrument,
            "pixel_scale_arcsec": geometry["pixel_scale"],
            "mask_radius_arcsec": geometry["mask_radius"],
            "real_space_shape": list(geometry["real_space_shape"]),
            "image_pixels_masked": n_pix,
            "extent_shape": [ny, nx],
            "extent_cells": ny * nx,
            "visibilities": geometry["n_visibilities"],
        },
        "mesh": {
            "name": mesh_name,
            "source_pixels": pix_pixels,
            "p_per_pixel": p_per_pixel,
            "nnz": inputs["nnz"],
            "nnz_per_source_column": inputs["nnz"] / pix_pixels,
        },
        "control_dgemm_head_s": control_dgemm_head_s,
        "control_dgemm_tail_s": control_dgemm_tail_s,
        "control_dgemm_drift": control_dgemm_tail_s / control_dgemm_head_s,
        "control_note": (
            "A fixed 1500x1500 dgemm at the head and tail of the cell. The machine "
            "throttles under sustained load, so absolute seconds drift across a "
            "long run; the round-robin interleaving makes the within-cell ratios "
            "immune to that, and this is how much drift there was."
        ),
        "pin_anchor": anchor_name,
        "anchor_first_call_s": anchor_first_call_s,
        "control": control_record,
        "kernels": results,
    }, {name: results[name].get("median_s") for name in results}


# ===================================================================
# Step 2 — thread scaling
# ===================================================================


def run_thread_cell(instrument, mesh_name, reps, threads):
    """``direct_conv_parallel`` and ``rfft2_numpy(workers=)`` at one thread count."""
    geometry = geometry_from(instrument)
    mesh = MESHES[mesh_name]
    pix_pixels = mesh["pix_pixels"]

    native = geometry["native_index_for_slim_index"]
    n_pix = geometry["n_pix"]

    indexes, sizes, weights = synthetic_mapper(n_pix, pix_pixels, mesh["p_per_pixel"])
    inputs = K.kernel_inputs_from(indexes, sizes, weights, native, pix_pixels)

    ny, nx = inputs["ny"], inputs["nx"]
    preload = synthetic_preload(ny, nx)

    reference = ref_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
        curvature_preload=preload,
        pix_indexes_for_sub_slim_index=indexes,
        pix_size_for_sub_slim_index=sizes,
        pix_weights_for_sub_slim_index=weights,
        native_index_for_slim_index=native,
        pix_pixels=pix_pixels,
    )

    parallel = K.direct_conv_parallel_kernel()

    sparse_operator = K.sparse_operator_from(
        inputs["rows_nnz"], inputs["col"], inputs["val"], ny * nx, pix_pixels
    )

    calls = {
        "direct_conv_serial": lambda: K.curvature_direct_conv(
            preload,
            inputs["iy"],
            inputs["ix"],
            inputs["flat"],
            inputs["indptr"],
            inputs["col"],
            inputs["val"],
            inputs["cscptr"],
            inputs["csc_row"],
            inputs["csc_val"],
            ny,
            nx,
            pix_pixels,
        ),
        "direct_conv_parallel": lambda: parallel(
            preload,
            inputs["iy"],
            inputs["ix"],
            inputs["flat"],
            inputs["indptr"],
            inputs["col"],
            inputs["val"],
            inputs["cscptr"],
            inputs["csc_row"],
            inputs["csc_val"],
            ny,
            nx,
            pix_pixels,
        ),
        "rfft2_numpy": lambda: K.curvature_fft_numpy(
            preload,
            inputs["rows_nnz"],
            inputs["col"],
            inputs["val"],
            ny,
            nx,
            pix_pixels,
            real_fft=True,
            batch_size=BATCH_SIZE,
            workers=threads,
            sparse_operator=sparse_operator,
        ),
    }

    out = {}
    for name, call in calls.items():
        pin = pin_result(np.asarray(call()), reference)
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            call()
            samples.append(time.perf_counter() - start)
        timed = samples[1:] if len(samples) > 1 else samples
        out[name] = {
            "median_s": statistics.median(timed),
            "all_rounds_s": samples,
            "pin": pin,
        }
        print(
            f"  {instrument}/{mesh_name} {name} @ {threads} threads: "
            f"{out[name]['median_s']:.4f} s  pin={'PASS' if pin['passed'] else 'FAIL'}",
            flush=True,
        )

    return out


# ===================================================================
# Charts
# ===================================================================


def save_chart(cells, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshes = sorted({c["mesh"]["name"] for c in cells})
    fig, axes = plt.subplots(1, len(meshes), figsize=(6.5 * len(meshes), 5.0), squeeze=False)

    for ax, mesh_name in zip(axes[0], meshes):
        cell = next(c for c in cells if c["mesh"]["name"] == mesh_name)
        names, values, colours = [], [], []
        for name in KERNEL_ORDER:
            record = cell["kernels"].get(name)
            if record is None:
                continue
            value = record.get("median_s") or record.get("extrapolated_seconds")
            if value is None:
                continue
            names.append(name if record.get("median_s") else f"{name} (extrap.)")
            values.append(value)
            colours.append("#4C72B0" if name in NUMBA_KERNELS else "#C44E52")

        y = range(len(names))
        bars = ax.barh(list(y), values, color=colours, edgecolor="white", height=0.6)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f} s",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("median seconds per curvature build (log scale)", fontsize=10)
        ax.set_title(
            f"{mesh_name}: S={cell['mesh']['source_pixels']}, P={cell['mesh']['p_per_pixel']}, "
            f"nnz/col={cell['mesh']['nnz_per_source_column']:.1f}",
            fontsize=10,
        )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================================================================
# Entry point
# ===================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bakeoff", "threads"), default="bakeoff")
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--meshes", default=",".join(MESHES))
    parser.add_argument("--kernels", default=",".join(KERNEL_ORDER))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-name", default=None)
    args, _ = parser.parse_known_args(argv)

    geometries = [g for g in args.geometries.split(",") if g]
    meshes = [m for m in args.meshes.split(",") if m]
    kernel_names = [k for k in args.kernels.split(",") if k]

    out_dir = _ROOT / "results" / "breakdown" / "interferometer"
    out_dir.mkdir(parents=True, exist_ok=True)

    version = aa.__version__

    if args.mode == "threads":
        payload = {
            "autoarray_version": version,
            "mode": "thread_scaling",
            "threads_requested": args.threads,
            "environment": env_record(),
            "pool_caveat": (
                "kernel threads are unavailable under a Nautilus multiprocessing pool; "
                "the single-thread number is the production number"
            ),
            "cells": {},
        }
        for instrument in geometries:
            for mesh_name in meshes:
                payload["cells"][f"{instrument}/{mesh_name}"] = run_thread_cell(
                    instrument, mesh_name, args.reps, args.threads
                )
        name = args.output_name or f"bakeoff_threads_{args.threads}t_v{version}.json"
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nthread-scaling results -> {path}")
        return 0

    payload = {
        "autoarray_version": version,
        "mode": "bakeoff",
        "batch_size": BATCH_SIZE,
        "pin": {
            "rtol": PIN_RTOL,
            "atol": "1e-10 * max|F_reference|",
            "note": "every kernel is pinned before any of its timings are recorded",
        },
        "environment": env_record(),
        "cells": {},
    }

    alma_estimates: dict = {}

    for instrument in geometries:
        cells = []
        for mesh_name in meshes:
            cell, medians = run_cell(
                instrument,
                mesh_name,
                kernel_names,
                args.reps,
                alma_estimates=alma_estimates,
                workers=args.threads,
            )
            payload["cells"][f"{instrument}/{mesh_name}"] = cell
            cells.append(cell)

            if instrument == "alma":
                alma_estimates["n_pix"] = cell["geometry"]["image_pixels_masked"]
                for name, median in medians.items():
                    if median is not None:
                        alma_estimates[(mesh_name, name)] = median

        chart_stem = (
            Path(args.output_name).stem if args.output_name else f"bakeoff_{instrument}_v{version}"
        )
        chart = out_dir / (
            f"{chart_stem}_{instrument}.png"
            if args.output_name
            else f"bakeoff_{instrument}_v{version}.png"
        )
        save_chart(
            cells,
            chart,
            f"Interferometer curvature kernels — {instrument.upper()} "
            f"(single thread, median of timed rounds)",
        )
        print(f"chart -> {chart}")

        # Written after every geometry: alma_high is tens of minutes and a crash there
        # must not cost the sma and alma cells.
        partial = out_dir / (args.output_name or f"bakeoff_v{version}.json")
        partial.write_text(json.dumps(payload, indent=2))

    # --- kill gate --------------------------------------------------------
    gate = {
        "rule": "best numba kernel must beat rfft2_numpy by >1.3x at sma or alma",
        "threshold": 1.3,
        "per_cell": {},
    }
    tripped = True
    for key, cell in payload["cells"].items():
        instrument = key.split("/")[0]
        if instrument not in ("sma", "alma"):
            continue
        rfft = cell["kernels"].get("rfft2_numpy", {}).get("median_s")
        best_name, best = None, None
        for name in NUMBA_KERNELS:
            median = cell["kernels"].get(name, {}).get("median_s")
            if median is not None and (best is None or median < best):
                best, best_name = median, name
        ratio = (rfft / best) if (rfft and best) else None
        gate["per_cell"][key] = {
            "best_numba_kernel": best_name,
            "best_numba_s": best,
            "rfft2_numpy_s": rfft,
            "speedup_rfft2_over_numba": ratio,
        }
        if ratio is not None and ratio > 1.3:
            tripped = False
    gate["kill_gate"] = "tripped" if tripped else "passed"
    payload["kill_gate"] = gate["kill_gate"]
    payload["kill_gate_detail"] = gate

    name = args.output_name or f"bakeoff_v{version}.json"
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nbake-off results -> {path}")
    print(f"KILL GATE: {gate['kill_gate']}")
    for key, record in gate["per_cell"].items():
        print(
            f"  {key}: best numba {record['best_numba_kernel']} = {record['best_numba_s']}, "
            f"rfft2 = {record['rfft2_numpy_s']}, ratio = {record['speedup_rfft2_over_numba']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
