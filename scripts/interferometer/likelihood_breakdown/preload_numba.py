"""
Numba CPU Profiling: The Interferometer ``W~`` Preload (Builder Bake-off)
========================================================================

The preload is the one-off, model-independent ``[2Ny, 2Nx]`` array
``P[i, j] = Σ_k w_k cos(2π(dx·u_k + dy·v_k))`` that every interferometer
w-tilde arm — numba *and* JAX — must pay for once per dataset before a single
likelihood is evaluated. Phase 2 of the ``numba-interferometer-revisit`` epic
measured that build at ~33 minutes at alma: longer than every likelihood number
in ``results/notes/numba_interferometer_verdict.md`` put together. It is not a
per-evaluation step, so no breakdown cell had ever timed it.

This cell times it, four ways, and tests the phase-3 hypothesis that three of
those four are unnecessary:

1. ``numba``   — the recovered ``w_tilde_curvature_preload_interferometer_from``
   (PyAutoArray ``0b90c401``), the quadruple-quadrant ``O(Ny·Nx·K)`` loop.
2. ``numpy``   — ``nufft_precision_operator_via_np_from``, today's library
   default (chunked over visibilities, ``chunk_k=2048``).
3. ``jax_cpu`` — ``nufft_precision_operator_via_jax_from`` on CPU
   (``lax.fori_loop`` over the same chunks).
4. ``nufft``   — ``numba_interferometer.preload.nufft_preload_from``, new in
   phase 3: ``P`` is exactly ``Re`` of a **type-1 (adjoint) NUFFT** of the
   weights ``1/σ²`` onto the doubled offset grid, so it costs
   ``O(K·nspread² + M log M)`` rather than ``O(N_pix·K)``.

All four take the same four arguments (``preload_inputs_from(dataset)``) and
must return the same array. PyAutoArray is untouched: the NUFFT builder lives in
the profiling pack, and a library follow-up is a separate decision the note
records the numbers for.

What is pinned
--------------
- **Parity** against the NumPy builder's array. At sma the rule is *mixed* —
  ``rtol=1e-10`` with ``atol = 1e-10·P[0,0]``. A type-1 NUFFT bounds its error
  against ``Σ_k |c_k|``, i.e. against the peak, so the preload's near-zero
  entries (five orders below it, at the fp64 noise floor) carry no relative
  guarantee; a relative-only test there measures round-off, not the builder.
  At alma the rule is the peak-scaled bound ``max|Δ| ≤ 10·eps·P[0,0]``.
  Both metrics are recorded for every builder either way, so neither rule has to
  be taken on trust.
- **The likelihood downstream of it.** For each ``eps``, the NUFFT preload is
  pushed through ``apply_sparse_operator`` → ``NumbaPreload.from_curvature_preload``
  → ``InversionInterferometerNumba(kernel="direct_conv")`` → ``log_evidence_from``
  and compared with the same chain on the reference preload at ``rtol=1e-6``.
  A preload that passes an array pin but moves the log evidence is not usable,
  and a builder that is fast but wrong is not a result.

Instrument rules
----------------
``alma_high`` runs the NUFFT builder **only**. The brute force there is
``O(61572 · 5e6)`` — hours — and is hard-refused rather than left to a flag.

Output
------
``results/breakdown/interferometer/preload_breakdown_<instrument>_v<version>.{json,png}``
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

import argparse
import json
import os

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import os as _smoke_os
import subprocess
import sys as _smoke_sys
import time
from datetime import UTC, datetime
from pathlib import Path

import autofit as af
import autolens as al
import numpy as np
from autoarray.fit import fit_util
from autoarray.inversion.inversion.interferometer import inversion_interferometer_util

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

from numba_interferometer import (  # noqa: E402
    inversion_interferometer_numba_util as numba_util_pack,
)
from numba_interferometer.inversion import InversionInterferometerNumba  # noqa: E402
from numba_interferometer.preload import (  # noqa: E402
    NumbaPreload,
    nufft_preload_from,
    preload_inputs_from,
)

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import (  # noqa: E402
    auto_simulate_if_missing,
    check_pinned,
    device_info_dict,
    parse_profile_cli,
    record_pinned_check,
    resolve_output_paths,
)
from instruments.interferometer import INSTRUMENTS  # noqa: E402

CELL = "preload_numba"

# ===================================================================
# The four builders
# ===================================================================
#
# Every builder maps the same four arrays to the same [2Ny, 2Nx] array, so the
# bake-off compares implementations of one function rather than four functions.
# ``eps`` is accepted by all of them and ignored by the three exact ones, which
# keeps the dispatch below free of per-builder special cases.

BRUTE_FORCE_BUILDERS = ("numba", "numpy", "jax_cpu")
ALL_BUILDERS = BRUTE_FORCE_BUILDERS + ("nufft",)

CHUNK_K = 2048  # the library default for both chunked brute-force builders

# Visibility chunking for the NUFFT builder — a memory ceiling, not a speed dial.
# The spreader's gather buffer is K x nspread^2 complex128 (~3 GB per 1e6
# visibilities at eps=1e-12), so alma_high's 5e6 needs ~15 GB in one shot and is
# killed by the OOM reaper on a 15 GB machine. Set from the instrument's own
# ``transformer_chunk_size`` once the CLI is parsed; the transform is linear in
# ``c``, so the chunks are summed and the array is unchanged.
NUFFT_CHUNK_SIZE: int | None = None


def _build_numba(inputs: dict, eps: float) -> np.ndarray:
    return np.asarray(
        numba_util_pack.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=inputs["noise_map_real"],
            uv_wavelengths=inputs["uv_wavelengths"],
            shape_masked_pixels_2d=tuple(int(s) for s in inputs["shape_masked_pixels_2d"]),
            grid_radians_2d=inputs["grid_radians_2d"],
        ),
        dtype=np.float64,
    )


def _build_numpy(inputs: dict, eps: float) -> np.ndarray:
    return np.asarray(
        inversion_interferometer_util.nufft_precision_operator_via_np_from(
            noise_map_real=inputs["noise_map_real"],
            uv_wavelengths=inputs["uv_wavelengths"],
            shape_masked_pixels_2d=inputs["shape_masked_pixels_2d"],
            grid_radians_2d=inputs["grid_radians_2d"],
            chunk_k=CHUNK_K,
        ),
        dtype=np.float64,
    )


def _build_jax_cpu(inputs: dict, eps: float) -> np.ndarray:
    return np.asarray(
        inversion_interferometer_util.nufft_precision_operator_via_jax_from(
            noise_map_real=inputs["noise_map_real"],
            uv_wavelengths=inputs["uv_wavelengths"],
            shape_masked_pixels_2d=inputs["shape_masked_pixels_2d"],
            grid_radians_2d=inputs["grid_radians_2d"],
            chunk_k=CHUNK_K,
        ),
        dtype=np.float64,
    )


def _build_nufft(inputs: dict, eps: float) -> np.ndarray:
    return nufft_preload_from(
        noise_map_real=inputs["noise_map_real"],
        uv_wavelengths=inputs["uv_wavelengths"],
        shape_masked_pixels_2d=inputs["shape_masked_pixels_2d"],
        grid_radians_2d=inputs["grid_radians_2d"],
        eps=eps,
        chunk_size=NUFFT_CHUNK_SIZE,
    )


BUILDERS = {
    "numba": _build_numba,
    "numpy": _build_numpy,
    "jax_cpu": _build_jax_cpu,
    "nufft": _build_nufft,
}

BUILDER_DESCRIPTION = {
    "numba": "recovered numba w_tilde_curvature_preload_interferometer_from (0b90c401)",
    "numpy": f"nufft_precision_operator_via_np_from (library default, chunk_k={CHUNK_K})",
    "jax_cpu": f"nufft_precision_operator_via_jax_from on CPU (chunk_k={CHUNK_K})",
    "nufft": "type-1 adjoint NUFFT (numba_interferometer.preload.nufft_preload_from)",
}

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
)


def thread_env_dict() -> dict:
    return {key.lower(): os.environ.get(key) for key in THREAD_ENV_KEYS}


def timed_call(fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, wall_s, cpu_s)``.

    ``cpu_s`` is ``time.process_time()``, which sums **every thread** of the
    process, so ``cpu_s / wall_s`` is the effective thread count of the call. It is
    recorded for every build because the thread pinning in this cell's
    environment does not reach all four builders equally: ``OMP/MKL/OPENBLAS/
    NUMBA_NUM_THREADS=1`` pin NumPy's BLAS and numba, but XLA's CPU runtime keeps
    its own intra-op pool, so the two JAX-backed builders (``jax_cpu`` and the
    ``nufft`` one, which calls nufftax through jax) can and do use several cores
    even with ``--xla_cpu_multi_thread_eigen=false``. Comparing their wall time
    against a genuinely single-threaded NumPy loop without saying so would flatter
    them by the thread factor, so both numbers are reported and the note quotes
    both.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - wall_start, time.process_time() - cpu_start


# ===================================================================
# Worker mode
# ===================================================================
#
# A brute-force build at alma is ~30 minutes and unbounded at jax_cpu, so each
# one is run in its own short-lived subprocess with a wall-clock timeout: a
# builder that overruns is killed and recorded as ``timed_out`` instead of
# taking the whole run down with it, and the two that did finish keep their
# numbers. The child writes its own ``.npy`` cache and ``.json`` sidecar *before*
# exiting, so a result is durable the moment it exists — the parent never holds
# a 30-minute measurement only in memory.
#
# The subprocess is this same file re-entered under PRELOAD_NUMBA_WORKER=1,
# rather than a second module or a multiprocessing fork: one file means the
# timed builder is literally the same function object in both regimes (a copy
# would be a second thing to keep in sync), and a fresh interpreter — not a fork
# of one with JAX already initialised — is the only safe way to run the JAX
# builder under a hard kill. The branch sits above every dataset/CLI line below
# so a worker pays for nothing but its own build.

if os.environ.get("PRELOAD_NUMBA_WORKER") == "1":
    _spec = json.loads(os.environ["PRELOAD_NUMBA_WORKER_SPEC"])

    _npz = np.load(_spec["inputs_path"])
    _worker_inputs = {
        "noise_map_real": _npz["noise_map_real"],
        "uv_wavelengths": _npz["uv_wavelengths"],
        "shape_masked_pixels_2d": tuple(int(s) for s in _npz["shape_masked_pixels_2d"]),
        "grid_radians_2d": _npz["grid_radians_2d"],
    }

    NUFFT_CHUNK_SIZE = _spec.get("nufft_chunk_size")

    _array, _build_s, _build_cpu_s = timed_call(
        BUILDERS[_spec["builder"]], _worker_inputs, _spec["eps"]
    )

    np.save(_spec["npy_path"], np.asarray(_array, dtype=np.float64))
    Path(_spec["sidecar_path"]).write_text(
        json.dumps(
            {
                "builder": _spec["builder"],
                "instrument": _spec["instrument"],
                "build_s": _build_s,
                "build_cpu_s": _build_cpu_s,
                "threads_effective": _build_cpu_s / _build_s if _build_s > 0 else None,
                "eps": _spec["eps"] if _spec["builder"] == "nufft" else None,
                "shape": [int(s) for s in np.asarray(_array).shape],
                "autolens_version": al.__version__,
                "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "threads": thread_env_dict(),
                "chunk_k": CHUNK_K,
            },
            indent=2,
        )
    )
    print(f"[worker] {_spec['builder']} built in {_build_s:.4f} s -> {_spec['npy_path']}")
    _smoke_sys.exit(0)


# ===================================================================
# CLI
# ===================================================================

_cli = parse_profile_cli()

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--builders",
    default="all",
    help=(
        "Comma-separated subset of "
        f"{','.join(ALL_BUILDERS)} (or 'all'). At alma_high only 'nufft' is "
        "permitted; the brute-force builders are hard-refused there."
    ),
)
_parser.add_argument(
    "--eps",
    default="1e-6,1e-9,1e-12",
    help="Comma-separated NUFFT precisions to sweep.",
)
_parser.add_argument(
    "--reps",
    type=int,
    default=3,
    help=(
        "Repeats for the cheap builds (always the NUFFT one; the brute-force "
        "trio too at sma). The brute-force builders run ONCE at every other "
        "instrument, where one build is tens of minutes."
    ),
)
_parser.add_argument(
    "--builder-timeout-s",
    type=float,
    default=5400.0,
    help=(
        "Per-brute-force-builder wall-clock cap (default 90 min). A builder that "
        "overruns is killed and recorded as timed_out. <= 0 disables the cap and "
        "runs the builder in-process."
    ),
)
_args = _parser.parse_known_args()[0]

instrument = _cli.instrument or "sma"

if instrument not in INSTRUMENTS:
    raise SystemExit(
        f"--instrument {instrument!r} is not in instruments.interferometer.INSTRUMENTS."
    )

requested_builders = (
    list(ALL_BUILDERS)
    if _args.builders.strip().lower() == "all"
    else [b.strip() for b in _args.builders.split(",") if b.strip()]
)

for _b in requested_builders:
    if _b not in BUILDERS:
        raise SystemExit(
            f"--builders {_b!r} is not available; choose from {','.join(ALL_BUILDERS)}."
        )

# alma_high: 61572 masked pixels x 5e6 visibilities. The brute force is hours,
# not minutes, and nothing in this cell needs it — the NUFFT builder is pinned
# against the brute force at sma and alma, which is where a pin is affordable.
# Refused rather than warned about: a flag that quietly starts an overnight
# build is the failure this guard exists to prevent.
if instrument == "alma_high":
    _refused = [b for b in requested_builders if b in BRUTE_FORCE_BUILDERS]
    if _args.builders.strip().lower() == "all":
        requested_builders = ["nufft"]
    elif _refused:
        raise SystemExit(
            f"Refusing to run the brute-force builder(s) {','.join(_refused)} at alma_high: "
            f"O(N_pix x K) = O(61572 x 5e6) is hours per builder. Only 'nufft' is permitted "
            f"here; the brute force is pinned against it at sma and alma."
        )

eps_sweep = [float(e) for e in _args.eps.split(",") if e.strip()]
if not eps_sweep:
    raise SystemExit("--eps must list at least one precision.")

# The eps used for the `nufft` builder's own timing row and for the parity
# reference: the tightest requested, i.e. the one a production build would use.
eps_primary = min(eps_sweep)

reps = max(1, int(_args.reps))
builder_timeout_s = float(_args.builder_timeout_s)

# The regime split. At sma every build is seconds, so every builder is repeated,
# run in-process and never cached — a cache would only serve to make a cheap
# measurement stale. Everywhere else the brute-force trio is a once-only,
# isolated, cached, timeout-capped build; the NUFFT builder stays cheap and is
# repeated at every instrument (that being the claim under test).
long_build_regime = instrument != "sma"


def _reps_for(builder: str) -> int:
    return 1 if (long_build_regime and builder in BRUTE_FORCE_BUILDERS) else reps


def _is_isolated(builder: str) -> bool:
    return long_build_regime and builder in BRUTE_FORCE_BUILDERS and builder_timeout_s > 0.0


def _is_cached(builder: str) -> bool:
    return long_build_regime and builder in BRUTE_FORCE_BUILDERS


# ===================================================================
# Dataset
# ===================================================================

print(f"\n--- Dataset loading [{instrument}] ---")

_workspace_root = _profiling_root()
pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]
real_space_shape = INSTRUMENTS[instrument]["real_space_shape"]
mask_radius = INSTRUMENTS[instrument]["mask_radius"]
transformer_chunk_size = INSTRUMENTS[instrument].get("transformer_chunk_size", None)

NUFFT_CHUNK_SIZE = transformer_chunk_size

dataset_path = Path("dataset") / "interferometer" / instrument

auto_simulate_if_missing(
    dataset_path,
    dataset_type="interferometer",
    instrument=instrument,
    workspace_root=_workspace_root,
)

real_space_mask = al.Mask2D.circular(
    shape_native=real_space_shape,
    pixel_scales=pixel_scale,
    radius=mask_radius,
)


def _build_transformer(uv_wavelengths, real_space_mask):
    """Per-instrument NUFFT chunk_size, as ``delaunay_numba.py`` does it (PyAutoArray#330)."""
    return al.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=transformer_chunk_size,
    )


dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=_build_transformer,
)

inputs = preload_inputs_from(dataset)

n_visibilities = int(inputs["uv_wavelengths"].shape[0])
extent = tuple(int(s) for s in inputs["shape_masked_pixels_2d"])
preload_shape = (2 * extent[0], 2 * extent[1])
n_image_pixels = int(dataset.real_space_mask.pixels_in_mask)

print(f"  Visibilities K:          {n_visibilities}")
print(f"  Masked pixels N_pix:     {n_image_pixels}")
print(f"  Masked bounding extent:  {extent[0]} x {extent[1]}")
print(f"  Preload shape:           {preload_shape[0]} x {preload_shape[1]}")
print(f"  Builders:                {', '.join(requested_builders)}")
print(f"  eps sweep:               {', '.join(f'{e:g}' for e in eps_sweep)}")

_cache_key = f"{real_space_shape[0]}x{real_space_shape[1]}_r{mask_radius}"

# The array `apply_sparse_operator` itself caches (written by delaunay_numba.py and
# its siblings). It is the NumPy builder's output, so it is a usable parity
# reference even in a run that does not rebuild it — but its build was never timed,
# which is why the `numpy` builder is still measured from scratch below.
_legacy_reference_path = dataset_path / f"nufft_precision_operator_{_cache_key}.npy"


def _cache_paths(builder: str) -> tuple[Path, Path]:
    """The ``.npy`` cache and its ``.json`` timing sidecar for one builder.

    The extension is appended textually, never through ``Path.with_suffix``: the
    cache key ends in the mask radius (``..._r3.5``), and ``with_suffix`` reads
    ``.5`` as the existing suffix and replaces it — silently keying alma's
    ``r3.5`` cache as ``r3`` and colliding it with a hypothetical ``r3`` run.
    """
    stem = dataset_path / f"preload_{builder}_{_cache_key}"
    return Path(f"{stem}.npy"), Path(f"{stem}.json")


# ===================================================================
# Machine-speed control
# ===================================================================


def dgemm_control_s(n: int = 1500, repeats: int = 3) -> float:
    """A fixed 1500x1500 ``dgemm``, timed identically in every run.

    The phase-2 machine-drift control, carried over unchanged: this laptop
    throttles, so a build_s compared across runs (and the alma trio *is* split
    across runs) is only believable if the same BLAS call costs the same in each.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))

    a @ b  # warm the BLAS thread pool / first-touch the pages

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        samples.append(time.perf_counter() - start)

    return float(np.median(samples))


print("\n--- Machine-speed control ---")
control_dgemm_s = dgemm_control_s()
print(f"  1500x1500 dgemm: {control_dgemm_s:.4f} s")


# ===================================================================
# Timed builds
# ===================================================================

_worker_inputs_path = None


def _worker_inputs_npz() -> str:
    """The four builder arguments on disk, written once, for the worker children."""
    global _worker_inputs_path

    if _worker_inputs_path is None:
        path = dataset_path / f"preload_inputs_{_cache_key}.npz"
        np.savez(
            path,
            noise_map_real=inputs["noise_map_real"],
            uv_wavelengths=inputs["uv_wavelengths"],
            shape_masked_pixels_2d=np.asarray(extent, dtype=np.int64),
            grid_radians_2d=inputs["grid_radians_2d"],
        )
        _worker_inputs_path = str(path.resolve())

    return _worker_inputs_path


def _run_isolated(builder: str, eps: float, npy_path: Path, sidecar_path: Path) -> dict:
    """Build ``builder`` in a fresh interpreter under ``builder_timeout_s``.

    Returns the sidecar the child wrote, or ``{"timed_out": True}`` if it was
    killed. The child persists its own result, so a timeout costs only the
    builder that overran.
    """
    env = dict(os.environ)
    env["PRELOAD_NUMBA_WORKER"] = "1"
    env["PRELOAD_NUMBA_WORKER_SPEC"] = json.dumps(
        {
            "builder": builder,
            "instrument": instrument,
            "eps": eps,
            "inputs_path": _worker_inputs_npz(),
            "npy_path": str(npy_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
            "nufft_chunk_size": NUFFT_CHUNK_SIZE,
        }
    )

    print(f"  [{builder}] isolated build, cap {builder_timeout_s / 60.0:.0f} min ...")

    try:
        subprocess.run(
            [_smoke_sys.executable, str(Path(__file__).resolve())],
            env=env,
            check=True,
            timeout=builder_timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"  [{builder}] TIMED OUT after {builder_timeout_s / 60.0:.0f} min — recorded, moving on."
        )
        return {"timed_out": True}

    return json.loads(sidecar_path.read_text())


def timed_build(builder: str, eps: float) -> dict:
    """Build ``builder``'s preload, honouring the cache / isolation / repeat regime.

    Returns a record carrying the array (``preload``, ``None`` on a timeout) and
    the timing provenance the JSON needs: whether the number came from this run
    or from the sidecar of the run that actually paid for it.
    """
    npy_path, sidecar_path = _cache_paths(builder)

    if _is_cached(builder) and npy_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        print(
            f"  [{builder}] cache hit: {npy_path} (build_s={sidecar['build_s']:.4f} from sidecar)"
        )
        return {
            "preload": np.load(npy_path),
            "build_s": float(sidecar["build_s"]),
            "build_cpu_s": sidecar.get("build_cpu_s"),
            "threads_effective": sidecar.get("threads_effective"),
            "cache_hit": True,
            "timed_out": False,
            "reps": 1,
            "warmup_s": None,
            "threads": sidecar.get("threads"),
            "build_s_source": "sidecar",
            "built_utc": sidecar.get("built_utc"),
        }

    if _is_isolated(builder):
        sidecar = _run_isolated(builder, eps, npy_path, sidecar_path)

        if sidecar.get("timed_out"):
            return {
                "preload": None,
                "build_s": None,
                "build_cpu_s": None,
                "threads_effective": None,
                "cache_hit": False,
                "timed_out": True,
                "reps": 0,
                "warmup_s": None,
                "threads": thread_env_dict(),
                "build_s_source": "timed_out",
                "built_utc": None,
            }

        return {
            "preload": np.load(npy_path),
            "build_s": float(sidecar["build_s"]),
            "build_cpu_s": sidecar.get("build_cpu_s"),
            "threads_effective": sidecar.get("threads_effective"),
            "cache_hit": False,
            "timed_out": False,
            "reps": 1,
            "warmup_s": None,
            "threads": sidecar.get("threads"),
            "build_s_source": "isolated_subprocess",
            "built_utc": sidecar.get("built_utc"),
        }

    n_reps = _reps_for(builder)

    warmup_s = None
    if n_reps > 1:
        # numba compiles lazily and JAX/nufftax compile on first call at a given
        # shape. A one-off preload build genuinely pays that once, so it is
        # reported (warmup_s) rather than hidden — but it is not what the
        # repeated rows measure.
        preload, warmup_s, _warmup_cpu_s = timed_call(BUILDERS[builder], inputs, eps)

    samples = []
    cpu_samples = []
    for _ in range(n_reps):
        preload, wall_s, cpu_s = timed_call(BUILDERS[builder], inputs, eps)
        samples.append(wall_s)
        cpu_samples.append(cpu_s)

    build_s = float(np.median(samples))
    build_cpu_s = float(np.median(cpu_samples))

    if _is_cached(builder):
        np.save(npy_path, np.asarray(preload, dtype=np.float64))
        sidecar_path.write_text(
            json.dumps(
                {
                    "builder": builder,
                    "instrument": instrument,
                    "build_s": build_s,
                    "build_cpu_s": build_cpu_s,
                    "threads_effective": build_cpu_s / build_s if build_s > 0 else None,
                    "eps": eps if builder == "nufft" else None,
                    "shape": [int(s) for s in np.asarray(preload).shape],
                    "autolens_version": al.__version__,
                    "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "threads": thread_env_dict(),
                    "chunk_k": CHUNK_K,
                },
                indent=2,
            )
        )

    return {
        "preload": np.asarray(preload, dtype=np.float64),
        "build_s": build_s,
        "build_cpu_s": build_cpu_s,
        "threads_effective": build_cpu_s / build_s if build_s > 0 else None,
        "cache_hit": False,
        "timed_out": False,
        "reps": n_reps,
        "warmup_s": warmup_s,
        "threads": thread_env_dict(),
        "build_s_source": "in_process",
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


print("\n--- Builder bake-off ---")

builder_records: dict[str, dict] = {}
builder_arrays: dict[str, np.ndarray] = {}

for builder in requested_builders:
    print(f"\n  {builder}: {BUILDER_DESCRIPTION[builder]}")
    record = timed_build(builder, eps_primary)
    preload = record.pop("preload")

    if preload is not None:
        builder_arrays[builder] = preload
        print(
            f"  [{builder}] build_s = {record['build_s']:.4f} s "
            f"({record['build_cpu_s']:.4f} cpu-s, "
            f"{record['threads_effective']:.2f}x threads, reps={record['reps']})"
        )

    record["description"] = BUILDER_DESCRIPTION[builder]
    record["eps"] = eps_primary if builder == "nufft" else None
    builder_records[builder] = record


# ===================================================================
# Parity
# ===================================================================
#
# The reference is the NumPy builder's array: the one today's
# `apply_sparse_operator` produces and every downstream number in this repo was
# computed from. Preference order — this run's own `numpy` build, then the
# committed-cache array beside the dataset (the same function's output, from an
# earlier run), then a build on the spot. At alma_high there is no reference and
# none is manufactured: the NUFFT builder is pinned where a pin is affordable.

RTOL_ELEMENTWISE = 1.0e-10
PEAK_SCALE_CONSTANT = 10.0

reference_source = None
reference_preload = None

if "numpy" in builder_arrays:
    reference_preload = builder_arrays["numpy"]
    reference_source = (
        "numpy_builder_cached"
        if builder_records["numpy"]["cache_hit"]
        else "numpy_builder_this_run"
    )
elif _legacy_reference_path.exists():
    reference_preload = np.load(_legacy_reference_path)
    reference_source = f"cached_np_builder_array ({_legacy_reference_path.name})"
elif not long_build_regime:
    reference_preload = _build_numpy(inputs, eps_primary)
    reference_source = "numpy_builder_built_for_reference"

if reference_preload is not None:
    reference_peak = float(reference_preload[0, 0])
    print(f"\n--- Parity reference: {reference_source} (P[0,0] = {reference_peak:.6e}) ---")
else:
    reference_peak = None
    print("\n--- Parity reference: NONE (no affordable brute force at this instrument) ---")


def parity_metrics(candidate, *, eps: float) -> dict:
    """Every parity metric for one array, plus the rule applied and its verdict.

    ``parity_rel_peak`` is the bound a type-1 NUFFT actually satisfies (its error
    is bounded against ``Σ_k |c_k|``, i.e. against the peak). ``parity_max_rel_elementwise``
    is the strictest thing that *could* be asked, reported alongside the count of
    entries that breach ``rtol=1e-10`` so the two rules can be compared rather
    than taken on trust — the preload's near-zero entries sit at the fp64 noise
    floor, where an elementwise relative test measures round-off, not the builder.

    ``eps`` scales the peak-scaled rule, so a sweep row is judged against the
    precision *it* asked for rather than against the tightest one in the run. The
    three exact builders have no ``eps``; they are held to ``eps_primary``, the
    strictest form of the same bound. Both verdicts (``passes_pin`` under the
    instrument's rule, ``passes_peak_scaled_pin`` always) are recorded, so the
    choice of rule is visible rather than baked into a single boolean.
    """
    if candidate is None or reference_preload is None:
        return {
            "parity_reference": reference_source,
            "parity_rule": None,
            "parity_peak_scaled_rule": None,
            "parity_max_abs": None,
            "parity_rel_peak": None,
            "parity_max_rel_elementwise": None,
            "parity_n_elementwise_over_1e10": None,
            "passes_pin": None,
            "passes_peak_scaled_pin": None,
        }

    candidate = np.asarray(candidate, dtype=np.float64)
    delta = np.abs(candidate - reference_preload)
    max_abs = float(delta.max())
    rel_peak = max_abs / abs(reference_peak)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_each = delta / np.maximum(np.abs(reference_preload), 1e-300)

    max_rel = float(rel_each.max())
    n_over = int((rel_each > RTOL_ELEMENTWISE).sum())

    # The plan's peak-scaled rule, evaluated at every instrument.
    bound = PEAK_SCALE_CONSTANT * eps * abs(reference_peak)
    peak_scaled_rule = f"max|d| <= {PEAK_SCALE_CONSTANT:g} * eps({eps:g}) * P[0,0] = {bound:.6e}"
    passes_peak_scaled = bool(max_abs <= bound)

    # The mixed sma rule: full relative strength on every entry that carries
    # signal, an absolute floor of rtol x peak under the ones that do not.
    atol = RTOL_ELEMENTWISE * abs(reference_peak)
    mixed_rule = f"allclose(rtol={RTOL_ELEMENTWISE:g}, atol={RTOL_ELEMENTWISE:g}*P[0,0]={atol:.6e})"
    passes_mixed = bool(np.allclose(candidate, reference_preload, rtol=RTOL_ELEMENTWISE, atol=atol))

    rule, passes = (
        (peak_scaled_rule, passes_peak_scaled) if long_build_regime else (mixed_rule, passes_mixed)
    )

    return {
        "parity_reference": reference_source,
        "parity_rule": rule,
        "parity_peak_scaled_rule": peak_scaled_rule,
        "parity_max_abs": max_abs,
        "parity_rel_peak": rel_peak,
        "parity_max_rel_elementwise": max_rel,
        "parity_n_elementwise_over_1e10": n_over,
        "passes_pin": passes,
        "passes_peak_scaled_pin": passes_peak_scaled,
    }


for builder in requested_builders:
    builder_records[builder].update(parity_metrics(builder_arrays.get(builder), eps=eps_primary))


# ===================================================================
# The downstream likelihood
# ===================================================================
#
# The chain of `delaunay_numba.py`: a preload that passes an array pin but moves
# the log evidence is not usable, so the pin that actually matters is on the
# number the sampler sees. Built once (the model, mesh and adapt image are
# preload-independent) and re-evaluated per candidate preload.

hilbert_pixels = 1500
regularization_coefficient = 1.0
RTOL_LOG_EVIDENCE = 1.0e-6

_evidence_state: dict = {}


def _evidence_setup():
    """The adapt image, Hilbert mesh, model instance and adapt images — built once."""
    if _evidence_state:
        return _evidence_state

    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

    image_mesh = al.image_mesh.Hilbert(pixels=hilbert_pixels, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.real_space_mask, adapt_data=adapt_image
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    _ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=0),
        regularization=al.reg.ConstantSplit(coefficient=regularization_coefficient),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)

    _evidence_state["instance"] = instance
    _evidence_state["adapt_images"] = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={instance.galaxies.source: image_plane_mesh_grid},
        galaxy_name_image_plane_mesh_grid_dict={"('galaxies', 'source')": image_plane_mesh_grid},
    )
    _evidence_state["mesh_vertices"] = int(image_plane_mesh_grid.shape[0])

    return _evidence_state


def log_evidence_for(preload: np.ndarray) -> float:
    """The numba log evidence this preload produces — ``delaunay_numba.py``'s chain.

    ``apply_sparse_operator`` is handed the candidate array so the dirty image and
    the operator come from the *same* preload the numba kernel then indexes; passing
    one and rebuilding the other is exactly the drift ``from_curvature_preload``
    exists to prevent.
    """
    state = _evidence_setup()

    preload = np.asarray(preload, dtype=np.float64)

    ds = dataset.apply_sparse_operator(use_jax=False, nufft_precision_operator=preload)
    numba_preload = NumbaPreload.from_curvature_preload(ds, preload)

    fit = al.FitInterferometer(
        dataset=ds,
        tracer=al.Tracer(galaxies=list(state["instance"].galaxies)),
        adapt_images=state["adapt_images"],
        settings=al.Settings(),
        xp=np,
    )

    inversion = InversionInterferometerNumba(
        dataset=fit.inversion.dataset,
        linear_obj_list=fit.inversion.linear_obj_list,
        settings=fit.inversion.settings,
        xp=np,
        numba_preload=numba_preload,
        kernel="direct_conv",
    )

    return float(
        fit_util.log_evidence_from(
            chi_squared=inversion.fast_chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=fit.noise_normalization,
        )
    )


# ===================================================================
# eps sweep
# ===================================================================

print("\n--- NUFFT eps sweep ---")

log_evidence_reference = None
if reference_preload is not None:
    print("  reference log evidence (brute-force preload) ...")
    log_evidence_reference = log_evidence_for(reference_preload)
    print(f"  log_evidence_reference = {log_evidence_reference!r}")

eps_records: dict[str, dict] = {}
eps_arrays: dict[float, np.ndarray] = {}

for eps in eps_sweep:
    array, warmup_s, _warmup_cpu_s = timed_call(_build_nufft, inputs, eps)

    samples = []
    cpu_samples = []
    for _ in range(reps):
        array, wall_s, cpu_s = timed_call(_build_nufft, inputs, eps)
        samples.append(wall_s)
        cpu_samples.append(cpu_s)

    eps_arrays[eps] = array
    build_s = float(np.median(samples))
    build_cpu_s = float(np.median(cpu_samples))

    record = {
        "build_s": build_s,
        "build_cpu_s": build_cpu_s,
        "threads_effective": build_cpu_s / build_s if build_s > 0 else None,
        "warmup_s": warmup_s,
        "reps": reps,
    }
    record.update(parity_metrics(array, eps=eps))

    if log_evidence_reference is not None:
        log_evidence = log_evidence_for(array)
        rel_diff = abs(log_evidence - log_evidence_reference) / max(
            abs(log_evidence_reference), 1e-300
        )
        record["log_evidence"] = log_evidence
        record["log_evidence_reference"] = log_evidence_reference
        record["log_evidence_rel_diff"] = rel_diff
        record["passes_log_evidence_pin"] = bool(rel_diff <= RTOL_LOG_EVIDENCE)
    else:
        record["log_evidence"] = None
        record["log_evidence_reference"] = None
        record["log_evidence_rel_diff"] = None
        record["passes_log_evidence_pin"] = None

    eps_records[f"{eps:g}"] = record

    print(
        f"  eps={eps:g}: build {build_s:.4f} s ({build_cpu_s:.4f} cpu-s, "
        f"{record['threads_effective']:.2f}x threads)"
        + (
            f" | rel_peak {record['parity_rel_peak']:.3e} (pin {record['passes_pin']})"
            if record["parity_rel_peak"] is not None
            else " | parity n/a"
        )
        + (
            f" | log_evidence rel diff {record['log_evidence_rel_diff']:.3e} "
            f"(pin {record['passes_log_evidence_pin']})"
            if record["log_evidence_rel_diff"] is not None
            else " | log evidence n/a"
        )
    )

# eps-to-eps self-consistency: the only internal check available where there is no
# brute-force reference (alma_high). A looser eps that agrees with the tightest one
# to the peak-scaled bound is doing the same arithmetic, just less carefully.
eps_self_consistency = {}
if len(eps_arrays) > 1:
    tightest = min(eps_arrays)
    anchor = eps_arrays[tightest]
    anchor_peak = float(anchor[0, 0])
    for eps, array in eps_arrays.items():
        if eps == tightest:
            continue
        max_abs = float(np.abs(array - anchor).max())
        eps_self_consistency[f"{eps:g}"] = {
            "vs_eps": f"{tightest:g}",
            "max_abs": max_abs,
            "rel_peak": max_abs / abs(anchor_peak),
        }


# ===================================================================
# JSON + PNG
# ===================================================================

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

al_version = al.__version__

print("\n" + "=" * 78)
print(f"INTERFEROMETER W~ PRELOAD BUILDER BAKE-OFF — {instrument.upper()} — v{al_version}")
print("=" * 78)

_max_label = max(len(b) for b in requested_builders)
for builder in requested_builders:
    record = builder_records[builder]
    if record["timed_out"]:
        print(f"  {builder:<{_max_label}}  TIMED OUT (> {builder_timeout_s / 60.0:.0f} min)")
    else:
        print(
            f"  {builder:<{_max_label}}  {record['build_s']:>12.4f} s"
            f"   parity_rel_peak={record['parity_rel_peak']}"
            f"   pin={record['passes_pin']}"
        )
print("=" * 78)

summary = {
    "autolens_version": al_version,
    "device": device_info_dict(),
    "instrument": instrument,
    "control_dgemm_s": control_dgemm_s,
    "control_note": (
        "A fixed 1500x1500 BLAS dgemm. This laptop throttles and the alma "
        "brute-force trio is deliberately split across processes/runs, so a "
        "build_s is only comparable across runs when this agrees."
    ),
    "geometry": {
        "K": n_visibilities,
        "N_pix": n_image_pixels,
        "extent": list(extent),
        "preload_shape": list(preload_shape),
        "pixel_scale_arcsec": pixel_scale,
        "mask_radius_arcsec": mask_radius,
        "real_space_shape": list(real_space_shape),
    },
    "configuration": {
        "builders": requested_builders,
        "eps_sweep": eps_sweep,
        "eps_primary": eps_primary,
        "reps": reps,
        "reps_note": (
            "Applies to the NUFFT builder at every instrument and to the "
            "brute-force trio at sma only; elsewhere the brute force runs once, "
            "isolated in its own interpreter under builder_timeout_s."
        ),
        "builder_timeout_s": builder_timeout_s,
        "chunk_k": CHUNK_K,
        "nufft_chunk_size": NUFFT_CHUNK_SIZE,
        # Read by scripts/misc/tooling/build_readme.py for the dashboard's
        # "Inversion path" column. This cell has no per-step decomposition, so its
        # step-sum total renders as "—"; without this key the row would be
        # mislabelled "dense (mapping)" by the filename fallback.
        "inversion_path": "sparse_numba",
        "kernel": "direct_conv",
        "mesh": f"delaunay_hilbert_{hilbert_pixels}",
        "regularization": "constant_split",
        "regularization_coefficient": regularization_coefficient,
        "threads": thread_env_dict(),
    },
    "builders": builder_records,
    "threads_note": (
        "build_s is wall clock; build_cpu_s is time.process_time(), which sums "
        "every thread, so threads_effective = build_cpu_s / build_s is the "
        "effective core count of the build. OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1 "
        "pin NumPy's BLAS and numba to one core, but XLA's CPU runtime keeps its "
        "own intra-op pool, so the two JAX-backed builders (jax_cpu, and nufft via "
        "nufftax) can use several cores even under "
        "--xla_cpu_multi_thread_eigen=false. Quote both numbers before comparing "
        "them with a single-threaded loop."
    ),
    "parity_reference": reference_source,
    "parity_reference_peak": reference_peak,
    "parity_note": (
        "parity_rel_peak = max|d| / P[0,0]; parity_max_rel_elementwise is the "
        "strictest possible reading, with parity_n_elementwise_over_1e10 counting "
        "the entries that breach rtol=1e-10. The sma rule is mixed "
        "(rtol=1e-10, atol=1e-10*P[0,0]): a type-1 NUFFT bounds its error against "
        "sum|c_k|, i.e. against the peak, so its near-zero entries — five orders "
        "below the peak, at the fp64 noise floor — carry no relative guarantee and "
        "a relative-only test there measures round-off, not the builder. The alma "
        "rule is the peak-scaled bound 10*eps*P[0,0], evaluated at each row's own "
        "eps. Both metrics and both verdicts (passes_pin under the instrument's "
        "rule, passes_peak_scaled_pin always) are recorded under either rule, so "
        "the choice of rule is visible rather than baked into one boolean."
    ),
    "nufft_eps_sweep": eps_records,
    "nufft_eps_self_consistency": eps_self_consistency,
    "log_evidence_rtol": RTOL_LOG_EVIDENCE,
    "log_evidence_note": (
        "Each eps's preload is pushed through apply_sparse_operator -> "
        "NumbaPreload.from_curvature_preload -> "
        "InversionInterferometerNumba(kernel='direct_conv') -> "
        "fit_util.log_evidence_from, the chain delaunay_numba.py profiles, and "
        "compared with the same chain on the brute-force reference preload. A "
        "preload that passes an array pin but moves the log evidence is not usable."
    ),
    "mesh_vertices": _evidence_state.get("mesh_vertices"),
    "hilbert_pixels": hilbert_pixels,
}

dict_path, chart_path = resolve_output_paths(
    _cli,
    default_dir=_workspace_root / "results" / "breakdown" / "interferometer",
    default_basename=f"preload_breakdown_{instrument}_v{al_version}",
    cell=CELL,
)
dict_path.write_text(json.dumps(summary, indent=2))
print(f"\n  Results dict saved to: {dict_path}")

# --- Save bar chart ---

bar_labels = []
bar_values = []
bar_colors = []

for builder in requested_builders:
    record = builder_records[builder]
    if record["build_s"] is None:
        continue
    bar_labels.append(f"{builder}")
    bar_values.append(record["build_s"])
    bar_colors.append("#DD8452" if builder == "nufft" else "#4C72B0")

for eps_label, record in eps_records.items():
    bar_labels.append(f"nufft eps={eps_label}")
    bar_values.append(record["build_s"])
    bar_colors.append("#55A868")

fig, ax = plt.subplots(figsize=(10, 5.5))
y_pos = range(len(bar_labels))
bars = ax.barh(y_pos, bar_values, color=bar_colors, edgecolor="white", height=0.6)

for bar, value in zip(bars, bar_values):
    ax.text(
        bar.get_width() * 1.08,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4g} s",
        va="center",
        fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_labels, fontsize=10)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Preload build time (s, log scale)", fontsize=11)
ax.margins(x=0.35)
fig.suptitle(
    f"Interferometer W~ preload — builder bake-off — {instrument.upper()}",
    fontsize=12,
    fontweight="bold",
)
ax.set_title(
    f"AutoLens v{al_version}  |  K = {n_visibilities}  |  N_pix = {n_image_pixels}  |  "
    f"preload {preload_shape[0]}x{preload_shape[1]}",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"  Bar chart saved to:    {chart_path}")


# ===================================================================
# Pinned-value drift record (soft)
# ===================================================================

_pinned_drift: list = []

# Pinned 2026-09-08 (v2026.8.17.1). The log evidence of the 1500-tier
# Hilbert/Delaunay fiducial through the numba direct_conv kernel on the
# brute-force reference preload — the same fiducial delaunay_numba.py pins, so a
# divergence here is the preload, not the mesh.
EXPECTED_LOG_EVIDENCE: dict[str, float] = {
    "sma": -3169.6493766794806,
}

_pinned_expected = EXPECTED_LOG_EVIDENCE.get(instrument)

if log_evidence_reference is None:
    print(f"  Pinned check SKIPPED for {instrument} (no brute-force reference log evidence).")
elif _pinned_expected is None:
    print(
        f"  Pinned check SKIPPED for {instrument} (no pinned value). "
        f"log_evidence_reference = {log_evidence_reference!r}"
    )
else:
    _rec = check_pinned(
        log_evidence_reference, _pinned_expected, label="preload_reference", rtol=1e-6
    )
    if _rec is not None:
        _pinned_drift.append(_rec)

record_pinned_check(dict_path, _pinned_expected, _pinned_drift)
if _pinned_expected is not None and log_evidence_reference is not None and not _pinned_drift:
    print("  Pinned-value check PASSED (recorded in result JSON).")
