"""
Interferometer Streaming Scaling: Shared Harness
================================================

The shared half of the ``streaming_scaling`` cells (``accumulate.py``,
``in_memory.py``, ``parity.py``, ``plot_scaling.py``). It owns four things, so
that the cells only choose grids and write JSON:

1. **The synthetic dataset.** Seeded, per-chunk visibilities generated in memory
   (``uv`` uniform in ``±1e5`` wavelengths, unit-sigma complex Gaussian data at
   ``1e-3``, noise ``1 + 1j``), a 400 x 400 circular mask at 0.05"/pix (radius
   10", 125 676 unmasked pixels) and ``TransformerNUFFT``. Chunk ``k`` is drawn
   from ``default_rng(SEED + k)``, so the stream and the in-memory arrays built
   from the same ``(N_vis, chunk)`` are the same visibilities.
2. **The builders.** ``Interferometer.from_stream`` (array-free, chunk by chunk)
   and ``Interferometer(...).apply_sparse_operator()`` (every visibility in memory
   at once), plus the 20 x 20 rectangular sparse inversion whose ``log_evidence``
   the parity cell compares.
3. **The child runner.** Every measurement runs in a **fresh child process** with
   an ``RLIMIT_AS`` address-space cap (10 GB by default) and a per-child timeout,
   so one measurement's heap, JIT cache or allocator state never leaks into the
   next, and an out-of-memory failure is a recorded outcome rather than a dead
   parent. Peak RSS is the child's own ``ru_maxrss``; when the child dies before
   reporting, the parent's ``/proc/<pid>/status`` ``VmHWM`` watchdog supplies it.
4. **The provenance.** ``_profile_cli.device_info_dict`` (with its
   ``device.provenance`` block) and ``machine_info_dict``, plus the release tag
   of each library checkout (``git describe --tags``), because source checkouts
   report a stale ``__version__``.

``RLIMIT_AS`` caps *virtual* address space, not resident memory. JAX and the
allocator reserve address space they never touch, so the cap fails a run before
its RSS reaches 10 GB. That makes it a conservative ceiling, and the cells record
peak RSS beside the outcome so the two are never confused.

Run as ``python _streaming.py --child '<json spec>'`` only by the runner.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def profiling_root() -> Path:
    for p in HERE.parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = profiling_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "streaming_scaling"

SEED = 1234
CAP_GB_DEFAULT = 10.0
MASK_SHAPE = (400, 400)
PIXEL_SCALE = 0.05
MASK_RADIUS = 10.0
MESH_SHAPE = (20, 20)
REGULARIZATION_COEFFICIENT = 1.0

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "XLA_FLAGS",
)

LIBRARY_DIRS = ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens")


def dataset_description() -> dict:
    return {
        "kind": "synthetic, seeded per chunk (default_rng(SEED + k))",
        "seed": SEED,
        "uv_wavelengths": "uniform in [-1e5, 1e5] for u and v",
        "data": "1e-3 * (N(0,1) + i N(0,1))",
        "noise_map": "1 + 1j",
        "mask": f"circular, shape {MASK_SHAPE}, {PIXEL_SCALE} arcsec/pix, radius {MASK_RADIUS} arcsec",
        "transformer": "TransformerNUFFT",
    }


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _gen_chunk(np, k: int, n: int):
    rng = np.random.default_rng(SEED + k)
    uv = rng.uniform(-1e5, 1e5, size=(n, 2))
    data = 1e-3 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    noise = np.full(n, 1.0 + 1.0j)
    return uv, data, noise


def _chunks(np, n_vis: int, chunk: int):
    k = 0
    done = 0
    while done < n_vis:
        n = min(chunk, n_vis - done)
        yield _gen_chunk(np, k, n)
        k += 1
        done += n


def _full_arrays(np, n_vis: int, chunk: int):
    uv = np.empty((n_vis, 2))
    data = np.empty(n_vis, complex)
    noise = np.empty(n_vis, complex)
    done = 0
    for u, d, s in _chunks(np, n_vis, chunk):
        n = len(d)
        uv[done : done + n] = u
        data[done : done + n] = d
        noise[done : done + n] = s
        done += n
    return uv, data, noise


def _mask(aa):
    return aa.Mask2D.circular(shape_native=MASK_SHAPE, pixel_scales=PIXEL_SCALE, radius=MASK_RADIUS)


def _build_stream(np, aa, n_vis, chunk, mask):
    return aa.Interferometer.from_stream(
        _chunks(np, n_vis, chunk), mask, transformer_class=aa.TransformerNUFFT
    )


def _build_memory(np, aa, n_vis, chunk, mask, nufft_chunk_size=None):
    uv, data, noise = _full_arrays(np, n_vis, chunk)
    dataset = aa.Interferometer(
        data=aa.Visibilities(visibilities=data),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=noise),
        uv_wavelengths=uv,
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
    )
    del uv, data, noise
    return dataset.apply_sparse_operator(nufft_chunk_size=nufft_chunk_size)


def _log_evidence(aa, dataset, mask) -> dict:
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        overlay_grid_from,
    )

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=1)
    mesh = aa.mesh.RectangularUniform(shape=MESH_SHAPE)
    mapper = aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=aa.Grid2DIrregular(
                overlay_grid_from(shape_native=MESH_SHAPE, grid=grid)
            ),
            adapt_data=None,
        ),
        regularization=aa.reg.Constant(coefficient=REGULARIZATION_COEFFICIENT),
    )
    inversion = aa.Inversion(dataset=dataset, linear_obj_list=[mapper])
    fit = aa.m.MockFitInterferometer(dataset=dataset, inversion=inversion)
    return {
        "log_evidence": float(fit.log_evidence),
        "inversion_class": type(inversion).__name__,
    }


def _profile(np, aa, n_vis, chunk, mask, top: int = 10) -> dict:
    import cProfile
    import pstats

    from autoarray.inversion.inversion.interferometer import (
        inversion_interferometer_util as util,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    util.sparse_terms_from_chunks(
        _chunks(np, n_vis, chunk),
        real_space_mask=mask,
        transformer_class=aa.TransformerNUFFT,
    )
    profiler.disable()
    stats = pstats.Stats(profiler)
    total = stats.total_tt

    def rows(key):
        entries = []
        for (filename, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
            entries.append(
                {
                    "function": f"{Path(filename).name}:{line}({name})",
                    "ncalls": nc,
                    "tottime_s": round(tt, 4),
                    "cumtime_s": round(ct, 4),
                    "tottime_frac": round(tt / total, 4) if total else None,
                }
            )
        entries.sort(key=lambda e: e[key], reverse=True)
        return entries[:top]

    return {
        "profiled_n_vis": n_vis,
        "profiled_chunk": chunk,
        "n_chunks": -(-n_vis // chunk),
        "profiled_total_s": round(total, 3),
        "top_tottime": rows("tottime_s"),
        "top_cumtime": rows("cumtime_s"),
    }


def _failure_site(exc: BaseException) -> str | None:
    """The innermost PyAuto* frame of a traceback, ``file:line(function)``."""
    site = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if "/autoarray/" in frame.filename or "/autolens/" in frame.filename:
            site = f"{Path(frame.filename).name}:{frame.lineno}({frame.name})"
    return site


def _child(spec: dict) -> None:
    cap = spec.get("cap_bytes")
    if cap:
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    t0 = time.perf_counter()
    import autoarray as aa
    import numpy as np

    mask = _mask(aa)
    out = {
        "mode": spec["mode"],
        "n_vis": spec["n_vis"],
        "chunk": spec["chunk"],
        "nufft_chunk_size": spec.get("nufft_chunk_size"),
        "pixels_in_mask": int(mask.pixels_in_mask),
        "import_s": round(time.perf_counter() - t0, 2),
    }
    mode = spec["mode"]
    t1 = time.perf_counter()
    try:
        if mode == "stream":
            _build_stream(np, aa, spec["n_vis"], spec["chunk"], mask)
        elif mode == "memory":
            _build_memory(np, aa, spec["n_vis"], spec["chunk"], mask, spec.get("nufft_chunk_size"))
        elif mode == "stream_evidence":
            dataset = _build_stream(np, aa, spec["n_vis"], spec["chunk"], mask)
            out["build_s"] = round(time.perf_counter() - t1, 2)
            out.update(_log_evidence(aa, dataset, mask))
        elif mode == "memory_evidence":
            dataset = _build_memory(
                np, aa, spec["n_vis"], spec["chunk"], mask, spec.get("nufft_chunk_size")
            )
            out["build_s"] = round(time.perf_counter() - t1, 2)
            out.update(_log_evidence(aa, dataset, mask))
        elif mode == "profile":
            out["profile"] = _profile(np, aa, spec["n_vis"], spec["chunk"], mask)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out["outcome"] = "OK"
    except MemoryError as exc:
        traceback.print_exc()
        out["outcome"] = "FAILED_MEM"
        out["error"] = f"MemoryError: {str(exc)[:300]}"
    except Exception as exc:  # noqa: BLE001 — a failure under the cap is a result
        traceback.print_exc()
        out["failed_in"] = _failure_site(exc)
        message = f"{type(exc).__name__}: {str(exc)[-400:]}"
        oom = "out of memory" in message.lower() or "resource_exhausted" in message.lower()
        out["outcome"] = "FAILED_MEM" if oom else "FAILED"
        out["error"] = message
    out["wall_s"] = round(time.perf_counter() - t1, 2)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print("RESULT " + json.dumps(out), flush=True)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def run_child(
    mode: str,
    n_vis: int,
    chunk: int,
    *,
    cap_gb: float | None = CAP_GB_DEFAULT,
    timeout_s: float = 1800.0,
    nufft_chunk_size: int | None = None,
    log_dir: Path | None = None,
) -> dict:
    """Run one measurement in a fresh child process and return its record."""
    cap_bytes = int(cap_gb * 1e9) if cap_gb else None
    spec = {
        "mode": mode,
        "n_vis": int(n_vis),
        "chunk": int(chunk),
        "nufft_chunk_size": nufft_chunk_size,
        "cap_bytes": cap_bytes,
    }
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT)
    )
    hwm_kb = [0]

    def watch():
        status = f"/proc/{proc.pid}/status"
        while proc.poll() is None:
            try:
                with open(status) as fh:
                    for line in fh:
                        if line.startswith("VmHWM"):
                            hwm_kb[0] = max(hwm_kb[0], int(line.split()[1]))
            except OSError:
                pass
            time.sleep(0.25)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    t0 = time.perf_counter()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    child_wall = time.perf_counter() - t0
    label = f"{mode}_{n_vis:.0e}_{chunk}" + (
        f"_nufft{nufft_chunk_size}" if nufft_chunk_size else ""
    )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{label}.log").write_text(stdout + "\n---STDERR---\n" + stderr[-20000:])
    record = None
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            record = json.loads(line[len("RESULT ") :])
    if record is None:
        tail = [ln for ln in stderr.strip().splitlines() if ln.strip()][-1:] or [""]
        oom = "out of memory" in stderr.lower() or "memoryerror" in stderr.lower()
        record = {
            "mode": mode,
            "n_vis": int(n_vis),
            "chunk": int(chunk),
            "nufft_chunk_size": nufft_chunk_size,
            "outcome": "TIMEOUT"
            if timed_out
            else ("FAILED_MEM" if oom else f"FAILED(rc={proc.returncode})"),
            "error": tail[0][-400:],
            "wall_s": None,
            "peak_rss_mb": round(hwm_kb[0] / 1024),
            "peak_rss_source": "parent /proc VmHWM watchdog (child did not report)",
        }
    else:
        record["peak_rss_source"] = "child ru_maxrss"
    record["child_wall_s"] = round(child_wall, 2)
    record["cap_gb"] = cap_gb
    record["timeout_s"] = timeout_s
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if mode in ("stream", "memory") and record.get("wall_s") and n_vis:
        record["s_per_1e6_vis"] = round(record["wall_s"] / (n_vis / 1e6), 3)
    if record.get("peak_rss_mb"):
        record["peak_rss_gb"] = round(record["peak_rss_mb"] / 1024, 3)
    print(json.dumps(record), flush=True)
    return record


def release_tags() -> dict:
    """``git describe --tags`` of each PyAuto* library checkout on ``PYTHONPATH``."""
    out = {}
    for name in LIBRARY_DIRS:
        try:
            module = {
                "PyAutoNerves": "autonerves",
                "PyAutoFit": "autofit",
                "PyAutoArray": "autoarray",
                "PyAutoGalaxy": "autogalaxy",
                "PyAutoLens": "autolens",
            }[name]
            import importlib.util

            spec = importlib.util.find_spec(module)
            repo = Path(spec.origin).resolve().parent.parent
            out[name] = (
                subprocess.check_output(
                    ["git", "-C", str(repo), "describe", "--tags"],
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
                .decode()
                .strip()
            )
        except Exception as exc:  # noqa: BLE001 — provenance never loses a result
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


def release_version() -> str:
    """The PyAutoArray release this run measured, for the ``_v<version>`` file suffix.

    ``git describe --tags`` prints ``<tag>`` when the checkout sits on a tag and
    ``<tag>-<n>-g<sha>`` when it is ``n`` commits past one; the second form keeps the
    ``+<n>`` so an unreleased run can never pass as a release row.
    """
    tag = release_tags().get("PyAutoArray", "")
    if not tag or tag.startswith("unavailable"):
        return "unknown"
    parts = tag.split("-")
    return parts[0] if len(parts) == 1 else f"{parts[0]}+{parts[1]}"


def run_metadata(cli_args: dict) -> dict:
    """``device`` (with provenance) + ``machine`` + thread env + release tags."""
    from _profile_cli import device_info_dict, machine_info_dict

    return {
        "device": device_info_dict(),
        "machine": machine_info_dict(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "release_tags": release_tags(),
        "dataset": dataset_description(),
        "cli": cli_args,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def plot_series(png: Path, series: dict, title: str) -> None:
    """Two panels (wall s, peak RSS GB) vs N_vis, one line per series.

    ``series`` maps a label to its row list; failed rows are drawn as crosses at the
    RSS they reached, so a failure is visible rather than missing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, rows in series.items():
        ok = [r for r in rows if r.get("outcome") == "OK"]
        bad = [r for r in rows if r.get("outcome") not in ("OK", "SKIPPED_BUDGET")]
        if ok:
            line = ax_t.plot(
                [r["n_vis"] for r in ok], [r["wall_s"] for r in ok], "o-", label=label
            )[0]
            ax_m.plot(
                [r["n_vis"] for r in ok],
                [r["peak_rss_mb"] / 1024 for r in ok],
                "o-",
                color=line.get_color(),
                label=label,
            )
            color = line.get_color()
        else:
            color = None
        if bad:
            ax_m.plot(
                [r["n_vis"] for r in bad],
                [(r.get("peak_rss_mb") or 0) / 1024 for r in bad],
                "x",
                ms=10,
                mew=2,
                color=color,
                label=f"{label}: failed",
            )
    for ax, ylabel in ((ax_t, "wall [s]"), (ax_m, "peak RSS [GB]")):
        ax.set_xscale("log")
        ax.set_xlabel("N_vis")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax_t.set_yscale("log")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"wrote {png}")


def parse_int_list(text: str) -> list[int]:
    return [int(float(v)) for v in text.split(",") if v.strip()]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]))
    else:
        raise SystemExit("_streaming.py is the child entry point; run a cell instead.")
