"""Witness: does a single-process profiling cell time the Sibson code a Nautilus worker runs?

Nautilus parallelises with a Python ``multiprocessing`` pool. PyAutoFit evaluates the likelihood
once in the parent, then forks one worker process per core, and every sample is evaluated inside a
worker. Since PyAutoArray#677, ``autoarray.numba_util.parallel_kernel_allowed()`` is False in any
multiprocessing child (``multiprocessing.parent_process()`` is set), so a worker runs the Sibson
interpolator's **serial** compilation (``parallel=False``, no numba threading layer). The parent
and any single-process script (a profiling cell) run the ``prange`` compilation instead.

That makes "pin NUMBA_NUM_THREADS=1" an approximation: it times the ``prange`` kernel on one
thread, not the serial kernel a worker runs. The DelaunayNN numba profiling cells instead force
the worker route in-process (patch ``parallel_kernel_allowed`` to return False). This script
checks that the patch reproduces a real worker, by timing the Sibson kernel at the profiled HST
size in six legs, each in a fresh interpreter so numba's thread count and threading layer start
clean:

``parallel_default``   single process, ``prange`` kernel, NUMBA_NUM_THREADS unset (all cores)
``parallel_t8``        single process, ``prange`` kernel, NUMBA_NUM_THREADS=8 (the RAL 8-CPU task
                       the day-zero row ran in)
``parallel_t1``        single process, ``prange`` kernel, NUMBA_NUM_THREADS=1 (a thread pin only)
``patched_serial``     single process, ``parallel_kernel_allowed`` patched False, NUMBA_NUM_THREADS=1
                       (what the DelaunayNN profiling cells now do)
``fork_worker``        the Nautilus pattern: the parent evaluates once (``prange`` kernel), then a
                       ``fork`` pool of ONE worker times the calls; production thread env
                       (BLAS/OpenMP pinned to 1, NUMBA_NUM_THREADS unset)
``fork_workers_all``   as ``fork_worker`` but one worker per core, all timing at once:
                       informational only (memory-bandwidth contention, the node-throughput view)

Every leg pins the five BLAS/OpenMP variables with ``_production_config.pin_thread_env(1)``,
as production submit scripts and profiling cells do. Per leg it records the median wall time
per call, CPU-seconds per wall-second in the timing process (1.0 = one core), which Sibson
kernels were compiled in the timing process, numba's thread count and threading layer, and a
hash of the outputs.

PASS requires all of:
1. every leg's outputs bit-identical;
2. ``fork_worker`` and ``patched_serial`` both ran only the serial kernel, on one core
   (CPU/wall below 1.3);
3. ``patched_serial`` median within ``--tolerance`` (default 10%) of ``fork_worker``.

Inputs: the real HST source-plane DelaunayNN vertices in
``results/misc/delaunay_nn/mesh_points_hst.npy`` (1,500 points, see ``extract_mesh_points.py``),
subsampled to ``--mesh-points`` (default 1,280, the profiled cell's ``source_pixels``). Queries,
``--queries`` (default 61,600, the profiled over-sampled grid), are scattered round the vertices
at the local vertex spacing, mimicking a traced image grid concentrated where the mesh is. Only
the route comparison matters here, not the absolute cost.

Needs a PyAutoArray carrying #677; without ``parallel_kernel_allowed`` it exits with an error.

Run from the repository root::

    python scripts/misc/delaunay_nn/worker_route_witness.py
    python scripts/misc/delaunay_nn/worker_route_witness.py --repeats 5 --queries 5000   # quick

Writes ``results/misc/delaunay_nn/worker_route_witness_<host>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()

LEGS = (
    "parallel_default",
    "parallel_t8",
    "parallel_t1",
    "patched_serial",
    "fork_worker",
    "fork_workers_all",
    "paired_kernels",
    "paired_patch_worker",
)

#: NUMBA_NUM_THREADS per leg; None = unset (numba defaults to the cores it can see).
LEG_NUMBA_THREADS = {
    "parallel_default": None,
    "parallel_t8": "8",
    "parallel_t1": "1",
    "patched_serial": "1",
    "fork_worker": None,
    "fork_workers_all": None,
    "paired_kernels": "1",
    "paired_patch_worker": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh-points", type=int, default=1280)
    parser.add_argument("--queries", type=int, default=61600)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10, help="timed calls per leg per round")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--leg", choices=LEGS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--leg-out", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# One leg, in its own interpreter
# ---------------------------------------------------------------------------


def run_leg(args):
    """Runs inside a fresh interpreter. Thread variables are set before numpy/numba import."""
    sys.path.insert(0, str(ROOT))
    from _production_config import pin_thread_env

    thread_env = pin_thread_env(1)
    want = LEG_NUMBA_THREADS[args.leg]
    if want is None:
        os.environ.pop("NUMBA_NUM_THREADS", None)
    else:
        os.environ["NUMBA_NUM_THREADS"] = want

    import hashlib
    import multiprocessing as mp

    import numba
    import numpy as np
    from autoarray import numba_util
    from autoarray.inversion.mesh.interpolator import sibson
    from autoarray.inversion.mesh.interpolator.delaunay import (
        pix_indexes_delaunay_walk_from,
        scipy_delaunay_tri_only,
    )
    from autoarray.inversion.mesh.mesh.delaunay_nn import DelaunayNN

    if not hasattr(numba_util, "parallel_kernel_allowed"):
        raise SystemExit(
            "This PyAutoArray predates #677 (no numba_util.parallel_kernel_allowed): "
            "there is no serial worker route to witness."
        )

    if args.leg == "patched_serial":
        numba_util.parallel_kernel_allowed = lambda: False

    # Record the route each call TAKES (the ``parallel=`` argument). A forked child inherits the
    # parent's compiled-kernel cache, numba thread count and threading layer, so those cannot
    # show which kernel a worker ran; this can.
    _original_kernel = sibson.sibson_numba_kernel

    def _recording_kernel(parallel=True):
        _ROUTES_CALLED.append("prange" if parallel else "serial")
        return _original_kernel(parallel=parallel)

    sibson.sibson_numba_kernel = _recording_kernel

    rng = np.random.default_rng(7)
    all_points = np.load(ROOT / "results" / "misc" / "delaunay_nn" / "mesh_points_hst.npy")
    pick = rng.choice(len(all_points), size=min(args.mesh_points, len(all_points)), replace=False)
    points = np.ascontiguousarray(all_points[np.sort(pick)], dtype=np.float64)

    # Queries round the vertices at the local spacing (a traced grid concentrated on the mesh).
    simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)
    from scipy.spatial import cKDTree

    spacing = cKDTree(points).query(points, k=2)[0][:, 1]
    owner = rng.integers(0, len(points), size=args.queries)
    queries = points[owner] + rng.normal(size=(args.queries, 2)) * spacing[owner, None]
    queries = np.ascontiguousarray(queries, dtype=np.float64)

    mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=queries,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )
    kwargs = dict(
        query_points=queries,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=mappings[:, 0],
        circumcircles=sibson.delaunay_circumcircles_from(points, simplices, xp=np),
        max_cavity_triangles=DelaunayNN.max_cavity_triangles,
        max_neighbors=DelaunayNN.max_neighbors,
    )

    def call():
        return sibson._sibson_mappings_weights_numba(**kwargs)

    if args.leg.startswith("fork"):
        # The PyAutoFit pattern: one parent evaluation, then fork the workers.
        call()
        n_workers = 1 if args.leg == "fork_worker" else (os.cpu_count() or 1)
        _WORKER_STATE["call"] = call
        _WORKER_STATE["args"] = args
        with mp.get_context("fork").Pool(n_workers) as pool:
            per_worker = pool.map(_time_in_worker, range(n_workers), chunksize=1)
        result = dict(per_worker[0])
        result["n_workers"] = n_workers
        if n_workers > 1:
            result["per_worker_median_s"] = [w["median_s"] for w in per_worker]
            result["median_s"] = float(np.median([w["median_s"] for w in per_worker]))
            result["cpu_per_wall"] = float(np.median([w["cpu_per_wall"] for w in per_worker]))
        result["in_multiprocessing_child"] = all(w["in_multiprocessing_child"] for w in per_worker)
        result["routes_called_all_workers"] = sorted(
            {r for w in per_worker for r in w["routes_called"]}
        )
    elif args.leg == "paired_patch_worker":
        result = _time_patch_vs_worker(call, args, numba_util)
        result["n_workers"] = 1
    elif args.leg == "paired_kernels":
        result = _time_paired(call, args, numba_util)
        result["n_workers"] = 0
    else:
        result = _time_calls(call, args)
        result["n_workers"] = 0

    result.update(
        leg=args.leg,
        numba_num_threads_env=os.environ.get("NUMBA_NUM_THREADS"),
        blas_thread_env=thread_env,
        numba_version=numba.__version__,
        points=int(len(points)),
        queries=int(len(queries)),
        output_sha256=_hash_outputs(call(), hashlib, np),
    )
    args.leg_out.write_text(json.dumps(result, indent=2))


_WORKER_STATE: dict = {}

#: The route of every Sibson call in this process, appended by the recording wrapper.
_ROUTES_CALLED: list = []


def _time_in_worker(_index):
    import multiprocessing as mp

    result = _time_calls(_WORKER_STATE["call"], _WORKER_STATE["args"])
    result["in_multiprocessing_child"] = mp.parent_process() is not None
    return result


def _one_timed_call_in_worker(_index):
    import multiprocessing as mp

    del _ROUTES_CALLED[:]
    t0 = time.perf_counter()
    _WORKER_STATE["call"]()
    return time.perf_counter() - t0, list(_ROUTES_CALLED), mp.parent_process() is not None


def _time_patch_vs_worker(call, args, numba_util):
    """Patched parent vs a real forked worker, alternating call by call.

    The parent evaluates once on its default route (as PyAutoFit does), forks a persistent
    one-process pool, and only THEN patches itself to the serial route, so the worker takes the
    serial route through its own ``parent_process()`` check, not through the patch. Calls then
    alternate parent / worker, each timed where it runs; the parent is idle while the worker
    computes, so the two never compete for the CPU.
    """
    import multiprocessing as mp

    import numpy as np

    call()
    _WORKER_STATE["call"] = call
    with mp.get_context("fork").Pool(1) as pool:
        numba_util.parallel_kernel_allowed = lambda: False
        for _ in range(args.warmup):
            call()
            pool.apply(_one_timed_call_in_worker, (0,))
        patched, worker, worker_routes, in_child = [], [], set(), True
        del _ROUTES_CALLED[:]
        for i in range(args.repeats):
            for side in ("parent", "worker") if i % 2 == 0 else ("worker", "parent"):
                if side == "parent":
                    t0 = time.perf_counter()
                    call()
                    patched.append(time.perf_counter() - t0)
                else:
                    wall, routes, child = pool.apply(_one_timed_call_in_worker, (0,))
                    worker.append(wall)
                    worker_routes.update(routes)
                    in_child = in_child and child
        parent_routes = sorted(set(_ROUTES_CALLED))
    ratios = [a / b for a, b in zip(patched, worker)]
    result = {
        "patched_walls_s": patched,
        "worker_walls_s": worker,
        "patched_over_worker_per_pair": ratios,
        "patched_over_worker": float(np.median(ratios)),
        "median_s": float(np.median(patched)),
        "worker_median_s": float(np.median(worker)),
        "parent_routes_called": parent_routes,
        "worker_routes_called": sorted(worker_routes),
        "routes_called": sorted(set(parent_routes) | worker_routes),
        "in_multiprocessing_child": in_child,
        "cpu_per_wall": None,
        "numba_get_num_threads": None,
        "numba_threading_layer": None,
    }
    return result


def _time_paired(call, args, numba_util):
    """Alternate the serial and prange kernels call by call in ONE process (1 thread).

    Slow machine drift hits both kernels equally, so the per-pair ratio isolates the
    difference between the two compilations of the same body.
    """
    import numpy as np

    def with_route(parallel):
        numba_util.parallel_kernel_allowed = (lambda: True) if parallel else (lambda: False)
        call()

    for _ in range(args.warmup):
        with_route(False)
        with_route(True)
    serial, prange_ = [], []
    for i in range(args.repeats):
        order = (False, True) if i % 2 == 0 else (True, False)
        for parallel in order:
            t0 = time.perf_counter()
            with_route(parallel)
            (prange_ if parallel else serial).append(time.perf_counter() - t0)
    numba_util.parallel_kernel_allowed = lambda: True
    ratios = [s / p for s, p in zip(serial, prange_)]
    result = _time_calls(call, argparse.Namespace(warmup=0, repeats=1))
    result.update(
        serial_walls_s=serial,
        prange_t1_walls_s=prange_,
        serial_over_prange_t1_per_pair=ratios,
        median_s=float(np.median(serial)),
        prange_t1_median_s=float(np.median(prange_)),
        serial_over_prange_t1=float(np.median(ratios)),
        routes_called=["prange", "serial"],
    )
    return result


def _time_calls(call, args):
    import numba
    import numpy as np
    from autoarray.inversion.mesh.interpolator import sibson

    for _ in range(args.warmup):
        call()
    del _ROUTES_CALLED[:]
    walls, cpus = [], []
    for _ in range(args.repeats):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        call()
        walls.append(time.perf_counter() - wall0)
        cpus.append(time.process_time() - cpu0)
    try:
        layer = numba.threading_layer()
    except ValueError:  # no parallel region launched in this process
        layer = None
    return {
        # The route of every TIMED call in this process.
        "routes_called": sorted(set(_ROUTES_CALLED)),
        "n_calls_timed": len(_ROUTES_CALLED),
        "median_s": float(np.median(walls)),
        "walls_s": walls,
        "cpu_per_wall": float(np.sum(cpus) / np.sum(walls)),
        # Inherited across fork, so NOT the route a forked worker took (see routes_called).
        "kernel_cache_keys": sorted(str(k) for k in sibson._SIBSON_NUMBA_CACHE),
        "numba_get_num_threads": int(numba.get_num_threads()),
        "numba_threading_layer": layer,
        "pid": os.getpid(),
    }


def _hash_outputs(outputs, hashlib, np):
    digest = hashlib.sha256()
    for array in outputs:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Orchestrator (stdlib only)
# ---------------------------------------------------------------------------


def orchestrate(args):
    rounds = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in range(args.rounds):
            # Rotate the leg order each round so no leg always runs first or last.
            order = LEGS[r % len(LEGS) :] + LEGS[: r % len(LEGS)]
            legs = {}
            for leg in order:
                out = Path(tmp) / f"r{r}_{leg}.json"
                cmd = [
                    sys.executable,
                    __file__,
                    "--leg",
                    leg,
                    "--leg-out",
                    str(out),
                    "--mesh-points",
                    str(args.mesh_points),
                    "--queries",
                    str(args.queries),
                    "--repeats",
                    str(args.repeats),
                    "--warmup",
                    str(args.warmup),
                ]
                env = dict(os.environ)
                env.pop("NUMBA_NUM_THREADS", None)
                print(f"[witness] round {r + 1}/{args.rounds}: leg {leg}", flush=True)
                subprocess.run(cmd, check=True, env=env, cwd=ROOT)
                legs[leg] = json.loads(out.read_text())
            rounds.append({leg: legs[leg] for leg in LEGS})

    summary = summarise(rounds)
    checks = verdict(rounds, summary, args.tolerance)
    doc = {
        "measurement": "delaunay_nn_worker_route_witness",
        "question": (
            "Does forcing parallel_kernel_allowed() False in a single process reproduce the "
            "Sibson cost a Nautilus multiprocessing worker pays?"
        ),
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "settings": {
            "mesh_points": args.mesh_points,
            "queries": args.queries,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "tolerance": args.tolerance,
        },
        "library": _library_identity(),
        "summary": summary,
        "checks": checks,
        "verdict": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL",
        "rounds": rounds,
    }

    output_dir = args.output_dir or ROOT / "results" / "misc" / "delaunay_nn"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"worker_route_witness_{_slug(socket.gethostname())}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")

    print()
    print("Per-leg median of round medians [min-max over rounds], ms:")
    for leg in LEGS:
        row = summary["legs"][leg]
        print(
            f"  {leg:<18} {row['median_ms']:>8.1f} [{row['min_ms']:.1f}-{row['max_ms']:.1f}]  "
            f"cpu/wall {row['cpu_per_wall'] if row['cpu_per_wall'] is None else round(row['cpu_per_wall'], 2)}  route {','.join(row['routes_called'])}"
        )
    for name, ratio in summary["paired_ratios"].items():
        print(
            f"  ratio {name}: median {ratio['median']:.3f} "
            f"[{ratio['min']:.3f}-{ratio['max']:.3f}] over rounds"
        )
    print()
    for name, check in checks.items():
        print(f"  [{'ok' if check['ok'] else 'FAIL'}] {name}: {check['detail']}")
    print(f"\nVerdict: {doc['verdict']}  ->  {path}")
    return 0 if doc["verdict"] == "PASS" else 1


def _stats(values):
    values = sorted(values)
    n = len(values)
    median = values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    return {"median": median, "min": values[0], "max": values[-1], "per_round": values}


def summarise(rounds):
    legs = {}
    for leg in LEGS:
        medians = [r[leg]["median_s"] * 1e3 for r in rounds]
        stats = _stats(medians)
        legs[leg] = {
            "median_ms": stats["median"],
            "min_ms": stats["min"],
            "max_ms": stats["max"],
            "round_medians_ms": [r[leg]["median_s"] * 1e3 for r in rounds],
            "cpu_per_wall": (
                _stats([r[leg]["cpu_per_wall"] for r in rounds])["median"]
                if all(r[leg]["cpu_per_wall"] is not None for r in rounds)
                else None
            ),
            "routes_called": sorted({x for r in rounds for x in r[leg]["routes_called"]}),
        }
    paired = {
        # Same round, back to back: drift between rounds cancels.
        "patched_serial / fork_worker": _stats(
            [r["patched_serial"]["median_s"] / r["fork_worker"]["median_s"] for r in rounds]
        ),
        # Same process, call by call: the two compilations of one body.
        "serial / prange@1 (paired_kernels)": _stats(
            [r["paired_kernels"]["serial_over_prange_t1"] for r in rounds]
        ),
        # One parent, one persistent forked worker, call by call: the decisive comparison.
        "patched / worker (paired_patch_worker)": _stats(
            [r["paired_patch_worker"]["patched_over_worker"] for r in rounds]
        ),
        "parallel_t1 / fork_worker": _stats(
            [r["parallel_t1"]["median_s"] / r["fork_worker"]["median_s"] for r in rounds]
        ),
        "fork_worker / parallel_t8 (worker vs the day-zero condition)": _stats(
            [r["fork_worker"]["median_s"] / r["parallel_t8"]["median_s"] for r in rounds]
        ),
    }
    return {"legs": legs, "paired_ratios": paired}


def verdict(rounds, summary, tolerance):
    hashes = {r[leg]["output_sha256"] for r in rounds for leg in LEGS}
    one_core = {
        leg: summary["legs"][leg]["routes_called"] == ["serial"]
        and all(r[leg]["cpu_per_wall"] < 1.3 for r in rounds)
        for leg in ("fork_worker", "patched_serial", "fork_workers_all")
    }
    in_child = all(
        r[leg].get("in_multiprocessing_child") for r in rounds for leg in ("fork_worker",)
    )
    ratio = summary["paired_ratios"]["patched / worker (paired_patch_worker)"]
    paired_routes_ok = all(
        r["paired_patch_worker"]["parent_routes_called"] == ["serial"]
        and r["paired_patch_worker"]["worker_routes_called"] == ["serial"]
        and r["paired_patch_worker"]["in_multiprocessing_child"]
        for r in rounds
    )
    return {
        "outputs_identical": {
            "ok": len(hashes) == 1,
            "detail": "all legs, all rounds bit-identical"
            if len(hashes) == 1
            else f"{len(hashes)} distinct output hashes",
        },
        "fork_worker_ran_in_multiprocessing_child": {
            "ok": in_child,
            "detail": "multiprocessing.parent_process() set in every fork_worker round",
        },
        "worker_and_patch_serial_one_core": {
            "ok": all(one_core.values()),
            "detail": ", ".join(
                f"{leg}: routes={summary['legs'][leg]['routes_called']} "
                f"cpu/wall={summary['legs'][leg]['cpu_per_wall']:.2f}"
                for leg in one_core
            ),
        },
        "paired_patch_worker_both_serial": {
            "ok": paired_routes_ok,
            "detail": "patched parent and forked worker both took only the serial route",
        },
        "patch_matches_worker": {
            "ok": abs(ratio["median"] - 1.0) <= tolerance,
            "detail": (
                f"paired patched / worker median {ratio['median']:.3f} over rounds "
                f"[{ratio['min']:.3f}-{ratio['max']:.3f}] (tolerance ±{tolerance:.0%})"
            ),
        },
    }


def _library_identity():
    identity = {}
    for name, repo in (("autoarray", "PyAutoArray"),):
        try:
            sha = subprocess.run(
                ["git", "-C", str(ROOT.parent / repo), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sha = None
        identity[name] = {"git_sha": sha}
    return identity


def _slug(value):
    return "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")


if __name__ == "__main__":
    _args = parse_args()
    if _args.leg is not None:
        run_leg(_args)
        raise SystemExit(0)
    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        print(f"[smoke] {__file__}: argument parsing OK; exiting.")
        raise SystemExit(0)
    raise SystemExit(orchestrate(_args))
