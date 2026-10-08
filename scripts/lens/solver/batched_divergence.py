"""
Linear-solver batched divergence: does ``jit(vmap)`` solve what ``jit`` solves, lane by lane?
=============================================================================================

Phase 4a of the linear-solver programme (autolens_profiling#397). Phase 3b found that on the A100
the Jacobi PDIP (``pdip_jacobi``, the library ``"jacobi"`` mode every Mapper inversion uses)
follows a different trajectory inside ``jax.jit(jax.vmap(solve))`` over 50 distinct SLaM systems
than when each system is solved alone (19/50 lanes in iterations, 13/50 in the flag), while CPU
batched == unbatched bit for bit and the released raw PDIP is identical everywhere. This probe
localises that. It is research only: it calls the library's solver entry points exactly as
``_solvers`` composes them and never transcribes PDIP internals.

Probes
------
``determinism``
    The batched solve over the B lanes twice, each through a *fresh* ``jax.jit(jax.vmap(...))``
    (new trace + compile), plus a second call of the first executable; the unbatched solve of every
    lane twice through fresh ``jax.jit``. Per lane: bitwise equality of ``x``, ``iterations`` and
    the flag; and batched vs unbatched (the phase-3b finding, reproduced). ``--deterministic``
    reruns everything in a separate process with ``XLA_FLAGS=--xla_gpu_deterministic_ops=true``
    set before JAX is imported (artefact suffix ``_det``).
``lowering``
    Each lane through ``jit(vmap)`` at B = 1 against plain ``jit``: does the vmap lowering alone
    change the result?
``tiled``
    One system tiled B times (``euclid_vis_lp`` and the first SLaM lane) at ``--tile-sizes``:
    are the lanes identical to each other, and to the unbatched solve?
``trajectory`` (``pdip_jacobi`` only)
    The library ``solve_nnls`` on the lane's Jacobi system (``_solvers._jacobi``, the quantities
    ``reconstruction_positive_only_from`` builds) with ``max_iter = k`` for k = 0..``--max-iter``,
    batched and unbatched. ``solve_nnls`` is the ``while_loop`` driver that
    ``solve_nnls_primal_with_status`` (the ``pdip_jacobi`` candidate) runs as its primal, and a
    ``while_loop`` capped at k reproduces the first k iterations of a longer run, so k indexes the
    PDIP trajectory. k = 0 is ``initialize`` alone. Per lane: the first k at which the batched
    and unbatched states ``(x, s, z)`` differ in any bit, which of them differ, the relative and
    ulp size of that first difference, and ``||Δx|| / ||x||`` and the max ulp distance at every k.
    A consistency check confirms that k = ``--max-iter`` reproduces the candidate's own ``x``
    bit for bit (batched and unbatched).
``primitives`` (context)
    The linear-algebra primitives jaxnnls's PDIP is built from — ``jax.scipy.linalg.cho_factor``,
    ``cho_solve`` and a matrix-vector product — applied once to each lane's Jacobi system
    (``Q_pc + I``, ``q_pc``: ``initialize``'s first operations), ``jit`` vs ``jit(vmap)``,
    bitwise per lane. Says which primitive's batched lowering differs from its unbatched one.

Join (per lane): ``cond_Q`` from the corpus manifest, ``cond(Q_pc)`` of the Jacobi-scaled system
(computed here, NumPy), the phase-3a ``pdip_jacobi`` divergence membership (unconverged on CPU
only / A100 only / both / neither, from ``accuracy_summary_all[_gpu]_v2026.10.7.1.json``) and the
phase-3b batched flag on each device (``timing_summary_all[_gpu]_v2026.10.7.1.json``).

No timings are reported. Batches take the first ``--batch-size`` distinct systems of the pooled
multi-system groups in manifest order (by default ``slam_fixture_571`` + ``slam48_hst``, the
phase-3b batch).

Run from the repo root::

    python scripts/lens/solver/batched_divergence.py                       # CPU (control)
    python scripts/lens/solver/batched_divergence.py --device gpu
    python scripts/lens/solver/batched_divergence.py --device gpu --deterministic

Output
------
``results/lens/solver/batched_divergence_summary_<corpus>[_gpu][_det]_v<version>.{json,png}``
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

import _driver  # noqa: E402

# --device must pin the JAX platform before anything imports jax.
_DEVICE = _driver.select_device_from_argv()

import os as _os  # noqa: E402

DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops=true"
# --deterministic must reach XLA_FLAGS before anything imports jax.
_DETERMINISTIC = "--deterministic" in _sys.argv[1:]
if _DETERMINISTIC and DETERMINISTIC_FLAG not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") + " " + DETERMINISTIC_FLAG).strip()

# AUTOLENS_PROFILING_SMOKE=1 short-circuit (CI lint smoke).
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import _corpus  # noqa: E402
import _solvers  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_GROUPS = ["slam_fixture_571", "slam48_hst", "euclid_vis_lp"]
DEFAULT_CANDIDATES = ["pdip_jacobi", "pdip_raw", "certified"]
TRAJECTORY_CANDIDATE = "pdip_jacobi"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ITER = 50
DEFAULT_TILE_SIZES = [2, 8, 50]
#: The cap ``_solvers._build_pdip_jacobi`` passes (the library's ``nnls_max_iter`` default).
PDIP_JACOBI_MAX_ITER = 50
#: The phase-3a / 3b artefacts this probe joins against (released tag 2026.10.7.1).
JOIN_TAG = "2026.10.7.1"


def _parse_args(argv=None):
    """``_driver.parse_cli`` plus ``--batch-size``, ``--max-iter``, ``--tile-sizes``,
    ``--deterministic``."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    pre.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    pre.add_argument("--tile-sizes", type=int, nargs="+", default=DEFAULT_TILE_SIZES)
    pre.add_argument("--deterministic", action="store_true")
    known, rest = pre.parse_known_args(_sys.argv[1:] if argv is None else argv)
    args = _driver.parse_cli(
        __doc__.splitlines()[1]
        + " (also: --batch-size B, --max-iter K, --tile-sizes B [B ...], --deterministic)",
        DEFAULT_CANDIDATES,
        argv=rest,
    )
    if args.groups is None:
        args.groups = list(DEFAULT_GROUPS)
    if known.batch_size < 1 or known.max_iter < 1 or any(b < 1 for b in known.tile_sizes):
        raise SystemExit("--batch-size, --max-iter and --tile-sizes must be >= 1")
    args.batch_size = known.batch_size
    args.max_iter = known.max_iter
    args.tile_sizes = sorted(set(known.tile_sizes))
    args.deterministic = known.deterministic
    return args


# ---------------------------------------------------------------------------
# Bit-level comparison
# ---------------------------------------------------------------------------


def _bits(a):
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype == np.float64:
        return a.view(np.uint64)
    return a


def _same(a, b) -> bool:
    """Bitwise equality (NaN payloads and signed zeros included)."""
    a, b = _bits(a), _bits(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _ordered(a):
    """float64 -> int64 whose order is the float order, so differences count ulps."""
    i = np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.int64)
    return np.where(i < 0, np.int64(-(2**63)) - i, i)


def _max_ulp(a, b):
    """Largest ulp distance between ``a`` and ``b`` (float; ``None`` if any value is non-finite)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    # Exact integer differences: the ordered ints reach ~2**62, beyond float64's 53-bit
    # mantissa, so subtracting them as floats quantises the distance to ~512-1024 ulp (the
    # A100 artefacts at profiling 6fb885e carry that quantisation; see the ledger).
    oa, ob = _ordered(a).ravel().tolist(), _ordered(b).ravel().tolist()
    return float(max((abs(u - v) for u, v in zip(oa, ob)), default=0))


def _fin(v):
    """``float(v)`` if finite, else ``None`` (the JSON is written with ``allow_nan=False``)."""
    v = float(v)
    return v if np.isfinite(v) else None


def _rel(a, b):
    """``||a - b|| / ||b||`` (``None`` if non-finite)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        nb = float(np.linalg.norm(b))
        nd = float(np.linalg.norm(a - b))
    return _fin(nd / nb) if nb > 0 else _fin(nd)


def _maxabs(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    return _fin(np.max(d)) if np.all(np.isfinite(d)) else None


# ---------------------------------------------------------------------------
# Calls (fresh jit objects, so every call site is its own trace + compile)
# ---------------------------------------------------------------------------


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    return jax


def _stack(lanes):
    import jax.numpy as jnp

    jax = _jax()
    Q = jax.device_put(jnp.asarray(np.stack([s.Q for s in lanes]), dtype=jnp.float64))
    q = jax.device_put(jnp.asarray(np.stack([s.q for s in lanes]), dtype=jnp.float64))
    return jax.block_until_ready((Q, q))


def _single(s):
    import jax.numpy as jnp

    jax = _jax()
    Q = jax.device_put(jnp.asarray(s.Q, dtype=jnp.float64))
    q = jax.device_put(jnp.asarray(s.q, dtype=jnp.float64))
    return jax.block_until_ready((Q, q))


def _host(out):
    """A dict of device arrays -> a dict of NumPy arrays."""
    jax = _jax()
    out = jax.block_until_ready(out)
    return {k: np.asarray(v) for k, v in out.items()}


def _fresh_batched(body):
    jax = _jax()
    return jax.jit(jax.vmap(lambda Q, q: body(Q, q)))


def _fresh_single(body):
    jax = _jax()
    return jax.jit(lambda Q, q: body(Q, q))


def _status(out, i=None):
    """``(x, iterations, converged)`` of one lane (``i``) or of an unbatched output."""
    pick = (lambda v: v) if i is None else (lambda v: v[i])
    return (
        np.asarray(pick(out["x"]), dtype=np.float64),
        int(np.asarray(pick(out["iterations"]))),
        bool(np.asarray(pick(out["converged"]))),
    )


def _compare(a, b) -> dict:
    """Compare two ``(x, iterations, converged)`` triples."""
    xa, ia, ca = a
    xb, ib, cb = b
    return {
        "x_bitwise_equal": _same(xa, xb),
        "iterations_equal": ia == ib,
        "converged_equal": ca == cb,
        "rel_dx": _rel(xa, xb),
        "max_ulp_x": _max_ulp(xa, xb),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_determinism(cand, lanes, target_kappa):
    """Batched twice (fresh jit each) + a repeat call; unbatched twice (fresh jit each)."""
    body = cand.build(float(target_kappa))
    Q, q = _stack(lanes)
    f1 = _fresh_batched(body)
    b1 = _host(f1(Q, q))
    b1r = _host(f1(Q, q))
    b2 = _host(_fresh_batched(body)(Q, q))
    g1, g2 = _fresh_single(body), _fresh_single(body)
    per_lane, unbatched = [], []
    for i, s in enumerate(lanes):
        Qs, qs = _single(s)
        u1 = _host(g1(Qs, qs))
        u2 = _host(g2(Qs, qs))
        unbatched.append(u1)
        sb1, sb2, sb1r, su1, su2 = (
            _status(b1, i),
            _status(b2, i),
            _status(b1r, i),
            _status(u1),
            _status(u2),
        )
        per_lane.append(
            {
                "lane": i,
                "system": f"{s.group}/{s.name}",
                "batched_iterations": sb1[1],
                "batched_converged": sb1[2],
                "unbatched_iterations": su1[1],
                "unbatched_converged": su1[2],
                "batched_run1_vs_run2": _compare(sb1, sb2),
                "batched_run1_vs_repeat_call": _compare(sb1, sb1r),
                "unbatched_run1_vs_run2": _compare(su1, su2),
                "batched_vs_unbatched": _compare(sb1, su1),
            }
        )

    def count(key, field):
        return sum(1 for r in per_lane if not r[key][field])

    summary = {
        key: {
            "n_lanes_x_differ": count(key, "x_bitwise_equal"),
            "n_lanes_iterations_differ": count(key, "iterations_equal"),
            "n_lanes_converged_differ": count(key, "converged_equal"),
            "max_rel_dx": max(
                (r[key]["rel_dx"] for r in per_lane if r[key]["rel_dx"] is not None), default=None
            ),
        }
        for key in (
            "batched_run1_vs_run2",
            "batched_run1_vs_repeat_call",
            "unbatched_run1_vs_run2",
            "batched_vs_unbatched",
        )
    }
    summary["n_unconverged_batched"] = sum(1 for r in per_lane if not r["batched_converged"])
    summary["n_unconverged_unbatched"] = sum(1 for r in per_lane if not r["unbatched_converged"])
    return {"summary": summary, "lanes": per_lane}, b1, unbatched


def probe_lowering(cand, lanes, unbatched, target_kappa):
    """``jit(vmap)`` at B = 1 vs plain ``jit``, per lane."""
    body = cand.build(float(target_kappa))
    f = _fresh_batched(body)
    rows = []
    for i, s in enumerate(lanes):
        Qs, qs = _single(s)
        o = _host(f(Qs[None], qs[None]))
        rows.append(
            {
                "lane": i,
                "system": f"{s.group}/{s.name}",
                **_compare(_status(o, 0), _status(unbatched[i])),
            }
        )
    return {
        "summary": {
            "n_lanes": len(rows),
            "n_lanes_x_differ": sum(1 for r in rows if not r["x_bitwise_equal"]),
            "n_lanes_iterations_differ": sum(1 for r in rows if not r["iterations_equal"]),
            "n_lanes_converged_differ": sum(1 for r in rows if not r["converged_equal"]),
        },
        "lanes": rows,
    }


def probe_tiled(cand, systems, tile_sizes, target_kappa):
    """One system tiled B times: lane-to-lane and lane-vs-unbatched."""
    body = cand.build(float(target_kappa))
    rows = []
    for s in systems:
        Qs, qs = _single(s)
        u = _status(_host(_fresh_single(body)(Qs, qs)))
        for b in tile_sizes:
            o = _host(_fresh_batched(body)(*_stack([s] * b)))
            lanes = [_status(o, i) for i in range(b)]
            lane_eq = all(_same(lanes[0][0], ln[0]) for ln in lanes[1:])
            lane_dx = max((_maxabs(ln[0], lanes[0][0]) or 0.0 for ln in lanes[1:]), default=0.0)
            rows.append(
                {
                    "system": f"{s.group}/{s.name}",
                    "tile_size": b,
                    "lanes_bitwise_identical": lane_eq,
                    "lane_to_lane_max_abs_dx": lane_dx,
                    "lane_iterations": sorted({ln[1] for ln in lanes}),
                    "lane_converged": sorted({ln[2] for ln in lanes}),
                    "unbatched_iterations": u[1],
                    "unbatched_converged": u[2],
                    "lane0_vs_unbatched": _compare(lanes[0], u),
                }
            )
    return rows


def _trajectory_body(k):
    """The ``pdip_jacobi`` solve with ``max_iter = k``, returning the full PDIP state."""
    from autoarray.util.jax_nnls import solve_nnls

    def f(Q, q):
        Q_pc, q_pc, D = _solvers._jacobi(Q, q)
        y, s, z, converged, iterations = solve_nnls(Q_pc, q_pc, solver_tol=None, max_iter=k)
        return {
            "x": y * D,
            "y": y,
            "s": s,
            "z": z,
            "converged": converged,
            "iterations": iterations,
        }

    return f


def _mu(s, z):
    """The complementarity ``s . z / n`` (``None`` if non-finite)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _fin(np.dot(s, z) / len(s))


def probe_trajectory(lanes, max_iter, batched_final, unbatched_final):
    """First-difference trajectory of every lane, batched vs unbatched, k = 0..max_iter."""
    Q, q = _stack(lanes)
    singles = [_single(s) for s in lanes]
    ks = list(range(max_iter + 1))
    B, n = len(lanes), lanes[0].n
    st = {
        key: np.empty((len(ks), B, n)) for key in ("xb", "yb", "sb", "zb", "xu", "yu", "su", "zu")
    }
    itb, itu = np.empty((len(ks), B), int), np.empty((len(ks), B), int)
    cvb, cvu = np.empty((len(ks), B), bool), np.empty((len(ks), B), bool)
    for k in ks:
        body = _trajectory_body(k)
        ob = _host(_fresh_batched(body)(Q, q))
        g = _fresh_single(body)
        for v in ("x", "y", "s", "z"):
            st[v + "b"][k] = ob[v]
        itb[k], cvb[k] = ob["iterations"], ob["converged"].astype(bool)
        for i, (Qs, qs) in enumerate(singles):
            ou = _host(g(Qs, qs))
            for v in ("x", "y", "s", "z"):
                st[v + "u"][k, i] = ou[v]
            itu[k, i], cvu[k, i] = int(ou["iterations"]), bool(ou["converged"])
        if k % 10 == 0 or k == max_iter:
            print(f"  trajectory k={k}/{max_iter}", flush=True)

    rows = []
    for i, s in enumerate(lanes):

        def differs(k, v, i=i):
            return not _same(st[v + "b"][k, i], st[v + "u"][k, i])

        first_state = next((k for k in ks if any(differs(k, v) for v in ("y", "s", "z"))), None)
        first_x = next((k for k in ks if differs(k, "x")), None)
        at = None
        if first_state is not None:
            k = first_state
            mu_b, mu_u = _mu(st["sb"][k, i], st["zb"][k, i]), _mu(st["su"][k, i], st["zu"][k, i])
            at = {
                "k": k,
                "which_differ": [v for v in ("y", "s", "z") if differs(k, v)],
                "n_components_differ_y": int(
                    np.sum(_bits(st["yb"][k, i]) != _bits(st["yu"][k, i]))
                ),
                "rel_dy": _rel(st["yb"][k, i], st["yu"][k, i]),
                "rel_ds": _rel(st["sb"][k, i], st["su"][k, i]),
                "rel_dz": _rel(st["zb"][k, i], st["zu"][k, i]),
                "max_ulp_y": _max_ulp(st["yb"][k, i], st["yu"][k, i]),
                "max_ulp_s": _max_ulp(st["sb"][k, i], st["su"][k, i]),
                "max_ulp_z": _max_ulp(st["zb"][k, i], st["zu"][k, i]),
                "mu_batched": mu_b,
                "mu_unbatched": mu_u,
                "lane_iterations_batched": int(itb[k, i]),
                "lane_iterations_unbatched": int(itu[k, i]),
            }
        rows.append(
            {
                "lane": i,
                "system": f"{s.group}/{s.name}",
                "first_k_state_differs": first_state,
                "first_k_x_differs": first_x,
                "at_first_difference": at,
                "iterations_batched": int(itb[-1, i]),
                "iterations_unbatched": int(itu[-1, i]),
                "converged_batched": bool(cvb[-1, i]),
                "converged_unbatched": bool(cvu[-1, i]),
                "rel_dx_by_k": [_rel(st["xb"][k, i], st["xu"][k, i]) for k in ks],
                "max_ulp_x_by_k": [_max_ulp(st["xb"][k, i], st["xu"][k, i]) for k in ks],
                "mu_batched_by_k": [_mu(st["sb"][k, i], st["zb"][k, i]) for k in ks],
                "mu_unbatched_by_k": [_mu(st["su"][k, i], st["zu"][k, i]) for k in ks],
            }
        )
    consistency = {
        "against": "the pdip_jacobi candidate (solve_nnls_primal_with_status) at max_iter="
        f"{max_iter}: batched via a fresh jit(vmap) of _solvers' body, unbatched via fresh jit",
        "batched_x_bitwise_equal_lanes": sum(
            1 for i in range(B) if _same(st["xb"][-1, i], batched_final["x"][i])
        ),
        "unbatched_x_bitwise_equal_lanes": sum(
            1 for i in range(B) if _same(st["xu"][-1, i], unbatched_final[i]["x"])
        ),
        "n_lanes": B,
        "candidate_max_iter": PDIP_JACOBI_MAX_ITER,
        "applicable": max_iter == PDIP_JACOBI_MAX_ITER,
    }
    firsts = [r["first_k_state_differs"] for r in rows if r["first_k_state_differs"] is not None]
    summary = {
        "n_lanes": B,
        "n_lanes_state_ever_differs": len(firsts),
        "first_k_state_differs_histogram": {str(k): firsts.count(k) for k in sorted(set(firsts))},
        "n_lanes_final_iterations_differ": int(np.sum(itb[-1] != itu[-1])),
        "n_lanes_final_converged_differ": int(np.sum(cvb[-1] != cvu[-1])),
        "consistency": consistency,
    }
    return {"summary": summary, "lanes": rows}


def _primitive_body(Q, q):
    import jax.numpy as jnp
    import jax.scipy as jsp

    Q_pc, q_pc, D = _solvers._jacobi(Q, q)
    c, low = jsp.linalg.cho_factor(Q_pc + jnp.eye(Q_pc.shape[0]))
    return {
        "jacobi_Q_pc": Q_pc,
        "cho_factor": c,
        "cho_solve": jsp.linalg.cho_solve((c, low), q_pc),
        "matvec": Q_pc @ q_pc,
    }


def probe_primitives(lanes):
    """jaxnnls's linear-algebra primitives on each lane's Jacobi system, jit vs jit(vmap)."""
    ob = _host(_fresh_batched(_primitive_body)(*_stack(lanes)))
    g = _fresh_single(_primitive_body)
    keys = list(ob)
    rows = []
    for i, s in enumerate(lanes):
        ou = _host(g(*_single(s)))
        rows.append(
            {
                "lane": i,
                "system": f"{s.group}/{s.name}",
                **{
                    key: {"bitwise_equal": _same(ob[key][i], ou[key]), "max_ulp": None}
                    for key in keys
                },
            }
        )
        for key in keys:
            if not rows[-1][key]["bitwise_equal"]:
                rows[-1][key]["max_ulp"] = _max_ulp(ob[key][i], ou[key])
    return {
        "summary": {key: sum(1 for r in rows if not r[key]["bitwise_equal"]) for key in keys},
        "lanes": rows,
    }


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


def _load_json(name):
    path = _driver.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.is_file() else None


def _phase3_join(lanes):
    """Per lane: the 3a ``pdip_jacobi`` divergence set and the 3b batched flags, per device."""
    out = {}
    acc = {
        "cpu": _load_json(f"accuracy_summary_all_v{JOIN_TAG}.json"),
        "gpu": _load_json(f"accuracy_summary_all_gpu_v{JOIN_TAG}.json"),
    }
    tim = {
        "cpu": _load_json(f"timing_summary_all_v{JOIN_TAG}.json"),
        "gpu": _load_json(f"timing_summary_all_gpu_v{JOIN_TAG}.json"),
    }
    conv3a = {dev: {} for dev in acc}
    for dev, d in acc.items():
        for r in (d or {}).get("rows", []):
            if r["candidate"] == "pdip_jacobi":
                conv3a[dev][r["system"]] = r["converged"]
    conv3b = {dev: {} for dev in tim}
    for dev, d in tim.items():
        for r in (d or {}).get("rows", []):
            if r["candidate"] == "pdip_jacobi" and not r["tiled"] and r["batch_size"] == len(lanes):
                conv3b[dev] = dict(zip(r["lanes"], r["converged"]))
    for s in lanes:
        key = f"{s.group}/{s.name}"
        c, g = conv3a["cpu"].get(key), conv3a["gpu"].get(key)
        if c is None or g is None:
            member = None
        else:
            member = {
                (False, False): "both",
                (False, True): "cpu_only",
                (True, False): "a100_only",
                (True, True): "neither",
            }[(c, g)]
        out[key] = {
            "phase3a_jacobi_converged_cpu": c,
            "phase3a_jacobi_converged_a100": g,
            "phase3a_divergence_set": member,
            "phase3b_batched_converged_cpu": conv3b["cpu"].get(key),
            "phase3b_batched_converged_a100": conv3b["gpu"].get(key),
            "phase3b_a100_flag_batched_vs_unbatched_differs": (
                None if conv3b["gpu"].get(key) is None or g is None else conv3b["gpu"][key] != g
            ),
        }
    return out, {
        "accuracy": [f"accuracy_summary_all{sfx}_v{JOIN_TAG}.json" for sfx in ("", "_gpu")],
        "timing": [f"timing_summary_all{sfx}_v{JOIN_TAG}.json" for sfx in ("", "_gpu")],
        "found": {f"accuracy_{k}": v is not None for k, v in acc.items()}
        | {f"timing_{k}": v is not None for k, v in tim.items()},
    }


def _ranks(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    for val in np.unique(v):  # average ties
        m = v == val
        r[m] = r[m].mean()
    return r


def _auc(score, label):
    """P(score of a labelled lane > score of an unlabelled lane) (Mann-Whitney; ties count 1/2)."""
    score, label = np.asarray(score, dtype=np.float64), np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = _ranks(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def join(lanes, determinism, trajectory):
    phase3, sources = _phase3_join(lanes)
    det = {r["system"]: r for r in determinism["lanes"]}
    traj = {r["system"]: r for r in (trajectory or {"lanes": []})["lanes"]}
    rows = []
    for i, s in enumerate(lanes):
        key = f"{s.group}/{s.name}"
        d = np.sqrt(np.diag(s.Q))
        Q_pc = s.Q / d[:, None] / d[None, :]
        bu = det[key]["batched_vs_unbatched"]
        t = traj.get(key)
        rows.append(
            {
                "lane": i,
                "system": key,
                "cond_Q": s.meta.get("cond_Q"),
                "cond_Q_pc": float(np.linalg.cond(Q_pc)),
                "batched_vs_unbatched_x_differs": not bu["x_bitwise_equal"],
                "batched_vs_unbatched_iterations_differ": not bu["iterations_equal"],
                "batched_vs_unbatched_converged_differs": not bu["converged_equal"],
                "batched_converged": det[key]["batched_converged"],
                "unbatched_converged": det[key]["unbatched_converged"],
                "first_k_state_differs": None if t is None else t["first_k_state_differs"],
                **phase3[key],
            }
        )

    differs = [r["batched_vs_unbatched_x_differs"] for r in rows]
    unconv = [not r["unbatched_converged"] for r in rows]
    stats = {
        "auc_cond_Q_predicts_x_differs": _auc([r["cond_Q"] for r in rows], differs),
        "auc_cond_Q_pc_predicts_x_differs": _auc([r["cond_Q_pc"] for r in rows], differs),
        "auc_cond_Q_predicts_unbatched_unconverged": _auc([r["cond_Q"] for r in rows], unconv),
        "auc_cond_Q_pc_predicts_unbatched_unconverged": _auc(
            [r["cond_Q_pc"] for r in rows], unconv
        ),
        "median_cond_Q_pc_differs": _median([r["cond_Q_pc"] for r in rows if r[_DX]]),
        "median_cond_Q_pc_same": _median([r["cond_Q_pc"] for r in rows if not r[_DX]]),
        "median_cond_Q_differs": _median([r["cond_Q"] for r in rows if r[_DX]]),
        "median_cond_Q_same": _median([r["cond_Q"] for r in rows if not r[_DX]]),
    }
    with_k = [r for r in rows if r["first_k_state_differs"] is not None]
    stats["spearman_first_k_vs_cond_Q_pc"] = _spearman(
        [r["first_k_state_differs"] for r in with_k], [r["cond_Q_pc"] for r in with_k]
    )
    crosstab: dict = {}
    for r in rows:
        k = str(r["phase3a_divergence_set"])
        c = crosstab.setdefault(
            k, {"n": 0, "x_differs": 0, "iterations_differ": 0, "converged_differs": 0}
        )
        c["n"] += 1
        c["x_differs"] += int(r["batched_vs_unbatched_x_differs"])
        c["iterations_differ"] += int(r["batched_vs_unbatched_iterations_differ"])
        c["converged_differs"] += int(r["batched_vs_unbatched_converged_differs"])
    stats["by_phase3a_divergence_set"] = crosstab
    return {"sources": sources, "statistics": stats, "lanes": rows}


_DX = "batched_vs_unbatched_x_differs"


def _median(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(png_path, trajectory, join_rows, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    rows = trajectory["lanes"] if trajectory else []
    firsts = [r["first_k_state_differs"] for r in rows if r["first_k_state_differs"] is not None]
    never = len(rows) - len(firsts)
    if firsts:
        ax1.hist(firsts, bins=np.arange(-0.5, max(firsts) + 1.5, 1.0), color="tab:red")
    ax1.set_xlabel("first PDIP iteration k where batched and unbatched (x, s, z) differ")
    ax1.set_ylabel("lanes")
    ax1.set_title(f"first difference ({never} of {len(rows)} lanes never differ)")

    cmap = plt.get_cmap("viridis")
    differing = [r for r in rows if r["first_k_state_differs"] is not None]
    for j, r in enumerate(differing):
        ks = range(len(r["rel_dx_by_k"]))
        col = cmap(j / max(1, len(differing) - 1))
        ax2.plot(ks, [_driver.log_floor(v) for v in r["rel_dx_by_k"]], color=col, lw=0.8)
        ax3.plot(ks, [_driver.log_floor(v) for v in r["max_ulp_x_by_k"]], color=col, lw=0.8)
    for ax, lab in (
        (ax2, "||x_batched - x_unbatched|| / ||x_unbatched||"),
        (ax3, "max ulp distance, x"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("max_iter = k")
        ax.set_ylabel(lab)
        ax.set_title(f"growth over k, the {len(differing)} differing lanes (floor = 0)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pool(systems):
    by_group: dict[str, list] = {}
    for s in systems:
        by_group.setdefault(s.group, []).append(s)
    pooled = [s for g, ss in by_group.items() if len(ss) > 1 for s in ss]
    singles = [ss[0] for ss in by_group.values() if len(ss) == 1]
    return pooled, singles


def main() -> int:
    args = _parse_args()
    unknown = [c for c in args.candidates if c not in _solvers.CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidate(s) {unknown}; known: {list(_solvers.CANDIDATES)}")
    if any(_solvers.CANDIDATES[c].backend != "jax" for c in args.candidates):
        raise SystemExit("batched_divergence probes JAX candidates only")
    systems = _corpus.load_corpus(args.groups)
    label = _driver.corpus_label(args.groups, _corpus.group_names())
    pooled, singles = _pool(systems)
    if len(pooled) < args.batch_size:
        raise SystemExit(f"--batch-size {args.batch_size} > {len(pooled)} pooled systems")
    lanes = pooled[: args.batch_size]
    tile_systems = singles + lanes[:1]
    print(
        f"  device={args.device} deterministic={args.deterministic} "
        f"XLA_FLAGS={_os.environ.get('XLA_FLAGS', '')!r} B={len(lanes)} "
        f"tile={[f'{s.group}/{s.name}' for s in tile_systems]}",
        flush=True,
    )

    probes: dict = {}
    trajectory = None
    for name in args.candidates:
        cand = _solvers.CANDIDATES[name]
        det, batched_final, unbatched_final = probe_determinism(cand, lanes, args.target_kappa)
        low = probe_lowering(cand, lanes, unbatched_final, args.target_kappa)
        tiled = probe_tiled(cand, tile_systems, args.tile_sizes, args.target_kappa)
        probes[name] = {"determinism": det, "lowering": low, "tiled": tiled}
        ds = det["summary"]
        print(
            f"  {name:<12s} det run1/run2 x-diff {ds['batched_run1_vs_run2']['n_lanes_x_differ']} "
            f"repeat {ds['batched_run1_vs_repeat_call']['n_lanes_x_differ']} "
            f"unbatched {ds['unbatched_run1_vs_run2']['n_lanes_x_differ']} | batched-vs-unbatched "
            f"x {ds['batched_vs_unbatched']['n_lanes_x_differ']} "
            f"it {ds['batched_vs_unbatched']['n_lanes_iterations_differ']} "
            f"flag {ds['batched_vs_unbatched']['n_lanes_converged_differ']} | "
            f"B=1 vmap-vs-jit x {low['summary']['n_lanes_x_differ']} | tiled "
            + ", ".join(
                f"{t['system'].split('/')[-1]}x{t['tile_size']}:"
                f"{'eq' if t['lanes_bitwise_identical'] else 'NE'}/"
                f"{'eq' if t['lane0_vs_unbatched']['x_bitwise_equal'] else 'NE'}"
                for t in tiled
            ),
            flush=True,
        )
        if name == TRAJECTORY_CANDIDATE:
            trajectory = probe_trajectory(lanes, args.max_iter, batched_final, unbatched_final)
            ts = trajectory["summary"]
            print(
                f"  trajectory: {ts['n_lanes_state_ever_differs']}/{ts['n_lanes']} lanes differ; "
                f"first-k histogram {ts['first_k_state_differs_histogram']}; "
                f"consistency {ts['consistency']['batched_x_bitwise_equal_lanes']}/"
                f"{ts['consistency']['unbatched_x_bitwise_equal_lanes']} of {ts['n_lanes']}",
                flush=True,
            )
            joined = join(lanes, det, trajectory)
    primitives = probe_primitives(lanes)
    print(f"  primitives (lanes differing jit vs jit(vmap)): {primitives['summary']}", flush=True)
    if trajectory is None:
        det0 = probes[args.candidates[0]]["determinism"]
        joined = join(lanes, det0, None)

    summary = _driver.summary_header("batched_divergence", args, systems, label)
    cfg = summary["configuration"]
    cfg.pop("repeats", None)
    cfg.update(
        {
            "batch_size": len(lanes),
            "lanes": [f"{s.group}/{s.name}" for s in lanes],
            "max_iter": args.max_iter,
            "tile_sizes": list(args.tile_sizes),
            "tile_systems": [f"{s.group}/{s.name}" for s in tile_systems],
            "deterministic": args.deterministic,
            "xla_flags_env": _os.environ.get("XLA_FLAGS", ""),
            "trajectory_candidate": TRAJECTORY_CANDIDATE,
            "trajectory_entry": "autoarray.util.jax_nnls.solve_nnls(Q_pc, q_pc, solver_tol=None, "
            "max_iter=k) on _solvers._jacobi(Q, q), k = 0..max_iter",
            "batched_call": "a fresh jax.jit(jax.vmap(body)) per call site",
            "unbatched_call": "a fresh jax.jit(body) per call site",
            "timing": "none recorded",
        }
    )
    summary["candidates"] = {
        name: {
            "description": _solvers.CANDIDATES[name].description,
            "library_function": _solvers.CANDIDATES[name].library_function,
        }
        for name in args.candidates
    }
    summary["probes"] = probes
    summary["trajectory"] = trajectory
    summary["primitives"] = primitives
    summary["join"] = joined

    json_path, png_path = _driver.output_paths(
        "batched_divergence", label, summary["autolens_version"], args.device, args.output_dir
    )
    if args.deterministic:
        stem = json_path.stem.replace(
            f"_v{summary['autolens_version']}", f"_det_v{summary['autolens_version']}"
        )
        json_path, png_path = (
            json_path.with_name(stem + ".json"),
            png_path.with_name(stem + ".png"),
        )
    _driver.write_json(json_path, summary)
    _plot(
        png_path,
        trajectory,
        joined["lanes"],
        f"pdip_jacobi batched vs unbatched — {label} B={len(lanes)} — {args.device}"
        f"{' (xla deterministic ops)' if args.deterministic else ''} — "
        f"v{summary['autolens_version']}",
    )
    print(f"  wrote {png_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
