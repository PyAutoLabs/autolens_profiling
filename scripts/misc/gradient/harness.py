"""Measure what a gradient evaluation of one likelihood costs and whether it is correct.

Likelihood lifecycle phase 6 (autolens_profiling#505; adopts PyAutoPulse
``gradient_cost_probe``). A cell's ``gradient_breakdown.py`` builds its fused, jit-able
likelihood ``f(params) -> log L`` exactly as the cell's runtime script does, then calls
:func:`measure` and :func:`write_witness`. The harness times the forward call and
``jax.value_and_grad`` (first call, steady single-call median, vmap per call), runs the
pre-registered finite-difference check of ``tooling/gradient_witness.py`` and writes the cell's
``gradient.json``. Gradients are taken with respect to the flattened parameter vector of
``params`` (``jax.flatten_util.ravel_pytree``), at the point the caller passes (the runtime
scripts' prior medians).
"""

from __future__ import annotations

import json
import socket
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_MISC = Path(__file__).resolve().parents[1]

#: The ``point`` labels a ``gradient.json`` records for where the gradient was taken.
POINT_PRIOR_MEDIANS = "prior_medians"
POINT_IID_DRAW_0 = "iid draw 0 of the seeded sequence"
if str(_MISC / "tooling") not in sys.path:
    sys.path.insert(0, str(_MISC / "tooling"))

import gradient_witness as gw  # noqa: E402


def _block(x):
    import jax

    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, x
    )


def _timed(fn, *args, n_repeats: int) -> tuple[float, float, object]:
    """(first call s, steady median s, last result) of ``fn(*args)``."""
    start = time.perf_counter()
    out = _block(fn(*args))
    first = time.perf_counter() - start
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        out = _block(fn(*args))
        times.append(time.perf_counter() - start)
    return first, statistics.median(times), out


def unit_directions(theta, seed: int):
    """The rule's seeded unit directions: a standard-normal draw scaled by ``max(|theta_i|, 1)``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for _ in range(gw.FD_DIRECTIONS):
        v = rng.standard_normal(theta.shape) * np.maximum(np.abs(theta), 1.0)
        yield v / np.linalg.norm(v)


def fd_directions(f_flat, theta, grad, *, seed: int = gw.FD_SEED) -> list[dict]:
    """The pre-registered FD rule along ``gw.FD_DIRECTIONS`` seeded random unit directions.

    Each direction scales a standard-normal draw by ``max(|theta_i|, 1)`` (so parameters of
    different sizes move comparably) and is normalised to unit length; the directional
    derivative is the Richardson extrapolation of central differences at ``gw.FD_STEP`` and
    half of it, compared with ``grad . v``.
    """
    import numpy as np

    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    out = []
    for v in unit_directions(theta, seed):
        diffs = []
        for h in (gw.FD_STEP, gw.FD_STEP / 2):
            plus = float(f_flat(theta + h * v))
            minus = float(f_flat(theta - h * v))
            diffs.append((plus - minus) / (2 * h))
        fd = gw.richardson(*diffs)
        analytic = float(grad @ v)
        out.append(
            {
                "step": gw.FD_STEP,
                "fd_h": diffs[0],
                "fd_half": diffs[1],
                "fd": fd,
                "analytic": analytic,
                "rel_error": gw.fd_relative_error(fd, analytic),
            }
        )
    return out


def fd_sweep_directions(f_flat, theta, grad, *, seed: int = gw.FD_SEED) -> list[dict]:
    """The FD step sweep (witness version 2) along the rule's seeded directions: the central
    difference at every step in ``gw.SWEEP_STEPS``, beside ``grad . v``, and the direction's
    ``gw.sweep_class``."""
    import numpy as np

    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    out = []
    for v in unit_directions(theta, seed):
        steps = []
        for h in gw.SWEEP_STEPS:
            plus = float(f_flat(theta + h * v))
            minus = float(f_flat(theta - h * v))
            steps.append({"step": h, "central": (plus - minus) / (2 * h)})
        analytic = float(grad @ v)
        out.append(
            {
                "analytic": analytic,
                "steps": steps,
                "class": gw.sweep_class(analytic, [r["central"] for r in steps]),
            }
        )
    return out


def measure(
    f, params, *, n_repeats: int = 10, vmap_batch: int | None = 8, point: str = "prior_medians"
) -> dict:
    """Timings, the gradient's health, the FD check, grad/forward and the hazard for ``f``."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree

    theta, unravel = ravel_pytree(params)

    def f_flat(vector):
        return f(unravel(vector))

    forward = jax.jit(f_flat)
    value_and_grad = jax.jit(jax.value_and_grad(f_flat))
    fwd_first, fwd_single, log_l = _timed(forward, theta, n_repeats=n_repeats)
    vg_first, vg_single, (vg_value, grad) = _timed(value_and_grad, theta, n_repeats=n_repeats)
    timings = {
        "forward.single_call_s": fwd_single,
        "forward.first_call_s": fwd_first,
        "forward.vmap_per_call_s": None,
        "value_and_grad.single_call_s": vg_single,
        "value_and_grad.first_call_s": vg_first,
        "value_and_grad.vmap_per_call_s": None,
    }
    vmap_note = None
    if vmap_batch:
        batch = jnp.tile(theta, (vmap_batch, 1))
        try:
            _, t, _ = _timed(jax.jit(jax.vmap(f_flat)), batch, n_repeats=n_repeats)
            timings["forward.vmap_per_call_s"] = t / vmap_batch
            _, t, _ = _timed(
                jax.jit(jax.vmap(jax.value_and_grad(f_flat))), batch, n_repeats=n_repeats
            )
            timings["value_and_grad.vmap_per_call_s"] = t / vmap_batch
        except Exception as exc:  # an OOM batch leaves the single-call numbers standing
            vmap_note = f"vmap batch {vmap_batch} failed: {type(exc).__name__}: {exc}"[:300]
    grad = np.asarray(grad, dtype=np.float64)
    finite = bool(np.isfinite(grad).all())
    l2 = float(np.linalg.norm(grad)) if finite else None
    if theta.dtype != jnp.float64:
        fd = {"status": "not_assessed", "reason": f"parameters are {theta.dtype}, not fp64"}
        sweep = dict(fd)
    elif not finite:
        fd = {"status": "fail", "reason": "non-finite gradient", "directions": []}
        sweep = dict(fd)
    else:
        directions = fd_directions(lambda x: forward(jnp.asarray(x)), theta, grad)
        fd = {"status": gw.fd_status(directions), "directions": directions}
        sweep_dirs = fd_sweep_directions(lambda x: forward(jnp.asarray(x)), theta, grad)
        classes = [d["class"] for d in sweep_dirs]
        sweep = {"status": gw.sweep_status(classes), "directions": sweep_dirs}
    if sweep["status"] != "not_assessed":
        sweep["rule"] = {
            "directions": gw.FD_DIRECTIONS,
            "steps": list(gw.SWEEP_STEPS),
            "run": gw.SWEEP_RUN,
            "tolerance": gw.FD_TOLERANCE,
            "tiny": gw.FD_TINY,
            "seed": gw.FD_SEED,
        }
    if fd["status"] != "not_assessed":
        fd["rule"] = {
            "directions": gw.FD_DIRECTIONS,
            "steps": [gw.FD_STEP, gw.FD_STEP / 2],
            "richardson": True,
            "tolerance": gw.FD_TOLERANCE,
            "tiny": gw.FD_TINY,
            "seed": gw.FD_SEED,
        }
        fd["max_rel_error"] = max((d["rel_error"] for d in fd["directions"]), default=None)
    ratio = vg_single / fwd_single
    return {
        "point": point,
        "n_parameters": int(theta.size),
        "precision": str(theta.dtype),
        "log_likelihood": float(log_l),
        "value_and_grad_log_likelihood": float(vg_value),
        "repeats": n_repeats,
        "statistic": "median",
        "vmap_batch": vmap_batch,
        "vmap_note": vmap_note,
        "timings": timings,
        "grad_over_forward": ratio,
        "grad_over_forward_vmap": (
            timings["value_and_grad.vmap_per_call_s"] / timings["forward.vmap_per_call_s"]
            if timings["forward.vmap_per_call_s"] and timings["value_and_grad.vmap_per_call_s"]
            else None
        ),
        "gradient": {"finite": finite, "l2_norm": l2, "nonzero": bool(l2) if finite else False},
        "fd": fd,
        "fd_sweep": sweep,
        "hazard": gw.hazard(
            ratio,
            sweep["status"],
            [d["class"] for d in sweep.get("directions", [])]
            if sweep["status"] != "not_assessed"
            else [],
        ),
    }


def host_class(device: dict) -> str:
    """The cell host class of this run (the day-zero migration map's host rule)."""
    sys.path.insert(0, str(_MISC / "tooling"))
    from build_migration_map import json_host

    return json_host({"device": device})[0] or "unknown_host"


def cell_dir(
    root: Path,
    *,
    dataset_type: str,
    variant: str,
    instrument: str,
    setup_id: str,
    host: str,
    model_variant: str | None = None,
) -> Path:
    parts = ["results", dataset_type, variant, instrument, setup_id, host]
    if model_variant and model_variant != "default":
        parts.append(model_variant)
    return Path(root).joinpath(*parts)


def witness(
    measured: dict,
    *,
    dataset_type: str,
    variant: str,
    instrument: str,
    setup_id: str,
    host: str,
    autolens_version: str,
    device: dict,
    resources: dict | None = None,
    configuration: dict | None = None,
) -> dict:
    """The ``gradient.json`` document (``tooling/gradient_witness.py``).

    ``configuration`` carries the cell facts the catalogue's size rule reads (for a pixelized
    cell its ``source_pixels`` / ``source_pixels_requested``), as the runtime rows do.
    """
    doc = {
        "schema": gw.SCHEMA,
        "version": gw.VERSION,
        "likelihood": {"dataset_type": dataset_type, "variant": variant},
        "instrument": instrument,
        "setup_id": setup_id,
        "host_class": host,
        "autolens_version": autolens_version,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "device": device,
        "resources": resources,
        "configuration": configuration or {},
        **measured,
    }
    errors = gw.validate(doc)
    if errors:
        raise ValueError("invalid gradient witness: " + "; ".join(errors))
    return doc


def write_witness(cell: Path, doc: dict) -> Path:
    """Write ``gradient.json`` into the cell folder (never under a smoke run)."""
    import os

    if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
        return None
    cell.mkdir(parents=True, exist_ok=True)
    path = cell / "gradient.json"
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return path


def summary(doc: dict) -> str:
    t = doc["timings"]
    fd = doc["fd"]
    lines = [
        f"  forward          {t['forward.single_call_s'] * 1e3:10.4f} ms (single)",
        f"  value_and_grad   {t['value_and_grad.single_call_s'] * 1e3:10.4f} ms (single)",
        f"  grad / forward   {doc['grad_over_forward']:10.2f}x",
        f"  FD check         {fd['status']}"
        + (f" (max rel error {fd['max_rel_error']:.2e})" if fd.get("max_rel_error") else ""),
    ]
    sweep = doc.get("fd_sweep")
    if sweep:
        classes = [d["class"] for d in sweep.get("directions", [])]
        lines.append(
            f"  FD step sweep    {sweep['status']}"
            + (f" ({', '.join(classes)})" if classes else "")
        )
    lines += [
        f"  hazard          {'; '.join(doc['hazard']['reasons']) or 'none'}",
    ]
    return "\n".join(lines)


# ---- the --gradient hook (phase 6b, autolens_profiling#507) ---------------------------------


def gradient_requested(argv=None) -> bool:
    """Whether the run was asked for the gradient witness instead of its timing row."""
    return "--gradient" in (sys.argv if argv is None else argv)


def run_gradient(
    f,
    params,
    *,
    dataset_type: str,
    variant: str,
    instrument: str,
    setup_id: str,
    host: str | None = None,
    path: Path | None = None,
    configuration: dict | None = None,
    n_repeats: int = 10,
    vmap_batch: int | None = 8,
    point: str,
) -> Path | None:
    """Measure ``f(params)`` and write the cell's ``gradient.json``; the construction site's
    ``--gradient`` branch calls this with its own fused likelihood and returns.

    ``point``: where ``params`` sits — ``POINT_PRIOR_MEDIANS``, ``POINT_IID_DRAW_0`` or the
    caller's own description. Required: a defaulted label once wrote ``prior_medians`` for
    cells evaluated at iid draw 0 (autolens_profiling#517 item 10).

    ``path``: the witness path (a campaign submit's ``--output-path``); else the mirrored cell
    of ``host`` (default: this run's host by the day-zero host rule). fp64 only.
    """
    import autolens as al
    import jax

    root = _MISC.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from _profile_cli import device_info_dict, resources_dict

    if not jax.config.x64_enabled:
        raise SystemExit("--gradient measures fp64 only: export JAX_ENABLE_X64=True")
    if "_mixed" in setup_id:
        raise SystemExit("--gradient measures fp64 only (phase 6); drop --use-mixed-precision")
    print(f"\n--- Gradient cost and correctness [{dataset_type}/{variant}, {instrument}] ---")
    measured = measure(f, params, n_repeats=n_repeats, vmap_batch=vmap_batch, point=point)
    device = device_info_dict()
    if path is not None:
        path = Path(path)
        if path.name != "gradient.json":
            path = path.with_name("gradient.json")
        host = path.parent.name if host is None else host
    host = host or host_class(device)
    doc = witness(
        measured,
        dataset_type=dataset_type,
        variant=variant,
        instrument=instrument,
        setup_id=setup_id,
        host=host,
        autolens_version=al.__version__,
        device=device,
        resources=resources_dict("jax"),
        configuration=configuration,
    )
    print(summary(doc))
    cell = (
        path.parent
        if path is not None
        else cell_dir(
            root,
            dataset_type=dataset_type,
            variant=variant,
            instrument=instrument,
            setup_id=setup_id,
            host=host,
        )
    )
    written = write_witness(cell, doc)
    print(f"  gradient.json: {written}")
    return written
