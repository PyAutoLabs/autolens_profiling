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
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(gw.FD_DIRECTIONS):
        v = rng.standard_normal(theta.shape) * np.maximum(np.abs(theta), 1.0)
        v /= np.linalg.norm(v)
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


def measure(f, params, *, n_repeats: int = 10, vmap_batch: int | None = 8) -> dict:
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
    elif not finite:
        fd = {"status": "fail", "reason": "non-finite gradient", "directions": []}
    else:
        directions = fd_directions(lambda x: forward(jnp.asarray(x)), theta, grad)
        fd = {"status": gw.fd_status(directions), "directions": directions}
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
        "point": "prior_medians",
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
        "hazard": gw.hazard(ratio, fd["status"]),
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
        f"  hazard           {'; '.join(doc['hazard']['reasons']) or 'none'}",
    ]
    return "\n".join(lines)
