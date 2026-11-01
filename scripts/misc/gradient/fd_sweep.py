"""
Finite-difference step-size sweep: a diagnostic of a gradient cell's FD check
=============================================================================

Likelihood lifecycle phase 6c (autolens_profiling#509). The pre-registered FD rule
(``harness.fd_directions``: central differences at ``gw.FD_STEP`` and half of it, Richardson
extrapolated, rel error <= 1e-4) failed on most cells of the first gradient campaign, and on the
pixelized cells the two steps disagree with each other, so it cannot tell a wrong gradient from
a likelihood that is not smooth at that step. This diagnostic separated the two and led to the
step sweep in the witness rule (``harness.fd_sweep_directions``, witness version 2); it adds the
likelihood slices the witness does not keep. It never writes a ``gradient.json``.

Run a cell's ``gradient_breakdown.py`` through this wrapper with the cell's own flags::

    FD_SWEEP_OUT=<dir>/fd_sweep.json python scripts/misc/gradient/fd_sweep.py \
        scripts/imaging/delaunay/gradient_breakdown.py --instrument hst --config-name ...

The wrapper replaces ``harness.measure`` in-process, so the cell builds exactly the likelihood,
point and parameters it hands the harness in the campaign; the replacement then records, along
the same seeded unit directions as the pre-registered rule:

* ``directions``: the witness rule's step sweep (``harness.fd_sweep_directions``): the central
  difference at each step in ``gw.SWEEP_STEPS`` (1e-1 down to 1e-8) beside ``grad . v``, and the
  direction's class (agree / mismatch / zero / no_plateau, ``gw.sweep_class``).
* ``slice``: ``f(theta + t v)`` and the analytic ``grad(theta + t v) . v`` along the first
  direction over a wide and a narrow window of ``t``, to see where the likelihood jumps or bends
  and whether the analytic slope tracks the slice between the jumps.

No ``gradient.json`` is written: the replacement writes ``FD_SWEEP_OUT`` and exits.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

_MISC = Path(__file__).resolve().parents[1]
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from gradient import harness  # noqa: E402

SLICE_WINDOWS = {"wide": (2e-2, 41), "narrow": (2e-4, 41)}
SLICE_GRAD_EVERY = 4


def sweep(f, params, **_ignored) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree

    gw = harness.gw
    theta, unravel = ravel_pytree(params)

    def f_flat(vector):
        return f(unravel(vector))

    forward = jax.jit(f_flat)
    value_and_grad = jax.jit(jax.value_and_grad(f_flat))
    start = time.perf_counter()
    log_l, grad = value_and_grad(theta)
    grad = np.asarray(grad, dtype=np.float64)
    theta_np = np.asarray(theta, dtype=np.float64)

    def f_at(x):
        return float(forward(jnp.asarray(x)))

    out_dirs = harness.fd_sweep_directions(f_at, theta_np, grad)
    v = next(harness.unit_directions(theta_np, gw.FD_SEED))
    slices = {}
    for name, (half_width, n) in SLICE_WINDOWS.items():
        ts = np.linspace(-half_width, half_width, n)
        points = []
        for i, t in enumerate(ts):
            x = theta_np + t * v
            row = {"t": float(t), "f": f_at(x)}
            if i % SLICE_GRAD_EVERY == 0 or i == n // 2:
                _, g = value_and_grad(jnp.asarray(x))
                row["analytic"] = float(np.asarray(g, dtype=np.float64) @ v)
            points.append(row)
        slices[name] = points

    return {
        "schema": "fd-sweep",
        "version": 2,
        "n_parameters": int(theta_np.size),
        "log_likelihood": float(log_l),
        "gradient_finite": bool(np.isfinite(grad).all()),
        "gradient_l2_norm": float(np.linalg.norm(grad)),
        "seed": gw.FD_SEED,
        "pre_registered_step": gw.FD_STEP,
        "directions": out_dirs,
        "slice_direction": 0,
        "slices": slices,
        "elapsed_s": time.perf_counter() - start,
    }


def _replacement(f, params, **kwargs):
    out = Path(os.environ["FD_SWEEP_OUT"])
    doc = sweep(f, params, **kwargs)
    doc["argv"] = sys.argv
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"  fd_sweep.json: {out}  ({doc['elapsed_s']:.1f} s)")
    raise SystemExit(0)


if __name__ == "__main__":
    if "FD_SWEEP_OUT" not in os.environ:
        raise SystemExit("set FD_SWEEP_OUT to the sweep's output path")
    harness.measure = _replacement
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    if "--gradient" not in sys.argv:
        sys.argv.append("--gradient")
    runpy.run_path(str(script), run_name="__main__")
