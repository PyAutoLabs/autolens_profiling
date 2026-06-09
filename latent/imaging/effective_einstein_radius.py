"""
Latent profiling: effective_einstein_radius
=============================================

Profiles the effective_einstein_radius computation in isolation.

This is the most expensive of the five default latents. It solves for the
tangential critical curve via ``LensCalc.einstein_radius_jit_from``, which
wraps ``jax_zero_contour.ZeroSolver`` — a marching-squares contour finder
that uses ``jax.lax.cond`` / ``jax.lax.while_loop`` for early termination.
The while-loop makes this latent incompatible with ``jax.vmap``; the
production code path therefore uses ``LATENT_BATCH_MODE='jit'`` (loop over
samples in Python, one JIT call per sample) rather than a batched vmap.

Cache behaviour: ``LensCalc`` maintains a ``_zero_contour_cache`` dict keyed
on ``(kind, pixel_scales, tol, max_newton)``. On the first call the
``(f, ZeroSolver)`` pair is built and cached; subsequent calls reuse it and
therefore hit JAX's XLA compile cache. This script explicitly surfaces the
first-call vs second-call cost by constructing two separate ``LensCalc``
instances (each starting with an empty cache) and timing them independently.
"""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tempfile

import autofit as af
import autolens as al
from autolens import fixtures
from autoconf import conf
from autolens.analysis.latent import LATENT_FUNCTIONS

# AUTOLENS_PROFILING_SMOKE=1 short-circuit.
import sys as _sys, os as _os
if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports OK; exiting.")
    _sys.exit(0)

LATENT_KEY = "effective_einstein_radius"


def _push_single_latent_config(latent_key: str) -> Path:
    """Write a temp config dir with only latent_key enabled and push it."""
    tmpdir = Path(tempfile.mkdtemp(prefix="latent_cfg_"))
    yaml_lines = [
        f"{k}: {'true' if k == latent_key else 'false'}"
        for k in LATENT_FUNCTIONS
    ]
    (tmpdir / "latent.yaml").write_text("\n".join(yaml_lines) + "\n")
    conf.instance.push(str(tmpdir))
    return tmpdir


def _time_closure_cache(tracer, dataset, xp=np):
    """Time first vs second call of the effective_einstein_radius latent function.

    Constructs a fresh LensCalc for each call so the closure cache starts empty,
    isolating the first-build vs cache-hit cost difference.

    Returns (first_call_s, second_call_s) floats, or (nan, nan) on error.
    """
    from autolens.analysis.latent import effective_einstein_radius
    from autolens.imaging.fit_imaging import FitImaging

    # Build a minimal fit object that effective_einstein_radius can use.
    # We need fit.tracer and fit.dataset.grids.lp.
    try:
        # Config already pushed by main(); the pushed config has this latent enabled.

        analysis = al.AnalysisImaging(dataset=dataset, use_jax=(xp is not np), magzero=25.0)

        lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, bulge=al.lp.Sersic)
        source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)
        model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

        params = model.physical_values_from_prior_medians
        instance = model.instance_from_vector(vector=params)
        fit = al.FitImaging(
            dataset=dataset,
            tracer=al.Tracer(galaxies=list(instance.galaxies)),
            xp=xp,
        )

        # First call — fresh LensCalc, cold closure cache.
        t0 = time.perf_counter()
        _ = effective_einstein_radius(fit=fit, magzero=25.0, xp=xp)
        first_call_s = time.perf_counter() - t0

        # Second call — LensCalc on `fit.tracer` still has its cache populated
        # from above. Construct a new LensCalc explicitly to test warm-cache path.
        from autogalaxy.operate.lens_calc import LensCalc
        lc2 = LensCalc.from_mass_obj(fit.tracer)
        init_guess = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        # Warm the ZeroSolver cache on lc2 first (mirrors the cold first call).
        if xp is not np:
            _ = lc2.einstein_radius_jit_from(init_guess=init_guess)
            t0 = time.perf_counter()
            _ = lc2.einstein_radius_jit_from(init_guess=init_guess)
            second_call_s = time.perf_counter() - t0
        else:
            _ = lc2.einstein_radius_from(grid=dataset.grids.lp)
            t0 = time.perf_counter()
            _ = lc2.einstein_radius_from(grid=dataset.grids.lp)
            second_call_s = time.perf_counter() - t0

        return first_call_s, second_call_s
    except Exception:
        return float("nan"), float("nan")


def main(config_name: str, output_dir: Path, use_mixed_precision: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Isolate THIS latent only via conf override
    _push_single_latent_config(LATENT_KEY)

    dataset = fixtures.make_masked_imaging_7x7()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, bulge=al.lp.Sersic)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)
    analysis_jax = al.AnalysisImaging(dataset=dataset, use_jax=True, magzero=25.0)

    params = jnp.array(model.physical_values_from_prior_medians)

    # === Eager numpy baseline ===
    t0 = time.perf_counter()
    eager_values = al.LatentLens.variables(analysis_np, np.asarray(params), model)
    eager_t = time.perf_counter() - t0
    eager_value = float(eager_values[0])

    # === Closure cache first-call vs second-call (numpy path) ===
    closure_cache_first_call_s, closure_cache_second_call_s = _time_closure_cache(
        tracer=None, dataset=dataset, xp=np
    )

    # === JIT compile + first call + steady-state ===
    fn = jax.jit(lambda p: al.LatentLens.variables(analysis_jax, p, model))
    lower_t = compile_t = first_t = steady_t = float("nan")
    jit_value = float("nan")
    jit_error = None
    try:
        t0 = time.perf_counter()
        lowered = fn.lower(params)
        lower_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        compiled = lowered.compile()
        compile_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        first = compiled(params)
        try:
            jax.block_until_ready(first[0])
        except Exception:
            pass
        first_t = time.perf_counter() - t0
        jit_value = float(first[0])

        steady_ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = compiled(params)
            try:
                jax.block_until_ready(r[0])
            except Exception:
                pass
            steady_ts.append(time.perf_counter() - t0)
        steady_t = float(np.mean(steady_ts))
    except Exception as exc:
        jit_error = repr(exc)

    # === vmap batched ===
    # NOTE: effective_einstein_radius uses jax.lax.while_loop / lax.cond under
    # jax_zero_contour, which is incompatible with jax.vmap. We still attempt
    # it so the sweep surfaces the error cleanly rather than silently skipping.
    vmap_t = float("nan")
    vmap_value = float("nan")
    vmap_error = None
    batch_size = 3
    batched = jnp.tile(params[None, :], (batch_size, 1))
    try:
        vfn = jax.jit(jax.vmap(lambda p: al.LatentLens.variables(analysis_jax, p, model)))
        warm = vfn(batched)
        try:
            jax.block_until_ready(warm[0])
        except Exception:
            pass
        t0 = time.perf_counter()
        r = vfn(batched)
        try:
            jax.block_until_ready(r[0])
        except Exception:
            pass
        vmap_t = (time.perf_counter() - t0) / batch_size
        vmap_value = float(r[0][0])
    except Exception as exc:
        vmap_error = repr(exc)

    record = {
        "latent_key": LATENT_KEY,
        "config_name": config_name,
        "use_mixed_precision": use_mixed_precision,
        "eager_value": eager_value,
        "eager_time_s": eager_t,
        "closure_cache_first_call_s": closure_cache_first_call_s,
        "closure_cache_second_call_s": closure_cache_second_call_s,
        "jit_value": jit_value,
        "jit_lower_s": lower_t,
        "jit_compile_s": compile_t,
        "jit_first_call_s": first_t,
        "jit_steady_state_s": steady_t,
        "jit_error": jit_error,
        "vmap_per_call_s": vmap_t,
        "vmap_value": vmap_value,
        "vmap_error": vmap_error,
    }
    out_path = output_dir / f"{config_name}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--use-mixed-precision", action="store_true")
    args = parser.parse_args()
    main(args.config_name, args.output_dir, args.use_mixed_precision)
