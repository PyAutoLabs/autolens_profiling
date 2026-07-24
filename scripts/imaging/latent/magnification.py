"""
Latent profiling: magnification
=================================

Profiles the magnification computation in isolation.

The magnification latent is defined as the ratio of image-plane (lensed)
to source-plane (intrinsic) flux: ``total_lensed_source_flux_mujy / total_source_flux_mujy``.
The magzero conversion cancels in the ratio, so the result is dimensionless and
independent of the chosen zero-point. Under JIT the two flux paths both compile
into the same trace; expect this latent to cost roughly the sum of those two
components minus any shared subexpressions XLA can fuse. Because the division
of two scalar JIT outputs is trivially vectorisable, vmap is expected to work
cleanly for this latent.
"""

import argparse
import json
import os
import os as _os

# AUTOLENS_PROFILING_SMOKE=1 short-circuit.
import sys as _sys
import tempfile
import time
from pathlib import Path

import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autolens import fixtures
from autolens.analysis.latent import LATENT_FUNCTIONS
from autonerves import conf

if _os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports OK; exiting.")
    _sys.exit(0)

LATENT_KEY = "magnification"


def _push_single_latent_config(latent_key: str) -> Path:
    """Write a temp config dir with only latent_key enabled and push it."""
    tmpdir = Path(tempfile.mkdtemp(prefix="latent_cfg_"))
    yaml_lines = [f"{k}: {'true' if k == latent_key else 'false'}" for k in LATENT_FUNCTIONS]
    (tmpdir / "latent.yaml").write_text("\n".join(yaml_lines) + "\n")
    conf.instance.push(str(tmpdir))
    return tmpdir


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
