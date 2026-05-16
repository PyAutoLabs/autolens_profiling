"""
Minimal Nautilus Example — pure-JAX HST MGE likelihood
------------------------------------------------------

Drives the Nautilus nested sampler against the HST MGE imaging likelihood
running fully under ``jax.jit``. The analysis is built with ``use_jax=True``
and the closure is passed through ``jax.jit`` once, ahead of sampling, so
the JIT compile cost is reported separately from the sampling wall time.

Nautilus itself is a NumPy sampler, so the wrapper does
``np.asarray(jit_loglike(jnp.asarray(params)))`` per call -- the JAX kernel
runs but every evaluation crosses the Python <-> JAX boundary.

``n_live`` is kept at the smoke-test values used by ``simple.py`` — this
is a wiring test, not a converged posterior.

Compare versus ``simple.py`` (NumPy likelihood under the same sampler) and
``likelihood/imaging/mge.py`` (single-likelihood JIT profiling of the same
MGE setup).

Requirements:
    pip install nautilus-sampler
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib


# AUTOLENS_PROFILING_SMOKE=1 short-circuit (Phase 5 / CI lint smoke).
# Verifies the import graph + module-level setup succeeded without running
# the full profiling pipeline. Skipped entirely when the env var is unset.
import os as _smoke_os
import sys as _smoke_sys
if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _smoke_sys.exit(0)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autolens as al

# Make ``from searches._{setup,metrics}`` importable regardless of how the
# script is invoked (``python searches/nautilus/jax.py``, ``python -m
# searches.nautilus.jax``, or a CI runner).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from searches._metrics import MLTracker
from searches._setup import (
    build_analysis,
    build_dataset,
    build_model,
    format_best_fit,
)

dataset = build_dataset()
model = build_model()
analysis = build_analysis(dataset, use_jax=True)

print(f"Model free parameters: {model.total_free_parameters}")

from nautilus import Sampler


def log_likelihood_jax(params):
    """Pure-JAX log likelihood: flat parameter vector -> scalar log L."""
    instance = model.instance_from_vector(vector=params, xp=jnp)
    return analysis.log_likelihood_function(instance=instance)


jit_log_likelihood = jax.jit(log_likelihood_jax)

# Warm up the JIT once so the compile cost is measured separately.
warmup_unit = [0.5] * model.prior_count
warmup_physical = jnp.asarray(model.vector_from_unit_vector(warmup_unit))
print("JIT-compiling MGE likelihood (one-shot)...", flush=True)
t_jit_start = time.time()
_ = float(jax.block_until_ready(jit_log_likelihood(warmup_physical)))
t_jit = time.time() - t_jit_start
print(f"  Compiled in {t_jit:.2f} s", flush=True)


def prior_transform(cube):
    """Map a unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    """Adapter: NumPy in, JIT'd JAX likelihood, Python float out."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    log_l = float(jit_log_likelihood(jnp.asarray(params)))
    tracker.record(log_l)
    return log_l


n_live = 200

sampler = Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
    n_dim=model.prior_count,
    n_live=n_live,
)

t_start = time.time()
# Run to Nautilus's default convergence (n_eff=10000, f_live=0.01) on the
# JAX-jitted MGE likelihood. JIT compile is paid once above; per-call cost
# inside sampling is the JAX kernel + Python<->JAX boundary.
sampler.run(verbose=True)
t_elapsed = time.time() - t_start

points, log_w, log_l = sampler.posterior()
best_idx = np.argmax(log_l)
best_instance = model.instance_from_vector(vector=list(points[best_idx]))
max_logl = float(np.max(log_l))

evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

# ---------------------------------------------------------------------------
# Print human-readable summary
# ---------------------------------------------------------------------------

summary = f"""\
--- Nautilus (JAX JIT) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {float(sampler.log_z):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compile, run ahead of time)
JIT compile time:    {t_jit:.2f} s     (one-shot warm-up before sampling)
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {float(sampler.n_eff):.1f}
Posterior samples:   {len(points)}
Sampler config:      n_live={n_live}, default n_eff=10000, f_live=0.01

--- Convergence ---
Converged:           yes (Nautilus default n_eff / f_live)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (first eval within 1 nat of max log L)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

# ---------------------------------------------------------------------------
# Write versioned JSON + PNG to results/searches/nautilus/
# ---------------------------------------------------------------------------

al_version = al.__version__
result_dict = {
    "sampler": "nautilus",
    "backend": "jax_jit",
    "instrument": "hst",
    "model": {
        "type": "MGE+Isothermal+ExternalShear",
        "free_parameters": int(model.total_free_parameters),
    },
    "sampler_config": {
        "n_live": n_live,
        "n_eff_target": 10000,
        "f_live": 0.01,
    },
    "results": {
        "max_log_likelihood": max_logl,
        "log_evidence": float(sampler.log_z),
        "best_fit_summary": format_best_fit(best_instance),
    },
    "performance": {
        "wall_time_s": t_elapsed,
        "jit_compile_s": t_jit,
        "likelihood_evals": int(n_likelihood_calls),
        "time_per_eval_ms": t_elapsed / max(n_likelihood_calls, 1) * 1e3,
        "ess": float(sampler.n_eff),
        "posterior_samples": int(len(points)),
    },
    "convergence": {
        "converged": True,
        "evals_to_ml": int(evals_to_ml) if evals_to_ml is not None else None,
        "time_to_ml_s": float(time_to_ml) if time_to_ml is not None else None,
    },
    "version": al_version,
}

results_dir = _REPO_ROOT / "results" / "searches" / "nautilus"
results_dir.mkdir(parents=True, exist_ok=True)
json_path = results_dir / f"jax_summary_v{al_version}.json"
json_path.write_text(json.dumps(result_dict, indent=2))
print(f"  Results JSON saved to: {json_path}")

# Bar chart of the headline timings
fig, ax = plt.subplots(figsize=(8, 3))
labels = [
    "jit_compile (s)",
    "wall_time (s)",
    "time_per_eval (ms)",
    "time_to_ml (s)" if time_to_ml is not None else "time_to_ml (n/a)",
]
times = [
    t_jit,
    t_elapsed,
    t_elapsed / max(n_likelihood_calls, 1) * 1e3,
    float(time_to_ml) if time_to_ml is not None else 0.0,
]
ax.barh(labels, times, color=["#8172B2", "#4C72B0", "#55A868", "#C44E52"])
ax.set_title(f"Nautilus (JAX JIT) — HST MGE — v{al_version}")
fig.tight_layout()
png_path = results_dir / f"jax_summary_v{al_version}.png"
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f"  Bar chart saved to:    {png_path}")
