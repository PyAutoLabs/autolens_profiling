"""
Minimal Nautilus Example (HST MGE lens likelihood)
--------------------------------------------------

Drives the Nautilus nested sampler directly against the HST MGE imaging
likelihood, bypassing ``af.NonLinearSearch``. Useful as a fast end-to-end
smoke test of the real PyAutoLens likelihood under a production sampler.

``n_live`` is kept small so the search finishes in a few minutes — this is
a wiring test, not a converged posterior.

Compare versus ``likelihood/imaging/mge.py`` (single-likelihood profiling
of the same MGE setup) and ``jax.py`` (JAX-JIT'd likelihood under the same
sampler).

Requirements:
    pip install nautilus-sampler
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import autolens as al

# Make ``from searches._{setup,metrics}`` importable regardless of how the
# script is invoked (``python searches/nautilus/simple.py``, ``python -m
# searches.nautilus.simple``, or a CI runner).
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
analysis = build_analysis(dataset, use_jax=False)

print(f"Model free parameters: {model.total_free_parameters}")

from nautilus import Sampler


def prior_transform(cube):
    """Map a unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=list(params))
    log_l = float(analysis.log_likelihood_function(instance=instance))
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
# Run to Nautilus's default convergence (n_eff=10000, f_live=0.01). This
# may take many thousands of likelihood evaluations against the NumPy MGE
# -- expect long wall times.
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
--- Nautilus (NumPy) Results ---
Best fit:        {format_best_fit(best_instance)}
Max log L:       {max_logl:.4f}
Log evidence:    {float(sampler.log_z):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s
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
    "backend": "numpy",
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
json_path = results_dir / f"simple_summary_v{al_version}.json"
json_path.write_text(json.dumps(result_dict, indent=2))
print(f"  Results JSON saved to: {json_path}")

# Bar chart of the headline timings
fig, ax = plt.subplots(figsize=(8, 3))
labels = [
    "wall_time (s)",
    "time_per_eval (ms)",
    "time_to_ml (s)" if time_to_ml is not None else "time_to_ml (n/a)",
]
times = [
    t_elapsed,
    t_elapsed / max(n_likelihood_calls, 1) * 1e3,
    float(time_to_ml) if time_to_ml is not None else 0.0,
]
ax.barh(labels, times, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_title(f"Nautilus (NumPy) — HST MGE — v{al_version}")
fig.tight_layout()
png_path = results_dir / f"simple_summary_v{al_version}.png"
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f"  Bar chart saved to:    {png_path}")
