# searches

Sampler / search profiling for the PyAutoLens HST MGE lens-modelling likelihood. Each subfolder drives a single sampler family directly against the real likelihood — bypassing `af.NonLinearSearch` — so the per-sampler convergence characteristics (wall time, likelihood evaluations, posterior ESS, evals/time to ML) can be compared on identical footing.

## Why bypass `af.NonLinearSearch`?

`af.NonLinearSearch` adds caching, multi-process forking, output formatting, and result hierarchies that are valuable for production fits but obscure the underlying sampler's cost. The scripts in this section call the sampler library directly and instrument every likelihood evaluation through a shared `MLTracker`. The result is a clean apples-to-apples comparison of:

- Wall time and likelihood-evaluation count to **Nautilus's default convergence** (`n_eff=10000`, `f_live=0.01`).
- Per-evaluation likelihood cost (NumPy baseline vs JAX-JIT'd path).
- Evals-to-ML and time-to-ML — the eval index and wall time at which the running max log L first came within 1 nat of the final maximum.

## Shared helpers

| File | Role |
|------|------|
| [`_setup.py`](./_setup.py) | Builds the HST imaging dataset, the MGE + Isothermal + ExternalShear lens model with an MGE source bulge, and the `AnalysisImaging` object. The dataset, mask, and model mirror the reference setup in [`likelihood/imaging/mge.py`](../likelihood/imaging/mge.py) so likelihood values are directly comparable across the two sections. |
| [`_metrics.py`](./_metrics.py) | `MLTracker` — records the log-likelihood and wall time of every evaluation, computes evals-to-ML and time-to-ML headline numbers. Also offers `MLTracker.from_log_l_history` for samplers that JIT their likelihood and only expose log-L per dead/live point post hoc. |

## Supported samplers

| Sampler | Folder | Status | Notes |
|---------|--------|--------|-------|
| Nautilus | [`nautilus/`](./nautilus/README.md) | ✓ profiled | Both NumPy (`simple.py`) and JAX-JIT (`jax.py`) variants. |
| Dynesty | _planned_ | not yet mirrored | Static nested sampling; reference scripts at `autolens_workspace_developer/searches_minimal/dynesty_simple.py`. |
| Emcee | _planned_ | not yet mirrored | Affine-invariant ensemble MCMC. |
| BlackJAX (NUTS, SMC) | _planned_ | not yet mirrored | Pure-JAX HMC family. Gradient pathology surfaced in upstream `sweep_findings.md`; HMC viability depends on first fixing NaN-gradient hot spots. |
| NumPyro (ESS) | _planned_ | not yet mirrored | Ensemble slice sampler under JAX. |
| PocoMC | _planned_ | not yet mirrored | Preconditioned Monte Carlo. |
| NSS (simple, jit, grad) | _planned_ | not yet mirrored | Nested slice sampler; `nss_jit.py` shows VRAM ceiling on consumer GPUs (see `sweep_findings.md`). |
| LBFGS | _planned_ | not yet mirrored | Not a sampler; serves as the maximum-likelihood reference point. |

Each row above corresponds to one or more scripts under `autolens_workspace_developer/searches_minimal/`; the mirror migration here under their own follow-up prompts.

## Versioned artifacts

Each script writes a JSON + PNG pair to:

```
results/searches/<sampler>/<script>_summary_v<al.__version__>.{json,png}
```

The JSON carries the structured timings + sampler config + best-fit summary. The PNG is a bar chart of the headline timings (wall time, time per eval, time to ML; plus JIT compile time on JAX scripts).

Old versions are retained alongside new ones; Phase 4's dashboard surfaces the latest per axis.

## Running a script

From the repo root (cwd matters because `_setup.build_dataset()` resolves `dataset/imaging/hst/` relative to the repo root via `Path(__file__).resolve().parent.parent`):

```bash
cd autolens_profiling
python searches/nautilus/simple.py
python searches/nautilus/jax.py
```

Or as modules:

```bash
python -m searches.nautilus.simple
python -m searches.nautilus.jax
```

Both invocation styles work — each script injects the repo root into `sys.path` before importing `searches._{setup,metrics}` for robustness.

**Requirements:** `nautilus-sampler` for the Nautilus scripts (`pip install nautilus-sampler`). The JAX variant additionally needs a working JAX install.

**Codex / sandboxed runs:**

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python searches/nautilus/simple.py
```
