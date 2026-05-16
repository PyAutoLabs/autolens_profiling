# likelihood

JAX JIT profiling for the PyAutoLens likelihood function across imaging, interferometer, point-source, and datacube datasets, and across the MGE, pixelization, and Delaunay model compositions used in real science cases.

## What "JIT likelihood profiling" means

For each science case, the likelihood function turns a parameter vector into a single number (log-likelihood) via a chain of array operations: instantiate the model, build the `Tracer`, ray-trace grids through the lens, compute a mapping matrix, blur it with the PSF, solve a linear-algebra reconstruction problem, and finally compute a chi-squared. Under `xp=jnp`, every step is dispatched as a JAX op and can be compiled into a single XLA program with `jax.jit`.

Profiling the **whole likelihood** as one JIT'd function gives the honest per-call cost a sampler will see in production. Profiling **each step individually** under its own JIT gives the breakdown that tells you where the time is going. Both numbers matter: the whole-function timing is the production cost, and the per-step breakdown is the optimisation target. Each script in this section reports both wherever the underlying pipeline supports per-step JIT-ing (the interferometer and datacube paths intentionally stay at full-pipeline JIT for now — see those subfolders' READMEs for why).

Every script also reports a **batched (`jax.vmap`) per-likelihood cost** to make explicit how much the JIT amortises across a population of evaluations — the regime an actual sampler operates in.

## How to read the per-script output

Each script prints a structured narrative to stdout, ending in:

- The eager (numpy) baseline log-likelihood for sanity.
- The single-JIT lower / compile / first-call / steady-state per-call timings.
- The vmap per-likelihood cost and speedup vs single-JIT.
- A correctness check: eager ≡ JIT ≡ vmap log-likelihoods at `rtol=1e-4`.
- A `results/likelihood/<type>/<script>_likelihood_summary_<instrument>_v<al.__version__>.{json,png}` write.

The JSON carries the structured timings keyed by step name plus the model / dataset metadata. The PNG is a bar chart of per-step costs (where applicable) plus the single-JIT vs vmap comparison.

## Versioned artifacts

Result files are tagged with the PyAutoLens release that produced them (`al.__version__`). Old versions remain alongside new ones so cross-release trends stay visible — Phase 4's dashboard will read the latest per axis and present the headline numbers framed by astronomy instrument.

See the top-level [results/README.md](../results/README.md) for the full filename convention.

## Sections

| Folder | Profiles |
|--------|----------|
| [`imaging/`](./imaging/README.md) | MGE, pixelization, and Delaunay likelihoods on imaging datasets (HST-resolution by default). |
| [`interferometer/`](./interferometer/README.md) | MGE, pixelization, and Delaunay likelihoods on interferometer (visibility-space) datasets (SMA by default). |
| [`point_source/`](./point_source/README.md) | Image-plane and source-plane chi-squared for lensed point sources. |
| [`datacube/`](./datacube/README.md) | Multi-channel datacube likelihoods (e.g. ALMA-style) with Delaunay pixelization. |

## Running a script

From the repo root:

```bash
cd autolens_profiling
python likelihood/imaging/mge.py
```

Scripts use the input datasets under `dataset/<type>/<instrument>/` (see top-level [README](../README.md)). The default instrument is encoded per-script; some support a CLI flag to switch instruments. Run with `--help` for the supported options.

**Codex / sandboxed runs** — set writable cache dirs so numba and matplotlib don't choke on read-only home/source paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python likelihood/imaging/mge.py
```

## Conventions inherited from `autolens_workspace_developer`

The scripts follow the JIT conventions documented at `autolens_workspace_developer/CLAUDE.md`:

- All autoarray types (`Array2D`, `Grid2D`, `Grid2DIrregular`, …) expose `.array` for the raw `np.ndarray` / `jax.Array` underneath. These are extracted before crossing the `jax.jit` boundary because autoarray types are not registered as JAX pytrees as **inputs**.
- The `xp` parameter (`xp=np` default, `xp=jnp` for JAX) controls the backend. JIT'd closures pass `xp=jnp` through every nested call.
- The model is converted to a JAX pytree via `autofit.jax.register_model(model)` so `af.ModelInstance` can cross the JIT boundary directly — no manual flat-vector unpacking.
