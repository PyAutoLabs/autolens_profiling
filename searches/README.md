# `searches/` — first-class search profiling

This section profiles **first-class PyAutoFit search objects** end-to-end:
`af.Nautilus` today, with the registry shape ready for `af.DynestyStatic`,
`af.BlackJAXNUTS`, `af.Emcee`, etc. Unlike `likelihood_runtime/` (which
profiles `analysis.log_likelihood_function` in isolation), every cell here
runs `search.fit(model=model, analysis=analysis)` — so visualization,
samples I/O, `samples_info.json`, latent variables, and every other piece
of PyAutoFit machinery is exercised and measured.

## Design

| Dimension      | Values                                                                    |
|----------------|---------------------------------------------------------------------------|
| Sampler        | `nautilus` (more to come via `_samplers.SAMPLER_BUILDERS`)                 |
| Dataset class  | `imaging`, `interferometer`, `point_source`, `datacube`                   |
| Model type     | `mge`, `pixelization`, `delaunay`, `image_plane`, `source_plane`          |
| Instrument     | per-dataset-class (HST/Euclid/JWST/AO; SMA/ALMA/ALMA-high/JVLA; simple)   |
| Hardware       | `local_cpu`, `local_gpu`, `hpc_a100` (external dispatch)                  |
| Precision      | `fp64`, `mp` (mixed precision via `al.Settings(use_mixed_precision=...)`) |

Layout:

```
searches/
  README.md                 # this file
  _setup.py                 # dataset/model/analysis dispatchers
  _samplers.py              # sampler registry + per-(ds, model) n_live
  _metrics.py               # viz wall-time interception + result reader
  _runner.py                # shared driver (every leaf calls run_search)
  sweep.py                  # matrix driver, resume-by-default
  aggregate.py              # comparison.json + comparison.png per cell
  nautilus/
    imaging/{mge, pixelization, delaunay}.py
    interferometer/{mge, pixelization, delaunay}.py
    point_source/{image_plane, source_plane}.py
    datacube/delaunay.py
```

## Key design choices

**First-class only.** No more wrapping `nautilus.Sampler` directly. The
old `simple.py` / `jax.py` scripts are deleted. Every cell goes through
`af.Nautilus.fit(model, analysis)`, so visualization, output writes,
sample I/O, and latent-variable computation are part of the profile.

**SLaM-matched `n_live`.** Per `autolens_workspace/scripts/guides/modeling/
slam_start_here.py`: MGE / point-source / parametric phases use
`n_live=200` (matches `source_lp[1]`); pixelization / Delaunay phases
use `n_live=150` (matches `source_pix[1]`).

**`number_of_cores=1` always.** This profile measures per-evaluation
end-to-end cost. Production scaling via `number_of_cores > 1` is a
separate axis a future sweep can introduce.

**JAX rows force `force_x1_cpu=True` and `use_jax_vmap=True`.** This is
mandatory: `nautilus.Sampler` forking under multiprocessing corrupts
JAX state. The trade-off is one batched evaluation per Nautilus step.

**Visualization wall-time is split out.** `_metrics.attach_viz_timer`
wraps every visualize-family hook on the analysis (`visualize`,
`visualize_combined`, `visualize_before_fit`,
`visualize_before_fit_combined`) plus the search's `plot_results`. The
JSON reports `total_wall_s`, `viz_wall_s` and the derived
`sampler_wall_s = total_wall_s - viz_wall_s` so you can ask both "how
long did the full first-class fit take?" and "how much was viz?".

**`sweep.py` wipes search state by default.** PyAutoFit's resume gate is
the `.completed` sentinel file under `<output_path>/searches/...` — once
a `search.fit()` finishes sampling, that file is written and the next
attempt at the same `path_prefix` short-circuits to a cached-result load.
For *production* (SLaM-style chained phases) this is correct behaviour.
For *profiling* it produces 2-3× phantom speedups when a re-run after
a post-fit crash hits the cached `samples.csv`. `sweep.py` therefore
removes `<output_path>/searches/<sampler>/<ds>/<model>/<instrument>/<config>/`
before each cell run by default. Pass `--keep-completed` to opt out
(e.g. when iterating on the post-fit visualization path).

`force_pickle_overwrite=True` is also set on every search, but it only
controls whether output pickles in the `files/` directory get re-written
when an existing search is *resumed* — it does **not** bypass the
`.completed` gate. The sweep-level wipe is what makes re-runs honest.

## Datacube multi-channel fitting

`datacube/delaunay.py` fits `_DATACUBE_N_CHANNELS` (default 4) identical
interferometer channels via `af.FactorGraphModel`. Each channel becomes
its own `al.AnalysisInterferometer`, wrapped in an `af.AnalysisFactor`
paired with `model.copy()`, then combined under a single global model —
the same pattern documented in
`autolens_workspace/scripts/multi/modeling.py`. The N channels are
identical copies of the per-instrument dataset; the profile measures
cube-cost scaling, not band-wavelength variation.

To change the channel count, edit `_DATACUBE_N_CHANNELS` in `_setup.py`
(34 matches the existing ALMA cube fiducial; 4 keeps profiling
turnaround sane).

## What this *doesn't* profile (yet)

- **Pool scaling.** `number_of_cores > 1` sweeps are future work.
- **Adapt-image regeneration across phases.** Pixelization / Delaunay
  cells use a truth-derived `lensed_source.fits` cached next to the
  dataset. Production SLaM regenerates this between phases.
- **A100 dispatch.** The local sweep generates only CPU and laptop-GPU
  rows. The `hpc_a100_fp64` / `hpc_a100_mp` config names exist in
  `sweep.py` for parity with `likelihood_runtime/`; the actual dispatch
  to RAL HPC happens externally (same mechanism as the likelihood
  sweep).
- **Samplers other than Nautilus.** The registry is in place; adding
  `dynesty`, `blackjax_nuts`, `emcee`, etc. is one function per sampler
  in `_samplers.py`.

## Running

Single cell (CPU NumPy, fastest path):

```bash
python searches/nautilus/imaging/mge.py \
    --instrument hst --config-name local_cpu_fp64
```

Single cell (laptop GPU, JAX-vmap):

```bash
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python searches/nautilus/imaging/mge.py \
    --instrument hst --config-name local_gpu_fp64
```

Full sweep (every cell × instrument × config) — warning, this is long:

```bash
python searches/sweep.py
```

Iteration sweep (one cell, one instrument, CPU only):

```bash
python searches/sweep.py \
    --only nautilus/imaging/mge \
    --instrument hst \
    --skip-gpu --skip-mp
```

Aggregate post-sweep:

```bash
python searches/aggregate.py
```

## Output layout

```
results/searches/
  <sampler>/<dataset_class>/<model>/<instrument>/
    <config_name>.json         # per-config headline metrics
    <config_name>.png          # per-config bar chart
    <config_name>.log          # subprocess stdout/stderr (sweep only)
    comparison.json            # cross-config aggregation (aggregate.py)
    comparison.png             # cross-config bar chart (aggregate.py)
```

The PyAutoFit search itself writes its own output (`samples.csv`,
`samples_info.json`, `search.summary`, visualization, ...) to the
autoconf `output_path` under `path_prefix=searches/<sampler>/
<dataset_class>/<model>/<instrument>`. The metric JSON+PNG above live
separately under `results/searches/`.
