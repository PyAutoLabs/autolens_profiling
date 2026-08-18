# `searches/` — first-class search profiling

This section profiles **first-class PyAutoFit search objects** end-to-end:
`af.Nautilus` today, with the registry shape ready for `af.DynestyStatic`,
`af.BlackJAXNUTS`, `af.Emcee`, etc. Unlike `likelihood_runtime/` (which
profiles `analysis.log_likelihood_function` in isolation), every cell here
runs `search.fit(model=model, analysis=analysis)` — so visualization,
samples I/O, `samples_info.json`, latent variables, and every other piece
of PyAutoFit machinery is exercised and measured.

## Latest results

<!-- BEGIN auto-table:searches -->
| Sampler | Cell | Config | max logL | logZ | Wall | Evals | Time / eval | Version |
|---------|------|--------|---------:|-----:|-----:|------:|------------:|---------|
| `multi_start_adam` | `group/mge/hst` | `local_gpu_fp64` | -231,891.0 | — | 2368.26 s | 33 | 71765.4 ms | v2026.7.9.1 |
| `multi_start_prodigy` | `cluster/image_plane_solved/simple` | `hpc_a100_fp64` | -1,708.9 | — | 832.23 s | 65 | 8727.0 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `cluster/source_plane_solved/simple` | `hpc_a100_fp64` | 21.4 | — | 250.86 s | 65 | 1853.8 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `cluster/source_plane_tensor/simple` | `hpc_a100_fp64` | -11,000.5 | — | 261.36 s | 65 | 1182.0 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `default` | -79.9 | — | 852.82 s | 65 | 13120.3 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `hpc_a100_fp64` | -68.5 | — | 90.31 s | 65 | 923.8 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane/simple` | `starts256` | -47.7 | — | 3515.50 s | 257 | 13679.0 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane_solved/simple` | `default` | 2.4 | — | 981.87 s | 65 | 15105.7 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/image_plane_solved/simple` | `hpc_a100_fp64` | 9.7 | — | 117.74 s | 65 | 1243.7 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane/simple` | `default` | -109.7 | — | 19.41 s | 65 | 298.6 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane/simple` | `hpc_a100_fp64` | -109.7 | — | 35.85 s | 65 | 401.4 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane_solved/simple` | `hpc_a100_fp64` | 4.5 | — | 44.77 s | 65 | 550.3 ms | v2026.7.23.1 |
| `multi_start_prodigy` | `point_source/source_plane_tensor/simple` | `hpc_a100_fp64` | 13.4 | — | 31.68 s | 65 | 338.6 ms | v2026.7.23.1 |
| `nautilus` | `cluster/image_plane_solved/simple` | `hpc_a100_fp64` | 31.5 | -1.8 | 742.08 s | 8,400 | 44.1 ms | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane/simple` | `hpc_a100_fp64` | 42.3 | -36.5 | 551.28 s | 34,600 | 8.5 ms | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane_solved/simple` | `hpc_a100_fp64` | 43.4 | 12.7 | 442.81 s | 7,200 | 14.4 ms | v2026.7.23.1 |
| `nautilus` | `cluster/source_plane_tensor/simple` | `hpc_a100_fp64` | 69.8 | 8.5 | 482.73 s | 20,500 | 8.8 ms | v2026.7.23.1 |
| `nautilus` | `imaging/delaunay/hst` | `hpc_a100_fp64` | 30,623.5 | 30,562.2 | 2722.51 s | 31,536 | 84.8 ms | v2026.5.21.1 |
| `nautilus` | `imaging/mge/hst` | `hpc_a100_fp64` | 31,786.8 | 31,690.5 | 831.28 s | 63,800 | 12.1 ms | v2026.5.21.1 |
| `nautilus` | `imaging/pixelization/hst` | `hpc_a100_fp64` | 29,143.3 | 29,066.3 | 2768.06 s | 58,464 | 46.5 ms | v2026.5.21.1 |
| `nautilus` | `point_source/image_plane/simple` | `default` | 9.6 | -16.9 | 739.70 s | 13,760 | 53.8 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane/simple` | `hpc_a100_fp64` | 9.6 | -16.8 | 217.11 s | 14,464 | 9.8 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_repeat_solved/simple` | `hpc_a100_fp64` | 7.9 | -8.0 | 162.72 s | 7,100 | 15.5 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_repeat_solved/simple_missing` | `hpc_a100_fp64` | 13.1 | -8.2 | 186.72 s | 9,800 | 13.5 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/near_caustic` | `hpc_a100_fp64` | 29.1 | 5.0 | 176.35 s | 10,500 | 11.7 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/simple` | `hpc_a100_fp64` | 10.6 | -5.3 | 147.21 s | 7,500 | 12.2 ms | v2026.7.23.1 |
| `nautilus` | `point_source/image_plane_solved/simple_missing` | `hpc_a100_fp64` | 22.1 | 4.8 | 193.29 s | 11,700 | 12.0 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane/simple` | `default` | -313.2 | -343.9 | 168.00 s | 15,552 | 10.8 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane/simple` | `hpc_a100_fp64` | -313.9 | -344.7 | 217.83 s | 14,400 | 11.2 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_solved/near_caustic` | `hpc_a100_fp64` | 16.5 | -7.8 | 112.85 s | 10,200 | 9.1 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_solved/simple` | `hpc_a100_fp64` | 6.4 | -10.6 | 86.58 s | 7,700 | 8.3 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_tensor/near_caustic` | `hpc_a100_fp64` | 34.5 | -8.9 | 270.32 s | 23,400 | 9.3 ms | v2026.7.23.1 |
| `nautilus` | `point_source/source_plane_tensor/simple` | `hpc_a100_fp64` | 15.3 | -11.8 | 184.36 s | 14,900 | 9.0 ms | v2026.7.23.1 |
| `nss` | `imaging/mge/hst` | `hpc_a100_fp64` | 31,786.5 | 31,697.7 | 679.40 s | 394,321 | 1.6 ms | v2026.5.21.1 |
<!-- END auto-table:searches -->

Auto-generated by `scripts/misc/tooling/build_readme.py` from the search-run artifacts under `results/searches/<sampler>/<class>/<model>/<instrument>/` (latest version per sampler × cell × config).

## Design

| Dimension      | Values                                                                    |
|----------------|---------------------------------------------------------------------------|
| Sampler        | `nautilus`, `multi_start_{adam,prodigy,prodigy_autoconv,lion,adabelief}` (via `_samplers.SAMPLER_BUILDERS`) |
| Dataset class  | `imaging`, `interferometer`, `point_source`, `datacube`, `group`         |
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

**MAP optimizers alongside samplers.** `multi_start_adam` (`af.MultiStartAdam`,
a JAX/optax multi-start gradient MAP optimizer) is registered as a first-class
search too, but only for the `imaging/mge` cell — the benchmark-proven cell where
a gradient MAP optimizer is meaningful (pixelization/Delaunay/interferometer/
point-source are outside its use case). It is JAX-only (a pure-NumPy config
raises) and has no `n_live` (it records `n_starts`/`n_steps`; the JSON stores
`n_live: null`).

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

## Group-scale truth-recovery benchmark (`group/mge`)

The `group` dataset class is the high-dimensional stress test for the JAX
gradient MAP optimizers (autolens_profiling#82): **4 deflector galaxies**
(MGE light + `Isothermal` mass, `ExternalShear` on the primary) lensing **4
background MGE sources** — ~54 free parameters, versus ~14 for the single-lens
`imaging/mge` cell. It answers "do the multi-start gradient optimizers scale to
a harder, higher-dimensional model, and do they recover the input truth?"

- **Simulator.** `simulators/group4_mge.py` builds the tracer from a single
  truth structure (`GROUP4_TRUTH`) and writes the dataset **plus a
  `truth.json`** to `dataset/imaging/group4_mge/<instrument>/`. Auto-simulated
  on first run via the standard `auto_simulate_if_missing` hook
  (`dataset_type="group4_mge"`).
- **Centres are seeded, geometry is not.** Every galaxy's light + mass centre
  gets a modest-sigma Gaussian prior at its known position — the honest prior
  for individually-visible group members, and what breaks the permutation
  symmetry among the 4 lenses / 4 sources. Einstein radii, ellipticities and
  shear keep broad default priors, so the search still has to *find* the mass
  model.
- **Truth recovery is scored.** `searches/_recovery.py` compares the fit's
  `max_log_likelihood_instance` to `truth.json` (per-lens Einstein-radius
  fractional error + mass-centre distance, primary shear, per-source centre)
  and writes a `"recovery"` block — including `overall_pass` — into the summary
  JSON. Nautilus (`nautilus/group/mge`) is the **reference anchor**: if it can't
  recover here, the simulation/model is wrong before any optimizer is judged.
- **Two optimizer modes.** The MultiStart family runs both **fixed-step**
  (`n_steps=300`) and, for `multi_start_prodigy_autoconv`,
  **auto-convergence** (each start early-stops via
  `af.MultiStartGradientConvergence` when its figure-of-merit plateaus). The
  `sampler_config` block records the convergence criterion so the
  early-stop-vs-fixed-300 comparison is self-describing.

## Datacube multi-channel fitting

`datacube/delaunay.py` fits `_DATACUBE_N_CHANNELS` (default 4) identical
interferometer channels via `af.FactorGraphModel`. Each channel becomes
its own `al.AnalysisInterferometer`, wrapped in an `af.AnalysisFactor`
paired with `model.copy()`, then combined under a single global model —
the same pattern documented in
`autolens_workspace/scripts/multi_dataset/modeling.py`. The N channels are
identical copies of the per-instrument dataset; the profile measures
cube-cost scaling, not band-wavelength variation.

To change the channel count, edit `_DATACUBE_N_CHANNELS` in `_setup.py`
(34 matches the existing ALMA cube fiducial; 4 keeps profiling
turnaround sane).

## Standalone instruments

- **`multi_start_nan_accounting_overhead.py`** — does `MultiStartGradient`'s
  per-step value-NaN / gradient-NaN accounting (PyAutoFit#1472) cost any run
  time? Times four ways of obtaining per-lane gradient finiteness against a
  real likelihood, with a duplicate-baseline **control** so the measurement
  noise floor is reported alongside the overhead — the shipped `fused` variant
  is meant to sit below it. Counter-intuitively the `eager` variant (reduce on
  device but outside the jit) is the *worst*, not the best: an un-jitted
  reduction buys a kernel dispatch plus a host round-trip to save a few KB that
  were never the cost. Run it on **both** CPU and GPU — on CPU a device→host
  copy is a same-address-space memcpy, so the `host` variant's true cost is
  invisible there.

  Local CPU row (`mge`, 16 starts, ndim 15, 1.03 s/step): **fused 4.1 us
  (0.0004% of a step)**, host 7.0 us, eager 29.4 us. Absolute values move ~50%
  between CPU runs; the *ordering* (fused < host < eager) has been stable across
  three independent measurements, so read the ranking as the result and the
  magnitudes as order-of-magnitude.

  The `fused` figure comes from a proxy objective at the real array shapes, not
  from the real likelihood — differencing two ~1 s jitted calls cannot resolve a
  ~4 us effect (it returns negative "costs"). That makes it an upper bound on
  the reduction in isolation; XLA's fusion into the real backward pass is not
  captured. The end-to-end loop is ~1200x too coarse to see any of this and
  reports itself as bounding, not measuring.

  Related but distinct: `misc/hazards/checks/nonfinite_gradient.py` *detects*
  non-finite gradients on a likelihood surface; this one measures what it costs
  to *count* them during a fit.

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

## Pixelized-mesh multi-start cells (#117 campaign knowledge)

The `multi_start_prodigy/{pixelization,knn,delaunay}.py` imaging cells were
promoted from the autolens_workspace_developer#117 broad-start campaign
(2026-07; full record: `autolens_workspace_developer/searches_minimal/
pix_prodigy_findings.md`). The durable lessons their configs encode:

- **Gradient multi-start works on pixelized sources** — the #100/#101
  "Nautilus wins pix decisively" verdict inverted once the library search
  gained per-start vmapped state, lr-free Prodigy, and resurrection. knn:
  +29724 @ r_E 1.599 vs a matched-settings Nautilus's +5704 @ r_E 1.011.
- **The regularization axis decides searchability**, not the mesh landscape:
  AdaptSplit's double-squared coefficients make its high-coefficient region
  an escape-taxed floor (knn) or an outright NaN wall (delaunay). Fixed or
  inherited reg (the SLaM `source_pix[1]` pattern) is the fast path
  (~150-250 steps to truth); free Matérn is the safe free parametrization
  (same fit ceiling, no wall) — hence the `delaunay_matern` model type.
- **Budgets**: 16 starts recover the basin; `batch_size=4` is mandatory
  (unbatched 16-start pixelized jvp ≈ 58 GB); 3000 steps because reg-mode
  crossings arrive late (~1300-2000 steps) — a long plateau is a reg mode,
  not convergence.
- **Mesh smoothness class predicts gradient efficiency** (kernel-CDF C∞ >
  knn Wendland > delaunay C0-at-flip-seams) and, by extension, which
  posterior kernels each mesh can host (Hamiltonian on the smooth meshes;
  tempered SMC or warm refits for delaunay).
- **Rectangular caveats (campaign-close state)**: the kernel-CDF
  `value_and_grad` step cost is anomalously high on CPU (~17x knn vs ~4.5x
  forward-eval ratio — profile it on the A100 before drawing landscape
  conclusions), and its *fixed-reg* arm stalled where every other mesh's
  converged — implicating the sharp `bandwidth=0.1` (narrow gradient
  support), with `bandwidth=1.0` costing only ~1.4k nats of fit ceiling.
  Prefer "search smooth, refine sharp" as an annealing schedule; do NOT free
  bandwidth as a model parameter without first checking the joint
  (bandwidth, reg) evidence scan for an interior optimum — a MAP objective
  may rail it at the staircase limit.
- **Ops**: multi-start resume chains do not survive library upgrades that
  touch FoM bookkeeping (the resume sanity check refuses, by design) — pin
  the HPC mirrors for a campaign or plan to restart in-flight chains.
