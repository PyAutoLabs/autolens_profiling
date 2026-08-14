# Numerical-hazard profiling

This instrument records numerical properties that change how a likelihood
surface behaves under sampling: flat saturations, non-finite derivatives,
backend divergence, and scale-dependent conditioning mechanisms. It complements
the runtime profiles; a fast evaluation is not sampler-friendly if its surface
contains a NaN gradient or a broad plateau.

Reusable detectors live here, independent of what they inspect. Dataset-specific
fixtures and cells belong under `scripts/<dataset>/hazards/`. A finding carries
its tier as metadata and declares one subject scope:

- `component` — a profile or lensing calculation, with no dataset;
- `matrix` — synthetic linear-algebra inputs, with no dataset;
- `likelihood` — a real dataset and complete likelihood (phase 2).

Risk is typed. The schema uses `prior_mass`, `epsilon_neighbourhood`,
`reachability`, or `error_curve`; it never forces measure-zero and continuous
hazards through Monte Carlo prior volume.

## Running

From the repository root:

```bash
python scripts/misc/hazards/scan.py
python scripts/misc/hazards/scan.py --subject component --backend jax
python scripts/misc/hazards/scan.py --subject likelihood
python scripts/misc/hazards/scan.py --check
```

The normal scan writes per-check JSON/PNG pairs and the generated seed summary
under `results/hazards/`. `--check` re-runs the reproducers without writing and
returns non-zero when a new semantic finding ID appears. A moved source anchor
does not create a new finding; persistence comes from the reproducer, while the
token fingerprint helps relocate the implementation.

## Findings

<!-- BEGIN auto-table:hazards -->
| Finding | Subject | Hazard | Risk basis | Backends |
|---|---|---|---|---|
| `component.ell_comps.magnitude-saturation` | `component` | `saturation` | prior_mass, reachability | numpy, jax |
| `component.isothermal.near-spherical-saturation` | `component` | `saturation` | epsilon_neighbourhood, reachability | numpy, jax |
| `component.power-law.series-vs-hyp2f1-divergence` | `component` | `backend_divergence` | error_curve, reachability | numpy, jax |
| `component.spherical-geometry.radial-sqrt-gradient-at-zero` | `component` | `nonfinite_gradient` | epsilon_neighbourhood, reachability | jax |
| `likelihood.imaging-pixelization.absolute-conditioning-floors` | `likelihood` | `conditioning_floor` | error_curve, reachability | numpy |
| `likelihood.imaging-pixelization.nnls-active-set-kinks` | `likelihood` | `active_set` | epsilon_neighbourhood, error_curve | numpy |
| `likelihood.imaging-sersic.ell-comps-origin-nonfinite-gradient` | `likelihood` | `nonfinite_gradient` | epsilon_neighbourhood, error_curve | jax |
| `matrix.curvature.absolute-diagonal-floor` | `matrix` | `conditioning_floor` | error_curve, reachability | numpy, jax |
<!-- END auto-table:hazards -->

The tier-1 slice intentionally proves four shapes rather than scanning the full
`component × backend` matrix:

1. `ell_comps` and Isothermal saturation;
2. the measure-zero radial `sqrt` gradient at `r=0`;
3. PowerLaw `hyp2f1` versus the fixed 20-term JAX series;
4. the absolute curvature-diagonal floor on synthetic matrix scales.

Tier 2 adds a full [`FitImaging` cell](../../imaging/hazards/README.md) around a
rectangular source inversion. The generated
`component/profile_registry_coverage.json` records public classes from `al.lp`,
`al.lp_linear`, `al.lmp`, and `al.mp`, de-duplicating aliases without running a
full profile-by-backend matrix.

The first index consumer is `scripts/misc/likelihood_runtime/aggregate.py`.
`comparison.json` includes a `hazard_findings` list for the
`imaging/pixelization` cell. A missing or invalid hazard index produces an empty
list plus a warning so runtime aggregation remains available.
