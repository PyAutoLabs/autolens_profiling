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
| `component.mge.faddeeva-seam-gradient` | `component` | `nonsmooth_objective` | error_curve, reachability | jax |
| `component.mge.spherical-clamp-bias` | `component` | `backend_divergence` | error_curve, reachability | numpy, jax |
| `component.power-law.series-vs-hyp2f1-divergence` | `component` | `backend_divergence` | error_curve, reachability | numpy, jax |
| `component.spherical-geometry.radial-sqrt-gradient-at-zero` | `component` | `nonfinite_gradient` | epsilon_neighbourhood, reachability | jax |
| `likelihood.imaging-pixelization.absolute-conditioning-floors` | `likelihood` | `conditioning_floor` | error_curve, reachability | numpy |
| `likelihood.imaging-pixelization.nnls-active-set-kinks` | `likelihood` | `active_set` | epsilon_neighbourhood, error_curve | numpy |
| `likelihood.positions-penalty.argmax-switch` | `likelihood` | `nonsmooth_objective` | error_curve, reachability | jax |
| `likelihood.positions-penalty.interior-plateau` | `likelihood` | `zero_gradient_region` | error_curve, reachability | jax |
| `likelihood.positions-penalty.threshold-hinge` | `likelihood` | `nonsmooth_objective` | error_curve, reachability | jax |
| `matrix.curvature.absolute-diagonal-floor` | `matrix` | `conditioning_floor` | error_curve, reachability | numpy, jax |
<!-- END auto-table:hazards -->

The tier-1 slice intentionally proves four shapes rather than scanning the full
`component × backend` matrix:

1. `ell_comps` and Isothermal saturation;
2. the measure-zero radial `sqrt` gradient at `r=0`;
3. PowerLaw `hyp2f1` versus the fixed 20-term JAX series;
4. the absolute curvature-diagonal floor on synthetic matrix scales.

The PowerLaw finding has a bounded convergence study in
`power_law_omega.py` and
`results/hazards/component/power_law/omega_convergence.json`. The 20-term
series exceeds `1e-4` angular error over 5.7% of the packaged default prior.
A fixed-bin counterfactual retains reverse-mode differentiation and `vmap`, but
covering the live `0.999` ellipticity clamp requires 10,240 terms and was 125x
slower than the current path there on the recorded CPU run. The complete 7x7
`FitImaging` fixture moved by at most 0.0056 log-likelihood units. That evidence
does not support routing the high-cost counterfactual into PyAutoGalaxy; the
stable finding remains profiling evidence.

The two `component.mge` findings have their own bounded study in `mge_faddeeva.py` and
`results/hazards/component/mge/faddeeva_audit.json` (issue PyAutoGalaxy#600, phase A). The JAX
Faddeeva routine's three `xp.where` regions leave its derivative discontinuous by up to 5.8e-5
relative at the `r2 = 2.5` seam, and finite differences of the JAX deflection field diverge as the
step shrinks (58-96 of 15,361 grid points wrong by more than 1 % at `h <= 1e-6`) while the same
comparison in a crossing-free parameter direction stays smooth to 2.9e-11. The kinks are not
measurable further downstream: on a 2000-step `centre_x` transect and on a bounded `FitImaging`
gradient the autodiff gradient is no rougher than the smooth-path baseline. The spherical clamp
biases every `*Sph` MGE-routed profile by 6.3e-5 (gNFWSph) / 1.1e-4 (Gaussian) relative, with a
spurious cross-axis deflection reaching 1.45e-4". The full verdict - lift the clamp on JAX, replace
the routine with the seam-free Weideman N=32 series (1.9e-13 accurate, 0.745x the cost) - is in
`results/notes/numpy_deflections_cpu.md`.

Phase B implemented both. On a PyAutoGalaxy checkout carrying that change `scan.py --check
--subject component` reports both `component.mge` findings **resolved**: the routine's derivative
now steps across the former `r2 = 2.5` boundary by exactly the amount `w'` genuinely moves there
(excess factor 1.000, against 2.28e4 for the routine it replaced), and a spherical profile's
deflection is purely radial again (cross-axis 5e-16", against 1.4e-4"). Each reproducer gates on
that mechanism rather than on an error magnitude, so the verdict does not depend on a tuned
threshold; both gates still fire against the pre-phase-B library. The records and index rows are
kept - they remain true of the released library until the change ships - and the after-state
numbers are the "After phase B" section of the note.

Tier 2 adds a full [`FitImaging` cell](../../imaging/hazards/README.md) around a
rectangular source inversion. The generated
`component/profile_registry_coverage.json` records public classes from `al.lp`,
`al.lp_linear`, `al.lmp`, and `al.mp`, de-duplicating aliases without running a
full profile-by-backend matrix.

The first index consumer is `scripts/misc/likelihood_runtime/aggregate.py`.
`comparison.json` includes a `hazard_findings` list for the
`imaging/pixelization` cell. A missing or invalid hazard index produces an empty
list plus a warning so runtime aggregation remains available.
