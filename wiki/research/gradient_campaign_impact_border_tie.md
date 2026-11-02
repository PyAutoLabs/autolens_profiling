# Did the PyAutoArray#671 gradient defects touch the phase-6 gradient campaign?

Research note for autolens_profiling#515, 2026-11-02. The human asked for it while PyAutoArray#671 was being fixed.

**Answer: no.** Neither defect changes any of the 73 phase-6 `gradient.json` verdicts in `results/`, and neither changes the [day_zero_gradient snapshot](https://github.com/PyAutoLabs/PyAutoPulse/blob/main/snapshots/day_zero_gradient/lens.json). No cell needs a re-run on account of #671.

## The two defects

PyAutoArray#671 started from a gradient of |g| ≈ 2e9 on the `imaging/rectangular/hst` *fiducial* cell (39x39, no `--standard-size`), a side run that is not in `results/`. It turned out to have two separate causes:

1. **Border-relocator tie (a library bug, fixed in PyAutoArray#672, pending release).**
   - The border relocator sizes its ellipse as `b = max(|y′|)` over the border points.
   - When the lens sits exactly on the mask's symmetry centre, the traced border comes in exact ± pairs, so that maximum is a tie.
   - Under `jit`, JAX's gradient of `max` at a tie came out 2x or 0.5x the true value (eager mode was correct).
   - So the defect needs all three: `jit`, the border relocator on, and a lens at exactly the symmetry centre.
2. **Isothermal core at a grid point (an evaluation-point artefact, not a bug).**
   - On an odd grid (HST is 141x141) at pixelization over-sampling 1, one source-grid point sits exactly at (0,0).
   - At the prior medians the Isothermal centre is also exactly (0,0).
   - The `psi` softening (a 1e-8 core) then gives d(deflection)/d(centre) ≈ 1e8. Autodiff is exact for the softened profile; FD steps cannot resolve the core.

## Why the campaign cells are clear

- **Every pixelized campaign cell is a `--standard-size` cell.** All the standard drivers (imaging via `standard_jax_cell`; interferometer, datacube and multi_dataset via their `likelihood_breakdown/*_standard.py`; group and multi-galaxy via `galaxy_group_standard.py`) evaluate at **iid draw 0 of the seeded production sequence** (`_production_config.iid_instances`, seed 235, unit values in [0.4, 0.6]), not at the prior medians.
  - For the imaging presets the mass centre at that draw is (0.00085, −0.00035)″. Other families draw from the same seeded stream in their own prior order, so their centre is a different but equally generic value; it would be exactly 0 only if a unit value were exactly 0.5.
  - That is not the mask's symmetry centre, so the border has no exact ± ties: defect 1 cannot trigger.
  - It is not a grid point (pixel scales 0.05–0.1″): defect 2 cannot trigger.
- **The cells that really are evaluated at the prior medians are not pixelized** (imaging and interferometer MGE, point source, cluster, weak). There is no border relocator, so defect 1 cannot apply.
  - In the imaging MGE cells the mass and shear are *fixed* (`mass.centre = (0.0, 0.0)`, not free), so the gradient never passes through the Isothermal core either, and the light sub-grid uses even sub-sizes (4/2/2).
- **Measured control.** `interferometer/rectangular/sma` was re-run locally with the campaign flags (`--standard-size --sparse --config-name hpc_ral_cpu_fp64`), with PyAutoArray before (`c7da8b0f`) and after (`d7b642e2`) #672:
  - the gradient is **bit-identical** (largest relative change 2.4e-16);
  - it reproduces the stored campaign record (log L to 1e-12, directional derivatives to 1e-5).
- **Small-step check over all 73 cells.** #672 does not change any likelihood value, so the stored FD numbers stay valid. In most pixelized cells the stored autodiff matches the stored central difference at h = 1e-8 to ≲1e-3. A defect-1 or defect-2 error would show up there as a disagreement at every small step.

## What the remaining disagreements are (none is from #671)

| Cells | Pattern | Most likely cause |
|---|---|---|
| `interferometer/rectangular/{sma, alma, alma_high, jvla}` | autodiff vs FD gap about 1 at every step, both backends | log L jitters by about 1e-6 between evaluations, so FD at small steps is noise against a small gradient (sma: \|g\| = 0.23; h = 1e-8 FD gave 52/5/50 in the campaign and 13/−40/15 locally) |
| `datacube/delaunay/sma`, `interferometer/delaunay/sma` | gap of order 1 at h = 1e-8 | the same noise floor against a small gradient (\|g\| 13–600) |
| `group/delaunay/euclid`, `imaging/delaunay/jwst`, `imaging/delaunay_nn/hst`, `interferometer/delaunay/jvla` | CPU-sparse disagrees at small steps, GPU-dense agrees | unexplained CPU-sparse vs GPU-dense gradient difference; not the border relocator (the evaluation point has no ties) |
| `imaging/mge/{ao, euclid, hst, jwst}` | constant 0.1–0.4 % gap at every step, CPU = GPU | unexplained; no pixelization and no mass gradient, so outside both defects |
| `interferometer/mge/sma` | 70 % gap in one direction | unexplained (NUFFT MGE) |
| `point_source_image/image_plane`, `cluster/image_plane` | autodiff exactly zero | image-plane solver blocks the gradient (known class) |

The pixelized `no_plateau` verdicts come from FD alone. #672 leaves values unchanged, so it cannot move them; the human's 2026-11-02 ruling that the Delaunay roughness is intrinsic stands.

## Side finding: the recorded evaluation point is mislabelled

Every standard driver except `galaxy_group_standard.py` calls `harness.run_gradient(...)` without a `point=` argument, so the imaging, interferometer, datacube and multi_dataset `gradient.json` files say `"point": "prior_medians"` although they evaluate at iid draw 0. The group and multi-galaxy cells pass it and are labelled correctly. The label is what made the #671 defects look like a campaign risk; a follow-up should pass the real point and correct the stored labels.

## Where the work is

- Classification script and before/after scripts: `Science/scratch/gradient_campaign_impact_2026-11-02/` (`small_step_check.py`, `ab_cell.sh`, `ab_compare.py`).
- #671 investigation: PyAutoMind `complete/2026/11/rectangular-gradient-rtu-sparse.md`; repro in `Science/scratch/rectangular_gradient_rtu_sparse_2026-11-02/`.
