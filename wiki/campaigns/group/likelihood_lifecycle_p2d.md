# Likelihood lifecycle phase 2d: group at the standard size

**Status:** draft
**Question:** What does every planned group/{mge, rectangular, delaunay} cell cost at the standard size on the pinned release 2026.10.25.1, on the §3.3 "as imaging" setups, with a parity witness?
**Pre-registered rule:** the 24 planned group cells (10 RAL CPU, 14 A100) land `current` with runtime and breakdown, provenance, `resources`, `fit.png` and a `parity.json` whose status is `match` against the eager NumPy dense reference (1e-4 x |reference| in fp64). The campaign's rule, decisions and batch plan are on the shared 2d page; this page holds the group-specific parts. Written 2026-10-28, before any run.
**Verdict:** pending (nothing run). Expected: group is slower than multi_galaxy for every setup, and it bounds multi_galaxy's walls (batch 0 runs every group leg, decision D9); the two dPIE extra galaxies add deflection cost only; every witness is `match`.
**Headline:** none yet.
**Library PRs:** none (measurement only, tag 2026.10.25.1)
**Profiling PRs:** the 2d draft PR, stacked on #471
**Ledger:** the shared campaign page [`../multi_galaxy/likelihood_lifecycle_p2d.md`](../multi_galaxy/likelihood_lifecycle_p2d.md); rows will land under `results/group/{mge,rectangular,delaunay}/euclid/`
**Mind contract:** epic `likelihood-lifecycle`, phase 2 sub-issue 2d; issue autolens_profiling#473
**Next:** the human rules on D1-D11 (shared page); then batch 0, which is every group leg
**Fit figures:** none yet

## The group cell

- Dataset (D2, D3): the workspace group simulator at 0.1"/px (`euclid`), 250x250, exposure
  300 s, sky 0.1, the euclid instrument PSF (21x21, sigma 0.1"); one main lens (SersicSph +
  IsothermalSph, Einstein radius 4.0"), two extra galaxies at (3.5, 2.5) and (-4.4, -5.0)
  (SersicSph + `dPIEMassSph`, sigma 212 and 239 km/s), a SersicCore source; 7.5" mask, 17,692
  masked pixels; light-profile over-sampling on all three galaxy centres.
- Model (D4): the main lens an MGE of 20 Gaussians + `Isothermal`, an external shear, each
  extra galaxy an MGE of 10 Gaussians (centre fixed) + `dPIEMassSph` (sigma free, Gaussian prior
  of width 10 km/s on the truth), the source an MGE of 20 Gaussians or the standard-size
  pixelization (D5).
- The point_source analogy (D6) does not apply to these cells: they are extended-source
  `AnalysisImaging` fits. A group point-source likelihood would follow the point_source rules.

## Cells

| Likelihood | RAL CPU setups | A100 setups |
|---|---|---|
| group/mge | `jax_cpu_dense_fp64` | `jax_gpu_dense_fp64` |
| group/rectangular, /delaunay | `numba_cpu_sparse_fp64`, `jax_cpu_sparse_fp64` | `jax_gpu_dense_fp64`, `jax_gpu_sparse_fp64`, `jax_cpu_sparse_fp64` |

## Journal

### 2026-10-28 — draft for review

Cells declared and scripts written with the multi_galaxy cells; see the shared page.
