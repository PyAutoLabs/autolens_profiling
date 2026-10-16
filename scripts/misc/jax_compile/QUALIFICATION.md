# HST/MGE pilot correctness qualification design

Reviewed: 2026-10-13. Status: **proposed; no scientific acceptance**.
Scope: the fixed-parameter CPU/fp64 `imaging-mge-runtime-v1` pilot, `jit` only.
This source audit implements the documentation phase in [project #417]. It
executes no likelihood, reference solver, inference or production measurement.
The [raw producer](PILOT.md) and [Pulse pilot contract] retain their existing
qualification blockers. A reviewed design is not a computed reference.

## Decision

Use an independently constructed imaging operator and an independent constrained
linear solve as the eventual witness. First implement a small synthetic
correctness harness in `autolens_workspace_test`, with a separate project-owned
adapter linking its artifacts to the pilot. Neither implementation is included
here. Freeze numerical tolerances only after reviewing that harness's evidence
and the selected production inputs; never tune them after viewing production
disagreements.

A NumPy `FitImaging` comparison is useful diagnostic evidence but shares lens
tracing, profiles, masking and inversion assembly with the JAX path. Re-solving
an exported library matrix independently certifies only the solve. Neither
establishes independent end-to-end correctness of the pilot.

## Sources inspected

These are clean source revisions inspected for this design, **not a frozen
production stack**. Dependencies, numerical builds, environment locks and the
selected host remain unverified. Re-audit changed sources before collection.

| Repository | Inspected revision | Relevant source |
|---|---|---|
| autolens_profiling | `e4266a1355a860d2274a82722858349cd3521e46` | [Workload and provenance][workload], [builder], [worker] |
| PyAutoLens | `794c94ac0bf891ad13b7444a96b23e97d4cc3ab4` | [AnalysisImaging][analysis], [position penalty][positions], [FitImaging][fit] |
| PyAutoGalaxy | `7e91650cdba94c159987f291f1fbd851727285c5` | [MGE model construction][mge], [Basis], [operated columns][columns], [Gaussian] |
| PyAutoArray | `ccddfba6744d612092ef3723361689b09bf65648` | [Figure of merit][fom], [inversion], [dense imaging][dense], [solve utilities][solve], [PDIP wrapper][pdip], [Settings], [packaged defaults][defaults] |
| PyAutoFit | `911fe05e1c6b570df51789f0ab5b2cbf54c1f43a` | Model/prior-median dependency used by the workload; no external prior term is called by its objective |
| PyAutoNerves | `7f3055b7f3d1967f372bdb41ee732135bca0e68d` | [Test-mode environment][test-mode]; configuration values are read through `conf.instance` |
| autolens_workspace_test | `04c01cee46e61141c32e5807c02e09a04ea5b684` | [Existing MGE NumPy/JAX regression][regression] |

## Exact calculation to reproduce

`current.build_workload` loads three existing FITS files, prepares the mask and
oversampling, builds the model and calls `current.from_dataset`. The latter
constructs an instance at the physical prior medians, registers the model as a
JAX pytree and returns `AnalysisImaging.log_likelihood_function(instance)`.
This is not the retired flat-vector posterior or a call through `Fitness`.

`AnalysisImaging` returns `fit.figure_of_merit - positions_penalty`. In this
builder no positions likelihood is supplied, so the penalty is zero. The fit
contains an inversion and therefore follows `FitDataset.log_evidence`. However,
the two MGE `Basis` objects have no regularization: `regularization_term`,
`log_det_curvature_reg_matrix_term` and `log_det_regularization_matrix_term` each
return zero when no `AbstractRegularization` is present. For this precise case,
the reported scalar reduces to the normalized Gaussian data log likelihood.
Assert these conditions in a future witness; do not assume them for other models.

Let `d` and `sigma` be the ordered unmasked data and effective positive noise,
`A` the PSF-operated unit-intensity columns in actual coefficient order, and
`b` the non-linear light/background contribution subtracted before inversion.
For the intended pure linear-light model `b` should be zero; capture and verify
the actual value. Define

```text
B = A / sigma[:, None]           y = (d - b) / sigma
q = B.T @ y                     Q = B.T @ B + lambda * I
c >= 0: minimize 0.5*c.T@Q@c - q.T@c
m = b + A@c                     r = (d - m) / sigma
chi2 = sum(r**2)                noise_norm = sum(log(2*pi*sigma**2))
reported_value = -0.5 * (chi2 + noise_norm)
```

The dense mapping path adds the configured
`no_regularization_add_to_curvature_diag_value` to all unregularized columns;
its packaged default is `1e-3`. Both bases are unregularized, so the displayed
identity term applies to all intended 40 columns. **This stabilization changes
the reconstruction but is not an extra `lambda*||c||²` likelihood penalty or
log-determinant term.** A plain least-squares solve, an unconstrained solve, or
an NNLS solve omitting that diagonal would compare different calculations.

The packaged positive-only solver is enabled. The MGE JAX path uses PDIP even
when the mapper-only `certified` option is requested. With the packaged
`nnls_preconditioning_no_mapper: raw`, the wrapper performs a raw solve followed
by at most ten Jacobi-space polish iterations; it retains the raw iterate if
polishing is unsuccessful. A missing explicit raw tolerance invokes
`0.01*n*jaxnnls.pdip.EPSILON*max(1,max(abs(q)))`; the wrapper defaults to 50 raw
iterations. These are solver stopping controls, not scientific error bounds.
The raw convergence flag does not certify the accepted polish. Record both
statuses and independently evaluate optimality of the returned coefficients.
NumPy uses the library's `fnnls_cholesky`, with optional memo state.

## Identity and configuration required at execution

The existing producer captures useful hashes but does not resolve this complete
contract. Record actual values and their source; null means unverified.

| Identity | Required capture and checks |
|---|---|
| FITS and preparation | Original byte hashes, provenance (observed/simulated), units, shape and dtype; effective data/noise arrays and their hashes, finite values and strictly positive fitted noise; selected mask bytes, native-to-slim row map, origin and pixel scales. Record any padding and PSF normalization, not just file hashes. |
| Spatial operator | Effective PSF values, orientation, centering, support, boundary/cropping rules and convolution oversampling; image and blurring grids, subpixel coordinates/weights and final binning. Include outside-mask light that convolves into fitted pixels. |
| Model | Full prior/model serialization and ordered physical median values, pytree structure, leaf paths/shapes/dtypes/hashes; resolved Gaussian widths/centres/ellipticities, mass and shear values, redshifts and cosmology. Link each matrix column to its galaxy/basis/profile path and amplitude convention. |
| Intended cell | 0.05 arcsec pixels, 3.5 arcsec circular mask, LP radial oversampling `[4,2,2]` at `[0.3,0.6]`, two 20-Gaussian bases, fixed Isothermal/shear, dense inversion and fp64. Verify constructed counts and classes rather than copying labels. |
| Solver | Resolved positive-only and edge-zero settings; actual retained indices, diagonal increment, requested and selected solver/preconditioning, raw/polish tolerances and iteration limits/status. Capture `nnls_jacobi_preconditioning`, `nnls_target_kappa`, numerical library versions and `EPSILON`; distinguish unused backward/mapper settings from forward controls. |
| Terms and defaults | Effective `Settings`, config search/layer order, hashes of all contributing library/user/project config files and overrides; mixed precision, non-linear contribution, positions penalty, regularization presence and all reported scalar terms. Model serialization alone does not contain these defaults. |
| Test controls | Require production test mode off and small-dataset shortcuts off; record `PYAUTO_TEST_MODE`, `PYAUTO_SMALL_DATASETS` and effective behavior. Reject fallback/sentinel reconstructions. |
| Runtime and pairing | Clean full source SHAs plus source hashes, dependency/build lock, CPU/backend/x64, effective thread/affinity/XLA settings. Bind reference and every compared observation to the same science identity; keep backend-specific reference settings explicit. |

**Known producer limitation:** `mge_model_from` changes `total_gaussians` to two
when `PYAUTO_SMALL_DATASETS=1`. `current.provenance` currently labels the counts
as twenty per basis, and the worker's recorded environment filters omit the
`PYAUTO_*` controls. The serialized model can expose a discrepancy, but the
labels are not an enforced 40-column assertion. A future guard must inspect the
constructed model and record these controls. This audit changes no runner code
and makes no claim that the development fixture used a shortcut.

## Independent reference: recommended construction

1. Start from the frozen FITS bytes and explicit physical parameters, not the
   library's traced grid, operated matrix, coefficients or scalar result.
   Implement the intended pixel/subpixel coordinates, mask, lens mapping,
   Isothermal/shear convention, elliptical unit-intensity Gaussians and binning
   separately. The reference design must explicitly review the [Isothermal],
   [external shear][shear] and [multi-plane tracing][tracing] conventions.
   Inspect [Gaussian] for its eccentric-radius/sigma convention;
   ordinary circular-Gaussian intuition is not enough. Identify any unavoidable
   shared I/O dependency and its checks. Validate the reference with analytic
   small cases before using it to judge the library.
2. Build each convolved column with explicit finite-support spatial convolution,
   including the blurring region and the same PSF normalization/crop. Use
   asymmetric-PSF and boundary fixtures to expose orientation and missing-flux
   errors. The library's NumPy real-space column path shares profile/grid code;
   merely selecting it is not an independent operator construction.
3. Form the augmented least-squares problem from that independent `A`:

   ```text
   C = concatenate_rows(B, sqrt(lambda) * I)
   z = concatenate(y, zeros(number_of_columns))
   c_ref = independent_nnls(C, z)
   ```

   Candidate: [SciPy's `optimize.nnls`][scipy-nnls], an active-set non-negative
   least-squares implementation distinct from the pilot's PDIP wrapper. This
   construction reproduces the stabilized quadratic above without reusing the
   library's curvature assembly or `fnnls`. Pin and audit the selected SciPy
   implementation and explicit iteration limit; do not assume an `atol` keyword
   controls reference accuracy across versions. No package/version is selected
   as the production reference by this recommendation.
4. Compute the image, chi-squared and noise normalization independently from
   `c_ref`. Preserve the augmented residual/objective as a solve diagnostic;
   **do not report it as the data likelihood**. Verify primal feasibility,
   dual feasibility and complementarity for `g=Q@c_ref-q`. Solver success or a
   finite scalar alone is insufficient.
5. Compare against a project adapter exposing the target's ordered matrix,
   coefficients, model image and terms outside measurement intervals. Match its
   scalar to the actual compiled pilot objective at the identical instance.
   Never insert witness construction or diagnostic outputs into the executable
   whose compile/runtime is being timed. New diagnostic execution and its budget
   require the follow-up plan.

A staged matrix-export comparison may localize a solver discrepancy, but its
artifact must say `shared_operator: true` and `end_to_end_qualified: false`.
Until the independent operator and reference validations exist, report
**reference unavailable**, not scientific correctness passed.

## Comparisons and proposed tolerance review

Always retain raw absolute and relative errors, maxima and their indices, RMS
errors, and both compared values. Never use relative error in the total log
likelihood as the sole check: noise normalization can hide a chi-squared error.
Compare chi-squared, noise normalization, zero terms, the final scalar, model
image and coefficients separately. Test the expected zero terms structurally
as well as numerically.

The following are **candidate screening budgets for human review**, not accepted
thresholds or measured performance. They make the next experiment falsifiable.
All absolute AND relative criteria in a row must hold; they are not the additive
`atol + rtol*abs(reference)` rule. Freeze exact definitions before collection.

| Quantity | Candidate budget / rule | Rationale and limitation |
|---|---|---|
| Reported scalar | `abs(delta) <= 1e-3`; also `abs(delta)/max(1,abs(reference)) <= 1e-8` | Small absolute log-likelihood perturbation plus a scale check; no blanket claim of scientific sufficiency. Independently compare the terms below to prevent cancellation. |
| Chi-squared and noise normalization | Each: absolute `<= 1e-3` and relative `<= 1e-8`, denominator `max(1,abs(reference))` | Makes their contributions separately accountable. Floor 1 is in dimensionless log-likelihood/chi-squared units, fixed before results. |
| Reconstructed image | Noise-whitened RMS error `<= 1e-5`, maximum absolute whitened error `<= 1e-3`; also relative image L2 error `<= 1e-8` using the whitened reference image norm with floor 1 | Controls spatially localized errors as well as the scalar; does not follow automatically from likelihood agreement. |
| Independent reference convergence | Reference-induced changes in each comparison metric must consume at most 1% of its proposed budget | Establish with analytic fixtures and a second audited solve/refinement, plus KKT residuals; a library return flag is insufficient. Such validation is separate, budgeted work. |
| Coefficients | Record componentwise absolute/relative errors and the full vector. Propose absolute floors in the documented intensity units and a relative budget only after conditioning and active-set sensitivity are assessed | Narrow/overlapping Gaussian columns make an arbitrary universal intensity tolerance misleading. No coefficient acceptance threshold is selected here; this remains a blocker. |

For primal/dual/complementarity diagnostics, record raw norms and dimensionless
versions with explicit positive scales: `s_c=max(c_floor,||c||inf)` and
`s_g=max(g_floor,||Q||inf*s_c,||q||inf)`. Report
`||min(c,0)||inf/s_c`, `||min(g,0)||inf/s_g`, and
`||c*g||inf/(s_c*s_g)`. The floors carry coefficient/gradient units and need
review; do not substitute machine epsilon as a universal scientific tolerance.
Record KKT diagnostics for both reference and target with the same definitions.

Report singular values/conditioning of the independent weighted matrix and its
augmented form, the actual diagonal increment, and sensitivity of coefficients
and reconstructed image. A positive diagonal on every column makes this exact
quadratic strictly convex: near-degenerate unregularized columns do not imply
multiple exact stabilized minimizers. Numerical conditioning can still make
coefficient agreement difficult. If the diagonal is absent, changes, or the
reference is rank deficient, revisit uniqueness and the reference contract;
do not silently drop coefficient comparison because the images agree.

The candidate budgets above are deliberately not inherited from the existing
regression's `rtol=1e-4`. That script varies mass/shear and passes through
`Fitness` for one check; its additional NumPy/JAX `FitImaging` check still shares
source. The pilot's protocol tests use fake workers/clocks and validate
orchestration, not these mathematical claims. Earlier tiny synthetic worker
runs provide development execution evidence only.

## Witness artifact proposal and collection blockers

Propose a separate versioned `hst-mge-correctness-witness` artifact, not a
renamed baseline observation. Preserve reference source/algorithm/dependency
identities, science-identity digest, inputs and coefficient/row maps, operator,
`Q/q`, reconstruction/image/term arrays, per-array shape/dtype/hash, raw metrics,
KKT/conditioning results, reference-validation receipts, tolerance-document hash
and human decision reference. Hash artifact bytes and link them from a manifest;
retain failures and missing evidence. Avoid publishing machine-specific paths.

Separate `integrity_passed`, `reference_validation_passed`,
`comparison_passed` (null without approved bounds), and
`scientifically_accepted` (false until explicit human review). Bind cold, seed,
warm and runtime scalar observations to the witness; one matching scalar does
not establish every worker's reconstruction. The current worker stores only
the scalar, so any required per-worker reconstruction evidence is an explicit
future adapter decision, never an inferred property of this design.

| Decision / deliverable | Current state and next owner |
|---|---|
| Raw measurement tooling | Shipped in [project #416]; scientific qualification remains absent. |
| Correctness harness | Proposed next phase in `autolens_workspace_test`: independent operator/augmented solve, analytic and deliberately faulty synthetic fixtures, then a project adapter. No new library implementation in this phase. |
| Candidate thresholds | Require scientific review and pre-production calibration, especially coefficient units/conditioning and reference KKT accuracy. This document does not accept them. |
| Production dataset/model | Unselected bytes/provenance, full resolved configuration and actual counts. Observational inputs require image/contaminant and mask review. |
| Stack and host | Freeze exact clean sources, numerical builds, reproducible lock, named CPU host/affinity/threads and memory only after selection. Source pins above are audit evidence. |
| Load and resources | Define a background-load method excluding campaign work; raw host load is insufficient. Keep Pulse's proposed ceiling separate from authorization; allow for reference preparation/validation and diagnostic runs explicitly. Do not assume they fit its reserved one-process/30-minute correctness allowance. |
| Collection / acceptance | Separate human collection window/resource authorization, followed by scientific result review. No production profiling, baseline freeze/promotion or inference is authorized here. |

Suggested next bounded implementation: immutable fixture/identity exporter and
independent analytic operator plus augmented NNLS on tiny synthetic arrays,
limited to this same fixed model family. Include nonzero and active-boundary
coefficients, asymmetric PSF, masked-edge light, ill-conditioned columns and
intentional wrong-diagonal/normalization/order mutations. Require the witness
to reject those faults and preserve failures. Plan exact process/time/memory
caps before executing; assess full HST feasibility afterwards. Route any
discovered library defect through the bug workflow and keep collection blocked.

[project #417]: https://github.com/PyAutoLabs/autolens_profiling/issues/417
[project #416]: https://github.com/PyAutoLabs/autolens_profiling/pull/416
[Pulse pilot contract]: https://github.com/PyAutoLabs/PyAutoPulse/blob/35b52bb7b9c918249de48d0c85236b5296f112d0/tasks/hst_mge_cpu_pilot.md
[workload]: https://github.com/PyAutoLabs/autolens_profiling/blob/e4266a1355a860d2274a82722858349cd3521e46/scripts/misc/jax_compile/current.py
[builder]: https://github.com/PyAutoLabs/autolens_profiling/blob/e4266a1355a860d2274a82722858349cd3521e46/scripts/misc/cell_builders/imaging_mge.py
[worker]: https://github.com/PyAutoLabs/autolens_profiling/blob/e4266a1355a860d2274a82722858349cd3521e46/scripts/misc/jax_compile/pilot_worker.py
[analysis]: https://github.com/PyAutoLabs/PyAutoLens/blob/794c94ac0bf891ad13b7444a96b23e97d4cc3ab4/autolens/imaging/model/analysis.py
[positions]: https://github.com/PyAutoLabs/PyAutoLens/blob/794c94ac0bf891ad13b7444a96b23e97d4cc3ab4/autolens/analysis/analysis/lens.py
[fit]: https://github.com/PyAutoLabs/PyAutoLens/blob/794c94ac0bf891ad13b7444a96b23e97d4cc3ab4/autolens/imaging/fit_imaging.py
[mge]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/analysis/model_util.py
[Basis]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/profiles/basis.py
[columns]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/profiles/light/linear/abstract.py
[Gaussian]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/profiles/light/standard/gaussian.py
[Isothermal]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/profiles/mass/total/isothermal.py
[shear]: https://github.com/PyAutoLabs/PyAutoGalaxy/blob/7e91650cdba94c159987f291f1fbd851727285c5/autogalaxy/profiles/mass/sheets/external_shear.py
[tracing]: https://github.com/PyAutoLabs/PyAutoLens/blob/794c94ac0bf891ad13b7444a96b23e97d4cc3ab4/autolens/lens/tracer_util.py
[fom]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/fit/fit_dataset.py
[inversion]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/inversion/inversion/abstract.py
[dense]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/inversion/inversion/imaging/mapping.py
[solve]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/inversion/inversion/inversion_util.py
[pdip]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/util/jax_nnls.py
[Settings]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/settings.py
[defaults]: https://github.com/PyAutoLabs/PyAutoArray/blob/ccddfba6744d612092ef3723361689b09bf65648/autoarray/config/general.yaml
[test-mode]: https://github.com/PyAutoLabs/PyAutoNerves/blob/7f3055b7f3d1967f372bdb41ee732135bca0e68d/autonerves/test_mode.py
[regression]: https://github.com/PyAutoLabs/autolens_workspace_test/blob/04c01cee46e61141c32e5807c02e09a04ea5b684/scripts/imaging/jax_likelihood/mge.py
[scipy-nnls]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
