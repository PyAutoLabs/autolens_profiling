"""Certified active-set NNLS kernels, and the fixed-lens-light source-only system.

Why this module exists
----------------------

The A100 pixelized baseline (``results/notes/a100_pixelized_baseline_2026_09.md``)
put the ``Regularized reconstruction`` row at 37 ms — 61-71 % of one pixelized
imaging likelihood call — and the reconstruction row-split
(``results/notes/reconstruction_row_split_2026_09.md``) resolved it into
21-22 PDIP iterations at ~1.7 ms each. The matrix-free sweep then found no
CG/SLQ crossover at any mesh size, and named the reason: ``cond(F + λH) ≈ 4e10``,
set by the 60 **unregularised linear-MGE columns** that share the solve with the
source pixels.

Freezing the lens light after SLaM ``light[1]`` — converting the linear MGE to
regular light profiles at the solved intensities and subtracting it — removes
that block from the linear system entirely. What is left is a *source-only*
inversion: one ``Mapper``, no linear func list, ``cond(F + λH) ≈ 1e6-1e7``. On
that system a **certified active-set** positive solve is a real option, because
the active set is small and the unconstrained solve already nearly identifies
it. This module is the kernel half of that measurement:

- :func:`fixed_light_system_from` builds the source-only system from an existing
  fit with *linear* lens light, using the library's own linear-to-regular
  conversion;
- :func:`active_set_certified` is the numpy/scipy fp64 **reference** scheme
  (seed, restricted Cholesky, KKT certificate, free-all / free-one release);
- :func:`active_set_masked_jax` is its jit-able, **static-shape** twin — the
  form an A100 cell (and, eventually, a library solver) can actually run;
- :func:`active_set_evidence_error` scores an iterate in nats against the PDIP
  solution, through the same ``reconstruction_steps.log_evidence_terms`` every
  profiling cell uses.

The QP
------

The problem is exactly the one the library hands to ``jaxnnls``. With

    ``D_s = diag(1 / sqrt(diag(F + λH)))``,
    ``Q = D_s (F + λH) D_s``,  ``q = D_s D``

(:func:`reconstruction_steps.jacobi_scaled`, which reproduces
``autoarray…inversion_util.reconstruction_positive_only_from``), minimise

    ``½ xᵀ Q x − qᵀ x``   subject to   ``x ≥ 0``,

and the physical reconstruction is ``x · D_s``. ``D_s`` is diagonal and strictly
positive, so the **active set is the same index set in either space** — every
set-valued quantity here is reported in the scaled space and is valid unscaled.

The scheme
----------

Pass 0 is the unconstrained Cholesky solve of the full system (1 factorisation);
the fixed set is seeded ``Z := {i : x_unc,i < 0}`` (plus ``fixed0``, below).
Each subsequent pass is one Cholesky of the free block ``Q[F, F]``, giving
``x_F = Q[F,F]⁻¹ q_F`` and ``x_Z = 0``, then the KKT test on ``g = Q x − q``:

- primal: ``x_i ≥ −τ_x`` for every free ``i``   (``τ_x = τ_rel · max|x|``)
- dual:   ``g_i ≥ −τ_g`` for every fixed ``i``  (``τ_g = τ_rel · max|q|``)
- complementarity holds by construction (``x_Z = 0`` exactly; ``max|g_F|`` is
  reported per pass and sits at solver round-off).

If both hold the iterate is **certified**: it is the exact KKT point of the QP,
not an approximation to it. Otherwise every free index that came out negative
moves into ``Z``, and the dual violators are released by one of two policies —
``"free_all"`` releases every fixed index with ``g_i < −τ_g``, ``"free_one"``
only the single most negative. ``τ_rel = 1e-9`` throughout.

Edge zeroing
------------

``PyAutoArray``'s positive-only solver does not solve the full system: when
``use_positive_only_solver`` **and** ``use_edge_zeroed_pixels`` are set and the
inversion has a ``Mapper``, ``Inversion.solve_ids_to_keep`` subsets the system
to the kept indices and scatters zeros back into the rest
(``autoarray/inversion/inversion/abstract.py``). On a rectangular mesh that is
the border ring — 152 of 1521 pixels on the profiling fiducial. Those indices
are *not* free variables of the QP; they are held at zero by the library.

:func:`fixed_light_system_from` therefore records ``edge_zero_mask`` straight
off ``Inversion.solve_ids_to_keep`` (the library property, called — not
reimplemented), and both schemes take it as ``fixed0``: seeded into ``Z`` at
pass 0 and **never released**, so the certified solution is the library's
edge-zeroed NNLS solution rather than the pure full-system one. Running with
``fixed0=None`` measures the pure problem; the difference is ~215 nats of
log-evidence on the rectangular fiducial, which is why both are reported.

Dense only
----------

Every system built here is a **dense** ``InversionImagingMapping``.
``sparse_steps.sparse_context_from`` raises unless the inversion is exactly one
``Mapper`` plus one ``AbstractLinearObjFuncList``, and the source-only system
has no func list at all; worse, ``InversionImagingSparse.psf_weighted_data``
reads ``dataset.sparse_operator.weight_map``, which is baked from the
*original* image at ``apply_sparse_operator`` time, so a subtracted dataset's
sparse data vector would silently ignore the subtraction. The dense and sparse
totals agree to <10 % on the matrix-free sweep, and every quantity this module
measures (``F + λH``, ``D``, the solve, the two log-dets) is identical code on
both legs — so nothing is lost by staying dense, and a wrong number is avoided.

Library provenance (verified 2026-09-12, autolens 2026.8.17.1)
--------------------------------------------------------------

- ``autolens/imaging/fit_imaging.py`` ``FitImaging.tracer_linear_light_profiles_to_light_profiles``
  — the linear-to-regular conversion at the solved intensities.
- ``autoarray/inversion/inversion/abstract.py`` ``Inversion.solve_ids_to_keep``
  / ``zeroed_ids_to_keep`` — the edge-zero index set, called directly.
- ``autoarray/inversion/inversion/abstract.py`` ``Inversion.operated_mapping_matrix``
  — the dense PSF-operated mapping matrix; ``blurred_image + Λ x`` reproduces
  ``FitImaging.model_data`` to 1e-15 and ``log_evidence_terms`` on it reproduces
  ``FitImaging.log_evidence`` exactly (pinned by the tests).
- ``autoarray/util/jax_nnls.py`` ``solve_nnls`` — the PDIP reference, reached
  through :func:`reconstruction_steps.nnls_pdip`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax import lax
from jax.scipy.linalg import cho_solve

from likelihood_breakdown import reconstruction_steps

__all__ = [
    "ActiveSetResult",
    "FixedLightSystem",
    "active_set_certified",
    "active_set_evidence_error",
    "active_set_masked_jax",
    "fixed_light_system_from",
    "linear_system_from",
    "unconstrained_solve",
]

TAU_REL_DEFAULT = 1.0e-9


# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------


@dataclass
class FixedLightSystem:
    """One linear system, plus everything needed to score an iterate on it.

    Built by :func:`linear_system_from` (any fit) or :func:`fixed_light_system_from`
    (the source-only, lens-light-fixed system). All arrays are numpy fp64 in the
    **unscaled** space except ``Q``/``q``, which are the Jacobi-scaled QP the
    library's positive-only solver actually receives.
    """

    name: str
    fit: Any
    inversion: Any
    dataset: Any

    curvature_reg_matrix: np.ndarray  # F + λH, (n, n)
    data_vector: np.ndarray  # D, (n,)
    mapper_indices: np.ndarray  # global indices of the mapper block
    reg_reduced: np.ndarray  # rank-stripped H
    curv_reg_reduced: np.ndarray  # rank-stripped F + λH

    Q: np.ndarray  # Jacobi-scaled curvature, unit diagonal
    q: np.ndarray  # Jacobi-scaled data vector
    d_scale: np.ndarray  # x_physical = x_scaled * d_scale

    edge_zero_mask: np.ndarray  # bool (n,), True where the library holds x at 0
    ids_to_keep: np.ndarray | None  # the library's kept-index array, or None

    n_params: int
    n_mapper: int
    n_funcs: int

    blurred_image: np.ndarray  # regular light profiles, PSF-convolved
    operated_mapping_matrix: np.ndarray  # (M, n)

    # Only populated by ``fixed_light_system_from``.
    source_only_tracer: Any = None
    subtracted_light_image: np.ndarray | None = None
    subtracted_light_flux: float | None = None

    _pdip: tuple | None = field(default=None, repr=False, compare=False)

    # -- scoring ----------------------------------------------------------

    def model_data_from(self, x) -> np.ndarray:
        """``blurred_image + Λ x`` — the dense model image of a parameter vector.

        Reproduces ``FitImaging.model_data`` for the library's own
        reconstruction (pinned to 1e-15 by ``test_active_set_steps.py``).
        """
        x = np.asarray(x, dtype=float)
        return self.blurred_image + self.operated_mapping_matrix @ x

    def log_evidence_terms(self, x) -> dict:
        """Every evidence term of an iterate, via ``reconstruction_steps``."""
        x = np.asarray(x, dtype=float)
        return reconstruction_steps.log_evidence_terms(
            jnp.asarray(np.asarray(self.dataset.data.array), dtype=jnp.float64),
            jnp.asarray(np.asarray(self.dataset.noise_map.array), dtype=jnp.float64),
            jnp.asarray(self.model_data_from(x), dtype=jnp.float64),
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(self.mapper_indices),
            jnp.asarray(self.reg_reduced, dtype=jnp.float64),
            jnp.asarray(self.curv_reg_reduced, dtype=jnp.float64),
        )

    def log_evidence(self, x) -> float:
        return float(self.log_evidence_terms(x)["log_evidence"])

    # -- references -------------------------------------------------------

    def pdip(self) -> tuple[np.ndarray, bool, int]:
        """The library's PDIP NNLS solution of this system (cached).

        ``reconstruction_steps.nnls_pdip`` calls the ``solve_nnls`` *driver*, so
        the iteration count and the convergence flag survive; the returned
        vector is the physical (unscaled) reconstruction.
        """
        if self._pdip is None:
            x_pc, converged, iters = jax.jit(reconstruction_steps.nnls_pdip)(
                jnp.asarray(self.Q, dtype=jnp.float64),
                jnp.asarray(self.q, dtype=jnp.float64),
            )
            x_pc = np.asarray(jax.block_until_ready(x_pc), dtype=float)
            self._pdip = (x_pc * self.d_scale, bool(converged), int(iters))
        return self._pdip

    def unconstrained(self, *, use_edge_zero: bool = True) -> np.ndarray:
        """The pass-0 unconstrained Cholesky solve, physical space."""
        fixed0 = self.edge_zero_mask if use_edge_zero else None
        return unconstrained_solve(self.Q, self.q, fixed0=fixed0) * self.d_scale


def _light_profile_cls():
    from autogalaxy.profiles.light.abstract import LightProfile

    return LightProfile


def _pixelization_cls():
    from autoarray.inversion.pixelization import Pixelization

    return Pixelization


def _profile_attrs(galaxy) -> dict:
    """The galaxy's profile attributes, by the name the ``Galaxy`` holds them under.

    ``Galaxy`` keeps its profiles as plain instance attributes (``bulge``,
    ``disk``, ``mass``, ``shear``, ``pixelization``, …) alongside bookkeeping
    (``id``, ``_label``, ``redshift``), so the attribute dict is the only way to
    rebuild a galaxy with a subset of its profiles under the *same* names — and
    the names matter, because ``af.Model`` paths and adapt-image dictionaries
    are keyed on them.
    """
    skip = {"id", "redshift"}
    return {
        key: value
        for key, value in vars(galaxy).items()
        if not key.startswith("_") and key not in skip
    }


def _light_only_galaxies(tracer):
    """Light-profile-only copies of every galaxy that carries light and no mesh.

    A galaxy whose light is a ``Basis`` is kept (``Basis`` subclasses both
    ``LightProfile`` and ``MassProfile``; the light branch wins, which is what a
    fixed *lens light* subtraction wants). A galaxy with a ``Pixelization`` is
    skipped outright — its emission is not a profile and cannot be subtracted.
    """
    import autolens as al

    light_cls = _light_profile_cls()
    pix_cls = _pixelization_cls()

    out = []
    for galaxy in tracer.galaxies:
        attrs = _profile_attrs(galaxy)
        if any(isinstance(v, pix_cls) for v in attrs.values()):
            continue
        light = {k: v for k, v in attrs.items() if isinstance(v, light_cls)}
        if light:
            out.append(al.Galaxy(redshift=float(galaxy.redshift), **light))
    return out


def _light_stripped_tracer(tracer):
    """The same tracer with every *light* profile removed (mass and mesh kept).

    A galaxy with no light to strip is passed through **by identity**, not
    rebuilt. ``AdaptImages`` keys its ``galaxy_image_dict`` on the galaxy object,
    so a rebuilt source galaxy — however identical — would miss the lookup and
    the adapt-image mesh would be handed ``adapt_data=None``.
    """
    import autolens as al

    light_cls = _light_profile_cls()

    galaxies = []
    for galaxy in tracer.galaxies:
        attrs = _profile_attrs(galaxy)
        stripped = {k: v for k, v in attrs.items() if not isinstance(v, light_cls)}
        if len(stripped) == len(attrs):
            galaxies.append(galaxy)
        else:
            galaxies.append(al.Galaxy(redshift=float(galaxy.redshift), **stripped))
    return al.Tracer(galaxies=galaxies)


def linear_system_from(fit, dataset=None, *, name: str = "system") -> FixedLightSystem:
    """Read one fit's linear system (and its scoring arrays) off the inversion.

    Everything is pulled from the library objects rather than rebuilt: ``F + λH``
    and ``D`` off the inversion, the reduced blocks off its rank-stripped
    properties, the edge-zero set off ``solve_ids_to_keep``.
    """
    from autoarray.inversion.mappers.abstract import Mapper

    dataset = fit.dataset if dataset is None else dataset
    inversion = fit.inversion

    n = int(inversion.total_params)
    mapper = inversion.cls_list_from(cls=Mapper)[0]
    n_mapper = int(mapper.params)

    curvature_reg_matrix = np.asarray(inversion.curvature_reg_matrix, dtype=float)
    data_vector = np.asarray(inversion.data_vector, dtype=float)

    ids_to_keep = inversion.solve_ids_to_keep
    edge_zero_mask = np.zeros(n, dtype=bool)
    if ids_to_keep is not None:
        ids_to_keep = np.asarray(ids_to_keep, dtype=int)
        edge_zero_mask[:] = True
        edge_zero_mask[ids_to_keep] = False

    Q, q, d_scale = reconstruction_steps.jacobi_scaled(
        jnp.asarray(curvature_reg_matrix, dtype=jnp.float64),
        jnp.asarray(data_vector, dtype=jnp.float64),
    )

    return FixedLightSystem(
        name=name,
        fit=fit,
        inversion=inversion,
        dataset=dataset,
        curvature_reg_matrix=curvature_reg_matrix,
        data_vector=data_vector,
        mapper_indices=np.asarray(inversion.mapper_indices),
        reg_reduced=np.asarray(inversion.regularization_matrix_reduced, dtype=float),
        curv_reg_reduced=np.asarray(inversion.curvature_reg_matrix_reduced, dtype=float),
        Q=np.asarray(Q, dtype=float),
        q=np.asarray(q, dtype=float),
        d_scale=np.asarray(d_scale, dtype=float),
        edge_zero_mask=edge_zero_mask,
        ids_to_keep=ids_to_keep,
        n_params=n,
        n_mapper=n_mapper,
        n_funcs=n - n_mapper,
        blurred_image=np.asarray(fit.blurred_image.array, dtype=float),
        operated_mapping_matrix=np.asarray(inversion.operated_mapping_matrix, dtype=float),
    )


def fixed_light_system_from(
    fit,
    dataset=None,
    *,
    adapt_images=None,
    settings=None,
    name: str = "fixed_light",
) -> FixedLightSystem:
    """The source-only system left once this fit's *linear* lens light is fixed.

    ``fit`` is a fit whose lens light is a **linear** light profile (the MGE-60
    basis the profiling cells build). The steps, in the order the library
    performs them:

    1. ``fit.tracer_linear_light_profiles_to_light_profiles`` converts every
       linear light profile to a regular one at its solved intensity — the same
       call SLaM makes after ``light[1]``;
    2. the PSF-convolved image of those regular profiles is subtracted **from
       the dataset**, and a fresh ``Imaging`` is built on the residual with the
       same noise map, PSF, mask and over-sampling;
    3. the tracer is rebuilt with every light profile stripped, leaving mass,
       shear and the source's pixelization;
    4. a dense ``FitImaging`` on (2) with (3) is the source-only inversion.

    The subtraction is applied to the *dataset* rather than carried as a regular
    light profile on the tracer. Both are algebraically the same on the dense
    path, but the w-tilde path is not: ``InversionImagingSparse`` reads the
    operator's ``weight_map``, baked from the original image, so a tracer-side
    subtraction never reaches the sparse data vector. Subtracting on the dataset
    is exact on both legs (see the module docstring).
    """
    import autolens as al

    dataset = fit.dataset if dataset is None else dataset
    adapt_images = getattr(fit, "adapt_images", None) if adapt_images is None else adapt_images
    settings = fit.settings if settings is None else settings

    converted = fit.tracer_linear_light_profiles_to_light_profiles
    light_galaxies = _light_only_galaxies(converted)
    if not light_galaxies:
        raise ValueError(
            f"{name}: the converted tracer carries no light profiles to fix — "
            "fixed_light_system_from expects a fit with linear lens light."
        )

    blurred = al.Galaxies(galaxies=light_galaxies).blurred_image_2d_from(
        grid=dataset.grids.lp,
        psf=dataset.psf,
        blurring_grid=dataset.grids.blurring,
        xp=np,
    )

    subtracted = dataset.__class__(
        data=dataset.data - blurred,
        noise_map=dataset.noise_map,
        psf=dataset.psf,
        over_sample_size_lp=dataset.over_sample_size_lp,
        over_sample_size_pixelization=dataset.over_sample_size_pixelization,
    )

    source_only_tracer = _light_stripped_tracer(fit.tracer)

    fit_source_only = al.FitImaging(
        dataset=subtracted,
        tracer=source_only_tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )

    system = linear_system_from(fit_source_only, subtracted, name=name)
    system.source_only_tracer = source_only_tracer
    system.subtracted_light_image = np.asarray(blurred.array, dtype=float)
    system.subtracted_light_flux = float(np.sum(system.subtracted_light_image))
    return system


# ---------------------------------------------------------------------------
# The numpy reference scheme
# ---------------------------------------------------------------------------


def _chol_solve_np(A, b):
    """One dense Cholesky factorisation plus two triangular solves, fp64."""
    c, low = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
    return scipy.linalg.cho_solve((c, low), b, check_finite=False)


def unconstrained_solve(Q, q, *, fixed0=None) -> np.ndarray:
    """The pass-0 solve: unconstrained on the free block, zero on ``fixed0``.

    One factorisation. With ``fixed0=None`` this is the full-system solve; with
    the edge-zero mask it is the system the library's positive-only solver
    actually receives.
    """
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    x = np.zeros(q.shape[0], dtype=float)
    if fixed0 is None:
        return _chol_solve_np(Q, q)
    free = ~np.asarray(fixed0, dtype=bool)
    if free.any():
        idx = np.where(free)[0]
        x[idx] = _chol_solve_np(Q[np.ix_(idx, idx)], q[idx])
    return x


@dataclass
class ActiveSetResult:
    """One run of :func:`active_set_certified`."""

    policy: str
    certified: bool
    cycle_detected: bool
    passes_to_certification: int | None
    passes_run: int
    n_factorisations: int
    factorisations_after_pass: list[int]
    history: list[dict]
    iterates: list[np.ndarray]
    fixed_set: np.ndarray
    tau_rel: float
    max_passes: int
    warm_start: bool
    log_evidence_per_pass: list[float] | None = None

    @property
    def x(self) -> np.ndarray:
        """The final iterate."""
        return self.iterates[-1]

    def as_dict(self) -> dict:
        """JSON-safe view: everything but the iterates and the mask itself."""
        return {
            "policy": self.policy,
            "certified": self.certified,
            "cycle_detected": self.cycle_detected,
            "passes_to_certification": self.passes_to_certification,
            "passes_run": self.passes_run,
            "n_factorisations": self.n_factorisations,
            "factorisations_after_pass": list(self.factorisations_after_pass),
            "history": self.history,
            "fixed_set_size": int(self.fixed_set.sum()),
            "tau_rel": self.tau_rel,
            "max_passes": self.max_passes,
            "warm_start": self.warm_start,
            "log_evidence_per_pass": self.log_evidence_per_pass,
        }


def active_set_certified(
    Q,
    q,
    x_unc,
    *,
    policy: str = "free_all",
    max_passes: int = 40,
    tau_rel: float = TAU_REL_DEFAULT,
    fixed0=None,
    initial_fixed=None,
    log_evidence_fn: Callable[[np.ndarray], float] | None = None,
) -> ActiveSetResult:
    """Certified active-set NNLS on the scaled system — numpy/scipy fp64 reference.

    Parameters
    ----------
    Q, q
        The Jacobi-scaled QP (``FixedLightSystem.Q`` / ``.q``). Everything below
        is in that space; multiply an iterate by ``d_scale`` for the physical
        reconstruction.
    x_unc
        The pass-0 unconstrained solve (:func:`unconstrained_solve`), **with the
        same** ``fixed0`` **already held at zero**. Pass ``None`` for a warm
        start from ``fixed0`` alone, which skips pass 0 and its factorisation.
    policy
        ``"free_all"`` releases every fixed index whose gradient is below
        ``−τ_g``; ``"free_one"`` releases only the most negative one.
    fixed0
        Boolean mask of indices the library holds at zero (the edge-zero set).
        They are seeded into the fixed set and **never released**, so the
        certificate is a certificate for the library's edge-zeroed problem. With
        ``None``, the pure full-system QP is solved.
    initial_fixed
        A *guess* at the active set — a warm start from a neighbouring
        parameter point's answer. Seeded into the fixed set like ``fixed0`` but
        **releasable**, which is the whole difference: a warm start that could
        not be released would certify the restricted problem it was handed
        rather than the QP, and a wrong guess would come back wearing a
        certificate (measured: 0.1 relative error and −8 nats on the ±1 % draws).
    log_evidence_fn
        Optional hook called on each iterate (pass 0 included) — typically
        ``lambda x: system.log_evidence(x * system.d_scale)``. Recorded in
        ``log_evidence_per_pass``; it is *not* used to steer the scheme.

    Returns
    -------
    :class:`ActiveSetResult` — per-pass iterates, ``|Z|``, primal/dual violation
    counts, the factorisation count and the certification flag.
    """
    if policy not in ("free_all", "free_one"):
        raise ValueError(f"unknown policy {policy!r} (expected 'free_all' or 'free_one')")

    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    n = Q.shape[0]

    permanent = np.zeros(n, dtype=bool) if fixed0 is None else np.asarray(fixed0, dtype=bool).copy()

    seed = (
        np.zeros(n, dtype=bool) if initial_fixed is None else np.asarray(initial_fixed, dtype=bool)
    )

    n_fact = 0
    if x_unc is None:
        if initial_fixed is None and fixed0 is None:
            raise ValueError(
                "a warm start (x_unc=None) needs an initial_fixed or fixed0 mask to start from"
            )
        x = np.zeros(n, dtype=float)
        Z = permanent | seed
    else:
        x = np.asarray(x_unc, dtype=float).copy()
        x[permanent] = 0.0
        n_fact += 1
        # Seed exactly as the scratch probe does: the strict negatives of the
        # unconstrained solve, no tolerance band, plus the permanent set.
        Z = permanent | seed | (x < 0.0)

    iterates = [x.copy()]
    fact_at_pass = [n_fact]
    evidence = [log_evidence_fn(x)] if log_evidence_fn is not None else None

    seen = {Z.tobytes()}
    history: list[dict] = []
    certified = False
    cycled = False

    for p in range(1, max_passes + 1):
        free = ~Z
        x = np.zeros(n, dtype=float)
        if free.any():
            idx = np.where(free)[0]
            x[idx] = _chol_solve_np(Q[np.ix_(idx, idx)], q[idx])
            n_fact += 1
        iterates.append(x.copy())
        fact_at_pass.append(n_fact)
        if evidence is not None:
            evidence.append(log_evidence_fn(x))

        g = Q @ x - q
        tau_x = tau_rel * max(float(np.max(np.abs(x))), 1e-300)
        tau_g = tau_rel * max(float(np.max(np.abs(q))), 1e-300)

        freeable = Z & ~permanent
        primal_violations = free & (x < -tau_x)
        dual_violations = freeable & (g < -tau_g)

        history.append(
            {
                "pass": p,
                "n_fixed": int(Z.sum()),
                "n_free": int(free.sum()),
                "n_primal_violations": int(primal_violations.sum()),
                "n_dual_violations": int(dual_violations.sum()),
                "min_x_free": float(x[free].min()) if free.any() else 0.0,
                "min_g_fixed": float(g[freeable].min()) if freeable.any() else 0.0,
                "max_abs_g_free": float(np.max(np.abs(g[free]))) if free.any() else 0.0,
                "tau_x": tau_x,
                "tau_g": tau_g,
                "n_factorisations": n_fact,
            }
        )

        if not primal_violations.any() and not dual_violations.any():
            certified = True
            break

        Z_new = Z | primal_violations
        if dual_violations.any():
            if policy == "free_all":
                Z_new = Z_new & ~dual_violations
            else:  # free_one
                i = int(np.argmin(np.where(freeable, g, np.inf)))
                Z_new = Z_new.copy()
                Z_new[i] = False

        key = Z_new.tobytes()
        if key in seen:
            cycled = True
        seen.add(key)
        Z = Z_new

    return ActiveSetResult(
        policy=policy,
        certified=certified,
        cycle_detected=cycled,
        passes_to_certification=(len(history) if certified else None),
        passes_run=len(history),
        n_factorisations=n_fact,
        factorisations_after_pass=fact_at_pass,
        history=history,
        iterates=iterates,
        fixed_set=Z,
        tau_rel=tau_rel,
        max_passes=max_passes,
        warm_start=x_unc is None,
        log_evidence_per_pass=evidence,
    )


# ---------------------------------------------------------------------------
# The jit-able, static-shape twin
# ---------------------------------------------------------------------------


def _masked_solve(Q, q, Z):
    """The restricted solve, at **full size**: ``Q`` with ``Z``'s rows/cols replaced by identity.

    ``Q_masked`` is block diagonal — ``Q[F, F]`` on the free block, ``I`` on the
    fixed one — and ``q_masked`` is zero on ``Z``, so the solve returns
    ``x_F = Q[F,F]⁻¹ q_F`` and ``x_Z = 0`` *exactly*, in one ``(n, n)``
    Cholesky. Costlier per pass than factorising the free block alone, and the
    only version with a shape a jit can keep: the free set changes every pass,
    an index list would retrace, a boolean mask does not.
    """
    keep = ~Z
    Q_masked = jnp.where(keep[:, None] & keep[None, :], Q, 0.0)
    Q_masked = Q_masked + jnp.diag(jnp.where(Z, 1.0, 0.0))
    q_masked = jnp.where(Z, 0.0, q)
    L = jnp.linalg.cholesky(Q_masked)
    return cho_solve((L, True), q_masked)


def active_set_masked_jax(Q, q, fixed0_mask, n_passes: int, *, tau_rel: float = TAU_REL_DEFAULT):
    """Fixed-budget masked active-set scheme — one full-size Cholesky per pass.

    The jit-able twin of :func:`active_set_certified` under the ``"free_all"``
    policy. Shapes are static throughout: the fixed set is a boolean vector, the
    pass count is a Python int (a ``lax.scan`` length), and every pass
    factorises the same ``(n, n)`` matrix, so one compile serves every pass and
    every iterate of a search.

    ``fixed0_mask`` is the library's edge-zero set: seeded at pass 0 and never
    released. Once a pass certifies, the fixed set is frozen, so the remaining
    budget reproduces the certified iterate rather than wandering off it.

    Parameters
    ----------
    Q, q
        The Jacobi-scaled QP.
    fixed0_mask
        Boolean ``(n,)``; use ``jnp.zeros(n, bool)`` for the pure problem.
    n_passes
        Number of restricted passes **after** pass 0 (static).

    Returns
    -------
    dict with ``x`` (the final iterate), ``x_unconstrained`` (pass 0), ``xs``
    (``(n_passes + 1, n)``, pass 0 first), and, per pass, ``n_fixed``,
    ``n_primal_violations``, ``n_dual_violations`` and ``certified``.
    """
    Q = jnp.asarray(Q, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    permanent = jnp.asarray(fixed0_mask, dtype=bool)

    tau_g = tau_rel * jnp.max(jnp.abs(q))

    x0 = _masked_solve(Q, q, permanent)
    # Same seed as the numpy reference: strict negatives, no tolerance band.
    Z0 = permanent | (x0 < 0.0)

    def step(carry, _):
        Z, done = carry
        x = _masked_solve(Q, q, Z)
        g = Q @ x - q
        tau_x = tau_rel * jnp.max(jnp.abs(x))

        free = ~Z
        freeable = Z & ~permanent
        primal_violations = free & (x < -tau_x)
        dual_violations = freeable & (g < -tau_g)
        certified = ~(jnp.any(primal_violations) | jnp.any(dual_violations))

        Z_new = (Z | primal_violations) & ~dual_violations
        Z_next = jnp.where(done | certified, Z, Z_new)

        out = {
            "x": x,
            "n_fixed": jnp.sum(Z),
            "n_primal_violations": jnp.sum(primal_violations),
            "n_dual_violations": jnp.sum(dual_violations),
            "certified": certified,
        }
        return (Z_next, done | certified), out

    (_, _), outs = lax.scan(step, (Z0, jnp.bool_(False)), None, length=n_passes)

    xs = jnp.concatenate([x0[None, :], outs["x"]], axis=0)
    return {
        "x": xs[-1],
        "x_unconstrained": x0,
        "xs": xs,
        "n_fixed": outs["n_fixed"],
        "n_primal_violations": outs["n_primal_violations"],
        "n_dual_violations": outs["n_dual_violations"],
        "certified": outs["certified"],
    }


# ---------------------------------------------------------------------------
# Evidence error
# ---------------------------------------------------------------------------


def active_set_evidence_error(
    system: FixedLightSystem,
    x,
    *,
    reference_log_evidence: float | None = None,
    clip: bool = False,
) -> float:
    """``log_evidence(iterate) − log_evidence(PDIP)``, nats.

    The iterate is **physical** (already multiplied by ``d_scale``). With
    ``clip=True`` it is passed through ``max(x, 0)`` first: an early active-set
    iterate can carry negative entries, and such an infeasible point can score
    *above* the constrained optimum — a positive delta is a warning, not a win,
    which is why both the raw and the clipped number are worth recording.

    The two log-det terms are properties of ``F + λH`` and ``H``, not of ``x``,
    so only ``χ²`` and ``sᵀHs`` actually move between iterates; they are
    recomputed per call anyway rather than assumed away.
    """
    x = np.asarray(x, dtype=float)
    if clip:
        x = np.maximum(x, 0.0)
    if reference_log_evidence is None:
        reference_log_evidence = system.log_evidence(system.pdip()[0])
    return system.log_evidence(x) - float(reference_log_evidence)
