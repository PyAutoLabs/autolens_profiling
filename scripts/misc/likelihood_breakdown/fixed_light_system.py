"""The fixed-lens-light source-only system — the JAX-free half of ``active_set_steps``.

Why this module exists
----------------------

``active_set_steps.py`` carries two things that have nothing to do with each
other: the **certified active-set kernels** (JAX, ``lax.scan``, ``cho_solve``)
and the **system builders** that turn a fit with linear lens light into the
source-only system the whole fixed-lens-light epic measures. It imports ``jax``,
``jax.numpy`` and ``jax.lax`` at module level (``active_set_steps.py:124-131``)
for the kernels, so a cell that wants only ``fixed_light_system_from`` — a numba
CPU cell, whose entire point is to measure what the CPU costs *without* JAX —
drags a JAX runtime into its process by importing it.

This module is the builders, extracted verbatim, with no JAX at import time and
none on the default numpy path. ``active_set_steps`` re-exports every name from
here, so nothing that imports it changes: the kernels keep their builders, their
callers keep their import line, and the pins are the same objects.

What changed in the move
------------------------

Three additions, each of which exists because the numba CPU leg needs it and
none of which alters the JAX path:

- :func:`jacobi_scaled_np` is the promotion of ``fixed_light_cpu_kernels``'s
  private ``_jacobi_scaled_np`` — the same preconditioning the library applies
  on both its paths, in pure numpy. ``fixed_light_cpu_kernels._jacobi_scaled_np``
  is now an alias of it, so there is one implementation rather than two.
- :func:`linear_system_from` takes ``scaling={"jax", "numpy"}``. ``"jax"`` is the
  default and calls ``reconstruction_steps.jacobi_scaled`` through a
  **function-local** ``import jax.numpy``, bit for bit what every existing pin
  was calibrated against; ``"numpy"`` calls :func:`jacobi_scaled_np` and imports
  nothing.
- :func:`fixed_light_system_from` takes
  ``sparse_operator={"drop", "carry", "rebake_cpu"}``. The subtraction rebuilds
  the ``Imaging``, and a rebuilt ``Imaging`` has no operator unless one is given:
  ``"drop"`` is the historic behaviour (dense S3, which is what the JAX cells
  measure), ``"carry"`` passes the original operator straight through, and
  ``"rebake_cpu"`` calls ``apply_sparse_operator_cpu()`` on the rebuilt dataset —
  the numba CPU leg's route, and the only one of the three that is correct on
  the sparse path (see below).

The rebuild also now carries ``convolve_over_sample_size_lp`` /
``convolve_over_sample_size_pixelization`` and ``noise_covariance_matrix``
through from the source dataset. The previous rebuild dropped all three silently
(``Imaging.__init__``, ``PyAutoArray/autoarray/dataset/imaging/dataset.py:65-79``),
and the PSF over-sampling one matters: ``apply_sparse_operator`` *raises* on an
over-sampled PSF (``dataset.py:622-627``), so defaulting the field back to 1
would have hidden that refusal behind a silently different dataset instead of
surfacing it.

Why a re-baked operator is the right one
----------------------------------------

``InversionImagingSparseNumba.psf_weighted_data`` recomputes the weighted data
from ``self.data`` on every inversion
(``imaging_numba/sparse.py:94-101``) — it reads no baked weight map — while the
operator itself (``psf_precision_operator_sparse``, ``indexes``, ``lengths``) is
a function of the **noise map, PSF and mask only**
(``dataset.py:682-706``). Subtracting the fixed lens light changes the data and
nothing else, so a re-baked operator is *identical* to the original's and the
sparse S3 data vector sees the subtraction. That identity is asserted, not
assumed, by the cell's P1 gate.

(The historic warning in ``active_set_steps``'s docstring — that a sparse data
vector would ignore the subtraction — describes ``InversionImagingSparse``, the
**JAX** sparse class, whose ``psf_weighted_data`` does read a baked
``sparse_operator.weight_map``. The numba class is a different class with a
different property, which is why the numba leg can do what the JAX leg cannot.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "FixedLightSystem",
    "fixed_light_system_from",
    "jacobi_scaled_np",
    "linear_system_from",
]

#: The three ways :func:`fixed_light_system_from` can treat the sparse operator
#: when it rebuilds the ``Imaging`` on the subtracted data.
SPARSE_OPERATOR_MODES = ("drop", "carry", "rebake_cpu")


def jacobi_scaled_np(curvature_reg_matrix, data_vector):
    """``reconstruction_steps.jacobi_scaled`` in pure numpy.

    The same preconditioning the library applies on both its paths
    (``inversion_util.py:355-375``): ``d = sqrt(diag Q)``, ``D = 1/d``, solve
    ``(D Q D) y = D q``, recover ``x = y * D``. Re-expressed in numpy here so a
    CPU kernel row imports no JAX — these rows exist precisely to measure what
    the CPU costs without it.
    """
    crm = np.asarray(curvature_reg_matrix, dtype=float)
    dv = np.asarray(data_vector, dtype=float)
    d_scale = 1.0 / np.sqrt(np.diag(crm))
    return (crm * d_scale[:, None]) * d_scale[None, :], dv * d_scale, d_scale


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
        """Every evidence term of an iterate, via ``reconstruction_steps``.

        JAX is imported **here**, not at module scope: scoring an iterate is a
        JAX operation, but building a system is not, and a numba CPU cell only
        ever does the latter.
        """
        import jax.numpy as jnp

        from likelihood_breakdown import reconstruction_steps

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
        import jax
        import jax.numpy as jnp

        from likelihood_breakdown import reconstruction_steps

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
        from likelihood_breakdown.active_set_steps import unconstrained_solve

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


def _jacobi_scaled(curvature_reg_matrix, data_vector, scaling: str):
    """``(Q, q, d_scale)`` through the requested arithmetic.

    ``"jax"`` reaches ``reconstruction_steps.jacobi_scaled`` — the path every
    existing pin was calibrated on, kept bit-identical — behind a function-local
    import, so nothing imports JAX unless a caller asks for it. ``"numpy"`` is
    :func:`jacobi_scaled_np` and imports nothing.
    """
    if scaling == "numpy":
        return jacobi_scaled_np(curvature_reg_matrix, data_vector)

    if scaling != "jax":
        raise ValueError(f"scaling must be 'jax' or 'numpy' (got {scaling!r})")

    import jax.numpy as jnp

    from likelihood_breakdown import reconstruction_steps

    return reconstruction_steps.jacobi_scaled(
        jnp.asarray(curvature_reg_matrix, dtype=jnp.float64),
        jnp.asarray(data_vector, dtype=jnp.float64),
    )


def linear_system_from(
    fit,
    dataset=None,
    *,
    name: str = "system",
    scaling: str = "jax",
) -> FixedLightSystem:
    """Read one fit's linear system (and its scoring arrays) off the inversion.

    Everything is pulled from the library objects rather than rebuilt: ``F + λH``
    and ``D`` off the inversion, the reduced blocks off its rank-stripped
    properties, the edge-zero set off ``solve_ids_to_keep``.

    ``scaling`` selects the arithmetic the Jacobi preconditioning is done in.
    The default ``"jax"`` is what every phase-0 to phase-5 pin was calibrated
    against; ``"numpy"`` produces the same numbers to fp64 round-off with no JAX
    in the process at all.
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

    Q, q, d_scale = _jacobi_scaled(curvature_reg_matrix, data_vector, scaling)

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
    scaling: str = "jax",
    sparse_operator: str = "drop",
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
       same noise map, PSF, mask, over-sampling, PSF convolution over-sampling
       and noise covariance matrix;
    3. the tracer is rebuilt with every light profile stripped, leaving mass,
       shear and the source's pixelization;
    4. a ``FitImaging`` on (2) with (3) is the source-only inversion.

    The subtraction is applied to the *dataset* rather than carried as a regular
    light profile on the tracer, so it reaches the data vector on every
    formalism rather than only the dense one.

    ``sparse_operator`` decides what the rebuilt dataset carries, and therefore
    which inversion class the factory dispatches
    (``inversion/inversion/factory.py:136-154``):

    - ``"drop"`` — no operator, a dense ``InversionImagingMapping``. The historic
      behaviour and the default, because every existing caller and pin expects
      the dense S3 system.
    - ``"carry"`` — the source dataset's own operator object, passed through
      unchanged. Correct only when the operator does not depend on the data;
      offered so a cell can *measure* that rather than assume it.
    - ``"rebake_cpu"`` — ``apply_sparse_operator_cpu()`` on the rebuilt dataset,
      giving a ``SparseLinAlgImagingNumba`` operator baked from the subtracted
      dataset's (identical) noise map, PSF and mask, and an
      ``InversionImagingSparseNumba``. The numba CPU route.
    """
    import autolens as al

    if sparse_operator not in SPARSE_OPERATOR_MODES:
        raise ValueError(
            f"sparse_operator must be one of {list(SPARSE_OPERATOR_MODES)} "
            f"(got {sparse_operator!r})"
        )

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

    # Every field `Imaging.__init__` accepts that this rebuild can carry is
    # carried (dataset.py:65-79). `convolve_over_sample_size_*` in particular:
    # `apply_sparse_operator` REFUSES an over-sampled PSF (dataset.py:622-627),
    # and silently defaulting the field back to 1 would turn that refusal into a
    # quietly different dataset.
    subtracted = dataset.__class__(
        data=dataset.data - blurred,
        noise_map=dataset.noise_map,
        psf=dataset.psf,
        noise_covariance_matrix=dataset.noise_covariance_matrix,
        over_sample_size_lp=dataset.over_sample_size_lp,
        over_sample_size_pixelization=dataset.over_sample_size_pixelization,
        convolve_over_sample_size_lp=dataset.convolve_over_sample_size_lp,
        convolve_over_sample_size_pixelization=dataset.convolve_over_sample_size_pixelization,
        sparse_operator=dataset.sparse_operator if sparse_operator == "carry" else None,
    )

    if sparse_operator == "rebake_cpu":
        subtracted = subtracted.apply_sparse_operator_cpu()

    source_only_tracer = _light_stripped_tracer(fit.tracer)

    fit_source_only = al.FitImaging(
        dataset=subtracted,
        tracer=source_only_tracer,
        adapt_images=adapt_images,
        settings=settings,
        xp=np,
    )

    system = linear_system_from(fit_source_only, subtracted, name=name, scaling=scaling)
    system.source_only_tracer = source_only_tracer
    system.subtracted_light_image = np.asarray(blurred.array, dtype=float)
    system.subtracted_light_flux = float(np.sum(system.subtracted_light_image))
    return system
