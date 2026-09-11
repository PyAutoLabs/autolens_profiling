"""Matrix-free pixelized-imaging likelihood kernels, for the matrix-free breakdown.

Why this module exists
----------------------

The 2026-09-10 A100 baseline and the 2026-09-11 reconstruction split
(``results/notes/a100_pixelized_baseline_2026_09.md``) took the dominant
likelihood row apart: at the 1500-source-pixel fiducial the *exact*
unconstrained solve of ``F + λH`` is 1.52-1.58 ms, each of the two log-det
Choleskys is ~1.2 ms, and the 37 ms ``Regularized reconstruction`` row is
positivity — 21-22 PDIP iterations, every one a fresh dense KKT Cholesky. So a
matrix-free line is **not** a speed play at n ≈ 1500. Its value is that the
dense path grows as ``n**3`` in the factorisation and ``N_pix × N_src`` in the
assembly (35 GB at 1521 source pixels, > 100 GB at 5000 — no longer an A100),
while a matrix-free matvec costs the sparse non-zero count plus a pair of FFTs
and is nearly independent of ``N_src``. This module is the **reference
implementation** whose numbers the results note quotes; it is not a production
path and nothing in ``PyAutoArray`` imports it.

"Matrix-free" here means: ``F + λH`` is never formed. Every consumer of it —
the linear solve, the positivity solve and both log determinants — is driven by
the single primitive :func:`curvature_reg_matvec`, which maps a
``(total_params, B)`` block of vectors to ``(F + λH) V`` in **one FFT pass per
matvec** (one forward blur and one adjoint blur of the batch, whatever ``B``
is), plus a scatter/gather through the mapper's COO triplets.

The operator, block by block
----------------------------

The profiled configuration is the ``func_list_and_mapper`` branch
(:mod:`sparse_steps`): one pixelized ``Mapper`` with sparse triplets ``A`` on
the rectangular FFT grid, and one ``AbstractLinearObjFuncList`` (the MGE-60
lens-light basis) whose columns are **already PSF-operated**. Writing ``H`` for
the blur, ``N⁻¹`` for the inverse variances and ``B̃`` for the operated basis
scattered onto the FFT grid, the library's ``F`` is

.. code-block:: text

    F = [ Aᵀ Hᵀ N⁻¹ H A     Aᵀ Hᵀ N⁻¹ B̃ ]
        [ B̃ᵀ N⁻¹ H A        B̃ᵀ N⁻¹ B̃    ]

so a joint matvec is one forward blur of ``A V_m``, one inverse-variance
weighting of ``H A V_m + B̃ V_f``, one adjoint blur, and two gathers:

.. code-block:: text

    u = H (A V_m) + B̃ V_f          # model image on the FFT grid
    y = N⁻¹ u
    (F V)_mapper = Aᵀ (Hᵀ y)
    (F V)_func   = B̃ᵀ y

**This is not ``ImagingSparseOperator.apply_operator``.** That method applies
the fused ``W = Hᵀ N⁻¹ H``, which is right for the mapper-mapper block and
wrong for anything touching the func block: ``B̃`` is already blurred, so
routing it through ``W`` convolves it a second time. The forward and adjoint
blurs are therefore split here, reusing the operator's own precomputed
``Khat_r`` / ``Khat_flip_r`` rFFT state and its crop offsets, so the arithmetic
is identical to ``apply_operator`` on the mapper block and correct on the
others. ``test_matvec_reproduces_curvature_reg_matrix`` pins that column by
column against ``inversion.curvature_reg_matrix``.

``add_to_diag`` goes on the **no-regularization** parameters
(``Settings.no_regularization_add_to_curvature_diag_value`` applied at
``inversion.no_regularization_index_list`` by
``inversion_util.curvature_matrix_with_added_to_diag_from``) — that is the MGE
block, not the mapper diagonal. It is held as a ``(total_params,)`` vector and
added to the matvec output.

What is exact and what is not
-----------------------------

- :func:`curvature_matvec` / :func:`curvature_reg_matvec` are **exact** — they
  reproduce ``inversion.curvature_reg_matrix`` to float round-off.
- :func:`pcg_solve` is exact up to its residual tolerance, and reports the
  iteration count that the crossover argument turns on.
- :func:`pdip_matrix_free` mirrors ``autoarray.util.jax_nnls.solve_nnls`` step
  for step (same ``jaxnnls`` initialisation, predictor-corrector centering,
  line searches and KKT residual test) with the dense KKT Cholesky replaced by
  a PCG solve of the same system. The solution is the same NNLS minimiser; the
  iteration *path* differs once the inner solve is inexact.
- :func:`jacobi_diag_estimate` is **approximate on the mapper block** by
  design: it uses only the diagonal of ``W``. It is a preconditioner, so this
  costs iterations, not correctness, and the iteration counts are recorded. The
  func block of the estimate is exact.
- :func:`slq_logdet` is a **stochastic** estimator: Hutchinson probes plus
  Lanczos quadrature. Its error is reported as a per-probe spread, and the
  probes come from a caller-supplied PRNG key so a sweep can use common random
  numbers and see a smooth bias rather than jitter.

Open item: ``λH`` is taken from the library's dense
``inversion.regularization_matrix`` and converted **once** to a
``jax.experimental.sparse.BCOO`` at context build, outside every timed row.
Assembling ``H`` directly from the mesh neighbours (never forming the dense
matrix at all) is left to the PyAutoArray phase, and is the one place this
reference still touches an ``(n, n)`` dense array.

Kernel API
----------

Every kernel takes either a :class:`MatrixFreeContext` or a plain ``matvec``
callable on ``(n, B)`` blocks, so the CPU unit tests exercise the solvers on
random dense SPD systems with no imaging setup at all
(:func:`as_matvec` is the dispatcher).

Library provenance (PyAutoArray, verified 2026-09-11)
-----------------------------------------------------

- ``inversion/inversion/imaging/inversion_imaging_util.py``:
  ``ImagingSparseOperator`` (``Khat_r`` / ``Khat_flip_r`` / ``fft_shape`` /
  ``inverse_variances_native``), ``apply_operator``,
  ``curvature_matrix_diag_from``, ``curvature_matrix_off_diag_func_list_from``.
- ``inversion/inversion/inversion_util.py``:
  ``curvature_matrix_with_added_to_diag_from``,
  ``reconstruction_positive_only_from`` (the Jacobi scaling).
- ``inversion/inversion/abstract.py``: ``no_regularization_index_list``,
  ``mapper_indices``, ``regularization_matrix_reduced``,
  ``curvature_reg_matrix_reduced``, ``_log_det_symmetric_from``.
- ``util/jax_nnls.py`` + ``jaxnnls/pdip.py``: ``solve_nnls``, ``initialize``,
  ``pdip_pc_step``, ``solve_kkt_rhs``, ``ort_line_search``,
  ``centering_params``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
from jax.ops import segment_sum

from likelihood_breakdown.sparse_steps import SparseContext

__all__ = [
    "MatrixFreeContext",
    "as_matvec",
    "bcoo_matvec",
    "curvature_matvec",
    "curvature_reg_matvec",
    "jacobi_diag_estimate",
    "matrix_free_context_from",
    "matrix_free_evidence_terms",
    "pcg_solve",
    "pcg_solve_batched",
    "pdip_matrix_free",
    "reduced_matvec_from",
    "slq_logdet",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixFreeContext:
    """Everything the matrix-free kernels need, resolved once and eagerly.

    Built by :func:`matrix_free_context_from`. Like :class:`SparseContext` it is
    closed over rather than passed as a traced argument: the FFT state, the
    triplets and the sparse ``λH`` are device-resident constants for a given
    model evaluation.

    Attributes
    ----------
    sparse_ctx
        The :class:`SparseContext` this one extends — it carries the operator,
        the param ranges and the mapper/func split.
    rows, cols, vals
        The mapper's COO triplets from ``Mapper.sparse_triplets_curvature``:
        ``rows`` are **rectangular FFT-grid** indices, ``cols`` source pixels,
        ``vals`` the mapping weights.
    funcs_on_grid
        ``(M_rect, n_funcs)`` — the operated (already PSF-convolved) MGE basis
        scattered from slim masked pixels onto the FFT grid via
        ``sparse_ctx.fft_index_for_masked_pixel``. Zero-width when there is no
        func list.
    inverse_variances_grid
        ``(M_rect,)`` — the operator's own ``inverse_variances_native``,
        flattened. Zero outside the mask, which is what makes the func-func
        block of the matvec equal the library's slim-grid GEMM.
    reg_bcoo
        ``λH`` as a ``(total_params, total_params)`` BCOO, built once from the
        library's dense ``inversion.regularization_matrix`` (zero rows/cols on
        the func block). Sparse assembly from mesh neighbours is an open item.
    reg_diag
        ``diag(λH)``, ``(total_params,)``.
    add_to_diag_vector
        ``(total_params,)`` — ``add_to_diag_value`` at every index in
        ``sparse_ctx.no_regularization_index_list``, zero elsewhere. Held as a
        vector (rather than applied by index inside the matvec) so the matvec
        is one fused add.
    """

    sparse_ctx: SparseContext
    rows: Any  # (nnz,) int32, FFT-grid indices
    cols: Any  # (nnz,) int32, source pixel indices
    vals: Any  # (nnz,) float64
    funcs_on_grid: Any  # (M_rect, n_funcs) float64
    inverse_variances_grid: Any  # (M_rect,) float64
    reg_bcoo: Any  # BCOO (total_params, total_params)
    reg_diag: Any  # (total_params,) float64
    add_to_diag_vector: Any  # (total_params,) float64
    add_to_diag_value: float
    total_params: int
    mapper_params: int
    n_funcs: int
    m_rect: int

    @property
    def operator(self):
        return self.sparse_ctx.operator

    @property
    def mapper_slice(self) -> slice:
        return slice(self.sparse_ctx.mapper_start, self.sparse_ctx.mapper_end)

    @property
    def func_slice(self) -> slice:
        return slice(self.sparse_ctx.func_start, self.sparse_ctx.func_end)


def matrix_free_context_from(inversion, dataset, sparse_ctx: SparseContext) -> MatrixFreeContext:
    """Build a :class:`MatrixFreeContext` from an ``InversionImagingSparse``.

    Eager: every array here is a one-off setup cost, measured as its own
    (untimed-row) context-build section by the cell, never inside a matvec.

    ``dataset`` is accepted for symmetry with
    :func:`sparse_steps.sparse_context_from` and because the operator's mask
    geometry is read through ``sparse_ctx``; nothing else is taken from it.
    """
    from autoarray.inversion.mappers.abstract import Mapper

    mapper = inversion.cls_list_from(cls=Mapper)[0]
    rows, cols, vals = mapper.sparse_triplets_curvature

    operator = sparse_ctx.operator
    m_rect = int(operator.y_shape) * int(operator.x_shape)

    if sparse_ctx.n_funcs > 0:
        operated = next(iter(inversion.linear_func_operated_mapping_matrix_dict.values()))
        operated = jnp.asarray(operated, dtype=jnp.float64)
    else:
        operated = jnp.zeros((sparse_ctx.fft_index_for_masked_pixel.shape[0], 0), jnp.float64)

    funcs_on_grid = jnp.zeros((m_rect, operated.shape[1]), dtype=jnp.float64)
    funcs_on_grid = funcs_on_grid.at[sparse_ctx.fft_index_for_masked_pixel, :].set(operated)

    regularization_matrix = jnp.asarray(inversion.regularization_matrix, dtype=jnp.float64)
    reg_bcoo = jsparse.BCOO.fromdense(regularization_matrix)

    total_params = sparse_ctx.total_params
    add_to_diag_vector = jnp.zeros(total_params, dtype=jnp.float64)
    if len(sparse_ctx.no_regularization_index_list) > 0:
        add_to_diag_vector = add_to_diag_vector.at[
            jnp.asarray(sparse_ctx.no_regularization_index_list, dtype=jnp.int32)
        ].set(sparse_ctx.add_to_diag_value)

    return MatrixFreeContext(
        sparse_ctx=sparse_ctx,
        rows=jnp.asarray(rows, dtype=jnp.int32),
        cols=jnp.asarray(cols, dtype=jnp.int32),
        vals=jnp.asarray(vals, dtype=jnp.float64),
        funcs_on_grid=funcs_on_grid,
        inverse_variances_grid=jnp.asarray(
            operator.inverse_variances_native, dtype=jnp.float64
        ).reshape(-1),
        reg_bcoo=reg_bcoo,
        reg_diag=jnp.diag(regularization_matrix),
        add_to_diag_vector=add_to_diag_vector,
        add_to_diag_value=float(sparse_ctx.add_to_diag_value),
        total_params=int(total_params),
        mapper_params=int(sparse_ctx.mapper_params),
        n_funcs=int(operated.shape[1]),
        m_rect=int(m_rect),
    )


# ---------------------------------------------------------------------------
# The operator: one FFT pass per matvec
# ---------------------------------------------------------------------------


def _blur(operator, X, kernel_hat):
    """Convolve a ``(M_rect, B)`` batch on the FFT grid with a precomputed rFFT.

    The body of ``ImagingSparseOperator.apply_operator``'s convolution, split
    out so the forward blur ``H`` (``Khat_r``) and the adjoint blur ``Hᵀ``
    (``Khat_flip_r``) can be applied separately — ``apply_operator`` only ever
    exposes the fused ``W = Hᵀ N⁻¹ H``, which double-blurs the already-operated
    MGE columns. Same padding, same ``s=(Fy, Fx)`` rFFT and the same
    ``Ky // 2, Kx // 2`` crop, so the mapper block is bit-comparable with the
    library's own curvature assembly.
    """
    y_shape, x_shape = operator.y_shape, operator.x_shape
    Fy, Fx = operator.fft_shape
    cy, cx = operator.Ky // 2, operator.Kx // 2

    B = X.shape[1]
    images = X.T.reshape((B, y_shape, x_shape))
    images_pad = jnp.pad(images, ((0, 0), (0, Fy - y_shape), (0, Fx - x_shape)))
    hat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))
    out_pad = jnp.fft.irfft2(hat * kernel_hat[None, :, :], s=(Fy, Fx))
    out = out_pad[:, cy : cy + y_shape, cx : cx + x_shape]
    return out.reshape((B, y_shape * x_shape)).T


def curvature_matvec(ctx: MatrixFreeContext, V):
    """``F V`` for a ``(total_params, B)`` block, without ever forming ``F``.

    Accepts a 1-D ``(total_params,)`` vector too and returns 1-D in that case.

    One forward blur, one inverse-variance weighting, one adjoint blur — for
    the whole batch, whatever ``B`` is. ``add_to_diag_value`` is added on the
    no-regularization (MGE) parameters, matching
    ``curvature_matrix_with_added_to_diag_from``.
    """
    one_d = V.ndim == 1
    V2 = V[:, None] if one_d else V

    V_m = V2[ctx.mapper_slice]
    V_f = V2[ctx.func_slice]

    # A V_m scattered onto the rectangular FFT grid.
    X = segment_sum(ctx.vals[:, None] * V_m[ctx.cols], ctx.rows, num_segments=ctx.m_rect)

    # The model image: blur the mapper part, add the already-blurred MGE part.
    u = _blur(ctx.operator, X, ctx.operator.Khat_r)
    if ctx.n_funcs > 0:
        u = u + ctx.funcs_on_grid @ V_f

    y = ctx.inverse_variances_grid[:, None] * u

    G = _blur(ctx.operator, y, ctx.operator.Khat_flip_r)
    out_mapper = segment_sum(
        ctx.vals[:, None] * G[ctx.rows], ctx.cols, num_segments=ctx.mapper_params
    )

    out = jnp.zeros_like(V2)
    out = out.at[ctx.mapper_slice].set(out_mapper)
    if ctx.n_funcs > 0:
        out = out.at[ctx.func_slice].set(ctx.funcs_on_grid.T @ y)

    out = out + ctx.add_to_diag_vector[:, None] * V2

    return out[:, 0] if one_d else out


def curvature_reg_matvec(ctx: MatrixFreeContext, V):
    """``(F + λH) V`` — the operator every solver and log-det in here drives."""
    return curvature_matvec(ctx, V) + _bcoo_apply(ctx.reg_bcoo, V)


def _bcoo_apply(bcoo, V):
    """``bcoo @ V`` — BCOO matmul already handles 1-D and ``(n, B)`` ``V``."""
    return bcoo @ V


def bcoo_matvec(bcoo) -> Callable:
    """A plain matvec closure over a BCOO — e.g. ``λH`` for its own log-det."""

    def matvec(V):
        return _bcoo_apply(bcoo, V)

    return matvec


def as_matvec(op) -> Callable:
    """Dispatch: a :class:`MatrixFreeContext` becomes its ``F + λH`` matvec.

    Anything else is assumed to already be a callable on ``(n, B)`` blocks (and
    on 1-D vectors), which is how the CPU unit tests drive these kernels on
    random dense SPD systems with no imaging setup.
    """
    if isinstance(op, MatrixFreeContext):
        return lambda V: curvature_reg_matvec(op, V)
    return op


def reduced_matvec_from(ctx: MatrixFreeContext, matvec: Callable | None = None) -> Callable:
    """The **reduced** (mapper-only) operator, by masking the func block out.

    The library's two log-dets are taken on the rank-stripped blocks
    (``curvature_reg_matrix_reduced`` / ``regularization_matrix_reduced``,
    i.e. ``M[mapper_indices][:, mapper_indices]``) because the full ``H`` is
    singular on the unregularised MGE rows. Embedding a mapper-length vector
    into a zero-padded full-length one, applying the full operator and reading
    the mapper rows back is exactly that submatrix product — no reduced matrix
    is formed.

    ``matvec`` defaults to ``F + λH``; pass ``bcoo_matvec(ctx.reg_bcoo)`` for
    the reduced ``λH``.
    """
    full_matvec = as_matvec(ctx) if matvec is None else matvec
    mapper_slice = ctx.mapper_slice

    def reduced(V):
        one_d = V.ndim == 1
        V2 = V[:, None] if one_d else V
        full = jnp.zeros((ctx.total_params, V2.shape[1]), dtype=V2.dtype)
        full = full.at[mapper_slice].set(V2)
        out = full_matvec(full)[mapper_slice]
        return out[:, 0] if one_d else out

    return reduced


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------


def jacobi_diag_estimate(ctx: MatrixFreeContext, *, floor: float = 1.0e-30):
    """An approximate ``diag(F + λH)`` for Jacobi preconditioning.

    The mapper block uses only the diagonal of ``W = Hᵀ N⁻¹ H``:

    .. code-block:: text

        w_j = (Σ_k psf_k²) · invvar_j        (on the FFT grid)
        diag(F)_i ≈ Σ_nnz vals² w[rows]      (segment_sum over cols)

    which is exact only where ``A`` has a single non-zero per column; in
    general it drops the off-diagonal of ``W``. That is deliberate — this is a
    preconditioner, so the approximation costs CG iterations (which every row
    records), not accuracy.

    The func block is **exact**: the operated basis is already blurred, so its
    curvature diagonal really is ``Σ_j B̃[j, f]² invvar_j``.

    ``add_to_diag_value`` (on the no-regularization parameters) and
    ``diag(λH)`` are added, and the result is floored at *floor* so ``1 / diag``
    is finite for a parameter no data touches.
    """
    # Σ_k psf_k², recovered from the operator's own precomputed half-spectrum
    # (it keeps no spatial copy of the padded kernel) — see :func:`_rfft_energy`.
    psf_energy = _rfft_energy(ctx.operator)

    w = psf_energy * ctx.inverse_variances_grid

    diag_mapper = segment_sum(ctx.vals**2 * w[ctx.rows], ctx.cols, num_segments=ctx.mapper_params)

    diag = jnp.zeros(ctx.total_params, dtype=jnp.float64)
    diag = diag.at[ctx.mapper_slice].set(diag_mapper)
    if ctx.n_funcs > 0:
        diag_func = jnp.sum(ctx.funcs_on_grid**2 * ctx.inverse_variances_grid[:, None], axis=0)
        diag = diag.at[ctx.func_slice].set(diag_func)

    diag = diag + ctx.add_to_diag_vector + ctx.reg_diag

    return jnp.maximum(diag, floor)


def _rfft_energy(operator):
    """``Σ_k psf_k²`` from the operator's precomputed half-spectrum.

    Parseval on the padded kernel: ``Σ psf² = (1 / (Fy·Fx)) Σ_full |Khat|²``.
    ``Khat_r`` is an rFFT, so every column except the zero column (and the
    Nyquist column when ``Fx`` is even) stands for two full-spectrum columns.
    """
    Fy, Fx = operator.fft_shape
    khat = jnp.asarray(operator.Khat_r)
    weights = jnp.full((khat.shape[1],), 2.0, dtype=jnp.float64).at[0].set(1.0)
    if Fx % 2 == 0:
        weights = weights.at[-1].set(1.0)
    return jnp.sum(jnp.abs(khat) ** 2 * weights[None, :]) / (Fy * Fx)


# ---------------------------------------------------------------------------
# Preconditioned conjugate gradients
# ---------------------------------------------------------------------------


def _pcg_core(matvec, Bmat, *, tol, maxiter, diag):
    """Column-independent preconditioned CG on ``(n, B)`` right-hand sides.

    One ``lax.while_loop``; every column carries its own ``alpha`` / ``beta``
    and its own convergence flag, and a converged column stops being updated
    (its search direction is zeroed) while the loop waits for the rest. The
    single-vector :func:`pcg_solve` is this with ``B = 1``, so there is one
    iteration body to get right rather than two.

    Stopping: ``‖r_c‖ ≤ tol · ‖b_c‖`` per column ``c``.
    """
    inv_diag = None if diag is None else (1.0 / diag)

    def precond(R):
        return R if inv_diag is None else (inv_diag[:, None] * R)

    b_norm = jnp.linalg.norm(Bmat, axis=0)
    target = tol * jnp.where(b_norm > 0.0, b_norm, 1.0)

    X0 = jnp.zeros_like(Bmat)
    R0 = Bmat
    Z0 = precond(R0)
    P0 = Z0
    rz0 = jnp.sum(R0 * Z0, axis=0)

    def cond(state):
        i, _, R, _, _, _ = state
        active = jnp.linalg.norm(R, axis=0) > target
        return jnp.logical_and(i < maxiter, jnp.any(active))

    def body(state):
        i, X, R, P, Z, rz = state

        active = (jnp.linalg.norm(R, axis=0) > target).astype(Bmat.dtype)

        AP = matvec(P)
        pAp = jnp.sum(P * AP, axis=0)
        alpha = jnp.where(pAp > 0.0, rz / jnp.where(pAp > 0.0, pAp, 1.0), 0.0) * active

        X = X + alpha[None, :] * P
        R = R - alpha[None, :] * AP
        Z = precond(R)
        rz_new = jnp.sum(R * Z, axis=0)
        beta = jnp.where(rz > 0.0, rz_new / jnp.where(rz > 0.0, rz, 1.0), 0.0)
        P = Z + beta[None, :] * P

        return i + 1, X, R, P, Z, rz_new

    n_iter, X, R, _, _, _ = lax.while_loop(cond, body, (0, X0, R0, P0, Z0, rz0))

    rel_residual = jnp.linalg.norm(R, axis=0) / jnp.where(b_norm > 0.0, b_norm, 1.0)
    return X, n_iter, rel_residual


def pcg_solve(op, D, *, tol=1.0e-10, maxiter=500, diag=None):
    """Solve ``(F + λH) x = D`` by preconditioned CG. Matrix-free, jit-able.

    Parameters
    ----------
    op
        A :class:`MatrixFreeContext` (its ``F + λH`` matvec is used) or any
        callable applying an SPD operator to ``(n, B)`` blocks and 1-D vectors.
    D
        The right-hand side, ``(n,)``.
    tol
        Relative residual target: the loop stops at ``‖r‖ ≤ tol ‖D‖``.
    maxiter
        Hard iteration cap; a solve that hits it returns its current iterate
        and its honest residual rather than raising.
    diag
        Jacobi preconditioner diagonal (``M⁻¹ = 1 / diag``). ``None`` runs
        unpreconditioned CG.

    Returns
    -------
    ``(x, n_iter, rel_residual)`` — the solution, the iterations actually run
    and ``‖r‖ / ‖D‖`` at exit.
    """
    matvec = as_matvec(op)
    X, n_iter, rel_residual = _pcg_core(matvec, D[:, None], tol=tol, maxiter=maxiter, diag=diag)
    return X[:, 0], n_iter, rel_residual[0]


def pcg_solve_batched(op, Dmat, *, tol=1.0e-10, maxiter=500, diag=None):
    """:func:`pcg_solve` for ``(n, B)`` right-hand sides, each its own CG state.

    The loop runs until every column has converged (or the cap is hit), so the
    reported iteration count is the worst column's — the same ``while_loop``
    straggler behaviour the batched NNLS row has. Not needed by the SLQ or PDIP
    paths; it is what the ``@vmap`` row of the cell drives.
    """
    matvec = as_matvec(op)
    X, n_iter, rel_residual = _pcg_core(matvec, Dmat, tol=tol, maxiter=maxiter, diag=diag)
    return X, n_iter, rel_residual


# ---------------------------------------------------------------------------
# Matrix-free positivity: PDIP with an inner PCG
# ---------------------------------------------------------------------------


def _ort_line_search(x, dx):
    """``jaxnnls.pdip.ort_line_search``: max ``alpha ≤ 1`` with ``x + alpha dx ≥ 0``."""
    min_batch = jnp.min(jnp.where(dx < 0, -x / dx, jnp.inf))
    return jnp.min(jnp.array([1.0, min_batch]))


def _solve_kkt_rhs(s, z, P_inv_vec, solve, v1, v2, v3):
    """``jaxnnls.pdip.solve_kkt_rhs`` with the Cholesky solve injected.

    *solve* applies ``(Q + diag(P_inv_vec))⁻¹`` — a dense ``cho_solve``
    upstream, a PCG here. Returns ``(dx, ds, dz)`` and the solve's iteration
    count.
    """
    r2 = -v3 + v2 / z
    p1 = -v1 - P_inv_vec * r2
    dx, cg_iter = solve(p1)
    ds = dx - v3
    dz = -(v2 + z * ds) / s
    return dx, ds, dz, cg_iter


def pdip_matrix_free(
    op,
    D,
    *,
    diag,
    solver_tol=None,
    max_iter=50,
    cg_tol=1.0e-8,
    cg_maxiter=500,
    jacobi_scaling=True,
):
    """Non-negative solve of ``(F + λH) x = D``, ``x ≥ 0``, without forming ``F + λH``.

    A step-for-step mirror of ``autoarray.util.jax_nnls.solve_nnls`` (whose
    iteration body is ``jaxnnls.pdip.pdip_pc_step``) with the **only** change
    being the linear solver inside each Newton step:

    - upstream: ``L = cho_factor(Q + diag(z / s))`` once per PDIP iteration,
      then two ``cho_solve`` calls (predictor, then corrector);
    - here: two :func:`pcg_solve` calls on the same system
      ``(Q + diag(z / s)) Δx = rhs``, with ``Q v = curvature_reg_matvec(ctx, v)``
      and a Jacobi preconditioner ``diag(Q) + z / s``.

    Everything else is upstream's: ``initialize`` (whose own ``(Q + I) x = q``
    solve is likewise a PCG here), the KKT residual
    ``‖(Qx - q - z, s z, s - x)‖_∞ < solver_tol`` test, the predictor-corrector
    centering ``sigma``/``mu``, the ``0.99`` fraction-to-boundary line search,
    the ``max_iter`` cap and the ``solver_tol=None`` default semantics
    (``min(n · eps · 5e3, 1e-2)``, taken from ``jaxnnls.pdip.EPSILON``).

    Jacobi scaling
    --------------
    ``reconstruction_positive_only_from`` rescales before calling the solver —
    ``d = sqrt(diag(Q))``, ``D_s = 1 / d``, solve ``(D_s Q D_s) y = D_s q`` for
    ``y ≥ 0`` and return ``y · D_s`` — so that ill-conditioned ``Q`` does not
    break the relaxed-KKT backward pass. With ``jacobi_scaling=True`` (the
    library default, ``general.yaml: nnls_jacobi_preconditioning``) the same
    scaling is applied here, from the supplied *diag*, and the returned ``x``
    is the **unscaled** reconstruction — directly comparable with
    ``reconstruction_steps.nnls_pdip``'s ``x_pc * D_scale`` and with
    ``inversion.reconstruction``. Because ``D_s`` is diagonal and positive the
    minimiser is unchanged; only the iterate path is. When *diag* is the
    approximate :func:`jacobi_diag_estimate` rather than the exact diagonal,
    the scaled system is merely near-unit-diagonal — the solution still matches,
    the iteration count need not.

    Parameters
    ----------
    op
        A :class:`MatrixFreeContext` or a plain matvec callable (the CPU tests
        pass ``lambda V: Q @ V``).
    D
        The data vector, ``(n,)``.
    diag
        ``diag(Q)``, exact or estimated — used for the scaling and for the
        inner CG preconditioner.
    solver_tol, max_iter
        As ``jax_nnls.solve_nnls``.
    cg_tol, cg_maxiter
        The inner PCG's relative-residual target and cap.

    Returns
    -------
    ``(x, pdip_iter, total_cg_iter, converged)`` — ``converged`` is upstream's
    ``0``/``1`` flag, ``total_cg_iter`` the summed inner CG iterations (three
    solves' worth per PDIP step, plus the initialisation).
    """
    from jaxnnls.pdip import EPSILON

    matvec = as_matvec(op)

    n = D.shape[0]
    diag = jnp.asarray(diag, dtype=D.dtype)

    if jacobi_scaling:
        scale = 1.0 / jnp.sqrt(diag)
    else:
        scale = jnp.ones_like(diag)

    def Q_matvec(V):
        if V.ndim == 1:
            return scale * matvec(scale * V)
        return scale[:, None] * matvec(scale[:, None] * V)

    q = D * scale
    diag_scaled = diag * scale**2

    if solver_tol is None:
        solver_tol = min(n * float(EPSILON), 1.0e-2)
    solver_tol = jnp.asarray(solver_tol, dtype=q.dtype)

    def shifted_solve(shift, rhs):
        """PCG on ``(Q + diag(shift)) y = rhs``."""

        def shifted(V):
            if V.ndim == 1:
                return Q_matvec(V) + shift * V
            return Q_matvec(V) + shift[:, None] * V

        y, n_iter, _ = pcg_solve(
            shifted, rhs, tol=cg_tol, maxiter=cg_maxiter, diag=diag_scaled + shift
        )
        return y, n_iter

    # --- jaxnnls.pdip.initialize, with the dense solve replaced ---------------
    ones = jnp.ones_like(q)
    x, init_cg = shifted_solve(ones, q)
    z = -x
    alpha_p = -jnp.min(-z)
    s = lax.select(alpha_p < 0, -z, 1 + alpha_p - z)
    alpha_d = -jnp.min(z)
    z = lax.select(alpha_d >= 0, 1 + alpha_d + z, z)

    # --- the predictor-corrector loop ---------------------------------------
    def cond(state):
        _, _, _, converged, pdip_iter, _ = state
        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    def step(state):
        x, s, z, _, pdip_iter, cg_total = state

        r1 = Q_matvec(x) - q - z
        r2 = s * z
        r3 = s - x

        kkt_res = jnp.concatenate((r1, r2, r3))
        converged = lax.select(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

        P_inv_vec = z / s

        def solve(rhs):
            return shifted_solve(P_inv_vec, rhs)

        _, ds_a, dz_a, cg_a = _solve_kkt_rhs(s, z, P_inv_vec, solve, r1, r2, r3)

        mu = jnp.dot(s, z) / s.shape[0]
        alpha = jnp.min(jnp.array([_ort_line_search(s, ds_a), _ort_line_search(z, dz_a)]))
        sigma = (jnp.dot(s + alpha * ds_a, z + alpha * dz_a) / jnp.dot(s, z)) ** 3

        r2c = r2 - (sigma * mu - (ds_a * dz_a))
        dx, ds, dz, cg_c = _solve_kkt_rhs(s, z, P_inv_vec, solve, r1, r2c, r3)

        alpha = 0.99 * jnp.min(jnp.array([_ort_line_search(s, ds), _ort_line_search(z, dz)]))

        return (
            x + alpha * dx,
            s + alpha * ds,
            z + alpha * dz,
            converged,
            pdip_iter + 1,
            cg_total + cg_a + cg_c,
        )

    x, _, _, converged, pdip_iter, cg_total = lax.while_loop(cond, step, (x, s, z, 0, 0, init_cg))

    return x * scale, pdip_iter, cg_total, converged


# ---------------------------------------------------------------------------
# Stochastic Lanczos quadrature log-determinant
# ---------------------------------------------------------------------------


def slq_logdet(
    matvec,
    n,
    *,
    n_probes: int = 16,
    n_lanczos: int = 40,
    key,
    reorth: bool = False,
    tiny: float = 1.0e-300,
):
    """``log det A`` by stochastic Lanczos quadrature, ``A`` given only as a matvec.

    Hutchinson with Rademacher probes ``z`` (``E[z zᵀ] = I``, ``‖z‖² = n``)
    turns the log-determinant into ``E[zᵀ (log A) z]``, and ``m`` steps of
    Lanczos starting from ``z / ‖z‖`` give the Gauss quadrature

    .. code-block:: text

        zᵀ (log A) z  ≈  n · Σ_k τ_k1² log θ_k

    with ``θ`` the eigenvalues of the ``(m, m)`` tridiagonal and ``τ_k1`` the
    first component of its ``k``-th eigenvector. All probes are carried as the
    columns of an ``(n, n_probes)`` block, so the operator is applied **once per
    Lanczos step** for the whole batch — the reason this costs ``m`` matvecs
    rather than ``m · n_probes``.

    Parameters
    ----------
    matvec
        Callable applying the SPD operator to an ``(n, n_probes)`` block (a
        :class:`MatrixFreeContext` is accepted and dispatched).
    n
        The operator's dimension.
    n_probes, n_lanczos
        Probe count and Lanczos steps — the two knobs the note's
        noise-vs-probes curve sweeps.
    key
        A ``jax.random`` key. Fixed key ⇒ bit-identical repeat, which is what
        makes the log-evidence error a smooth bias across a parameter sweep
        (common random numbers) rather than jitter.
    reorth
        Full re-orthogonalisation: keep the Lanczos basis and run one
        Gram-Schmidt pass per step. Costs ``O(m² n p)`` flops and ``m n p``
        memory; buys back the loss of orthogonality that otherwise duplicates
        eigenvalues at large ``m``.
    tiny
        Floor applied to the Ritz values before the log. Non-positive ``θ`` can
        only come from round-off on a positive-definite operator; the count of
        clamped values is returned so a run can say whether it happened.

    Returns
    -------
    ``(estimate, per_probe, n_clamped)`` — the mean over probes, the
    ``(n_probes,)`` per-probe values (whose spread is the estimator's own error
    bar: ``std / sqrt(n_probes)``) and how many Ritz values were floored.
    """
    matvec = as_matvec(matvec)

    Z = jax.random.rademacher(key, (n, n_probes), dtype=jnp.float64).astype(jnp.float64)
    z_norm = jnp.linalg.norm(Z, axis=0)
    Q0 = Z / z_norm[None, :]

    m = int(n_lanczos)

    alphas0 = jnp.zeros((m, n_probes), dtype=jnp.float64)
    betas0 = jnp.zeros((m, n_probes), dtype=jnp.float64)

    if reorth:
        basis0 = jnp.zeros((m, n, n_probes), dtype=jnp.float64)

        def body(j, state):
            q_prev, q_cur, beta_prev, alphas, betas, basis = state
            basis = basis.at[j].set(q_cur)

            w = matvec(q_cur)
            alpha = jnp.sum(q_cur * w, axis=0)
            w = w - alpha[None, :] * q_cur - beta_prev[None, :] * q_prev

            # One Gram-Schmidt pass against the stored basis (rows beyond j are
            # still zero, so they contribute nothing).
            coeffs = jnp.einsum("knp,np->kp", basis, w)
            w = w - jnp.einsum("knp,kp->np", basis, coeffs)

            beta = jnp.linalg.norm(w, axis=0)
            q_next = w / jnp.where(beta > 0.0, beta, 1.0)[None, :]

            alphas = alphas.at[j].set(alpha)
            betas = betas.at[j].set(beta)
            return q_cur, q_next, beta, alphas, betas, basis

        _, _, _, alphas, betas, _ = lax.fori_loop(
            0,
            m,
            body,
            (jnp.zeros_like(Q0), Q0, jnp.zeros(n_probes, jnp.float64), alphas0, betas0, basis0),
        )
    else:

        def body(j, state):
            q_prev, q_cur, beta_prev, alphas, betas = state

            w = matvec(q_cur)
            alpha = jnp.sum(q_cur * w, axis=0)
            w = w - alpha[None, :] * q_cur - beta_prev[None, :] * q_prev

            beta = jnp.linalg.norm(w, axis=0)
            q_next = w / jnp.where(beta > 0.0, beta, 1.0)[None, :]

            alphas = alphas.at[j].set(alpha)
            betas = betas.at[j].set(beta)
            return q_cur, q_next, beta, alphas, betas

        _, _, _, alphas, betas = lax.fori_loop(
            0, m, body, (jnp.zeros_like(Q0), Q0, jnp.zeros(n_probes, jnp.float64), alphas0, betas0)
        )

    # Tridiagonals, (n_probes, m, m).
    diag = alphas.T
    off = betas[: m - 1].T
    T = jnp.zeros((n_probes, m, m), dtype=jnp.float64).at[:, jnp.arange(m), jnp.arange(m)].set(diag)
    if m > 1:
        idx = jnp.arange(m - 1)
        T = T.at[:, idx, idx + 1].set(off)
        T = T.at[:, idx + 1, idx].set(off)

    theta, vectors = jnp.linalg.eigh(T)
    tau = vectors[:, 0, :]

    n_clamped = jnp.sum(theta <= tiny)
    theta_safe = jnp.maximum(theta, tiny)

    per_probe = n * jnp.sum(tau**2 * jnp.log(theta_safe), axis=1)

    return jnp.mean(per_probe), per_probe, n_clamped


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def matrix_free_evidence_terms(
    ctx: MatrixFreeContext,
    D,
    x,
    *,
    log_det_curvature_reg,
    log_det_regularization,
    data,
    noise_map,
    model_data_fn,
) -> dict:
    """The five evidence terms, with the two log-dets supplied by SLQ.

    Same keys and the same ``-2 ln e = chi² + sᵀHs + ln det(F+λH) - ln det(H)
    + noise_norm`` assembly as
    :func:`reconstruction_steps.log_evidence_terms`, so a matrix-free result
    JSON is term-by-term comparable with the exact one. Two differences, both
    forced by being matrix-free:

    - the log-dets are **arguments**, not Choleskys of reduced dense blocks —
      they come from :func:`slq_logdet` on
      :func:`reduced_matvec_from`-masked operators;
    - the regularization term is ``xᵀ (λH x)`` through the BCOO, which equals
      the library's ``s_reducedᵀ H_reduced s_reduced`` exactly because the
      func rows and columns of ``H`` are zero.

    ``model_data_fn`` maps the reconstruction to the model image (lens light
    plus the mapped source) — the one piece that differs between the dense and
    w-tilde legs, which this helper stays blind to, exactly as
    ``log_evidence_terms`` does.

    ``D`` is accepted for signature symmetry with the exact comparator and to
    make the call site read as "these terms belong to this linear system"; the
    evidence does not use it.

    Returned values are Python floats: this is called **eagerly, once**, for
    the JSON, never inside a timed row.
    """
    del D

    model_data = model_data_fn(x)

    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    regularization_term = jnp.dot(x, _bcoo_apply(ctx.reg_bcoo, x))

    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map**2))

    log_evidence = -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )

    return {
        "chi_squared": float(chi_squared),
        "regularization_term": float(regularization_term),
        "log_det_curvature_reg": float(log_det_curvature_reg),
        "log_det_regularization": float(log_det_regularization),
        "noise_normalization": float(noise_normalization),
        "log_evidence": float(log_evidence),
    }
