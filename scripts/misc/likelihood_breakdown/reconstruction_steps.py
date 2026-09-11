"""Standalone reconstruction / log-evidence steps, for the reconstruction sub-rows.

Why this module exists
----------------------

The 2026-09-10 A100 baseline (``results/notes/a100_pixelized_baseline_2026_09.md``)
found the ``Regularized reconstruction`` row to be 36.7-37.9 ms on every
pixelized cell — 61-71 % of the per-call likelihood cost, and identical dense
vs sparse. That row is **one fused JIT unit**: the cells call
``al.util.inversion.reconstruction_positive_only_from`` on ``F + λH``, which
(``PyAutoArray/autoarray/inversion/inversion/inversion_util.py``) Jacobi-rescales
the system and hands it to ``autoarray.util.jax_nnls.solve_nnls_primal`` — a
``lax.while_loop`` over ``jaxnnls.pdip.pdip_pc_step``, capped at 50 iterations.
Nothing in that row is separable by timing the cell's own code: **every PDIP
iteration is a fresh dense Cholesky of the (n, n) KKT system**, inside the
external ``jaxnnls`` package, and the ``custom_vjp`` primal discards the
iteration count and the convergence flag on the way out.

So the split is not a partition of the row into stages. It is a set of
*comparators* measured on the same matrices:

- how many PDIP iterations the solve actually takes (``nnls_pdip`` calls the
  ``solve_nnls`` **driver** rather than the ``custom_vjp`` primal, so the count
  and the flag survive), and what one iteration costs on its own
  (``nnls_pdip_one_iteration``);
- what a single Cholesky of the same ``F + λH`` costs
  (``cholesky_curvature_reg``), and what a full unconstrained solve — one
  factorisation plus two triangular solves — costs (``cholesky_solve``). That
  is the number a matrix-free CG line has to beat, not the 37 ms;
- the two log-det Choleskys of the reduced blocks (``log_det_cholesky``) that
  step 13 folds into its single row, and every term of the evidence
  (``log_evidence_terms``).

Overlap rule
------------

Every row fed from this module is an **overlapping sub-row**: it re-measures
part of the work the parent row already contains, on the same arrays. The
sub-rows are never appended to a cell's ``likelihood_steps`` and never enter
``total_step_by_step`` — summing them against the parent row is meaningless,
the same discipline ``sparse_steps`` uses for ``steps_sparse_sub_rows``.

Dense and sparse
----------------

These functions take ``F``, ``H``, ``D`` and the reduced blocks as explicit
arrays, so the sub-rows are measured identically on the dense and ``--sparse``
legs. That is not an accident of the harness: the w-tilde path replaces the
mapping matrix, not the solve, so step 12 and the two log-dets are the same
code on both legs (``sparse_steps`` docstring).

Library provenance (PyAutoArray, verified 2026-09-10)
-----------------------------------------------------

- ``inversion/inversion/inversion_util.py`` ``reconstruction_positive_only_from``
  — the Jacobi scaling reproduced by :func:`jacobi_scaled` (``d = sqrt(diag(Q))``,
  ``D = 1/d``, ``Q_pc = D Q D``, ``q_pc = D q``, ``x = solve(Q_pc, q_pc) * D``),
  and its ``max_iter`` default of 50.
- ``util/jax_nnls.py`` ``solve_nnls`` — the configurable PDIP driver, returning
  ``(x, s, z, converged, pdip_iter)``. ``solve_nnls_primal`` (what the library
  calls) wraps it in a ``custom_vjp`` whose primal keeps only ``x``.
- ``inversion/inversion/abstract.py`` ``log_det_curvature_reg_matrix_term`` /
  ``log_det_regularization_matrix_term`` — the ``2 * sum(log(diag(cholesky(M))))``
  formula reproduced by :func:`log_det_cholesky`, on the rank-stripped blocks.

Not reproduced here: ``target_kappa`` and the relaxed-KKT backward pass. They
are gradient-side only — the forward solve this module times is identical to
the library's with them absent.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve


def jacobi_scaled(curvature_reg_matrix, data_vector):
    """Jacobi-precondition ``(F + λH) s = D``, exactly as the library does.

    Returns ``(Q_pc, q_pc, D_scale)`` with ``Q_pc = D_scale Q D_scale`` unit on
    the diagonal and ``q_pc = D_scale q``; the solution of the scaled system
    recovers the unscaled reconstruction as ``x_pc * D_scale``. ``D_scale`` is
    diagonal positive, so non-negativity is preserved.
    """
    d = jnp.sqrt(jnp.diag(curvature_reg_matrix))
    D_scale = 1.0 / d
    Q_pc = (curvature_reg_matrix * D_scale[:, None]) * D_scale[None, :]
    q_pc = data_vector * D_scale
    return Q_pc, q_pc, D_scale


def nnls_pdip(Q_pc, q_pc, max_iter=50):
    """Solve the (already scaled) NNLS system, keeping the PDIP iteration count.

    Calls ``autoarray.util.jax_nnls.solve_nnls`` — the driver — rather than
    ``solve_nnls_primal``, whose ``custom_vjp`` discards ``converged`` and
    ``pdip_iter``. The forward solve is the same ``lax.while_loop``; only the
    (unused here) differentiability is given up.

    ``max_iter`` is a Python int held as a default argument, not a traced one,
    so ``jax.jit(nnls_pdip)`` compiles it as a static loop bound.

    Returns ``(x_pc, converged, pdip_iter)``.
    """
    from autoarray.util.jax_nnls import solve_nnls

    x_pc, _, _, converged, pdip_iter = solve_nnls(Q_pc, q_pc, solver_tol=None, max_iter=max_iter)
    return x_pc, converged, pdip_iter


def nnls_pdip_one_iteration(Q_pc, q_pc):
    """One PDIP iteration (initialisation + a single ``pdip_pc_step``).

    The direct per-iteration cost, and the cross-check on dividing the
    cell-driven NNLS row by its iteration count.
    """
    return nnls_pdip(Q_pc, q_pc, max_iter=1)


def cholesky_curvature_reg(curvature_reg_matrix):
    """One Cholesky factorisation of the full ``F + λH`` the NNLS receives."""
    return jnp.linalg.cholesky(curvature_reg_matrix)


def cholesky_solve(curvature_reg_matrix, data_vector):
    """The unconstrained solve: one factorisation plus two triangular solves.

    What the reconstruction would cost if the non-negativity constraint were
    dropped — the floor an iterative (matrix-free CG) line is really competing
    against, since the NNLS row pays this cost once per PDIP iteration on a
    KKT system of the same size.
    """
    L = jnp.linalg.cholesky(curvature_reg_matrix)
    return cho_solve((L, True), data_vector)


def log_det_cholesky(matrix):
    """``2 * sum(log(diag(cholesky(M))))`` — the library's log-det formula."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(matrix))))


def log_evidence_terms(
    data,
    noise_map,
    model_data,
    reconstruction,
    reduced_indices,
    reg_reduced,
    curv_reg_reduced,
) -> dict:
    """Every term of ``-2 ln e = chi^2 + s^T H s + ln det(F+λH) - ln det(H) + noise_norm``.

    Same arithmetic as each cell's ``compute_log_evidence`` (and
    ``sparse_steps.log_evidence``), returned term by term instead of collapsed
    to one number, so a result JSON records *which* term the evidence is made
    of. ``model_data`` is passed in already built — the mapped reconstruction
    is the one piece that differs between the dense and w-tilde legs, and this
    helper is deliberately blind to which one built it.

    The regularised terms use the rank-stripped (mapper-only) blocks: the full
    ``H`` is rank-deficient by construction (non-regularised linear components
    such as an MGE lens light), so its log-det is ``-inf``.

    Returned values are Python floats — this is called **eagerly, once**, for
    the JSON, never inside a timed row.
    """
    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    s_reduced = reconstruction[reduced_indices]
    regularization_term = jnp.dot(s_reduced, jnp.dot(reg_reduced, s_reduced))

    log_det_curvature_reg = log_det_cholesky(curv_reg_reduced)
    log_det_regularization = log_det_cholesky(reg_reduced)

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
