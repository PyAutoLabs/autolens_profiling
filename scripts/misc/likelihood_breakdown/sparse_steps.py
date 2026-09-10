"""Standalone w-tilde (sparse) inversion steps, for the ``--sparse`` breakdown.

Why this module exists
----------------------

Until 2026-09-10 the ``_sparse`` breakdown rows were **dense tables with a
sparse dataset attached**: the three imaging cells called
``dataset.apply_sparse_operator()`` and then timed
``inversion.operated_mapping_matrix``,
``curvature_matrix_via_mapping_matrix_from`` and
``mapped_reconstructed_data_via_mapping_matrix_from`` — none of which the
w-tilde path ever calls. Dense and ``_sparse`` rows agreed to <1 % because they
were the same computation twice.

The functions here reproduce
``autoarray.inversion.inversion.imaging.sparse.InversionImagingSparse`` step by
step for the configuration the production-fiducial cells build: **one pixelized
``Mapper`` plus one ``AbstractLinearObjFuncList``** (the MGE-60 lens-light
basis). That is the ``func_list_and_mapper`` branch of both ``data_vector`` and
``curvature_matrix``; the pure-mapper and multi-mapper branches are not
reproduced because no profiled cell takes them.

Every function takes explicit arrays so it can be handed straight to
``timing.jit_profile`` as its own compiled program. The pieces that are *not*
part of the w-tilde formalism — the regularization matrix H, ``F + λH``, the
NNLS reconstruction and both Choleskys — are deliberately absent: they are the
same call on the same reduced matrices as the dense leg, and the cells keep
using their own dense code for them. That is the whole point of the
comparison.

Step map (dense row -> sparse row)
----------------------------------

============================== =========================================
dense                          sparse
============================== =========================================
Mapping matrix                 Sparse triplets (data + curvature)
Blurred mapping matrix (PSF)   MGE operated basis  +  PSF-weighted data
Data vector (D)                Data vector (D, w-tilde)
Curvature matrix (F)           Curvature matrix (F, w-tilde)
                                 + F diag (FFT blocks)
                                 + F off-diag (mapper x MGE)
                                 + F MGE x MGE GEMM
Mapped recon + log evidence    Mapped recon (sparse) + log evidence
============================== =========================================

The MGE operated basis stays **dense** — 60 PSF-convolved columns built by
``psf.convolved_mapping_matrix_from`` — and is reported as its own row rather
than folded away, because a matrix-free line has to beat it too.

Library provenance (PyAutoArray, verified 2026-09-10)
-----------------------------------------------------

- ``inversion/inversion/imaging/sparse.py``:
  ``_data_vector_func_list_and_mapper``, ``_curvature_matrix_func_list_and_mapper``,
  ``_mapped_reconstructed_data_dict_from``.
- ``inversion/inversion/imaging/inversion_imaging_util.py``:
  ``psf_weighted_data_from``, ``data_vector_via_psf_weighted_data_from``,
  ``data_vector_via_blurred_mapping_matrix_from``,
  ``curvature_matrix_mirrored_from``, ``curvature_matrix_with_added_to_diag_from``,
  ``mapped_reconstructed_operated_data_via_sparse_operator_from``,
  ``ImagingSparseOperator.{curvature_matrix_diag_from,
  curvature_matrix_off_diag_func_list_from}``.
- ``inversion/mappers/abstract.py``: ``sparse_triplets_data`` (slim rows) /
  ``sparse_triplets_curvature`` (rectangular FFT rows).

Mixed precision
---------------

``ImagingSparseOperator`` hard-casts its triplets and vectors to ``float64``
(``inversion_imaging_util.py``, the ``jnp.asarray(..., dtype=jnp.float64)``
lines), so ``--use-mixed-precision`` does **not** reach the w-tilde blocks. The
cells record that as ``mixed_precision_note`` in the result JSON rather than
pretending the flag applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from autoarray.inversion.inversion.imaging import inversion_imaging_util
from autoarray.inversion.mappers import mapper_util
from autoarray.structures.arrays.uniform_2d import Array2D


@dataclass(frozen=True)
class SparseContext:
    """Everything the sparse steps need that is *static* at trace time.

    Resolved once, eagerly, by :func:`sparse_context_from`. Closed over by the
    step functions rather than passed as a traced argument: the operator's FFT
    state is device-resident constant data, and the ``psf`` / ``mask`` objects
    are Python objects the JAX path treats as compile-time constants.
    """

    operator: Any  # ImagingSparseOperator
    psf: Any  # dataset.psf (Kernel2D)
    mask: Any  # dataset.mask (Mask2D)
    fft_index_for_masked_pixel: Any  # (M_pix,) int32
    total_params: int
    mapper_start: int
    mapper_end: int
    func_start: int
    func_end: int
    mapper_params: int
    n_funcs: int
    add_to_diag_value: float
    no_regularization_index_list: tuple[int, ...]

    @property
    def sparse_batch_size(self) -> int:
        return int(self.operator.batch_size)


def sparse_context_from(inversion, dataset) -> SparseContext:
    """Build a :class:`SparseContext` from an ``InversionImagingSparse``.

    Reads the param ranges, the mapper/func split and the curvature-diagonal
    settings off the inversion itself so the standalone steps assemble F and D
    into exactly the layout the library does. Raises if the inversion is not
    the one-mapper + one-func-list shape these steps reproduce.
    """
    from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
    from autoarray.inversion.mappers.abstract import Mapper

    mapper_ranges = inversion.param_range_list_from(cls=Mapper)
    func_ranges = inversion.param_range_list_from(cls=AbstractLinearObjFuncList)

    if len(mapper_ranges) != 1 or len(func_ranges) != 1:
        raise ValueError(
            "sparse_steps reproduces the func_list_and_mapper branch only "
            f"(1 Mapper + 1 AbstractLinearObjFuncList); got {len(mapper_ranges)} "
            f"mapper(s) and {len(func_ranges)} func list(s)."
        )

    mapper = inversion.cls_list_from(cls=Mapper)[0]
    linear_func = inversion.cls_list_from(cls=AbstractLinearObjFuncList)[0]
    operated = inversion.linear_func_operated_mapping_matrix_dict[linear_func]

    return SparseContext(
        operator=dataset.sparse_operator,
        psf=dataset.psf,
        mask=dataset.mask,
        fft_index_for_masked_pixel=jnp.asarray(
            dataset.mask.fft_index_for_masked_pixel, dtype=jnp.int32
        ),
        total_params=int(inversion.total_params),
        mapper_start=int(mapper_ranges[0][0]),
        mapper_end=int(mapper_ranges[0][1]),
        func_start=int(func_ranges[0][0]),
        func_end=int(func_ranges[0][1]),
        mapper_params=int(mapper.params),
        n_funcs=int(operated.shape[1]),
        add_to_diag_value=float(inversion.settings.no_regularization_add_to_curvature_diag_value),
        no_regularization_index_list=tuple(int(i) for i in inversion.no_regularization_index_list),
    )


# ---------------------------------------------------------------------------
# Setup rows — the sparse replacements for "Mapping matrix" / "Blurred
# mapping matrix (PSF)"
# ---------------------------------------------------------------------------


def sparse_triplets(
    pix_indexes_for_sub,
    pix_weights_for_sub,
    slim_index_for_sub,
    fft_index_for_masked_pixel,
    sub_fraction_slim,
):
    """Both triplet representations of the mapper's sparse mapping operator.

    Mirrors ``Mapper.sparse_triplets_data`` (rows in **slim masked** pixel
    indices, consumed by the data vector) and
    ``Mapper.sparse_triplets_curvature`` (rows converted to **rectangular FFT
    grid** indices, consumed by F and the model image). ``cols``/``vals`` are
    identical between the two, but both calls are timed because both run in
    the library — the second is what the two curvature blocks and the mapped
    reconstruction all read.

    Returns ``(rows_data, cols, vals, rows_curvature)``.
    """
    rows_data, cols, vals = mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=pix_indexes_for_sub,
        pix_weights_for_sub=pix_weights_for_sub,
        slim_index_for_sub=slim_index_for_sub,
        fft_index_for_masked_pixel=fft_index_for_masked_pixel,
        sub_fraction_slim=sub_fraction_slim,
        xp=jnp,
    )
    rows_curvature, _, _ = mapper_util.sparse_triplets_from(
        pix_indexes_for_sub=pix_indexes_for_sub,
        pix_weights_for_sub=pix_weights_for_sub,
        slim_index_for_sub=slim_index_for_sub,
        fft_index_for_masked_pixel=fft_index_for_masked_pixel,
        sub_fraction_slim=sub_fraction_slim,
        xp=jnp,
        return_rows_slim=False,
    )
    return rows_data, cols, vals, rows_curvature


def psf_weighted_data(weight_map_native, kernel_native, native_index_for_slim_index):
    """The ``psf_weighted_data`` vector: PSF correlated with ``data / noise**2``.

    ``InversionImagingSparse.psf_weighted_data`` — the once-per-call precompute
    that lets the mapper block of D be formed without a mapping matrix. The
    weight map comes off the dataset's ``sparse_operator``; it is built from the
    dataset's own data, which equals the profile-subtracted image whenever every
    lens-light component is linear (the MGE-60 fiducial).
    """
    return inversion_imaging_util.psf_weighted_data_from(
        weight_map_native=weight_map_native,
        kernel_native=kernel_native,
        native_index_for_slim_index=native_index_for_slim_index,
        xp=jnp,
    )


# ---------------------------------------------------------------------------
# Data vector (D)
# ---------------------------------------------------------------------------


def data_vector(
    ctx: SparseContext,
    psf_weighted_data_vector,
    rows_data,
    cols,
    vals,
    operated_mapping_matrix,
    image,
    noise_map,
):
    """D assembled exactly as ``_data_vector_func_list_and_mapper`` does.

    The mapper block is ``data_vector_via_psf_weighted_data_from`` on the slim
    triplets; the MGE block is ``data_vector_via_blurred_mapping_matrix_from``
    on the dense operated basis, against the inversion's own ``data`` (the
    profile-subtracted image) — not against ``psf_weighted_data``.
    """
    out = jnp.zeros(ctx.total_params, dtype=jnp.float64)

    mapper_block = inversion_imaging_util.data_vector_via_psf_weighted_data_from(
        psf_weighted_data=psf_weighted_data_vector,
        rows=rows_data,
        cols=cols,
        vals=vals,
        S=ctx.mapper_params,
    )
    out = out.at[ctx.mapper_start : ctx.mapper_end].set(mapper_block)

    func_block = inversion_imaging_util.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=operated_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )
    return out.at[ctx.func_start : ctx.func_end].set(func_block)


# ---------------------------------------------------------------------------
# Curvature matrix (F) — one row plus three sub-rows
# ---------------------------------------------------------------------------


def curvature_diag(ctx: SparseContext, rows_curvature, cols, vals):
    """F's mapper x mapper block: ``A^T W A`` via the FFT column-block loop.

    ``ImagingSparseOperator.curvature_matrix_diag_from`` — a
    ``lax.fori_loop`` over ``ceil(S / batch_size)`` blocks of source columns,
    each one an rfft2 forward blur, an inverse-variance weighting and an
    irfft2 backprojection. This is the sub-row a matrix-free CG replaces.
    """
    return ctx.operator.curvature_matrix_diag_from(
        rows=rows_curvature,
        cols=cols,
        vals=vals,
        S=ctx.mapper_params,
    )


def curvature_off_diag_func_list(
    ctx: SparseContext, rows_curvature, cols, vals, operated_mapping_matrix, noise_map
):
    """F's mapper x MGE block: ``A^T H^T (H B / noise**2)``.

    ``ImagingSparseOperator.curvature_matrix_off_diag_func_list_from``. One
    batched flipped-PSF convolution over the ``n_funcs`` columns (60 here), so
    it is cheap next to the diagonal block's S-column sweep.
    """
    curvature_weights = operated_mapping_matrix / noise_map[:, None] ** 2
    return ctx.operator.curvature_matrix_off_diag_func_list_from(
        curvature_weights=curvature_weights,
        fft_index_for_masked_pixel=ctx.fft_index_for_masked_pixel,
        rows=rows_curvature,
        cols=cols,
        vals=vals,
        S=ctx.mapper_params,
    )


def curvature_func_func(operated_mapping_matrix, noise_map):
    """F's MGE x MGE block: the ``(N, 60)^T (N, 60)`` GEMM on ``B / noise``.

    The one block of the w-tilde F that is a plain dense matmul; the library
    forms each weighted matrix once and mirrors the upper triangle.
    """
    weighted = operated_mapping_matrix / noise_map[:, None]
    return jnp.dot(weighted.T, weighted)


def curvature_matrix(
    ctx: SparseContext, rows_curvature, cols, vals, operated_mapping_matrix, noise_map
):
    """The whole of ``InversionImagingSparse.curvature_matrix``, one program.

    Assembles the three blocks into ``(total_params, total_params)``, then
    applies ``curvature_matrix_mirrored_from`` and
    ``curvature_matrix_with_added_to_diag_from`` exactly as the property does.
    Timed as one row so it is directly comparable with the dense leg's
    "Curvature matrix (F)"; the three sub-rows above say where inside it the
    time goes.
    """
    F = jnp.zeros((ctx.total_params, ctx.total_params), dtype=jnp.float64)

    diag = curvature_diag(ctx, rows_curvature, cols, vals)
    F = F.at[ctx.mapper_start : ctx.mapper_end, ctx.mapper_start : ctx.mapper_end].set(diag)

    off_diag = curvature_off_diag_func_list(
        ctx, rows_curvature, cols, vals, operated_mapping_matrix, noise_map
    )
    F = F.at[ctx.mapper_start : ctx.mapper_end, ctx.func_start : ctx.func_end].set(off_diag)

    func_block = curvature_func_func(operated_mapping_matrix, noise_map)
    F = F.at[ctx.func_start : ctx.func_end, ctx.func_start : ctx.func_end].set(func_block)

    F = inversion_imaging_util.curvature_matrix_mirrored_from(curvature_matrix=F, xp=jnp)

    if len(ctx.no_regularization_index_list) > 0:
        F = inversion_imaging_util.curvature_matrix_with_added_to_diag_from(
            curvature_matrix=F,
            value=ctx.add_to_diag_value,
            no_regularization_index_list=list(ctx.no_regularization_index_list),
            xp=jnp,
        )

    return F


# ---------------------------------------------------------------------------
# Model image + log evidence
# ---------------------------------------------------------------------------


def mapped_reconstructed_operated_data(
    ctx: SparseContext, reconstruction, rows_curvature, cols, vals, operated_mapping_matrix
):
    """The PSF-convolved model image, w-tilde style.

    ``_mapped_reconstructed_data_dict_from(use_operated_for_linear_func=True)``:
    the mapper's contribution is scattered onto the rectangular grid by
    ``mapped_reconstructed_operated_data_via_sparse_operator_from`` and then
    convolved by the PSF (the w-tilde path has no mapping matrix to blur
    column-wise), while the MGE's contribution is
    ``sum(reconstruction * operated_mapping_matrix, axis=1)`` on the already
    convolved basis. Their sum is the dense leg's
    ``operated_mapping_matrix @ reconstruction``.
    """
    mapper_recon = reconstruction[ctx.mapper_start : ctx.mapper_end]
    func_recon = reconstruction[ctx.func_start : ctx.func_end]

    mapped = inversion_imaging_util.mapped_reconstructed_operated_data_via_sparse_operator_from(
        reconstruction=mapper_recon,
        rows=rows_curvature,
        cols=cols,
        vals=vals,
        fft_index_for_masked_pixel=ctx.fft_index_for_masked_pixel,
        data_shape=ctx.mask.shape_native,
    )
    mapped = ctx.psf.convolved_image_from(
        image=Array2D(values=mapped, mask=ctx.mask),
        blurring_image=None,
        xp=jnp,
    ).array

    mapped_func = jnp.sum(func_recon * operated_mapping_matrix, axis=1)

    return mapped + mapped_func


def log_evidence(
    ctx: SparseContext,
    data,
    noise_map,
    blurred_image,
    reconstruction,
    rows_curvature,
    cols,
    vals,
    operated_mapping_matrix,
    reduced_indices,
    reg_reduced,
    curv_reg_reduced,
):
    """The five-term log evidence with the model image built the sparse way.

    Everything except the model image is byte-for-byte the dense cells'
    ``compute_log_evidence``:

        -2 ln e = chi^2 + s^T H s + ln det(F + H) - ln det(H) + noise_norm

    with ``s^T H s`` and both log-dets on the rank-stripped (mapper-only)
    blocks, and the log-dets as ``2 * sum(log(diag(cholesky(M))))`` to match
    ``Inversion.log_det_*``. That the two log-det terms are the *same* call on
    the *same* reduced matrices in both legs is the reason the sparse row is a
    fair comparator for a matrix-free line.
    """
    mapped_recon = mapped_reconstructed_operated_data(
        ctx, reconstruction, rows_curvature, cols, vals, operated_mapping_matrix
    )

    model_data = blurred_image + mapped_recon

    residual = data - model_data
    chi_squared = jnp.sum((residual / noise_map) ** 2)

    s_reduced = reconstruction[reduced_indices]
    regularization_term = jnp.dot(s_reduced, jnp.dot(reg_reduced, s_reduced))

    L_cr = jnp.linalg.cholesky(curv_reg_reduced)
    log_det_curvature_reg = 2.0 * jnp.sum(jnp.log(jnp.diag(L_cr)))
    L_r = jnp.linalg.cholesky(reg_reduced)
    log_det_regularization = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))

    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map**2))

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_det_curvature_reg
        - log_det_regularization
        + noise_normalization
    )
