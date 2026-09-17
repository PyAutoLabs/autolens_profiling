"""Curvature-matrix kernels for the numba CPU fixed-lens-light cell — jax-free.

Why this module exists
----------------------

``curvature_matrix_via_sparse_operator_from`` (PyAutoArray
``inversion/inversion/imaging_numba/inversion_imaging_numba_util.py:542``) is a
**dispatcher**. It picks between two kernels with identical inputs, outputs and
contracts on a single pixel-count cap::

    if pix_pixels <= two_stage_max_pix_pixels:
        return curvature_matrix_via_sparse_operator_two_stage_from(...)
    return curvature_matrix_via_sparse_operator_direct_from(...)

``CURVATURE_TWO_STAGE_MAX_PIX_PIXELS = 4096``, and the source spaces this
campaign measures are 1500 pixels, so **production always takes the two-stage
branch**. That cap was calibrated on the HST *rectangular* fiducial — bilinear
mappings, ``u0 = 4``, where two-stage wins 2.9x — and the two-stage kernel's own
docstring says it can lose on barycentric Delaunay (``u0 ~ 1.55``), because
stage 2 does a dense ``pix_pixels``-long AXPY per mapping of every data pixel and
then re-zeroes the whole accumulator regardless of how few indices stage 1
actually touched. It has never been measured on the Delaunay fixed-light cell.

Nothing in the library reaches that choice from the outside. The inversion calls
the dispatcher by **module attribute** at ``imaging_numba/sparse.py:385``, with
the seven keywords and **without** ``two_stage_max_pix_pixels``; there is no
setting, no environment variable and no config key. So the seam is the same one
``fixed_light_numpy_solvers.numpy_solver_injected`` uses for the solver: rebind
the module attribute for the duration of a row, count the calls, restore by
identity.

The three kernels
-----------------

``two_stage``
    The library dispatcher with ``two_stage_max_pix_pixels=10**9``. This is
    **what production runs today** at any source size this campaign measures, and
    it is the A/B's control. Injecting it rather than leaving route ``b``
    un-patched is deliberate: route ``b`` is the unpatched production path and
    stays that way, and ``b`` vs ``b`` + this control is how the injection's own
    cost is read off the table.

``direct``
    The library dispatcher with ``two_stage_max_pix_pixels=0``, i.e. the
    library's own quadruple loop, which production only reaches above 4096
    source pixels. The two agree to floating-point reassociation (~4e-13 on the
    production HST geometries), not bit-identically: the two-stage form sums the
    same products in a different order.

``two_stage_touched``
    The candidate. The two-stage kernel with stage 2 and the re-zero restricted
    to the source indices stage 1 actually touched, tracked in an ``int64``
    index list behind an ``int8`` seen-flag. Everything else — the loop nest,
    the per-element summation order, the halved-diagonal / ``A + A.T`` tail — is
    the library's, verbatim.

Why ``two_stage_touched`` is expected BIT-IDENTICAL to ``two_stage``
-------------------------------------------------------------------

It is not a numerical claim about tolerances; it is a claim that the removed
arithmetic is an exact no-op, and it rests on three facts:

1. The entries stage 2 skips hold **exactly** ``+0.0``. ``source_accumulator``
   is allocated with ``np.zeros`` and every touched entry is written back to
   ``0.0`` before the next data pixel, so an untouched entry is the literal
   ``+0.0`` and never a denormal.
2. ``x + w * (+0.0)`` is exactly ``x`` in IEEE-754 for every finite ``x`` and
   finite ``w``. ``w * (+0.0)`` is ``+0.0`` for ``w >= 0`` and ``-0.0`` for
   ``w < 0``, and ``x + (±0.0) == x`` for every ``x`` **except** ``x == -0.0``
   with ``+0.0``, which gives ``+0.0``. ``curvature_matrix`` starts at ``+0.0``
   and ``+0.0 + y`` is ``y`` for ``y != -0.0`` and ``+0.0`` for ``y == -0.0``,
   so no entry is ever ``-0.0`` and that one exception cannot arise.
3. The order of the surviving additions is unchanged. Each
   ``curvature_row[pix_1]`` is its own memory location, visited at most once per
   ``(data_0, pix_0_index)`` iteration, and the ``data_0`` loop runs in the same
   order in both kernels — so for any one entry of ``F`` the *sequence* of values
   accumulated into it is identical, and floating-point addition's
   non-associativity has nothing to bite on.

That is the expectation. It is **asserted, not assumed**: the seam's unit tests
compare ``two_stage_touched`` against ``two_stage`` with ``np.array_equal`` on a
real fixture, and the s4 witness does the same on the production HST Delaunay
``N=1500`` system and records ``bit_identical`` explicitly rather than only
whether a tolerance cleared. A pass at 1e-15 and a pass at exact equality are
different facts about the kernel, and only the second one licenses promoting it
into ``curvature_matrix_via_sparse_operator_two_stage_from`` without touching the
library's existing reference test.

**Nothing here may import jax** — at module level or inside a function. The cell
that uses it asserts ``"jax" not in sys.modules``.
"""

from __future__ import annotations

import contextlib

import numpy as np
from autoarray import numba_util
from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util

__all__ = [
    "KERNELS",
    "KERNEL_DOTTED_NAMES",
    "LIBRARY_CURVATURE_DISPATCHER_DOTTED",
    "TWO_STAGE_ALWAYS",
    "TWO_STAGE_NEVER",
    "curvature_kernel_injected",
    "curvature_matrix_via_sparse_operator_two_stage_touched_from",
]

#: The dotted name this module's injection seam rebinds. Recorded in every row
#: that runs an injected kernel, so a reader never has to infer which attribute
#: was patched.
LIBRARY_CURVATURE_DISPATCHER_DOTTED = (
    "autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util."
    "curvature_matrix_via_sparse_operator_from"
)

#: ``two_stage_max_pix_pixels`` values that force each branch of the library
#: dispatcher. Written as constants rather than inline literals because they are
#: the whole content of the ``two_stage`` and ``direct`` kernels: everything else
#: about those two rows is the library's own code.
TWO_STAGE_ALWAYS = 10**9
TWO_STAGE_NEVER = 0


# ===================================================================
# (a) The candidate kernel
# ===================================================================


@numba_util.jit()
def curvature_matrix_via_sparse_operator_two_stage_touched_from(
    psf_precision_operator: np.ndarray,
    psf_precision_indexes: np.ndarray,
    psf_precision_lengths: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """The library's two-stage kernel, with stage 2 restricted to touched indices.

    A verbatim copy of
    ``inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_two_stage_from``
    (PyAutoArray ``:734``) with one change, in two places:

    * stage 1 records each **newly** touched source index in ``touched`` behind an
      ``int8`` ``seen`` flag (one branch per scatter it already performs);
    * stage 2's AXPY and the accumulator re-zero run over ``touched[:n_touched]``
      instead of over all ``pix_pixels``.

    The cost stage 2 pays in the library's form is
    ``(sum_data u0 + data_pixels) * pix_pixels`` dense operations per call,
    whatever the geometry. On barycentric Delaunay at ``pix_pixels = 1500`` the
    PSF overlap of one data pixel reaches only a small fraction of the source
    space, so nearly all of that is arithmetic on exact zeros — which is what
    this kernel skips, and only that.

    It is expected to be **bit-identical** to the library kernel. See the module
    docstring for why, and the seam's tests and the s4 witness for where that is
    asserted rather than assumed. The halved-diagonal / ``A + A.T`` contract and
    the parameters are exactly the library's.
    """

    data_pixels = psf_precision_lengths.shape[0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    # The per-data-pixel source-space accumulator, allocated once. Only the
    # entries this data pixel touched are written back to zero below, so the
    # whole array is `+0.0` on entry to every data pixel exactly as it is in the
    # library kernel — which is the premise the bit-identity argument rests on.
    source_accumulator = np.zeros(pix_pixels)

    # `seen[p]` is 1 while `p` is in `touched[:n_touched]`. An int8 flag array
    # rather than a set: the whole point of the kernel is to stay in L1, and at
    # pix_pixels = 1500 this is 1.5 kB beside the 12 kB accumulator.
    seen = np.zeros(pix_pixels, dtype=np.int8)
    touched = np.zeros(pix_pixels, dtype=np.int64)

    curvature_index = 0

    for data_0 in range(data_pixels):
        pair_length_0 = psf_precision_lengths[data_0]
        pix_lengths_0 = pix_lengths[data_0]

        if pix_lengths_0 == 0:
            # No mappings for this data pixel: it contributes nothing, but its
            # stored pairs must still be stepped over.
            curvature_index += pair_length_0
            continue

        # -- stage 1: accumulate the dense source-space vector, recording which
        #    entries of it were reached -------------------------------------
        n_touched = 0

        for data_1_index in range(pair_length_0):
            data_1 = psf_precision_indexes[curvature_index]
            psf_precision_value = psf_precision_operator[curvature_index]

            curvature_index += 1

            pix_row_1 = data_to_pix_unique[data_1]
            weight_row_1 = data_weights[data_1]

            for pix_1_index in range(pix_lengths[data_1]):
                pix_1 = pix_row_1[pix_1_index]
                if seen[pix_1] == 0:
                    seen[pix_1] = 1
                    touched[n_touched] = pix_1
                    n_touched += 1
                source_accumulator[pix_1] += weight_row_1[pix_1_index] * psf_precision_value

        # -- stage 2: one gathered AXPY per mapping of data_0, over the touched
        #    indices only ----------------------------------------------------
        pix_row_0 = data_to_pix_unique[data_0]
        weight_row_0 = data_weights[data_0]

        for pix_0_index in range(pix_lengths_0):
            data_0_weight = weight_row_0[pix_0_index]
            curvature_row = curvature_matrix[pix_row_0[pix_0_index]]

            for touched_index in range(n_touched):
                pix_1 = touched[touched_index]
                curvature_row[pix_1] += data_0_weight * source_accumulator[pix_1]

        for touched_index in range(n_touched):
            pix_1 = touched[touched_index]
            source_accumulator[pix_1] = 0.0
            seen[pix_1] = 0

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[i, j] += curvature_matrix[j, i]

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


# ===================================================================
# (b) The three kernels, as the seam installs them
# ===================================================================
#
# Each entry of `KERNELS` is a BUILDER: it is handed the dispatcher the seam is
# about to replace and returns the callable that stands in for it. The two
# library branches need it (they are the library's own code with one keyword
# supplied); the candidate does not, and ignores it. A uniform interface keeps
# `curvature_kernel_injected` from having to know which kind it is installing.


def _two_stage_builder(original):
    """The library dispatcher, forced onto its two-stage branch."""

    def kernel(
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
        pix_pixels,
    ):
        return original(
            psf_precision_operator=psf_precision_operator,
            psf_precision_indexes=psf_precision_indexes,
            psf_precision_lengths=psf_precision_lengths,
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=pix_pixels,
            two_stage_max_pix_pixels=TWO_STAGE_ALWAYS,
        )

    return kernel


def _direct_builder(original):
    """The library dispatcher, forced onto its direct quadruple loop."""

    def kernel(
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
        pix_pixels,
    ):
        return original(
            psf_precision_operator=psf_precision_operator,
            psf_precision_indexes=psf_precision_indexes,
            psf_precision_lengths=psf_precision_lengths,
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=pix_pixels,
            two_stage_max_pix_pixels=TWO_STAGE_NEVER,
        )

    return kernel


def _two_stage_touched_builder(_original):
    """The candidate. The dispatcher it replaces is not consulted."""

    def kernel(
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
        pix_pixels,
    ):
        return curvature_matrix_via_sparse_operator_two_stage_touched_from(
            psf_precision_operator,
            psf_precision_indexes,
            psf_precision_lengths,
            data_to_pix_unique,
            data_weights,
            pix_lengths,
            pix_pixels,
        )

    return kernel


#: The kernels this seam can install, by name.
KERNELS = {
    "two_stage": _two_stage_builder,
    "direct": _direct_builder,
    "two_stage_touched": _two_stage_touched_builder,
}

#: What each name actually runs, as a dotted path. This is the string the cell
#: and the witness record: "two_stage" is not a kernel, it is a branch of the
#: library dispatcher, and a row that only said "two_stage" would leave a reader
#: to guess which function's milliseconds they are reading.
KERNEL_DOTTED_NAMES = {
    "two_stage": (
        "autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util."
        "curvature_matrix_via_sparse_operator_two_stage_from"
    ),
    "direct": (
        "autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util."
        "curvature_matrix_via_sparse_operator_direct_from"
    ),
    "two_stage_touched": (
        "likelihood_breakdown.fixed_light_numpy_kernels."
        "curvature_matrix_via_sparse_operator_two_stage_touched_from"
    ),
}


# ===================================================================
# (c) The injection seam
# ===================================================================


@contextlib.contextmanager
def curvature_kernel_injected(kernel_name: str, *, label: str):
    """Rebind the library's curvature-matrix dispatcher to one named kernel.

    The mirror of ``fixed_light_numpy_solvers.numpy_solver_injected``, one level
    down: that one replaces the positive-only *solve*, this one replaces the
    mapper-block *assembly*. Both rebind a module attribute, because both of the
    library call sites resolve their target by attribute at call time — this
    one at ``imaging_numba/sparse.py:385``, inside
    ``_curvature_matrix_mapper_diag``, with the seven keywords and without
    ``two_stage_max_pix_pixels``.

    ``kernel_name`` must be a key of :data:`KERNELS`. There is no "pass a
    callable" form on purpose: the A/B this seam exists for compares three named
    kernels, and a row whose JSON said only "some function" would not be a row.

    Yields a mutable dict, ``{"n_calls": n, "kernel": kernel_name,
    "dotted_name": ..., "label": label, "last_pix_pixels": int | None}``. The
    counter exists to be **asserted**: an injection that never fired would report
    the unpatched route's timing wearing the candidate's label, and on this
    library it would fire zero times the moment the inversion stops calling the
    dispatcher by module attribute.

    Restoration is by identity. On exit the attribute must still be the wrapper
    this manager installed; if something else has rebound it in the meantime the
    original is *not* restored over the top — that is a bug in the caller's
    nesting, and silently winning the race would hide it.

    Install it **outside** ``call_accounting.install``, for the same reason the
    solver seam is: the accounting wrapper closes over whatever it finds, so
    installed outside, the decomposition's ``sparse_numba.curvature_matrix`` site
    attributes the injected kernel; installed inside, ``uninstall()`` would
    restore the library dispatcher over the injection and the row would measure
    the wrong kernel. ``__wrapped__``, ``__name__`` and ``__doc__`` are carried
    over so a site that introspects the callable still resolves it.
    """
    if kernel_name not in KERNELS:
        raise KeyError(
            f"curvature_kernel_injected({label!r}): unknown kernel {kernel_name!r}. "
            f"Known kernels: {', '.join(sorted(KERNELS))}."
        )

    original = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from
    kernel = KERNELS[kernel_name](original)

    counts: dict = {
        "n_calls": 0,
        "kernel": kernel_name,
        "dotted_name": KERNEL_DOTTED_NAMES[kernel_name],
        "label": label,
        "last_pix_pixels": None,
    }

    def patched(
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
        pix_pixels,
        two_stage_max_pix_pixels=None,
    ):
        # `two_stage_max_pix_pixels` is accepted and DROPPED. The inversion never
        # passes it (`sparse.py:385`), and a caller that does is asking the
        # dispatcher to choose — which is exactly the decision this injection has
        # taken over for the duration of the row.
        counts["n_calls"] += 1
        counts["last_pix_pixels"] = int(pix_pixels)
        return kernel(
            psf_precision_operator,
            psf_precision_indexes,
            psf_precision_lengths,
            data_to_pix_unique,
            data_weights,
            pix_lengths,
            pix_pixels,
        )

    patched.__wrapped__ = original
    patched.__name__ = getattr(original, "__name__", "curvature_matrix_via_sparse_operator_from")
    patched.__doc__ = getattr(original, "__doc__", None)

    inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from = patched
    try:
        yield counts
    finally:
        current = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from
        if current is not patched:
            raise RuntimeError(
                f"curvature_kernel_injected({label!r}): "
                f"{LIBRARY_CURVATURE_DISPATCHER_DOTTED} was rebound by something else "
                f"while the injection was open (found {current!r}, expected the injected "
                f"wrapper). Restoring the original here would clobber whatever holds it "
                f"now; fix the nesting instead."
            )
        inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from = original
