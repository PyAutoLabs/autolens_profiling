"""Curvature-matrix kernel variants for the numba w-tilde interferometer path.

Phase 2 of the ``numba-interferometer-revisit`` epic. Every function here
assembles the *same* matrix as the phase-1 reference kernel

    ``inversion_interferometer_numba_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from``

namely ``F = Mᵀ W~ M`` with ``W~`` the translation-invariant operator stored as the
``(2Ny, 2Nx)`` offset array ``preload``. They differ only in *how* the triple product
is evaluated. ``test_parity.py`` pins each of them to the reference before any of its
timings are believed.

Common input layout
-------------------
The reference kernel consumes the mapper's dense ``[N_pix, P]`` triplet arrays and a
``[N_pix, 2]`` native index map. Every variant here consumes the same information in
one of two flat forms, built by :func:`kernel_inputs_from`:

- ``iy``, ``ix``   ``[N_pix]`` int64 — the pixel's row/column on the **unmasked extent**
  grid (``Ny x Nx``). Only *differences* of these index ``preload``, so an arbitrary
  common origin shift (native grid → extent grid) leaves ``F`` unchanged.
- ``indptr``, ``col``, ``val`` — CSR over image pixels: image pixel ``i``'s mappings are
  ``col[indptr[i]:indptr[i+1]]`` with weights ``val[...]``.
- ``flat`` ``[N_pix]`` and ``rows_nnz`` ``[nnz]`` — the extent-flat cell index
  ``iy·Nx + ix``, per image pixel and per non-zero respectively (the COO row index the
  FFT routes and ``InterferometerSparseOperator`` consume).
- ``cscptr``, ``csc_row``, ``csc_val`` — the same triplets sorted source-major, which the
  extent-grid convolution needs.

``preload`` symmetry
--------------------
``W~[i,j] = Σ_k σ_k⁻² cos(2π(Δx·u_k + Δy·v_k))`` is even in ``(Δy, Δx)``, so
``preload[-dy, -dx] == preload[dy, dx]`` (with the wrap-around negative indexing the
recovered kernel already relies on). The ``symmetric`` variant is the only one that
uses this; it is a property of the operator, not an assumption about the data.
"""

from __future__ import annotations

import numpy as np
from autoarray import numba_util

# ---------------------------------------------------------------------------
# Input marshalling
# ---------------------------------------------------------------------------


def kernel_inputs_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
) -> dict:
    """Flatten the reference kernel's argument set into the shared layout above.

    ``native_index_for_slim_index`` is re-origined onto the unmasked extent
    (``iy -= iy.min()``, ``ix -= ix.min()``) so ``Ny x Nx`` is the extent rectangle the
    FFT and direct-convolution kernels operate on. Only differences of these indexes
    are ever used, so this is a relabelling, not a change of ``F``.
    """
    native = np.asarray(native_index_for_slim_index, dtype=np.int64)
    sizes = np.asarray(pix_size_for_sub_slim_index, dtype=np.int64)
    indexes = np.asarray(pix_indexes_for_sub_slim_index, dtype=np.int64)
    weights = np.asarray(pix_weights_for_sub_slim_index, dtype=np.float64)

    n_pix = native.shape[0]

    iy = native[:, 0] - int(native[:, 0].min())
    ix = native[:, 1] - int(native[:, 1].min())

    ny = int(iy.max()) + 1
    nx = int(ix.max()) + 1

    indptr = np.zeros(n_pix + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    nnz = int(indptr[-1])

    # A real `Mapper` pads its `[N_pix, P_max]` triplet rows to the longest row, so the
    # valid entries are the first `pix_size_for_sub_slim_index[i]` of each. Selecting them
    # with a boolean mask keeps C (row-major) order, which is exactly CSR order, and keeps
    # this out of Python: the marshalling runs once per likelihood evaluation and a
    # per-pixel loop here would show up as kernel cost in the breakdown.
    valid = np.arange(indexes.shape[1], dtype=np.int64)[None, :] < sizes[:, None]

    col = np.ascontiguousarray(indexes[valid])
    val = np.ascontiguousarray(weights[valid])

    # Source-major (CSC) view of the same triplets.
    row_of_nnz = np.repeat(np.arange(n_pix, dtype=np.int64), sizes)
    order = np.argsort(col, kind="stable")
    csc_row = row_of_nnz[order]
    csc_val = val[order]
    counts = np.bincount(col, minlength=int(pix_pixels)).astype(np.int64)
    cscptr = np.zeros(int(pix_pixels) + 1, dtype=np.int64)
    np.cumsum(counts, out=cscptr[1:])

    return {
        "preload": None,  # filled by the caller
        "iy": np.ascontiguousarray(iy),
        "ix": np.ascontiguousarray(ix),
        "indptr": indptr,
        "col": col,
        "val": val,
        "cscptr": cscptr,
        "csc_row": np.ascontiguousarray(csc_row),
        "csc_val": np.ascontiguousarray(csc_val),
        "flat": np.ascontiguousarray(iy * nx + ix),
        "rows_nnz": np.ascontiguousarray(np.repeat(iy * nx + ix, sizes)),
        "n_pix": n_pix,
        "nnz": nnz,
        "ny": ny,
        "nx": nx,
        "pix_pixels": int(pix_pixels),
    }


# ---------------------------------------------------------------------------
# 2. hoisted — the reference pair loop with the row gathers hoisted
# ---------------------------------------------------------------------------


@numba_util.jit()
def curvature_hoisted(preload, iy, ix, indptr, col, val, pix_pixels):
    """The reference ``O(N² P²)`` pair loop with every loop-invariant read hoisted.

    Three hoists relative to the reference kernel, none of which changes the sum:
    the destination row ``F[sp0]`` and the ``ip0`` weight leave the ``ip1`` loop, and
    the mapper triplets are read from flat CSR arrays rather than the ``[N, P]``
    two-dimensional gathers.
    """
    n_pix = indptr.shape[0] - 1
    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    for ip0 in range(n_pix):
        y0 = iy[ip0]
        x0 = ix[ip0]

        for t0 in range(indptr[ip0], indptr[ip0 + 1]):
            sp0 = col[t0]
            w0 = val[t0]
            row = curvature_matrix[sp0]

            for ip1 in range(n_pix):
                pw = preload[iy[ip1] - y0, ix[ip1] - x0] * w0

                for t1 in range(indptr[ip1], indptr[ip1 + 1]):
                    row[col[t1]] += pw * val[t1]

    return curvature_matrix


# ---------------------------------------------------------------------------
# 3. symmetric — hoisted, upper image-pixel triangle only, with a mirror pass
# ---------------------------------------------------------------------------


@numba_util.jit()
def _curvature_symmetric_half(preload, iy, ix, indptr, col, val, pix_pixels):
    """Half of ``F``: the ``ip1 > ip0`` pairs plus half of each ``ip1 == ip0`` pair.

    ``preload`` is even in the offset, so the ``(ip1, ip0)`` pair contributes exactly
    the transpose of the ``(ip0, ip1)`` pair. Accumulating the strict upper image-pixel
    triangle plus ``0.5 x`` the diagonal pairs therefore gives a matrix ``A`` with
    ``F = A + Aᵀ`` — the mirror pass the caller applies.
    """
    n_pix = indptr.shape[0] - 1
    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    p_self = 0.5 * preload[0, 0]

    for ip0 in range(n_pix):
        y0 = iy[ip0]
        x0 = ix[ip0]

        for t0 in range(indptr[ip0], indptr[ip0 + 1]):
            sp0 = col[t0]
            w0 = val[t0]
            row = curvature_matrix[sp0]

            pw_self = p_self * w0
            for t1 in range(indptr[ip0], indptr[ip0 + 1]):
                row[col[t1]] += pw_self * val[t1]

            for ip1 in range(ip0 + 1, n_pix):
                pw = preload[iy[ip1] - y0, ix[ip1] - x0] * w0

                for t1 in range(indptr[ip1], indptr[ip1 + 1]):
                    row[col[t1]] += pw * val[t1]

    return curvature_matrix


def curvature_symmetric(preload, iy, ix, indptr, col, val, pix_pixels):
    """``F`` from the halved pair loop plus its mirror."""
    half = _curvature_symmetric_half(preload, iy, ix, indptr, col, val, pix_pixels)
    return half + half.T


# ---------------------------------------------------------------------------
# 4. two_stage — the imaging two-stage source-space accumulator, pair-loop form
# ---------------------------------------------------------------------------


@numba_util.jit()
def curvature_two_stage(preload, iy, ix, indptr, col, val, pix_pixels):
    """``F`` via a dense per-data-pixel source accumulator.

    The transplant of PyAutoArray's imaging
    ``curvature_matrix_via_sparse_operator_two_stage_from``: for each data pixel
    ``ip0`` accumulate ``acc[s1] = Σ_{ip1} W~[ip0, ip1] · A[ip1, s1]`` over the whole
    image, then AXPY ``F[s0, :] += w0 · acc`` once per mapping of ``ip0``. Cost
    ``O(N² P + N P S)`` instead of the reference's ``O(N² P²)``, and the second stage
    is a contiguous AXPY.
    """
    n_pix = indptr.shape[0] - 1
    curvature_matrix = np.zeros((pix_pixels, pix_pixels))
    acc = np.zeros(pix_pixels)

    for ip0 in range(n_pix):
        y0 = iy[ip0]
        x0 = ix[ip0]

        acc[:] = 0.0

        for ip1 in range(n_pix):
            pv = preload[iy[ip1] - y0, ix[ip1] - x0]

            for t1 in range(indptr[ip1], indptr[ip1 + 1]):
                acc[col[t1]] += pv * val[t1]

        for t0 in range(indptr[ip0], indptr[ip0 + 1]):
            sp0 = col[t0]
            w0 = val[t0]
            row = curvature_matrix[sp0]

            for s1 in range(pix_pixels):
                row[s1] += w0 * acc[s1]

    return curvature_matrix


# ---------------------------------------------------------------------------
# 5. direct_conv — the extent-grid direct convolution, one source column at a time
# ---------------------------------------------------------------------------


@numba_util.jit()
def curvature_direct_conv(
    preload, iy, ix, flat, indptr, col, val, cscptr, csc_row, csc_val, ny, nx, pix_pixels
):
    """``F`` by convolving each source column of ``A`` over the extent rectangle.

    For source column ``s``:

    1. ``u = W~ A[:, s]`` on the ``(Ny, Nx)`` extent grid — each of the ``nnz_s``
       non-zeros of the column scatters a shifted copy of the ``W~`` kernel onto ``u``.
       Splitting the row into the two contiguous halves of the wrapped ``preload`` row
       makes the inner loop a pure contiguous AXPY.
    2. ``F[s, :] = Aᵀ u`` — one gather per non-zero of the whole mapping operator.

    Cost ``O(nnz·M + S·nnz)`` with ``M = Ny·Nx``, versus the pair loop's ``O(N² P²)``:
    the convolution replaces the ``N²`` pixel-pair space with the ``M``-cell extent
    rectangle, which is the whole point of the extent-grid form.
    """
    n_pix = indptr.shape[0] - 1
    m_cells = ny * nx
    nx2 = 2 * nx

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))
    u = np.zeros(m_cells)

    for sp in range(pix_pixels):
        u[:] = 0.0

        for t in range(cscptr[sp], cscptr[sp + 1]):
            i0 = csc_row[t]
            wi = csc_val[t]
            i_y = iy[i0]
            i_x = ix[i0]
            off = nx2 - i_x

            for jy in range(ny):
                dy = jy - i_y
                base = jy * nx

                for jx in range(i_x):
                    u[base + jx] += wi * preload[dy, off + jx]

                for jx in range(i_x, nx):
                    u[base + jx] += wi * preload[dy, jx - i_x]

        row = curvature_matrix[sp]

        for i1 in range(n_pix):
            ui = u[flat[i1]]

            for t in range(indptr[i1], indptr[i1 + 1]):
                row[col[t]] += val[t] * ui

    return curvature_matrix


_PARALLEL_CACHE: dict = {}


def direct_conv_parallel_kernel():
    """Compile (once) :func:`curvature_direct_conv` with ``prange`` over source columns.

    Built lazily rather than decorated at import time for two reasons: ``numba.prange``
    has to be resolvable in the function's own scope under ``nopython``, and the
    thread-count numba bakes in is read from ``NUMBA_NUM_THREADS`` at *its* import, so
    a thread-scaling arm must set that variable before this is first called.

    Each source column owns its accumulator ``u`` and writes only its own row of ``F``,
    so the parallel loop needs neither a reduction nor a lock.
    """
    if "kernel" in _PARALLEL_CACHE:
        return _PARALLEL_CACHE["kernel"]

    import numba
    from numba import prange

    @numba.njit(cache=True, parallel=True, nogil=True)
    def _curvature_direct_conv_parallel(
        preload, iy, ix, flat, indptr, col, val, cscptr, csc_row, csc_val, ny, nx, pix_pixels
    ):
        n_pix = indptr.shape[0] - 1
        m_cells = ny * nx
        nx2 = 2 * nx

        curvature_matrix = np.zeros((pix_pixels, pix_pixels))

        for sp in prange(pix_pixels):
            u = np.zeros(m_cells)

            for t in range(cscptr[sp], cscptr[sp + 1]):
                i0 = csc_row[t]
                wi = csc_val[t]
                i_y = iy[i0]
                i_x = ix[i0]
                off = nx2 - i_x

                for jy in range(ny):
                    dy = jy - i_y
                    base = jy * nx

                    for jx in range(i_x):
                        u[base + jx] += wi * preload[dy, off + jx]

                    for jx in range(i_x, nx):
                        u[base + jx] += wi * preload[dy, jx - i_x]

            row = curvature_matrix[sp]

            for i1 in range(n_pix):
                ui = u[flat[i1]]

                for t in range(indptr[i1], indptr[i1 + 1]):
                    row[col[t]] += val[t] * ui

        return curvature_matrix

    _PARALLEL_CACHE["kernel"] = _curvature_direct_conv_parallel
    return _curvature_direct_conv_parallel


# ---------------------------------------------------------------------------
# 6. source_loop — the recovered `..._from_2` source-pixel-major variant
# ---------------------------------------------------------------------------


def source_loop_inputs_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pix_pixels: int,
):
    """The three source-major arrays ``..._from_2`` takes, as **int64** indexes.

    Two changes to the recovered ``sub_slim_indexes_for_pix_index`` helper, both
    required to wire ``..._from_2`` at all; neither changes a value it would have
    produced on inputs it handled correctly:

    1. **Integer dtypes.** The recovered helper builds all three arrays from
       ``np.zeros`` / ``np.ones``, i.e. float64, and ``..._from_2`` then uses them as
       loop bounds and array indexes — legal in the interpreter, a numba typing error
       under ``nopython``.
    2. **``pix_size_for_sub_slim_index`` is honoured.** The recovered helper iterates
       every column of the ``[N_pix, P_max]`` index array, ignoring the per-pixel size.
       A real ``Mapper`` pads short rows (a border-relocated Delaunay pixel can map to
       fewer than three vertices) and the padding is not a valid source index, so the
       recovered helper raises on the production mapper it was written for. Reading only
       the first ``size`` entries of each row is what the rest of the pack does.
    """
    indexes = np.asarray(pix_indexes_for_sub_slim_index, dtype=np.int64)
    weights = np.asarray(pix_weights_for_sub_slim_index, dtype=np.float64)
    row_sizes = np.asarray(pix_size_for_sub_slim_index, dtype=np.int64)

    n_pix = indexes.shape[0]
    pix_pixels = int(pix_pixels)

    valid = np.arange(indexes.shape[1], dtype=np.int64)[None, :] < row_sizes[:, None]

    flat_pix = indexes[valid]
    flat_weight = weights[valid]
    flat_slim = np.repeat(np.arange(n_pix, dtype=np.int64), row_sizes)

    sizes = np.bincount(flat_pix, minlength=pix_pixels).astype(np.int64)
    max_size = int(sizes.max()) if sizes.size else 0

    # Group the non-zeros by source pixel. `stable` keeps each group in ascending slim
    # order, i.e. the order the recovered helper's append loop produced.
    order = np.argsort(flat_pix, kind="stable")
    grouped_pix = flat_pix[order]

    starts = np.zeros(pix_pixels, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    slot = np.arange(grouped_pix.size, dtype=np.int64) - starts[grouped_pix]

    sub_slim_indexes = -np.ones((pix_pixels, max_size), dtype=np.int64)
    sub_slim_weights = -np.ones((pix_pixels, max_size), dtype=np.float64)

    sub_slim_indexes[grouped_pix, slot] = flat_slim[order]
    sub_slim_weights[grouped_pix, slot] = flat_weight[order]

    return sub_slim_indexes, sizes, sub_slim_weights


# ---------------------------------------------------------------------------
# 7/8. FFT convolution on the NumPy/scipy stack
# ---------------------------------------------------------------------------


def sparse_operator_from(flat, col, val, m_cells, pix_pixels):
    from scipy.sparse import csc_matrix

    return csc_matrix(
        (np.asarray(val), (np.asarray(flat), np.asarray(col))),
        shape=(int(m_cells), int(pix_pixels)),
    )


def curvature_fft_numpy(
    preload,
    flat,
    col,
    val,
    ny,
    nx,
    pix_pixels,
    *,
    real_fft: bool = True,
    batch_size: int = 128,
    workers: int = 1,
    sparse_operator=None,
):
    """``F = Aᵀ W~ A`` with ``W~`` applied as a padded circular convolution.

    ``real_fft=True`` is the ``rfft2``/``irfft2`` route; ``real_fft=False`` is the
    complex ``fft2``/``ifft2`` route, i.e. the algorithm
    ``InterferometerSparseOperator.apply_operator`` implements, so the pair isolates
    the real-FFT saving from JAX overhead.

    The sparse scatter (``A[:, block]`` densified) and the projection (``Aᵀ G``) are
    done with ``scipy.sparse``, the NumPy stack's counterpart of the JAX route's
    ``segment_sum``; only the operator application differs between the arms.
    """
    import scipy.fft as sfft

    m_cells = int(ny) * int(nx)
    if sparse_operator is None:
        sparse_operator = sparse_operator_from(flat, col, val, m_cells, pix_pixels)

    a_csc = sparse_operator
    a_csr_t = sparse_operator.T.tocsr()

    if real_fft:
        khat = sfft.rfft2(preload, workers=workers)
    else:
        khat = sfft.fft2(preload, workers=workers)

    curvature_matrix = np.zeros((int(pix_pixels), int(pix_pixels)))

    for start in range(0, int(pix_pixels), int(batch_size)):
        stop = min(start + int(batch_size), int(pix_pixels))

        block = np.asarray(a_csc[:, start:stop].todense()).T  # (B, M)
        b_size = block.shape[0]

        padded = np.zeros((b_size, 2 * ny, 2 * nx))
        padded[:, :ny, :nx] = block.reshape(b_size, ny, nx)

        if real_fft:
            spectrum = sfft.rfft2(padded, axes=(-2, -1), workers=workers)
            spectrum *= khat[None, :, :]
            convolved = sfft.irfft2(spectrum, s=(2 * ny, 2 * nx), axes=(-2, -1), workers=workers)
        else:
            spectrum = sfft.fft2(padded, axes=(-2, -1), workers=workers)
            spectrum *= khat[None, :, :]
            convolved = sfft.ifft2(spectrum, axes=(-2, -1), workers=workers).real

        g_block = convolved[:, :ny, :nx].reshape(b_size, m_cells).T  # (M, B)

        curvature_matrix[:, start:stop] = a_csr_t @ g_block

    return curvature_matrix


# ---------------------------------------------------------------------------
# 9. fft2_jax — the library's own InterferometerSparseOperator route
# ---------------------------------------------------------------------------


def jax_curvature_callable(preload, flat, col, val, ny, nx, pix_pixels, *, batch_size: int = 128):
    """A jitted callable returning ``F`` from ``InterferometerSparseOperator``.

    This is the production JAX/FFT algorithm on exactly the same synthetic inputs:
    ``from_nufft_precision_operator`` builds ``Khat = fft2(preload)`` once and
    ``curvature_matrix_diag_from`` assembles ``F`` in ``batch_size`` source-column
    blocks. The operator is a frozen dataclass rather than a pytree, so it is closed
    over (its ``Khat`` becomes a jit constant) exactly as an inversion closes over it.
    """
    import jax
    import jax.numpy as jnp
    from autoarray.inversion.inversion.interferometer.inversion_interferometer_util import (
        InterferometerSparseOperator,
    )

    m_cells = int(ny) * int(nx)

    operator = InterferometerSparseOperator.from_nufft_precision_operator(
        np.asarray(preload),
        dirty_image=np.zeros(m_cells),
        batch_size=int(batch_size),
    )

    rows = jnp.asarray(np.asarray(flat), dtype=jnp.int32)
    cols = jnp.asarray(np.asarray(col), dtype=jnp.int32)
    vals = jnp.asarray(np.asarray(val), dtype=jnp.float64)

    @jax.jit
    def _curvature(rows_, cols_, vals_):
        # `xp=jnp` keeps this on the operator's JAX branch; since PyAutoArray#544 the
        # default `xp=np` would call `np.asarray` on a tracer under `jit`.
        return operator.curvature_matrix_diag_from(rows_, cols_, vals_, S=int(pix_pixels), xp=jnp)

    def call():
        result = _curvature(rows, cols, vals)
        result.block_until_ready()
        return result

    return call
