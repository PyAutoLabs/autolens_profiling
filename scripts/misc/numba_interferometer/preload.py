"""The once-per-dataset quantities the numba w-tilde interferometer path needs.

``NumbaPreload`` is the pack's replacement for the deleted
``autoarray.dataset.interferometer.w_tilde.WTildeInterferometer`` dataclass
(removed by PyAutoArray ``222ee046``, 2026-02-02). It carries exactly the four
model-independent arrays the recovered kernels consume:

- ``curvature_preload``           ``[2Ny, 2Nx]`` — the translation-invariant
  ``W~ = Re(Fᴴ W F)`` operator, one entry per pixel offset.
- ``dirty_image``                 ``[N_pix]``    — ``d~ = Re(Fᴴ W d)``.
- ``real_space_mask``             the mask the two above live on.
- ``native_index_for_slim_index`` ``[N_pix, 2]`` — the slim → native (y, x)
  index map the scatter kernel differences to index ``curvature_preload``.

Both construction routes produce the same object; they differ only in which
implementation builds ``curvature_preload``:

- :meth:`NumbaPreload.from_sparse_operator` — the modern NumPy builder
  ``nufft_precision_operator_via_np_from``, i.e. the array today's
  ``dataset.apply_sparse_operator()`` was itself built from, with the dirty
  image read straight off ``dataset.sparse_operator`` (no recomputation).
- :meth:`NumbaPreload.via_numba` — the recovered numba kernel
  ``w_tilde_curvature_preload_interferometer_from``. Phase 3 of the epic times
  this against the modern builder; phase 1 only pins that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from autoarray.inversion.inversion.interferometer import inversion_interferometer_util

from numba_interferometer import inversion_interferometer_numba_util as numba_util_pack


def _preload_inputs_from(dataset) -> dict:
    """The four arguments both preload builders take, read off the dataset.

    These are exactly the arguments ``Interferometer.psf_precision_operator_from``
    passes today and ``Interferometer.apply_w_tilde()`` passed at ``0b90c401``.

    ``uv_wavelengths`` is read off the transformer rather than the dataset, because an
    inversion's ``dataset`` may be a ``DatasetInterface`` (the object PyAutoLens hands the
    inversion), which carries the transformer but no ``uv_wavelengths`` attribute.
    """
    mask = dataset.transformer.grid.mask

    return {
        "noise_map_real": np.asarray(dataset.noise_map.array.real, dtype=np.float64),
        "uv_wavelengths": np.asarray(dataset.transformer.uv_wavelengths, dtype=np.float64),
        "shape_masked_pixels_2d": mask.shape_native_masked_pixels,
        "grid_radians_2d": np.asarray(
            mask.derive_grid.all_false.in_radians.native.array, dtype=np.float64
        ),
    }


def preload_inputs_from(dataset) -> dict:
    """Public alias of :func:`_preload_inputs_from`.

    The four builder arguments are the profiling harness's unit of work — phase 3
    times four independent implementations of the *same* map from these four arrays
    to ``[2Ny, 2Nx]`` — so a script needs to read them off a dataset once and hand
    them to every builder. That is a public need, not a private one.
    """
    return _preload_inputs_from(dataset)


def nufft_preload_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    eps: float = 1.0e-12,
    chunk_size: int | None = None,
) -> np.ndarray:
    """The ``W~`` preload built as the real part of a **type-1 (adjoint) NUFFT**.

    Same signature family, and the same return value, as the brute-force builders
    ``nufft_precision_operator_via_np_from`` / ``_via_jax_from`` and the recovered
    numba ``w_tilde_curvature_preload_interferometer_from`` — but ``O(K·nspread² +
    M log M)`` instead of ``O(N_pix·K)``, where ``M = 4·Ny·Nx`` is the doubled
    offset grid. At alma (``K = 1e6``, ``N_pix = 15380``) that is the difference
    between minutes and seconds.

    The construction
    ----------------
    The brute-force builders compute, over the mask's bounding extent
    ``(Ny, Nx) = shape_native_masked_pixels``::

        P[i, j] = Σ_k w_k cos(2π(dx·u_k + dy·v_k)),   w_k = 1/σ_k²

    with ``dx = −j·Δ_rad`` and ``dy = +i·Δ_rad`` on autoarray's radian grid (native
    ``y`` *decreases* down the rows, ``x`` increases along the columns), the four
    quadrants filled from the four corners so that offset ``0`` sits at ``[0, 0]``
    and negative offsets sit at negative indices (wraparound / FFT ordering), with
    the middle row ``Ny`` and column ``Nx`` left zero as padding.

    Writing ``x_k = 2π u_k Δ_rad`` and ``y_k = 2π v_k Δ_rad`` — the transformer's own
    scaled frequencies (``transformer.py:334-335``) — that is exactly

        P[i, j] = Re Σ_k w_k exp(i(−j·x_k + i·y_k))

    i.e. the real part of a type-1 NUFFT of the weights onto the ``(2Ny, 2Nx)``
    mode grid. ``nufftax.nufft2d1(x, y, c, n_modes=(N1, N2), eps, isign)`` returns
    ``f[m2, m1] = Σ_k c_k exp(isign·i(m1·x_k + m2·y_k))`` on the **centred** mode
    grid, shape ``(N2, N1)``, so the mapping is

        f = nufft2d1(−x, y, w, n_modes=(2Nx, 2Ny), eps, isign=+1)
        P = ifftshift(Re f);  P[Ny, :] = 0;  P[:, Nx] = 0

    Why this mapping and not one of the other seven
    -----------------------------------------------
    Pinned empirically at sma against ``nufft_precision_operator_via_np_from``
    (``test_parity.py::test_preload_via_nufft_matches_modern_np_builder``). Of the
    eight candidates (axis swap × sign of ``x`` × sign of ``y``) exactly two agree
    with the brute force — ``(−x, +y)`` above and ``(+x, −y)`` — at
    ``max|Δ| = 1.7e-17``, i.e. ``8.7e-14`` of the peak ``P[0, 0]``. The other six
    are wrong by ``2.1e-1`` of the peak, so the identification is not marginal: the
    discrimination is thirteen orders of magnitude.

    The two survivors are the *same* construction. ``w`` is real, so
    ``f(−x, +y)`` and ``f(+x, −y)`` are complex conjugates and their real parts are
    identical; ``(−x, +y)`` is kept because it is the one that reads off the formula
    above term by term. That degeneracy is also why ``P[i, j] == P[−i, −j]``
    (cosine evenness) holds — pinned separately.

    ``ifftshift`` vs ``fftshift`` is likewise not a choice here: both axes have even
    length ``2N``, and for even ``N`` the two shifts are the same permutation. The
    canonical ``ifftshift`` (centred → wraparound) is used because that is the
    direction the transform actually goes.

    The padding row/column is at index ``Ny`` / ``Nx``, not ``Ny-1`` / ``Nx-1``:
    after ``ifftshift`` index ``Ny`` carries mode ``−Ny``, the Nyquist mode, which
    the brute force never evaluates (its quadrants span offsets ``−(Ny−1) … Ny−1``).
    The NUFFT *does* return a value there, so it is zeroed explicitly.

    Accuracy
    --------
    ``eps`` is the NUFFT's requested precision and the error is **peak-scaled**, not
    elementwise-relative: a type-1 NUFFT bounds ``max|Δ|`` against ``Σ_k |c_k|``, so
    the near-zero entries of ``P`` — five orders below its peak — carry no relative
    accuracy guarantee at all. At sma with ``eps = 1e-12`` the measured
    ``max|Δ| = 1.7e-17`` is already the fp64 round-off floor (``eps = 1e-14`` only
    reaches ``1.4e-17``) yet the worst *elementwise* relative error is ``6.0e-10``,
    on 32 of 19600 entries. The sma pin is therefore **mixed** —
    ``rtol = 1e-10`` with ``atol = 1e-10 · P[0, 0]`` — which keeps a full relative
    test on every entry that carries signal and puts an absolute floor under the
    ones sitting at the fp64 noise floor, where a relative test measures round-off
    rather than the builder. At alma the pin is the peak-scaled bound
    ``max|Δ| ≤ 10·eps·P[0, 0]``. Never ``rtol`` with ``atol = 0``.

    Parameters
    ----------
    noise_map_real
        ``[K]`` real noise map; ``w = 1/σ²``.
    uv_wavelengths
        ``[K, 2]`` ``(u, v)`` baselines in wavelengths.
    shape_masked_pixels_2d
        ``(Ny, Nx)`` — the mask's bounding extent, ``mask.shape_native_masked_pixels``.
    grid_radians_2d
        ``[ny, nx, 2]`` native ``(y, x)`` grid in radians; only its pixel spacing is
        used, so the full native grid and the extent sub-grid give the same answer.
    eps
        Requested NUFFT precision. ``1e-12`` saturates fp64 at every instrument
        profiled.
    chunk_size
        Cap on the visibilities passed to ``nufft2d1`` in one call, or ``None`` for
        one shot. The transform is linear in ``c``, so the chunks' transforms are
        summed and the result is the same array (to summation order).

        This is not an optimisation — it is a memory ceiling, the same one
        ``TransformerNUFFT`` carries as its own ``chunk_size`` (PyAutoArray#330).
        The spreader's gather buffer is ``K · nspread²`` complex128; at ``eps=1e-12``
        ``nspread ≈ 14``, so alma's ``K = 1e6`` needs ~3 GB and alma_high's
        ``K = 5e6`` needs ~15 GB — which is where an unchunked call on a 15 GB
        machine is killed by the OOM reaper rather than returning slowly. Use the
        instrument's own ``transformer_chunk_size``.

    Returns
    -------
    ``[2Ny, 2Nx]`` float64, wraparound-ordered, padding row/column zero.
    """
    import jax.numpy as jnp
    import nufftax

    noise_map_real = np.asarray(noise_map_real, dtype=np.float64)
    uv_wavelengths = np.asarray(uv_wavelengths, dtype=np.float64)
    grid_radians_2d = np.asarray(grid_radians_2d, dtype=np.float64)

    y_shape, x_shape = (int(s) for s in shape_masked_pixels_2d)

    pixel_scale_radians = _pixel_scale_radians_from(grid_radians_2d)

    # The transformer's own scaled frequencies (transformer.py:334-335).
    x = 2.0 * np.pi * uv_wavelengths[:, 0] * pixel_scale_radians
    y = 2.0 * np.pi * uv_wavelengths[:, 1] * pixel_scale_radians

    w = 1.0 / (noise_map_real**2)

    n_modes = (2 * x_shape, 2 * y_shape)
    total_visibilities = int(x.shape[0])

    if chunk_size is None or chunk_size >= total_visibilities:
        chunk_size = total_visibilities

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size}.")

    # Only Re(f) is ever used, so each chunk's real part is accumulated and the
    # complex block is released before the next one is spread.
    real_modes = np.zeros((2 * y_shape, 2 * x_shape), dtype=np.float64)

    for k0 in range(0, total_visibilities, chunk_size):
        k1 = min(total_visibilities, k0 + chunk_size)

        f = nufftax.nufft2d1(
            jnp.asarray(-x[k0:k1]),
            jnp.asarray(y[k0:k1]),
            jnp.asarray(w[k0:k1], dtype=jnp.complex128),
            n_modes,
            eps,
            1,
        )

        real_modes += np.asarray(np.real(f), dtype=np.float64)

        del f

    preload = np.ascontiguousarray(np.fft.ifftshift(real_modes))

    preload[y_shape, :] = 0.0
    preload[:, x_shape] = 0.0

    return preload


def _pixel_scale_radians_from(grid_radians_2d: np.ndarray) -> float:
    """``Δ_rad`` read off the radian grid as an adjacent-pixel difference.

    Taken from the grid rather than from ``mask.pixel_scales`` because the grid is
    what the brute-force builders differenced to get their ``dx``/``dy``: deriving
    it any other way would let a unit or half-pixel convention drift in between the
    two implementations this function has to reproduce exactly.

    Square pixels are asserted rather than handled. Every interferometer preset in
    ``instruments/interferometer.py`` is square, the ``[2Ny, 2Nx]`` offset grid has
    a single mode spacing per axis by construction, and a rectangular-pixel dataset
    would silently produce a *plausible* wrong preload — the failure mode this
    module exists to rule out.
    """
    if grid_radians_2d.ndim != 3 or grid_radians_2d.shape[-1] != 2:
        raise ValueError(
            f"grid_radians_2d must be [ny, nx, 2] native; got {grid_radians_2d.shape}."
        )

    n_y, n_x = grid_radians_2d.shape[:2]

    if n_y < 2 or n_x < 2:
        raise ValueError(
            "grid_radians_2d must be at least 2x2 for the pixel scale to be read off "
            f"as an adjacent-pixel difference; got {(n_y, n_x)}."
        )

    # Native y decreases down the rows, x increases along the columns.
    delta_y = float(grid_radians_2d[0, 0, 0] - grid_radians_2d[1, 0, 0])
    delta_x = float(grid_radians_2d[0, 1, 1] - grid_radians_2d[0, 0, 1])

    if not np.isclose(delta_y, delta_x, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            "nufft_preload_from requires square pixels: the radian grid's row spacing "
            f"{delta_y!r} and column spacing {delta_x!r} differ."
        )

    return delta_x


def curvature_preload_from(dataset) -> np.ndarray:
    """The real-space ``W~`` preload of ``dataset``, built by the modern NumPy builder.

    Public because it is the expensive object: ``O(N_pix * K)``, ~6 s at sma and
    10-15 minutes at alma's million visibilities. A harness that runs several arms over
    one dataset should build it once, cache it, and hand it to both
    ``Interferometer.apply_sparse_operator(nufft_precision_operator=...)`` and
    :meth:`NumbaPreload.from_curvature_preload`, so the cost is paid once rather than
    twice per arm.
    """
    return np.asarray(
        inversion_interferometer_util.nufft_precision_operator_via_np_from(
            **_preload_inputs_from(dataset)
        ),
        dtype=np.float64,
    )


def _dirty_image_from(dataset) -> np.ndarray:
    """``d~ = Re(Fᴴ W d)``, read off the dataset's sparse operator.

    Raises rather than recomputing: a dataset without a sparse operator has not
    had its one-off adjoint NUFFT paid, and silently paying it here would hide
    (from a profiling harness, of all things) a cost the caller did not ask for.
    """
    if getattr(dataset, "sparse_operator", None) is None:
        raise ValueError(
            "The interferometer dataset has no `sparse_operator`, so the numba "
            "w-tilde preload cannot read its dirty image `d~ = Re(F^H W d)`.\n\n"
            "Call `dataset = dataset.apply_sparse_operator(use_jax=False)` before "
            "building a `NumbaPreload`."
        )

    return np.asarray(dataset.sparse_operator.dirty_image, dtype=np.float64)


def _native_index_for_slim_index_from(dataset) -> np.ndarray:
    return np.asarray(dataset.transformer.real_space_mask.derive_indexes.native_for_slim).astype(
        "int"
    )


@dataclass(frozen=True)
class NumbaPreload:
    """The model-independent w-tilde quantities of one interferometer dataset."""

    curvature_preload: np.ndarray
    dirty_image: np.ndarray
    real_space_mask: object
    native_index_for_slim_index: np.ndarray

    @classmethod
    def from_sparse_operator(cls, dataset) -> NumbaPreload:
        """Build the preload from a dataset that already has a sparse operator.

        The dirty image is read off ``dataset.sparse_operator`` verbatim.

        ``curvature_preload`` is rebuilt with ``nufft_precision_operator_via_np_from``
        rather than read off the operator, because
        ``InterferometerSparseOperator.from_nufft_precision_operator`` keeps only
        ``Khat = fft2(nufft_precision_operator)`` and discards the real-space array
        the numba scatter kernel indexes. The rebuild calls the same NumPy function
        ``apply_sparse_operator(use_jax=False)`` itself calls with the same
        arguments, so the array is the one the operator was built from — it is a
        recomputation, not a re-derivation, and ``test_parity.py`` pins it
        elementwise against ``nufft_precision_operator_via_np_from``.
        """
        curvature_preload = inversion_interferometer_util.nufft_precision_operator_via_np_from(
            **_preload_inputs_from(dataset)
        )

        return cls(
            curvature_preload=np.asarray(curvature_preload, dtype=np.float64),
            dirty_image=_dirty_image_from(dataset),
            real_space_mask=dataset.transformer.real_space_mask,
            native_index_for_slim_index=_native_index_for_slim_index_from(dataset),
        )

    @classmethod
    def from_curvature_preload(cls, dataset, curvature_preload) -> NumbaPreload:
        """Build the preload around an already-computed ``curvature_preload`` array.

        The same object :meth:`from_sparse_operator` produces, without paying the
        ``O(N_pix * K)`` build a second time. The array must be the one the dataset's
        sparse operator was itself built from — pass it to
        ``apply_sparse_operator(nufft_precision_operator=...)`` and to this method, so
        the two paths cannot drift apart.
        """
        return cls(
            curvature_preload=np.asarray(curvature_preload, dtype=np.float64),
            dirty_image=_dirty_image_from(dataset),
            real_space_mask=dataset.transformer.real_space_mask,
            native_index_for_slim_index=_native_index_for_slim_index_from(dataset),
        )

    @classmethod
    def via_numba(cls, dataset) -> NumbaPreload:
        """Build the preload with the recovered numba kernel.

        The kernel is ``w_tilde_curvature_preload_interferometer_from`` from
        PyAutoArray ``0b90c401`` — the quadruple-quadrant ``O(Ny·Nx·K)`` loop the
        modern chunked NumPy/JAX builders replaced. Phase 3 of the
        ``numba-interferometer-revisit`` epic profiles this against them; here it
        exists so the parity test can pin that they agree.
        """
        inputs = _preload_inputs_from(dataset)

        curvature_preload = numba_util_pack.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=inputs["noise_map_real"],
            uv_wavelengths=inputs["uv_wavelengths"],
            shape_masked_pixels_2d=tuple(int(s) for s in inputs["shape_masked_pixels_2d"]),
            grid_radians_2d=inputs["grid_radians_2d"],
        )

        return cls(
            curvature_preload=np.asarray(curvature_preload, dtype=np.float64),
            dirty_image=_dirty_image_from(dataset),
            real_space_mask=dataset.transformer.real_space_mask,
            native_index_for_slim_index=_native_index_for_slim_index_from(dataset),
        )

    def w_matrix(self) -> np.ndarray:
        """The dense ``W~`` expanded from the preload — testing only.

        ``[N_pix, N_pix]``, so this is only ever tractable on a small mask. The
        parity test uses it as the dense oracle for ``F = Mᵀ W~ M``.
        """
        return numba_util_pack.w_tilde_via_preload_from(
            w_tilde_preload=self.curvature_preload,
            native_index_for_slim_index=self.native_index_for_slim_index,
        )
