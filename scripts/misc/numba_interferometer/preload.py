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
