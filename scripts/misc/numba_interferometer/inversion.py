"""``InversionInterferometerNumba`` — the recovered numba w-tilde inversion.

This is the pack's re-wiring of ``InversionInterferometerWTilde`` (PyAutoArray
``0b90c401``, ``autoarray/inversion/inversion/interferometer/w_tilde.py``) onto
today's API. The linear algebra is untouched:

    D = Mᵀ d~                       (``d~`` = the dirty image ``Re(Fᴴ W d)``)
    F = Mᵀ W~ M                     (assembled by the numba scatter kernel)
    χ² = sᵀ F s − 2 sᵀ D + Σ d²/σ²  (``AbstractInversionInterferometer.fast_chi_squared``)

Only the *application* of ``W~`` differs from ``InversionInterferometerSparse``
on ``main``: a numba scatter-accumulate over image-pixel pairs here, an FFT
convolution on a padded ``(2Ny, 2Nx)`` grid there. Everything downstream —
``regularization_matrix``, ``curvature_reg_matrix``, ``reconstruction``, the two
log-determinants, ``regularization_term``, ``log_evidence`` — is inherited
unchanged from ``AbstractInversion``.

Preconditions (raised, never worked around)
-------------------------------------------
The recovered kernel consumes the mapper as the dense triplet arrays
``pix_indexes/sizes/weights_for_sub_slim_index`` indexed on **slim** rows, and
supports exactly one mapper and no linear-function lists. It also does **not**
fold ``over_sampler.sub_fraction`` into the weights, whereas the modern sparse
triplets do (``interferometer/sparse.py``'s ``_sparse_triplets_curvature_from``).
Anything else is a precondition violation and raises, rather than agreeing with
the modern path by accident:

- a linear-function list (e.g. a linear light profile / MGE basis),
- more than one mapper,
- ``over_sample_size != 1`` on the pixelization grid.
"""

from __future__ import annotations

import numpy as np
from autoarray import exc
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.interferometer.abstract import (
    AbstractInversionInterferometer,
)
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.structures.visibilities import Visibilities
from autonerves import cached_property

from numba_interferometer import inversion_interferometer_numba_util as numba_util_pack
from numba_interferometer import kernels as kernels_pack
from numba_interferometer.preload import NumbaPreload

# Every kernel here assembles the same F; they differ only in how the triple product
# M^T W~ M is evaluated. `kernels.py` documents each one and `test_parity.py` pins all
# of them to `reference` before any timing of them is believed.
KERNELS = (
    "reference",
    "hoisted",
    "symmetric",
    "two_stage",
    "direct_conv",
    "source_loop",
)


class InversionInterferometerNumba(AbstractInversionInterferometer):
    def __init__(
        self,
        dataset,
        linear_obj_list,
        settings=None,
        xp=np,
        preloads=None,
        numba_preload: NumbaPreload = None,
        kernel: str = "reference",
    ):
        """
        The numba w-tilde interferometer inversion, recovered from PyAutoArray ``0b90c401``.

        Parameters
        ----------
        dataset
            The interferometer dataset (or ``DatasetInterface``) being reconstructed. It must
            carry a ``sparse_operator`` so the dirty image is available without recomputation.
        linear_obj_list
            The linear objects reconstructing the data. Exactly one ``Mapper`` and nothing else.
        settings
            The inversion settings (``autoarray.settings.Settings``).
        xp
            The array module. Must be ``numpy``: the recovered kernels are numba, not JAX.
        preloads
            Optional ``AbstractPreloads``, forwarded to ``AbstractInversion`` unchanged.
        numba_preload
            The dataset's :class:`NumbaPreload`. Built via
            ``NumbaPreload.from_sparse_operator(dataset)`` when omitted.
        kernel
            Which curvature kernel assembles ``F`` (one of :data:`KERNELS`). They are
            algebraically identical and pinned to each other; ``"reference"`` is the
            phase-1 recovered kernel.
        """
        if xp is not np:
            raise exc.InversionException(
                "`InversionInterferometerNumba` was passed a non-NumPy array module "
                f"({xp!r}). The recovered kernels are numba `@jit` functions with no JAX "
                "path; use `InversionInterferometerSparse` for the JAX/FFT route."
            )

        if kernel not in KERNELS:
            raise exc.InversionException(
                f"Unknown curvature kernel {kernel!r} for `InversionInterferometerNumba`. "
                f"Available kernels: {KERNELS}."
            )

        try:
            import numba  # noqa: F401
        except ModuleNotFoundError as error:
            raise exc.InversionException(
                "The numba w-tilde interferometer inversion requires numba, which is not "
                "installed. Install it, or use `InversionInterferometerSparse`."
            ) from error

        self.kernel = kernel

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

        self._check_preconditions()

        self.numba_preload = (
            numba_preload
            if numba_preload is not None
            else NumbaPreload.from_sparse_operator(dataset)
        )

    def _check_preconditions(self) -> None:
        """Raise on every configuration the recovered kernel cannot represent."""
        if self.has(cls=AbstractLinearObjFuncList):
            raise exc.InversionException(
                "A linear-function list (e.g. a linear light profile or MGE basis) was passed "
                "to `InversionInterferometerNumba`. The recovered numba w-tilde kernel assembles "
                "F only from a mapper's `pix_indexes/sizes/weights_for_sub_slim_index` triplets "
                "and has no mapper x function or function x function block — the deleted "
                "`InversionInterferometerWTilde` fell back to the dense mapping formalism for "
                "these.\n\n"
                "Use `InversionInterferometerSparse` (which does support mixed linear objects) "
                "for such a model."
            )

        total_mappers = self.total(cls=Mapper)

        if total_mappers != 1:
            raise exc.InversionException(
                f"`InversionInterferometerNumba` was passed {total_mappers} mappers. The "
                "recovered numba w-tilde kernel builds a single [pix_pixels, pix_pixels] "
                "curvature matrix from one mapper and has no off-diagonal mapper x mapper "
                "block.\n\n"
                "Use `InversionInterferometerSparse` for a multi-mapper model."
            )

        mapper = self.cls_list_from(cls=Mapper)[0]

        sub_fraction = np.asarray(mapper.over_sampler.sub_fraction.array)

        if not np.all(sub_fraction == 1.0):
            raise exc.InversionException(
                "`InversionInterferometerNumba` was passed a mapper whose over-sampler has "
                f"`sub_fraction != 1` (minimum {float(sub_fraction.min())}, i.e. "
                f"`over_sample_size` up to {int(round(1.0 / float(sub_fraction.min())))}).\n\n"
                "The modern sparse triplets fold `over_sampler.sub_fraction` into the mapping "
                "weights (`interferometer/sparse.py::_sparse_triplets_curvature_from`); the "
                "recovered numba kernel uses `pix_weights_for_sub_slim_index` as-is and its "
                "rows are indexed on the slim grid, so with over-sampling it would silently "
                "compute a different F.\n\n"
                "Apply `over_sample_size_pixelization=1` to the dataset, or use "
                "`InversionInterferometerSparse`."
            )

    @property
    def curvature_preload(self) -> np.ndarray:
        return self.numba_preload.curvature_preload

    @property
    def dirty_image(self) -> np.ndarray:
        return self.numba_preload.dirty_image

    @cached_property
    def mapper_index_arrays(self):
        """The mapper's dense triplet arrays plus the slim → native index map.

        These are the five arrays the scatter kernel takes. They are pulled out of the
        curvature step so a breakdown script can time the mapper's index/weight array
        construction separately from the ``O(N² P²)`` scatter it feeds.
        """
        mapper = self.cls_list_from(cls=Mapper)[0]

        return (
            np.asarray(mapper.pix_indexes_for_sub_slim_index),
            np.asarray(mapper.pix_sizes_for_sub_slim_index),
            np.asarray(mapper.pix_weights_for_sub_slim_index),
            self.numba_preload.native_index_for_slim_index,
            int(mapper.params),
        )

    @property
    def data_vector(self) -> np.ndarray:
        """``D = Mᵀ d~`` — the real-space mapping matrix against the dirty image.

        Identical in form to ``InversionInterferometerSparse.data_vector``; the transform
        is never applied because ``d~ = Re(Fᴴ W d)`` already carries it.
        """
        return np.dot(self.mapping_matrix.T, self.dirty_image)

    @cached_property
    def kernel_index_arrays(self) -> dict:
        """The flat (CSR / CSC / extent-flat) form of the mapper the variants consume.

        The recovered kernel reads the mapper's ``[N_pix, P]`` triplet arrays directly;
        every phase-2 variant reads flat arrays instead, and the extent-grid and FFT
        forms additionally need extent-local row/column indexes. Marshalling is done
        here, inside the ``F`` step (``curvature_matrix_scatter`` touches this property),
        so a variant's timing carries its own input-preparation cost rather than hiding
        it in an earlier row. A reinstated library kernel would either pay the same cost
        or have the mapper emit this layout directly; charging it here is the
        conservative choice.

        The extent implied by the mapper's own native indexes is checked against the
        preload's ``(2Ny, 2Nx)`` shape, because the convolution kernels index the
        preload by extent offsets and a mismatch would silently wrap.
        """
        (
            pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index,
            native_index_for_slim_index,
            pix_pixels,
        ) = self.mapper_index_arrays

        inputs = kernels_pack.kernel_inputs_from(
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            native_index_for_slim_index=native_index_for_slim_index,
            pix_pixels=pix_pixels,
        )

        preload_shape = self.curvature_preload.shape

        if (2 * inputs["ny"], 2 * inputs["nx"]) != preload_shape:
            raise exc.InversionException(
                "The unmasked extent implied by the mapper's native indexes "
                f"({inputs['ny']} x {inputs['nx']}) does not match the curvature "
                f"preload's shape {preload_shape}, which must be (2Ny, 2Nx). The "
                "extent-grid and FFT kernels index the preload by extent offsets, so a "
                "mismatch would wrap silently instead of failing."
            )

        return inputs

    @cached_property
    def source_loop_index_arrays(self):
        """The source-major arrays the recovered ``..._from_2`` variant consumes."""
        (
            pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index,
            _,
            pix_pixels,
        ) = self.mapper_index_arrays

        return kernels_pack.source_loop_inputs_from(
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            pix_pixels=pix_pixels,
        )

    @cached_property
    def curvature_matrix_scatter(self) -> np.ndarray:
        """``F = Mᵀ W~ M`` from the selected numba kernel.

        ``reference`` is the recovered kernel: it loops the full ``N x N`` image-pixel
        pair space (it does **not** halve on symmetry), so the matrix it returns is
        already complete and symmetric — no mirroring pass is applied to it here, and
        ``test_parity.py`` pins that against the dense oracle ``Mᵀ W~ M``. ``symmetric``
        does halve and mirrors internally; every variant therefore returns a complete
        symmetric ``F`` from this property, whatever it does inside.
        """
        (
            pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index,
            native_index_for_slim_index,
            pix_pixels,
        ) = self.mapper_index_arrays

        if self.kernel == "reference":
            return (
                numba_util_pack.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
                    curvature_preload=self.curvature_preload,
                    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
                    pix_size_for_sub_slim_index=pix_sizes_for_sub_slim_index,
                    pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
                    native_index_for_slim_index=native_index_for_slim_index,
                    pix_pixels=pix_pixels,
                )
            )

        if self.kernel == "source_loop":
            sub_indexes, sub_sizes, sub_weights = self.source_loop_index_arrays

            return numba_util_pack.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from_2(
                curvature_preload=self.curvature_preload,
                native_index_for_slim_index=native_index_for_slim_index,
                pix_pixels=pix_pixels,
                sub_slim_indexes_for_pix_index=sub_indexes,
                sub_slim_sizes_for_pix_index=sub_sizes,
                sub_slim_weights_for_pix_index=sub_weights,
            )

        inputs = self.kernel_index_arrays
        preload = self.curvature_preload

        if self.kernel == "hoisted":
            return kernels_pack.curvature_hoisted(
                preload,
                inputs["iy"],
                inputs["ix"],
                inputs["indptr"],
                inputs["col"],
                inputs["val"],
                pix_pixels,
            )

        if self.kernel == "symmetric":
            return kernels_pack.curvature_symmetric(
                preload,
                inputs["iy"],
                inputs["ix"],
                inputs["indptr"],
                inputs["col"],
                inputs["val"],
                pix_pixels,
            )

        if self.kernel == "two_stage":
            return kernels_pack.curvature_two_stage(
                preload,
                inputs["iy"],
                inputs["ix"],
                inputs["indptr"],
                inputs["col"],
                inputs["val"],
                pix_pixels,
            )

        # `direct_conv` — the only remaining member of KERNELS.
        return kernels_pack.curvature_direct_conv(
            preload,
            inputs["iy"],
            inputs["ix"],
            inputs["flat"],
            inputs["indptr"],
            inputs["col"],
            inputs["val"],
            inputs["cscptr"],
            inputs["csc_row"],
            inputs["csc_val"],
            inputs["ny"],
            inputs["nx"],
            pix_pixels,
        )

    @cached_property
    def curvature_matrix(self) -> np.ndarray:
        """``F``, with the no-regularization diagonal addition applied.

        The scatter kernel returns a complete symmetric matrix, so unlike the imaging numba
        class (``imaging_numba/sparse.py``) and the modern interferometer sparse class there
        is no mirroring step — only the ``no_regularization_index_list`` diagonal add, which
        both of those also apply.
        """
        curvature_matrix = self.curvature_matrix_scatter

        if len(self.no_regularization_index_list) > 0:
            curvature_matrix = inversion_util.curvature_matrix_with_added_to_diag_from(
                curvature_matrix=curvature_matrix,
                value=self.settings.no_regularization_add_to_curvature_diag_value,
                no_regularization_index_list=self.no_regularization_index_list,
                xp=np,
            )

        return curvature_matrix

    @property
    def mapped_reconstructed_operated_data_dict(self):
        """The model visibilities of each linear object.

        Only formed when explicit model visibilities are needed (plots, residual maps);
        ``fast_chi_squared`` never touches this. One forward NUFFT per linear object, on the
        real-space ``M s`` image that ``mapped_reconstructed_data_dict`` returns.
        """
        mapped_reconstructed_operated_data_dict = {}

        image_dict = self.mapped_reconstructed_data_dict

        for linear_obj in self.linear_obj_list:
            visibilities = self.transformer.visibilities_from(image=image_dict[linear_obj], xp=np)

            mapped_reconstructed_operated_data_dict[linear_obj] = Visibilities(
                visibilities=visibilities
            )

        return mapped_reconstructed_operated_data_dict
