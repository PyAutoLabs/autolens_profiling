"""
Recovered numba w-tilde interferometer kernels.

Provenance
----------
Verbatim recovery (bodies unchanged) of the numba kernels that lived in
PyAutoArray at::

    autoarray/inversion/inversion/interferometer/inversion_interferometer_util.py

at commit ``0b90c401`` (2025-12-17, "fast chi squared implemented") — the last
commit on ``main`` at which the whole numba w-tilde interferometer likelihood
was intact. The module was dismantled by PR #201 (``c9605b5b``, merged
``e72defbb``, 2025-12-21) and PR #204 (``222ee046``, merged ``a655bb9e``,
2026-02-03), which replaced the numba scatter-accumulate with the JAX/FFT
``InterferometerSparseOperator`` that lives under the same path today.

Recover them with::

    git -C PyAutoArray show \
        0b90c401:autoarray/inversion/inversion/interferometer/inversion_interferometer_util.py

Kept: the six core kernels (lines 13-608 of the recovered file) plus
``sub_slim_indexes_for_pix_index`` (~line 1838). Dropped: the COSMA
multiprocessing experiments (``parallel_*``, ``jit_loop*``, ``make_2d``,
``*_stage_*``, ``*_in_stages_with_chunks_from``, ``*_para_from``) and the two
pure-NumPy helpers (``data_vector_via_transformed_mapping_matrix_from``,
``mapped_reconstructed_visibilities_from``) which this pack does not use — its
data vector is ``Mᵀ dirty_image`` and it never forms model visibilities.

Algebra
-------
``curvature_preload`` here is the *same array* the modern PyAutoArray calls
``nufft_precision_operator``::

    curvature_preload[Δy, Δx] = Σ_k σ_r[k]⁻² cos(2π [Δx·u_k + Δy·v_k])

so ``W~ = Re(Fᴴ W F)``, ``D = Mᵀ Re(Fᴴ W d)`` and ``F = Mᵀ W~ M`` are unchanged
between the recovered and the modern path; only the *application* differs (a
numba scatter over image-pixel pairs here, an FFT convolution there). See
``README.md`` in this directory.

``jit`` still comes from ``autoarray.numba_util``, which survives on ``main``.
"""

import logging

import numpy as np
from autoarray import numba_util

logger = logging.getLogger(__name__)


@numba_util.jit()
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray,
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
    native_index_for_slim_index,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    image_native
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros(image_pixels)

    weight_map_real = visibilities_real / noise_map_real**2.0

    for ip0 in range(image_pixels):
        value = 0.0

        y = grid_radians_slim[ip0, 1]
        x = grid_radians_slim[ip0, 0]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            value += weight_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0
                * np.pi
                * (y * uv_wavelengths[vis_1d_index, 0] + x * uv_wavelengths[vis_1d_index, 1])
            )

        w_tilde_data[ip0] = value

    return w_tilde_data


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels] that encodes the
    NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute the curvature
    matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
    a preload matrix that can compute the matrix w_tilde via an efficient preloading scheme which exploits the
    symmetries in the NUFFT.

    To compute w_tilde, one first defines a real space mask where every False entry is an unmasked pixel which is
    used in the calculation, for example:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an imaging.Mask2D, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
        IxIxIxIoIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI


    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
    downwards, therefore:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxI0I1I2IxIxIxIxI
        IxIxIxI3I4I5IxIxIxIxI
        IxIxIxI6I7I8IxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

    In the standard calculation of `w_tilde` it is a matrix of
    dimensions [unmasked_image_pixels, unmasked_pixel_images], therefore for the example mask above it would be
    dimensions [9, 9]. One performs a double for loop over `unmasked_image_pixels`, using the (y,x) spatial offset
    between every possible pair of unmasked image pixels to precompute values that depend on the properties of the NUFFT.

    This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
    example, if two image pixel are next to one another by the same spacing the same value will be computed via the
    NUFFT. For the example mask above:

    - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,9].
    - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].
    - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
    times using the mask above).

    The `w_tilde_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.

    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - w_tilde_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 0 - the values of pixels paired with themselves.
    - w_tilde_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
    in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]
    - w_tilde_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

    Flipped pairs:

    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `w_tilde_preload[:,:,1]`, and the ordering of these pairings is flipped in the
    x direction to make it straight forward to use this matrix when computing w_tilde.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    shape_masked_pixels_2d
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that precomputes the values for fast computation of w_tilde.
    """

    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = grid_radians_2d[0, -1, 0] - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                x_offset = grid_radians_2d[0, -1, 1] - grid_radians_2d[i, grid_x_shape - j - 1, 1]

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[i, -j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = grid_radians_2d[-1, 0, 0] - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                x_offset = grid_radians_2d[-1, 0, 1] - grid_radians_2d[grid_y_shape - i - 1, j, 1]

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, -j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload


@numba_util.jit()
def w_tilde_via_preload_from(w_tilde_preload, native_index_for_slim_index):
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    w_tilde_preload
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    slim_size = len(native_index_for_slim_index)

    w_tilde_via_preload = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            w_tilde_via_preload[i, j] = w_tilde_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelizaiton pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelizaiton pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    image_pixels = len(native_index_for_slim_index)

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += (
                        curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
                    )

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from_2(
    curvature_preload: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
    sub_slim_indexes_for_pix_index,
    sub_slim_sizes_for_pix_index,
    sub_slim_weights_for_pix_index,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
       [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
       pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
       compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization's mesh pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization's mesh that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    for sp0 in range(pix_pixels):
        ip_size_0 = sub_slim_sizes_for_pix_index[sp0]

        for sp1 in range(sp0, pix_pixels):
            val = 0.0
            ip_size_1 = sub_slim_sizes_for_pix_index[sp1]

            for ip0_tmp in range(ip_size_0):
                ip0 = sub_slim_indexes_for_pix_index[sp0, ip0_tmp]
                ip0_weight = sub_slim_weights_for_pix_index[sp0, ip0_tmp]

                ip0_y, ip0_x = native_index_for_slim_index[ip0]

                for ip1_tmp in range(ip_size_1):
                    ip1 = sub_slim_indexes_for_pix_index[sp1, ip1_tmp]
                    ip1_weight = sub_slim_weights_for_pix_index[sp1, ip1_tmp]

                    ip1_y, ip1_x = native_index_for_slim_index[ip1]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    val += curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight

            curvature_matrix[sp0, sp1] += val

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@numba_util.jit()
def sub_slim_indexes_for_pix_index(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pix_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_slim_sizes_for_pix_index = np.zeros(pix_pixels)

    for pix_indexes in pix_indexes_for_sub_slim_index:
        for pix_index in pix_indexes:
            sub_slim_sizes_for_pix_index[pix_index] += 1

    max_pix_size = np.max(sub_slim_sizes_for_pix_index)

    sub_slim_indexes_for_pix_index = -1 * np.ones(shape=(pix_pixels, int(max_pix_size)))
    sub_slim_weights_for_pix_index = -1 * np.ones(shape=(pix_pixels, int(max_pix_size)))
    sub_slim_sizes_for_pix_index = np.zeros(pix_pixels)

    for slim_index, pix_indexes in enumerate(pix_indexes_for_sub_slim_index):
        pix_weights = pix_weights_for_sub_slim_index[slim_index]

        for pix_index, pix_weight in zip(pix_indexes, pix_weights):
            sub_slim_indexes_for_pix_index[
                pix_index, int(sub_slim_sizes_for_pix_index[pix_index])
            ] = slim_index

            sub_slim_weights_for_pix_index[
                pix_index, int(sub_slim_sizes_for_pix_index[pix_index])
            ] = pix_weight

            sub_slim_sizes_for_pix_index[pix_index] += 1

    return (
        sub_slim_indexes_for_pix_index,
        sub_slim_sizes_for_pix_index,
        sub_slim_weights_for_pix_index,
    )
