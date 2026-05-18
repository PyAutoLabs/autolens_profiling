"""Shared adapt-image construction for profiling likelihood scripts.

Loads or computes the lensed-source image used as the ``adapt_image`` for
adaptive pixelizations (``RectangularAdaptImage`` and image-meshes such as
``image_mesh.Hilbert``).

The function looks for a cached ``lensed_source.fits`` next to the rest of
the curated dataset artifacts (``data.fits``, ``tracer.json``, etc.). When
the file is absent, it derives the image from the truth tracer on
``dataset.grid`` and writes the result to disk so that sibling scripts on
the same instrument reuse it.

For imaging datasets the cached image is PSF-convolved (matches the spatial
scales the data resolves). For interferometer / datacube datasets the raw
real-space lensed source is cached; PSF / NUFFT operations live downstream
in the likelihood path.
"""

from pathlib import Path
from typing import Optional, Union

import autoarray as aa
import autolens as al


def adapt_image_for_dataset(
    *,
    dataset_path: Union[str, Path],
    dataset,
    tracer: Optional[al.Tracer] = None,
) -> aa.Array2D:
    """Return the lensed-source image masked to ``dataset.mask``.

    Parameters
    ----------
    dataset_path
        Directory containing ``tracer.json`` and (optionally) the cached
        ``lensed_source.fits``. Cache writes land here.
    dataset
        The masked dataset (``al.Imaging`` or ``al.Interferometer``). The
        function reads ``dataset.grid``, ``dataset.mask``, and (for imaging)
        ``dataset.psf`` + ``dataset.grids.blurring``.
    tracer
        Optional truth tracer; loaded from ``<dataset_path>/tracer.json``
        when not provided.
    """
    dataset_path = Path(dataset_path)
    cache_path = dataset_path / "lensed_source.fits"

    if cache_path.exists():
        native_image = aa.Array2D.from_fits(
            file_path=cache_path,
            pixel_scales=dataset.pixel_scales,
        )
        return native_image.apply_mask(mask=dataset.mask)

    if tracer is None:
        tracer = al.from_json(file_path=dataset_path / "tracer.json")

    has_psf = getattr(dataset, "psf", None) is not None

    if has_psf:
        plane_image_list = tracer.blurred_image_2d_list_from(
            grid=dataset.grids.lp,
            blurring_grid=dataset.grids.blurring,
            psf=dataset.psf,
        )
    else:
        plane_image_list = tracer.image_2d_list_from(grid=dataset.grid)

    lensed_source = plane_image_list[-1]

    aa.output_to_fits(
        values=lensed_source.native.array,
        file_path=str(cache_path),
        overwrite=True,
    )

    return lensed_source
