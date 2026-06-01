"""
Shared dataset/model/analysis builders for the ``searches/`` profiling scripts.

Generalises across the cells defined in
``autolens_profiling/instruments/{imaging,interferometer}.py`` and the
point-source presets in ``simulators/point_source.py``, with model-type
dispatch across ``mge`` / ``pixelization`` / ``delaunay`` (and point-source
``image_plane`` / ``source_plane``).

The builders use **uniform priors** rather than the ``GaussianPrior``-near-truth
pattern that the ``likelihood_runtime/`` scripts use. The likelihood scripts are
profiling deterministic per-call cost at the truth; the search scripts need the
sampler to actually search a realistic prior volume so its convergence cost
reflects production use.

Pixelization / Delaunay sources consume a truth-derived adapt image cached
next to the dataset as ``lensed_source.fits`` (built by
``_adapt_image_util.adapt_image_for_dataset`` on first call). This is a
profiling-convenience simplification — production SLaM regenerates the adapt
image across phases.

Usage::

    from searches._setup import build_for_cell

    dataset, model, analysis = build_for_cell(
        dataset_class="imaging",
        model_type="mge",
        instrument="hst",
        use_jax=True,
        use_mixed_precision=False,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import autofit as af
import autolens as al

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/

# ``_adapt_image_util`` lives at the workspace root.
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import auto_simulate_if_missing  # noqa: E402
from instruments.imaging import INSTRUMENTS as _IMAGING_INSTRUMENTS  # noqa: E402
from instruments.interferometer import (  # noqa: E402
    INSTRUMENTS as _INTERFEROMETER_INSTRUMENTS,
)
from simulators.point_source import INSTRUMENTS as _POINT_SOURCE_INSTRUMENTS  # noqa: E402


_PIXELIZATION_MESH_SHAPE: tuple[int, int] = (39, 39)  # 1521 source pixels — production fiducial
_HILBERT_PIXELS: int = 1500
_MGE_TOTAL_GAUSSIANS: int = 20  # ``source_lp[1]`` SLaM fiducial; lighter than likelihood_runtime's 60
_DATACUBE_N_CHANNELS: int = 4  # matches the "quick iteration" value in likelihood_runtime/datacube/delaunay.py


# -----------------------------------------------------------------------------
# Top-level dispatcher
# -----------------------------------------------------------------------------


def build_for_cell(
    *,
    dataset_class: str,
    model_type: str,
    instrument: str,
    use_jax: bool,
    use_mixed_precision: bool = False,
) -> tuple[Any, Any, Any]:
    """Build dataset, model and analysis for one profiling cell.

    Returns ``(dataset, model, analysis)``. The analysis has all per-cell
    plumbing (adapt images for pix/delaunay; transformer choice for
    interferometer; solver for point_source) already attached.

    Datacube cells return ``(dataset_list, factor_graph.global_prior_model,
    factor_graph)`` — the search treats the factor graph as both the model
    source and the analysis, per the multi-dataset pattern in
    ``autolens_workspace/scripts/multi/modeling.py``.
    """
    if dataset_class == "datacube":
        return _build_for_datacube(
            model_type=model_type,
            instrument=instrument,
            use_jax=use_jax,
            use_mixed_precision=use_mixed_precision,
        )

    dataset, dataset_path = _build_dataset(dataset_class, instrument)
    mask_radius = _mask_radius_for(dataset_class, instrument)
    model = _build_model(dataset_class, model_type, mask_radius=mask_radius)
    adapt_images = _adapt_images_for(
        dataset_class, model_type, dataset_path=dataset_path, dataset=dataset
    )
    analysis = _build_analysis(
        dataset_class=dataset_class,
        model_type=model_type,
        dataset=dataset,
        use_jax=use_jax,
        use_mixed_precision=use_mixed_precision,
        adapt_images=adapt_images,
    )
    return dataset, model, analysis


def _build_for_datacube(
    *,
    model_type: str,
    instrument: str,
    use_jax: bool,
    use_mixed_precision: bool,
) -> tuple[list, Any, Any]:
    """Multi-channel datacube fit via ``af.FactorGraphModel``.

    Mirrors ``autolens_workspace/scripts/multi/modeling.py``: build N
    per-channel interferometer datasets, wrap each in an
    ``AnalysisInterferometer``, pair each with a copy of the shared model
    via ``af.AnalysisFactor``, then combine into an ``af.FactorGraphModel``.

    The N channels are identical copies of the per-instrument dataset (the
    profiling concern is cube-cost scaling, not band-wavelength variation),
    so the adapt image is computed once and shared across every channel's
    AnalysisInterferometer.
    """
    dataset_list, dataset_path = _build_datacube_channels(instrument)
    mask_radius = _mask_radius_for("datacube", instrument)
    model = _build_model("datacube", model_type, mask_radius=mask_radius)

    adapt_images = _adapt_images_for(
        "datacube",
        model_type,
        dataset_path=dataset_path,
        dataset=dataset_list[0],
    )

    analysis_list = [
        al.AnalysisInterferometer(
            dataset=ds,
            adapt_images=adapt_images,
            settings=al.Settings(
                use_border_relocator=model_type in ("pixelization", "delaunay"),
                use_mixed_precision=use_mixed_precision,
            ),
            use_jax=use_jax,
        )
        for ds in dataset_list
    ]

    # One AnalysisFactor per channel, each with its own copy of the model so
    # PyAutoFit's factor graph treats them as independent likelihood factors
    # sharing the same global parameters.
    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
        for analysis in analysis_list
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=use_jax)
    return dataset_list, factor_graph.global_prior_model, factor_graph


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def _mask_radius_for(dataset_class: str, instrument: str) -> float:
    if dataset_class == "imaging":
        return float(_IMAGING_INSTRUMENTS[instrument]["mask_radius"])
    if dataset_class in ("interferometer", "datacube"):
        return float(_INTERFEROMETER_INSTRUMENTS[instrument]["mask_radius"])
    if dataset_class == "point_source":
        # Point-source mask radius isn't applied to a 2D image; reuse the
        # imaging value so MGE/source-bulge priors share a sensible scale.
        return 3.5
    raise ValueError(f"Unknown dataset_class: {dataset_class!r}")


def _build_dataset(dataset_class: str, instrument: str) -> tuple[Any, Path]:
    if dataset_class == "imaging":
        return _build_imaging(instrument)
    if dataset_class == "interferometer":
        return _build_interferometer(instrument)
    if dataset_class == "datacube":
        # Datacube takes the FactorGraphModel path in build_for_cell; this
        # branch is only here so direct callers of _build_dataset still
        # work — it returns the first channel only.
        dataset_list, dataset_path = _build_datacube_channels(instrument)
        return dataset_list[0], dataset_path
    if dataset_class == "point_source":
        return _build_point_source(instrument)
    raise ValueError(f"Unknown dataset_class: {dataset_class!r}")


def _build_datacube_channels(instrument: str) -> tuple[list, Path]:
    """Build ``_DATACUBE_N_CHANNELS`` identical-channel interferometer datasets.

    Channels are identical copies of the same per-instrument dataset (the
    profile is cube-cost scaling, not band-wavelength variation). Each
    channel is built via a fresh ``from_fits + apply_sparse_operator`` so
    the analyses don't share mutable dataset state — mirrors the existing
    ``likelihood_runtime/datacube/delaunay.py`` pattern.
    """
    dataset_list = []
    dataset_path: Path | None = None
    for _ in range(_DATACUBE_N_CHANNELS):
        ds, dataset_path = _build_interferometer(instrument)
        dataset_list.append(ds)
    assert dataset_path is not None  # _DATACUBE_N_CHANNELS >= 1
    return dataset_list, dataset_path


def _build_imaging(instrument: str) -> tuple[al.Imaging, Path]:
    cfg = _IMAGING_INSTRUMENTS[instrument]
    pixel_scale = cfg["pixel_scale"]
    mask_radius = cfg["mask_radius"]
    dataset_path = Path("dataset") / "imaging" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="imaging",
        instrument=instrument,
        workspace_root=_WORKSPACE_ROOT,
    )
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=pixel_scale,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=1,
    )
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )
    # The w-tilde sparse operator is what the inversion factory consults when
    # selecting InversionImagingSparse for models with a pixelization Mapper.
    # A100 fp64 vmap-probe (autolens_profiling#44):
    #   - dense path:  931 MB / replica -> n_live=150 needs 140 GB (OOM @ 80 GB)
    #   - sparse path:  95 MB / replica -> n_live=150 needs  14 GB (comfortable)
    # Pure-MGE-source cells short-circuit to dense in the factory regardless,
    # so attaching here only adds the w-tilde kernel-construction one-shot
    # cost (~tens of MB, sub-second) without changing per-eval cost.
    # Eliminates the need for PyAutoFit#1303/#1305's chunked-vmap workaround
    # on pixelization / Delaunay search runs.
    dataset = dataset.apply_sparse_operator()
    return dataset, dataset_path


def _build_interferometer(instrument: str) -> tuple[al.Interferometer, Path]:
    cfg = _INTERFEROMETER_INSTRUMENTS[instrument]
    pixel_scale = cfg["pixel_scale"]
    mask_radius = cfg["mask_radius"]
    real_space_shape = cfg["real_space_shape"]
    transformer_kind = cfg["transformer"]
    chunk_size = cfg.get("transformer_chunk_size")
    dataset_path = Path("dataset") / "interferometer" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="interferometer",
        instrument=instrument,
        workspace_root=_WORKSPACE_ROOT,
    )
    real_space_mask = al.Mask2D.circular(
        shape_native=real_space_shape,
        pixel_scales=pixel_scale,
        radius=mask_radius,
    )

    if transformer_kind == "dft":
        transformer_class: Any = al.TransformerDFT
    elif transformer_kind == "nufft":
        # Inject per-instrument chunk_size into TransformerNUFFT — required
        # for alma_high / jvla to cap the nufftax gather buffer (see
        # PyAutoArray#330 and the same idiom in
        # likelihood_runtime/datacube/delaunay.py).
        def _build_transformer(uv_wavelengths, real_space_mask):
            return al.TransformerNUFFT(
                uv_wavelengths=uv_wavelengths,
                real_space_mask=real_space_mask,
                chunk_size=chunk_size,
            )

        transformer_class = _build_transformer
    else:
        raise ValueError(
            f"Unknown transformer kind {transformer_kind!r} for instrument {instrument!r}"
        )

    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=transformer_class,
    )
    dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=False)
    return dataset, dataset_path


def _build_point_source(instrument: str) -> tuple[Any, Path]:
    cfg = _POINT_SOURCE_INSTRUMENTS[instrument]
    dataset_path = Path("dataset") / "point_source" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="point_source",
        instrument=instrument,
        workspace_root=_WORKSPACE_ROOT,
    )
    dataset = al.from_json(
        file_path=dataset_path / "point_dataset_positions_only.json",
    )
    # Stash the per-instrument PointSolver geometry alongside the dataset so
    # _build_analysis can construct it without re-reading the instrument dict.
    dataset._profiling_solver_kwargs = {  # type: ignore[attr-defined]
        "grid_shape": cfg["grid_shape"],
        "pixel_scale": cfg["pixel_scale"],
        "pixel_scale_precision": cfg["pixel_scale_precision"],
        "magnification_threshold": cfg["magnification_threshold"],
    }
    return dataset, dataset_path


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def _build_model(dataset_class: str, model_type: str, *, mask_radius: float) -> af.Collection:
    if model_type == "mge":
        return _mge_model(mask_radius=mask_radius)
    if model_type == "pixelization":
        return _pixelization_model(mask_radius=mask_radius)
    if model_type == "delaunay":
        return _delaunay_model(mask_radius=mask_radius)
    if model_type in ("image_plane", "source_plane"):
        return _point_source_model()
    raise ValueError(f"Unknown model_type: {model_type!r}")


def _lens_mass_and_shear() -> tuple[af.Model, af.Model]:
    """Isothermal + ExternalShear with uniform default priors — used by every
    non-point-source model.
    """
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    return mass, shear


def _mge_model(*, mask_radius: float) -> af.Collection:
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=True,
    )
    mass, shear = _lens_mass_and_shear()
    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=False,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _pixelization_model(*, mask_radius: float) -> af.Collection:
    """RectangularAdaptImage source, mirrors ``source_pix[1]`` init mesh.

    The lens light is MGE so the lens-light + source-pixelization
    inversion runs the full Gaussians + mesh columns through the same
    linear inversion path a real source_pix phase would.
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=True,
    )
    mass, shear = _lens_mass_and_shear()
    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularAdaptImage(
            shape=_PIXELIZATION_MESH_SHAPE,
            weight_power=1.0,
            weight_floor=0.0,
        ),
        regularization=al.reg.Constant,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _delaunay_model(*, mask_radius: float) -> af.Collection:
    """Hilbert image_mesh + Delaunay mesh + ConstantSplit regularization.

    Matches the ``source_pix[2]``-style production pipeline shape, with the
    Hilbert vertex count fixed at the production fiducial. The lens light is
    MGE for parity with the pixelization cell.
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=True,
    )
    mass, shear = _lens_mass_and_shear()
    lens = af.Model(
        al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear
    )
    # al.Pixelization only accepts mesh + regularization (no image_mesh kwarg).
    # The Hilbert image_mesh is applied OUTSIDE the model: the precomputed
    # image_plane_mesh_grid is passed to AnalysisImaging via AdaptImages's
    # galaxy_name_image_plane_mesh_grid_dict — see _adapt_images_for. The
    # mesh.Delaunay instance pins all parameters so PyAutoFit treats it as
    # a fixed value (the bare class form auto-promotes to af.Model and then
    # looks up priors for areas_factor, which has no entry in default config).
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.Delaunay(
            pixels=_HILBERT_PIXELS, areas_factor=0.5, zeroed_pixels=0
        ),
        regularization=al.reg.ConstantSplit,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _point_source_model() -> af.Collection:
    mass, _ = _lens_mass_and_shear()  # No shear for the point-source profile.
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    point_0 = af.Model(al.ps.PointFlux)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


# -----------------------------------------------------------------------------
# Adapt image (pix/delaunay only)
# -----------------------------------------------------------------------------


def _adapt_images_for(
    dataset_class: str,
    model_type: str,
    *,
    dataset_path: Path,
    dataset: Any,
) -> Optional[al.AdaptImages]:
    if model_type not in ("pixelization", "delaunay"):
        return None
    if dataset_class not in ("imaging", "interferometer", "datacube"):
        return None
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    galaxy_key = "('galaxies', 'source')"

    extra: dict = {}
    if model_type == "delaunay":
        # Delaunay's mapper.interpolator_from chain expects to find a
        # precomputed image_plane_mesh_grid via
        # AdaptImages.galaxy_name_image_plane_mesh_grid_dict — al.Pixelization
        # has no image_mesh field of its own. Mirror the workspace pattern
        # (autolens_workspace/scripts/imaging/features/pixelization/delaunay.py):
        # compute it once from the Hilbert image-mesh + truth-derived adapt
        # image, then ship via AdaptImages.
        mask = (
            dataset.mask
            if dataset_class == "imaging"
            else dataset.real_space_mask
        )
        image_mesh = al.image_mesh.Hilbert(
            pixels=_HILBERT_PIXELS, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=mask, adapt_data=adapt_image
        )
        extra["galaxy_name_image_plane_mesh_grid_dict"] = {
            galaxy_key: image_plane_mesh_grid
        }

    return al.AdaptImages(
        galaxy_name_image_dict={galaxy_key: adapt_image},
        **extra,
    )


# -----------------------------------------------------------------------------
# Analysis construction
# -----------------------------------------------------------------------------


def _build_analysis(
    *,
    dataset_class: str,
    model_type: str,
    dataset: Any,
    use_jax: bool,
    use_mixed_precision: bool,
    adapt_images: Optional[al.AdaptImages],
) -> Any:
    # Pixelization / Delaunay analyses normally require ``positions_likelihood_list``
    # to guard against the demagnified-source systematic. For pure profiling we
    # don't care about solution quality — we're measuring sampler + likelihood
    # cost — so disable the check rather than wire up truth-position plumbing.
    raise_positions_exc = model_type not in ("pixelization", "delaunay")

    if dataset_class == "imaging":
        return al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=al.Settings(
                use_border_relocator=model_type in ("pixelization", "delaunay"),
                use_mixed_precision=use_mixed_precision,
            ),
            raise_inversion_positions_likelihood_exception=raise_positions_exc,
            use_jax=use_jax,
        )
    if dataset_class in ("interferometer", "datacube"):
        return al.AnalysisInterferometer(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=al.Settings(
                use_border_relocator=model_type in ("pixelization", "delaunay"),
                use_mixed_precision=use_mixed_precision,
            ),
            raise_inversion_positions_likelihood_exception=raise_positions_exc,
            use_jax=use_jax,
        )
    if dataset_class == "point_source":
        solver_kwargs = getattr(dataset, "_profiling_solver_kwargs", None)
        if solver_kwargs is None:
            raise RuntimeError(
                "point_source dataset is missing the solver kwargs stash; "
                "construct it via _build_point_source first."
            )
        grid = al.Grid2D.uniform(
            shape_native=solver_kwargs["grid_shape"],
            pixel_scales=solver_kwargs["pixel_scale"],
        )
        solver = al.PointSolver.for_grid(
            grid=grid,
            pixel_scale_precision=solver_kwargs["pixel_scale_precision"],
            magnification_threshold=solver_kwargs["magnification_threshold"],
        )
        fit_positions_cls = (
            al.FitPositionsImagePairAll
            if model_type == "image_plane"
            else al.FitPositionsSource
        )
        return al.AnalysisPoint(
            dataset=dataset,
            solver=solver,
            fit_positions_cls=fit_positions_cls,
            use_jax=use_jax,
        )
    raise ValueError(f"Unknown dataset_class: {dataset_class!r}")


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------


def format_best_fit(instance: Any) -> str:
    """One-line summary of an instance's lens mass + shear (best-effort).

    Works across mge / pix / delaunay / point-source models; falls back to a
    generic representation when fields are missing.
    """
    try:
        mass = instance.galaxies.lens.mass
        out = (
            f"lens.mass.einstein_radius={mass.einstein_radius:.4f}  "
            f"lens.mass.centre=({mass.centre[0]:.3f}, {mass.centre[1]:.3f})"
        )
    except AttributeError:
        return repr(instance)
    try:
        shear = instance.galaxies.lens.shear
        out += f"  shear=({shear.gamma_1:.4f}, {shear.gamma_2:.4f})"
    except AttributeError:
        pass
    return out
