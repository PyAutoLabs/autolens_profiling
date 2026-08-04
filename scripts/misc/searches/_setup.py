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

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)


import sys
from pathlib import Path
from typing import Any, Optional

import autofit as af
import autolens as al
import numpy as np

_WORKSPACE_ROOT = _profiling_root()  # autolens_profiling/

# ``_adapt_image_util`` lives at the workspace root.
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
from simulators.point_source import INSTRUMENTS as _POINT_SOURCE_INSTRUMENTS  # noqa: E402

from _adapt_image_util import adapt_image_for_dataset  # noqa: E402
from _profile_cli import auto_simulate_if_missing  # noqa: E402
from instruments.imaging import INSTRUMENTS as _IMAGING_INSTRUMENTS  # noqa: E402
from instruments.interferometer import (  # noqa: E402
    INSTRUMENTS as _INTERFEROMETER_INSTRUMENTS,
)

_PIXELIZATION_MESH_SHAPE: tuple[int, int] = (39, 39)  # 1521 source pixels — production fiducial
_HILBERT_PIXELS: int = 1500

# The pixelized-source model types. ``knn`` and ``delaunay_matern`` were
# promoted from the autolens_workspace_developer#117 multi-start gradient
# campaign (searches_minimal/pix_prodigy_findings.md): the Delaunay-family
# meshes share the Hilbert image-mesh + AdaptImages plumbing, and the choice
# of regularization parametrization is load-bearing for gradient search (see
# the model builders below).
_PIX_MODEL_TYPES: tuple[str, ...] = ("pixelization", "delaunay", "knn", "delaunay_matern")
# The subset whose mesh vertices come from a precomputed image-plane mesh grid.
_DELAUNAY_FAMILY: tuple[str, ...] = ("delaunay", "knn", "delaunay_matern")
_MGE_TOTAL_GAUSSIANS: int = (
    20  # ``source_lp[1]`` SLaM fiducial; lighter than likelihood_runtime's 60
)
_GROUP4_MGE_TOTAL_GAUSSIANS: int = (
    10  # per-galaxy basis for the 4+4 group cell — 8 galaxies, so kept lean
)
_DATACUBE_N_CHANNELS: int = (
    4  # matches the "quick iteration" value in likelihood_runtime/datacube/delaunay.py
)

# Multi-band imaging datacube channel presets (compile-time A/B, issue: multi-band
# FactorGraphModel value_and_grad cold compile). Both arms use 4 channels so the
# only variable is per-band shape:
#   homogeneous  -> all jwst (0.03 arcsec/px) -> identical masked-pixel counts
#   heterogeneous-> 2x jwst (0.03) + 2x jwst_lw (0.06) -> two distinct shapes,
#                   mirroring JWST F115W/F150W vs F277W/F444W.
_DATACUBE_IMG_HOMO: list[str] = ["jwst", "jwst", "jwst", "jwst"]
_DATACUBE_IMG_HETERO: list[str] = ["jwst", "jwst", "jwst_lw", "jwst_lw"]


class FitPositionsSourceTensor(al.FitPositionsSource):
    """Free-centre tensor-weighted source-plane fit (PyAutoLens#679)."""

    weighting = "jacobian"


class _ClusterDatasetList(list):
    """Plain ``list`` subclass — a builtin ``list`` cannot carry attributes
    (``x = []; x.foo = 1`` raises), but the cluster cell's ``dataset`` is a
    ``List[PointDataset]`` (one per lensed system) rather than the single
    ``PointDataset`` the point_source cell stashes ``_profiling_solver_kwargs``
    on. This subclass lets the cluster dataset carry the same stash (#678
    phase B chunk 2), so ``_build_analysis`` and the runner's truth-anchor
    step read it identically to the point_source convention in
    ``_build_point_source``.
    """


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
    ``autolens_workspace/scripts/multi_dataset/modeling.py``.
    """
    if dataset_class == "datacube":
        return _build_for_datacube(
            model_type=model_type,
            instrument=instrument,
            use_jax=use_jax,
            use_mixed_precision=use_mixed_precision,
        )

    if dataset_class in ("datacube_img", "datacube_img_hetero"):
        # ``instrument`` is ignored here — the channels come from the fixed
        # multi-band presets; the dataset_class selects homogeneous vs mixed.
        instruments = (
            _DATACUBE_IMG_HETERO if dataset_class == "datacube_img_hetero" else _DATACUBE_IMG_HOMO
        )
        return _build_for_datacube_imaging(
            model_type=model_type,
            instruments=instruments,
            use_jax=use_jax,
            use_mixed_precision=use_mixed_precision,
        )

    if dataset_class == "cluster":
        return _build_for_cluster(
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

    Mirrors ``autolens_workspace/scripts/multi_dataset/modeling.py``: build N
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
                use_border_relocator=model_type in _PIX_MODEL_TYPES,
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
        af.AnalysisFactor(prior_model=model.copy(), analysis=analysis) for analysis in analysis_list
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=use_jax)
    return dataset_list, factor_graph.global_prior_model, factor_graph


def _build_for_datacube_imaging(
    *,
    model_type: str,
    instruments: list[str],
    use_jax: bool,
    use_mixed_precision: bool,
) -> tuple[list, Any, Any]:
    """Multi-band imaging datacube fit via ``af.FactorGraphModel``.

    Unlike ``_build_for_datacube`` (identical interferometer channels), each
    channel is an ``al.Imaging`` dataset built from a possibly *different*
    instrument preset, so the per-band masked-pixel count varies whenever the
    ``instruments`` list mixes pixel scales (jwst 0.03 + jwst_lw 0.06). This is
    the multi-wavelength JWST reproducer (F115W/F150W vs F277W/F444W): it lets
    the compile-time probe measure whether heterogeneous per-factor shapes stop
    XLA sharing fused sub-graphs across the factors.

    A homogeneous ``instruments`` list (all one preset) is the control arm; a
    mixed list is the heterogeneous arm. The model is defined in arcsec units
    (``mask_radius`` is instrument-independent at 3.5), so a single shared model
    is copied into each ``AnalysisFactor`` and the factors share global
    parameters — the only cross-factor difference is data shape.
    """
    mask_radius = _mask_radius_for("imaging", instruments[0])
    model = _build_model("imaging", model_type, mask_radius=mask_radius)

    dataset_list: list = []
    analysis_list: list = []
    for instrument in instruments:
        dataset, dataset_path = _build_imaging(instrument)
        adapt_images = _adapt_images_for(
            "imaging", model_type, dataset_path=dataset_path, dataset=dataset
        )
        analysis_list.append(
            _build_analysis(
                dataset_class="imaging",
                model_type=model_type,
                dataset=dataset,
                use_jax=use_jax,
                use_mixed_precision=use_mixed_precision,
                adapt_images=adapt_images,
            )
        )
        dataset_list.append(dataset)

    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model.copy(), analysis=analysis) for analysis in analysis_list
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=use_jax)
    return dataset_list, factor_graph.global_prior_model, factor_graph


def _build_for_cluster(
    *,
    model_type: str,
    instrument: str,
    use_jax: bool,
    use_mixed_precision: bool,
) -> tuple[list, Any, Any]:
    """Cluster point-source fit via ``af.FactorGraphModel`` (#678 phase B chunk 2).

    Mirrors ``autolens_workspace/scripts/cluster/modeling.py``: one
    ``al.AnalysisPoint`` per lensed system (an entry of the cluster's
    ``dataset_list``), each wrapped in an ``af.AnalysisFactor`` against a
    *copy* of the shared cluster model, combined into one
    ``af.FactorGraphModel``. Returns ``(dataset_list,
    factor_graph.global_prior_model, factor_graph)`` — same
    dataset_list/model/analysis convention ``_build_for_datacube`` uses above,
    so the runner's generic ``search.fit(model=model, analysis=analysis)``
    path needs no cluster-specific branching.
    """
    dataset_list, dataset_path = _build_dataset("cluster", instrument)
    mask_radius = _mask_radius_for("cluster", instrument)  # 0.0, unused by point models
    model = _build_model(
        "cluster",
        model_type,
        mask_radius=mask_radius,
        dataset_list=dataset_list,
        dataset_path=dataset_path,
    )
    analysis = _build_analysis(
        dataset_class="cluster",
        model_type=model_type,
        dataset=dataset_list,
        use_jax=use_jax,
        use_mixed_precision=use_mixed_precision,
        adapt_images=None,
        model=model,
    )
    return dataset_list, analysis.global_prior_model, analysis


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def _mask_radius_for(dataset_class: str, instrument: str) -> float:
    if dataset_class in ("imaging", "group"):
        return float(_IMAGING_INSTRUMENTS[instrument]["mask_radius"])
    if dataset_class in ("interferometer", "datacube"):
        return float(_INTERFEROMETER_INSTRUMENTS[instrument]["mask_radius"])
    if dataset_class == "point_source":
        # Point-source mask radius isn't applied to a 2D image; reuse the
        # imaging value so MGE/source-bulge priors share a sensible scale.
        return 3.5
    if dataset_class == "cluster":
        # Cluster point models have no 2D image mask either (and no MGE
        # source whose prior scale would need it) — unused, kept only so
        # this function's signature stays uniform across dataset_class.
        return 0.0
    raise ValueError(f"Unknown dataset_class: {dataset_class!r}")


def _build_dataset(dataset_class: str, instrument: str) -> tuple[Any, Path]:
    if dataset_class == "imaging":
        return _build_imaging(instrument)
    if dataset_class == "group":
        return _build_group_imaging(instrument)
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
    if dataset_class == "cluster":
        return _build_cluster(instrument)
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


def _build_group_imaging(instrument: str) -> tuple[al.Imaging, Path]:
    """Load the 4-lens + 4-source group imaging dataset (auto-simulate if missing).

    Identical mask / over-sampling / sparse-operator pipeline to
    ``_build_imaging``, but the dataset is the group-scale one written by
    ``simulators/group4_mge.py`` under ``dataset/imaging/group4_mge/<instrument>/``
    (which also writes the ``truth.json`` the recovery check consumes).
    """
    cfg = _IMAGING_INSTRUMENTS[instrument]
    pixel_scale = cfg["pixel_scale"]
    mask_radius = cfg["mask_radius"]
    dataset_path = Path("dataset") / "imaging" / "group4_mge" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="group4_mge",
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
    # Over-sample densely at every deflector centre — each group member hosts a
    # cuspy MGE light + mass profile, mirroring the simulator's centre list.
    from simulators.group4_mge import GROUP4_TRUTH

    centre_list = [tuple(map(float, ln["centre"])) for ln in GROUP4_TRUTH["lenses"]]
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=centre_list,
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )
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


def _build_cluster(instrument: str) -> tuple[_ClusterDatasetList, Path]:
    """Load the cluster's per-system point datasets (#678 phase B chunk 2).

    ``instrument`` is always ``"simple"`` today (``simulators/cluster.py``
    hardcodes its output path; there is no per-instrument ``INSTRUMENTS``
    dict to key off, unlike ``_build_point_source``).
    """
    dataset_path = Path("dataset") / "cluster" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="cluster",
        instrument=instrument,
        workspace_root=_WORKSPACE_ROOT,
    )
    dataset_list = _ClusterDatasetList(
        al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
    )
    # Solver kwargs stashed on the LIST (a plain list can't take attributes —
    # see _ClusterDatasetList) so _build_analysis and the runner's truth-anchor
    # step construct the identical PointSolver without re-deriving these
    # values. Taken verbatim from
    # scripts/cluster/likelihood_breakdown/image_plane.py's solver (the
    # "tutorial-scale configuration of the workspace cluster scripts") so
    # breakdown and search cells agree exactly.
    dataset_list._profiling_solver_kwargs = {  # type: ignore[attr-defined]
        "grid_shape": (200, 200),
        "pixel_scale": 0.7,
        "pixel_scale_precision": 0.01,
        "magnification_threshold": 0.1,
    }
    return dataset_list, dataset_path


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def _build_model(
    dataset_class: str,
    model_type: str,
    *,
    mask_radius: float,
    dataset_list: list | None = None,
    dataset_path: Path | None = None,
) -> af.Collection:
    if dataset_class == "group":
        if model_type == "mge":
            return _group_mge_model(mask_radius=mask_radius)
        raise ValueError(f"group cell only supports model_type='mge', got {model_type!r}")
    if dataset_class == "cluster":
        if dataset_list is None or dataset_path is None:
            raise RuntimeError(
                "cluster model construction requires dataset_list + dataset_path "
                "(build via _build_cluster first)."
            )
        return _cluster_point_model(
            dataset_list,
            dataset_path,
            solved=model_type in ("source_plane_solved", "image_plane_solved"),
        )
    if model_type == "mge":
        return _mge_model(mask_radius=mask_radius)
    if model_type == "pixelization":
        return _pixelization_model(mask_radius=mask_radius)
    if model_type == "delaunay":
        return _delaunay_model(mask_radius=mask_radius)
    if model_type == "knn":
        return _knn_model(mask_radius=mask_radius)
    if model_type == "delaunay_matern":
        return _delaunay_matern_model(mask_radius=mask_radius)
    if model_type in ("image_plane", "source_plane", "source_plane_tensor"):
        return _point_source_model()
    if model_type in ("image_plane_solved", "source_plane_solved", "image_plane_repeat_solved"):
        return _point_source_model(solved=True)
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
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=False,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _group_mge_model(*, mask_radius: float) -> af.Collection:
    """4 deflectors (MGE light + Isothermal mass) + 4 MGE sources.

    The high-dimensional (~54 free-param) group cell. Every galaxy's light and
    mass **centres are seeded** near the known truth position (`GROUP4_TRUTH`)
    with a modest-sigma Gaussian — the honest prior for a group whose members
    are individually visible, and (critically) the thing that breaks the
    permutation symmetry among the 4 lenses / 4 sources that would otherwise
    make the search hopeless. The genuinely-unknown quantities — Einstein
    radii, ellipticities, shear — keep their broad default priors, so the
    search still has to *find* the mass model.

    Amplitudes stay linear (solved by the inversion), so the non-linear count is
    geometry only. `ExternalShear` is on the primary deflector only, matching the
    simulator truth.
    """
    from simulators.group4_mge import GROUP4_TRUTH

    galaxies: dict[str, af.Model] = {}

    for lens in GROUP4_TRUTH["lenses"]:
        cy, cx = float(lens["centre"][0]), float(lens["centre"][1])
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=_GROUP4_MGE_TOTAL_GAUSSIANS,
            centre_prior_is_uniform=False,
            centre=(cy, cx),
            centre_sigma=0.1,
        )
        mass = af.Model(al.mp.Isothermal)
        mass.centre_0 = af.GaussianPrior(mean=cy, sigma=0.1)
        mass.centre_1 = af.GaussianPrior(mean=cx, sigma=0.1)
        kwargs: dict = dict(redshift=GROUP4_TRUTH["redshift_lens"], bulge=bulge, mass=mass)
        if lens["shear"] is not None:
            kwargs["shear"] = af.Model(al.mp.ExternalShear)
        galaxies[lens["name"]] = af.Model(al.Galaxy, **kwargs)

    for source in GROUP4_TRUTH["sources"]:
        cy, cx = float(source["centre"][0]), float(source["centre"][1])
        source_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=_GROUP4_MGE_TOTAL_GAUSSIANS,
            centre_prior_is_uniform=False,
            centre=(cy, cx),
            centre_sigma=0.3,
        )
        galaxies[source["name"]] = af.Model(
            al.Galaxy, redshift=GROUP4_TRUTH["redshift_source"], bulge=source_bulge
        )

    return af.Collection(galaxies=af.Collection(**galaxies))


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
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
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
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    # al.Pixelization only accepts mesh + regularization (no image_mesh kwarg).
    # The Hilbert image_mesh is applied OUTSIDE the model: the precomputed
    # image_plane_mesh_grid is passed to AnalysisImaging via AdaptImages's
    # galaxy_name_image_plane_mesh_grid_dict — see _adapt_images_for. The
    # mesh.Delaunay instance pins all parameters so PyAutoFit treats it as
    # a fixed value (the bare class form auto-promotes to af.Model and then
    # looks up priors for areas_factor, which has no entry in default config).
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.Delaunay(pixels=_HILBERT_PIXELS, areas_factor=0.5, zeroed_pixels=0),
        regularization=al.reg.ConstantSplit,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _knn_model(*, mask_radius: float) -> af.Collection:
    """Hilbert image_mesh + KNearestNeighbor mesh + free AdaptSplit reg.

    The Wendland-C4 KNN mesh is the Delaunay-family member built for gradient
    inference: its interpolation is smooth within each neighbour set, so
    descent information carries much further than through the Delaunay mesh's
    C0 flip seams. In the #117 broad-start MultiStartProdigy campaign it was
    the fastest pixelized converger — good basin in ~250 steps, exact truth
    (r_E to 3 d.p.) once resurrection crossed regularization modes.

    The regularization is the free split-family scheme the campaign validated.
    LESSON (#117): AdaptSplit double-squares its coefficients, so its
    high-coefficient region is an over-regularized floor the search must
    escape via resurrection — on this mesh that region is *finite* (escapable,
    late breakout ~step 1300), unlike on the Delaunay mesh where it is a NaN
    wall (see ``_delaunay_matern_model``). Budget accordingly
    (``_MULTI_START_N_STEPS_BY_CELL``): a long plateau is a reg mode, not
    convergence.
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=True,
    )
    mass, shear = _lens_mass_and_shear()
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    # Mesh instance pins all parameters (see _delaunay_model note on why the
    # bare class form cannot be used). Regularization: free (inner, outer)
    # with signal_scale pinned, matching the #117 validated surface.
    regularization = af.Model(al.reg.AdaptSplit)
    regularization.signal_scale = 1.0
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.KNearestNeighbor(pixels=_HILBERT_PIXELS, zeroed_pixels=0),
        regularization=regularization,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _delaunay_matern_model(*, mask_radius: float) -> af.Collection:
    """Hilbert image_mesh + Delaunay mesh + free MaternKernel regularization.

    The gradient-search variant of the ``delaunay`` cell. Same mesh, different
    regularization parametrization — and that difference is the whole point.

    LESSON (#117, searches_minimal/pix_prodigy_findings.md): with the
    split-family AdaptSplit reg, broad-start gradient search on the Delaunay
    mesh hits a **NaN wall** at high coefficients (the #104 double-squared
    lambda^4 fragility): lanes die instead of learning, and escaping the
    resulting +8.5k-logL plateau took a ~2000-step resurrection lottery. The
    Matérn kernel scheme reaches the SAME final fit quality (truth-point bars
    +29682 vs +30079) but degrades *gracefully* at high coefficient — no NaNs
    anywhere on its axis — giving a smooth, low-churn climb to the bar.
    Ordering measured on this mesh: Matérn >= fixed/inherited reg >> free
    AdaptSplit. (The nautilus ``delaunay`` cell keeps ConstantSplit for
    comparability with its own history; use THIS cell for gradient searches.)

    ``nu`` is pinned at 0.5 so the free reg is 2-parameter (coefficient,
    scale), dimension-matched to the AdaptSplit surface it replaces. Kernel
    regularizations build from pairwise vertex distances (no scipy neighbours)
    and are the other JAX-safe Delaunay-family pairing; they require
    tfp-nightly.
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=_MGE_TOTAL_GAUSSIANS,
        centre_prior_is_uniform=True,
    )
    mass, shear = _lens_mass_and_shear()
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass, shear=shear)
    regularization = af.Model(al.reg.MaternKernel)
    regularization.nu = 0.5
    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.Delaunay(pixels=_HILBERT_PIXELS, areas_factor=0.5, zeroed_pixels=0),
        regularization=regularization,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _point_source_model(solved: bool = False) -> af.Collection:
    mass, _ = _lens_mass_and_shear()  # No shear for the point-source profile.
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    # solved: parameter-free PointSolved — the *Solved fit classes solve the source
    # centre analytically (#657), dropping 3 free parameters vs PointFlux.
    point_0 = af.Model(al.ps.PointSolved) if solved else af.Model(al.ps.PointFlux)
    source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


# Point-source fit class per model_type — the single source of truth shared by
# _build_analysis (below) and searches/_runner.py's truth-anchor step (#678
# phase B), so the anchor evaluates the truth tracer through the EXACT fit
# class the cell itself searches with.
_POINT_SOURCE_FIT_CLS: dict[str, type] = {
    "image_plane": al.FitPositionsImagePairAll,
    "source_plane": al.FitPositionsSource,
    "image_plane_solved": al.FitPositionsImagePairAllSolved,
    "source_plane_solved": al.FitPositionsSourceSolved,
    "source_plane_tensor": FitPositionsSourceTensor,
    "image_plane_repeat_solved": al.FitPositionsImagePairRepeatSolved,
}


def point_source_fit_cls_for(model_type: str) -> type:
    """Resolve the ``fit_positions_cls`` a point_source ``model_type`` fits with."""
    try:
        return _POINT_SOURCE_FIT_CLS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown point_source model_type: {model_type!r}. "
            f"Add a row to _POINT_SOURCE_FIT_CLS in searches/_setup.py."
        ) from exc


# Cluster fit class per model_type (#678 phase B chunk 2) — deliberately a
# SEPARATE table from _POINT_SOURCE_FIT_CLS even though several model_type
# strings are shared: cluster's "image_plane_solved" fits with
# FitPositionsImagePairRepeatSolved (multi-image per-system pairing — the
# "model-fit default" per likelihood_breakdown/image_plane.py's docstring),
# not FitPositionsImagePairAllSolved (the galaxy-tier 2-image convention). No
# "image_plane" (free) or "image_plane_repeat_solved" row: forward-solving a
# free source centre is wall-time-prohibitive at cluster scale (~0.3 s/call —
# see the leaf cells' docstrings), and cluster has no pair-repeat variant.
_CLUSTER_FIT_CLS: dict[str, type] = {
    "source_plane": al.FitPositionsSource,
    "source_plane_solved": al.FitPositionsSourceSolved,
    "source_plane_tensor": FitPositionsSourceTensor,
    "image_plane_solved": al.FitPositionsImagePairRepeatSolved,
}


def cluster_point_fit_cls_for(model_type: str) -> type:
    """Resolve the ``fit_positions_cls`` a cluster ``model_type`` fits with.

    Shared by ``_build_analysis``'s cluster branch and the runner's cluster
    truth-anchor step (#678 phase B chunk 2), mirroring
    ``point_source_fit_cls_for`` above.
    """
    try:
        return _CLUSTER_FIT_CLS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown cluster model_type: {model_type!r}. "
            f"Add a row to _CLUSTER_FIT_CLS in searches/_setup.py."
        ) from exc


# Reference-anchored scaling-tier constants (#678 phase B chunk 2) — verbatim
# from scripts/misc/simulators/cluster.py / likelihood_breakdown/*.py, so the
# search model's fixed values match the simulator truth exactly. Both `b0`
# and `rs` scale with the SAME luminosity exponent here (this repo's own
# simulator convention) — unlike autolens_workspace/scripts/cluster/
# modeling.py's differing sigma/r_cut exponents (its dPIEMassSph
# parametrization ties them via the Bergamini mass-to-light tilt). See
# _cluster_point_model's docstring for the full port-vs-source deviation.
_CLUSTER_SCALING_RA: float = 0.1
_CLUSTER_SCALING_RS_REF: float = 10.0
_CLUSTER_SCALING_EXPONENT: float = 0.5

# Main-lens truth (centre, core radius `ra`) — centre + core stay fixed;
# `rs` / `b0` are the 2 free parameters per lens, matching the workspace
# modeling.py convention of "centre fixed; 2 free mass params per lens".
_CLUSTER_MAIN_LENS_TRUTH: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.0, 0.0), 8.0),
    ((10.0, 8.0), 5.0),
)
_CLUSTER_HOST_HALO_CENTRE: tuple[float, float] = (0.0, 0.0)
_CLUSTER_REDSHIFT_LENS: float = 0.5


def _cluster_point_model(
    dataset_list: list, dataset_path: Path, *, solved: bool = False
) -> af.Collection:
    """Cluster lens model (#678 phase B chunk 2) — ported from
    ``autolens_workspace/scripts/cluster/modeling.py`` onto this repo's own
    ``dataset/cluster/<instrument>`` truth.

    **Deviation from the workspace script**: the workspace's ``modeling.py``
    loads ``mass.csv`` / ``point.csv`` family CSVs via
    ``al.galaxy_models_from_csv`` / ``al.galaxy_af_models_from_csv_tables``,
    and its main lenses + host halo use the Lenstool-native ``dPIEMassSph``
    (``sigma``/``r_core``/``r_cut``, cosmology-pinned via ``H0``/``Om0``).
    This profiling repo's own cluster simulator
    (``scripts/misc/simulators/cluster.py``, mirrored by
    ``scripts/cluster/likelihood_breakdown/{image_plane,source_plane}.py``)
    writes only ``point_datasets.csv`` + ``scaling_galaxies.csv`` — no
    ``mass.csv``/``point.csv`` — and parametrizes every dPIE with
    ``dPIEMassB0Sph`` (``ra``/``rs``/``b0``, no ``H0``/``Om0`` attributes at
    all). Truth-anchoring loads ``tracer.json`` (whatever profile types the
    simulator wrote), so the model here MUST use the same parametrization for
    the ports to be physically comparable — hence ``dPIEMassB0Sph`` in place
    of ``dPIEMassSph``, with the sigma/r_cut free pair reinterpreted as the
    analogous b0/rs free pair (core radius ``ra`` fixed, like the workspace's
    fixed ``r_core=0`` — except this repo's truth core is non-zero, so ``ra``
    is fixed at its own truth value instead of 0).

    Components (mirrors the workspace's 4 categories):

    - 2 main lens galaxies (``dPIEMassB0Sph``): centre + core radius ``ra``
      fixed at truth; ``rs`` (truncation) + ``b0`` (lens strength) free — 2
      free parameters each [4 total].
    - The scaling tier (``dPIEMassB0Sph`` per member, loaded from
      ``scaling_galaxies.csv`` via ``al.galaxy_table_from_csv``): centre
      fixed per member; core ``ra`` fixed; ``b0`` and ``rs`` both derive from
      the SAME free ``scaling_b0_ref`` × ``(L_i / L_ref) ** exponent``
      relation (the profiling repo's simulator's own single-exponent
      convention) — 1 free parameter for the whole tier regardless of
      member count [1 total].
    - 1 standalone host halo (``NFWMCRLudlowSph``): centre fixed; free
      ``mass_at_200``. ``redshift_object`` / ``redshift_source`` are fixed at
      the truth values (0.5 / the furthest source redshift) — **never
      sampled**: these drive the cosmology-dependent Ludlow concentration-
      mass relation, and the PointSolver's implicit-diff ``custom_jvp``
      treats them as static aux, not a traced argument (#678, the
      "cosmology pinned" constraint) [1 total].
    - ``len(dataset_list)`` source galaxies, one per system, name-paired to
      ``dataset.name`` (``point_0``, ``point_1``, ...): ``solved=True`` gives
      each a parameter-free ``al.ps.PointSolved`` [0 total]; ``solved=False``
      gives each a free-centre ``al.ps.Point`` with a uniform prior ±1.0"
      around that system's OWN observed-positions centroid (a data-driven,
      per-source initialisation — 2 free parameters per source).

    Free-parameter total: 6 for every ``*_solved`` model_type (matching the
    workspace's own "N=6" cluster fiducial); 6 + 2*``len(dataset_list)`` for
    the free-centre variants (``source_plane`` / ``source_plane_tensor``).
    """
    source_redshifts = [float(d.redshift) for d in dataset_list]

    main_lens_models: list[af.Model] = []
    for centre, ra in _CLUSTER_MAIN_LENS_TRUTH:
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre = centre
        mass.ra = ra
        mass.rs = af.UniformPrior(lower_limit=1.0, upper_limit=40.0)
        mass.b0 = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)
        main_lens_models.append(af.Model(al.Galaxy, redshift=_CLUSTER_REDSHIFT_LENS, mass=mass))

    host_halo_mass = af.Model(al.mp.NFWMCRLudlowSph)
    host_halo_mass.centre = _CLUSTER_HOST_HALO_CENTRE
    host_halo_mass.redshift_object = _CLUSTER_REDSHIFT_LENS
    host_halo_mass.redshift_source = max(source_redshifts)
    host_halo_mass.mass_at_200 = af.LogUniformPrior(lower_limit=10**14.5, upper_limit=10**16.0)
    host_halo_model = af.Model(al.Galaxy, redshift=_CLUSTER_REDSHIFT_LENS, dark=host_halo_mass)

    galaxies: dict[str, af.Model] = {
        "lens_0": main_lens_models[0],
        "lens_1": main_lens_models[1],
        "host_halo": host_halo_model,
    }

    for i, dataset in enumerate(dataset_list):
        if solved:
            point = af.Model(al.ps.PointSolved)
        else:
            centroid = np.asarray(dataset.positions).mean(axis=0)
            point = af.Model(al.ps.Point)
            point.centre_0 = af.UniformPrior(
                lower_limit=float(centroid[0]) - 1.0, upper_limit=float(centroid[0]) + 1.0
            )
            point.centre_1 = af.UniformPrior(
                lower_limit=float(centroid[1]) - 1.0, upper_limit=float(centroid[1]) + 1.0
            )
        galaxies[f"source_{i}"] = af.Model(
            al.Galaxy, redshift=source_redshifts[i], **{dataset.name: point}
        )

    scaling_table = al.galaxy_table_from_csv(file_path=dataset_path / "scaling_galaxies.csv")
    scaling_b0_ref = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    reference_luminosity = max(scaling_table.luminosities)
    scaling_galaxies_list: list[af.Model] = []
    for centre, luminosity in zip(scaling_table.centres, scaling_table.luminosities):
        luminosity_ratio = luminosity / reference_luminosity
        mass = af.Model(al.mp.dPIEMassB0Sph)
        mass.centre = tuple(centre)
        mass.ra = _CLUSTER_SCALING_RA
        mass.rs = _CLUSTER_SCALING_RS_REF * luminosity_ratio**_CLUSTER_SCALING_EXPONENT
        mass.b0 = scaling_b0_ref * luminosity_ratio**_CLUSTER_SCALING_EXPONENT
        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=_CLUSTER_REDSHIFT_LENS, mass=mass)
        )
    scaling_galaxies = af.Collection(scaling_galaxies_list)

    return af.Collection(galaxies=af.Collection(**galaxies), scaling_galaxies=scaling_galaxies)


# -----------------------------------------------------------------------------
# Adapt image (pix/delaunay only)
# -----------------------------------------------------------------------------


def _adapt_images_for(
    dataset_class: str,
    model_type: str,
    *,
    dataset_path: Path,
    dataset: Any,
) -> al.AdaptImages | None:
    if model_type not in _PIX_MODEL_TYPES:
        return None
    if dataset_class not in ("imaging", "interferometer", "datacube"):
        return None
    adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)
    galaxy_key = "('galaxies', 'source')"

    extra: dict = {}
    if model_type in _DELAUNAY_FAMILY:
        # The Delaunay family's mapper.interpolator_from chain expects to find a
        # precomputed image_plane_mesh_grid via
        # AdaptImages.galaxy_name_image_plane_mesh_grid_dict — al.Pixelization
        # has no image_mesh field of its own. Mirror the workspace pattern
        # (autolens_workspace/scripts/imaging/features/pixelization/delaunay.py):
        # compute it once from the Hilbert image-mesh + truth-derived adapt
        # image, then ship via AdaptImages.
        mask = dataset.mask if dataset_class == "imaging" else dataset.real_space_mask
        image_mesh = al.image_mesh.Hilbert(
            pixels=_HILBERT_PIXELS, weight_power=1.0, weight_floor=0.0
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=mask, adapt_data=adapt_image
        )
        extra["galaxy_name_image_plane_mesh_grid_dict"] = {galaxy_key: image_plane_mesh_grid}

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
    adapt_images: al.AdaptImages | None,
    model: Any | None = None,
) -> Any:
    # Pixelization / Delaunay analyses normally require ``positions_likelihood_list``
    # to guard against the demagnified-source systematic. For pure profiling we
    # don't care about solution quality — we're measuring sampler + likelihood
    # cost — so disable the check rather than wire up truth-position plumbing.
    raise_positions_exc = model_type not in _PIX_MODEL_TYPES

    if dataset_class in ("imaging", "group"):
        return al.AnalysisImaging(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=al.Settings(
                use_border_relocator=model_type in _PIX_MODEL_TYPES,
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
                use_border_relocator=model_type in _PIX_MODEL_TYPES,
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
        fit_positions_cls = point_source_fit_cls_for(model_type)
        return al.AnalysisPoint(
            dataset=dataset,
            solver=solver,
            fit_positions_cls=fit_positions_cls,
            use_jax=use_jax,
        )
    if dataset_class == "cluster":
        # ``dataset`` is the cluster's List[PointDataset] (see
        # _build_cluster); ``model`` is the shared cluster prior model built
        # by _cluster_point_model — required here (unlike every other
        # dataset_class) because the factor-graph pattern needs a fresh
        # COPY of it per per-system AnalysisFactor (#678 phase B chunk 2,
        # mirrors autolens_workspace/scripts/cluster/modeling.py's
        # analysis_factor_list).
        if model is None:
            raise RuntimeError(
                "cluster analysis construction requires the shared cluster "
                "model (build via _cluster_point_model first)."
            )
        solver_kwargs = getattr(dataset, "_profiling_solver_kwargs", None)
        if solver_kwargs is None:
            raise RuntimeError(
                "cluster dataset_list is missing the solver kwargs stash; "
                "construct it via _build_cluster first."
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
        fit_positions_cls = cluster_point_fit_cls_for(model_type)
        analysis_list = [
            al.AnalysisPoint(
                dataset=ds,
                solver=solver,
                fit_positions_cls=fit_positions_cls,
                use_jax=use_jax,
            )
            for ds in dataset
        ]
        analysis_factor_list = [
            af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
            for analysis in analysis_list
        ]
        return af.FactorGraphModel(*analysis_factor_list, use_jax=use_jax)
    raise ValueError(f"Unknown dataset_class: {dataset_class!r}")


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------


def format_best_fit(instance: Any) -> str:
    """One-line summary of an instance's lens mass + shear (best-effort).

    Works across mge / pix / delaunay / point-source models; falls back to a
    generic representation when fields are missing. For the multi-galaxy group
    cell (``lens_0..lens_N``) it summarises each deflector's Einstein radius.
    """
    group_lenses = sorted(name for name in vars(instance.galaxies) if str(name).startswith("lens_"))
    if group_lenses:
        parts = []
        for name in group_lenses:
            try:
                er = getattr(instance.galaxies, name).mass.einstein_radius
                parts.append(f"{name}.theta_E={er:.3f}")
            except AttributeError:
                continue
        if parts:
            return "  ".join(parts)

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
