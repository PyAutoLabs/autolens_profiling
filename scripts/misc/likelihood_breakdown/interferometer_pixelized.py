"""Shared harness for the interferometer pixelized (W~ / sparse) breakdown cells.

``scripts/interferometer/delaunay/likelihood_breakdown.py`` (Hilbert-1500 Delaunay)
and ``.../pixelization.py`` (rectangular 39x39) differ only in their mesh; everything
else — dataset, sparse operator, the timed steps, the dense comparison arm, the lever
sub-rows and the JSON schema — lives here, so the two cells cannot drift
(autolens_profiling#320, interferometer likelihood campaign 2/3).

What the sparse arm times (the library's per-evaluation work)
--------------------------------------------------------------

``FitInterferometer`` on a dataset with ``apply_sparse_operator()`` applied builds an
``InversionInterferometerSparse`` for a one-mapper source. Per evaluation it runs, in
order (PyAutoArray ``inversion/inversion/interferometer/sparse.py``, ``abstract.py``,
``fit/fit_interferometer.py``):

1. ray-trace the pixelization grid (and, for Delaunay, the image-plane mesh vertices);
2. border relocation + mesh interpolation -> the mapper (``pix_indexes`` / ``weights``);
3. the dense real-space mapping matrix ``L`` (``M_pix x S``, used only by ``D``);
4. the regularization matrix ``H``;
5. the sparse triplets of ``L`` on the unmasked-extent grid
   (``_sparse_triplets_curvature_from``);
6. ``D = Lᵀ d~`` against the cached dirty image;
7. ``F = Aᵀ W~ A`` via ``InterferometerSparseOperator.curvature_matrix_diag_from``:
   ``lax.fori_loop`` over ``ceil(S / batch_size)`` column blocks, each a scatter into
   ``(M, B)``, one ``rfft2``/``irfft2`` pair on ``(B, 2y, 2x)`` and a gather +
   ``segment_sum`` back to ``(S, B)``;
8. the positive-only solve of ``(F + H) s = D`` (PDIP, or the certified active set when
   ``Settings(positive_only_solver="certified")`` — honoured here, because the inversion
   is mapper-only on JAX);
9. the two log-dets (dense Choleskys of ``F + H`` and ``H`` on JAX);
10. ``fast_chi_squared`` (``sᵀFs - 2sᵀD + dᵀN⁻¹d``) + ``sᵀHs`` + noise normalisation ->
    the figure of merit.

There is **no transformed-mapping-matrix row** in this arm: the sparse path never forms
one. Steps 1-4 are nested ``params -> stage`` prefixes on the library fit, attributed by
successive differences (``setup_split``; the MGE cell's pattern); steps 5-10 are timed
standalone on the arrays the library produced. The standalone figure of merit is
checked against the library's (``step_fidelity``) and the fused
``AnalysisInterferometer`` pipeline is the ``full_pipeline_single_jit`` comparator.

The dense comparison arm (``dense_steps``)
-------------------------------------------

The same fit through ``InversionInterferometerMapping``: steps 1-4 are shared, then the
transformed mapping matrix ``T`` (NUFFT of every column), dense ``D`` and ``F`` (real +
imaginary), and the same solve / log-det / figure-of-merit functions. The library forms
``T`` with ONE ``nufft2d2`` over all columns, whose oversampled grid is
``S x (2Ny) x (2Nx)`` complex128 — 6.3 GB at sma for S = 1500, which is what put the old
cell at 14.6 GB RSS. The arm therefore forms ``T`` column-chunked (``jax.lax.map``,
``--dense-transform-chunk`` columns per step through the transformer's own
``_forward_native``: same arithmetic, bounded buffer — the MGE cell's
``--transform-chunk`` arm), and ``--dense-library`` additionally jits the library's own
dense ``FitInterferometer`` (guarded: a device OOM is recorded, not raised). The arm runs
only up to ``--dense-max-vis`` visibilities (``T`` alone is ``N_vis x S x 16`` bytes).
Witness: ``|figure_of_merit_sparse - figure_of_merit_dense| <= 1e-6`` nats in fp64.

Lever sub-rows (never summed)
-----------------------------

- ``curvature_batch_size_sweep`` (``--batch-size-sweep``): step 7 re-timed with the
  operator's ``batch_size`` replaced (``dataclasses.replace``), i.e. the block width /
  ``fori_loop`` trip count lever.
- ``curvature_kernel_split``: one column block's scatter, FFT apply and gather +
  ``segment_sum`` jitted separately and scaled by the block count — scatter-, FFT- or
  launch-bound.
- ``solver_ab``: PDIP and certified on the same ``(D, F + H)`` system, with iterations /
  passes and the figure-of-merit gap; ``--solver-ab`` also times the full pipeline with
  the other solver.
- ``fp32_fft_arm`` (``--fp32-fft-arm``): a measurement-only float32 copy of step 7
  (scatter, ``rfft2`` / ``irfft2``, gather in fp32; the library hard-casts to float64), its
  per-call time and the figure-of-merit shift against the 0.5-nat bar.
- ``mixed_precision``: with ``--use-mixed-precision``, the fp64 library figure of merit
  of the same run, so the mp shift is measured in-run.

Shared setup
------------

The dataset / mask inputs, the mass + source model and the adapt images are built by the
module-level helpers ``load_dataset``, ``build_model`` and ``adapt_images_builder``, and
the output names by ``result_paths``, and the two meshes by ``build_delaunay_mesh`` /
``build_rectangular_mesh``, so the NumPy / numba sibling
``interferometer_pixelized_numpy.py`` (campaign 3/3, autolens_profiling#326) runs on
exactly the same inputs. ``--mask-radius`` overrides the preset's real-space mask radius
(3.5"); a non-default radius appends ``_r<radius>`` to the output names.

Memory
------

Every jitted section's compiled executable is dropped and ``jax.clear_caches()`` +
``gc.collect()`` run between sections; the old cell's ``jit_profile`` kept every
executable alive. The JSON is rewritten after each section, so a SIGKILL (host OOM)
cannot lose the rows already measured.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import re
import resource
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import autoarray as aa
import autofit as af
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.interferometer import inversion_interferometer_util
from autoarray.inversion.mappers import mapper_util
from autoarray.inversion.mappers.abstract import Mapper
from autofit.jax import register_model as _register_model_pytrees

from . import timing
from .provenance import source_revisions

#: Above this visibility count the laptop CPU uses fewer repeats and the dense arm is off
#: by default (T alone is N_vis x S x 16 B).
LARGE_VIS = 200_000

#: Sparse-vs-dense structural witness (fp64).
WITNESS_NATS = 1e-6

#: The mixed-precision / fp32 bar the prompt sets for a precision lever.
PRECISION_BAR_NATS = 0.5


def add_mask_radius_arg(parser) -> None:
    """``--mask-radius``: shared by this harness and the NumPy / numba sibling."""
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=None,
        help="Real-space circular mask radius in arcsec (default: the instrument preset's, "
        "3.5). A non-default radius appends _r<radius> to the output names.",
    )


def add_cell_args(parser) -> None:
    """The cell-local flags both pixelized cells declare (on their ``_cell_parser``)."""
    add_mask_radius_arg(parser)
    parser.add_argument("--solver", choices=("pdip", "certified"), default="pdip")
    parser.add_argument(
        "--solver-ab",
        action="store_true",
        help="Also time the full pipeline with the other positive-only solver.",
    )
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument(
        "--vmap-batch",
        type=str,
        default=None,
        help="Comma-separated vmap batches for the sparse full pipeline, largest first; "
        "the first that fits is kept.",
    )
    parser.add_argument(
        "--batch-size-sweep",
        type=str,
        default=None,
        help="Comma-separated sparse-operator batch sizes to re-time step 7 (F) with.",
    )
    parser.add_argument("--fp32-fft-arm", action="store_true")
    parser.add_argument(
        "--dense-max-vis",
        type=int,
        default=LARGE_VIS,
        help="Run the dense comparison arm only up to this many visibilities.",
    )
    parser.add_argument(
        "--dense-transform-chunk",
        type=int,
        default=50,
        help="Columns per jax.lax.map step of the dense arm's chunked transform.",
    )
    parser.add_argument(
        "--dense-library",
        action="store_true",
        help="Also jit the library's one-shot dense FitInterferometer (OOM recorded).",
    )


@dataclass
class MeshSetup:
    """What a cell's mesh contributes: its pixelization and JSON fields."""

    pixelization: Any
    n_source_pixels: int
    image_plane_mesh_grid: Any | None  # Delaunay only
    regularization: dict
    configuration: dict = field(default_factory=dict)


@dataclass
class CellSpec:
    """One pixelized cell: its name, fiducial size and mesh builder."""

    cell: str  # "delaunay" | "pixelization"
    title: str
    fiducial_source_pixels: int
    # (cli, dataset, adapt_image, n_requested) -> MeshSetup
    build_mesh: Callable[..., MeshSetup]
    pinned: dict = field(default_factory=dict)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _nufftax_version():
    try:
        from importlib.metadata import version

        return version("nufftax")
    except Exception:  # noqa: BLE001
        return None


def _is_oom(exc) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "Out of memory" in text or "out of memory" in text


def _oom_record(exc, where):
    msg = str(exc).splitlines()[0][:500] if str(exc) else type(exc).__name__
    requested = None
    m = re.search(r"allocate ([0-9.]+)\s*(B|KiB|MiB|GiB|TiB)", str(exc))
    if m:
        scale = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}[m.group(2)]
        requested = int(float(m.group(1)) * scale)
    print(f"  RESOURCE_EXHAUSTED in {where} (recorded as a result): {msg}")
    return {"status": "oom", "requested_bytes": requested, "where": where, "error": msg}


def _free():
    """Drop compiled executables and device buffers held by JAX's caches."""
    jax.clear_caches()
    gc.collect()


def _arr(x):
    return jnp.asarray(getattr(x, "array", x))


def _abs_diff(a, b):
    return None if a is None or b is None else float(abs(float(a) - float(b)))


_HLO_PATTERNS = {
    "fft": r"\bfft\(",
    "while": r"\bwhile\(",
    "cholesky": r"cholesky|potrf",
    "triangular_solve": r"triangular-solve|trsm",
    "scatter": r"\bscatter\(",
    "custom_call": r"custom-call\(",
}


def hlo_census(func, *args) -> dict:
    """Op counts in the *optimized* HLO of ``jax.jit(func)`` (after XLA's CSE).

    A count above the structural minimum (e.g. two ``while`` loops holding an ``fft``
    where the likelihood needs one F build) is duplicated work the fused pipeline pays.
    """
    try:
        text = jax.jit(func).lower(*args).compile().as_text()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)[:300]}
    out = {k: len(re.findall(v, text)) for k, v in _HLO_PATTERNS.items()}
    out["status"] = "ok"
    return out


@dataclass
class DatasetSetup:
    """The loaded interferometer dataset and the geometry it was masked with."""

    instrument: str
    preset: dict
    dataset_path: Path
    pixel_scale: float
    real_space_shape: tuple
    mask_radius: float
    mask_radius_default: float
    real_space_mask: Any
    transformer_chunk: int | None
    preset_chunk: int | None
    dataset: Any
    n_vis: int
    n_image_pixels: int

    @property
    def mask_suffix(self) -> str:
        """``""`` at the preset radius, else ``_r<radius>`` (appended to output names)."""
        return mask_radius_suffix(self.mask_radius, self.mask_radius_default)


def mask_radius_suffix(mask_radius: float, mask_radius_default: float) -> str:
    """``""`` at the preset radius, else ``_r<radius>``: r3.5 outputs keep their names."""
    if float(mask_radius) == float(mask_radius_default):
        return ""
    return f"_r{float(mask_radius)}"


def load_dataset(instrument: str, mask_radius, profiling_root: Path, timer) -> DatasetSetup:
    """Auto-simulate if missing, mask at ``mask_radius`` (``None`` = preset) and load."""
    from simulators.interferometer import INSTRUMENTS

    from _profile_cli import auto_simulate_if_missing
    from instruments.interferometer import transformer_chunk_size_for

    print(f"\n--- Dataset loading [{instrument}] ---")
    preset = INSTRUMENTS[instrument]
    pixel_scale = preset["pixel_scale"]
    real_space_shape = preset["real_space_shape"]
    mask_radius_default = preset["mask_radius"]
    mask_radius = mask_radius_default if mask_radius is None else float(mask_radius)
    dataset_path = Path("dataset") / "interferometer" / instrument
    auto_simulate_if_missing(
        dataset_path,
        dataset_type="interferometer",
        instrument=instrument,
        workspace_root=profiling_root,
    )
    real_space_mask = al.Mask2D.circular(
        shape_native=real_space_shape, pixel_scales=pixel_scale, radius=mask_radius
    )
    preset_chunk = transformer_chunk_size_for(instrument)
    n_vis_preset = int(preset["n_visibilities"])
    # The per-call sparse path runs no NUFFT; the transformer's chunk only bounds the
    # one-off operator build + dirty image (unchunked, alma killed the laptop at 9.6 GB).
    transformer_chunk = (
        preset_chunk
        if preset_chunk is not None
        else (100_000 if n_vis_preset > LARGE_VIS else None)
    )

    def _transformer(uv_wavelengths, real_space_mask):
        return al.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            chunk_size=transformer_chunk,
        )

    with timer.section("dataset_load"):
        dataset = al.Interferometer.from_fits(
            data_path=dataset_path / "data.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
            real_space_mask=real_space_mask,
            transformer_class=_transformer,
        )
    n_vis = int(dataset.uv_wavelengths.shape[0])
    n_image_pixels = int(dataset.grids.pixelization.shape[0])
    print(f"  Visibilities: {n_vis}; masked pixels: {n_image_pixels}")
    if mask_radius != mask_radius_default:
        print(f"  mask radius {mask_radius} arcsec (preset {mask_radius_default})")
    print(f"  TransformerNUFFT chunk_size={transformer_chunk} (preset {preset_chunk})")
    return DatasetSetup(
        instrument=instrument,
        preset=preset,
        dataset_path=dataset_path,
        pixel_scale=pixel_scale,
        real_space_shape=real_space_shape,
        mask_radius=mask_radius,
        mask_radius_default=mask_radius_default,
        real_space_mask=real_space_mask,
        transformer_chunk=transformer_chunk,
        preset_chunk=preset_chunk,
        dataset=dataset,
        n_vis=n_vis,
        n_image_pixels=n_image_pixels,
    )


def build_model(mesh: MeshSetup):
    """Isothermal + ExternalShear near the simulator truth, the pixelized source, no lens light."""
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=ell[1], sigma=0.01)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
    field_model = af.Model(al.MassField, redshift=0.5, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=mesh.pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field_model)


def adapt_images_builder(adapt_image, mesh: MeshSetup) -> Callable:
    """``source_galaxy -> al.AdaptImages`` (the dicts are keyed on the galaxy object)."""

    def adapt_images_for(source_galaxy):
        kwargs = dict(
            galaxy_image_dict={source_galaxy: adapt_image},
            galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        )
        if mesh.image_plane_mesh_grid is not None:
            kwargs.update(
                galaxy_image_plane_mesh_grid_dict={source_galaxy: mesh.image_plane_mesh_grid},
                galaxy_name_image_plane_mesh_grid_dict={
                    "('galaxies', 'source')": mesh.image_plane_mesh_grid
                },
            )
        return al.AdaptImages(**kwargs)

    return adapt_images_for


def result_paths(
    cli, profiling_root: Path, instrument: str, cell: str, al_version: str, mask_suffix: str = ""
) -> tuple[Path, Path]:
    """``(json, png)`` for one pixelized cell; ``cell`` is the output-name stem."""
    from _profile_cli import resolve_output_paths

    default_dir = profiling_root / "results" / "breakdown" / "interferometer"
    if cli.config_name is not None and instrument != "alma":
        default_dir = default_dir / instrument
    basename = f"{cell}_breakdown_{instrument}_v{al_version}"
    dict_path, chart_path = resolve_output_paths(
        cli, default_dir=default_dir, default_basename=basename, cell=cell
    )
    if cli.config_name is None and cli.regularization == "constant_split":
        dict_path = dict_path.with_name(dict_path.stem + "_constant_split.json")
        chart_path = chart_path.with_name(chart_path.stem + "_constant_split.png")
    if mask_suffix:
        dict_path = dict_path.with_name(dict_path.stem + mask_suffix + ".json")
        chart_path = chart_path.with_name(chart_path.stem + mask_suffix + ".png")
    return dict_path, chart_path


#: The Delaunay cells' fiducial Hilbert count and the rectangular cells' fiducial side.
DELAUNAY_N_FIDUCIAL = 1500
RECT_SIDE_FIDUCIAL = 39  # 39 x 39 = 1521, the imaging campaign's rectangular tier
RECT_REGULARIZATION_COEFFICIENT = 1.0


def build_delaunay_mesh(cli, dataset, adapt_image, n_requested) -> MeshSetup:
    """Hilbert image mesh on the adapt image + ``al.mesh.Delaunay`` (``--regularization``)."""
    from _profile_cli import delaunay_regularization

    n = DELAUNAY_N_FIDUCIAL if n_requested is None else int(n_requested)
    image_mesh = al.image_mesh.Hilbert(pixels=n, weight_power=1.0, weight_floor=0.0)
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.real_space_mask, adapt_data=adapt_image
    )
    n_vertices = int(image_plane_mesh_grid.shape[0])
    scheme, regularization, provenance = delaunay_regularization(cli)
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(pixels=n_vertices, zeroed_pixels=0),
        regularization=regularization,
    )
    return MeshSetup(
        pixelization=pixelization,
        n_source_pixels=n_vertices,
        image_plane_mesh_grid=image_plane_mesh_grid,
        regularization=provenance,
        configuration={
            "mesh": "Delaunay",
            "image_mesh": "Hilbert(weight_power=1.0, weight_floor=0.0)",
            "hilbert_pixels": n,
            "delaunay_vertices": n_vertices,
            "edge_zeroed_pixels": 0,
        },
    )


def build_rectangular_mesh(cli, dataset, adapt_image, n_requested) -> MeshSetup:
    """Adaptive rectangular ``side x side`` mesh (``--rect-mesh``), ``Constant(1.0)``."""
    from _profile_cli import rect_mesh_classes

    side = RECT_SIDE_FIDUCIAL if n_requested is None else int(round(math.sqrt(n_requested)))
    mesh_cls = rect_mesh_classes(cli)[1]
    pixelization = al.Pixelization(
        mesh=mesh_cls(shape=(side, side), weight_power=1.0, weight_floor=0.0),
        regularization=al.reg.Constant(coefficient=RECT_REGULARIZATION_COEFFICIENT),
    )
    return MeshSetup(
        pixelization=pixelization,
        n_source_pixels=side * side,
        image_plane_mesh_grid=None,
        regularization={"scheme": "constant", "coefficient": RECT_REGULARIZATION_COEFFICIENT},
        configuration={
            "mesh": mesh_cls.__name__,
            "mesh_shape": [side, side],
            "rect_mesh": cli.rect_mesh,
        },
    )


def run(spec: CellSpec, cli, cell_args, script_file: str, profiling_root: Path) -> None:
    """Run one pixelized interferometer breakdown cell end to end."""
    from _production_config import observe_thread_env
    from _profile_cli import (
        check_pinned,
        device_info_dict,
        record_pinned_check,
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    revisions = source_revisions(profiling_root)
    print("--- Import provenance ---")
    print(f"  cell __file__ = {script_file}")
    import autogalaxy as ag

    for module in (aa, ag, al, af):
        print(f"  {module.__name__}.__file__ = {module.__file__}")
    for repo, rev in revisions.items():
        print(f"  {repo:<18} {rev}")
    print(f"  jax {jax.__version__} backend={jax.default_backend()} x64={jax.config.x64_enabled}")
    print(f"  nufftax {_nufftax_version()}")
    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is off: export JAX_ENABLE_X64=True (fp32 truncation).")

    load_start = list(os.getloadavg())
    instrument = cli.instrument or "sma"
    timer = timing.Timer()
    jit_records: dict[str, dict] = {}
    on_cpu = jax.default_backend() == "cpu"

    # ------------------------------------------------------------------
    # Dataset + sparse operator
    # ------------------------------------------------------------------
    ds = load_dataset(instrument, cell_args.mask_radius, profiling_root, timer)
    pixel_scale = ds.pixel_scale
    real_space_shape = ds.real_space_shape
    mask_radius = ds.mask_radius
    dataset_path = ds.dataset_path
    real_space_mask = ds.real_space_mask
    preset_chunk = ds.preset_chunk
    transformer_chunk = ds.transformer_chunk
    dataset = ds.dataset
    n_vis = ds.n_vis
    n_image_pixels = ds.n_image_pixels

    with timer.section("apply_sparse_operator"):
        dataset_sparse = dataset.apply_sparse_operator(
            method="nufft",
            batch_size=cli.sparse_batch_size,
            nufft_chunk_size=transformer_chunk,
            use_jax=True,
        )
    operator_build_s = timer.records[-1][1]
    operator = dataset_sparse.sparse_operator
    print(
        f"  sparse operator: extent {operator.y_shape}x{operator.x_shape} "
        f"(M={operator.M}), batch_size={operator.batch_size}, build {operator_build_s:.2f} s, "
        f"peak RSS {_peak_rss_mb():.0f} MB"
    )

    large_on_cpu = on_cpu and n_vis > LARGE_VIS
    n_repeats = (
        int(cell_args.n_repeats) if cell_args.n_repeats is not None else (3 if large_on_cpu else 10)
    )

    def jit_profile(func, label, *args):
        _, result = timing.jit_profile(
            func, label, *args, n_repeats=n_repeats, timer=timer, jit_records=jit_records
        )
        return result

    def per_call(label):
        return jit_records[label]["steady_per_call_s"]

    # ------------------------------------------------------------------
    # Adapt image + mesh + model
    # ------------------------------------------------------------------
    print("\n--- Adapt image + mesh ---")
    from _adapt_image_util import adapt_image_for_dataset

    with timer.section("adapt_image_build"):
        adapt_image = adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)

    n_requested = cli.source_pixels
    mesh = spec.build_mesh(cli, dataset, adapt_image, n_requested)
    n_src = int(mesh.n_source_pixels)
    pin_is_fiducial = n_requested is None and cli.regularization in (None, "adapt_split")
    print(
        f"  source pixels: {n_src} (requested {n_requested}); regularization {mesh.regularization}"
    )

    model = build_model(mesh)
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    _register_model_pytrees(model)
    params_tree = jax.tree_util.tree_map(jnp.asarray, instance)

    def settings_for(solver=None, mixed=None):
        return al.Settings(
            use_mixed_precision=cli.use_mixed_precision if mixed is None else mixed,
            positive_only_solver=solver or cell_args.solver,
        )

    settings = settings_for()

    adapt_images_for = adapt_images_builder(adapt_image, mesh)

    adapt_images = adapt_images_for(instance.galaxies.source)

    def tracer_from(params):
        return al.Tracer(galaxies=list(params.galaxies), fields=[params.fields])

    def fit_from(params, ds, fit_settings=None):
        return al.FitInterferometer(
            dataset=ds,
            tracer=tracer_from(params),
            adapt_images=adapt_images_for(params.galaxies.source),
            settings=fit_settings or settings,
            xp=jnp,
        )

    # ------------------------------------------------------------------
    # Library reference (sparse): every quantity the standalone steps need
    # ------------------------------------------------------------------
    print("\n--- Library sparse fit (jax.jit), reference quantities ---")
    trace_info: dict = {}

    def library_quantities(params):
        fit = fit_from(params, dataset_sparse)
        inv = fit.inversion
        mapper = inv.cls_list_from(cls=Mapper)[0]
        trace_info["inversion_class"] = type(inv).__name__
        trace_info["solver_used"] = inv.positive_only_solver_used
        trace_info["preconditioning_used"] = inv.positive_only_preconditioning_used
        trace_info["mapper_class"] = type(mapper).__name__
        ids = inv.solve_ids_to_keep
        trace_info["solve_subset"] = ids is not None
        return {
            "solve_ids_to_keep": _arr(ids) if ids is not None else jnp.arange(inv.total_params),
            "pix_indexes": _arr(mapper.pix_indexes_for_sub_slim_index),
            "pix_weights": _arr(mapper.pix_weights_for_sub_slim_index),
            "slim_index_for_sub": _arr(mapper.slim_index_for_sub_slim_index),
            "sub_fraction": _arr(mapper.over_sampler.sub_fraction),
            "mapping_matrix": _arr(inv.mapping_matrix),
            "data_vector": _arr(inv.data_vector),
            "curvature_matrix": _arr(inv.curvature_matrix),
            "regularization_matrix": _arr(inv.regularization_matrix),
            "reconstruction": _arr(inv.reconstruction),
            "log_det_curvature_reg": inv.log_det_curvature_reg_matrix_term,
            "log_det_regularization": inv.log_det_regularization_matrix_term,
            "regularization_term": inv.regularization_term,
            "fast_chi_squared": inv.fast_chi_squared,
            "figure_of_merit": fit.figure_of_merit,
        }

    with timer.section("library_sparse_reference_jit"):
        ref = jax.jit(library_quantities)(params_tree)
        ref = jax.tree_util.tree_map(lambda x: x.block_until_ready(), ref)
    _free()

    fom_library = float(ref["figure_of_merit"])
    print(f"  inversion class: {trace_info['inversion_class']} ({trace_info['mapper_class']})")
    print(
        f"  solver requested {cell_args.solver} -> used {trace_info['solver_used']} "
        f"(preconditioning {trace_info['preconditioning_used']})"
    )
    print(f"  library figure_of_merit = {fom_library!r}")
    if trace_info["inversion_class"] != "InversionInterferometerSparse":
        raise SystemExit(f"expected InversionInterferometerSparse, got {trace_info}")

    # Structural arrays the triplet step closes over (static per dataset).
    slim_index_for_sub = ref["slim_index_for_sub"]
    sub_fraction = ref["sub_fraction"]
    extent_index = jnp.asarray(
        np.asarray(dataset_sparse.transformer.real_space_mask.extent_index_for_masked_pixel),
        dtype=jnp.int32,
    )
    mapper_indices = jnp.asarray(np.arange(n_src))
    # Edge-zeroed pixels (the rectangular meshes): the library solves the subset system
    # and scatters exact zeros back (``AbstractInversion.reconstruction``).
    solve_ids = ref["solve_ids_to_keep"] if trace_info["solve_subset"] else None
    n_solved = int(solve_ids.shape[0]) if solve_ids is not None else n_src
    print(f"  solve subset: {trace_info['solve_subset']} ({n_solved} of {n_src} pixels solved)")

    dirty_image = jnp.asarray(np.asarray(operator.dirty_image))
    data_jnp = jnp.asarray(dataset.data.array)
    noise_jnp = jnp.asarray(dataset.noise_map.array)

    # ------------------------------------------------------------------
    # Steps 1-4: nested params -> stage prefixes on the library sparse fit
    # ------------------------------------------------------------------
    ray_trace_mesh = mesh.image_plane_mesh_grid is not None
    prefix_labels = {
        1: "Ray-trace grids" + (" (data + mesh vertices)" if ray_trace_mesh else " (data)"),
        2: "Border relocation + mesh interpolation (mapper)",
        3: "Mapping matrix L",
        4: "Regularization matrix H",
    }

    def setup_prefix_fn(upto: int):
        def fn(params):
            if upto == 1:
                tracer = tracer_from(params)
                out = [
                    jnp.stack(
                        [
                            _arr(g)
                            for g in tracer.traced_grid_2d_list_from(
                                grid=dataset_sparse.grids.pixelization, xp=jnp
                            )
                        ]
                    )
                ]
                if ray_trace_mesh:
                    out.append(
                        jnp.stack(
                            [
                                _arr(g)
                                for g in tracer.traced_grid_2d_list_from(
                                    grid=al.Grid2DIrregular(mesh.image_plane_mesh_grid), xp=jnp
                                )
                            ]
                        )
                    )
                return tuple(out)
            inv = fit_from(params, dataset_sparse).inversion
            mapper = inv.cls_list_from(cls=Mapper)[0]
            out = (
                _arr(mapper.pix_indexes_for_sub_slim_index),
                _arr(mapper.pix_weights_for_sub_slim_index),
            )
            if upto >= 3:
                out = out + (_arr(inv.mapping_matrix),)
            if upto >= 4:
                out = out + (_arr(inv.regularization_matrix),)
            return out

        return fn

    print("\n" + "=" * 70)
    print("SPARSE ARM — library per-evaluation steps")
    print("=" * 70)
    prefix_per_call: dict[int, float] = {}
    for upto in sorted(prefix_labels):
        print(f"\n--- Setup prefix 1..{upto}: {prefix_labels[upto]} ---")
        jit_profile(setup_prefix_fn(upto), f"setup_prefix_{upto}", params_tree)
        prefix_per_call[upto] = per_call(f"setup_prefix_{upto}")
        _free()
        print(f"  peak RSS so far: {_peak_rss_mb():.0f} MB")
    setup_split = timing.split_by_successive_differences(prefix_per_call, prefix_labels)

    # ------------------------------------------------------------------
    # Steps 5-10: standalone on the library's arrays
    # ------------------------------------------------------------------
    def sparse_triplets(pix_indexes, pix_weights):
        return mapper_util.sparse_triplets_from(
            pix_indexes_for_sub=pix_indexes,
            pix_weights_for_sub=pix_weights,
            slim_index_for_sub=slim_index_for_sub,
            fft_index_for_masked_pixel=extent_index,
            sub_fraction_slim=sub_fraction,
            return_rows_slim=False,
            xp=jnp,
        )

    def data_vector_sparse(mapping_matrix, dirty):
        return jnp.dot(mapping_matrix.T, dirty)

    def curvature_sparse_with(op):
        def fn(rows, cols, vals):
            return op.curvature_matrix_diag_from(rows=rows, cols=cols, vals=vals, S=n_src, xp=jnp)

        return fn

    curvature_sparse = curvature_sparse_with(operator)

    def solve_with(solver):
        def fn(data_vector, curvature_matrix, regularization_matrix):
            stats: dict = {}
            creg = curvature_matrix + regularization_matrix
            if solve_ids is not None:
                data_vector = data_vector[solve_ids]
                creg = creg[solve_ids][:, solve_ids]
            recon = inversion_util.reconstruction_positive_only_from(
                data_vector=data_vector,
                curvature_reg_matrix=creg,
                settings=settings_for(solver=solver),
                xp=jnp,
                solver=solver,
                stats=stats,
                preconditioning="jacobi",
            )
            if solve_ids is not None:
                recon = jnp.zeros(n_src).at[solve_ids].set(recon)
            info = {
                k: v
                for k, v in stats.items()
                if k in ("converged", "iterations", "certified", "passes")
            }
            return recon, info

        return fn

    def log_dets(curvature_matrix, regularization_matrix):
        creg = (curvature_matrix + regularization_matrix)[mapper_indices][:, mapper_indices]
        reg = regularization_matrix[mapper_indices][:, mapper_indices]
        ld_creg = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(creg))))
        ld_reg = 2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(reg))))
        return ld_creg, ld_reg

    def figure_of_merit_fn(
        recon,
        data_vector,
        curvature_matrix,
        regularization_matrix,
        ld_creg,
        ld_reg,
        visibilities,
        noise_map,
    ):
        chi_squared = (
            recon @ curvature_matrix @ recon
            - 2.0 * recon @ data_vector
            + jnp.sum(visibilities.real**2 / noise_map.real**2)
            + jnp.sum(visibilities.imag**2 / noise_map.imag**2)
        )
        reg_term = recon @ regularization_matrix @ recon
        noise_norm = jnp.sum(jnp.log(2 * jnp.pi * noise_map.real**2)) + jnp.sum(
            jnp.log(2 * jnp.pi * noise_map.imag**2)
        )
        return -0.5 * (chi_squared + reg_term + ld_creg - ld_reg + noise_norm)

    steps: list[tuple[str, float]] = [
        (prefix_labels[k], setup_split[prefix_labels[k]]) for k in sorted(prefix_labels)
    ]
    sub_rows: dict[str, float] = {}

    print("\n--- Step 5: sparse triplets ---")
    rows, cols, vals = jit_profile(
        sparse_triplets, "sparse_triplets", ref["pix_indexes"], ref["pix_weights"]
    )
    steps.append(("Sparse triplets (extent grid)", per_call("sparse_triplets")))
    nnz = int(rows.shape[0])
    _free()

    print("\n--- Step 6: D = Lᵀ d~ ---")
    dv = jit_profile(data_vector_sparse, "data_vector_sparse", ref["mapping_matrix"], dirty_image)
    steps.append(("Data vector D = Lᵀ d~", per_call("data_vector_sparse")))
    _free()

    print("\n--- Step 7: F = Aᵀ W~ A (blocked rfft2) ---")
    fm = jit_profile(curvature_sparse, "curvature_sparse", rows, cols, vals)
    steps.append(
        (
            f"Curvature matrix F = Aᵀ W~ A ({-(-n_src // operator.batch_size)} blocks of {operator.batch_size})",
            per_call("curvature_sparse"),
        )
    )
    _free()

    hm = ref["regularization_matrix"]
    solver = trace_info["solver_used"]
    print(f"\n--- Step 8: reconstruction ({solver}) ---")
    recon, solve_info = jit_profile(solve_with(solver), f"reconstruction_{solver}", dv, fm, hm)
    steps.append((f"Reconstruction ({solver})", per_call(f"reconstruction_{solver}")))
    solve_info = {k: int(v) for k, v in solve_info.items()}
    print(f"  solver stats: {solve_info}")
    _free()

    print("\n--- Step 9: log-det terms ---")
    ld_creg, ld_reg = jit_profile(log_dets, "log_dets", fm, hm)
    steps.append(("Log-det terms (2 Choleskys)", per_call("log_dets")))
    _free()

    print("\n--- Step 10: fast chi-squared + figure of merit ---")
    fom_steps = jit_profile(
        figure_of_merit_fn,
        "figure_of_merit",
        recon,
        dv,
        fm,
        hm,
        ld_creg,
        ld_reg,
        data_jnp,
        noise_jnp,
    )
    steps.append(("Fast chi-squared + figure of merit", per_call("figure_of_merit")))
    fom_steps = float(fom_steps)
    _free()

    def rel(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))

    step_fidelity = {
        "figure_of_merit_steps": fom_steps,
        "figure_of_merit_library": fom_library,
        "abs_diff_nats": _abs_diff(fom_steps, fom_library),
        "data_vector_max_rel_diff": rel(dv, ref["data_vector"]),
        "curvature_matrix_max_rel_diff": rel(fm, ref["curvature_matrix"]),
        "reconstruction_max_rel_diff": rel(recon, ref["reconstruction"]),
        "log_det_curvature_reg_abs_diff": _abs_diff(ld_creg, ref["log_det_curvature_reg"]),
        "log_det_regularization_abs_diff": _abs_diff(ld_reg, ref["log_det_regularization"]),
        "nnz_triplets": nnz,
    }
    print(f"  step fidelity: {step_fidelity}")

    # ------------------------------------------------------------------
    # Results writer (called after every section)
    # ------------------------------------------------------------------
    state: dict[str, Any] = {
        "full_pipeline": {"status": "not_run"},
        "dense": {"status": "not_run"},
        "solver_ab": None,
        "batch_size_sweep": None,
        "kernel_split": None,
        "fp32": None,
        "mixed_precision": None,
        "vmap": [],
    }
    al_version = al.__version__
    paths: dict = {}

    def write_results(stage: str):
        step_total = float(sum(t for _, t in steps))
        full = state["full_pipeline"]
        full_s = full.get("per_call_s")
        dense = state["dense"]
        summary = {
            "stage": stage,
            "autolens_version": al_version,
            "device": device_info_dict(),
            "instrument": instrument,
            "model": spec.cell,
            "transformer": "TransformerNUFFT",
            "use_mixed_precision": bool(cli.use_mixed_precision),
            "configuration": {
                "pixel_scale_arcsec": pixel_scale,
                "mask_radius_arcsec": mask_radius,
                "real_space_shape": list(real_space_shape),
                "image_pixels_masked": n_image_pixels,
                "visibilities": n_vis,
                "source_pixels": n_src,
                "source_pixels_requested": n_requested,
                **mesh.configuration,
                "lens_light": None,
                "lens_light_note": "simulators/interferometer.py puts no lens emission in the "
                "visibilities; mass fixed near truth (tight Gaussian priors), no lens light.",
                "inversion_path": "sparse",
                "inversion_class": trace_info["inversion_class"],
                "mapper_class": trace_info["mapper_class"],
                "solve_subset_edge_zeroed": bool(trace_info["solve_subset"]),
                "pixels_solved": n_solved,
                "positive_only_solver_requested": cell_args.solver,
                "positive_only_solver_used": trace_info["solver_used"],
                "nnls_preconditioning": trace_info["preconditioning_used"],
                "sparse_operator_method": "nufft",
                "sparse_batch_size": int(operator.batch_size),
                "curvature_blocks": int(-(-n_src // operator.batch_size)),
                "operator_extent_shape": [int(operator.y_shape), int(operator.x_shape)],
                "operator_M": int(operator.M),
                "operator_fft_shape": [2 * int(operator.y_shape), 2 * int(operator.x_shape)],
                "transformer_chunk_size": transformer_chunk,
                "transformer_chunk_size_preset": preset_chunk,
                "nufftax_version": _nufftax_version(),
                "n_repeats": n_repeats,
                "thread_env": observe_thread_env(),
                "host_load_avg_start": load_start,
                "host_load_avg_end": list(os.getloadavg()),
            },
            "regularization": mesh.regularization,
            "operator_build_s": float(operator_build_s),
            "steps": {k: float(v) for k, v in steps},
            "total_step_by_step": step_total,
            "setup_split": {k: float(v) for k, v in setup_split.items()},
            "setup_prefix_per_call_s": {str(k): float(v) for k, v in prefix_per_call.items()},
            "steps_sub_rows": {k: float(v) for k, v in sub_rows.items()},
            "full_pipeline_single_jit": float(full_s) if full_s is not None else None,
            "full_pipeline": full,
            "step_sum_over_full_jit": (step_total / full_s) if full_s else None,
            "figure_of_merit_reference": fom_library,
            "figure_of_merit_step_by_step": fom_steps,
            "step_fidelity": step_fidelity,
            "nnls": {"solver": trace_info["solver_used"], **solve_info},
            "dense_steps": dense.get("steps"),
            "dense_total_step_by_step": dense.get("total_step_by_step"),
            "dense": {k: v for k, v in dense.items() if k not in ("steps",)},
            "solver_ab": state["solver_ab"],
            "curvature_batch_size_sweep": state["batch_size_sweep"],
            "curvature_kernel_split": state["kernel_split"],
            "fp32_fft_arm": state["fp32"],
            "mixed_precision": state["mixed_precision"],
            "vmap": state["vmap"],
            "source_revisions": revisions,
            "jit_phases": jit_records,
            "peak_rss_mb": float(_peak_rss_mb()),
        }

        dict_path, chart_path = result_paths(
            cli, profiling_root, instrument, spec.cell, al_version, ds.mask_suffix
        )
        dict_path.write_text(json.dumps(summary, indent=2, default=float))
        paths["json"], paths["png"] = dict_path, chart_path
        _plot(summary, chart_path, spec)

        w = max(len(k) for k, _ in steps)
        print("\n" + "=" * 78)
        print(f"{spec.title} — {instrument.upper()} — sparse (W~) — stage {stage}")
        for i, (k, v) in enumerate(steps, 1):
            print(f"  {i:>2}. {k:<{w}} {v * 1e3:11.3f} ms")
        print(f"      {'TOTAL (step-by-step)':<{w}} {step_total * 1e3:11.3f} ms")
        if full_s:
            print(f"      {'Full pipeline (single JIT)':<{w}} {full_s * 1e3:11.3f} ms")
        if dense.get("steps"):
            print("  Dense (mapping) arm:")
            for k, v in dense["steps"].items():
                print(f"      {k:<{w}} {v * 1e3:11.3f} ms")
        print(f"  Peak RSS {summary['peak_rss_mb']:.0f} MB -> {dict_path}")
        print("=" * 78)

    write_results("sparse_steps")

    # ------------------------------------------------------------------
    # Full pipeline (AnalysisInterferometer on the sparse dataset)
    # ------------------------------------------------------------------
    print("\n--- Full pipeline: AnalysisInterferometer.log_likelihood_function (sparse) ---")

    def full_pipeline_with(solver):
        analysis = al.AnalysisInterferometer(
            dataset=dataset_sparse,
            adapt_images=adapt_images,
            settings=settings_for(solver=solver),
            use_jax=True,
        )

        def fn(params):
            return analysis.log_likelihood_function(instance=params)

        return fn

    full_fn = full_pipeline_with(cell_args.solver)
    try:
        fom_full = float(jit_profile(full_fn, "full_pipeline", params_tree))
        state["full_pipeline"] = {
            "status": "ok",
            "per_call_s": per_call("full_pipeline"),
            "figure_of_merit": fom_full,
            "abs_diff_vs_library_nats": _abs_diff(fom_full, fom_library),
            "solver": cell_args.solver,
        }
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        state["full_pipeline"] = _oom_record(exc, "full pipeline")
    _free()
    if state["full_pipeline"].get("status") == "ok":
        # The library fit's figure of merit without the Analysis wrapper, and an op census
        # of both fused programs against the standalone F / solve (duplicated work shows
        # up as extra while / fft ops that XLA's CSE did not merge).
        def library_fom(params):
            return fit_from(params, dataset_sparse).figure_of_merit

        jit_profile(library_fom, "library_fit_figure_of_merit", params_tree)
        sub_rows["FitInterferometer.figure_of_merit (library, single JIT)"] = per_call(
            "library_fit_figure_of_merit"
        )
        _free()
        state["full_pipeline"]["hlo_census"] = {
            "full_pipeline": hlo_census(full_fn, params_tree),
            "library_fit_figure_of_merit": hlo_census(library_fom, params_tree),
            "standalone_curvature_F": hlo_census(curvature_sparse, rows, cols, vals),
            "standalone_reconstruction": hlo_census(solve_with(solver), dv, fm, hm),
            "standalone_log_dets": hlo_census(log_dets, fm, hm),
        }
        print(f"  HLO census: {state['full_pipeline']['hlo_census']}")
        _free()
    write_results("full_pipeline")

    # ------------------------------------------------------------------
    # Solver A/B on the same system
    # ------------------------------------------------------------------
    print("\n--- Solver A/B (same D, F + H) ---")
    solver_ab: dict = {
        "note": "standalone step-8 rows on the same (D, F + H); fom recomputed "
        "with the step-9/10 functions"
    }
    for sv in ("pdip", "certified"):
        try:
            label = f"solver_ab_{sv}"
            r_sv, info_sv = jit_profile(solve_with(sv), label, dv, fm, hm)
            ld_c, ld_r = jax.jit(log_dets)(fm, hm)
            fom_sv = float(
                jax.jit(figure_of_merit_fn)(r_sv, dv, fm, hm, ld_c, ld_r, data_jnp, noise_jnp)
            )
            solver_ab[sv] = {
                "per_call_s": per_call(label),
                **{k: int(v) for k, v in info_sv.items()},
                "figure_of_merit": fom_sv,
                "negative_pixels": int(jnp.sum(r_sv < 0)),
                "zero_pixels": int(jnp.sum(r_sv == 0)),
            }
        except Exception as exc:  # noqa: BLE001
            solver_ab[sv] = (
                _oom_record(exc, f"solver {sv}")
                if _is_oom(exc)
                else {"status": "error", "error": traceback.format_exc()[-2000:]}
            )
        _free()
    if "per_call_s" in solver_ab.get("pdip", {}) and "per_call_s" in solver_ab.get("certified", {}):
        solver_ab["certified_over_pdip"] = (
            solver_ab["certified"]["per_call_s"] / solver_ab["pdip"]["per_call_s"]
        )
        solver_ab["figure_of_merit_abs_diff_nats"] = _abs_diff(
            solver_ab["certified"]["figure_of_merit"], solver_ab["pdip"]["figure_of_merit"]
        )
    share_denominator = state["full_pipeline"].get("per_call_s") or float(sum(t for _, t in steps))
    solver_ab["solve_share_of_full_pipeline"] = (
        per_call(f"reconstruction_{solver}") / share_denominator
    )
    if cell_args.solver_ab:
        other = "certified" if cell_args.solver == "pdip" else "pdip"
        try:
            fom_o = float(
                jit_profile(full_pipeline_with(other), f"full_pipeline_{other}", params_tree)
            )
            solver_ab[f"full_pipeline_{other}"] = {
                "per_call_s": per_call(f"full_pipeline_{other}"),
                "figure_of_merit": fom_o,
                "abs_diff_vs_library_nats": _abs_diff(fom_o, fom_library),
            }
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            solver_ab[f"full_pipeline_{other}"] = _oom_record(exc, f"full pipeline {other}")
        _free()
    state["solver_ab"] = solver_ab
    print(f"  solver A/B: {json.dumps(solver_ab, default=float)[:600]}")
    write_results("solver_ab")

    # ------------------------------------------------------------------
    # Lever 1: F block size sweep + kernel split
    # ------------------------------------------------------------------
    print("\n--- F kernel split (one block: scatter / FFT apply / gather) ---")
    B = int(operator.batch_size)
    n_blocks = -(-n_src // B)

    def block_scatter(rows, cols, vals):
        in_block = (cols >= 0) & (cols < B)
        bc = jnp.where(in_block, cols, 0).astype(jnp.int32)
        v = jnp.where(in_block, vals, 0.0)
        return jnp.zeros((operator.M, B), dtype=jnp.float64).at[rows, bc].add(v)

    def block_apply(fblock):
        return operator.apply_operator(fblock, xp=jnp)

    def block_gather(gblock, rows, cols, vals):
        from jax.ops import segment_sum

        return segment_sum(vals[:, None] * gblock[rows, :], cols, num_segments=n_src)

    try:
        fb = jit_profile(block_scatter, "kernel_scatter", rows, cols, vals)
        gb = jit_profile(block_apply, "kernel_fft_apply", fb)
        jit_profile(block_gather, "kernel_gather_segment_sum", gb, rows, cols, vals)
        parts = {
            "scatter_per_block_s": per_call("kernel_scatter"),
            "fft_apply_per_block_s": per_call("kernel_fft_apply"),
            "gather_segment_sum_per_block_s": per_call("kernel_gather_segment_sum"),
        }
        pred = n_blocks * sum(parts.values())
        state["kernel_split"] = {
            **parts,
            "n_blocks": n_blocks,
            "batch_size": B,
            "predicted_F_s": pred,
            "measured_F_s": per_call("curvature_sparse"),
            "fft_share_of_predicted": n_blocks * parts["fft_apply_per_block_s"] / pred,
            "scatter_share_of_predicted": n_blocks * parts["scatter_per_block_s"] / pred,
            "gather_share_of_predicted": n_blocks * parts["gather_segment_sum_per_block_s"] / pred,
            "note": "Each part jitted alone on one block (block 0) and scaled by n_blocks; the "
            "measured F runs them fused inside lax.fori_loop.",
        }
        del fb, gb
    except Exception as exc:  # noqa: BLE001
        state["kernel_split"] = (
            _oom_record(exc, "kernel split")
            if _is_oom(exc)
            else {"status": "error", "error": traceback.format_exc()[-2000:]}
        )
    _free()

    if cell_args.batch_size_sweep:
        sweep = []
        for bs in [int(b) for b in cell_args.batch_size_sweep.split(",") if b.strip()]:
            print(f"\n--- F with batch_size={bs} ---")
            op_b = dataclasses.replace(operator, batch_size=bs)
            try:
                f_b = jit_profile(
                    curvature_sparse_with(op_b), f"curvature_sparse_b{bs}", rows, cols, vals
                )
                sweep.append(
                    {
                        "batch_size": bs,
                        "n_blocks": -(-n_src // bs),
                        "per_call_s": per_call(f"curvature_sparse_b{bs}"),
                        "max_rel_diff_vs_default": rel(f_b, fm),
                        "status": "ok",
                    }
                )
                del f_b
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                sweep.append({"batch_size": bs, **_oom_record(exc, f"F batch_size {bs}")})
            del op_b
            _free()
        state["batch_size_sweep"] = sweep
    write_results("kernel_levers")

    # ------------------------------------------------------------------
    # Lever 5: fp32 FFT arm (measurement only)
    # ------------------------------------------------------------------
    if cell_args.fp32_fft_arm:
        print("\n--- fp32 FFT arm: F in float32 (measurement only) ---")
        khat32 = jnp.asarray(operator.Khat, dtype=jnp.complex64)
        y_s, x_s, M = int(operator.y_shape), int(operator.x_shape), int(operator.M)

        def curvature_fp32(rows, cols, vals):
            from jax import lax
            from jax.ops import segment_sum

            rows = rows.astype(jnp.int32)
            cols = cols.astype(jnp.int32)
            v32 = vals.astype(jnp.float32)
            offsets = jnp.arange(B, dtype=jnp.int32)
            c0 = jnp.zeros((n_src, n_blocks * B), dtype=jnp.float32)

            def body(i, C):
                start = i * B
                in_block = (cols >= start) & (cols < start + B)
                bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
                v = jnp.where(in_block, v32, 0.0)
                F = jnp.zeros((M, B), dtype=jnp.float32).at[rows, bc].add(v)
                img = F.T.reshape((B, y_s, x_s))
                pad = jnp.pad(img, ((0, 0), (0, y_s), (0, x_s)))
                G = jnp.fft.irfft2(jnp.fft.rfft2(pad) * khat32[None], s=(2 * y_s, 2 * x_s))
                G = G[:, :y_s, :x_s].reshape((B, M)).T
                cb = segment_sum(v32[:, None] * G[rows, :], cols, num_segments=n_src)
                width = jnp.minimum(B, jnp.maximum(0, n_src - start))
                cb = cb * (offsets < width)[None, :]
                return lax.dynamic_update_slice(C, cb, (0, start))

            C = lax.fori_loop(0, n_blocks, body, c0)[:, :n_src]
            return (0.5 * (C + C.T)).astype(jnp.float64)

        try:
            f32 = jit_profile(curvature_fp32, "curvature_sparse_fp32", rows, cols, vals)
            r32, _ = jax.jit(solve_with("pdip"))(dv, f32, hm)
            ld_c32, ld_r32 = jax.jit(log_dets)(f32, hm)
            fom32 = float(
                jax.jit(figure_of_merit_fn)(r32, dv, f32, hm, ld_c32, ld_r32, data_jnp, noise_jnp)
            )
            fom64 = solver_ab.get("pdip", {}).get("figure_of_merit", fom_steps)
            d = _abs_diff(fom32, fom64)
            state["fp32"] = {
                "per_call_s": per_call("curvature_sparse_fp32"),
                "fp64_per_call_s": per_call("curvature_sparse"),
                "speedup": per_call("curvature_sparse") / per_call("curvature_sparse_fp32"),
                "curvature_max_rel_diff": rel(f32, fm),
                "figure_of_merit_fp32_F": fom32,
                "figure_of_merit_fp64_F_pdip": fom64,
                "abs_diff_nats": d,
                "bar_nats": PRECISION_BAR_NATS,
                "holds_bar": bool(d is not None and np.isfinite(d) and d <= PRECISION_BAR_NATS),
                "note": "Script copy of curvature_matrix_diag_from in float32 (scatter, rfft2 / "
                "irfft2 with complex64 Khat, gather); solve / log-dets / chi-squared stay fp64 on "
                "the upcast F. The library hard-casts this path to float64, so "
                "use_mixed_precision never reaches it.",
            }
            del f32, r32
        except Exception as exc:  # noqa: BLE001
            state["fp32"] = (
                _oom_record(exc, "fp32 F")
                if _is_oom(exc)
                else {"status": "error", "error": traceback.format_exc()[-2000:]}
            )
        _free()
        write_results("fp32_arm")

    # ------------------------------------------------------------------
    # Mixed precision: fp64 library value in the same run
    # ------------------------------------------------------------------
    if cli.use_mixed_precision:
        print("\n--- Mixed precision: fp64 library figure of merit (same run) ---")
        fom64_lib = float(
            jax.jit(
                lambda p: fit_from(p, dataset_sparse, settings_for(mixed=False)).figure_of_merit
            )(params_tree)
        )
        d = _abs_diff(fom_library, fom64_lib)
        state["mixed_precision"] = {
            "figure_of_merit_mp": fom_library,
            "figure_of_merit_fp64": fom64_lib,
            "abs_diff_nats": d,
            "bar_nats": PRECISION_BAR_NATS,
            "holds_bar": bool(d is not None and d <= PRECISION_BAR_NATS),
            "note": "use_mixed_precision reaches the mapper / mapping matrix only; "
            "InterferometerSparseOperator hard-casts the W~ path to float64.",
        }
        _free()
        write_results("mixed_precision")

    # ------------------------------------------------------------------
    # Dense (mapping) comparison arm
    # ------------------------------------------------------------------
    dense_state: dict = {"status": "skipped"}
    if n_vis > cell_args.dense_max_vis:
        dense_state["reason"] = (
            f"N_vis {n_vis} > --dense-max-vis {cell_args.dense_max_vis}: T alone is "
            f"{n_vis * n_src * 16 / 2**30:.1f} GiB complex128"
        )
        print(f"\n--- Dense arm skipped: {dense_state['reason']} ---")
    else:
        print("\n" + "=" * 70)
        print("DENSE ARM — InversionInterferometerMapping (chunked transform)")
        print("=" * 70)
        dense_state = {"status": "running"}
        try:
            chunk_tr = al.TransformerNUFFT(
                uv_wavelengths=np.asarray(dataset.uv_wavelengths),
                real_space_mask=real_space_mask,
                chunk_size=transformer_chunk,
            )
            slim_rows, slim_cols = real_space_mask.slim_to_native_tuple
            slim_rows, slim_cols = jnp.asarray(slim_rows), jnp.asarray(slim_cols)
            n_y, n_x = real_space_mask.shape_native
            col_batch = int(cell_args.dense_transform_chunk)

            def transform_chunked(mapping_matrix):
                def one(column):
                    image = jnp.zeros((n_y, n_x), dtype=jnp.float64)
                    image = image.at[slim_rows, slim_cols].set(column).astype(jnp.complex128)
                    return chunk_tr._forward_native(image, xp=jnp)

                return jax.lax.map(one, mapping_matrix.T, batch_size=col_batch).T

            def dense_dv(tmm, vis, noise):
                return (
                    inversion_interferometer_util.data_vector_via_transformed_mapping_matrix_from(
                        transformed_mapping_matrix=tmm, visibilities=vis, noise_map=noise
                    )
                )

            def dense_curvature(tmm, noise):
                return inversion_util.curvature_matrix_via_mapping_matrix_from(
                    mapping_matrix=tmm.real, noise_map=noise.real, settings=settings, xp=jnp
                ) + inversion_util.curvature_matrix_via_mapping_matrix_from(
                    mapping_matrix=tmm.imag, noise_map=noise.imag, settings=settings, xp=jnp
                )

            dsteps = {k: float(v) for k, v in steps[:4]}
            tmm = jit_profile(
                transform_chunked, "dense_transformed_mapping_matrix", ref["mapping_matrix"]
            )
            dsteps[f"Transformed mapping matrix T (NUFFT, {col_batch}-column chunks)"] = per_call(
                "dense_transformed_mapping_matrix"
            )
            _free()
            dv_d = jit_profile(dense_dv, "dense_data_vector", tmm, data_jnp, noise_jnp)
            dsteps["Data vector D (dense, Re + Im)"] = per_call("dense_data_vector")
            _free()
            fm_d = jit_profile(dense_curvature, "dense_curvature_matrix", tmm, noise_jnp)
            dsteps["Curvature matrix F (dense Tᵀ N⁻¹ T, Re + Im)"] = per_call(
                "dense_curvature_matrix"
            )
            _free()
            r_d, info_d = jit_profile(solve_with(solver), "dense_reconstruction", dv_d, fm_d, hm)
            dsteps[f"Reconstruction ({solver})"] = per_call("dense_reconstruction")
            _free()
            ldc_d, ldr_d = jit_profile(log_dets, "dense_log_dets", fm_d, hm)
            dsteps["Log-det terms (2 Choleskys)"] = per_call("dense_log_dets")
            _free()
            fom_d = float(
                jit_profile(
                    figure_of_merit_fn,
                    "dense_figure_of_merit",
                    r_d,
                    dv_d,
                    fm_d,
                    hm,
                    ldc_d,
                    ldr_d,
                    data_jnp,
                    noise_jnp,
                )
            )
            dsteps["Fast chi-squared + figure of merit"] = per_call("dense_figure_of_merit")
            _free()
            gap = _abs_diff(fom_d, fom_steps)
            dense_state = {
                "status": "ok",
                "steps": dsteps,
                "total_step_by_step": float(sum(dsteps.values())),
                "transform": "script_chunked",
                "transform_column_batch": col_batch,
                "figure_of_merit": fom_d,
                "figure_of_merit_sparse": fom_steps,
                "sparse_vs_dense_abs_diff_nats": gap,
                "witness_threshold_nats": WITNESS_NATS,
                "witness_pass": (
                    bool(gap <= WITNESS_NATS) if not cli.use_mixed_precision else None
                ),
                "curvature_sparse_vs_dense_max_rel_diff": rel(fm, fm_d),
                "data_vector_sparse_vs_dense_max_rel_diff": rel(dv, dv_d),
                "nnls": {k: int(v) for k, v in info_d.items()},
                "note": "Steps 1-4 are shared with the sparse arm (same library mapper / L / H); "
                "T is formed column-chunked (the library's one-shot nufft2d2 over every column "
                "needs S x (2Ny) x (2Nx) complex128).",
            }
            print(f"  dense fom {fom_d!r} vs sparse {fom_steps!r}: |Δ| = {gap} nats")
            del tmm, dv_d, fm_d, r_d
        except Exception as exc:  # noqa: BLE001
            dense_state = (
                _oom_record(exc, "dense arm")
                if _is_oom(exc)
                else {"status": "error", "error": traceback.format_exc()[-3000:]}
            )
            print(f"  dense arm: {dense_state}")
        _free()
        if cell_args.dense_library and dense_state.get("status") == "ok":
            print("\n--- Dense library FitInterferometer (one-shot transform), jitted ---")

            def dense_library(params):
                return fit_from(params, dataset).figure_of_merit

            try:
                fom_dl = float(jit_profile(dense_library, "dense_library_fit", params_tree))
                dense_state["library"] = {
                    "status": "ok",
                    "per_call_s": per_call("dense_library_fit"),
                    "figure_of_merit": fom_dl,
                    "abs_diff_vs_sparse_library_nats": _abs_diff(fom_dl, fom_library),
                    "abs_diff_vs_dense_chunked_nats": _abs_diff(
                        fom_dl, dense_state["figure_of_merit"]
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                dense_state["library"] = _oom_record(exc, "dense library FitInterferometer")
            _free()
    state["dense"] = dense_state
    write_results("dense")

    # ------------------------------------------------------------------
    # vmap of the sparse full pipeline
    # ------------------------------------------------------------------
    if cell_args.vmap_batch and state["full_pipeline"].get("status") == "ok":
        for nb in sorted({int(b) for b in cell_args.vmap_batch.split(",")}, reverse=True):
            print(f"\n--- Sparse full pipeline under vmap, batch {nb} ---")
            pb = jax.tree_util.tree_map(
                lambda leaf, n=nb: jnp.broadcast_to(leaf, (n, *leaf.shape)), params_tree
            )
            try:
                t = timing.vmap_profile(
                    full_fn,
                    "full_pipeline",
                    pb,
                    nb,
                    n_repeats=n_repeats,
                    timer=timer,
                    jit_records=jit_records,
                )
                state["vmap"].append(
                    {
                        "batch": nb,
                        "status": "ok",
                        "per_call_s": float(t),
                        "batch_per_call_s": float(t * nb),
                    }
                )
                _free()
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                state["vmap"].append({"batch": nb, **_oom_record(exc, f"vmap batch {nb}")})
            _free()

    write_results("complete")

    # ------------------------------------------------------------------
    # Pinned figure of merit (recorded, never asserted)
    # ------------------------------------------------------------------
    pin = spec.pinned.get(instrument) if pin_is_fiducial and not cli.use_mixed_precision else None
    drift = []
    if pin is not None:
        for lbl, val in (("library", fom_library), ("step_by_step", fom_steps)):
            r = check_pinned(val, pin, label=lbl, rtol=1e-9)
            if r is not None:
                drift.append(r)
    record_pinned_check(paths["json"], pin, drift)

    # Fail loudly (after the JSON is on disk) on a broken reproduction.
    gap = step_fidelity["abs_diff_nats"]
    atol = 1.0 if cli.use_mixed_precision else 1e-3
    if gap is None or not np.isfinite(gap) or gap > atol:
        raise SystemExit(f"{spec.cell}: step-by-step figure of merit off the library by {gap} nats")
    if dense_state.get("witness_pass") is False:
        raise SystemExit(
            f"{spec.cell}: sparse vs dense witness failed ({dense_state['sparse_vs_dense_abs_diff_nats']} nats)"
        )
    print(
        f"\n  Done: witness step fidelity {gap:.3e} nats; dense {dense_state.get('sparse_vs_dense_abs_diff_nats')}"
    )


def _plot(summary: dict, chart_path: Path, spec: CellSpec) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["steps"].keys())
    times = list(summary["steps"].values())
    colors = ["#4C72B0"] * len(labels)
    if summary.get("dense_steps"):
        labels += [f"dense: {k}" for k in summary["dense_steps"]]
        times += list(summary["dense_steps"].values())
        colors += ["#C44E52"] * len(summary["dense_steps"])
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(labels) + 2))
    bars = ax.barh(range(len(labels)), times, color=colors, edgecolor="white", height=0.6)
    tmax = max(times) if times else 1.0
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + tmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t * 1e3:.3f} ms",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time per call (s)")
    c = summary["configuration"]
    full = summary.get("full_pipeline_single_jit")
    fig.suptitle(
        f"{spec.title} — {summary['instrument'].upper()} — sparse (W~) vs dense",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"AutoLens v{summary['autolens_version']} | {c['visibilities']} vis | "
        f"{c['source_pixels']} source px | step sum {summary['total_step_by_step'] * 1e3:.2f} ms | "
        + (f"full JIT {full * 1e3:.2f} ms" if full else "full JIT: n/a"),
        fontsize=8,
    )
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
