"""
Production presets for the CPU numba imaging cells
==================================================

The default imaging cells historically measured a configuration nothing in
production runs: ``over_sample_size_pixelization=1``, a fixed regularization
coefficient, an un-zeroed 1250/1500-vertex mesh, no positions penalty, a
60-Gaussian single-basis MGE, unpinned threads, and one prior-median instance
repeated ten times with the NNLS cross-evaluation warm-start memo silently on
(so repeats 2..10 self-seed). This module encodes what production *does* run,
field by field, with the file:line provenance of every value, so a cell can be
asked for the Euclid or the HST configuration instead of carrying its own.

Two production references are encoded, because no single cell can match both:

- ``EUCLID_VIS_PIX`` — the Euclid DR1 ``vis_pix`` stage,
  ``euclid_strong_lens_modeling_pipeline/scripts/initial_lens_model.py``
  (job 342301). Selected by ``--instrument euclid``.
- ``HST_SOURCE_PIX_2`` — the subhalo-validation ``source_pix[2]`` stage,
  ``subhalo_validation/scripts/imaging.py`` +
  ``scripts/pipelines/delaunay_adapt_split.py`` (job 342311). Selected by
  ``--instrument hst``.
- ``HST_RECT_ADAPT`` — the rectangular sibling of the above,
  ``subhalo_validation/scripts/pipelines/rectangular_adapt.py``, for the
  rectangular (``pixelization_numba``) cells.

Every preset has a ``legacy`` counterpart holding the pre-2026-09-08 cell
configuration verbatim, so ``--variant legacy`` reproduces the historic rows
rather than being a second, subtly different code path. The one field the
legacy presets do *not* restore is the light-profile radial-bin recipe: the
outer sub-size-1 bin was retired repo-wide (``[4, 2, 1]`` -> ``[4, 2, 2]``,
autolens_profiling#235) because sub-size 1 causes gradient issues, and that
change is unconditional.

Nothing here imports numpy or autolens at module scope: ``pin_thread_env`` has
to run *before* numpy is imported to have any effect on the BLAS thread pools,
so cells import this module first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Thread pinning
# ---------------------------------------------------------------------------

#: The five thread-count environment variables both production submit scripts
#: export to 1 before the Python process starts:
#: ``euclid_dr1_prelim/hpc/batch_cpu/submit_initial_lens_model_two_stage:48,161-166``
#: and ``subhalo_validation/hpc/batch_cpu/submit_source_pix:5,51-55``. A
#: Nautilus pool of 8 workers on 8 cores must not have each worker spawn its own
#: BLAS pool, so the per-worker likelihood is a genuinely single-threaded cost —
#: which is exactly the quantity a single-process profiling cell measures.
THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_thread_env(n_threads: int = 1) -> dict[str, Any]:
    """Set every ``THREAD_ENV_VARS`` entry to ``n_threads``, returning what happened.

    **Must be called before numpy is imported** — OpenBLAS / MKL read these once,
    when their shared library loads. Cells therefore import this module and call
    this in their path-setup header, above ``import numpy``.

    A variable already set to a *different* value is overridden and reported in
    the returned ``overridden`` dict rather than silently honoured, so a JSON row
    can never claim single-threaded when the shell said otherwise. The returned
    dict is what the cells record under ``configuration.thread_env``.
    """
    want = str(int(n_threads))
    overridden: dict[str, str] = {}
    preexisting: dict[str, str] = {}

    for name in THREAD_ENV_VARS:
        before = os.environ.get(name)
        if before is not None:
            preexisting[name] = before
            if before != want:
                overridden[name] = before
        os.environ[name] = want

    return {
        "n_threads": int(n_threads),
        "vars": list(THREAD_ENV_VARS),
        "set_to": want,
        "preexisting": preexisting,
        "overridden": overridden,
        "note": (
            "Set by _production_config.pin_thread_env before numpy import; "
            "mirrors the production SLURM submit scripts."
        ),
    }


def observe_thread_env() -> dict[str, Any]:
    """Record the thread environment as it stands, pinning nothing.

    What ``--variant legacy`` records: the historic rows were measured with
    whatever the shell happened to export (usually nothing), and pinning would
    change the number the legacy variant exists to reproduce.
    """
    return {
        "n_threads": None,
        "vars": list(THREAD_ENV_VARS),
        "set_to": None,
        "preexisting": {name: os.environ[name] for name in THREAD_ENV_VARS if name in os.environ},
        "overridden": {},
        "note": "Not pinned (legacy variant): recorded as found.",
    }


# ---------------------------------------------------------------------------
# The preset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionPreset:
    """One profiling cell's configuration, production or legacy.

    Every field carries the production file:line it was read from in the
    comment above it (audit 2026-09-08). A cell builds its dataset, mesh, model
    and analysis from these fields alone, so "does the cell match production?"
    is answered by reading this dataclass rather than by diffing two scripts.
    """

    #: Short name, e.g. ``euclid_vis_pix``; goes into the result JSON.
    name: str
    #: ``production`` or ``legacy``.
    variant: str
    #: Free text naming the production script + line range this encodes.
    provenance: str

    # --- source mesh --------------------------------------------------------
    #: ``delaunay`` (Hilbert image mesh) or ``rectangular`` (fixed grid).
    mesh_kind: str = "delaunay"
    #: Hilbert vertices requested. Euclid vis_pix :392 (500); subhalo
    #: delaunay_adapt_split.py :125-129 (1250).
    hilbert_pixels: int = 1250
    #: Edge points appended to the image-plane mesh grid *and* zeroed in the
    #: mesh. Euclid :448-455 + :613-615; subhalo :41 (`EDGE_PIXELS_TOTAL = 30`).
    #: PyAutoArray zeroes the last ``zeroed_pixels`` vertices of the image-mesh
    #: grid (autoarray/inversion/mesh/mesh/delaunay.py:58-72), so the mesh ends
    #: up with ``hilbert_pixels + edge_pixels`` vertices, 30 of them zeroed.
    edge_pixels: int = 0
    #: Hilbert weighting. Euclid :394-396; subhalo :37-38 — both 3.5 / 0.01.
    hilbert_weight_power: float = 1.0
    hilbert_weight_floor: float = 0.0
    #: Rectangular mesh side, for ``mesh_kind == "rectangular"``. Subhalo
    #: rectangular_adapt.py ``MESH_PIXELS_YX = 32`` (:38).
    rect_pixels_yx: int = 28

    # --- adapt image --------------------------------------------------------
    #: Signal-to-noise cap applied to the source adapt image before it drives
    #: the Hilbert mesh / adaptive regularization. Euclid: none (:422-428,
    #: :474-479). Subhalo: 3.0 (``pipelines/spec.py:9,:31``, ruled 2026-09-02 —
    #: an uncapped map starves fainter multiply-imaged features of mesh points).
    adapt_snr_cap: float | None = None

    # --- over-sampling ------------------------------------------------------
    #: Light-profile radial-bin recipe. The default is the subhalo recipe,
    #: ``subhalo_validation/scripts/imaging.py:179-184`` — `[4, 2, 2]` at
    #: `[0.1, 0.3]` after autolens_profiling#235 retired the outer sub-size-1
    #: bin. The two Euclid presets override it to `[4, 4, 2]`: the Euclid
    #: pipeline raised the middle bin in ``util.py:931-937``
    #: (euclid_strong_lens_modeling_pipeline#56, 2026-09-08) because sub-size 2
    #: in the 0.1-0.3" annulus under-integrates a compact source by ~0.6 %.
    lp_sub_size_list: tuple[int, ...] = (4, 2, 2)
    lp_radial_list: tuple[float, ...] = (0.1, 0.3)
    lp_centre: tuple[float, float] = (0.0, 0.0)
    #: Pixelization over-sampling rule: sub-size ``pix_high`` where the source
    #: adapt image (already a S/N map) exceeds ``pix_snr_cut``, ``pix_low``
    #: elsewhere. Euclid :498-511; subhalo :213,:216-255
    #: (``OVER_SAMPLE_SNR_CUT = 3.0``). ``None`` means the flat legacy value in
    #: ``pix_low``.
    pix_snr_cut: float | None = None
    pix_high: int = 4
    pix_low: int = 1

    # --- regularization -----------------------------------------------------
    #: ``adapt_split`` (free, production), ``constant_split`` or ``constant``
    #: (fixed coefficient, legacy).
    regularization: str = "constant_split"
    #: Fixed coefficient for the legacy schemes.
    regularization_coefficient: float = 1.0
    #: Priors for the free ``AdaptSplit``, identical in both production configs
    #: (``euclid .../config/priors/regularization/adapt_split.yaml``;
    #: ``subhalo .../adaptive_brightness.yaml:32-62``).
    adapt_split_priors: dict[str, Any] = field(
        default_factory=lambda: {
            "inner_coefficient": {"type": "LogUniform", "lower": 1.0e-6, "upper": 1.0e6},
            "outer_coefficient": {"type": "LogUniform", "lower": 1.0e-6, "upper": 1.0e6},
            "signal_scale": {"type": "Uniform", "lower": 0.0, "upper": 1.0},
        }
    )

    # --- positions penalty --------------------------------------------------
    #: ``positions_likelihood_from(factor=, minimum_threshold=)``. Euclid
    #: :544-546 (3.0 / 0.2); subhalo ``source_pix[2]`` has none (:384-388).
    positions_factor: float | None = None
    positions_minimum_threshold: float | None = None

    # --- lens light ---------------------------------------------------------
    #: ``mge_model_from``. Euclid :162-168 (20 x 2 bases = 40 Gaussians);
    #: subhalo :273-278 (30 x 2 = 60). The cells' historic 60 x 1 is a
    #: different basis structure, not just a different count.
    mge_total_gaussians: int = 60
    mge_gaussian_per_basis: int = 1

    # --- sequence + protocol ------------------------------------------------
    #: Production samples every free parameter, so the profiled stream is iid
    #: draws from the central 20 % of each prior rather than one instance
    #: repeated (which lets the NNLS memo self-seed).
    sequence: str = "iid"
    iid_unit_low: float = 0.4
    iid_unit_high: float = 0.6
    seed: int = 235

    #: NNLS cross-evaluation warm-start memo. Off in the default cells (see
    #: ``scripts/misc/nnls_warm_start/README.md``); the library default is
    #: ``true`` (``PyAutoArray/autoarray/config/general.yaml:11``) and both
    #: production projects leave it unset, so production runs with it on.
    memo: str = "off"

    #: Set by :func:`standard_size_preset` when the standard source-pixel count (or the
    #: borrowed instrument reference) changed the production preset: the reason, which makes
    #: the preset's pin and production witness inapplicable. ``None`` = the preset as recorded.
    size_override: str | None = None

    #: Set by :func:`standard_size_preset` (``--standard-size``, human ruling 2026-10-24): build
    #: the data and mesh as production does, not as the historic cells did — the source adapt
    #: image is production's subtracted S/N map (:func:`production_source_snr`), the over-sampling
    #: cut reads it UNCAPPED, the cap feeds only the mesh and regularization, and the rectangular
    #: mesh is ``RectangularBilinearAdaptImage`` with its free weight priors
    #: (:data:`PRODUCTION_DATA_SOURCES`). False keeps every existing preset as it was.
    production_data: bool = False

    #: ``imaging`` or ``interferometer``. An interferometer preset (:func:`interferometer_standard_preset`,
    #: likelihood-lifecycle 2b ruling R3) has no lens light, no over-sampling and no positions
    #: penalty, and its adapt image is the truth lensed source with no S/N cap.
    dataset_type: str = "imaging"

    @property
    def mesh_vertices(self) -> int:
        """Total Delaunay vertices: Hilbert points plus appended edge points."""
        return self.hilbert_pixels + self.edge_pixels

    def as_json(self) -> dict[str, Any]:
        """The ``configuration`` sub-dict every cell embeds in its result JSON."""
        payload: dict[str, Any] = {
            "preset": self.name,
            "variant": self.variant,
            "provenance": self.provenance,
            "mesh_kind": self.mesh_kind,
            "adapt_snr_cap": self.adapt_snr_cap,
            "over_sample_size_lp_rule": {
                "sub_size_list": list(self.lp_sub_size_list),
                "radial_list": list(self.lp_radial_list),
                "centre": list(self.lp_centre),
            },
            "over_sample_size_pixelization_rule": (
                {
                    "source_snr_cut": self.pix_snr_cut,
                    "sub_size_above": self.pix_high,
                    "sub_size_below": self.pix_low,
                }
                if self.pix_snr_cut is not None
                else {"flat_sub_size": self.pix_low}
            ),
            "regularization": (
                {"scheme": self.regularization, "free": True, "priors": self.adapt_split_priors}
                if self.regularization == "adapt_split"
                else {
                    "scheme": self.regularization,
                    "free": False,
                    "coefficient": self.regularization_coefficient,
                }
            ),
            "positions_penalty": (
                {
                    "factor": self.positions_factor,
                    "minimum_threshold": self.positions_minimum_threshold,
                }
                if self.positions_factor is not None
                else None
            ),
            "lens_light": (
                f"mge_{self.mge_total_gaussians * self.mge_gaussian_per_basis}"
                f"_linear_{self.mge_gaussian_per_basis}_basis"
            ),
            "mge_total_gaussians": self.mge_total_gaussians,
            "mge_gaussian_per_basis": self.mge_gaussian_per_basis,
            "sequence": {
                "kind": self.sequence,
                "unit_low": self.iid_unit_low,
                "unit_high": self.iid_unit_high,
                "seed": self.seed,
            },
            "memo": self.memo,
        }
        if self.size_override is not None:
            payload["size_override"] = self.size_override
        if self.mesh_kind == "delaunay":
            payload["mesh"] = {
                "kind": "delaunay",
                "hilbert_pixels": self.hilbert_pixels,
                "edge_pixels": self.edge_pixels,
                "zeroed_pixels": self.edge_pixels,
                "vertices": self.mesh_vertices,
                "hilbert_weight_power": self.hilbert_weight_power,
                "hilbert_weight_floor": self.hilbert_weight_floor,
            }
        else:
            payload["mesh"] = {
                "kind": "rectangular",
                "shape": [self.rect_pixels_yx, self.rect_pixels_yx],
                "source_pixels": self.rect_pixels_yx**2,
            }
        if self.dataset_type == "interferometer":
            # Ruling R3 (2b): no lens light, no over-sampling and no positions penalty in the
            # interferometer cells; the adapt image is the truth lensed source, uncapped.
            payload.update(
                dataset_type="interferometer",
                lens_light=None,
                mge_total_gaussians=0,
                over_sample_size_lp_rule=None,
                over_sample_size_pixelization_rule=None,
                positions_penalty=None,
                adapt_image="truth lensed source (lensed_source.fits), no S/N cap",
            )
            if self.mesh_kind == "rectangular":
                payload["mesh"]["class"] = "RectangularBilinearAdaptImage"
                payload["mesh"]["free_weights"] = PRODUCTION_RECT_WEIGHT_PRIORS
            return payload
        if self.production_data:
            payload["production_data"] = {
                "enabled": True,
                # The dataset's truth tracer stands in for the previous stage's maximum-likelihood
                # lens model whose light production subtracts (coordinator ruling, #456 fork a).
                "lens_light_source": "truth",
            }
            payload["production_source"] = PRODUCTION_DATA_SOURCES[
                "euclid" if self.name.startswith("euclid") else "subhalo"
            ]
            if self.mesh_kind == "rectangular":
                payload["mesh"]["class"] = "RectangularBilinearAdaptImage"
                payload["mesh"]["free_weights"] = PRODUCTION_RECT_WEIGHT_PRIORS
        return payload


# ---------------------------------------------------------------------------
# The presets
# ---------------------------------------------------------------------------

EUCLID_VIS_PIX = ProductionPreset(
    name="euclid_vis_pix",
    variant="production",
    provenance=(
        "euclid_strong_lens_modeling_pipeline/scripts/initial_lens_model.py "
        "(vis_pix stage, job 342301): mesh :613-615, image mesh :392-396, edge "
        ":448-455, regularization :617, over-sampling :498-511, sparse operator "
        ":364-365, positions :544-546, MGE :162-168, threads "
        "euclid_dr1_prelim/hpc/batch_cpu/submit_initial_lens_model_two_stage:48,161-166; "
        "light-profile radial bins util.py:931-937 ([4, 4, 2] since "
        "euclid_strong_lens_modeling_pipeline#56, 2026-09-08)"
    ),
    mesh_kind="delaunay",
    hilbert_pixels=500,
    edge_pixels=30,
    hilbert_weight_power=3.5,
    hilbert_weight_floor=0.01,
    adapt_snr_cap=None,
    lp_sub_size_list=(4, 4, 2),
    pix_snr_cut=3.0,
    pix_high=4,
    pix_low=2,
    regularization="adapt_split",
    positions_factor=3.0,
    positions_minimum_threshold=0.2,
    mge_total_gaussians=20,
    mge_gaussian_per_basis=2,
)

HST_SOURCE_PIX_2 = ProductionPreset(
    name="hst_source_pix_2",
    variant="production",
    provenance=(
        "subhalo_validation/scripts/imaging.py + scripts/pipelines/"
        "delaunay_adapt_split.py (source_pix[2], job 342311): mesh :41,:61-63, "
        "image mesh :37-38,:125-129, regularization :151, adapt S/N cap "
        "pipelines/spec.py:9,:31, over-sampling :213,:216-255, sparse operator "
        ":252-253,:807, no positions penalty :384-388, MGE :273-278, threads "
        "hpc/batch_cpu/submit_source_pix:5,51-55"
    ),
    mesh_kind="delaunay",
    hilbert_pixels=1250,
    edge_pixels=30,
    hilbert_weight_power=3.5,
    hilbert_weight_floor=0.01,
    adapt_snr_cap=3.0,
    pix_snr_cut=3.0,
    pix_high=4,
    pix_low=2,
    regularization="adapt_split",
    positions_factor=None,
    positions_minimum_threshold=None,
    mge_total_gaussians=30,
    mge_gaussian_per_basis=2,
)

HST_RECT_ADAPT = ProductionPreset(
    name="hst_rect_adapt",
    variant="production",
    provenance=(
        "subhalo_validation/scripts/pipelines/rectangular_adapt.py "
        "(rect_adapt source_pix[2], job 342311): RectangularBilinearAdaptImage "
        "shape (32, 32) (MESH_PIXELS_YX :38), regularization Adapt; the rest of "
        "the stage (adapt S/N cap, over-sampling, MGE, threads) as "
        "hst_source_pix_2"
    ),
    mesh_kind="rectangular",
    rect_pixels_yx=32,
    adapt_snr_cap=3.0,
    pix_snr_cut=3.0,
    pix_high=4,
    pix_low=2,
    regularization="adapt",
    positions_factor=None,
    positions_minimum_threshold=None,
    mge_total_gaussians=30,
    mge_gaussian_per_basis=2,
)

#: The Euclid rectangular reference does not exist — the Euclid pipeline's
#: pixelized stage is Delaunay only — so the rectangular cells run the subhalo
#: rectangular configuration under either instrument, with the instrument
#: choosing only the dataset. Recorded here rather than silently reused.
EUCLID_RECT_ADAPT = ProductionPreset(
    name="euclid_rect_adapt",
    variant="production",
    provenance=(
        HST_RECT_ADAPT.provenance + " — no Euclid rectangular production stage exists (the Euclid "
        "pipeline's pixelized stage is Delaunay only), so the Euclid "
        "rectangular row takes the rectangular mesh from this configuration and "
        "everything the stage owns (adapt S/N cap, light-profile radial bins, "
        "positions penalty, MGE) from the Euclid vis_pix stage, on the Euclid "
        "dataset"
    ),
    mesh_kind="rectangular",
    rect_pixels_yx=32,
    adapt_snr_cap=None,
    lp_sub_size_list=(4, 4, 2),
    pix_snr_cut=3.0,
    pix_high=4,
    pix_low=2,
    regularization="adapt",
    positions_factor=3.0,
    positions_minimum_threshold=0.2,
    mge_total_gaussians=20,
    mge_gaussian_per_basis=2,
)


def _legacy(name: str, mesh_kind: str, **overrides: Any) -> ProductionPreset:
    """A legacy preset: the pre-2026-09-08 cell configuration, verbatim.

    Everything the cells did before autolens_profiling#235 — an un-zeroed mesh
    with flat Hilbert weights, a fixed-coefficient regularization, a flat
    ``over_sample_size_pixelization = 1``, no positions penalty, a 60 x 1 MGE,
    the prior-median instance repeated, and the memo at its library default —
    except the light-profile radial bins, which moved to ``[4, 2, 2]``
    repo-wide (sub-size 1 causes gradient issues) and are not restored here.
    """
    base: dict[str, Any] = dict(
        name=name,
        variant="legacy",
        provenance=(
            "The cell's own pre-2026-09-08 configuration. Reproduces the "
            "historic rows except the light-profile radial bins, retired "
            "repo-wide from [4, 2, 1] to [4, 2, 2] by autolens_profiling#235."
        ),
        mesh_kind=mesh_kind,
        edge_pixels=0,
        hilbert_weight_power=1.0,
        hilbert_weight_floor=0.0,
        adapt_snr_cap=None,
        lp_sub_size_list=(4, 2, 2),
        lp_radial_list=(0.3, 0.6),
        pix_snr_cut=None,
        pix_low=1,
        positions_factor=None,
        positions_minimum_threshold=None,
        mge_total_gaussians=60,
        mge_gaussian_per_basis=1,
        sequence="repeated_prior_median",
        memo="library_default",
    )
    base.update(overrides)
    return ProductionPreset(**base)


LEGACY_DELAUNAY_NUMBA = _legacy(
    "legacy_delaunay_numba",
    "delaunay",
    hilbert_pixels=1250,
    regularization="constant_split",
    regularization_coefficient=1.0,
)

LEGACY_PIXELIZATION_NUMBA = _legacy(
    "legacy_pixelization_numba",
    "rectangular",
    rect_pixels_yx=28,
    regularization="constant",
    regularization_coefficient=1.0,
)


#: ``(cell_family, variant, instrument) -> preset``. ``cell_family`` is
#: ``delaunay_numba`` for the two Delaunay CPU cells and
#: ``pixelization_numba`` for the two rectangular ones.
_PRESETS: dict[tuple[str, str, str], ProductionPreset] = {
    ("delaunay_numba", "production", "euclid"): EUCLID_VIS_PIX,
    ("delaunay_numba", "production", "hst"): HST_SOURCE_PIX_2,
    ("pixelization_numba", "production", "euclid"): EUCLID_RECT_ADAPT,
    ("pixelization_numba", "production", "hst"): HST_RECT_ADAPT,
}

_LEGACY_PRESETS: dict[str, ProductionPreset] = {
    "delaunay_numba": LEGACY_DELAUNAY_NUMBA,
    "pixelization_numba": LEGACY_PIXELIZATION_NUMBA,
}


def preset_for(cell_family: str, instrument: str, variant: str = "production"):
    """Resolve the preset for ``cell_family`` at ``instrument`` under ``variant``.

    ``--variant legacy`` returns the cell's own pre-2026-09-08 configuration
    (instrument-independent, as the cells were). An instrument with no
    production reference raises rather than silently falling back, so a new
    instrument cannot quietly inherit Euclid's mesh.
    """
    if variant == "legacy":
        try:
            return _LEGACY_PRESETS[cell_family]
        except KeyError:
            raise KeyError(f"No legacy preset for cell family {cell_family!r}") from None

    try:
        return _PRESETS[(cell_family, variant, instrument)]
    except KeyError:
        raise KeyError(
            f"No {variant!r} preset for cell family {cell_family!r} at instrument "
            f"{instrument!r}. Production references exist for euclid and hst only "
            f"(see _production_config.EUCLID_VIS_PIX / HST_SOURCE_PIX_2); add one "
            f"with its production file:line provenance rather than reusing another."
        ) from None


#: The instrument whose production reference a standard-size cell borrows when its own
#: instrument has none (jwst, ao): the subhalo ``source_pix[2]`` stage is the high-resolution
#: imaging reference, so the high-resolution instruments take it, with only the dataset swapped.
STANDARD_SIZE_REFERENCE_INSTRUMENT = "hst"

#: The standard source-pixel count by numba cell family (human ruling 2026-10-23; the catalogue's
#: ``STANDARD_SOURCE_PIXELS``): Delaunay 1250 Hilbert vertices plus the preset's 30 zeroed edge
#: points (1280 built, in every standard-size cell), rectangular 35 x 35 = 1225.
STANDARD_HILBERT_PIXELS = 1250
STANDARD_RECT_PIXELS_YX = 35


def standard_size_preset(
    cell_family: str, instrument: str, source_pixels: int | None = None
) -> ProductionPreset:
    """The production preset of ``cell_family`` at ``instrument``, at the standard size.

    ``--standard-size`` (phase 2 of the likelihood-lifecycle epic) measures every instrument at
    one source-pixel count, so the size is the only field this changes: Delaunay presets keep
    their Hilbert weights and their ``edge_pixels`` zeroed edge points (1250 + 30 = 1280 vertices
    built; the ruling counts it as standard) and rectangular presets go to 35 x 35. An
    instrument with no production reference (jwst, ao — :func:`preset_for` raises for them)
    borrows :data:`STANDARD_SIZE_REFERENCE_INSTRUMENT`'s preset, recorded in its name, provenance
    and ``size_override``. Whenever anything changed, ``size_override`` says why, and the cells
    skip the preset's pinned value and production witness (both describe the preset as
    recorded). HST Delaunay is already 1250 + 30, so it comes back unchanged.

    ``source_pixels`` (rectangular only) builds the same cell at a declared size variant
    instead (``catalogue/size_variants.json``, ``--standard-size --source-pixels``; #468): an
    exact square, e.g. 784 = 28 x 28, the hst production preset's own mesh. None, or the
    standard 1225, is the standard size.
    """
    from dataclasses import replace

    if cell_family not in ("delaunay_numba", "pixelization_numba"):
        raise KeyError(f"No standard-size rule for cell family {cell_family!r}")
    borrowed = (cell_family, "production", instrument) not in _PRESETS
    base = preset_for(
        cell_family,
        instrument=STANDARD_SIZE_REFERENCE_INSTRUMENT if borrowed else instrument,
        variant="production",
    )
    if base.mesh_kind == "delaunay":
        if source_pixels is not None and source_pixels != STANDARD_HILBERT_PIXELS:
            raise ValueError("Delaunay has no declared size variant; it runs at the standard size")
        resized = base.hilbert_pixels != STANDARD_HILBERT_PIXELS
        size = {"hilbert_pixels": STANDARD_HILBERT_PIXELS}
        what = (
            f"Hilbert {base.hilbert_pixels} -> {STANDARD_HILBERT_PIXELS} (+{base.edge_pixels} edge)"
        )
    else:
        side = STANDARD_RECT_PIXELS_YX
        if source_pixels is not None and source_pixels != STANDARD_RECT_PIXELS_YX**2:
            side = int(round(source_pixels**0.5))
            if side * side != source_pixels:
                raise ValueError(
                    f"a rectangular size variant must be a square, not {source_pixels}"
                )
        resized = base.rect_pixels_yx != side
        size = {"rect_pixels_yx": side}
        what = f"rectangular {base.rect_pixels_yx}x{base.rect_pixels_yx} -> {side}x{side}"
    reasons = [
        "production data (human ruling 2026-10-24): subtracted S/N adapt image, uncapped "
        "over-sampling cut"
        + ("; RectangularBilinearAdaptImage mesh" if base.mesh_kind == "rectangular" else "")
    ]
    if borrowed:
        reasons.append(
            f"no {instrument} production reference: borrows the "
            f"{STANDARD_SIZE_REFERENCE_INSTRUMENT} preset {base.name!r} with the {instrument} dataset"
        )
    variant_size = size.get("rect_pixels_yx", STANDARD_RECT_PIXELS_YX) != STANDARD_RECT_PIXELS_YX
    if variant_size:
        reasons.append(
            f"declared size variant (catalogue/size_variants.json, #468): "
            f"{size['rect_pixels_yx'] ** 2} source pixels"
            + (f", {what}" if resized else f", the preset's own {what.split(' -> ')[1]} mesh")
        )
    elif resized:
        reasons.append(f"standard size (human ruling 2026-10-23): {what}")
    note = "; ".join(reasons)
    return replace(
        base,
        production_data=True,
        name=f"{base.name}_standard"
        + (f"_n{size['rect_pixels_yx'] ** 2}" if variant_size else "")
        + (f"_{instrument}" if borrowed else ""),
        provenance=f"{base.provenance} — {note}",
        size_override=note,
        **size,
    )


def witness_for(
    preset: ProductionPreset,
    cold_eval_median_s: float,
    instrument: str,
    host_class,
    mesh_class: str = "delaunay",
):
    """:func:`witness_verdict`, or ``not_applicable`` when no production job describes the cell.

    The production witness compares a cold eval against the production job of the preset as
    recorded, which ran a ``Delaunay`` mesh: a resized or borrowed preset (``size_override``) or
    a ``DelaunayNN`` mesh (``--mesh-class delaunay_nn``) is a different likelihood, so no band
    applies.
    """
    reason = None
    if preset.size_override is not None:
        reason = f"size_override: {preset.size_override}"
    elif mesh_class == "delaunay_nn":
        reason = "mesh_class delaunay_nn: the production job ran a Delaunay mesh"
    if reason is not None:
        return {
            "verdict": "not_applicable",
            "instrument": instrument,
            "host_class": host_class,
            "measured_cold_eval_median_s": cold_eval_median_s,
            "reason": reason,
        }
    return witness_verdict(cold_eval_median_s, instrument, host_class)


def numpy_dense_twin(dataset):
    """A shallow copy of ``dataset`` without its sparse operator: the eager NumPy dense route.

    The inversion factory dispatches on ``dataset.sparse_operator``, so the twin fits the same
    masked, over-sampled data through ``InversionImagingMapping`` with ``xp=np`` — the eager
    NumPy reference a numba sparse cell's parity witness compares against.
    """
    import copy

    twin = copy.copy(dataset)
    twin.sparse_operator = None
    return twin


# ---------------------------------------------------------------------------
# Builders (autolens imported lazily, as in _profile_cli)
# ---------------------------------------------------------------------------


def adapt_image_capped(adapt_image, preset: ProductionPreset):
    """The source adapt image with the preset's S/N cap applied, on a copy.

    Mirrors ``subhalo_validation/scripts/pipelines/spec.py:12-36``: every
    adaptive consumer (Hilbert image mesh, adaptive rectangular meshes,
    ``Adapt`` / ``AdaptSplit`` regularization) weights by the image's value, so
    an uncapped S/N map concentrates mesh points and loosens regularization on
    the brightest peak alone.
    """
    if preset.adapt_snr_cap is None:
        return adapt_image

    import autolens as al
    import numpy as np

    return al.Array2D(
        values=np.minimum(np.asarray(adapt_image), preset.adapt_snr_cap),
        mask=adapt_image.mask,
    )


def apply_production_over_sampling(
    dataset, adapt_image, preset: ProductionPreset, sparse_operator: bool = True
):
    """Apply both over-sampling maps, then re-apply the sparse CPU operator.

    Order matters and is production's: the light-profile radial bins go on
    first, then the pixelization map derived from the source adapt image, then
    ``apply_sparse_operator_cpu()`` — because ``apply_over_sampling`` returns a
    fresh ``Imaging`` that does not carry the precomputed sparse operator
    across, and losing it silently drops the fit to the dense inversion
    (``subhalo_validation/scripts/imaging.py:249-253``).

    ``adapt_image`` must already be capped (see ``adapt_image_capped``): the
    threshold is applied to the same map the mesh and regularization see.
    """
    import autolens as al
    import numpy as np

    over_sample_size_lp = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=list(preset.lp_sub_size_list),
        radial_list=list(preset.lp_radial_list),
        centre_list=[tuple(preset.lp_centre)],
    )

    if preset.pix_snr_cut is None:
        over_sample_size_pixelization: Any = preset.pix_low
    else:
        # The adapt image already IS a signal-to-noise map, so it is
        # thresholded directly. `over_sample_size_via_adapt_from` divides by the
        # noise map a second time, which put 76-90 % of the mask at sub-size 4
        # on the three validation lenses — rejected in production at
        # subhalo_validation/scripts/imaging.py:234-237.
        over_sample_size_pixelization = al.Array2D(
            values=np.where(
                np.asarray(adapt_image) > preset.pix_snr_cut,
                preset.pix_high,
                preset.pix_low,
            ),
            mask=dataset.mask,
        )

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size_lp,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    # The JAX standard-size cells pass sparse_operator=False and apply their own JAX operator.
    return dataset.apply_sparse_operator_cpu() if sparse_operator else dataset


# ---------------------------------------------------------------------------
# Production over-sampling for the JAX Delaunay-family cells
# ---------------------------------------------------------------------------

#: ``--instrument`` -> the production stage whose over-sampling the JAX
#: Delaunay-family imaging cells adopt under ``--over-sampling production``.
#: Only the two over-sampling fields are taken from the preset (the light-profile
#: radial bins and the pixelization S/N rule); the JAX cells keep their own mesh,
#: MGE and regularization, so the A100 re-measure isolates over-sampling.
_JAX_OVER_SAMPLING_PRESETS: dict[str, ProductionPreset] = {
    "euclid": EUCLID_VIS_PIX,
    "hst": HST_SOURCE_PIX_2,
}

#: Where each production pixelization over-sampling rule was read, re-checked
#: 2026-10-17 (autolens_profiling#430). Both threshold the source adapt image,
#: which in production is already a signal-to-noise map, and both threshold it
#: *uncapped*: the subhalo S/N cap of 3.0 (``pipelines/spec.py``) feeds only the
#: mesh and the regularization, never the over-sampling map.
JAX_OVER_SAMPLING_SOURCES: dict[str, str] = {
    "euclid": (
        "euclid_strong_lens_modeling_pipeline@9b1bc4c "
        "scripts/initial_lens_model.py:614-626 (source S/N > 3.0 -> 4, else 2, on "
        "the uncapped source adapt S/N map) + util.py:1505-1512 (light-profile "
        "bins [4, 4, 2] at [0.1, 0.3])"
    ),
    "hst": (
        "subhalo_validation@ab4bf32 scripts/imaging.py:213-255 "
        "apply_adaptive_pixelization_over_sampling (OVER_SAMPLE_SNR_CUT = 3.0 -> 4, "
        "else 2, on the uncapped source S/N map from "
        "galaxy_name_image_dict_via_result_from) + :175-184 over_sample_size_lp_from "
        "(light-profile bins [4, 2, 2] at [0.1, 0.3])"
    ),
}


def jax_over_sampling_preset(instrument: str) -> ProductionPreset:
    """The production preset whose over-sampling ``--over-sampling production`` uses.

    An instrument with no production stage raises rather than borrowing another
    instrument's rule.
    """
    try:
        return _JAX_OVER_SAMPLING_PRESETS[instrument]
    except KeyError:
        raise KeyError(
            f"--over-sampling production has no production reference for instrument "
            f"{instrument!r}; only {sorted(_JAX_OVER_SAMPLING_PRESETS)} exist "
            f"(EUCLID_VIS_PIX / HST_SOURCE_PIX_2). Use --over-sampling flat1."
        ) from None


def apply_production_over_sampling_jax(dataset, lensed_source, instrument: str):
    """Re-over-sample a masked JAX-cell dataset at ``instrument``'s production rule.

    Returns ``(dataset, record)``. ``record`` is the result-JSON block. Its
    ``over_sample_size_pixelization_rule`` has the shape the task witness reads:
    ``{source_snr_cut, sub_size_above, sub_size_below}``.

    Production thresholds a *signal-to-noise* map: the source adapt image
    ``galaxy_name_image_dict_via_result_from`` returns. The profiling datasets
    cache the truth lensed source as *flux* (``_adapt_image_util``), so the S/N
    map here is ``lensed_source / noise_map``, the truth-model equivalent of
    production's fitted-model S/N map, and uncapped as in both production rules.
    The cell's own adapt image (mesh + regularization) is left untouched.

    The caller re-applies the sparse operator afterwards when it uses one:
    ``apply_over_sampling`` returns a fresh ``Imaging`` without it.
    """
    import autolens as al
    import numpy as np

    preset = jax_over_sampling_preset(instrument)

    over_sample_size_lp = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=list(preset.lp_sub_size_list),
        radial_list=list(preset.lp_radial_list),
        centre_list=[tuple(preset.lp_centre)],
    )

    source_snr = np.asarray(lensed_source) / np.asarray(dataset.noise_map)
    pix_values = np.where(source_snr > preset.pix_snr_cut, preset.pix_high, preset.pix_low)
    over_sample_size_pixelization = al.Array2D(values=pix_values, mask=dataset.mask)

    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size_lp,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    n_above = int(np.sum(pix_values == preset.pix_high))
    record = {
        "over_sampling": "production",
        "over_sampling_preset": preset.name,
        "over_sampling_provenance": JAX_OVER_SAMPLING_SOURCES[instrument],
        "over_sample_size_lp_rule": {
            "sub_size_list": list(preset.lp_sub_size_list),
            "radial_list": list(preset.lp_radial_list),
            "centre": list(preset.lp_centre),
        },
        "over_sample_size_pixelization_rule": {
            "source_snr_cut": preset.pix_snr_cut,
            "sub_size_above": preset.pix_high,
            "sub_size_below": preset.pix_low,
            "snr_map": "truth lensed_source / noise_map (uncapped)",
            "pixels_above": n_above,
            "pixels_below": int(pix_values.size - n_above),
        },
    }
    return dataset, record


#: The ``configuration`` fields a ``--over-sampling flat1`` row records beside
#: its existing ``over_sample_size_lp_rule``, so every JAX Delaunay-family row
#: states its pixelization over-sampling instead of leaving it implicit.
FLAT1_OVER_SAMPLING_RECORD: dict[str, Any] = {
    "over_sampling": "flat1",
    "over_sample_size_pixelization_rule": {"flat_sub_size": 1},
}


def regularization_model(preset: ProductionPreset):
    """The regularization the preset names: a free ``af.Model`` or a fixed instance.

    Production samples the ``AdaptSplit`` coefficients, so the production
    presets hand back an ``af.Model`` with the production priors set explicitly
    (this repo ships no ``config/priors/regularization/``, so relying on config
    lookup would silently give the library defaults). The legacy presets hand
    back the fixed instance the historic rows were measured with.
    """
    import autofit as af
    import autolens as al

    if preset.regularization == "adapt_split":
        reg = af.Model(al.reg.AdaptSplit)
        priors = preset.adapt_split_priors
        reg.inner_coefficient = af.LogUniformPrior(
            lower_limit=priors["inner_coefficient"]["lower"],
            upper_limit=priors["inner_coefficient"]["upper"],
        )
        reg.outer_coefficient = af.LogUniformPrior(
            lower_limit=priors["outer_coefficient"]["lower"],
            upper_limit=priors["outer_coefficient"]["upper"],
        )
        reg.signal_scale = af.UniformPrior(
            lower_limit=priors["signal_scale"]["lower"],
            upper_limit=priors["signal_scale"]["upper"],
        )
        return reg

    if preset.regularization == "constant_split":
        return al.reg.ConstantSplit(coefficient=preset.regularization_coefficient)

    if preset.regularization == "constant":
        return al.reg.Constant(coefficient=preset.regularization_coefficient)

    if preset.regularization == "adapt":
        return af.Model(al.reg.Adapt)

    raise ValueError(f"Unknown regularization scheme {preset.regularization!r}")


def mge_lens_bulge(preset: ProductionPreset, mask_radius: float):
    """The MGE lens-light model the preset names.

    Basis structure, not just Gaussian count: Euclid runs 20 Gaussians across 2
    bases, subhalo 30 across 2; the cells' historic 60 x 1 is a different model.
    """
    import autolens as al

    return al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=preset.mge_total_gaussians,
        gaussian_per_basis=preset.mge_gaussian_per_basis,
        centre_prior_is_uniform=True,
    )


def positions_likelihood(preset: ProductionPreset, dataset_path: Path, tracer):
    """The Euclid positions penalty, rebuilt from the dataset's ``positions.json``.

    Production gets this from a previous stage's result
    (``result.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)``,
    ``initial_lens_model.py:544-546``), which solves the multiple images from
    the max-likelihood tracer and takes ``factor x`` their maximum source-plane
    separation, floored at ``minimum_threshold``. A profiling cell has no
    previous stage, so the committed simulator positions stand in for the
    solved ones and the threshold is computed the same way, from this cell's
    own tracer (``PyAutoLens/autolens/analysis/result.py:254-275``).

    Returns ``None`` when the preset has no positions penalty.
    """
    if preset.positions_factor is None:
        return None

    import autolens as al
    import numpy as np

    positions = al.from_json(file_path=str(Path(dataset_path) / "positions.json"))

    separations = al.SourceMaxSeparation(
        data=positions, noise_map=None, tracer=tracer
    ).max_separation_of_plane_positions

    threshold = preset.positions_factor * float(np.nanmax(separations))
    if (
        preset.positions_minimum_threshold is not None
        and threshold < preset.positions_minimum_threshold
    ):
        threshold = preset.positions_minimum_threshold

    return al.PositionsLH(positions=positions, threshold=threshold)


# ---------------------------------------------------------------------------
# Instance sequence
# ---------------------------------------------------------------------------


def iid_instances(model, n: int, preset: ProductionPreset) -> list:
    """``n`` independent instances, unit values uniform in the central 20 % of every prior.

    Lifted from ``scripts/misc/nnls_warm_start/delaunay_numba_nnls_iterations.py``
    (:377-388), which is where the regime was first measured. Production samples
    every free parameter, so no two consecutive likelihood evaluations a Nautilus
    worker sees are related; repeating one prior-median instance instead lets the
    NNLS cross-evaluation warm-start memo seed itself from a 100 %-correct
    previous solve, which is not a cost production ever pays.
    """
    import numpy as np

    rng = np.random.default_rng(preset.seed)
    return [
        model.instance_from_unit_vector(
            unit_vector=list(
                rng.uniform(preset.iid_unit_low, preset.iid_unit_high, size=model.prior_count)
            )
        )
        for _ in range(n)
    ]


def adapt_images_for(instance, image_plane_mesh_grid=None, adapt_image=None):
    """Rebuild ``al.AdaptImages`` for ``instance``.

    The dicts are keyed on the instance's own source-galaxy object, so they
    cannot be shared across an iid sequence — each instance needs its own.
    Lifted from ``delaunay_numba_nnls_iterations.py:392-411``.

    ``adapt_image`` is passed whenever it is available: adaptive schemes
    (``AdaptSplit``, ``Adapt``, the adaptive rectangular meshes) read it, and
    the non-adaptive ones ignore it, so the variants then differ only in the
    scheme.
    """
    import autolens as al

    kwargs: dict[str, Any] = {}
    if image_plane_mesh_grid is not None:
        kwargs["galaxy_image_plane_mesh_grid_dict"] = {
            instance.galaxies.source: image_plane_mesh_grid
        }
        kwargs["galaxy_name_image_plane_mesh_grid_dict"] = {
            "('galaxies', 'source')": image_plane_mesh_grid
        }
    if adapt_image is not None:
        kwargs["galaxy_image_dict"] = {instance.galaxies.source: adapt_image}
        kwargs["galaxy_name_image_dict"] = {"('galaxies', 'source')": adapt_image}
    return al.AdaptImages(**kwargs)


# ---------------------------------------------------------------------------
# Settings + timing protocol
# ---------------------------------------------------------------------------


def analysis_settings(
    preset: ProductionPreset, memo: str, use_mixed_precision: bool = False, **extra: Any
) -> tuple[Any, dict[str, Any]]:
    """``(al.Settings, provenance)`` with the NNLS memo explicitly controlled.

    Belt and braces: the memo is turned off both through
    ``al.Settings(nnls_warm_start_memo=False)`` and through the
    ``AUTOARRAY_NNLS_WARM_START=0`` env kill-switch the library reads at
    ``autoarray/inversion/inversion/nnls_memo.py:63``, because the two gates sit
    at different layers. The setting is then read back off the constructed
    ``Settings`` object and recorded, so the JSON states what actually took
    effect rather than what was asked for.

    ``memo="library_default"`` (the ``legacy`` variant) touches neither gate.
    """
    import autolens as al

    kwargs: dict[str, Any] = {"use_border_relocator": True}
    env_before = os.environ.get("AUTOARRAY_NNLS_WARM_START")

    if memo == "off":
        kwargs["nnls_warm_start_memo"] = False
        os.environ["AUTOARRAY_NNLS_WARM_START"] = "0"
    elif memo == "on":
        kwargs["nnls_warm_start_memo"] = True
        os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)

    if use_mixed_precision:  # the JAX standard-size cells' --use-mixed-precision
        kwargs["use_mixed_precision"] = True
    # Further Settings fields a cell sets explicitly (the interferometer numba arms' route /
    # gate, likelihood-lifecycle 2b R4); recorded as given.
    kwargs.update(extra)
    settings = al.Settings(**kwargs)

    provenance = {
        "requested": memo,
        **({"settings_extra": {k: v for k, v in extra.items()}} if extra else {}),
        "settings_nnls_warm_start_memo": bool(settings.nnls_warm_start_memo),
        "env_AUTOARRAY_NNLS_WARM_START": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
        "env_before": env_before,
        "note": (
            "Both gates are set for memo=off: the Settings flag and the "
            "AUTOARRAY_NNLS_WARM_START env kill-switch "
            "(autoarray/inversion/inversion/nnls_memo.py:63). "
            "settings_nnls_warm_start_memo is read back off the constructed "
            "Settings, so it records what took effect. The library default is "
            "true and both production projects leave it unset."
        ),
    }
    return settings, provenance


def timing_summary(per_eval_s: list[float], n_cold: int) -> dict[str, Any]:
    """Split a per-instance timing sequence into the cold and warm quantities.

    The protocol, and why it is what production is comparable to:

    - instance 0 is the compile / warm-up evaluation and is **discarded** (the
      numba kernels JIT on first call, and on a cold numba cache the very first
      ``psf_weighted_data_from`` can return garbage);
    - the next ``n_cold`` evaluations are the **cold evals**, the quantity
      PyAutoFit logs as "Log Likelihood Function Evaluation Time"
      (``autofit/non_linear/search/updater.py:311-317``) — one timed evaluation
      in a single process, which is what a pooled ``s/sample x speed-up`` comes
      back to;
    - every remaining evaluation is a **warm iid eval**, reported as median and
      mean. The median leads: an iid stream has a long right tail (each draw is
      a different active-set size) that drags the mean, and every historic row
      in this repo is an arithmetic mean of ten repeats of ONE instance, which
      is a different quantity entirely.
    """
    import numpy as np

    if len(per_eval_s) < 2 + n_cold:
        raise ValueError(
            f"timing_summary needs at least {2 + n_cold} evaluations "
            f"(1 warm-up + {n_cold} cold + >=1 warm), got {len(per_eval_s)}"
        )

    warmup_s = float(per_eval_s[0])
    cold = [float(t) for t in per_eval_s[1 : 1 + n_cold]]
    warm = [float(t) for t in per_eval_s[1 + n_cold :]]

    return {
        "protocol": (
            "instance 0 discarded (numba compile / cold-cache hazard); the next "
            "n_cold evaluations are cold evals comparable to PyAutoFit's logged "
            "'Log Likelihood Function Evaluation Time'; the rest are warm iid "
            "evaluations."
        ),
        "warmup_incl_numba_compile_s": warmup_s,
        "cold_eval_s": cold,
        "cold_eval_median_s": float(np.median(cold)),
        "warm_iid_median_s": float(np.median(warm)),
        "warm_iid_mean_s": float(np.mean(warm)),
        "n_warm": len(warm),
        "n_cold": len(cold),
    }


#: The production reference every production-mode row is judged against
#: (search.summary files, audit 2026-09-08). ``cold_eval_s`` is the range of the
#: logged "Log Likelihood Function Evaluation Time"; ``s_per_sample`` is the
#: pooled 8-core throughput and ``speed_up`` the ratio PyAutoFit records, so
#: ``s_per_sample x speed_up ~ cold_eval_s``.
PRODUCTION_REFERENCE: dict[str, dict[str, Any]] = {
    "euclid": {
        "job": "342301",
        "stage": "vis_pix",
        "cold_eval_s": [0.69, 1.13],
        "s_per_sample": [0.2099, 0.3261],
        "speed_up": [3.15, 4.35],
        "cores": 8,
    },
    "hst": {
        "job": "342311",
        "stage": "source_pix[2] / rect_adapt",
        "cold_eval_s": [0.52, 0.81],
        "s_per_sample": [0.1678, 0.2327],
        "speed_up": [3.0, 4.3],
        "cores": 8,
    },
}

#: The witness: a production-mode cold eval must land within this factor of the
#: production range (decision 2 on autolens_profiling#235, human, 2026-09-08).
WITNESS_FACTOR = 1.5


def _is_reference_host_class(host_class: str) -> bool:
    """The dashboard's own rule (``build_dashboard.is_reference_host_class``), imported lazily.

    One definition of "reference host class" for trend points (P6), GPU-only markers (P9) and
    this witness (P5). The tooling module is stdlib-only at import, so the lazy import keeps
    this module importable before numpy (``pin_thread_env``).
    """
    import sys

    tooling = str(Path(__file__).resolve().parent / "scripts" / "misc" / "tooling")
    if tooling not in sys.path:
        sys.path.insert(0, tooling)
    from build_dashboard import is_reference_host_class

    return is_reference_host_class(host_class)


def witness_verdict(
    cold_eval_median_s: float, instrument: str, host_class: str | None
) -> dict[str, Any]:
    """Judge one cold-eval median against the production reference range.

    ``host_class`` is the row's ``--config-name`` (``hpc_ral_cpu_fp64``, ``local_cpu_fp64``, …;
    ``None`` when the cell ran untagged). The reference range was logged on 8-core RAL
    production nodes, so a measurement is judged against it only on a reference host class
    (timing-noise audit row P5, fix phase 6). Anywhere else the verdict is ``INCONCLUSIVE`` with
    the reason "off reference host class (<class>)": no band turns a laptop row into a
    reference-host row. The band itself (``WITNESS_FACTOR``) is unchanged.
    """
    reference = PRODUCTION_REFERENCE.get(instrument)
    if reference is None:
        return {"verdict": "no_reference", "instrument": instrument, "host_class": host_class}

    low, high = reference["cold_eval_s"]
    allowed = [low / WITNESS_FACTOR, high * WITNESS_FACTOR]
    in_band = (low / WITNESS_FACTOR) <= cold_eval_median_s <= (high * WITNESS_FACTOR)
    if host_class and _is_reference_host_class(host_class):
        verdict, reason = (
            ("PASS", "cold-eval median inside the band")
            if in_band
            else ("FAIL", "cold-eval median outside the band")
        )
    else:
        verdict = "INCONCLUSIVE"
        reason = f"off reference host class ({host_class or 'untagged'})"
    return {
        "instrument": instrument,
        "job": reference["job"],
        "reference_cold_eval_s": [low, high],
        "factor": WITNESS_FACTOR,
        "allowed_s": allowed,
        "measured_cold_eval_median_s": cold_eval_median_s,
        "host_class": host_class,
        # Where the median falls, recorded whatever the host: a reader can still see it.
        "in_band": in_band,
        "verdict": verdict,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# The production cell model: one definition for the numba cells and the JAX
# standard-size cells (likelihood-lifecycle 2a, human ruling N1 on #456)
# ---------------------------------------------------------------------------

#: The production cell families and the likelihood variants they build.
STANDARD_FAMILY = {
    "delaunay": "delaunay_numba",
    "delaunay_nn": "delaunay_numba",
    "rectangular": "pixelization_numba",
}


def production_image_mesh_grid(
    dataset, adapt_image, preset: ProductionPreset, mask_radius: float, mask=None
):
    """The preset's Hilbert image-plane mesh plus its zeroed circle-edge points.

    Production appends ``edge_pixels`` points on the mask edge and zeroes exactly those vertices,
    so the reconstruction cannot put flux outside the data (Euclid initial_lens_model.py:448-455;
    subhalo delaunay_adapt_split.py:49-56). ``adapt_image`` is the (capped) source adapt image.
    ``mask`` defaults to ``dataset.mask``; an interferometer cell passes its ``real_space_mask``.
    """
    import autolens as al

    mask = dataset.mask if mask is None else mask
    image_mesh = al.image_mesh.Hilbert(
        pixels=preset.hilbert_pixels,
        weight_power=preset.hilbert_weight_power,
        weight_floor=preset.hilbert_weight_floor,
    )
    grid = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_image)
    if preset.edge_pixels:
        grid = al.image_mesh.append_with_circle_edge_points(
            image_plane_mesh_grid=grid,
            centre=mask.mask_centre,
            radius=mask_radius + mask.pixel_scale / 2.0,
            n_points=preset.edge_pixels,
        )
    return grid


def production_mesh(preset: ProductionPreset, *, n_vertices: int | None = None, mesh_class=None):
    """The source mesh: ``mesh_class`` (default ``al.mesh.Delaunay``) over ``n_vertices`` with the
    preset's edge points zeroed, or — rectangular presets — ``mesh_class(shape=(n, n))`` at the
    preset's side (the numba cells pass ``rect_mesh_classes(cli)[0]``)."""
    import autolens as al

    if preset.mesh_kind == "delaunay":
        cls = mesh_class or al.mesh.Delaunay
        return cls(pixels=int(n_vertices), zeroed_pixels=preset.edge_pixels)
    shape = (preset.rect_pixels_yx, preset.rect_pixels_yx)
    # production's source_pix[2] mesh, weights free (imaging --standard-size, interferometer R3)
    if preset.production_data or preset.dataset_type == "interferometer":
        import autofit as af

        mesh = af.Model(al.mesh.RectangularBilinearAdaptImage, shape=shape)
        priors = PRODUCTION_RECT_WEIGHT_PRIORS
        mesh.weight_power = af.UniformPrior(
            lower_limit=priors["weight_power"]["lower"], upper_limit=priors["weight_power"]["upper"]
        )
        mesh.weight_floor = af.LogUniformPrior(
            lower_limit=priors["weight_floor"]["lower"], upper_limit=priors["weight_floor"]["upper"]
        )
        return mesh
    return mesh_class(shape=shape)


def production_model(preset: ProductionPreset, mask_radius: float, mesh, regularization=None):
    """The cell model: the preset's MGE lens light, Isothermal + shear mass, and a source
    pixelization of ``mesh`` with ``regularization`` (default :func:`regularization_model`, made
    here after the lens priors: the production priors on a free coefficient). Pass only a fixed
    regularization instance (no priors), e.g. the legacy Delaunay ``--regularization`` choice."""
    import autofit as af
    import autolens as al

    # Construction order is part of the model: PyAutoFit orders a unit vector by prior creation,
    # so the seeded iid draws only land on the same parameters if the MGE, mass, shear and
    # regularization priors are made in this order (the numba cells' historic order).
    lens_bulge = mge_lens_bulge(preset, mask_radius=mask_radius)
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

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass)
    field = af.Model(al.MassField, redshift=0.5, shear=shear)
    if regularization is None:
        regularization = regularization_model(preset)
    if isinstance(regularization, af.Model) or isinstance(mesh, af.Model):
        pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
    else:
        pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)


def standard_jax_dataset(dataset, dataset_path, preset: ProductionPreset, sparse_batch_size=None):
    """``(dataset, mesh adapt image, record)`` for a JAX standard-size cell:
    :func:`production_standard_data` with the JAX w-tilde operator under ``--sparse``
    (``sparse_batch_size``) instead of the numba CPU one — the numba cells' exact data."""
    return production_standard_data(dataset, dataset_path, preset, sparse=sparse_batch_size)


def standard_jax_cell(
    variant: str,
    instrument: str,
    *,
    dataset,
    adapt_image,
    dataset_path,
    mask_radius: float,
    memo: str = "off",
    use_mixed_precision: bool = False,
    rect_mesh_cls=None,
    source_pixels: int | None = None,
):
    """Everything a JAX standard-size cell takes from the production preset, built once.

    ``dataset`` must already carry the cell's over-sampling (:func:`standard_jax_dataset`) and
    ``adapt_image`` be the capped one. Returns a dict: ``preset``, ``image_plane_mesh_grid``
    (Delaunay family, else None), ``n_source_pixels``, ``model``, ``instance`` (the first draw of
    the preset's seeded iid sequence — the numba cells' instance 0), ``settings``,
    ``memo_provenance``, ``positions_likelihood_list``, ``reg_scheme``, ``reg_provenance`` and
    ``record`` (what the row stores as ``configuration.production_preset``). ``source_pixels``:
    a declared rectangular size variant (:func:`standard_size_preset`), else the standard size.
    """
    import autolens as al

    preset = standard_size_preset(STANDARD_FAMILY[variant], instrument, source_pixels)
    if preset.mesh_kind == "delaunay":
        grid = production_image_mesh_grid(dataset, adapt_image, preset, mask_radius)
        n_source_pixels = int(grid.shape[0])
        mesh_cls = al.mesh.DelaunayNN if variant == "delaunay_nn" else al.mesh.Delaunay
        mesh = production_mesh(preset, n_vertices=n_source_pixels, mesh_class=mesh_cls)
    else:
        grid = None
        n_source_pixels = preset.rect_pixels_yx**2
        mesh = production_mesh(preset, mesh_class=rect_mesh_cls)
    model = production_model(preset, mask_radius, mesh)
    instance = iid_instances(model, 1, preset)[0]
    settings, memo_provenance = analysis_settings(
        preset, memo, use_mixed_precision=use_mixed_precision
    )
    positions = positions_likelihood(
        preset,
        dataset_path=dataset_path,
        tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
    )
    record = {**preset.as_json(), "instance": "iid draw 0 of the preset sequence (seeded)"}
    return {
        "preset": preset,
        "image_plane_mesh_grid": grid,
        "n_source_pixels": n_source_pixels,
        "model": model,
        "instance": instance,
        "settings": settings,
        "memo_provenance": memo_provenance,
        "positions_likelihood_list": [positions] if positions is not None else None,
        "reg_scheme": preset.regularization,
        "reg_provenance": preset.as_json()["regularization"],
        "record": record,
    }


def numpy_dense_reference(dataset, adapt_images, instance, **analysis_kwargs) -> float:
    """The parity reference: the eager NumPy dense-mapping log-likelihood of ``instance``.

    ``al.AnalysisImaging(use_jax=False)`` on :func:`numpy_dense_twin` of the cell's dataset, with
    the cell analysis's own ``settings`` / ``positions_likelihood_list``: independent of the
    cell's sparse operator (numba or JAX w-tilde), so a sparse cell's algorithm is checked too
    (human ruling (c) on #456).
    """
    import autolens as al

    analysis = al.AnalysisImaging(
        dataset=numpy_dense_twin(dataset),
        adapt_images=adapt_images,
        use_jax=False,
        **analysis_kwargs,
    )
    return float(analysis.log_likelihood_function(instance=instance))


#: Where the ``production_data`` construction is read (re-checked 2026-10-24 against
#: subhalo_validation@122079f on /mnt/c and euclid_strong_lens_modeling_pipeline@9b1bc4c).
PRODUCTION_DATA_SOURCES = {
    "subhalo": {
        "adapt_image": (
            "PyAutoGalaxy@7472211 autogalaxy/analysis/adapt_images/adapt_images.py:138 "
            "galaxy_name_image_dict_via_result_from: the source's subtracted S/N map (data minus "
            "the other galaxies' model images, over the noise map), floored at "
            "adapt_minimum_percent (0.01, subhalo_validation config/general.yaml:28) x max"
        ),
        "mesh_and_regularization_adapt_image": (
            "subhalo_validation scripts/pipelines/spec.py:9-36 capped_galaxy_name_image_dict_from "
            "(S/N capped at 3.0); used by delaunay_adapt_split.py:123,:165 and "
            "rectangular_adapt.py:109"
        ),
        "over_sampling": (
            "subhalo_validation scripts/imaging.py:213-255 apply_adaptive_pixelization_over_sampling: "
            "sub-size 4 where the UNCAPPED source S/N > 3.0, else 2; light-profile bins :175-184"
        ),
        "rectangular_mesh": (
            "subhalo_validation scripts/pipelines/rectangular_adapt.py:74-96 (source_pix[2]): "
            "af.Model(al.mesh.RectangularBilinearAdaptImage, shape) with al.reg.Adapt; weight "
            "priors config/priors/mesh/rectangular.yaml:43-61"
        ),
        "model_stand_in": (
            "no previous-stage fit exists in a profiling cell: the dataset's truth tracer "
            "(tracer.json) stands in for the maximum-likelihood model whose lens light is "
            "subtracted"
        ),
    },
    "euclid": {
        "adapt_image": (
            "euclid_strong_lens_modeling_pipeline scripts/initial_lens_model.py:538-543 "
            "galaxy_name_image_dict_via_result_from (subtracted S/N, uncapped) for the Hilbert "
            "mesh; adapt_minimum_percent 0.01 (config/general.yaml:22)"
        ),
        "over_sampling": "initial_lens_model.py:609-626: sub-size 4 where the source S/N > 3.0, else 2",
        "model_stand_in": (
            "the dataset's truth tracer stands in for the previous stage's maximum-likelihood model"
        ),
    },
}

#: Production's free ``RectangularBilinearAdaptImage`` weights (subhalo_validation
#: config/priors/mesh/rectangular.yaml:43-61, equal to the library default config).
PRODUCTION_RECT_WEIGHT_PRIORS = {
    "weight_power": {"type": "Uniform", "lower": 0.0, "upper": 10.0},
    "weight_floor": {"type": "LogUniform", "lower": 1e-5, "upper": 1.0},
}


def production_source_snr(dataset, dataset_path):
    """Production's source adapt image: the subtracted S/N map of the lensed source.

    ``(data - lens-galaxy model image) / noise_map``, the lens light PSF-blurred on the dataset's
    light-profile grid, floored at ``adapt_minimum_percent`` x its maximum — what
    ``galaxy_name_image_dict_via_result_from`` returns for ``('galaxies', 'source')``. The
    dataset's truth tracer stands in for the previous stage's maximum-likelihood model.
    Returns ``(snr, minimum_percent)``; the map is uncapped.
    """
    import autolens as al
    import numpy as np

    try:
        from autoconf import conf

        minimum_percent = float(conf.instance["general"]["adapt"]["adapt_minimum_percent"])
    except Exception:  # noqa: BLE001 — the library default, as both production configs set it
        minimum_percent = 0.01
    tracer = al.from_json(file_path=Path(dataset_path) / "tracer.json")
    planes = tracer.blurred_image_2d_list_from(
        grid=dataset.grids.lp, blurring_grid=dataset.grids.blurring, psf=dataset.psf
    )
    lens = np.asarray(planes[0])
    snr = (np.asarray(dataset.data) - lens) / np.asarray(dataset.noise_map)
    snr = np.maximum(snr, minimum_percent * np.max(snr))
    return al.Array2D(values=snr, mask=dataset.mask), minimum_percent


def production_standard_data(dataset, dataset_path, preset: ProductionPreset, sparse=None):
    """``(dataset, mesh adapt image, record)`` built as production builds them.

    Light-profile bins first, then the subtracted S/N map (:func:`production_source_snr`), then
    the pixelization over-sampling from the UNCAPPED map (sub-size ``pix_high`` where S/N >
    ``pix_snr_cut``, else ``pix_low``), and the mesh / regularization adapt image capped at
    ``adapt_snr_cap`` when the preset has one. ``sparse``: ``"cpu"`` applies the numba CPU
    operator, an int the JAX operator at that batch size, None neither. The numba and JAX
    standard-size cells both call this, so they fit the same data.
    """
    import autolens as al
    import numpy as np

    lp = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=list(preset.lp_sub_size_list),
        radial_list=list(preset.lp_radial_list),
        centre_list=[tuple(preset.lp_centre)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=lp, over_sample_size_pixelization=preset.pix_low
    )
    snr, minimum_percent = production_source_snr(dataset, dataset_path)
    above = np.asarray(snr) > preset.pix_snr_cut
    pix = al.Array2D(values=np.where(above, preset.pix_high, preset.pix_low), mask=dataset.mask)
    dataset = dataset.apply_over_sampling(over_sample_size_lp=lp, over_sample_size_pixelization=pix)
    if sparse == "cpu":
        dataset = dataset.apply_sparse_operator_cpu()
    elif sparse:
        dataset = dataset.apply_sparse_operator(batch_size=int(sparse))
    mesh_adapt = snr
    if preset.adapt_snr_cap is not None:
        mesh_adapt = al.Array2D(
            values=np.minimum(np.asarray(snr), preset.adapt_snr_cap), mask=dataset.mask
        )
    config = preset.as_json()
    record = {
        "over_sampling_preset": preset.name,
        "over_sample_size_lp_rule": config["over_sample_size_lp_rule"],
        "over_sample_size_pixelization_rule": {
            **config["over_sample_size_pixelization_rule"],
            "threshold_map": "uncapped subtracted S/N (production)",
            "pixels_above": int(above.sum()),
            "pixels_below": int((~above).sum()),
        },
        "adapt_image": "subtracted S/N (truth tracer), floored",
        "lens_light_source": "truth",
        "adapt_minimum_percent": minimum_percent,
        "mesh_adapt_snr_cap": preset.adapt_snr_cap,
    }
    return dataset, mesh_adapt, record


# ---------------------------------------------------------------------------
# Likelihood-lifecycle 2b: the interferometer standard-size cell (ruling R3) and its
# visibility-chunked dense parity reference (rulings R6, R7)
# ---------------------------------------------------------------------------

#: The interferometer likelihood variants with a standard-size production cell.
INTERFEROMETER_STANDARD_VARIANTS = ("delaunay", "rectangular")

#: Ruling R3 (human 2026-10-25, autolens_profiling#469), recorded on every preset it builds.
INTERFEROMETER_STANDARD_RULING = (
    "likelihood-lifecycle 2b ruling R3 (human 2026-10-25): the hst source_pix[2] mesh and "
    "regularization at the standard size, no lens light, the truth lensed-source adapt image "
    "with no S/N cap"
)


def interferometer_standard_preset(variant: str) -> ProductionPreset:
    """The standard-size interferometer production cell of ``variant`` (ruling R3).

    No interferometer production pipeline exists to read a preset from, so the cells borrow the
    hst ``source_pix[2]`` stage's mesh and regularization (the workspace interferometer SLaM uses
    the same, ``autolens_workspace/scripts/interferometer/features/pixelization/slam.py``):

    - ``delaunay``: Hilbert 1250 (weight power 3.5, floor 0.01) plus 30 zeroed circle-edge points
      (1280 vertices built), free ``AdaptSplit`` with the production priors;
    - ``rectangular``: ``RectangularBilinearAdaptImage`` 35 x 35 = 1225 with free weight priors,
      free ``Adapt``.

    Everything else differs from the imaging stage and is recorded as such: no lens light (the
    simulated visibilities carry none), no over-sampling and no positions penalty, and the adapt
    image is the dataset's truth lensed source (``lensed_source.fits``) with no S/N cap. The mass
    model and its priors are the imaging cells' (Isothermal + ExternalShear near the truth).
    ``size_override`` is set, so no production witness or imaging pin applies.
    """
    from dataclasses import replace

    if variant not in INTERFEROMETER_STANDARD_VARIANTS:
        raise KeyError(
            f"No interferometer standard preset for {variant!r}; "
            f"choose from {INTERFEROMETER_STANDARD_VARIANTS}"
        )
    base = HST_SOURCE_PIX_2 if variant == "delaunay" else HST_RECT_ADAPT
    if variant == "delaunay":
        size = {"hilbert_pixels": STANDARD_HILBERT_PIXELS}
        what = f"Hilbert {STANDARD_HILBERT_PIXELS} + {base.edge_pixels} zeroed edge points"
    else:
        size = {"rect_pixels_yx": STANDARD_RECT_PIXELS_YX}
        what = f"RectangularBilinearAdaptImage {STANDARD_RECT_PIXELS_YX}x{STANDARD_RECT_PIXELS_YX}"
    note = (
        f"{INTERFEROMETER_STANDARD_RULING}: {what}, {base.regularization} regularization; "
        "borrows the hst preset's mesh and regularization only"
    )
    return replace(
        base,
        name=f"interferometer_{base.name}_standard",
        provenance=f"{base.provenance} — {note}",
        dataset_type="interferometer",
        adapt_snr_cap=None,
        pix_snr_cut=None,
        pix_low=1,
        positions_factor=None,
        positions_minimum_threshold=None,
        mge_total_gaussians=0,
        production_data=False,
        size_override=note,
        **size,
    )


def interferometer_production_model(preset: ProductionPreset, mesh, regularization=None):
    """The interferometer cell model: Isothermal + shear near the truth and the pixelized source.

    :func:`production_model` without the lens light (the interferometer cells have none). The
    priors and their construction order (mass, shear, then the regularization) are fixed here,
    so every setup's seeded iid draw lands on the same parameters.
    """
    import autofit as af
    import autolens as al

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
    field = af.Model(al.MassField, redshift=0.5, shear=shear)
    if regularization is None:
        regularization = regularization_model(preset)
    if isinstance(regularization, af.Model) or isinstance(mesh, af.Model):
        pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
    else:
        pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    return af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)


def iid_unit_vectors(n: int, prior_count: int, preset: ProductionPreset) -> list[list[float]]:
    """The unit vectors :func:`iid_instances` draws (same seed, same order), for the record."""
    import numpy as np

    rng = np.random.default_rng(preset.seed)
    return [
        [float(u) for u in rng.uniform(preset.iid_unit_low, preset.iid_unit_high, size=prior_count)]
        for _ in range(n)
    ]


def standard_interferometer_cell(
    variant: str,
    instrument: str,
    *,
    dataset,
    adapt_image,
    mask_radius: float,
    memo: str = "off",
    use_mixed_precision: bool = False,
    n_instances: int = 1,
    **settings_extra: Any,
) -> dict[str, Any]:
    """Everything an interferometer standard-size cell takes from :func:`interferometer_standard_preset`.

    One definition for every setup (JAX sparse / dense, numba runtime and breakdown, the parity
    reference task), so they all evaluate the same likelihood. ``dataset`` is the masked
    ``al.Interferometer`` (its ``real_space_mask`` carries the Hilbert mesh and the edge points);
    ``adapt_image`` the truth lensed source (``_adapt_image_util.adapt_image_for_dataset``).

    Returns a dict: ``preset``, ``image_plane_mesh_grid`` (Delaunay, else None),
    ``n_source_pixels``, ``source_pixels_requested``, ``model``, ``instances`` (the preset's
    seeded iid stream, ``n_instances`` long; draw 0 is the profiled / parity instance),
    ``instance`` (draw 0), ``unit_vector`` (draw 0's unit vector), ``settings`` and
    ``memo_provenance`` (:func:`analysis_settings`, with ``settings_extra`` passed through) and
    ``record`` (what a row stores as ``configuration.production_preset``).
    """
    preset = interferometer_standard_preset(variant)
    mask = dataset.real_space_mask
    if preset.mesh_kind == "delaunay":
        grid = production_image_mesh_grid(dataset, adapt_image, preset, mask_radius, mask=mask)
        n_source_pixels = int(grid.shape[0])
        mesh = production_mesh(preset, n_vertices=n_source_pixels)
        requested = preset.hilbert_pixels
    else:
        grid = None
        n_source_pixels = preset.rect_pixels_yx**2
        mesh = production_mesh(preset)
        requested = n_source_pixels
    model = interferometer_production_model(preset, mesh)
    instances = iid_instances(model, max(1, int(n_instances)), preset)
    unit_vector = iid_unit_vectors(1, model.prior_count, preset)[0]
    settings, memo_provenance = analysis_settings(
        preset, memo, use_mixed_precision=use_mixed_precision, **settings_extra
    )
    record = {
        **preset.as_json(),
        "instance": "iid draw 0 of the preset sequence (seeded)",
        "instance_unit_vector": unit_vector,
        "instrument": instrument,
    }
    return {
        "preset": preset,
        "image_plane_mesh_grid": grid,
        "n_source_pixels": n_source_pixels,
        "source_pixels_requested": requested,
        "model": model,
        "instances": instances,
        "instance": instances[0],
        "unit_vector": unit_vector,
        "settings": settings,
        "memo_provenance": memo_provenance,
        "reg_scheme": preset.regularization,
        "reg_provenance": preset.as_json()["regularization"],
        "record": record,
    }


#: The visibility chunk and column block of the chunked dense reference (R6): memory
#: O(S^2 + chunk x S) — at jvla S = 1280 one 5e5-visibility chunk of T is 10.2 GB complex128 (peak
#: ~25 GB with the real / imaginary weighted copies; the reference tasks ask 64 GB).
INTERFEROMETER_REFERENCE_VIS_CHUNK = 500_000
INTERFEROMETER_REFERENCE_COLUMN_CHUNK = 64


def interferometer_dense_reference(
    dataset,
    adapt_images,
    instance,
    settings,
    *,
    vis_chunk: int = INTERFEROMETER_REFERENCE_VIS_CHUNK,
    column_chunk: int = INTERFEROMETER_REFERENCE_COLUMN_CHUNK,
    log=print,
) -> dict[str, Any]:
    """The eager NumPy dense log evidence of ``instance``, its ``T`` formed visibility-chunked (R6).

    The parity reference of every interferometer pixelized cell: independent of the W~ operator,
    of numba and of JAX. It runs the library's own dense route
    (``InversionInterferometerMapping``, built by ``FitInterferometer(xp=np)`` on
    :func:`numpy_dense_twin` of ``dataset``) for everything except the transformed mapping matrix
    ``T``, which the library forms in one ``nufft2d2`` over every column and every visibility (at
    jvla ``T`` alone is 512 GB). Here ``T`` is never held whole: for each chunk of ``vis_chunk``
    visibilities, a ``TransformerNUFFT`` over that chunk's uv points transforms the mapping matrix
    ``L`` in blocks of ``column_chunk`` columns (finufft on the NumPy path), and

        F += Re(T_k)^T N_r^-1 Re(T_k) + Im(T_k)^T N_i^-1 Im(T_k)
        D += Re(T_k)^T N_r^-1 Re(d_k) + Im(T_k)^T N_i^-1 Im(d_k)

    through the library's own ``curvature_matrix_via_mapping_matrix_from`` and
    ``data_vector_via_transformed_mapping_matrix_from``. ``F`` and ``D`` are then handed to the
    library inversion (its cached ``curvature_matrix`` / ``data_vector``), so the positive-only
    solve, the edge-zeroed subset, both log determinants and the regularization term are the
    library's. The chi-squared is ``s^T F s - 2 s^T D + d^T N^-1 d`` (identical to the residual
    form) and the evidence is assembled by ``fit_util.log_evidence_from``.

    Returns a dict with ``log_evidence`` and its terms, the chunking and the timings.
    """
    import time

    import autolens as al
    import numpy as np
    from autoarray.fit import fit_util
    from autoarray.inversion.inversion import inversion_util
    from autoarray.inversion.inversion.interferometer import inversion_interferometer_util

    t_start = time.perf_counter()
    twin = numpy_dense_twin(dataset)
    tracer = al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields])
    fit = al.FitInterferometer(
        dataset=twin, tracer=tracer, adapt_images=adapt_images, settings=settings, xp=np
    )
    inv = fit.inversion
    inversion_class = type(inv).__name__
    if inversion_class != "InversionInterferometerMapping":
        raise RuntimeError(
            f"the dense reference needs InversionInterferometerMapping, got {inversion_class}"
        )
    mapping_matrix = np.asarray(inv.mapping_matrix, dtype=np.float64)
    n_src = int(mapping_matrix.shape[1])
    visibilities = np.asarray(inv.data)
    noise_map = np.asarray(inv.noise_map)
    uv = np.asarray(dataset.uv_wavelengths)
    mask = dataset.real_space_mask
    eps = float(getattr(dataset.transformer, "eps", 1e-12))
    n_vis = int(uv.shape[0])
    vis_chunk = max(1, int(vis_chunk))
    column_chunk = max(1, int(column_chunk))
    n_chunks = -(-n_vis // vis_chunk)

    curvature = np.zeros((n_src, n_src), dtype=np.float64)
    data_vector = np.zeros(n_src, dtype=np.float64)
    transform_s = accumulate_s = 0.0
    for k, start in enumerate(range(0, n_vis, vis_chunk)):
        stop = min(n_vis, start + vis_chunk)
        t0 = time.perf_counter()
        transformer = al.TransformerNUFFT(
            uv_wavelengths=uv[start:stop], real_space_mask=mask, eps=eps
        )
        tmm = np.empty((stop - start, n_src), dtype=np.complex128)
        for c in range(0, n_src, column_chunk):
            tmm[:, c : c + column_chunk] = transformer.transform_mapping_matrix(
                mapping_matrix[:, c : c + column_chunk], xp=np
            )
        t1 = time.perf_counter()
        noise_k = noise_map[start:stop]
        curvature += inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=tmm.real, noise_map=noise_k.real, settings=settings, xp=np
        )
        curvature += inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=tmm.imag, noise_map=noise_k.imag, settings=settings, xp=np
        )
        data_vector += (
            inversion_interferometer_util.data_vector_via_transformed_mapping_matrix_from(
                transformed_mapping_matrix=tmm,
                visibilities=visibilities[start:stop],
                noise_map=noise_k,
            )
        )
        del tmm, transformer
        t2 = time.perf_counter()
        transform_s += t1 - t0
        accumulate_s += t2 - t1
        if n_chunks > 1:
            log(
                f"  [dense reference] chunk {k + 1}/{n_chunks}: T {t1 - t0:.1f} s, "
                f"F/D {t2 - t1:.1f} s"
            )
    if len(inv.no_regularization_index_list) > 0:
        curvature = inversion_util.curvature_matrix_with_added_to_diag_from(
            curvature_matrix=curvature,
            value=inv.settings.no_regularization_add_to_curvature_diag_value,
            no_regularization_index_list=inv.no_regularization_index_list,
            xp=np,
        )
    # The library inversion's cached properties: everything downstream (the positive-only solve,
    # the edge-zeroed subset, the log determinants) reads these, and T is never formed.
    inv.__dict__["curvature_matrix"] = curvature
    inv.__dict__["data_vector"] = data_vector
    t3 = time.perf_counter()
    reconstruction = np.asarray(inv.reconstruction)
    data_term = float(
        np.sum(visibilities.real**2 / noise_map.real**2)
        + np.sum(visibilities.imag**2 / noise_map.imag**2)
    )
    chi_squared = float(
        reconstruction @ curvature @ reconstruction - 2.0 * reconstruction @ data_vector + data_term
    )
    regularization_term = float(inv.regularization_term)
    log_det_curvature_reg = float(inv.log_det_curvature_reg_matrix_term)
    log_det_regularization = float(inv.log_det_regularization_matrix_term)
    noise_normalization = float(fit.noise_normalization)
    log_evidence = float(
        fit_util.log_evidence_from(
            chi_squared=chi_squared,
            regularization_term=regularization_term,
            log_curvature_regularization_term=log_det_curvature_reg,
            log_regularization_term=log_det_regularization,
            noise_normalization=noise_normalization,
        )
    )
    t4 = time.perf_counter()
    return {
        "log_evidence": log_evidence,
        "chi_squared": chi_squared,
        "regularization_term": regularization_term,
        "log_det_curvature_reg": log_det_curvature_reg,
        "log_det_regularization": log_det_regularization,
        "noise_normalization": noise_normalization,
        "inversion_class": inversion_class,
        "source_pixels": n_src,
        "visibilities": n_vis,
        "vis_chunk": vis_chunk,
        "column_chunk": column_chunk,
        "n_vis_chunks": n_chunks,
        "nufft_eps": eps,
        "transform_s": transform_s,
        "accumulate_s": accumulate_s,
        "solve_and_evidence_s": t4 - t3,
        "total_s": t4 - t_start,
        "method": (
            "eager NumPy dense route (InversionInterferometerMapping via FitInterferometer(xp=np) "
            "on the dataset without its sparse operator); T formed per visibility chunk and "
            "column block by TransformerNUFFT (finufft on the NumPy path) and accumulated into "
            "F and D, which the library inversion then solves (positive-only, edge-zeroed "
            "subset, log determinants); chi-squared s^T F s - 2 s^T D + d^T N^-1 d. "
            "Independent of the W~ operator, numba and JAX (ruling R6)."
        ),
    }


#: Where the reference task caches each (variant, instrument) reference, under the checkout.
INTERFEROMETER_REFERENCE_DIR = "output/p2b_references"
#: The dataset files a cached reference is keyed on (their md5).
_REFERENCE_DATASET_FILES = (
    "data.fits",
    "noise_map.fits",
    "uv_wavelengths.fits",
    "lensed_source.fits",
)


def interferometer_reference_path(root, variant: str, instrument: str) -> Path:
    """``<root>/output/p2b_references/interferometer_<variant>_<instrument>.json``."""
    return Path(root) / INTERFEROMETER_REFERENCE_DIR / f"interferometer_{variant}_{instrument}.json"


def dataset_fingerprint(dataset_path) -> dict[str, str | None]:
    """md5 of each dataset file a reference depends on (None when absent)."""
    import hashlib

    out: dict[str, str | None] = {}
    for name in _REFERENCE_DATASET_FILES:
        path = Path(dataset_path) / name
        if not path.is_file():
            out[name] = None
            continue
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        out[name] = digest.hexdigest()
    return out


def load_interferometer_reference(
    root, variant: str, instrument: str, *, dataset_path, library_revisions, unit_vector
) -> tuple[dict | None, str]:
    """``(cached reference, why)``: the reference task's result if it describes this cell.

    It must have been computed on the same dataset files (md5), the same library revisions and
    the same instance (iid draw 0's unit vector); otherwise ``(None, reason)``.
    """
    import json

    path = interferometer_reference_path(root, variant, instrument)
    if not path.is_file():
        return None, f"no cached reference at {path}"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"unreadable reference {path}: {exc}"
    if doc.get("schema") != "p2b-interferometer-reference" or doc.get("version") != 1:
        return None, f"{path}: not a p2b-interferometer-reference v1 document"
    if doc.get("dataset_fingerprint") != dataset_fingerprint(dataset_path):
        return None, f"{path}: computed on different dataset files"
    if isinstance(library_revisions, dict) and doc.get("library_revisions") != library_revisions:
        return None, f"{path}: computed on different library revisions"
    if unit_vector is not None and doc.get("instance_unit_vector") != unit_vector:
        return None, f"{path}: computed at a different instance"
    return doc, "ok"


def write_interferometer_parity(
    cell_dir,
    *,
    log_likelihood,
    variant: str,
    instrument: str,
    dataset_path,
    root,
    unit_vector,
    use_mixed_precision: bool,
    autolens_version: str,
    source_pixels: int,
    source_pixels_requested: int | None,
    quantity: str,
    compute_reference=None,
):
    """Write ``parity.json`` for an interferometer pixelized cell (R6, R7) and return the reference.

    The reference is the reference task's cache (:func:`load_interferometer_reference`) when it
    describes this cell; otherwise ``compute_reference()`` (a callable returning
    :func:`interferometer_dense_reference`'s dict) when the cell can afford it in-process (sma);
    otherwise the witness is ``not_assessed`` with the reason, and the reference task's
    ``--finalise`` pass rewrites it from the row's ``parity_candidate`` once the reference
    exists. The tolerance is the interferometer rule (``_profile_cli.PARITY_RTOL_BY_DATASET``).
    """
    from _profile_cli import provenance_dict, write_parity_json

    revisions = provenance_dict().get("library_revisions")
    doc, why = load_interferometer_reference(
        root,
        variant,
        instrument,
        dataset_path=dataset_path,
        library_revisions=revisions,
        unit_vector=unit_vector,
    )
    source = None
    reference = float("nan")
    if doc is not None:
        reference = float(doc["log_evidence"])
        source = (
            f"cached reference task {interferometer_reference_path(root, variant, instrument).name}"
            f" ({doc.get('method_short', 'chunked eager NumPy dense')})"
        )
    elif compute_reference is not None:
        try:
            computed = compute_reference()
            reference = float(computed["log_evidence"])
            source = (
                "visibility-chunked eager NumPy dense reference computed in the same process "
                f"({computed['n_vis_chunks']} chunk(s) of {computed['vis_chunk']} visibilities)"
            )
            doc = computed
        except Exception as exc:  # noqa: BLE001 — a failed reference must not cost the row
            why = f"in-process reference failed: {type(exc).__name__}: {exc}"
            print(f"  WARNING: parity reference failed: {why}")
    write_parity_json(
        Path(cell_dir),
        log_likelihood=log_likelihood,
        reference_log_likelihood=reference,
        use_mixed_precision=use_mixed_precision,
        reference_source=source
        or "visibility-chunked eager NumPy dense reference (reference task pending)",
        autolens_version=autolens_version,
        source_pixels=int(source_pixels),
        source_pixels_requested=source_pixels_requested,
        quantity=quantity,
        dataset_type="interferometer",
        pending_reason=None if source else f"reference pending: {why}",
    )
    return doc
