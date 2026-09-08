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
    #: Light-profile radial-bin recipe. Euclid ``util.py:831-836``; subhalo
    #: ``scripts/imaging.py:179-184``. Both are `[4, 2, 2]` at `[0.1, 0.3]`
    #: after autolens_profiling#235 retired the outer sub-size-1 bin.
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
        "euclid_dr1_prelim/hpc/batch_cpu/submit_initial_lens_model_two_stage:48,161-166"
    ),
    mesh_kind="delaunay",
    hilbert_pixels=500,
    edge_pixels=30,
    hilbert_weight_power=3.5,
    hilbert_weight_floor=0.01,
    adapt_snr_cap=None,
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
        "rectangular row runs this configuration on the Euclid dataset"
    ),
    mesh_kind="rectangular",
    rect_pixels_yx=32,
    adapt_snr_cap=None,
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


def apply_production_over_sampling(dataset, adapt_image, preset: ProductionPreset):
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

    return dataset.apply_sparse_operator_cpu()


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


def analysis_settings(preset: ProductionPreset, memo: str) -> tuple[Any, dict[str, Any]]:
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

    settings = al.Settings(**kwargs)

    provenance = {
        "requested": memo,
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


def witness_verdict(cold_eval_median_s: float, instrument: str) -> dict[str, Any]:
    """Judge one cold-eval median against the production reference range."""
    reference = PRODUCTION_REFERENCE.get(instrument)
    if reference is None:
        return {"verdict": "no_reference", "instrument": instrument}

    low, high = reference["cold_eval_s"]
    return {
        "instrument": instrument,
        "job": reference["job"],
        "reference_cold_eval_s": [low, high],
        "factor": WITNESS_FACTOR,
        "allowed_s": [low / WITNESS_FACTOR, high * WITNESS_FACTOR],
        "measured_cold_eval_median_s": cold_eval_median_s,
        "verdict": (
            "PASS"
            if (low / WITNESS_FACTOR) <= cold_eval_median_s <= (high * WITNESS_FACTOR)
            else "FAIL"
        ),
    }
