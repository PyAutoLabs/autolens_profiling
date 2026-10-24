"""Shared CLI / JSON / auto-simulate helpers for the likelihood scripts.

Used by every per-cell script under ``likelihood_runtime/`` and
``likelihood_breakdown/`` so the per-script boilerplate stays minimal
and the sweep-driver flags (``--config-name``, ``--output-dir``,
``--use-mixed-precision``) and dataset auto-simulate hook are defined
in one place.

Designed to be imported with relative path manipulation since the scripts
live under multiple sibling directories::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _profile_cli import (
        parse_profile_cli, device_info_dict, resolve_output_paths,
        auto_simulate_if_missing,
    )
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProfileCLI:
    config_name: str | None
    output_dir: Path | None
    use_mixed_precision: bool
    instrument: str | None
    vmap_probe: bool
    use_sparse_operator: bool
    rect_mesh: str
    regularization: str | None
    variant: str
    memo: str
    n_instances: int
    cold_evals: int
    sparse_batch_size: int
    source_pixels: int | None
    over_sampling: str = "flat1"
    #: ``--output-path``: the exact result JSON a driver (``likelihood_runtime/sweep.py``) has
    #: already placed; it wins over every other naming rule.
    output_path: Path | None = None
    #: ``--no-fit-png`` turns this off: the cell's ``fit.png`` (:func:`render_fit_png`).
    fit_png: bool = True
    #: ``--standard-size``: build the human's standard source-pixel count for the cell's variant
    #: (:data:`STANDARD_SOURCE_PIXELS`, :func:`requested_source_pixels`). A canonical cell, so the
    #: row keeps its canonical name (no ``_n<N>`` suffix) unlike ``--source-pixels``.
    standard_size: bool = False

    def parse_cell_args(self, cell_parser: argparse.ArgumentParser, *, argv=None):
        """Finish staged parsing, rejecting flags neither parser understands.

        Reparse the complete argv through the union so shared/local duplicates
        such as --source-pixels retain their cell values. Legacy shared-only
        callers remain permissive until they opt into this final boundary.
        """
        parser = argparse.ArgumentParser(
            parents=[_profile_parser(), cell_parser],
            add_help=False,
            allow_abbrev=False,
            conflict_handler="resolve",
        )
        args = parser.parse_args(argv)
        for name in (
            "source_pixels",
            "threads",
            "n_repeats",
            "n_stream",
            "trace_calls",
            "safe_budget",
            "pass_budget_max",
            "n_draws",
            "n_random",
            "walk_max_evals",
            "n_instances",
            "cold_evals",
            "sparse_batch_size",
        ):
            value = getattr(args, name, None)
            if value is not None and value < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "budgets"):
            try:
                budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
            except ValueError:
                parser.error("--budgets must be a comma-separated list of positive integers")
            if not budgets or any(value < 1 for value in budgets):
                parser.error("--budgets must contain positive integers")
        return args


def _profile_parser() -> argparse.ArgumentParser:
    """Build the shared parser; cells compose it at their final boundary."""
    parser = argparse.ArgumentParser(
        description="Multi-config likelihood profiling driver flags.",
        # Keep unknown args; per-script argparse is not exhaustive.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help=(
            "Output-filename label for the multi-config sweep "
            "(e.g. local_cpu_fp64, local_gpu_mp, hpc_a100_fp64). "
            "When omitted, the script keeps its single-config filename pattern."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override results dir. Each per-cell script defaults to its "
            "package's section under <autolens_profiling>/results/."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Exact result JSON path (the PNG goes beside it). Set by drivers that place the "
            "row themselves (likelihood_runtime/sweep.py); overrides --output-dir and naming."
        ),
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help=(
            "Pass use_mixed_precision=True to al.Settings — "
            "targeted fp32 paths in the JAX inversion."
        ),
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help=(
            "Instrument preset to profile. When omitted, the per-cell "
            "script's module-level default applies (typically 'sma' for "
            "interferometer/datacube cells, 'hst' for imaging)."
        ),
    )
    parser.add_argument(
        "--vmap-probe",
        action="store_true",
        help=(
            "Probe mode: JIT-vmap the full pipeline at batch=2 and batch=4, "
            "read compiled.memory_analysis(), write a vmap_probe.json with "
            "the recommended A100 batch_size, and exit before the steady-"
            "state timing loop. See vram/README.md for methodology."
        ),
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Call ``dataset.apply_sparse_operator(batch_size=...)`` "
            "(the width is ``--sparse-batch-size``, default 128) after "
            "dataset construction so the inversion factory selects the "
            "w-tilde sparse path (``InversionImagingSparse``) instead of "
            "the dense ``InversionImagingMapping``. The sparse path "
            "supports mixed linear-obj lists — the production "
            "pixelization / Delaunay cells include an MGE lens-light "
            "basis alongside the Mapper source, and the sparse "
            "InversionImagingSparse handles the MGE Basis columns via "
            "``linear_func_operated_mapping_matrix_dict`` while the "
            "Mapper columns go through the w-tilde sparse-operator "
            "assembly. The only short-circuit-to-dense case is when "
            "*every* linear object is an ``AbstractLinearObjFuncList`` "
            "(e.g. the pure-MGE-source reference cell). Per-cell scripts "
            "that read this flag embed the chosen path into the result "
            "JSON as ``inversion_path``."
        ),
    )

    parser.add_argument(
        "--sparse-batch-size",
        type=int,
        default=128,
        help=(
            "Column-block width handed to ``dataset.apply_sparse_operator("
            "batch_size=...)`` — how many source-pixel columns of the w-tilde "
            "curvature sweep share one rFFT2/irFFT2 batch "
            "(``ImagingSparseOperator.curvature_matrix_diag_from`` runs "
            "``ceil(S / batch_size)`` blocks). Larger blocks raise GPU "
            "occupancy and VRAM together; 128 is the library default and what "
            "every recorded sparse row was measured with. Ignored without "
            "``--sparse``; recorded in the result JSON as "
            "``configuration.sparse_batch_size``."
        ),
    )

    parser.add_argument(
        "--rect-mesh",
        choices=("bilinear", "rtu"),
        default="bilinear",
        help=(
            "Which adaptive rectangular mesh family the cell profiles "
            "(PyAutoArray#462 split): 'bilinear' — "
            "RectangularBilinearAdaptDensity/AdaptImage, the empirical "
            "rank-CDF workspace default and CPU-speed campaign target — or "
            "'rtu' — RectangularRTUAdaptDensity/AdaptImage, the kernel-CDF "
            "advanced/GPU/gradient meshes (the pre-split behaviour, so "
            "'--rect-mesh rtu' reproduces pre-split recorded results). "
            "Cells that read this flag resolve the classes via "
            "``rect_mesh_classes`` and embed ``rect_mesh`` in the result "
            "JSON; '_rtu' is appended to the output basename so the two "
            "families' results never clobber each other."
        ),
    )

    parser.add_argument(
        "--regularization",
        choices=("adapt_split", "constant_split"),
        default=None,
        help=(
            "Regularization scheme for the Delaunay-family cells. "
            "'adapt_split' — ``al.reg.AdaptSplit(inner_coefficient=0.1, "
            "outer_coefficient=10.0, signal_scale=0.1)``, what production "
            "(SLaM, the Euclid pipeline) pairs Delaunay with and the cells' "
            "default — or 'constant_split' — ``al.reg.ConstantSplit("
            "coefficient=1.0)``, the scheme every Delaunay row recorded before "
            "2026-09-08 was measured with, kept reachable so those rows stay "
            "comparable. Omitted (None) leaves each cell on its own default; "
            "cells outside the Delaunay family ignore the flag. The resolved "
            "scheme selects the cell's pinned log-evidence and is embedded in "
            "the result JSON as ``regularization``."
        ),
    )

    parser.add_argument(
        "--variant",
        choices=("production", "legacy"),
        default="production",
        help=(
            "Which configuration the production-preset cells build. "
            "'production' (the default) resolves the instrument's production "
            "preset from ``_production_config`` — the Euclid ``vis_pix`` stage "
            "for ``--instrument euclid``, the subhalo ``source_pix[2]`` stage "
            "for ``--instrument hst`` — matching mesh, over-sampling, "
            "regularization, MGE basis, positions penalty and thread pinning "
            "field for field. 'legacy' rebuilds the cell's own pre-2026-09-08 "
            "configuration, so the historic rows stay reproducible. Cells that "
            "carry no preset ignore the flag."
        ),
    )

    parser.add_argument(
        "--memo",
        choices=("off", "on"),
        default="off",
        help=(
            "The NNLS cross-evaluation warm-start memo "
            "(``aa.Settings(nnls_warm_start_memo=...)``, PyAutoArray#498). Off "
            "by default in the preset cells: its measured gains come from a "
            "random-walk stream a Nautilus pool never hands one worker, and "
            "with the memo on a cell that repeats one instance seeds itself "
            "from a 100 %%-correct previous solve. The library default is "
            "``true`` and production leaves it unset, so 'on' is what "
            "production pays; the resolved flag is recorded in every result "
            "JSON. ``--variant legacy`` leaves both gates untouched."
        ),
    )

    parser.add_argument(
        "--n-instances",
        type=int,
        default=20,
        help=(
            "Length of the seeded iid instance sequence the production-preset "
            "cells profile (default 20): one warm-up, ``--cold-evals`` cold "
            "evaluations, the rest warm."
        ),
    )

    parser.add_argument(
        "--cold-evals",
        type=int,
        default=3,
        help=(
            "How many of the iid instances are timed as cold evaluations "
            "(default 3) — the quantity comparable to PyAutoFit's logged 'Log "
            "Likelihood Function Evaluation Time'."
        ),
    )

    parser.add_argument(
        "--source-pixels",
        type=int,
        default=None,
        help=(
            "Override the cell's fiducial source-pixel count, for the N_src "
            "sweep (autolens_profiling#247). Rectangular cells take the nearest "
            "square (``round(sqrt(N))`` per side, so 3000 -> 55x55 = 3025); the "
            "Delaunay family takes N vertices directly. The count actually "
            "built is what lands in the result JSON (``configuration."
            "source_pixels`` / ``delaunay_vertices``), alongside "
            "``configuration.source_pixels_requested``. Omitted (None) leaves "
            "every cell on its fiducial (39x39 / 1500 / 1500) and changes "
            "nothing. A non-fiducial count SKIPS the cell's pinned-evidence "
            "check — the pin describes the fiducial mesh and nothing else — and "
            "the JSON then carries ``pinned_expected: null``."
        ),
    )

    parser.add_argument(
        "--over-sampling",
        choices=("flat1", "production"),
        default="flat1",
        help=(
            "Over-sampling recipe for the JAX Delaunay-family imaging cells "
            "(``scripts/imaging/delaunay{,_nn}/likelihood_{runtime,breakdown}.py``). "
            "'flat1' (the default) keeps the historic cell setting: pixelization "
            "sub-size 1 everywhere and light-profile bins [4, 2, 2] at [0.3, 0.6]. "
            "'production' resolves the production over-sampling of "
            "``--instrument`` from ``_production_config`` (euclid: the Euclid "
            "``vis_pix`` stage; hst: the subhalo ``source_pix[2]`` stage): "
            "pixelization sub-size 4 where the source S/N exceeds 3, 2 elsewhere, "
            "and that stage's light-profile radial bins. Only the over-sampling "
            "changes; the cell's mesh, MGE and regularization stay as they are. "
            "Recorded in the result JSON as "
            "``configuration.over_sample_size_pixelization_rule``; production rows "
            "default to an ``os_production/`` subfolder of the cell's results "
            "directory, so the dashboard keys them as their own series. Other "
            "cells ignore the flag."
        ),
    )

    parser.add_argument(
        "--no-fit-png",
        dest="fit_png",
        action="store_false",
        help=(
            "Do not render the cell's fit.png. By default each cell renders one fit figure "
            "beside its result JSON, after every timed region, with the plotter the "
            "autolens_visualization harness uses for the dataset type. Never rendered "
            "under AUTOLENS_PROFILING_SMOKE."
        ),
    )

    parser.add_argument(
        "--standard-size",
        action="store_true",
        help=(
            "Evaluate the standard-size production cell of the likelihood (human rulings "
            "2026-10-23/24): every pixelized imaging cell, JAX or numba, builds the numba "
            "production preset of its instrument (_production_config; jwst/ao borrow the hst "
            "preset) at Delaunay / DelaunayNN 1250 Hilbert vertices + 30 zeroed edge points "
            "(1280 built) or rectangular 35x35 = 1225, so every setup of one likelihood x "
            "instrument evaluates the same likelihood, with production's S/N adapt image, "
            "uncapped over-sampling cut and RectangularBilinearAdaptImage mesh. The row records "
            "``configuration.source_pixels_requested`` (1250 / 1225) beside the count actually "
            "built, and the catalogue's size rule matches on the requested count. Unlike "
            "``--source-pixels`` the row is a canonical cell: no ``_n<N>`` suffix, no scan "
            "placement. Cells without a source mesh (MGE) ignore it. Exclusive with "
            "``--source-pixels``."
        ),
    )

    return parser


#: Flags whose value ``--standard-size`` replaces (mesh family, size, over-sampling,
#: regularization, preset variant): passing one beside it is an error, never a silent override.
STANDARD_SIZE_EXCLUSIVE = (
    "--source-pixels",
    "--rect-mesh",
    "--over-sampling",
    "--regularization",
    "--variant",
)


def parse_profile_cli(default_config_name: str | None = None, *, argv=None) -> ProfileCLI:
    """Parse the sweep CLI flags accepted by every per-cell profile script.

    Returns ``ProfileCLI(config_name, output_dir, use_mixed_precision,
    instrument)``.

    When ``--config-name`` is omitted, falls back to ``default_config_name``
    (typically inferred from ``JAX_PLATFORM_NAME`` env var or left as ``None``
    to preserve the existing single-config filename pattern).

    ``--instrument`` is optional; when omitted (None) per-cell scripts keep
    their module-level hardcoded default (typically ``"sma"`` or ``"hst"``).
    """
    parser = _profile_parser()
    args, _unknown = parser.parse_known_args(argv)
    if args.standard_size:
        # --standard-size builds the production cell's mesh, size, over-sampling, regularization
        # and preset; any explicit flag it would silently replace is refused (#456 fork b).
        tokens = list(sys.argv[1:] if argv is None else argv)
        given = [
            f
            for f in STANDARD_SIZE_EXCLUSIVE
            if any(t == f or t.startswith(f + "=") for t in tokens)
        ]
        if given:
            parser.error(
                f"--standard-size builds the production cell and would silently replace "
                f"{', '.join(given)}; drop {'it' if len(given) == 1 else 'them'}"
            )
    config_name = args.config_name or default_config_name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    return ProfileCLI(
        config_name=config_name,
        output_dir=output_dir,
        use_mixed_precision=bool(args.use_mixed_precision),
        instrument=args.instrument,
        vmap_probe=bool(args.vmap_probe),
        use_sparse_operator=bool(args.sparse),
        rect_mesh=args.rect_mesh,
        regularization=args.regularization,
        variant=args.variant,
        memo=args.memo,
        n_instances=int(args.n_instances),
        cold_evals=int(args.cold_evals),
        sparse_batch_size=int(args.sparse_batch_size),
        source_pixels=int(args.source_pixels) if args.source_pixels is not None else None,
        over_sampling=args.over_sampling,
        output_path=Path(args.output_path).resolve() if args.output_path else None,
        fit_png=bool(args.fit_png),
        standard_size=bool(args.standard_size),
    )


#: The human's standard source-pixel counts (2026-10-23), by likelihood variant. The one copy is
#: ``scripts/misc/tooling/catalogue_lifecycle.STANDARD_SOURCE_PIXELS`` (the catalogue's size rule);
#: this mirrors it so the cells stay importable without the tooling path, and
#: ``test_standard_size.py`` pins the two equal.
STANDARD_SOURCE_PIXELS = {"delaunay": 1250, "delaunay_nn": 1250, "rectangular": 35 * 35}


def requested_source_pixels(cli: ProfileCLI, variant: str) -> int | None:
    """The source-pixel count a cell of likelihood ``variant`` is asked to build, or None.

    ``--standard-size`` -> :data:`STANDARD_SOURCE_PIXELS` of ``variant``; else ``--source-pixels``;
    else None (the cell keeps its own fiducial). Rectangular cells build the nearest square
    (``round(sqrt(N))`` per side; 1225 -> 35x35 exactly); Delaunay-family cells build N Hilbert
    vertices (the standard-size production cells add the preset's 30 zeroed edge points).
    """
    if getattr(cli, "standard_size", False):
        try:
            return STANDARD_SOURCE_PIXELS[variant]
        except KeyError:
            raise ValueError(
                f"--standard-size has no standard count for variant {variant!r}"
            ) from None
    return cli.source_pixels


def rectangular_side(requested: int | None, fiducial: int) -> int:
    """The rectangular mesh side for a requested source-pixel count (``fiducial`` when None)."""
    import math

    return fiducial if requested is None else int(round(math.sqrt(requested)))


#: What ``--regularization`` resolves to in the Delaunay-family cells when the
#: flag is omitted. Production pairs Delaunay with ``AdaptSplit``, so that is
#: what the profiled rows measure; ``constant_split`` reproduces the pre-
#: 2026-09-08 rows.
DELAUNAY_REGULARIZATION_DEFAULT = "adapt_split"


def delaunay_regularization(cli: ProfileCLI):
    """Resolve ``--regularization`` into the Delaunay-family regularization object.

    Returns ``(scheme, regularization, provenance)``: the resolved scheme name,
    the ``al.reg`` object to hand ``al.Pixelization``, and the dict every
    Delaunay-family result JSON records under ``regularization``.

    ``AdaptSplit`` is given the in-repo production-shaped coefficients
    (``inner=0.1``, ``outer=10.0``, ``signal_scale=0.1``, as used by
    ``likelihood_breakdown/delaunay_numba_nnls_iterations.py``) rather than its
    ``inner == outer == 1.0`` defaults, which make the per-pixel weights uniform
    and the scheme numerically indistinguishable from ``ConstantSplit``.

    ``AdaptSplit`` reads the mapper's ``adapt_data``, so a cell using it must
    also pass ``galaxy_image_dict`` / ``galaxy_name_image_dict`` to its
    ``al.AdaptImages``; the cells do so unconditionally, since ``ConstantSplit``
    ignores them and the two legs then differ only in the scheme.

    Imports autolens lazily so ``_profile_cli`` stays importable without the
    modelling stack.
    """
    import autolens as al

    scheme = cli.regularization or DELAUNAY_REGULARIZATION_DEFAULT

    if scheme == "constant_split":
        return (
            scheme,
            al.reg.ConstantSplit(coefficient=1.0),
            {"scheme": scheme, "coefficient": 1.0},
        )

    return (
        scheme,
        al.reg.AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.1),
        {
            "scheme": scheme,
            "inner_coefficient": 0.1,
            "outer_coefficient": 10.0,
            "signal_scale": 0.1,
        },
    )


def rect_mesh_classes(cli: ProfileCLI):
    """Resolve the explicit adaptive rectangular mesh classes for this run.

    Returns ``(density_cls, image_cls)`` for ``cli.rect_mesh``:
    ``RectangularBilinearAdaptDensity/AdaptImage`` (rank-CDF, the workspace
    default) or ``RectangularRTUAdaptDensity/AdaptImage`` (kernel-CDF, the
    pre-split behaviour). Imports autolens lazily so ``_profile_cli`` stays
    importable without the modelling stack.
    """
    import autolens as al

    if cli.rect_mesh == "rtu":
        return al.mesh.RectangularRTUAdaptDensity, al.mesh.RectangularRTUAdaptImage
    return (
        al.mesh.RectangularBilinearAdaptDensity,
        al.mesh.RectangularBilinearAdaptImage,
    )


#: Name of the per-fusion autotune subdirectory XLA writes inside
#: ``JAX_COMPILATION_CACHE_DIR``. A seeded autotune cache silently changes which
#: GPU kernels a run uses — the F row moving between ~4.8 ms and ~25.6 ms on the
#: A100 was the only tell — so every result records how many entries were
#: already there when the process started.
AUTOTUNE_CACHE_SUBDIR = "xla_gpu_per_fusion_autotune_cache_dir"


def _autotune_cache_entries(cache_dir: str | None) -> int:
    """Count entries in ``<cache_dir>/xla_gpu_per_fusion_autotune_cache_dir``.

    Zero when the cache dir is unset, absent, or has no autotune subdirectory —
    i.e. zero means "nothing was seeded", which is what a comparable run wants.
    """
    if not cache_dir:
        return 0
    autotune_dir = Path(cache_dir) / AUTOTUNE_CACHE_SUBDIR
    try:
        return sum(1 for _ in autotune_dir.iterdir())
    except OSError:
        return 0


#: Compilation-cache state as found at **import** time, before any JAX
#: compilation this process performs can add to it. Captured here rather than at
#: JSON-write time for exactly that reason.
_CACHE_DIR_AT_IMPORT = os.environ.get("JAX_COMPILATION_CACHE_DIR") or None
_AUTOTUNE_ENTRIES_AT_IMPORT = _autotune_cache_entries(_CACHE_DIR_AT_IMPORT)


def _loadavg() -> list[float] | None:
    try:
        return [round(x, 3) for x in os.getloadavg()]
    except Exception:  # noqa: BLE001 — no load average (Windows, sandbox) is a fact, not an error
        return None


#: Host load average as found at **import** time — the "start" of the run, before
#: this process's own compile and timing work moves it. ``provenance_dict`` pairs it
#: with the load average at write time.
LOADAVG_AT_IMPORT = _loadavg()

#: The provenance block's keys, in order. ``scripts/misc/tooling/check_results_layout.py``
#: requires every one of them on each device-recording result JSON (its
#: ``REQUIRED_PROVENANCE_KEYS`` mirrors this tuple).
PROVENANCE_KEYS = (
    "provenance_schema",
    "captured_at",
    "host",
    "slurm",
    "loadavg_at_import",
    "loadavg_at_write",
    "profiling_revision",
    "library_revisions",
    "library_versions",
    "dependency_versions",
)

#: Library repo -> importable package, for ``library_versions``.
_LIBRARY_PACKAGES = (
    ("PyAutoNerves", "autonerves"),
    ("PyAutoFit", "autofit"),
    ("PyAutoArray", "autoarray"),
    ("PyAutoGalaxy", "autogalaxy"),
    ("PyAutoLens", "autolens"),
)

#: Third-party packages whose version rescales or re-routes a measurement (a JAX bump
#: changes XLA, a nufftax below its floor routed x64 GPU NUFFTs to fp32 Pallas —
#: interferometer MGE round 1, RAL jobs 351055-351057).
_DEPENDENCY_PACKAGES = ("jax", "jaxlib", "numpy", "scipy", "numba", "nufftax")


def _source_revisions() -> dict[str, str]:
    """``likelihood_breakdown.provenance.source_revisions`` for this checkout + libraries."""
    import importlib
    import sys

    misc = str(Path(__file__).resolve().parent / "scripts" / "misc")
    if misc not in sys.path:
        sys.path.append(misc)
    provenance = importlib.import_module("likelihood_breakdown.provenance")
    return provenance.source_revisions(Path(__file__).resolve().parent)


def _package_versions(packages) -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _library_versions() -> dict[str, str | None]:
    """``__version__`` of every importable PyAuto* library (source checkouts, not pip)."""
    import importlib

    out = {}
    for repo, package in _LIBRARY_PACKAGES:
        try:
            out[repo] = getattr(importlib.import_module(package), "__version__", None)
        except Exception:  # noqa: BLE001 — an unimportable library is a fact to record
            out[repo] = None
    return out


def provenance_dict() -> dict:
    """The provenance block every result JSON carries (``device.provenance``).

    The fields the wiki index and the run-time dashboard read off a result: which
    host ran it and how loaded it was at start and at write, which SLURM job, which
    revision of this repo and of every PyAuto* library, and which versions of the
    dependencies that can silently re-route a measurement. Attached by
    :func:`device_info_dict`, so every cell that records its device gets it without
    a per-script edit; ``check_results_layout.py --check`` refuses a new
    device-recording JSON without it.

    Never raises: a helper that fails degrades its field to ``None`` or an
    ``unavailable: ...`` string, because a provenance failure must not lose the
    measurement it describes (autolens_profiling#297 lost a job's JSONs that way).
    """
    import datetime as _dt
    import socket

    try:
        revisions = _source_revisions()
    except Exception as exc:  # noqa: BLE001 — see the docstring
        revisions = f"unavailable: {type(exc).__name__}"
    profiling_revision = (
        revisions.get("autolens_profiling") if isinstance(revisions, dict) else revisions
    )
    return {
        "provenance_schema": 1,
        "captured_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID") or None,
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or None,
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID") or None,
        },
        "loadavg_at_import": LOADAVG_AT_IMPORT,
        "loadavg_at_write": _loadavg(),
        "profiling_revision": profiling_revision,
        "library_revisions": revisions,
        "library_versions": _library_versions(),
        "dependency_versions": _package_versions(_DEPENDENCY_PACKAGES),
    }


def device_info_dict() -> dict:
    """Capture backend / device / nvidia-smi summary for the current JAX process.

    Imports jax lazily so callers can collect this near the JSON write without
    re-importing.
    """
    import socket

    import jax

    info = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        # Which node ran it. RAL's $HOME is node-local, so two legs of one grid
        # landing on different nodes do not share a compilation cache and are
        # not comparable at the compile-time level; the hostname is what says so.
        "hostname": socket.gethostname(),
        # Environment provenance: a stray XLA_FLAGS (e.g. disabling
        # constant_folding) or thread pinning silently rescales every timing
        # in a result by integer factors — record them so drift between runs
        # is attributable (found the hard way: autolens_profiling#59).
        "xla_flags": os.environ.get("XLA_FLAGS") or None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS") or None,
        "cpu_count": os.cpu_count(),
        # Compilation / autotune cache provenance. ``cache_fresh`` is the flag
        # a harvest gates on: a run that inherited a populated autotune cache
        # measured somebody else's kernel choices.
        "jax_compilation_cache_dir": _CACHE_DIR_AT_IMPORT,
        "autotune_cache_entries_at_start": _AUTOTUNE_ENTRIES_AT_IMPORT,
        "cache_fresh": bool(_CACHE_DIR_AT_IMPORT) and _AUTOTUNE_ENTRIES_AT_IMPORT == 0,
        # The mandatory provenance block (results/README.md "Artefact policy"):
        # host + load, SLURM job, library revisions, dependency versions.
        "provenance": provenance_dict(),
    }
    if info["backend"] == "gpu":
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.used,memory.total",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                )
                .decode()
                .strip()
            )
            info["nvidia_smi"] = out.replace("\n", "; ")
        except Exception:
            pass
    return info


#: The two DISJOINT thread-knob families a CPU timing depends on. ``NPROC``
#: sizes XLA's CPU intra-op thread pool — it is what throttles a JAX-CPU row
#: (autolens_profiling `reference_nproc_throttles_jax_cpu`) — while the BLAS
#: knobs pin OpenBLAS / MKL / OpenMP and govern the numpy/scipy rows without
#: touching JAX at all. A CPU number is uninterpretable without both, so both
#: are recorded on every CPU timing.
JAX_THREAD_ENV_VARS = ("NPROC",)
BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _proc_field(path: str, key: str) -> str | None:
    """First ``key`` line of a ``/proc`` file, value only. ``None`` if absent."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def machine_info_dict() -> dict:
    """The host a timing was taken on — CPU, RAM, kernel, threads, GPU.

    The companion of :func:`device_info_dict`, which records the *JAX* device.
    This records the *machine*, and it exists because a laptop row and a
    cluster row are only comparable when the laptop is described: which CPU,
    how many cores were actually usable, how much RAM, which kernel, and —
    when one is present — which GPU at which driver.

    **``nproc`` is not the core count.** GNU ``nproc`` honours
    ``OMP_NUM_THREADS``, so on a shell that exports ``OMP_NUM_THREADS=1`` it
    prints 1 on an 8-core machine. ``nproc --all`` is the physical answer and
    both are recorded, precisely so the discrepancy is visible rather than
    silently halving a reader's understanding of a row.
    """
    import platform
    import shutil

    def _nproc(*args: str) -> int | None:
        exe = shutil.which("nproc")
        if exe is None:
            return None
        try:
            return int(subprocess.check_output([exe, *args], timeout=5).decode().strip())
        except Exception:  # noqa: BLE001 — an absent nproc is not an error
            return None

    mem_total_kb = _proc_field("/proc/meminfo", "MemTotal")
    info = {
        "cpu_model": _proc_field("/proc/cpuinfo", "model name"),
        "cpu_count_os": os.cpu_count(),
        "nproc": _nproc(),
        "nproc_all": _nproc("--all"),
        "nproc_note": (
            "GNU nproc honours OMP_NUM_THREADS, so `nproc` can print 1 on a "
            "multi-core host; `nproc_all` is the core count."
        ),
        "ram_total": mem_total_kb,
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_thread_env": {k: os.environ.get(k) for k in JAX_THREAD_ENV_VARS},
        "blas_thread_env": {k: os.environ.get(k) for k in BLAS_THREAD_ENV_VARS},
        "thread_env_note": (
            "NPROC sizes XLA's CPU intra-op pool (the JAX rows); the BLAS knobs "
            "pin OpenBLAS/MKL/OpenMP (the numpy/scipy rows) and do not touch "
            "JAX. Two disjoint families; both are recorded on every CPU timing."
        ),
        "gpu": None,
    }

    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )
        if out:
            info["gpu"] = out.replace("\n", "; ")
    except Exception:  # noqa: BLE001 — no GPU on this host is a fact, not an error
        pass

    return info


#: Legacy result folders a writer may still name; ``resolve_output_paths`` places their rows
#: in the mirrored tree (autolens_profiling#442 PR 2).
_LEGACY_RESULT_DIRS = (("results", "runtime"), ("results", "breakdown"))
_REPO_ROOT = Path(__file__).resolve().parent


def _setup_hints(cli: ProfileCLI, default_basename: str, hints: dict | None) -> dict:
    """Setup facts known before the result JSON exists, in the JSON's own keys."""
    import socket
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "misc" / "tooling"))
    from build_migration_map import INSTRUMENTS

    instrument = cli.instrument
    if instrument is None:
        tokens = f"_{default_basename}_"
        instrument = next((i for i in INSTRUMENTS if f"_{i}_" in tokens), None)
    backend = "cpu"
    if "jax" in sys.modules:
        backend = sys.modules["jax"].default_backend()
    doc = {"use_mixed_precision": bool(cli.use_mixed_precision)}
    if cli.config_name is None:
        # A config name carries its host tier; an unnamed run is placed by the host it runs on.
        doc["device"] = {"backend": backend, "provenance": {"host": socket.gethostname()}}
    if instrument:
        doc["instrument"] = instrument
    doc.update(hints or {})
    return doc


def _in_legacy_results(path: Path) -> bool:
    try:
        rel = Path(path).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return rel.parts[:2] in _LEGACY_RESULT_DIRS


def _row_instrument(cli: ProfileCLI, default_basename: str) -> str | None:
    """``--instrument``, else the instrument token of ``default_basename``."""
    if getattr(cli, "instrument", None):
        return cli.instrument
    tokens = f"_{default_basename}_"
    return next((i for i in _migration_map_module().INSTRUMENTS if f"_{i}_" in tokens), None)


def runtime_instrument_dir(legacy_dir: Path, instrument: str | None) -> Path:
    """``results/runtime/<class>/<model>`` -> ``.../<model>/<instrument>`` (any other path as is).

    The sweep layout names every runtime row under its instrument; a two-level legacy runtime
    folder would make a config-named row (``delaunay_local_cpu_fp64.json``) of every instrument
    the same name, so a row whose cell cannot be derived would collide in the landing folder.
    """
    try:
        rel = Path(legacy_dir).resolve().relative_to(_REPO_ROOT).parts
    except ValueError:
        return Path(legacy_dir)
    if instrument and len(rel) == 4 and rel[:2] == ("results", "runtime"):
        return Path(legacy_dir) / instrument
    return Path(legacy_dir)


def placed_path(legacy_path: Path, doc: dict) -> Path:
    """:func:`mirrored_output_path` without creating or announcing anything (for readers)."""
    legacy_path = Path(legacy_path)
    if not _in_legacy_results(legacy_path):
        return legacy_path
    rel = legacy_path.resolve().relative_to(_REPO_ROOT)
    new, _kind = _migration_map_module().destination(_REPO_ROOT, rel.as_posix(), doc)
    return _REPO_ROOT / new


def place_output_path(
    json_path: Path, cli: ProfileCLI, default_basename: str, hints: dict | None = None
) -> Path:
    """``mirrored_output_path`` with the setup facts the CLI knows (see ``resolve_output_paths``).

    For writers that finish a legacy name themselves (suffixes appended after
    ``resolve_output_paths(..., place=False)``) and then place it.
    """
    if not _in_legacy_results(json_path):
        return Path(json_path)
    placed = mirrored_output_path(json_path, _setup_hints(cli, default_basename, hints))
    placed.parent.mkdir(parents=True, exist_ok=True)
    return placed


def mirrored_output_path(legacy_path: Path, doc: dict) -> Path:
    """Where a result a writer would have put at ``legacy_path`` lives in the mirrored tree.

    ``legacy_path`` under ``results/runtime/`` or ``results/breakdown/`` is placed by the
    day-zero migration map's own classification (``scripts/misc/tooling/
    build_migration_map.destination``): ``results/<dataset_type>/<variant>/<instrument>/
    <setup_id>/<host_class>[/<model_variant>]/<measurement>.json`` for a canonical row,
    ``.../scans|experiments/<name>/`` for a scan or arm, ``results/misc/<group>/`` when no
    cell is derivable. Any other path is returned unchanged.
    """
    import sys

    legacy_path = Path(legacy_path)
    if not _in_legacy_results(legacy_path):
        return legacy_path
    rel = legacy_path.resolve().relative_to(_REPO_ROOT)
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "misc" / "tooling"))
    from build_migration_map import destination

    new, kind = destination(_REPO_ROOT, rel.as_posix(), doc)
    if kind == "misc":
        print(f"[results] no derivable cell for {rel.as_posix()}; writing {new}")
    return _REPO_ROOT / new


def _migration_map_module():
    """``scripts/misc/tooling/build_migration_map`` (stdlib-only), imported on first use."""
    import sys

    tooling = str(_REPO_ROOT / "scripts" / "misc" / "tooling")
    if tooling not in sys.path:
        sys.path.insert(0, tooling)
    import build_migration_map

    return build_migration_map


def _legacy_rel(path: Path) -> tuple[str, ...] | None:
    """``("results", <old group>, ...)`` for a path under a pre-#442 results folder, else None."""
    try:
        rel = Path(path).resolve().relative_to(_REPO_ROOT).parts
    except ValueError:
        return None
    mm = _migration_map_module()
    if len(rel) < 2 or rel[0] != "results" or (rel[1] not in mm.CATEGORY | mm.MISC_GROUPS):
        return None
    return rel


def legacy_landing_dir(legacy_dir: Path) -> Path:
    """The folder where a file a writer names under the legacy ``legacy_dir`` keeps its name.

    ``results/runtime/<rest>`` -> ``results/misc/likelihood_runtime/<rest>``,
    ``results/breakdown/<rest>`` -> ``results/misc/likelihood_breakdown/<rest>`` (where
    :func:`mirrored_output_path` puts a row with no derivable cell) and
    ``results/<group>/<rest>`` -> ``results/misc/<MISC_GROUPS[group]>/<rest>`` for the other
    pre-#442 folders (``build_migration_map.MISC_GROUPS``). Any other path is returned unchanged,
    so an explicit ``--output-dir`` passes through.
    """
    rel = _legacy_rel(legacy_dir)
    if rel is None:
        return Path(legacy_dir)
    mm = _migration_map_module()
    group = mm.CATEGORY.get(rel[1]) or mm.MISC_GROUPS[rel[1]]
    return _REPO_ROOT.joinpath("results", "misc", group, *rel[2:])


def legacy_read_dirs(legacy_dir: Path) -> list[Path]:
    """Folders a reader globs for files a writer names under ``legacy_dir``, by that name.

    The landing folder (:func:`legacy_landing_dir`) first, then ``legacy_dir`` itself (the
    unresolved rows the day-zero move left in place, and runs given that folder verbatim as
    ``--output-dir``). A path outside the legacy folders is its own only folder.
    """
    landing = legacy_landing_dir(legacy_dir)
    return [landing] if landing == Path(legacy_dir) else [landing, Path(legacy_dir)]


def _day_zero_moves() -> dict[str, str]:
    """{old repo-relative path: new path} of ``catalogue/migration_day_zero.json`` ({} if absent)."""
    import json

    global _DAY_ZERO_MOVES
    if _DAY_ZERO_MOVES is None:
        try:
            doc = json.loads((_REPO_ROOT / "catalogue" / "migration_day_zero.json").read_text())
            entries = doc.get("entries", [])
        except (OSError, ValueError):
            entries = []
        _DAY_ZERO_MOVES = {e["old"]: e["new"] for e in entries if e.get("new") != e.get("old")}
    return _DAY_ZERO_MOVES


_DAY_ZERO_MOVES: dict[str, str] | None = None


def locate_legacy_result(legacy_path: Path, doc: dict | None = None) -> Path | None:
    """Where the file a writer names ``legacy_path`` lives now, or None when it is nowhere.

    Tried in order: ``legacy_path`` itself (an unresolved row left in place, or an explicit
    ``--output-dir`` run), the day-zero map's destination (a committed row), the landing folder
    (a placed row with no derivable cell keeps its name), and -- for a ``results/runtime`` /
    ``results/breakdown`` name -- the slot :func:`mirrored_output_path` gives it for ``doc``
    (the setup facts the reader knows, e.g. ``{"instrument": "hst"}``; a row written after the
    move with those facts) unless the map put a committed row of another legacy name there.
    Readers that parse legacy file names use this instead of joining
    the legacy folder.
    """
    legacy_path = Path(legacy_path)
    candidates = [legacy_path]
    rel = _legacy_rel(legacy_path)
    if rel is not None:
        moved = _day_zero_moves().get("/".join(rel))
        if moved:
            candidates.append(_REPO_ROOT / moved)
        candidates.append(legacy_landing_dir(legacy_path.parent) / legacy_path.name)
        if rel[1] in ("runtime", "breakdown"):
            new, _kind = _migration_map_module().destination(_REPO_ROOT, "/".join(rel), doc or {})
            # A committed row the map moved there belongs to its own legacy name, not this one.
            claimed = {o for o, n in _day_zero_moves().items() if n == new}
            if not claimed or claimed == {"/".join(rel)}:
                candidates.append(_REPO_ROOT / new)
    return next((p for p in candidates if p.exists()), None)


def legacy_result_files(legacy_dir: Path, pattern: str = "*.json") -> list[tuple[Path, str]]:
    """``(current path, legacy file name)`` of every file once or now named under ``legacy_dir``.

    The legacy-name equivalent of ``sorted(legacy_dir.glob(pattern))`` (``pattern`` matched on
    the legacy name, sorted by it): committed rows the day-zero map moved (renamed into a
    mirrored cell, archived, or kept by name), rows in the landing folder, and rows still in
    ``legacy_dir``. A later source wins a name clash. A row a run placed into a *new* mirrored cell after the move has
    no legacy name on disk; find it with :func:`locate_legacy_result`.
    """
    import fnmatch

    found: dict[str, Path] = {}
    rel = _legacy_rel(legacy_dir)
    if rel is not None:
        prefix = "/".join(rel) + "/"
        for old, new in _day_zero_moves().items():
            name = old[len(prefix) :]
            if old.startswith(prefix) and "/" not in name and fnmatch.fnmatch(name, pattern):
                if (_REPO_ROOT / new).is_file():
                    found[name] = _REPO_ROOT / new
    for folder in legacy_read_dirs(legacy_dir):
        if folder.is_dir():
            for path in folder.glob(pattern):
                if path.is_file():
                    found[path.name] = path
    return sorted(((p, n) for n, p in found.items()), key=lambda t: t[1])


def legacy_result_dirs(legacy_root: Path) -> list[Path]:
    """Every legacy folder under ``legacy_root`` that names a file, as legacy paths, sorted.

    The folders of the committed rows the day-zero map moved, plus the folders on disk under
    the landing folder and under ``legacy_root`` itself (mapped back onto ``legacy_root``).
    Pair with :func:`legacy_result_files` to walk a legacy tree after the move.
    """
    legacy_root = Path(legacy_root)
    dirs: set[Path] = set()
    rel = _legacy_rel(legacy_root)
    if rel is not None:
        prefix = "/".join(rel) + "/"
        for old in _day_zero_moves():
            if old.startswith(prefix) and "/" in old[len(prefix) :]:
                dirs.add(legacy_root / old[len(prefix) :].rsplit("/", 1)[0])
    for base in legacy_read_dirs(legacy_root):
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and path.parent != base:
                    dirs.add(legacy_root / path.parent.relative_to(base))
    return sorted(dirs)


def resolve_output_paths(
    cli: ProfileCLI,
    default_dir: Path,
    default_basename: str,
    cell: str | None = None,
    hints: dict | None = None,
    place: bool = True,
) -> tuple[Path, Path]:
    """Resolve (json_path, png_path) for the per-cell write.

    - When ``cli.config_name`` is unset: use
      ``<output_dir>/<default_basename>.{json,png}`` (the single-config
      filename pattern).
    - When ``cli.config_name`` is set: use ``<output_dir>/<cell>_<config_name>.{json,png}``,
      where ``<cell>`` defaults to the first ``_``-separated token of
      ``default_basename`` (the leaf scripts use ``<cell>_likelihood_summary_...`` /
      ``<cell>_breakdown_...`` so the cell name is usually the leading token).
      This keeps per-cell JSONs disjoint even when the same config name is
      shared across cells in a sweep — without it, every cell writes to the
      same ``<config_name>.json`` and the sweep loses 5 of 6 results to
      clobbering (the bug surfaced by the first A100 sparse-vs-dense sweep,
      autolens_profiling#44).
    - ``cell`` overrides that first-token derivation. **Required for any cell
      whose name itself contains an underscore**: ``delaunay_nn`` derives to
      ``delaunay`` under the default rule and would silently clobber the
      Delaunay cell's ``delaunay_<config_name>.json`` (autolens_profiling#219).
      Callers that pass nothing keep the pre-existing behaviour exactly.
    - ``cli.output_dir`` overrides ``default_dir`` when set.
    - A ``default_dir`` under ``results/runtime/`` or ``results/breakdown/`` (the pre-#442
      layout) is not written: the row is placed in the mirrored tree by
      :func:`mirrored_output_path` from the legacy name plus the setup facts known now
      (``--instrument`` or the instrument token of ``default_basename``,
      ``--use-mixed-precision``, the host and JAX backend). ``hints`` adds JSON-keyed facts
      the CLI does not carry (e.g. ``{"transformer": "TransformerNUFFT"}``). An explicit
      ``--output-dir`` is honoured verbatim: ``sweep.py`` -> ``aggregate.py`` and the HPC
      submits still name legacy cell folders whose config-grammar file names they parse,
      until the catalogue builder reads the mirrored tree (#442 PR 3). ``place=False``
      returns the legacy name for a caller that appends to it and then calls
      :func:`place_output_path`.
    - When ``cli.use_sparse_operator`` is set, ``_sparse`` is appended to the
      resolved basename so dense and sparse JSONs from the same config don't
      clobber each other.
    - When ``cli.source_pixels`` is set **and no ``--config-name`` was given**,
      ``_n<N>`` is appended, so an ad-hoc local run at a non-fiducial mesh can
      never overwrite the canonical single-config row. A sweep passes its own
      label (``--config-name hpc_a100_fp64_n3000``) and gets no extra suffix —
      the label already says the size.
    """
    if getattr(cli, "output_path", None) is not None:
        cli.output_path.parent.mkdir(parents=True, exist_ok=True)
        return cli.output_path, cli.output_path.with_suffix(".png")
    results_dir = cli.output_dir if cli.output_dir is not None else default_dir
    if cli.output_dir is None and _in_legacy_results(results_dir):
        results_dir = runtime_instrument_dir(results_dir, _row_instrument(cli, default_basename))
    if cli.config_name is None:
        basename = default_basename
    else:
        # First underscore-separated token of default_basename is the cell,
        # unless the caller named it explicitly. All callers
        # (likelihood_runtime, likelihood_breakdown) follow the
        # ``<cell>_<purpose>_<inst>_v<version>`` convention.
        cell_name = cell if cell is not None else default_basename.split("_", 1)[0]
        basename = f"{cell_name}_{cli.config_name}"
    if cli.config_name is None and cli.source_pixels is not None:
        basename = f"{basename}_n{int(cli.source_pixels)}"
    if cli.use_sparse_operator:
        basename = f"{basename}_sparse"
    if cli.rect_mesh == "rtu":
        # Keep the two rectangular-mesh families' results disjoint; the
        # bilinear default keeps the unsuffixed (pre-split) filenames.
        basename = f"{basename}_rtu"
    json_path = results_dir / f"{basename}.json"
    if place and cli.output_dir is None and _in_legacy_results(json_path):
        placed = place_output_path(json_path, cli, default_basename, hints)
        return placed, placed.with_suffix(".png")
    if place or not _in_legacy_results(json_path):
        results_dir.mkdir(parents=True, exist_ok=True)
    return json_path, results_dir / f"{basename}.png"


#: The file name every cell's fit figure is kept under, beside its result JSON.
FIT_PNG = "fit.png"

#: Dataset types :func:`render_fit_png` draws, each with the plotter call the matching
#: ``autolens_visualization/scripts/<type>/visualization.py`` producer keeps as ``fit.png``:
#:
#: - ``imaging``: ``PlotterImaging.fit_imaging`` -> ``subplot_fit`` (the imaging producer's
#:   ``VisualizerImaging.visualize``);
#: - ``interferometer``: ``PlotterInterferometer.fit_interferometer`` -> ``subplot_fit`` (the
#:   interferometer and interferometer_datacube producers' ``VisualizerInterferometer.visualize``;
#:   a datacube cell passes one channel's fit);
#: - ``point_source``: ``PlotterPoint.fit_point`` -> point ``subplot_fit`` with the curves of
#:   ``curves_for_point_fits`` (both point-source producers' ``VisualizerPoint.visualize``);
#: - ``cluster``: ``PlotterPoint.fit_point_combined`` -> ``plot_fit_positions`` of every source
#:   (the cluster producer's ``VisualizerPoint.visualize_combined``); ``fit`` is the list of fits.
#:
#: Each is called with ``quick_update=True``, the switch that always writes the plain ``fit.png``
#: whatever ``config/visualize/plots.yaml`` says.
FIT_PNG_DATASET_TYPES = ("imaging", "interferometer", "point_source", "cluster")


def fit_png_enabled(cli: ProfileCLI | None = None) -> bool:
    """Whether a cell renders its ``fit.png``: never under smoke, else unless ``--no-fit-png``."""
    if os.environ.get("AUTOLENS_PROFILING_SMOKE"):
        return False
    return bool(getattr(cli, "fit_png", True))


def _draw_fit(fit, image_path: Path, dataset_type: str, solver=None) -> None:
    """Write ``image_path/fit.png`` with the viz producer's plotter (see FIT_PNG_DATASET_TYPES)."""
    if dataset_type == "imaging":
        from autolens.imaging.model.plotter import PlotterImaging

        PlotterImaging(image_path=image_path).fit_imaging(fit=fit, quick_update=True)
    elif dataset_type == "interferometer":
        from autolens.interferometer.model.plotter import PlotterInterferometer

        PlotterInterferometer(image_path=image_path).fit_interferometer(fit=fit, quick_update=True)
    elif dataset_type in ("point_source", "cluster"):
        from autolens.point.model.plotter import PlotterPoint
        from autolens.point.model.visualizer import curves_for_point_fits

        fits = list(fit) if dataset_type == "cluster" else [fit]
        solver = solver if solver is not None else getattr(fits[0], "solver", None)
        curves = curves_for_point_fits(fits=fits, solver=solver)
        plotter = PlotterPoint(image_path=image_path)
        if dataset_type == "cluster":
            plotter.fit_point_combined(
                fit_list=fits, curves=curves, solver=solver, quick_update=True
            )
        else:
            plotter.fit_point(fit=fits[0], quick_update=True, curves=curves)
    else:
        raise ValueError(
            f"no fit.png plotter for dataset_type={dataset_type!r}; "
            f"expected one of {FIT_PNG_DATASET_TYPES}"
        )


def fit_png_cell(cell_dir: Path) -> bool:
    """Whether ``cell_dir`` holds one lifecycle row: a mirrored-tree cell (or its own arm/scan).

    Only there is ``fit.png`` unambiguous: ``build_catalogue`` attaches it to the rows beside it.
    A flat folder an explicit ``--output-dir`` names (``results/misc/...``, a legacy cell folder,
    a scratch path) collects several configs' rows, so one ``fit.png`` there would belong to
    whichever wrote last.

    The test is the path from the folder's last ``results`` component, so a scratch tree laid out
    like ``results/`` (``<scratch>/results/imaging/delaunay/hst/<setup_id>/<host_class>/``, inside
    the repo's gitignored ``output/`` or anywhere else) is a cell too: it is just as unambiguous,
    and it is how a laptop proof or a timing arm keeps its row, figure and witness out of
    ``results/``. A folder with no ``results`` component is never a cell.
    """
    parts = Path(cell_dir).resolve().parts
    if "results" not in parts:
        return False
    rel = Path(*parts[len(parts) - 1 - parts[::-1].index("results") :])
    tooling = str(_REPO_ROOT / "scripts" / "misc" / "tooling")
    if tooling not in sys.path:
        sys.path.insert(0, tooling)
    import catalogue_lifecycle

    return catalogue_lifecycle.mirrored((rel / "runtime.json").as_posix()) is not None


def render_fit_png(
    fit,
    cell_dir: Path,
    *,
    dataset_type: str,
    cli: ProfileCLI | None = None,
    solver=None,
) -> Path | None:
    """Render the cell's ``fit.png`` into ``cell_dir`` and return its path (None when not drawn).

    Call it right after the row is written and after every timed region, with ``cell_dir`` the
    folder of the placed result JSON (``resolve_output_paths`` / ``place_output_path`` /
    ``--output-path``), so the figure sits beside the row and ``build_catalogue`` attaches it as
    ``figures[]``. ``fit`` must be an eager (NumPy) fit at the profiled instance.

    The figure beside a row always belongs to that row:

    - the old ``fit.png`` is removed before drawing, so a failed draw leaves none;
    - with ``--no-fit-png`` the old one is removed and nothing is drawn;
    - under ``AUTOLENS_PROFILING_SMOKE`` nothing is touched (a smoke run writes no row);
    - outside a mirrored-tree cell (:func:`fit_png_cell`) nothing is touched or drawn.

    The plotter draws into a system temp folder; ``fit.png`` is copied into the cell as
    ``.fit_png.tmp`` (gitignored) and renamed into place. Never raises: a figure that fails to
    draw must not cost the measurement it describes.
    """
    import shutil
    import tempfile

    if os.environ.get("AUTOLENS_PROFILING_SMOKE"):
        return None
    cell_dir = Path(cell_dir)
    target = cell_dir / FIT_PNG
    try:
        if not fit_png_cell(cell_dir):
            print(f"  [fit.png] skipped: {cell_dir} is not a mirrored-tree cell")
            return None
        target.unlink(missing_ok=True)
        if not fit_png_enabled(cli):
            return None
        cell_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fit_png_") as tmp:
            _draw_fit(fit, Path(tmp), dataset_type, solver=solver)
            written = Path(tmp) / FIT_PNG
            if not written.is_file():
                raise FileNotFoundError(f"the {dataset_type} plotter wrote no {FIT_PNG}")
            staged = cell_dir / ".fit_png.tmp"
            shutil.copyfile(written, staged)
            os.replace(staged, target)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        (cell_dir / ".fit_png.tmp").unlink(missing_ok=True)
        print(
            f"  WARNING: [fit.png] NOT RENDERED for {cell_dir} ({dataset_type}): "
            f"{type(exc).__name__}: {exc}. The row is kept without a figure."
        )
        return None
    print(f"  [fit.png] {target}")
    return target


#: The file name every cell's parity witness is kept under, beside its result JSON
#: (``scripts/misc/tooling/parity_witness.py``; results/README.md "Parity witness").
PARITY_JSON = "parity.json"

#: Relative tolerance of a parity witness, as a fraction of |reference log-likelihood| (human
#: ruling 2026-10-24): 1e-4 for an fp64 cell, 1e-3 for a mixed-precision one. The witness stores
#: the resulting ABSOLUTE tolerance.
PARITY_RTOL = {"fp64": 1e-4, "mixed": 1e-3}


def parity_witness_doc(
    *,
    log_likelihood,
    reference_log_likelihood,
    use_mixed_precision: bool,
    reference_source: str,
    autolens_version: str,
    source_pixels: int | None = None,
    source_pixels_requested: int | None = None,
    quantity: str | None = None,
) -> dict:
    """One ``parity-witness`` v1 document (the schema of ``parity_witness.validate``).

    ``status`` is ``match`` when ``|log_likelihood - reference| <= tolerance`` and ``diverged``
    above it, with ``tolerance = PARITY_RTOL[precision] * |reference|``; a non-finite value on
    either side is ``not_assessed`` with its reason (the numbers are then ``null``).
    """
    import datetime as _dt
    import math

    now = _dt.datetime.now(_dt.UTC)
    precision = "mixed" if use_mixed_precision else "fp64"
    rtol = PARITY_RTOL[precision]
    doc = {
        "schema": "parity-witness",
        "version": 1,
        "status": "not_assessed",
        "log_likelihood": None,
        "reference_log_likelihood": None,
        "tolerance": None,
        "tolerance_rule": f"{rtol:g} x |reference_log_likelihood| ({precision})",
        "reference": {"source": reference_source, "date": now.date().isoformat()},
        "measured_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "autolens_version": str(autolens_version),
    }
    if quantity:
        doc["quantity"] = quantity
    if source_pixels is not None:
        doc["source_pixels"] = int(source_pixels)
    if source_pixels_requested is not None:
        doc["source_pixels_requested"] = int(source_pixels_requested)
    try:
        value, reference = float(log_likelihood), float(reference_log_likelihood)
    except (TypeError, ValueError):
        value = reference = float("nan")
    if not (math.isfinite(value) and math.isfinite(reference)):
        doc["reason"] = (
            f"non-finite log-likelihood (cell {log_likelihood!r}, reference "
            f"{reference_log_likelihood!r}); no comparison made"
        )
        return doc
    tolerance = rtol * abs(reference)
    doc.update(
        status="match" if abs(value - reference) <= tolerance else "diverged",
        log_likelihood=value,
        reference_log_likelihood=reference,
        tolerance=tolerance,
    )
    return doc


def _parity_validate(doc: dict) -> list[str]:
    tooling = str(_REPO_ROOT / "scripts" / "misc" / "tooling")
    if tooling not in sys.path:
        sys.path.insert(0, tooling)
    import parity_witness

    return parity_witness.validate(doc)


def write_parity_json(cell_dir: Path, **witness) -> Path | None:
    """Write the cell's ``parity.json`` into ``cell_dir`` and return its path (None when not).

    ``witness`` is :func:`parity_witness_doc`'s keywords: the cell's log-likelihood, the eager
    NumPy reference computed in the same process on the same dataset, size, instrument and model,
    and its provenance. Call it after the row is written and after every timed region, with
    ``cell_dir`` the folder of the placed result JSON — the same placement and guard rules as
    :func:`render_fit_png`, so the witness beside a row always belongs to that row:

    - the old ``parity.json`` is removed first, so a run that fails here (or a cell rewritten
      without a witness: pass ``log_likelihood=None`` or call :func:`remove_parity_json`) leaves
      none;
    - under ``AUTOLENS_PROFILING_SMOKE`` nothing is touched (a smoke run writes no row);
    - outside a mirrored-tree cell (:func:`fit_png_cell`) nothing is touched or written.

    The document is validated with ``parity_witness.validate`` before it is written. Never raises:
    a witness that fails must not cost the measurement it describes.
    """
    if os.environ.get("AUTOLENS_PROFILING_SMOKE"):
        return None
    import json

    cell_dir = Path(cell_dir)
    target = cell_dir / PARITY_JSON
    staged = cell_dir / ".parity_json.tmp"
    try:
        if not fit_png_cell(cell_dir):
            print(f"  [parity.json] skipped: {cell_dir} is not a mirrored-tree cell")
            return None
        target.unlink(missing_ok=True)
        if witness.get("log_likelihood") is None:
            return None
        doc = parity_witness_doc(**witness)
        errors = _parity_validate(doc)
        if errors:
            raise ValueError("invalid parity witness: " + "; ".join(errors))
        staged.write_text(json.dumps(doc, indent=2) + "\n")
        os.replace(staged, target)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        staged.unlink(missing_ok=True)
        print(
            f"  WARNING: [parity.json] NOT WRITTEN for {cell_dir}: {type(exc).__name__}: {exc}. "
            "The row is kept without a parity witness."
        )
        return None
    delta = (
        abs(doc["log_likelihood"] - doc["reference_log_likelihood"])
        if doc["status"] != "not_assessed"
        else None
    )
    print(
        f"  [parity.json] {target}: {doc['status']} (|delta log L| = {delta!r}, "
        f"tolerance {doc['tolerance']!r})"
    )
    return target


def remove_parity_json(cell_dir: Path) -> None:
    """Remove a stale ``parity.json`` from a mirrored cell (same guards as :func:`write_parity_json`)."""
    write_parity_json(cell_dir, log_likelihood=None)


def auto_simulate_if_missing(
    dataset_path: Path,
    *,
    dataset_type: str,
    instrument: str,
    workspace_root: Path,
) -> None:
    """If the dataset is missing, invoke the matching simulator script.

    ``dataset_type`` maps to ``scripts/misc/simulators/<dataset_type>.py``
    (imaging / interferometer / point_source / cluster / …). The simulator is
    invoked via subprocess with ``--instrument <instrument>``, so both the
    likelihood-fit dataset and a versioned simulator-profiling JSON+PNG
    land at the right path in one shot.

    The dataset gate uses ``al.util.dataset.should_simulate`` (which also
    handles the ``PYAUTO_SMALL_DATASETS=1`` cleanup case). ``autolens`` is
    imported lazily so this helper can sit in any module without forcing
    the heavy import chain on every caller.
    """
    import sys

    import autolens as al  # noqa: F401 — imported lazily to defer side effects

    if not al.util.dataset.should_simulate(str(dataset_path)):
        return

    simulator_script = workspace_root / "scripts" / "misc" / "simulators" / f"{dataset_type}.py"
    if not simulator_script.exists():
        raise FileNotFoundError(
            f"Auto-simulate could not find simulator script at {simulator_script}. "
            f"Expected <dataset_type>.py (imaging / interferometer / point_source / cluster / …) "
            f"under scripts/misc/simulators/."
        )

    print(
        f"  [auto-simulate] {dataset_path} missing; invoking "
        f"scripts/misc/simulators/{dataset_type}.py --instrument {instrument}"
    )
    subprocess.run(
        [
            sys.executable,
            str(simulator_script),
            "--instrument",
            instrument,
            "--output-root",
            str(workspace_root),
        ],
        check=True,
    )


def check_pinned(got, expected, *, label: str, rtol: float = 1e-4):
    """Compare a computed likelihood/evidence against its pinned baseline.

    Profiling runs **record and flag** drift; they never adjudicate library
    correctness (that is autolens_workspace_test's remit — see
    ``wiki/campaigns/misc/design_lock_in.md``). Returns ``None`` when ``got`` is
    within ``rtol`` of ``expected``; otherwise prints a loud warning and
    returns a drift record for the result JSON, which PyAutoHeart's vitals
    scan picks up. Never raises — a changed computation must not kill a
    profiling job (the timing numbers are still data; the flag marks them
    non-comparable to the pinned baseline).
    """
    import numpy as _np

    arr = _np.asarray(got, dtype=float).ravel()
    rel = float(_np.max(_np.abs(arr - expected)) / max(abs(expected), 1e-300))
    got_f = float(arr[0]) if arr.size == 1 else float(arr[int(_np.argmax(_np.abs(arr - expected)))])
    if rel <= rtol:
        return None
    print(
        f"  WARNING: PINNED-VALUE DRIFT [{label}] — got {got_f!r}, "
        f"pinned {expected!r} (rel diff {rel:.3e} > rtol {rtol:g}). "
        f"Timings from this run are NOT comparable to the pinned baseline; "
        f"file a bug / check autolens_workspace_test before trusting trends."
    )
    return {
        "label": label,
        "expected": expected,
        "got": got_f,
        "rel_diff": rel,
        "rtol": rtol,
    }


def check_pinned_vector(got, expected, *, label: str, rtol: float = 1e-6):
    """Vector sibling of :func:`check_pinned` — compare an array of pinned values.

    Used by the deflection cells (``scripts/lens/deflections/``), whose pin is a
    16-coordinate ``(y, x)`` deflection sample rather than a single scalar. The
    comparison reduces to the **maximum relative deviation** over the flattened
    pair, and keeps ``check_pinned``'s soft-failure discipline exactly: returns
    ``None`` when every element is within ``rtol``; otherwise prints a loud
    warning and returns a drift record for the result JSON. Never raises — a
    changed computation must not kill a profiling job.
    """
    import numpy as _np

    got_arr = _np.asarray(got, dtype=float).ravel()
    exp_arr = _np.asarray(expected, dtype=float).ravel()

    if got_arr.shape != exp_arr.shape:
        print(
            f"  WARNING: PINNED-VECTOR SHAPE CHANGE [{label}] — got {got_arr.size} value(s), "
            f"pinned {exp_arr.size}. The pin and the computation no longer describe the same "
            f"quantity; re-pin deliberately with --repin --repin-reason."
        )
        return {
            "label": label,
            "expected_size": int(exp_arr.size),
            "got_size": int(got_arr.size),
            "rel_diff": float("inf"),
            "rtol": rtol,
        }

    # NaN is a legitimate pinned value here (a mass profile whose deflection
    # field is non-finite somewhere on the grid pins that fact). Two NaNs at the
    # same index match; a NaN facing a number — in either direction — is the
    # loudest possible drift, so it reduces to inf rather than to NaN.
    with _np.errstate(invalid="ignore"):
        rel_each = _np.abs(got_arr - exp_arr) / _np.maximum(_np.abs(exp_arr), 1e-300)
    rel_each = _np.where(_np.isnan(got_arr) & _np.isnan(exp_arr), 0.0, rel_each)
    rel_each = _np.where(_np.isnan(rel_each), _np.inf, rel_each)
    idx = int(_np.argmax(rel_each))
    rel = float(rel_each[idx])
    if rel <= rtol:
        return None
    print(
        f"  WARNING: PINNED-VALUE DRIFT [{label}] — element {idx} got {float(got_arr[idx])!r}, "
        f"pinned {float(exp_arr[idx])!r} (max rel diff {rel:.3e} > rtol {rtol:g}). "
        f"Timings from this run are NOT comparable to the pinned baseline; "
        f"file a bug / check autolens_workspace_test before trusting trends."
    )
    return {
        "label": label,
        "index": idx,
        "expected": float(exp_arr[idx]),
        "got": float(got_arr[idx]),
        "rel_diff": rel,
        "rtol": rtol,
    }


def record_pinned_check(json_path, expected, drift_records) -> None:
    """Merge the pinned-value check outcome into an already-written result JSON.

    Adds ``pinned_expected`` (the baseline value, or ``None`` when the
    instrument has no pin) and ``pinned_drift`` (list of drift records —
    empty means every compared value matched). The guard blocks run after
    the summary JSON is written, so this rewrites the file in place.
    """
    import json

    data = json.loads(json_path.read_text())
    data["pinned_expected"] = expected
    data["pinned_drift"] = drift_records
    json_path.write_text(json.dumps(data, indent=2))
