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


def parse_profile_cli(default_config_name: str | None = None) -> ProfileCLI:
    """Parse the sweep CLI flags accepted by every per-cell profile script.

    Returns ``ProfileCLI(config_name, output_dir, use_mixed_precision,
    instrument)``.

    When ``--config-name`` is omitted, falls back to ``default_config_name``
    (typically inferred from ``JAX_PLATFORM_NAME`` env var or left as ``None``
    to preserve the existing single-config filename pattern).

    ``--instrument`` is optional; when omitted (None) per-cell scripts keep
    their module-level hardcoded default (typically ``"sma"`` or ``"hst"``).
    """
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
            "from a 100 %-correct previous solve. The library default is "
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

    args, _unknown = parser.parse_known_args()
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
    )


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


def resolve_output_paths(
    cli: ProfileCLI,
    default_dir: Path,
    default_basename: str,
    cell: str | None = None,
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
    - When ``cli.use_sparse_operator`` is set, ``_sparse`` is appended to the
      resolved basename so dense and sparse JSONs from the same config don't
      clobber each other.
    """
    results_dir = cli.output_dir if cli.output_dir is not None else default_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if cli.config_name is None:
        basename = default_basename
    else:
        # First underscore-separated token of default_basename is the cell,
        # unless the caller named it explicitly. All callers
        # (likelihood_runtime, likelihood_breakdown) follow the
        # ``<cell>_<purpose>_<inst>_v<version>`` convention.
        cell_name = cell if cell is not None else default_basename.split("_", 1)[0]
        basename = f"{cell_name}_{cli.config_name}"
    if cli.use_sparse_operator:
        basename = f"{basename}_sparse"
    if cli.rect_mesh == "rtu":
        # Keep the two rectangular-mesh families' results disjoint; the
        # bilinear default keeps the unsuffixed (pre-split) filenames.
        basename = f"{basename}_rtu"
    return results_dir / f"{basename}.json", results_dir / f"{basename}.png"


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
    ``results/notes/design_lock_in.md``). Returns ``None`` when ``got`` is
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
