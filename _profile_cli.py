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
    config_name: Optional[str]
    output_dir: Optional[Path]
    use_mixed_precision: bool
    instrument: Optional[str]
    vmap_probe: bool
    use_sparse_operator: bool


def parse_profile_cli(default_config_name: Optional[str] = None) -> ProfileCLI:
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
            "Call ``dataset.apply_sparse_operator(use_jax=True)`` after "
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
    )


def device_info_dict() -> dict:
    """Capture backend / device / nvidia-smi summary for the current JAX process.

    Imports jax lazily so callers can collect this near the JSON write without
    re-importing.
    """
    import jax

    info = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }
    if info["backend"] == "gpu":
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            info["nvidia_smi"] = out.replace("\n", "; ")
        except Exception:
            pass
    return info


def resolve_output_paths(
    cli: ProfileCLI,
    default_dir: Path,
    default_basename: str,
) -> tuple[Path, Path]:
    """Resolve (json_path, png_path) for the per-cell write.

    - When ``cli.config_name`` is unset: use
      ``<output_dir>/<default_basename>.{json,png}`` (the single-config
      filename pattern).
    - When ``cli.config_name`` is set: use ``<output_dir>/<cell>_<config_name>.{json,png}``,
      where ``<cell>`` is the first ``_``-separated token of ``default_basename``
      (the leaf scripts use ``<cell>_likelihood_summary_...`` /
      ``<cell>_breakdown_...`` so the cell name is always the leading token).
      This keeps per-cell JSONs disjoint even when the same config name is
      shared across cells in a sweep — without it, every cell writes to the
      same ``<config_name>.json`` and the sweep loses 5 of 6 results to
      clobbering (the bug surfaced by the first A100 sparse-vs-dense sweep,
      autolens_profiling#44).
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
        # First underscore-separated token of default_basename is the cell.
        # All callers (likelihood_runtime, likelihood_breakdown) follow the
        # ``<cell>_<purpose>_<inst>_v<version>`` convention.
        cell = default_basename.split("_", 1)[0]
        basename = f"{cell}_{cli.config_name}"
    if cli.use_sparse_operator:
        basename = f"{basename}_sparse"
    return results_dir / f"{basename}.json", results_dir / f"{basename}.png"


def auto_simulate_if_missing(
    dataset_path: Path,
    *,
    dataset_type: str,
    instrument: str,
    workspace_root: Path,
) -> None:
    """If the dataset is missing, invoke the matching simulator script.

    ``dataset_type`` maps to ``simulators/<dataset_type>.py`` (one of
    ``imaging``, ``interferometer``, ``point_source``). The simulator is
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

    simulator_script = workspace_root / "simulators" / f"{dataset_type}.py"
    if not simulator_script.exists():
        raise FileNotFoundError(
            f"Auto-simulate could not find simulator script at {simulator_script}. "
            f"Expected one of imaging.py / interferometer.py / point_source.py "
            f"under simulators/."
        )

    print(
        f"  [auto-simulate] {dataset_path} missing; invoking "
        f"simulators/{dataset_type}.py --instrument {instrument}"
    )
    subprocess.run(
        [
            sys.executable,
            str(simulator_script),
            "--instrument", instrument,
            "--output-root", str(workspace_root),
        ],
        check=True,
    )
