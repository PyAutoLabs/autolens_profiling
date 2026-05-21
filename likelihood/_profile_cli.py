"""Shared CLI/JSON helpers for the multi-config likelihood profiling sweep.

The single-config likelihood scripts under ``likelihood/<class>/<model>.py``
each emit one JSON+PNG per run. To drive the CPU/GPU/A100 x fp64/mp matrix
the sweep harness invokes each script multiple times with different env
(``JAX_PLATFORM_NAME``) and flags, and needs the JSONs to land at a
matrix-friendly path (``local_cpu_fp64.json`` etc.) rather than the
single-config ``<model>_likelihood_summary_<instrument>_v<version>.json``
pattern.

This module centralises the parse / resolve / device-info logic so each
script only needs three lines of glue (parse args, override Settings flag,
override results path).

Designed to be imported with relative path manipulation since the scripts
live under multiple sibling directories::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _profile_cli import parse_profile_cli, device_info_dict
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


def parse_profile_cli(default_config_name: Optional[str] = None) -> ProfileCLI:
    """Parse the three sweep CLI flags accepted by every per-cell profile script.

    Returns ``ProfileCLI(config_name, output_dir, use_mixed_precision)``.

    When ``--config-name`` is omitted, falls back to ``default_config_name``
    (typically inferred from ``JAX_PLATFORM_NAME`` env var or left as ``None``
    to preserve the existing single-config filename pattern).
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
            "Override results dir. Defaults to "
            "<autolens_profiling>/results/likelihood/<class>/."
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

    args, _unknown = parser.parse_known_args()
    config_name = args.config_name or default_config_name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    return ProfileCLI(
        config_name=config_name,
        output_dir=output_dir,
        use_mixed_precision=bool(args.use_mixed_precision),
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

    - When ``cli.config_name`` is set: use ``<output_dir>/<config_name>.{json,png}``.
    - Otherwise: use ``<output_dir>/<default_basename>.{json,png}`` to preserve
      the existing single-config filename pattern.
    - ``cli.output_dir`` overrides ``default_dir`` when set.
    """
    results_dir = cli.output_dir if cli.output_dir is not None else default_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    basename = cli.config_name or default_basename
    return results_dir / f"{basename}.json", results_dir / f"{basename}.png"
