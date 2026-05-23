"""Multi-config latent profiling driver.

Runs each latent cell across the CPU/GPU x fp64/mp matrix (4 configs per
cell locally; HPC A100 configs are dispatched separately via
`z_projects/profiling/hpc/sync`).

Each subprocess invokes the per-latent script under
``autolens_profiling/latent/<class>/<latent>.py`` with the CLI args
``--config-name``, ``--output-dir``, ``--use-mixed-precision``.
Per-config JSONs land at::

    <output_root>/<class>/<latent>/<config_name>.json
    <output_root>/<class>/<latent>/<config_name>.log    (captured stdout/stderr)

Default ``--output-root`` is
``autolens_workspace_developer/jax_profiling/results/latent`` — mirrors the
``jit/`` convention used by ``likelihood_runtime/sweep.py`` and is read by
``aggregate.py`` to produce ``comparison.json`` / ``comparison.png``.

Usage::

    # All latent cells, all backends
    python latent/sweep.py

    # Skip the heaviest latent during iteration
    python latent/sweep.py --skip imaging/effective_einstein_radius

    # Single latent, CPU fp64 only
    python latent/sweep.py --only imaging/magnification --skip-gpu --skip-mp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]                 # autolens_profiling/
_WT_ROOT = _REPO_ROOT.parent                                     # PyAutoLabs-wt/<task>/ (or PyAutoLabs/)
_DEFAULT_OUTPUT_ROOT = _WT_ROOT / "autolens_workspace_developer" / "jax_profiling" / "results" / "latent"
_DEFAULT_PYTHON = "/home/jammy/venv/PyAutoGPU/bin/python"


# (dataset_class, latent_name). Order is roughly cheapest -> heaviest so
# failures surface quickly during iteration.
CELLS: list[tuple[str, str]] = [
    ("imaging", "total_lens_flux_mujy"),
    ("imaging", "total_lensed_source_flux_mujy"),
    ("imaging", "total_source_flux_mujy"),
    ("imaging", "magnification"),
    ("imaging", "effective_einstein_radius"),
]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    env_overrides: dict[str, str]
    extra_args: tuple[str, ...]
    is_gpu: bool


# CPU configs explicitly pin platform to cpu. GPU configs explicitly pin to
# cuda — we DO NOT let JAX fall back to CPU on GPU rows, so a missing CUDA
# device fails the run loudly rather than silently producing a CPU number.
CONFIGS: list[SweepConfig] = [
    SweepConfig(
        name="local_cpu_fp64",
        env_overrides={"JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"},
        extra_args=(),
        is_gpu=False,
    ),
    SweepConfig(
        name="local_cpu_mp",
        env_overrides={"JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"},
        extra_args=("--use-mixed-precision",),
        is_gpu=False,
    ),
    SweepConfig(
        name="local_gpu_fp64",
        env_overrides={"JAX_PLATFORM_NAME": "cuda", "JAX_PLATFORMS": "cuda,cpu"},
        extra_args=(),
        is_gpu=True,
    ),
    SweepConfig(
        name="local_gpu_mp",
        env_overrides={"JAX_PLATFORM_NAME": "cuda", "JAX_PLATFORMS": "cuda,cpu"},
        extra_args=("--use-mixed-precision",),
        is_gpu=True,
    ),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="CLASS/LATENT",
        help="Only run these cells (e.g. imaging/magnification imaging/effective_einstein_radius).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=(),
        metavar="CLASS/LATENT",
        help="Skip these cells (applied after --only).",
    )
    p.add_argument("--skip-cpu", action="store_true", help="Skip local_cpu_* configs.")
    p.add_argument("--skip-gpu", action="store_true", help="Skip local_gpu_* configs.")
    p.add_argument(
        "--skip-mp",
        action="store_true",
        help="Skip the use_mixed_precision rows (just fp64).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Root output dir. Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--python",
        default=_DEFAULT_PYTHON,
        help=f"Python interpreter to invoke per subprocess. Default: {_DEFAULT_PYTHON}",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned subprocess commands but don't execute.",
    )
    return p.parse_args()


def _resolve_cells(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected = CELLS
    if args.only:
        wanted = {c for c in args.only}
        selected = [(c, m) for (c, m) in selected if f"{c}/{m}" in wanted]
        missing = wanted - {f"{c}/{m}" for (c, m) in selected}
        if missing:
            sys.stderr.write(f"warning: --only includes unknown cells: {sorted(missing)}\n")
    skip = set(args.skip)
    selected = [(c, m) for (c, m) in selected if f"{c}/{m}" not in skip]
    return selected


def _resolve_configs(args: argparse.Namespace) -> list[SweepConfig]:
    configs = list(CONFIGS)
    if args.skip_cpu:
        configs = [c for c in configs if c.is_gpu]
    if args.skip_gpu:
        configs = [c for c in configs if not c.is_gpu]
    if args.skip_mp:
        configs = [c for c in configs if "--use-mixed-precision" not in c.extra_args]
    return configs


def _run_one(
    python: str,
    script_path: Path,
    config: SweepConfig,
    out_dir: Path,
    dry_run: bool,
) -> tuple[bool, float, str]:
    """Run one (cell, config) pair as a subprocess. Returns (ok, elapsed, log_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{config.name}.log"

    cmd = [
        python,
        str(script_path),
        "--config-name", config.name,
        "--output-dir", str(out_dir),
        *config.extra_args,
    ]

    env = dict(os.environ)
    env.update(config.env_overrides)
    # numba + matplotlib cache dirs — same workaround the per-cell scripts use.
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    print(f"\n--- [{config.name}] {script_path.relative_to(_REPO_ROOT)} ---")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"    env: {config.env_overrides}")

    if dry_run:
        return True, 0.0, ""

    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        print(f"    {'OK ' if ok else 'FAIL'} ({elapsed:.1f}s, exit={proc.returncode}) -> {log_path.name}")
        return ok, elapsed, str(log_path)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"    INTERRUPTED after {elapsed:.1f}s; partial log -> {log_path}")
        raise


def main() -> int:
    args = _parse_args()
    cells = _resolve_cells(args)
    configs = _resolve_configs(args)

    print(f"sweep_latent: {len(cells)} cells x {len(configs)} configs "
          f"= {len(cells) * len(configs)} runs")
    print(f"  cells:    {[f'{c}/{m}' for (c, m) in cells]}")
    print(f"  configs:  {[c.name for c in configs]}")
    print(f"  output:   {args.output_root}")
    print(f"  python:   {args.python}")
    if args.dry_run:
        print("  (dry-run)")

    summary: list[tuple[str, str, bool, float]] = []
    overall_t0 = time.time()

    for (cls, latent) in cells:
        script_path = _REPO_ROOT / "latent" / cls / f"{latent}.py"
        if not script_path.exists():
            print(f"\n!!! missing script: {script_path}")
            for cfg in configs:
                summary.append((f"{cls}/{latent}", cfg.name, False, 0.0))
            continue

        out_dir = args.output_root / cls / latent

        for cfg in configs:
            try:
                ok, elapsed, _log = _run_one(
                    args.python, script_path, cfg, out_dir, args.dry_run
                )
            except KeyboardInterrupt:
                print("\n\nsweep interrupted by user")
                return 130
            summary.append((f"{cls}/{latent}", cfg.name, ok, elapsed))

    total = time.time() - overall_t0
    print("\n" + "=" * 70)
    print(f"sweep_latent summary  ({total:.1f}s total)")
    print("=" * 70)
    print(f"  {'cell':<40}{'config':<22}{'ok':<6}{'elapsed':>10}")
    print(f"  {'-'*40}{'-'*22}{'-'*6}{'-'*10}")
    failures = 0
    for cell, cfg, ok, t in summary:
        flag = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  {cell:<40}{cfg:<22}{flag:<6}{t:>9.1f}s")
    if failures:
        print(f"\n  {failures} run(s) FAILED — check the .log files in each cell's output dir.")
        return 1
    print("\n  All runs OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
