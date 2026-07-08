"""Multi-config likelihood profiling driver.

Runs each in-scope cell across the CPU/GPU x fp64/mp matrix (4 configs per
cell locally; HPC A100 configs are dispatched separately via the SLURM
submit scripts under ``hpc/``).

Each subprocess invokes the existing per-cell likelihood script under
``autolens_profiling/likelihood_runtime/<class>/<model>.py`` with the
CLI args ``--config-name``, ``--output-dir``, ``--use-mixed-precision``.
Per-config JSONs land at::

    <output_root>/<class>/<model>/<config_name>.json
    <output_root>/<class>/<model>/<config_name>.png
    <output_root>/<class>/<model>/<config_name>.log    (captured stdout/stderr)

Default ``--output-root`` is ``results/runtime/`` in this repo, read by
``aggregate.py`` to produce ``comparison.json`` / ``comparison.png``.
(Earlier sweeps wrote to ``autolens_workspace_developer/jax_profiling/
results/jit`` — that tree remains readable history but is no longer the
default; see ``results/notes/design_lock_in.md``.)

Usage::

    # All in-scope cells, both backends
    python likelihood_runtime/sweep.py

    # Skip the heaviest cell during iteration
    python likelihood_runtime/sweep.py --skip datacube/delaunay

    # Single cell, single backend
    python likelihood_runtime/sweep.py --only interferometer/mge --skip-cpu

    # Imaging sparse-operator (w-tilde) rows — filenames gain a _sparse suffix
    python likelihood_runtime/sweep.py --only imaging/pixelization imaging/delaunay --sparse
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "runtime"
_DEFAULT_PYTHON = sys.executable


# (dataset_class, model, instruments). Order is roughly cheapest -> heaviest
# so failures surface quickly during iteration. An empty instrument tuple
# means "the per-cell script's module default" (pre-campaign behaviour);
# named instruments run once each, with per-instrument output subdirs
# (results/runtime/<class>/<model>/<instrument>/ — the 3-level layout
# aggregate.py already understands). The PreOptimizationTimes campaign
# matrix (autolens_profiling#56): imaging at ao/jwst/hst, interferometer
# and datacube across their instrument presets.
CELLS: list[tuple[str, str, tuple[str, ...]]] = [
    ("imaging", "mge", ("hst", "jwst", "ao")),
    ("imaging", "pixelization", ("hst", "jwst", "ao")),
    ("imaging", "delaunay", ("hst", "jwst", "ao")),
    ("interferometer", "mge", ("sma", "alma", "alma_high", "jvla")),
    ("interferometer", "pixelization", ("sma", "alma", "alma_high", "jvla")),
    ("interferometer", "delaunay", ("sma", "alma", "alma_high", "jvla")),
    ("datacube", "delaunay", ("sma", "alma", "alma_high")),
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
        metavar="CLASS/MODEL",
        help="Only run these cells (e.g. interferometer/mge datacube/delaunay).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=(),
        metavar="CLASS/MODEL",
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
        "--skip-existing",
        action="store_true",
        help=(
            "Skip any (cell, config) whose result JSON already exists in the "
            "output dir — resume an interrupted campaign without redoing "
            "completed runs (the in-flight run at interruption left no JSON, "
            "so it re-runs)."
        ),
    )
    p.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Pass --sparse to every selected cell (w-tilde sparse-operator "
            "inversion path; imaging cells only — combine with --only). "
            "Result filenames gain a _sparse suffix so dense and sparse "
            "rows coexist."
        ),
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


def _expand_cells() -> list[tuple[str, str, str | None]]:
    """Flatten CELLS into (class, model, instrument-or-None) rows."""
    rows: list[tuple[str, str, str | None]] = []
    for cls, model, instruments in CELLS:
        if instruments:
            rows.extend((cls, model, inst) for inst in instruments)
        else:
            rows.append((cls, model, None))
    return rows


def _cell_id(cls: str, model: str, inst: str | None) -> str:
    return f"{cls}/{model}/{inst}" if inst else f"{cls}/{model}"


def _resolve_cells(args: argparse.Namespace) -> list[tuple[str, str, str | None]]:
    """--only/--skip match class/model (all instruments) or class/model/instrument."""
    selected = _expand_cells()

    def _matches(spec: str, row: tuple[str, str, str | None]) -> bool:
        cls, model, inst = row
        return spec in (f"{cls}/{model}", _cell_id(cls, model, inst))

    if args.only:
        selected = [r for r in selected if any(_matches(s, r) for s in args.only)]
        matched = {s for s in args.only if any(_matches(s, r) for r in selected)}
        missing = set(args.only) - matched
        if missing:
            sys.stderr.write(f"warning: --only includes unknown cells: {sorted(missing)}\n")
    selected = [r for r in selected if not any(_matches(s, r) for s in args.skip)]
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
    sparse: bool = False,
    instrument: str | None = None,
) -> tuple[bool, float, str]:
    """Run one (cell, config) pair as a subprocess. Returns (ok, elapsed, log_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_suffix = "_sparse" if sparse else ""
    log_path = out_dir / f"{config.name}{log_suffix}.log"

    cmd = [
        python,
        str(script_path),
        "--config-name",
        config.name,
        "--output-dir",
        str(out_dir),
        *(("--instrument", instrument) if instrument else ()),
        *config.extra_args,
        *(("--sparse",) if sparse else ()),
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
        print(
            f"    {'OK ' if ok else 'FAIL'} ({elapsed:.1f}s, exit={proc.returncode}) -> {log_path.name}"
        )

        # Verify the device.backend in the JSON matches expectations.
        if ok:
            import json

            json_path = out_dir / f"{config.name}{log_suffix}.json"
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text())
                    actual = data.get("device", {}).get("backend")
                    expected = "gpu" if config.is_gpu else "cpu"
                    if actual != expected:
                        print(
                            f"    WARN: device.backend={actual!r} but config expected {expected!r}; "
                            f"check {json_path.name}"
                        )
                except Exception as exc:
                    print(f"    WARN: could not validate device.backend: {exc}")
        return ok, elapsed, str(log_path)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"    INTERRUPTED after {elapsed:.1f}s; partial log -> {log_path}")
        raise


def main() -> int:
    args = _parse_args()
    cells = _resolve_cells(args)
    configs = _resolve_configs(args)

    print(
        f"sweep_likelihood: {len(cells)} cells x {len(configs)} configs "
        f"= {len(cells) * len(configs)} runs"
    )
    print(f"  cells:    {[_cell_id(c, m, i) for (c, m, i) in cells]}")
    print(f"  configs:  {[c.name for c in configs]}")
    print(f"  output:   {args.output_root}")
    print(f"  python:   {args.python}")
    if args.dry_run:
        print("  (dry-run)")

    summary: list[tuple[str, str, bool, float]] = []
    overall_t0 = time.time()

    for cls, model, inst in cells:
        script_path = _REPO_ROOT / "likelihood_runtime" / cls / f"{model}.py"
        cell_id = _cell_id(cls, model, inst)
        if not script_path.exists():
            print(f"\n!!! missing script: {script_path}")
            for cfg in configs:
                summary.append((cell_id, cfg.name, False, 0.0))
            continue

        out_dir = args.output_root / cls / model
        if inst:
            out_dir = out_dir / inst

        for cfg in configs:
            if args.skip_existing:
                suffix = "_sparse" if args.sparse else ""
                existing = out_dir / f"{model}_{cfg.name}{suffix}.json"
                if existing.exists():
                    print(f"--- [{cfg.name}] {cell_id}: SKIP (result exists)")
                    summary.append((cell_id, cfg.name, True, 0.0))
                    continue
            try:
                ok, elapsed, _log = _run_one(
                    args.python,
                    script_path,
                    cfg,
                    out_dir,
                    args.dry_run,
                    sparse=args.sparse,
                    instrument=inst,
                )
            except KeyboardInterrupt:
                print("\n\nsweep interrupted by user")
                return 130
            summary.append((cell_id, cfg.name, ok, elapsed))

    total = time.time() - overall_t0
    print("\n" + "=" * 70)
    print(f"sweep_likelihood summary  ({total:.1f}s total)")
    print("=" * 70)
    print(f"  {'cell':<32}{'config':<22}{'ok':<6}{'elapsed':>10}")
    print(f"  {'-' * 32}{'-' * 22}{'-' * 6}{'-' * 10}")
    failures = 0
    for cell, cfg, ok, t in summary:
        flag = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  {cell:<32}{cfg:<22}{flag:<6}{t:>9.1f}s")
    if failures:
        print(f"\n  {failures} run(s) FAILED — check the .log files in each cell's output dir.")
        return 1
    print("\n  All runs OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
