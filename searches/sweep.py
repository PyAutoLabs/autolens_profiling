"""Multi-config first-class search profiling driver.

Runs each in-scope cell across the CPU/GPU × fp64/mp matrix (4 configs per
cell locally; HPC A100 configs are dispatched separately via the same
external mechanism used by ``likelihood_runtime/sweep.py``).

Each cell is a ``(sampler, dataset_class, model, instrument)`` quadruple.
Per-config JSONs land at::

    <output_root>/<sampler>/<dataset_class>/<model>/<instrument>/<config_name>.json
    <output_root>/<sampler>/<dataset_class>/<model>/<instrument>/<config_name>.png
    <output_root>/<sampler>/<dataset_class>/<model>/<instrument>/<config_name>.log

Resume-by-default: if the per-config JSON already exists, the cell is
skipped. Pass ``--force`` to re-run.

Usage::

    # All in-scope cells × instruments × configs (warning: long)
    python searches/sweep.py

    # One cell, one instrument, CPU only (fast iteration)
    python searches/sweep.py \\
        --only nautilus/imaging/mge --instrument hst --skip-gpu --skip-mp

    # Force re-run of one cell (bypass resume)
    python searches/sweep.py --only nautilus/imaging/mge --instrument hst --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # autolens_profiling/
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "searches"
# PyAutoFit's autoconf ``output_path`` defaults to ``<cwd>/output``. The
# searches package writes search state under ``<output_path>/searches/...``
# via the ``path_prefix`` set in ``_samplers.build_nautilus``. Wiping this
# subtree before a (cell, config) run is what gives honest timing — see the
# ``--keep-completed`` flag.
_DEFAULT_SEARCH_OUTPUT_ROOT = _REPO_ROOT / "output" / "searches"
_DEFAULT_PYTHON = sys.executable


# Per-(sampler, dataset_class, model) the canonical instrument set defaults
# come from the instrument dicts. If the user passes --instrument, that wins.
_INSTRUMENT_SETS: dict[str, tuple[str, ...]] = {
    "imaging": ("hst", "euclid", "jwst", "ao"),
    "interferometer": ("sma", "alma", "alma_high", "jvla"),
    "point_source": ("simple",),
    "datacube": ("sma",),
}


# (sampler, dataset_class, model). Order is roughly cheapest -> heaviest so
# failures surface quickly during iteration.
CELLS: list[tuple[str, str, str]] = [
    ("nautilus", "point_source", "image_plane"),
    ("nautilus", "point_source", "source_plane"),
    ("nautilus", "imaging", "mge"),
    ("nautilus", "imaging", "pixelization"),
    ("nautilus", "imaging", "delaunay"),
    ("nautilus", "interferometer", "mge"),
    ("nautilus", "interferometer", "pixelization"),
    ("nautilus", "interferometer", "delaunay"),
    ("nautilus", "datacube", "delaunay"),
    ("nss", "point_source", "image_plane"),
    ("nss", "point_source", "source_plane"),
    ("nss", "imaging", "mge"),
    ("nss", "imaging", "pixelization"),
    ("nss", "imaging", "delaunay"),
    ("nss", "interferometer", "mge"),
    ("nss", "interferometer", "pixelization"),
    ("nss", "interferometer", "delaunay"),
    ("nss", "datacube", "delaunay"),
]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    env_overrides: dict[str, str]
    extra_args: tuple[str, ...]
    is_gpu: bool


# CPU configs explicitly pin platform to cpu. GPU configs explicitly pin to
# cuda so a missing CUDA device fails loudly rather than silently producing
# a CPU number.
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
        metavar="SAMPLER/CLASS/MODEL",
        help="Only run these cells (e.g. nautilus/imaging/mge).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=(),
        metavar="SAMPLER/CLASS/MODEL",
        help="Skip these cells (applied after --only).",
    )
    p.add_argument(
        "--sampler",
        nargs="+",
        default=None,
        help="Restrict to these samplers (e.g. --sampler nautilus).",
    )
    p.add_argument(
        "--dataset-class",
        nargs="+",
        default=None,
        help="Restrict to these dataset classes (imaging / interferometer / ...).",
    )
    p.add_argument(
        "--instrument",
        nargs="+",
        default=None,
        help=(
            "Restrict to these instruments. Default: every instrument valid for "
            "the dataset class of each cell."
        ),
    )
    p.add_argument("--skip-cpu", action="store_true", help="Skip local_cpu_* configs.")
    p.add_argument("--skip-gpu", action="store_true", help="Skip local_gpu_* configs.")
    p.add_argument(
        "--skip-mp",
        action="store_true",
        help="Skip the use_mixed_precision rows (just fp64).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run cells whose per-config JSON already exists. "
            "Default behaviour is to resume (skip on JSON present)."
        ),
    )
    p.add_argument(
        "--keep-completed",
        action="store_true",
        help=(
            "Do NOT wipe the search-output dir (output/searches/...) before "
            "running each cell. By default sweep.py removes any "
            "``.completed`` sentinel + cached ``samples.csv`` + Nautilus "
            "checkpoint left by a prior run so the new run is a fresh "
            "fit. Use this flag to deliberately resume cached samples "
            "(e.g. for debugging the post-fit visualization path)."
        ),
    )
    p.add_argument(
        "--search-output-root",
        type=Path,
        default=_DEFAULT_SEARCH_OUTPUT_ROOT,
        help=(
            f"Where PyAutoFit writes its own per-search output (samples.csv, "
            f"search.summary, visualization). Default: "
            f"{_DEFAULT_SEARCH_OUTPUT_ROOT}. Must match the autoconf "
            f"``output_path`` + ``searches/`` prefix used by the leaf scripts."
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


def _resolve_cells(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    """Expand CELLS into per-instrument (sampler, ds_class, model, instrument) tuples."""
    selected = list(CELLS)
    if args.only:
        wanted = set(args.only)
        selected = [c for c in selected if f"{c[0]}/{c[1]}/{c[2]}" in wanted]
        missing = wanted - {f"{c[0]}/{c[1]}/{c[2]}" for c in selected}
        if missing:
            sys.stderr.write(f"warning: --only includes unknown cells: {sorted(missing)}\n")
    skip = set(args.skip)
    selected = [c for c in selected if f"{c[0]}/{c[1]}/{c[2]}" not in skip]
    if args.sampler:
        wanted_samplers = set(args.sampler)
        selected = [c for c in selected if c[0] in wanted_samplers]
    if args.dataset_class:
        wanted_ds = set(args.dataset_class)
        selected = [c for c in selected if c[1] in wanted_ds]

    instrument_filter = set(args.instrument) if args.instrument else None
    expanded: list[tuple[str, str, str, str]] = []
    for sampler, ds_class, model in selected:
        for instrument in _INSTRUMENT_SETS.get(ds_class, ()):
            if instrument_filter and instrument not in instrument_filter:
                continue
            expanded.append((sampler, ds_class, model, instrument))
    return expanded


def _resolve_configs(args: argparse.Namespace) -> list[SweepConfig]:
    configs = list(CONFIGS)
    if args.skip_cpu:
        configs = [c for c in configs if c.is_gpu]
    if args.skip_gpu:
        configs = [c for c in configs if not c.is_gpu]
    if args.skip_mp:
        configs = [c for c in configs if "--use-mixed-precision" not in c.extra_args]
    return configs


def _script_path(sampler: str, ds_class: str, model: str) -> Path:
    return _REPO_ROOT / "searches" / sampler / ds_class / f"{model}.py"


def _wipe_search_state(
    *,
    search_output_root: Path,
    sampler: str,
    ds_class: str,
    model: str,
    instrument: str,
    config_name: str,
    dry_run: bool,
) -> None:
    """Remove PyAutoFit's per-search output dir for one (cell, config).

    PyAutoFit gates fresh-vs-resume on the ``.completed`` sentinel inside the
    search's ``path_prefix/name`` directory. Without wiping, a re-run after a
    prior successful sampling will load the cached ``samples.csv`` + Nautilus
    pickle and report bogus 2-3x speedups (no real sampling fires). See PR #30
    follow-up for the diagnosis.
    """
    cell_root = search_output_root / sampler / ds_class / model / instrument / config_name
    if not cell_root.exists():
        return
    if dry_run:
        print(f"    [clear-completed] (dry-run) would remove {cell_root}")
        return
    shutil.rmtree(cell_root)
    try:
        display = cell_root.relative_to(_REPO_ROOT)
    except ValueError:
        display = cell_root
    print(f"    [clear-completed] removed {display} (set --keep-completed to suppress)")


def _run_one(
    *,
    python: str,
    script_path: Path,
    config: SweepConfig,
    sampler: str,
    ds_class: str,
    model: str,
    instrument: str,
    out_dir: Path,
    search_output_root: Path,
    keep_completed: bool,
    dry_run: bool,
    force: bool,
) -> tuple[bool, float, str]:
    """Run one (cell, instrument, config) triple as a subprocess.

    Returns (ok, elapsed, log_path). Resume-by-default: ``ok=True`` with
    ``elapsed=0`` is returned when the JSON already exists and ``--force``
    is not set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{config.name}.json"
    log_path = out_dir / f"{config.name}.log"

    if json_path.exists() and not force:
        print(f"    SKIP: {json_path.name} exists (use --force to re-run)")
        return True, 0.0, ""

    if not keep_completed:
        _wipe_search_state(
            search_output_root=search_output_root,
            sampler=sampler,
            ds_class=ds_class,
            model=model,
            instrument=instrument,
            config_name=config.name,
            dry_run=dry_run,
        )

    cmd = [
        python,
        str(script_path),
        "--config-name",
        config.name,
        "--output-dir",
        str(out_dir),
        "--instrument",
        instrument,
        *config.extra_args,
    ]

    env = dict(os.environ)
    env.update(config.env_overrides)
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    print(f"\n--- [{config.name}] {script_path.relative_to(_REPO_ROOT)} [{instrument}] ---")
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
            f"    {'OK ' if ok else 'FAIL'} ({elapsed:.1f}s, exit={proc.returncode})"
            f" -> {log_path.name}"
        )
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
        f"sweep_searches: {len(cells)} (cell,instrument) × {len(configs)} configs "
        f"= {len(cells) * len(configs)} runs"
    )
    print(f"  cells:    {[f'{s}/{c}/{m}/{i}' for (s, c, m, i) in cells]}")
    print(f"  configs:  {[c.name for c in configs]}")
    print(f"  output:   {args.output_root}")
    print(f"  python:   {args.python}")
    print(f"  resume:   {'OFF (--force)' if args.force else 'ON (default)'}")
    print(
        f"  .completed wipe: {'OFF (--keep-completed)' if args.keep_completed else 'ON (default)'}"
    )
    if args.dry_run:
        print("  (dry-run)")

    summary: list[tuple[str, str, bool, float]] = []
    overall_t0 = time.time()

    for sampler, ds_class, model, instrument in cells:
        script_path = _script_path(sampler, ds_class, model)
        cell_id = f"{sampler}/{ds_class}/{model}/{instrument}"
        if not script_path.exists():
            print(f"\n!!! missing script: {script_path}")
            for cfg in configs:
                summary.append((cell_id, cfg.name, False, 0.0))
            continue

        out_dir = args.output_root / sampler / ds_class / model / instrument

        for cfg in configs:
            try:
                ok, elapsed, _log = _run_one(
                    python=args.python,
                    script_path=script_path,
                    config=cfg,
                    sampler=sampler,
                    ds_class=ds_class,
                    model=model,
                    instrument=instrument,
                    out_dir=out_dir,
                    search_output_root=args.search_output_root,
                    keep_completed=args.keep_completed,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            except KeyboardInterrupt:
                print("\n\nsweep interrupted by user")
                return 130
            summary.append((cell_id, cfg.name, ok, elapsed))

    total = time.time() - overall_t0
    print("\n" + "=" * 80)
    print(f"sweep_searches summary  ({total:.1f}s total)")
    print("=" * 80)
    print(f"  {'cell':<46}{'config':<22}{'ok':<6}{'elapsed':>10}")
    print(f"  {'-' * 46}{'-' * 22}{'-' * 6}{'-' * 10}")
    failures = 0
    for cell, cfg, ok, t in summary:
        flag = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  {cell:<46}{cfg:<22}{flag:<6}{t:>9.1f}s")
    if failures:
        print(f"\n  {failures} run(s) FAILED — check the .log files in each cell's output dir.")
        return 1
    print("\n  All runs OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
