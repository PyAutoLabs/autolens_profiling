"""Where a sweep row and a cell's comparison live: one rule for ``sweep.py`` and ``aggregate.py``.

A sweep row is named, as before the day-zero move (autolens_profiling#442), under the legacy
``results/runtime/<class>/<model>/<instrument>/<model>_<config>[_sparse].json``. The sweep
does not let the leaf script guess where that lands: it places the row itself
(:func:`placed_row`, the day-zero map's classification fed the setup facts the sweep matrix
fixes -- instrument, precision and, for the dense interferometric routes, the transformer) and
passes the exact path as ``--output-path``. ``aggregate.py`` asks the same function, so writer
and reader agree by construction; every (class, model, instrument, config, sparse) of the
matrix has its own path (``test_sweep_layout.py``).

With a custom ``--output-root`` (not under ``results/runtime``) nothing is placed: rows keep
their legacy names in that folder, as they always did.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from _profile_cli import legacy_landing_dir, placed_path  # noqa: E402

#: The legacy root sweep rows are named under.
LEGACY_RUNTIME_ROOT = REPO_ROOT / "results" / "runtime"
#: Dataset classes whose dense route needs a transformer to name its setup.
INTERFEROMETRIC = ("interferometer", "datacube")
#: Every sweep leaf runs its dense interferometric route through TransformerNUFFT (the sweep
#: never passes ``--use-dft``); sparse rows are the W~ route whatever the transformer.
SWEEP_TRANSFORMER = "TransformerNUFFT"

# (dataset_class, model, instruments). Order is roughly cheapest -> heaviest
# so failures surface quickly during iteration. An empty instrument tuple
# means "the per-cell script's module default" (pre-campaign behaviour);
# named instruments run once each, with per-instrument output subdirs
# (<output_root>/<class>/<model>/<instrument>/ — the 3-level layout
# aggregate.py already understands). The PreOptimizationTimes campaign
# matrix (autolens_profiling#56): imaging at ao/jwst/hst, interferometer
# and datacube across their instrument presets.
CELLS: list[tuple[str, str, tuple[str, ...]]] = [
    ("imaging", "mge", ("hst", "jwst", "ao")),
    ("imaging", "pixelization", ("hst", "jwst", "ao")),
    ("imaging", "delaunay", ("hst", "jwst", "ao")),
    # DelaunayNN mesh-family comparison against imaging/delaunay — hst only,
    # the instrument the two cells are compared at (autolens_profiling#219).
    ("imaging", "delaunay_nn", ("hst",)),
    # mge interferometer runs sma ONLY: the dense MGE mapping matrix needs a
    # ~62 GB gather buffer at 1M+ vis (vram/config.py marks alma+ INHERENTLY
    # blocked) — confirmed empirically by the first campaign pass (#56):
    # alma OOMed at exactly 62,720,000,000 bytes on any backend.
    ("interferometer", "mge", ("sma",)),
    ("interferometer", "pixelization", ("sma", "alma", "alma_high", "jvla")),
    ("interferometer", "delaunay", ("sma", "alma", "alma_high", "jvla")),
    ("datacube", "delaunay", ("sma", "alma", "alma_high")),
]

#: Every config label a sweep row can carry (sweep.CONFIGS and the HPC tiers).
CONFIG_NAMES = (
    "local_cpu_fp64",
    "local_cpu_mp",
    "local_gpu_fp64",
    "local_gpu_mp",
    "hpc_ral_cpu_fp64",
    "hpc_a100_fp64",
    "hpc_a100_mp",
)


def is_placed(root: Path) -> bool:
    """True when ``root`` is (under) the legacy runtime root: rows are placed, not kept there."""
    return legacy_landing_dir(Path(root)) != Path(root)


def row_facts(cls: str, instrument: str | None, config_name: str, sparse: bool) -> dict:
    """The setup facts the sweep matrix fixes for one row, in the result JSON's own keys."""
    doc = {"use_mixed_precision": config_name.endswith("_mp")}
    if instrument:
        doc["instrument"] = instrument
    if cls in INTERFEROMETRIC and not sparse:
        doc["transformer"] = SWEEP_TRANSFORMER
    return doc


def row_name(model: str, config_name: str, sparse: bool) -> str:
    return f"{model}_{config_name}{'_sparse' if sparse else ''}.json"


def placed_row(
    cls: str,
    model: str,
    instrument: str | None,
    config_name: str,
    sparse: bool,
    root: Path = LEGACY_RUNTIME_ROOT,
) -> Path:
    """Where the sweep writes this row (and where ``aggregate.py`` reads it)."""
    cell = Path(root) / cls / model
    if instrument:
        cell = cell / instrument
    legacy = cell / row_name(model, config_name, sparse)
    if not is_placed(Path(root)):
        return legacy
    return placed_path(legacy, row_facts(cls, instrument, config_name, sparse))


def placed_comparison(cell_dir: Path, instrument: str | None = None) -> Path:
    """Where a cell's ``comparison.json`` goes: ``results/<dataset_type>/<variant>/<instrument>/``.

    A two-level legacy cell (``imaging/delaunay_nn``) is named under the instrument its rows
    record (``instrument``), so it lands in its mirrored cell rather than the landing folder.
    """
    cell_dir = Path(cell_dir)
    if not is_placed(cell_dir):
        return cell_dir / "comparison.json"
    rel = cell_dir.relative_to(LEGACY_RUNTIME_ROOT).parts
    if len(rel) == 2 and instrument:
        cell_dir = cell_dir / instrument
    return placed_path(
        cell_dir / "comparison.json", {"instrument": instrument} if instrument else {}
    )


@lru_cache(maxsize=1)
def _map_claims() -> dict[str, str]:
    """{current path: sha256} of every file the day-zero map moved (committed rows)."""
    try:
        doc = json.loads((REPO_ROOT / "catalogue" / "migration_day_zero.json").read_text())
    except (OSError, ValueError):
        return {}
    return {e["new"]: e["sha256"] for e in doc.get("entries", []) if e["new"] != e["old"]}


def is_fresh(path: Path) -> bool:
    """True when ``path`` holds a row a sweep wrote after the move.

    A slot the map filled with a committed row of another legacy name (e.g. a versioned
    summary moved into the cell) only counts once a run has rewritten it (its bytes differ
    from the map's sha256).
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return True
    sha = _map_claims().get(rel)
    return sha is None or hashlib.sha256(path.read_bytes()).hexdigest() != sha
