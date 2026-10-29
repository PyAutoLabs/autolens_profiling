"""
Group: delaunay numba CPU runtime (likelihood-lifecycle 2d, autolens_profiling#473)
===================================================================================

``AnalysisImaging(use_jax=False)`` on the numba CPU sparse operator (setup
``numba_cpu_sparse_fp64``), threads pinned to 1, over the seeded iid stream: instance 0 discarded
(numba compile), ``--cold-evals`` cold, the rest warm; the headline is the warm iid median. Then
``parity.json`` and ``fit.png`` beside the row.

The cell is the group standard cell (Delaunay 1250 Hilbert vertices + 30 zeroed edge points, free AdaptSplit), built by
``scripts/misc/cell_builders/galaxy_group.py``; the driver is
``scripts/misc/likelihood_breakdown/galaxy_group_standard.py``. Dataset: ``dataset/group/euclid``,
simulated on first use. Output: the mirrored cell named by ``--output-path`` (the campaign submits
always pass one), else the scratch tree ``output/p2d_local/results/...``. ``--fast`` is a local
check; ``AUTOLENS_PROFILING_SMOKE=1`` exits after the imports.

    python scripts/group/delaunay/likelihood_runtime_numba.py --instrument euclid --fast
"""

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
_sys.path.insert(0, str(_profiling_root()))

# Threads are pinned BEFORE numpy (and numba) is imported: OpenBLAS / MKL read their thread
# counts once, when the library loads. _production_config is stdlib-only at import.
from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402

_pin_thread_env(1)

# Stdlib only until the driver runs (numba: threads pinned above).
from likelihood_breakdown import galaxy_group_standard as _driver  # noqa: E402

_driver.main("group", "delaunay", "runtime", "numba", script_file=__file__)
