"""
Multi-Galaxy: rectangular JAX runtime (likelihood-lifecycle 2d, autolens_profiling#473)
=======================================================================================

``AnalysisImaging(use_jax=True)`` on the dense mapping route (setup ``jax_*_dense_fp64``) or, with
``--sparse``, the JAX w-tilde sparse operator (``jax_*_sparse_fp64``): the jitted full pipeline
(lower / compile / first call / steady block), its steady median, and the vmap per call the
catalogue reads as the JAX headline. Then ``parity.json`` (vs the eager NumPy dense reference) and
``fit.png`` beside the row.

The cell is the multi_galaxy standard cell (RectangularBilinearAdaptImage 35 x 35, free Adapt), built by
``scripts/misc/cell_builders/galaxy_group.py``; the driver is
``scripts/misc/likelihood_breakdown/galaxy_group_standard.py``. Dataset: ``dataset/multi_galaxy/hst``,
simulated on first use. Output: the mirrored cell named by ``--output-path`` (the campaign submits
always pass one), else the scratch tree ``output/p2d_local/results/...``. ``--fast`` is a local
check; ``AUTOLENS_PROFILING_SMOKE=1`` exits after the imports.

    python scripts/multi_galaxy/rectangular/likelihood_runtime.py --instrument hst --sparse --fast
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

# Stdlib only until the driver runs.
from likelihood_breakdown import galaxy_group_standard as _driver  # noqa: E402

_driver.main("multi_galaxy", "rectangular", "runtime", "jax", script_file=__file__)
