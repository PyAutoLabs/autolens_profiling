"""
Gradient cost and correctness: imaging/delaunay
===============================================

Likelihood lifecycle phase 6 (autolens_profiling#507). Runs ``likelihood_runtime.py`` with ``--gradient``:
the runtime script builds its own fused likelihood (same dataset, size, model and instance as its
runtime row) and hands it to ``scripts/misc/gradient/harness.py`` instead of timing it. The
harness times the forward call and ``jax.value_and_grad``, runs the pre-registered
finite-difference check and writes the cell's ``gradient.json``
(``scripts/misc/tooling/gradient_witness.py``). Every flag is passed through, so a campaign
submit gives this script the same flags as the cell's runtime row (plus its ``--output-path``).
fp64 only.
"""

import runpy
import sys
from pathlib import Path

_RUNTIME = Path(__file__).resolve().with_name("likelihood_runtime.py")

if __name__ == "__main__":
    if "--gradient" not in sys.argv:
        sys.argv.append("--gradient")
    sys.argv[0] = str(_RUNTIME)
    runpy.run_path(str(_RUNTIME), run_name="__main__")
