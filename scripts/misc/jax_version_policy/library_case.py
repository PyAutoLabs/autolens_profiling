"""
One hang-matrix cell through the real PyAuto library: a jitted 2d galaxy/group likelihood
called N times, with the same line protocol as :mod:`hang_case` (HEADER / COMPILED / CALL /
DONE). Used for the cells ``jax.export`` cannot serialize (Delaunay's host callback).

Run by :mod:`run_matrix` under the PyAuto environment; the jax version under test is put ahead
of the venv's own by a ``pip install --target`` directory on ``PYTHONPATH`` (the venv itself is
never changed). Issue PyAutoLabs/PyAutoNerves#213.

    python library_case.py --cell group/delaunay --calls 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _p in (ROOT, ROOT / "scripts" / "misc", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main():
    p = argparse.ArgumentParser(description="one library hang-matrix cell")
    p.add_argument("--cell", required=True, help="<dataset_type>/<variant>, e.g. group/delaunay")
    p.add_argument("--calls", type=int, default=30)
    p.add_argument(
        "--stage", type=int, default=7, help="breakdown prefix stage (7 = full likelihood)"
    )
    a = p.parse_args()
    dataset_type, variant = a.cell.split("/")

    import jax
    from autofit.jax import register_model
    from hang_case import _header
    from likelihood_breakdown import galaxy_group_standard as gs

    from _profile_cli import parse_profile_cli

    head = _header(f"library:{a.cell}", None)
    head["jax_file"] = os.path.dirname(jax.__file__).rsplit(os.sep, 2)[-2]  # venv vs shadow
    print("HEADER " + json.dumps(head), flush=True)

    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        cli = parse_profile_cli()
    finally:
        sys.argv = argv
    cell = gs._jax_cell(dataset_type, variant, cli, ROOT)
    instance = cell["instances"][0]
    register_model(cell["model"])
    params, treedef = gs._flat_params(instance)
    analysis = gs._analysis(cell, instance, use_jax=True)

    def prefix(params):
        fit = analysis.fit_from(instance=jax.tree_util.tree_unflatten(treedef, params))
        return gs._stage_value(fit, a.stage)

    t0 = time.perf_counter()
    compiled = jax.jit(prefix).lower(params).compile()
    print(f"COMPILED {time.perf_counter() - t0:.4f}", flush=True)
    for i in range(a.calls):
        t = time.perf_counter()
        jax.block_until_ready(compiled(params))
        print(f"CALL {i} {time.perf_counter() - t:.6f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
