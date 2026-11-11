"""
Export real PyAuto likelihoods as ``jax.export`` modules for the hang matrix.

Builds the likelihood-lifecycle 2d galaxy/group cells (the ones that deadlocked on jaxlib
0.11.2, PyAutoPulse ``knowledge/xla_cpu_deadlock_jaxlib_0_11_2.md``), jits a prefix of the
breakdown at iid draw 0 on the flat pytree leaves, and pickles ``exp.serialize()`` with the
leaves as NumPy arrays. :mod:`hang_case` then replays each in a process that imports only jax,
under any jax version that can deserialize it.

    python export_reproducers.py                       # all modules in MODULES
    python export_reproducers.py --only group_mge_p3

Writes ``output/jax_version_policy/modules/<name>.pkl`` (gitignored; ~MBs each, never
committed). A cell that will not export (e.g. a host callback) is reported and skipped.
Run in the normal PyAuto environment (PyAuto313 + activate.sh); it traces, it does not run
the likelihood. Issue PyAutoLabs/PyAutoNerves#213.
"""

import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
for _p in (ROOT, ROOT / "scripts" / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: name -> (dataset_type, variant, breakdown prefix stage). Prefix 7 is the full likelihood.
MODULES = {
    "group_mge_p3": ("group", "mge", 3),
    "group_mge_full": ("group", "mge", 7),
    "group_delaunay_p4": ("group", "delaunay", 4),
    "multi_galaxy_mge_full": ("multi_galaxy", "mge", 7),
    "multi_galaxy_rectangular_full": ("multi_galaxy", "rectangular", 7),
}


def export_one(name, dataset_type, variant, stage, out_dir):
    import jax
    import numpy as np
    from autofit.jax import register_model
    from likelihood_breakdown import galaxy_group_standard as gs

    from _profile_cli import parse_profile_cli

    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        cli = parse_profile_cli()  # the driver's defaults: dense, memo as the campaign ran it
    finally:
        sys.argv = argv
    cell = gs._jax_cell(dataset_type, variant, cli, ROOT)
    instance = cell["instances"][0]
    register_model(cell["model"])
    params, treedef = gs._flat_params(instance)
    analysis = gs._analysis(cell, instance, use_jax=True)

    def prefix(params):
        fit = analysis.fit_from(instance=jax.tree_util.tree_unflatten(treedef, params))
        return gs._stage_value(fit, stage)

    exp = jax.export.export(jax.jit(prefix))(params)
    path = out_dir / f"{name}.pkl"
    with path.open("wb") as fh:
        pickle.dump(
            {
                "exp": exp.serialize(),
                "params": [np.asarray(p) for p in params],
                "meta": {
                    "name": name,
                    "dataset_type": dataset_type,
                    "variant": variant,
                    "prefix_stage": stage,
                    "exported_with_jax": jax.__version__,
                    "serialization_version": getattr(exp, "calling_convention_version", None),
                },
            },
            fh,
        )
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--only", action="append", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "output" / "jax_version_policy" / "modules"))
    a = p.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, spec in MODULES.items():
        if a.only and name not in a.only:
            continue
        try:
            path = export_one(name, *spec, out_dir)
            print(f"exported {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"SKIP {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            traceback.print_exc(limit=3)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
