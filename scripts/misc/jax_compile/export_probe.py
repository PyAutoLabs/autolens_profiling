"""
Can ``jax.export`` beat the tracing floor? (final compile-time census, issue #77)

After the cache (#128) and autotune-off (#132) defaults, the residual warm-run
cost is Python *tracing* (~5-17 s per transform per process), which the
compilation cache cannot remove. ``jax.export`` serializes the traced/lowered
StableHLO to disk, so a repeat process could deserialize instead of retracing —
this probe measures whether that wins.

Two modes, run as separate processes so nothing is warm in-process::

    python jax_compile/export_probe.py --mode save   # trace + export + serialize to disk
    python jax_compile/export_probe.py --mode load   # deserialize + first call + steady

Compare `load`'s deserialize+first against the standard warm path's
trace+compile+first (probe.py census-warm rows). Uses the MGE ``vag`` cell.
Appends records to ``jax_compile/results/<hardware>/export_probe.json``.
"""

from __future__ import annotations

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


import argparse
import json
import sys
import time
from pathlib import Path

_WORKSPACE_ROOT = _profiling_root()
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from jax_compile.probe import build_objective, hardware_label

EXPORT_PATH = Path("/tmp/claude-1000/census_export_mge_vag.bin")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--mode", choices=("save", "load"), required=True)
    p.add_argument("--dataset-class", default="imaging")
    p.add_argument("--model-type", default="mge")
    p.add_argument("--instrument", default="hst")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()

    import jax
    from jax import export as jax_export

    record = {"mode": args.mode, "model_type": args.model_type, "transform": "vag"}

    if args.mode == "save":
        f, x0, ndim = build_objective(args)
        vag = jax.value_and_grad(f)

        t0 = time.perf_counter()
        exported = jax_export.export(jax.jit(vag))(jax.ShapeDtypeStruct(x0.shape, x0.dtype))
        record["trace_export_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        blob = exported.serialize()
        EXPORT_PATH.write_bytes(blob)
        record["serialize_s"] = round(time.perf_counter() - t0, 3)
        record["blob_mb"] = round(len(blob) / 1e6, 3)
        record["ndim"] = ndim
        print(
            f"[export_probe] save: trace+export {record['trace_export_s']}s  "
            f"serialize {record['serialize_s']}s  blob {record['blob_mb']}MB"
        )
    else:
        # The load path deliberately does NOT build the objective — the whole
        # point is skipping the model/dataset tracing. It only needs an input
        # array of the right shape.
        import jax.numpy as jnp

        t0 = time.perf_counter()
        exported = jax_export.deserialize(EXPORT_PATH.read_bytes())
        record["deserialize_s"] = round(time.perf_counter() - t0, 3)

        in_shape = exported.in_avals[0].shape
        x = jnp.full(in_shape, 0.5, dtype=exported.in_avals[0].dtype)

        t0 = time.perf_counter()
        out = exported.call(x)
        jax.block_until_ready(out)
        record["first_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        for _ in range(3):
            out = exported.call(x)
        jax.block_until_ready(out)
        record["steady_s"] = round((time.perf_counter() - t0) / 3, 4)
        print(
            f"[export_probe] load: deserialize {record['deserialize_s']}s  "
            f"first {record['first_s']}s  steady {record['steady_s']}s"
        )

    record["jax_version"] = jax.__version__
    record["tag"] = args.tag
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_dir = _WORKSPACE_ROOT / "scripts" / "misc" / "jax_compile" / "results" / hardware_label(jax)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "export_probe.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append(record)
    out_path.write_text(json.dumps(existing, indent=2) + "\n")


if __name__ == "__main__":
    main()
