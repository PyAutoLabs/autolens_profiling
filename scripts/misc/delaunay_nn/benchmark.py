"""Profile the full JAX DelaunayNN mapper across fixed-shape caps.

This is a dataset-agnostic microbenchmark for the geometry that distinguishes
``DelaunayNN`` from barycentric ``Delaunay``. It includes data-query weights,
dual areas, split points and split-regularization weights, so the warm timing
is the cost paid by the mapper rather than an isolated Watson-weight kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _profiling_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import autolens as al  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from autoarray.inversion.mesh.interpolator.delaunay import jax_delaunay  # noqa: E402
from autoarray.inversion.mesh.interpolator.sibson import jax_delaunay_nn  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-points", type=int, default=1200)
    parser.add_argument("--queries", type=int, default=15000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--caps", type=int, nargs="+", default=[16, 24, 32, 64])
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def adaptive_mesh(count, rng):
    """Production-shaped central source concentration plus boundary coverage."""
    blob_count = count // 2
    blob = rng.normal(size=(blob_count, 2)) * 0.15
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count - blob_count)
    radius = 1.0 + rng.normal(size=count - blob_count) * 0.12
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return np.concatenate([blob, ring])


def measure(function, points, queries, repeats):
    jitted = jax.jit(function)

    start = time.perf_counter()
    output = jitted(points, queries)
    jax.block_until_ready(output)
    compile_and_first = time.perf_counter() - start

    warm = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = jitted(points, queries)
        jax.block_until_ready(output)
        warm.append(time.perf_counter() - start)

    return output, compile_and_first, float(np.median(warm)), warm


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


args = parse_args()
rng = np.random.default_rng(4)
points = jnp.asarray(adaptive_mesh(args.mesh_points, rng))
queries = jnp.asarray(adaptive_mesh(args.queries, rng))

baseline_output, baseline_compile, baseline_warm, baseline_runs = measure(
    lambda mesh_points, query_points: jax_delaunay(mesh_points, query_points),
    points,
    queries,
    args.repeats,
)

rows = [
    {
        "name": "Delaunay",
        "cap": None,
        "compile_and_first_seconds": baseline_compile,
        "warm_median_seconds": baseline_warm,
        "warm_seconds": baseline_runs,
        "ratio_to_delaunay": 1.0,
    }
]

for cap in args.caps:
    output, compile_and_first, warm_median, warm_runs = measure(
        lambda mesh_points, query_points, fixed_cap=cap: jax_delaunay_nn(
            mesh_points,
            query_points,
            max_cavity_triangles=fixed_cap,
            max_neighbors=fixed_cap,
            query_chunk=args.query_chunk,
        ),
        points,
        queries,
        args.repeats,
    )

    main_overflow = int(np.asarray(output[10]).sum())
    split_overflow = int(np.asarray(output[13]).sum())
    main_degenerate = int(np.asarray(output[11]).sum())
    split_degenerate = int(np.asarray(output[14]).sum())

    rows.append(
        {
            "name": f"DelaunayNN cap {cap}",
            "cap": cap,
            "compile_and_first_seconds": compile_and_first,
            "warm_median_seconds": warm_median,
            "warm_seconds": warm_runs,
            "ratio_to_delaunay": warm_median / baseline_warm,
            "main_max_neighbors": int(np.asarray(output[3]).max()),
            "main_max_cavity": int(np.asarray(output[9]).max()),
            "split_max_neighbors": int(np.asarray(output[7]).max()),
            "split_max_cavity": int(np.asarray(output[12]).max()),
            "main_overflow_rows": main_overflow,
            "split_overflow_rows": split_overflow,
            "main_degenerate_rows": main_degenerate,
            "split_degenerate_rows": split_degenerate,
        }
    )

    if cap >= 32:
        assert main_overflow == 0
        assert split_overflow == 0
        assert main_degenerate == 0
        assert split_degenerate == 0

device = jax.devices()[0]
device_label = getattr(device, "device_kind", str(device))
backend = jax.default_backend()
hardware = backend if backend == "cpu" else f"{backend}_{slug(device_label)}"
version = getattr(al, "__version__", "unknown")
output_dir = args.output_dir or ROOT / "results" / "misc" / "delaunay_nn"
output_dir.mkdir(parents=True, exist_ok=True)
stem = f"delaunay_nn_benchmark_{hardware}_v{version}"
json_path = output_dir / f"{stem}.json"
png_path = output_dir / f"{stem}.png"

payload = {
    "pyautolens_version": version,
    "backend": backend,
    "hardware": hardware,
    "device": device_label,
    "mesh_points": args.mesh_points,
    "queries": args.queries,
    "query_chunk": args.query_chunk,
    "repeats": args.repeats,
    "rows": rows,
}
json_path.write_text(json.dumps(payload, indent=2) + "\n")

labels = [row["name"] for row in rows]
warm_times = [row["warm_median_seconds"] for row in rows]
fig, axis = plt.subplots(figsize=(10, 5))
bars = axis.bar(labels, warm_times, color=["#808080"] + ["#3B82F6"] * (len(rows) - 1))
axis.set_ylabel("Warm full-mapper time (seconds)")
axis.set_title(
    f"DelaunayNN fixed-shape cap cost\n{args.mesh_points} vertices × {args.queries} queries"
)
axis.tick_params(axis="x", rotation=25)
for bar, value in zip(bars, warm_times):
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}s",
        ha="center",
        va="bottom",
    )
fig.tight_layout()
fig.savefig(png_path, dpi=160)
plt.close(fig)

print(f"device: {backend} / {device_label}")
print(f"shape: {args.mesh_points} mesh points x {args.queries} queries")
for row in rows:
    print(
        f"{row['name']}: warm={row['warm_median_seconds']:.6f}s, "
        f"compile+first={row['compile_and_first_seconds']:.6f}s, "
        f"ratio={row['ratio_to_delaunay']:.3f}x"
    )
print(f"wrote {json_path}")
print(f"wrote {png_path}")
