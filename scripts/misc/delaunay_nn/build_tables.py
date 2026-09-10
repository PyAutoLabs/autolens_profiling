"""Build the real (S, K) DelaunayNN ConstantSplit stencil tables from a mesh point set.

Runs the NumPy Sibson path (``scipy_delaunay_nn``) on the source-plane mesh
points, then applies ``InterpolatorDelaunayNN._mappings_sizes_weights_split``'s
spare-column hstack, giving the exact (4P, 33) tables ConstantSplit consumes.

The checked-in tables used by ``assembly_bench.py`` were built from the HST
breakdown cell's 1,500-vertex source-plane mesh, run from the autolens_profiling
root:

```bash
python scripts/misc/delaunay_nn/extract_mesh_points.py \
    --out results/delaunay_nn/mesh_points_hst.npy
python scripts/misc/delaunay_nn/build_tables.py \
    --points results/delaunay_nn/mesh_points_hst.npy \
    --out results/delaunay_nn/tables_hst_1500.npz
```
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--points", required=True)
parser.add_argument("--out", required=True)
parser.add_argument(
    "--pixels", type=int, default=0, help="subsample the point set to this many pixels"
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

from autoarray.inversion.mesh.interpolator.sibson import scipy_delaunay_nn

pts = np.load(args.points).astype(np.float64)
if args.pixels and args.pixels < pts.shape[0]:
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(pts.shape[0], size=args.pixels, replace=False))
    pts = pts[idx]
print("points:", pts.shape)

# The data-grid query set is irrelevant to the split tables; keep it tiny.
query = pts[:4].copy()

out = scipy_delaunay_nn(points_np=pts, query_points_np=query, areas_factor=0.5)
(
    points,
    simplices_padded,
    mappings,
    sizes,
    weights,
    split_points,
    splitted_mappings,
    splitted_sizes,
    splitted_weights,
    cavity_sizes,
    overflow,
    degenerate,
    split_cavity_sizes,
    split_overflow,
    split_degenerate,
) = out

print("split tables:", splitted_mappings.shape, splitted_weights.shape)
print("split overflow:", int(np.sum(split_overflow)), "degenerate:", int(np.sum(split_degenerate)))
print("nan weights rows:", int(np.sum(np.isnan(splitted_weights).any(axis=1))))
print(
    f"split sizes: min {splitted_sizes.min()} max {splitted_sizes.max()} "
    f"mean {splitted_sizes.mean():.2f}"
)

# InterpolatorDelaunayNN._mappings_sizes_weights_split: one spare padded column.
row_count = splitted_mappings.shape[0]
m = np.hstack([splitted_mappings.astype(np.int32), -np.ones((row_count, 1), dtype=np.int32)])
w = np.hstack([splitted_weights.astype(np.float64), np.zeros((row_count, 1), dtype=np.float64)])
s = splitted_sizes.astype(np.int32)

np.savez_compressed(args.out, mappings=m, sizes=s, weights=w, points=points)
print("wrote", args.out, m.shape, w.shape, s.shape)
