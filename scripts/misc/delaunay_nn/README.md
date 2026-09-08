# DelaunayNN profiling

This profile isolates the full geometry cost of the Sibson natural-neighbour
mapper added as `mesh.DelaunayNN`. It times:

- qhull connectivity and JAX point location;
- data-grid natural-neighbour weights;
- barycentric dual areas and split-cross coordinates;
- natural-neighbour weights for all split-regularization points.

The comparison is therefore the complete mapper-table construction against
the equivalent barycentric `Delaunay` path, rather than an isolated
circumcircle kernel.

Run the production-shaped CPU/GPU benchmark from the repository root:

```bash
python scripts/misc/delaunay_nn/benchmark.py
```

For the laptop GPU, activate the CUDA environment first:

```bash
PyAutoGPU
JAX_PLATFORMS=cuda,cpu python scripts/misc/delaunay_nn/benchmark.py --repeats 10
```

The CPU backend remains registered because the Delaunay connectivity callback
runs qhull on the host even when the mapped JAX work targets CUDA. Ten warm
repeats give a more representative median across the laptop GPU's dynamic clock
states than the five-repeat CPU default.

Useful overrides:

```bash
python scripts/misc/delaunay_nn/benchmark.py \
  --mesh-points 1500 --queries 20000 --caps 16 24 32 64 --repeats 10
```

The script writes a versioned JSON and PNG pair under
`results/delaunay_nn/`. GPU filenames include the JAX device identity so a
laptop result cannot overwrite a later A100 run; the JSON also records the
same hardware key. The cap sweep exposes the speed/headroom trade-off;
the separate workspace assertion
`scripts/misc/jax_assertions/delaunay_nn_caps.py` decides correctness using
actual Hilbert meshes ray-traced through a mass-model ensemble.

The current cap decision and measured tail distributions are recorded in
[`results/notes/delaunay_nn_cap_audit.md`](../../../results/notes/delaunay_nn_cap_audit.md).

## ConstantSplit assembly bench (`assembly_bench.py`)

`assembly_bench.py` times the *regularization* half of the DelaunayNN cell —
`reg_split_from` plus `pixel_splitted_regularization_matrix_from` — which the
launch-latency work left as the largest remaining per-call cost
([`results/notes/delaunay_nn_launch_latency.md`](../../../results/notes/delaunay_nn_launch_latency.md)).
It runs the real `(6000, 33)` HST split stencil tables, checked in at
`results/delaunay_nn/tables_hst_1500.npz`, through every candidate formulation:

| variant | what it measures |
|---|---|
| `V0` | the pre-#536 path: the full padded `(4P, K, K)` scatter (`compact_width=K`); the reference every other variant is diffed against |
| `V0a_4wide` | the same scatter on 4-wide `Delaunay`-style tables, as the width calibration |
| `V1_dense_gemm` / `V1_highest` | dense `B^T diag(s) B` |
| `V2_bcoo_dense` / `V2b_bcoo_bcoo` | sparse BCOO assembly |
| `V3_dedup_segsum` | dedup sort + `segment_sum` |
| `V5_compact_k<N>` | a fixed compaction to `N` columns with no wide-row supplement — the width sweep that priced the compaction |
| `V6_library` | **the shipped library path**: default `SPLIT_REG_COMPACT_WIDTH` compaction plus the wide-row supplement |

```bash
python scripts/misc/delaunay_nn/assembly_bench.py
python scripts/misc/delaunay_nn/assembly_bench.py --pixels 300 --repeats 3   # smoke
```

Defaults read `results/delaunay_nn/tables_hst_1500.npz` and write
`results/delaunay_nn/assembly_bench.json`. The tables are checked in (330 KB);
rebuild them from the HST breakdown cell with

```bash
python scripts/misc/delaunay_nn/extract_mesh_points.py \
    --out results/delaunay_nn/mesh_points_hst.npy
python scripts/misc/delaunay_nn/build_tables.py \
    --points results/delaunay_nn/mesh_points_hst.npy \
    --out results/delaunay_nn/tables_hst_1500.npz
```

`V6_library` must reproduce `V0` exactly on these tables (max abs diff 0.0 on
CPU): every real stencil row is narrower than the compact width, so the
supplement contributes only zeros. The tail geometries that do exceed it are
audited by `autolens_workspace_test/scripts/misc/jax_assertions/delaunay_nn_caps.py`,
whose per-geometry count of above-width split rows is the margin the wide-row
budget is set against.
