#!/bin/bash
#
# 2026-09 pixelized baseline grid launcher — autolens_profiling#241.
#
# Submits the twelve legs of the A100 fp64 pixelized reference set,
# {pixelization, delaunay, delaunay_nn} x {runtime, breakdown} x {dense, sparse},
# in one command. Runtime legs go first (they are the cheap ones and produce the
# vmap per-call and VRAM rows the breakdown tables are reconciled against), then
# the compile-heavy breakdown legs; dense before sparse within each.
#
# Pass --node to pin every leg to ONE A100 node: the twelve rows are only
# comparable to each other if they ran on the same GPU (results/notes/
# xla_autotune_triton_gemm.md — the same code measured F at 4.8 vs 25.6 ms once
# the autotune cache differed, and node-to-node A100 differences are the same
# class of confound). Each submit wipes its own JAX_COMPILATION_CACHE_DIR, so
# co-tenancy on one node does not seed one leg's kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_baseline_grid.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_baseline_grid.sh --dry-run          # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed
# from the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_baseline_grid.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_baseline_grid.sh: --node needs a node name" >&2
                usage
                exit 2
            fi
            NODE="$2"
            shift 2
            ;;
        --node=*)
            NODE="${1#--node=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "submit_baseline_grid.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Order: runtime before breakdown, dense before sparse, then the mesh family.
LEGS="
submit_runtime_imaging_pixelization_a100_hst_fp64
submit_runtime_imaging_delaunay_a100_hst_fp64
submit_runtime_imaging_delaunay_nn_a100_hst_fp64
submit_runtime_imaging_pixelization_a100_hst_fp64_sparse
submit_runtime_imaging_delaunay_a100_hst_fp64_sparse
submit_runtime_imaging_delaunay_nn_a100_hst_fp64_sparse
submit_breakdown_imaging_pixelization_a100_hst_fp64
submit_breakdown_imaging_delaunay_a100_hst_fp64
submit_breakdown_imaging_delaunay_nn_a100_hst_fp64
submit_breakdown_imaging_pixelization_a100_hst_fp64_sparse
submit_breakdown_imaging_delaunay_a100_hst_fp64_sparse
submit_breakdown_imaging_delaunay_nn_a100_hst_fp64_sparse
"

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_baseline_grid.sh: missing submit '$leg' in $(pwd)" >&2
        exit 1
    fi
done

count=0
for leg in $LEGS; do
    count=$((count + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD $leg"
        continue
    fi
    out=$($SBATCH_CMD "$leg")
    echo "job=$(echo "$out" | awk '{print $NF}') $leg"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "would submit $count legs"
else
    echo "submitted $count legs"
fi
