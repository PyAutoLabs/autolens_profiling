#!/bin/bash
#
# 2026-09 matrix-free fiducial legs launcher — autolens_profiling#247, phase 2.
#
# Submits the three fiducial legs of the matrix-free reference set,
# imaging/matrix_free x {rectangular, delaunay, delaunay_nn} x A100 fp64, in one
# command. Each rebuilds the model of the sibling breakdown cell of the same
# mesh, drives the likelihood matrix-free (PCG + matrix-free PDIP + SLQ log-dets)
# and measures the exact dense comparators in the same process.
#
# Pass --node to pin every leg to ONE A100 node: these rows are compared to each
# other and to the 2026-09-11 `_recon_split` rows, which ran on
# euclid-ral-gpu-2 — and the same code measured F at 4.8 vs 25.6 ms once the
# autotune cache differed (results/notes/xla_autotune_triton_gemm.md), which is
# the same class of confound as a different node. Each submit wipes its own
# JAX_COMPILATION_CACHE_DIR, so co-tenancy on one node does not seed one leg's
# kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_matrix_free_fiducial.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_matrix_free_fiducial.sh --dry-run       # print, submit nothing
#
# Every leg runs the cell at its fiducial mesh with
# `--slq-probes 4,8,16,32 --slq-steps 20,40,80,160,320,640 --vmap-batch 16`
# (the Delaunay family also passes `--regularization adapt_split`, matching the
# `_recon_split` legs these rows are compared against). The SLQ step grid is
# wider than the cell's own default because the bias is Lanczos-step-limited,
# not probe-noise-limited — see the per-leg header. `--dry-run` prints each
# leg's exact invocation.
#
# Record the printed job ids and the node in the issue. Results are committed
# from the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_matrix_free_fiducial.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_matrix_free_fiducial.sh: --node needs a node name" >&2
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
            echo "submit_matrix_free_fiducial.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first: it is the cheapest mesh to build and the one whose pins are
# recorded on both the breakdown and runtime sides, so it is the fastest leg to
# read a gate off if something is wrong with the branch.
LEGS="
submit_breakdown_imaging_matrix_free_pixelization_a100_hst_fp64
submit_breakdown_imaging_matrix_free_delaunay_a100_hst_fp64
submit_breakdown_imaging_matrix_free_delaunay_nn_a100_hst_fp64
"

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_matrix_free_fiducial.sh: missing submit '$leg' in $(pwd)" >&2
        exit 1
    fi
done

count=0
for leg in $LEGS; do
    count=$((count + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD $leg"
        # Print what the leg will actually run, read out of the submit itself,
        # so a dry run says which mesh and which SLQ grid without opening it.
        sed -n '/^python3 -u scripts/,/^$/p' "$leg" | sed 's/^/      /'
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
