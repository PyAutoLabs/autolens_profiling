#!/bin/bash
#
# 2026-09 fixed-lens-light SOURCE-PIXEL SCALING legs launcher — autolens_profiling#257,
# phase 4 of the fixed-lens-light-profiling epic.
#
# Submits both arrays of the scaling sweep, imaging/fixed_light x {rectangular,
# delaunay} x A100 fp64, in one command. Each array is five arms —
# --source-pixels 500 / 1000 / 1500 / 2500 / 4000 — so this is ten legs.
#
# Each arm measures, at its N: the certified active set at every pass budget
# 1..12 (which covers phase 3's safe budgets 11 rectangular and 7 Delaunay, so
# the leg reports BOTH the budget that certifies at that N and the cost of the
# safe fixed budget at that N), the unconstrained Cholesky, the cell-driven PDIP
# reference, both log determinants, the batched @vmap 16 rows, and one whole
# library likelihood call. The deliverable is the ms-vs-N curve per row and its
# fitted scaling exponent, beside the memory ceiling.
#
# Pass --node to pin every arm to ONE A100 node: these rows are compared to each
# other and to the phase-0/1/2/3 rows, which ran on euclid-ral-gpu-2 — and the
# same code measured F at 4.8 vs 25.6 ms once the autotune cache differed
# (results/notes/xla_autotune_triton_gemm.md), which is the same class of
# confound as a different node. Each arm wipes its own
# JAX_COMPILATION_CACHE_DIR, so co-tenancy on one node does not seed one arm's
# kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_fixed_light_scaling.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_fixed_light_scaling.sh --dry-run       # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed
# from the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_fixed_light_scaling.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_fixed_light_scaling.sh: --node needs a node name" >&2
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
            echo "submit_fixed_light_scaling.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first, for the reason the phase-0/1/3 launchers give and one of
# this phase's own: it is the cheapest mesh to build, and it is the mesh whose
# pass budget phase 0 already saw move with N (6 passes at n=3025 against 7 at
# n=1521). If the certifying budget scales with the mesh at all, this is the
# array where it shows first.
LEGS="
submit_breakdown_imaging_fixed_light_scaling_pixelization_a100_hst_fp64
submit_breakdown_imaging_fixed_light_scaling_delaunay_a100_hst_fp64
"

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_fixed_light_scaling.sh: missing submit '$leg' in $(pwd)" >&2
        exit 1
    fi
done

count=0
for leg in $LEGS; do
    count=$((count + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD $leg"
        # Print what the leg will actually run, read out of the submit itself, so
        # a dry run says which mesh and which flags without opening it.
        sed -n '/^python3 -u scripts/,/^$/p' "$leg" | sed 's/^/      /'
        continue
    fi
    out=$($SBATCH_CMD "$leg")
    echo "job=$(echo "$out" | awk '{print $NF}') $leg (array 0-4)"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "would submit $count arrays (10 legs)"
else
    echo "submitted $count arrays (10 legs)"
fi
