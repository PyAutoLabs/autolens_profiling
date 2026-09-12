#!/bin/bash
#
# 2026-09 fixed-lens-light legs launcher — autolens_profiling#248, phase 2.
#
# Submits the three fiducial legs of the fixed-lens-light comparator set,
# imaging/fixed_light x {rectangular, delaunay, delaunay_nn} x A100 fp64, in one
# command. Each rebuilds the model of the sibling breakdown cell of the same
# mesh, measures the reconstruction row on S0 (the system the library solves
# today) and on S3 (the source-only system left once the lens light is fixed),
# and times the certified active set on S3 at pass budgets 1..8.
#
# Pass --node to pin every leg to ONE A100 node: these rows are compared to each
# other and to the 2026-09-11 `_recon_split` rows, which ran on
# euclid-ral-gpu-2 — and the same code measured F at 4.8 vs 25.6 ms once the
# autotune cache differed (results/notes/xla_autotune_triton_gemm.md), which is
# the same class of confound as a different node. Each submit wipes its own
# JAX_COMPILATION_CACHE_DIR, so co-tenancy on one node does not seed one leg's
# kernels from another's.
#
# Pass --with-n3000 to add the two mesh-size legs (rectangular and Delaunay at
# --source-pixels 3000). They are off by default because they are a different
# question from the fiducial comparison — whether the certifying pass budget
# grows with the mesh — and they skip the pins.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_fixed_light.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_fixed_light.sh --node euclid-ral-gpu-2 --with-n3000
#     hpc/batch_gpu/submit_fixed_light.sh --dry-run       # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed
# from the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_fixed_light.sh [--node <name>] [--with-n3000] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0
WITH_N3000=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_fixed_light.sh: --node needs a node name" >&2
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
        --with-n3000)
            WITH_N3000=1
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
            echo "submit_fixed_light.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first: it is the cheapest mesh to build, it is the mesh the
# phase-1 probe measured the hardest case on (7 passes to certify, against
# Delaunay's 2), and its pins are recorded on both the breakdown and runtime
# sides — so it is the fastest leg to read a gate off if the branch is wrong.
LEGS="
submit_breakdown_imaging_fixed_light_pixelization_a100_hst_fp64
submit_breakdown_imaging_fixed_light_delaunay_a100_hst_fp64
submit_breakdown_imaging_fixed_light_delaunay_nn_a100_hst_fp64
"

if [ "$WITH_N3000" -eq 1 ]; then
    LEGS="$LEGS
submit_breakdown_imaging_fixed_light_pixelization_a100_hst_fp64_n3000
submit_breakdown_imaging_fixed_light_delaunay_a100_hst_fp64_n3000
"
fi

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_fixed_light.sh: missing submit '$leg' in $(pwd)" >&2
        exit 1
    fi
done

count=0
for leg in $LEGS; do
    count=$((count + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD $leg"
        # Print what the leg will actually run, read out of the submit itself,
        # so a dry run says which mesh and which flags without opening it.
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
