#!/bin/bash
#
# 2026-09 fixed-lens-light LIBRARY-PATH legs launcher — autolens_profiling#251,
# phase 1 of the fixed-lens-light-profiling epic.
#
# Submits the three fiducial legs of the library-path comparator set,
# imaging/fixed_light_library x {rectangular, delaunay, delaunay_nn} x A100 fp64,
# in one command. Each rebuilds the model of the sibling breakdown cell of the
# same mesh and times ONE WHOLE library likelihood call per solver route: S0 PDIP,
# S3 PDIP, S3 positive-negative (xp.linalg.solve), S3 certified active set at that
# mesh's certifying pass budget with and without the PDIP fallback, and the same
# scheme at budget 1 where the fallback always fires.
#
# This is the row phase 0 (#248) could not measure. Its "~24 ms/call certified"
# figure is arithmetic on measured kernel rows and says so; these legs measure it.
#
# Pass --node to pin every leg to ONE A100 node: these rows are compared to each
# other and to the phase-0 fixed_light rows, which ran on euclid-ral-gpu-2 — and the
# same code measured F at 4.8 vs 25.6 ms once the autotune cache differed
# (results/notes/xla_autotune_triton_gemm.md), which is the same class of confound
# as a different node. Each submit wipes its own JAX_COMPILATION_CACHE_DIR, so
# co-tenancy on one node does not seed one leg's kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_fixed_light_library.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_fixed_light_library.sh --dry-run       # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed from
# the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_fixed_light_library.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_fixed_light_library.sh: --node needs a node name" >&2
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
            echo "submit_fixed_light_library.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first, for the reason the phase-0 launcher gives: it is the cheapest
# mesh to build, it is the hard case for the active set (7 passes to certify against
# Delaunay's 2, because its dual violation set drains behind 152 edge-zeroed pixels),
# and its pins are recorded on both the breakdown and runtime sides — so it is the
# fastest leg to read a gate off if the branch is wrong.
LEGS="
submit_breakdown_imaging_fixed_light_library_pixelization_a100_hst_fp64
submit_breakdown_imaging_fixed_light_library_delaunay_a100_hst_fp64
submit_breakdown_imaging_fixed_light_library_delaunay_nn_a100_hst_fp64
"

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_fixed_light_library.sh: missing submit '$leg' in $(pwd)" >&2
        exit 1
    fi
done

count=0
for leg in $LEGS; do
    count=$((count + 1))
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD $leg"
        # Print what the leg will actually run, read out of the submit itself, so a
        # dry run says which mesh and which flags without opening it.
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
