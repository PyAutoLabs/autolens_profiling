#!/bin/bash
#
# 2026-09 fixed-lens-light VERDICT legs launcher — autolens_profiling#259, phase 5
# of the fixed-lens-light-profiling epic (the last).
#
# Submits all four arrays of the A100 grid, imaging/fixed_light_library x
# {rectangular, delaunay} x {hst, euclid} x A100 fp64, in one command. Each array
# is three arms — --source-pixels 500 / 1250 / 2500 — so this is twelve legs.
#
# Each arm measures one whole library likelihood call per route (a S0 PDIP today,
# b S3 PDIP reference, c S3 positive-negative, d S3 certified active set at phase
# 3's production budget with the PDIP fallback), the certification sweep over
# budgets 1..12 that records the smallest certifying budget at that (dataset, mesh,
# N), and route c's Δlog-evidence in nats with its negative-pixel count. The
# deliverable is the 2 x 3 x hardware table and, from it, a stated production
# configuration per hardware.
#
# THE EUCLID DATASET IS SIMULATED ONCE, HERE, BEFORE ANY SUBMIT. It is gitignored
# and auto-simulated on first use, and three concurrent arms racing to write the
# same dataset/imaging/euclid/ would interleave their FITS writes. Each submit
# re-checks it and fails loudly rather than auto-simulating.
#
# Pass --node to pin every arm to ONE A100 node: these rows are compared to each
# other and to the phase-0/1/2/3/4 rows, which ran on euclid-ral-gpu-2 — and the
# same code measured F at 4.8 vs 25.6 ms once the autotune cache differed
# (results/notes/xla_autotune_triton_gemm.md), which is the same class of confound
# as a different node. Each arm wipes its own JAX_COMPILATION_CACHE_DIR, so
# co-tenancy on one node does not seed one arm's kernels from another's.
#
# SUBMIT ONLY ONCE THE CHECKOUT IS COMPLETE. Phase 4 lost ten arms in four seconds
# because `git worktree add` was still writing its 948 files when sbatch ran.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_fixed_light_verdict.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_fixed_light_verdict.sh --dry-run       # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed from
# the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_fixed_light_verdict.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_fixed_light_verdict.sh: --node needs a node name" >&2
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
            echo "submit_fixed_light_verdict.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first, for the reason the phase-0/1/3/4 launchers give: it is the
# cheapest mesh to build. HST before Euclid, because HST is the dataset every
# earlier phase measured and the one whose numbers can be checked against them.
LEGS="
submit_breakdown_imaging_fixed_light_verdict_pixelization_a100_hst_fp64
submit_breakdown_imaging_fixed_light_verdict_delaunay_a100_hst_fp64
submit_breakdown_imaging_fixed_light_verdict_pixelization_a100_euclid_fp64
submit_breakdown_imaging_fixed_light_verdict_delaunay_a100_euclid_fp64
"

AP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# One simulate, before any submit. `should_simulate` is the same gate the cells
# use, so this is a no-op when the dataset is already there.
for ds in hst euclid; do
    if [ ! -f "$AP_ROOT/dataset/imaging/$ds/data.fits" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "would simulate dataset/imaging/$ds"
        else
            echo "simulating dataset/imaging/$ds (once, before submitting)"
            ( cd "$AP_ROOT" && source activate.sh && \
              JAX_PLATFORMS=cpu python3 scripts/misc/simulators/imaging.py \
                  --instrument "$ds" >/dev/null )
        fi
    fi
done

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_fixed_light_verdict.sh: missing submit '$leg' in $(pwd)" >&2
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
    echo "job=$(echo "$out" | awk '{print $NF}') $leg (array 0-2)"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "would submit $count arrays (12 legs)"
else
    echo "submitted $count arrays (12 legs)"
fi
