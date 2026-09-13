#!/bin/bash
#
# 2026-09 fixed-lens-light LOW-LIKELIHOOD DRAW legs launcher — autolens_profiling#255,
# phase 3 of the fixed-lens-light-profiling epic.
#
# Submits both fiducial legs of the draw-set study,
# imaging/fixed_light_draws x {rectangular, delaunay} x A100 fp64, in one command.
# Each rebuilds the whole S0 -> S3 chain at a graded set of deliberately poor models
# — four one-parameter walks bisected onto delta log L ~ -10 / -100 / -1000 / -1e4,
# plus 24 seeded random draws from SLaM-like priors — and measures, per draw, the
# certified pass count and ms, the PDIP iteration count and ms, the seed-set size,
# and the unconstrained solve's delta log-evidence.
#
# The deliverable is the FALLBACK RATE at fixed pass budgets 2 / 4 / 6 / 7: the
# fraction of draws whose certification needs more passes than the budget. Phases
# 0-2 measured the scheme only at the fiducial, where it certifies in 2 passes
# (Delaunay) and 7 (rectangular); this is the leg that says whether those budgets
# survive a search's actual population of models.
#
# Pass --node to pin both legs to ONE A100 node: these rows are compared to each
# other and to the phase-0/1/2 rows, which ran on euclid-ral-gpu-2 — and the same
# code measured F at 4.8 vs 25.6 ms once the autotune cache differed
# (results/notes/xla_autotune_triton_gemm.md), which is the same class of confound
# as a different node. Each submit wipes its own JAX_COMPILATION_CACHE_DIR, so
# co-tenancy on one node does not seed one leg's kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_fixed_light_draws.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_fixed_light_draws.sh --dry-run       # print, submit nothing
#
# Record the printed job ids and the node in the issue. Results are committed from
# the RAL worktree with plain git; this script never touches hpc/sync.

set -e

usage() {
    echo "usage: submit_fixed_light_draws.sh [--node <name>] [--dry-run]" >&2
}

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_fixed_light_draws.sh: --node needs a node name" >&2
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
            echo "submit_fixed_light_draws.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

# Rectangular first, for the reason the phase-0/1 launchers give and one of this
# phase's own: it is the cheapest mesh to build, its pins are recorded on both the
# breakdown and runtime sides, AND it is the hard case for the scheme — 7 passes at
# the fiducial against Delaunay's 2, because its dual violation set drains behind
# 152 edge-zeroed pixels. If the pass count grows with model error at all, this is
# the leg where it shows first.
LEGS="
submit_breakdown_imaging_fixed_light_draws_pixelization_a100_hst_fp64
submit_breakdown_imaging_fixed_light_draws_delaunay_a100_hst_fp64
"

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_fixed_light_draws.sh: missing submit '$leg' in $(pwd)" >&2
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
