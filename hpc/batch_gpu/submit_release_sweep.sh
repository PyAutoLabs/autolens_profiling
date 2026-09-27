#!/bin/bash
#
# Release-sweep launcher — autolens_profiling#345 (the run-time dashboard).
#
# Submits the A100 likelihood-runtime legs that feed the run-time-over-time dashboard,
# every one pinned to the reference node in hpc/release_sweep.conf (RELEASE_SWEEP_NODE),
# so that a release's points sit on the same host as the previous release's. Runtime legs
# only: the dashboard plots per-call cost per release, and the breakdown legs are campaign
# instruments, not trend rows.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir because the
# submits' -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_release_sweep.sh                  # node from release_sweep.conf
#     hpc/batch_gpu/submit_release_sweep.sh --node <name>    # override the pin (say why in the issue)
#     hpc/batch_gpu/submit_release_sweep.sh --dry-run        # print, submit nothing
#
# Run it on the RAL checkout AFTER `HPCPullPyAuto` has moved the libraries to the release
# tag; the rows are versioned by the PyAutoLens release the process imports, and the
# provenance block (`device.provenance`) records the node, the load average and the job id
# the dashboard qualifies rows on. Record the printed job ids in the release issue.

set -e

usage() {
    echo "usage: submit_release_sweep.sh [--node <name>] [--dry-run]" >&2
}

HERE="$(cd "$(dirname "$0")" && pwd)"
CONF="$HERE/../release_sweep.conf"

NODE=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --node)
            if [ $# -lt 2 ]; then
                echo "submit_release_sweep.sh: --node needs a node name" >&2
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
            echo "submit_release_sweep.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

if [ -z "$NODE" ]; then
    if [ ! -f "$CONF" ]; then
        echo "submit_release_sweep.sh: $CONF missing and no --node given" >&2
        exit 1
    fi
    NODE="$(sed -n 's/^RELEASE_SWEEP_NODE=//p' "$CONF" | tail -1)"
    if [ -z "$NODE" ]; then
        echo "submit_release_sweep.sh: RELEASE_SWEEP_NODE not set in $CONF" >&2
        exit 1
    fi
fi

cd "$HERE"

# The dashboard's trend rows: every per-cell A100 runtime submit (dense, then sparse).
LEGS="
submit_runtime_imaging_mge_a100_hst_fp64
submit_runtime_imaging_pixelization_a100_hst_fp64
submit_runtime_imaging_delaunay_a100_hst_fp64
submit_runtime_imaging_delaunay_nn_a100_hst_fp64
submit_runtime_imaging_mge_a100_hst_fp64_sparse
submit_runtime_imaging_pixelization_a100_hst_fp64_sparse
submit_runtime_imaging_delaunay_a100_hst_fp64_sparse
submit_runtime_imaging_delaunay_nn_a100_hst_fp64_sparse
"

SBATCH_CMD="sbatch --nodelist=$NODE"

# Fail before submitting anything if a leg is missing, rather than half-way in.
for leg in $LEGS; do
    if [ ! -f "$leg" ]; then
        echo "submit_release_sweep.sh: missing submit '$leg' in $(pwd)" >&2
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
    echo "would submit $count legs to $NODE"
else
    echo "submitted $count legs to $NODE"
fi
