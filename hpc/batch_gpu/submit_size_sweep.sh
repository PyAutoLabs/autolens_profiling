#!/bin/bash
#
# 2026-09 reduced N_src size sweep launcher — autolens_profiling#247, phase 3.
#
# Submits the 26 legs of the reduced sweep, all through the ONE templated
# submit `submit_size_sweep_leg`, which reads SWEEP_MESH / SWEEP_PATH / SWEEP_N
# from the environment.
#
# The sweep is reduced from the 45 legs first planned. The CPU reference
# measured cond(F+lambda*H) ~ 4e10, which settles the speed question at the
# fiducial tier: Jacobi PCG needs thousands of matvecs and SLQ is Lanczos-step
# limited, so the sweep's job is now mainly the MEMORY question — at what mesh
# size do the dense and sparse exact paths stop fitting an A100 — plus
# matrix-free at three sizes for the note's crossover table.
#
#   path         meshes x sizes                                          legs
#   dense        rectangular, delaunay x {3000,5000,8000,12000}            8
#                delaunay_nn x {3000,5000}                                 2
#   sparse       the same ten, with --sparse                              10
#   matrix_free  rectangular, delaunay x {3000,5000,12000}                 6
#                                                                        ----
#                                                                          26
#
# The fiducial n=1500 rows already exist and are NOT re-run here: the baseline
# `_hpc_a100_fp64{,_sparse}` legs, the 2026-09-11 `_recon_split` legs (which
# carry the exact Cholesky / log-det comparators), and the three pending
# `_matrix_free` legs 342664-342666.
#
# Submission order — cheapest and most likely to survive, first:
#   1. matrix_free  (~30 min each; the rows the note's crossover table needs)
#   2. sparse
#   3. dense        (heaviest, most likely to OOM)
# and within each path, small N before large. A queue that drains only partway
# should therefore leave the most informative rows already measured. An OOM is a
# datum, not a failure — see the leg submit's header.
#
# Pass --node to pin every leg to ONE A100 node. These rows are compared to each
# other AND to the 2026-09-11 `_recon_split` / fiducial matrix-free rows, which
# ran on euclid-ral-gpu-2, and the same code measured F at 4.8 vs 25.6 ms once
# the autotune cache differed (results/notes/xla_autotune_triton_gemm.md) —
# node-to-node A100 differences are the same class of confound. Each leg wipes
# its own JAX_COMPILATION_CACHE_DIR, so co-tenancy on one node does not seed one
# leg's kernels from another's.
#
# Usage (on the RAL login node, from anywhere — the script cd's to its own dir
# because the submit's -o/-e paths are relative to hpc/batch_gpu/):
#
#     hpc/batch_gpu/submit_size_sweep.sh --node euclid-ral-gpu-2
#     hpc/batch_gpu/submit_size_sweep.sh --dry-run                  # print, submit nothing
#     hpc/batch_gpu/submit_size_sweep.sh --dry-run --paths matrix_free --sizes 3000
#
# --paths / --meshes / --sizes SELECT from the table above; they are filters,
# not generators. They cannot invent a leg the table does not contain (the table
# IS the reduced sweep the human chose) — to add one, edit LEGS below. A filter
# that selects nothing is an error, not a silent no-op.
#
# Every submitted leg is appended to results/sweep/size_sweep_manifest.jsonl —
# that file, not this terminal, is the provenance the aggregator reads to tell a
# leg that OOMed from a leg that was never submitted. Commit it (from the RAL
# worktree, with plain git) after a real submit.

set -e

usage() {
    cat >&2 <<'EOF'
usage: submit_size_sweep.sh [--node <name>] [--dry-run]
                            [--paths dense,sparse,matrix_free]
                            [--meshes rectangular,delaunay,delaunay_nn]
                            [--sizes 3000,5000,8000,12000]

  --node     pin every leg to one A100 node (strongly recommended)
  --dry-run  print the legs and their exact invocations, submit nothing
  --paths    filter the leg table by path      (default: all three)
  --meshes   filter the leg table by mesh      (default: all three)
  --sizes    filter the leg table by requested N (default: all four)
EOF
}

NODE=""
DRY_RUN=0
PATHS_FILTER=""
MESHES_FILTER=""
SIZES_FILTER=""

need_value() {
    if [ "$2" -lt 2 ]; then
        echo "submit_size_sweep.sh: $1 needs a value" >&2
        usage
        exit 2
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --node)     need_value "$1" $#; NODE="$2"; shift 2 ;;
        --node=*)   NODE="${1#--node=}"; shift ;;
        --paths)    need_value "$1" $#; PATHS_FILTER="$2"; shift 2 ;;
        --paths=*)  PATHS_FILTER="${1#--paths=}"; shift ;;
        --meshes)   need_value "$1" $#; MESHES_FILTER="$2"; shift 2 ;;
        --meshes=*) MESHES_FILTER="${1#--meshes=}"; shift ;;
        --sizes)    need_value "$1" $#; SIZES_FILTER="$2"; shift 2 ;;
        --sizes=*)  SIZES_FILTER="${1#--sizes=}"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)
            echo "submit_size_sweep.sh: unknown argument '$1'" >&2
            usage
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")"

AP_ROOT="$(cd ../.. && pwd)"
LEG_SUBMIT="submit_size_sweep_leg"
MANIFEST="$AP_ROOT/results/sweep/size_sweep_manifest.jsonl"

if [ ! -f "$LEG_SUBMIT" ]; then
    echo "submit_size_sweep.sh: missing '$LEG_SUBMIT' in $(pwd)" >&2
    exit 1
fi

# The leg table, in submission order: matrix_free, then sparse, then dense;
# small N before large inside each path; rectangular before delaunay before
# delaunay_nn at equal N. One leg per line: "<path> <mesh> <N>".
LEGS="
matrix_free rectangular 3000
matrix_free delaunay    3000
matrix_free rectangular 5000
matrix_free delaunay    5000
matrix_free rectangular 12000
matrix_free delaunay    12000
sparse      rectangular 3000
sparse      delaunay    3000
sparse      delaunay_nn 3000
sparse      rectangular 5000
sparse      delaunay    5000
sparse      delaunay_nn 5000
sparse      rectangular 8000
sparse      delaunay    8000
sparse      rectangular 12000
sparse      delaunay    12000
dense       rectangular 3000
dense       delaunay    3000
dense       delaunay_nn 3000
dense       rectangular 5000
dense       delaunay    5000
dense       delaunay_nn 5000
dense       rectangular 8000
dense       delaunay    8000
dense       rectangular 12000
dense       delaunay    12000
"

# `--paths a,b` -> "a b"; empty filter matches everything.
in_filter() {
    local value="$1" filter="$2"
    [ -z "$filter" ] && return 0
    local item
    for item in $(echo "$filter" | tr ',' ' '); do
        [ "$item" = "$value" ] && return 0
    done
    return 1
}

SBATCH_CMD="sbatch"
if [ -n "$NODE" ]; then
    SBATCH_CMD="sbatch --nodelist=$NODE"
fi

SHA="$(git -C "$AP_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$(dirname "$MANIFEST")"
fi

count=0
while read -r leg_path leg_mesh leg_n; do
    [ -z "$leg_path" ] && continue
    in_filter "$leg_path" "$PATHS_FILTER" || continue
    in_filter "$leg_mesh" "$MESHES_FILTER" || continue
    in_filter "$leg_n" "$SIZES_FILTER" || continue

    count=$((count + 1))
    job_name="sweep_${leg_mesh}_${leg_path}_n${leg_n}"
    export_list="ALL,SWEEP_MESH=${leg_mesh},SWEEP_PATH=${leg_path},SWEEP_N=${leg_n}"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$SBATCH_CMD --export=$export_list -J $job_name $LEG_SUBMIT"
        # What the leg will actually run, so a dry run says the cell and the
        # flags without opening the submit.
        case "$leg_mesh" in
            rectangular) cell=pixelization; reg="" ;;
            delaunay)    cell=delaunay;     reg=" --regularization adapt_split" ;;
            delaunay_nn) cell=delaunay_nn;  reg=" --regularization adapt_split" ;;
        esac
        case "$leg_path" in
            dense)
                echo "      python3 -u scripts/imaging/likelihood_breakdown/${cell}.py --config-name hpc_a100_fp64_n${leg_n} --source-pixels ${leg_n}${reg} --split-setup --vmap-batch 16"
                ;;
            sparse)
                echo "      python3 -u scripts/imaging/likelihood_breakdown/${cell}.py --config-name hpc_a100_fp64_n${leg_n} --source-pixels ${leg_n}${reg} --split-setup --vmap-batch 16 --sparse"
                ;;
            matrix_free)
                echo "      python3 -u scripts/imaging/likelihood_breakdown/matrix_free.py --mesh ${leg_mesh} --source-pixels ${leg_n} --config-name hpc_a100_fp64_matrix_free_n${leg_n}${reg} --no-pdip --slq-probes 16 --slq-steps 20,80,320 --vmap-batch 16"
                ;;
        esac
        continue
    fi

    out=$($SBATCH_CMD --export="$export_list" -J "$job_name" "$LEG_SUBMIT")
    job=$(echo "$out" | awk '{print $NF}')
    echo "job=$job $leg_mesh $leg_path n=$leg_n"
    printf '{"job": "%s", "mesh": "%s", "path": "%s", "n": %s, "node": "%s", "submitted_utc": "%s", "sha": "%s"}\n' \
        "$job" "$leg_mesh" "$leg_path" "$leg_n" "${NODE:-any}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHA" >> "$MANIFEST"
done <<EOF
$LEGS
EOF

if [ "$count" -eq 0 ]; then
    echo "submit_size_sweep.sh: no leg in the table matches those filters" >&2
    echo "  --paths '$PATHS_FILTER' --meshes '$MESHES_FILTER' --sizes '$SIZES_FILTER'" >&2
    echo "  The filters SELECT from the 26-leg table; they cannot add a leg." >&2
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "would submit $count legs"
else
    echo "submitted $count legs"
    echo "manifest: $MANIFEST"
fi
