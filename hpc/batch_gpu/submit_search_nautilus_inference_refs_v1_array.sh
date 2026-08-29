#!/bin/bash -l
#
# W4 / issue #161 (Phase 1 targets registry) — InferenceRefs_v1 reference-row
# array. Produces the 11 still-needed Nautilus fp64 reference rows listed in
# results/baselines/InferenceRefs_v1/SUBMIT_LIST.md, one per array task.
#
# *** THIS SCRIPT HAS NOT BEEN SUBMITTED. *** It is Phase 1 prep — writing
#   the array so it exists and is reviewable is one plan step; running an
#   ~11-task, multi-hour-per-task A100 campaign is a separate decision this
#   commit does NOT make. Do not `sbatch` this without a human go-ahead.
#
# WHY THIS EXISTS
#   InferenceRefs_v1 today certifies exactly two targets (mge_fp64,
#   delaunay_fp64), both adopted RETROACTIVELY from existing runs — see
#   results/baselines/InferenceRefs_v1/{mge_fp64,delaunay_fp64}/README.md.
#   Every other registered target (pixelization, delaunay_nn, knn,
#   delaunay_matern, slam_source_pix, slam_source_pix_nn, and the
#   positions-on arms of mge/delaunay) has NO certified reference posterior
#   at all. This script is what would produce them.
#
# ARM (every task)
#   sampler       nautilus
#   precision     fp64
#   n_live        >= 2x the scripts/misc/searches/_samplers._N_LIVE fiducial
#                 for the row's cell, via SEARCHES_NAUTILUS_N_LIVE (added in
#                 this same commit series — see _samplers.build_nautilus)
#   seed          0 (SEARCHES_NAUTILUS_SEED)
#   config-name   hpc_a100_fp64_ref  (writes results/searches/nautilus/
#                 imaging/<model>/hst/hpc_hpc_a100_fp64_ref.json — distinct
#                 from any existing hpc_a100_fp64 row, never overwrites one)
#   positions     off for most rows; on (fixed 0.3 or auto — see table) for
#                 the positions-arm rows
#
# TASK MAPPING — mirrors SUBMIT_LIST.md's table exactly; keep both in sync.
#   0  pixelization           off  --
#   1  delaunay_nn            off  --
#   2  delaunay_nn            on   fixed 0.3
#   3  slam_source_pix        off  --
#   4  slam_source_pix        on   auto (SLaM convention)
#   5  slam_source_pix_nn     off  --
#   6  slam_source_pix_nn     on   auto (SLaM convention)
#   7  knn                    off  --
#   8  delaunay_matern        off  --
#   9  mge                    on   auto (reference-row-specific choice)
#   10 delaunay               on   auto (reference-row-specific choice)
#
# WALL ESTIMATE
#   No row here has ever been run at 2x n_live, so there is no measured
#   number to anchor a per-row estimate on. Conservatively budgeting from the
#   heaviest KNOWN comparable (imaging/delaunay/hst at n_live=150 took
#   1891 s sampler wall — see delaunay_fp64/README.md) and doubling for 2x
#   n_live plus doubling again for margin gives ~2.1 h; --time 6:00:00 leaves
#   generous headroom for the untested rows (delaunay_nn / slam_source_pix*
#   are new meshes with no wall-time history at all).
#
# BEFORE SUBMITTING (do this regardless of who submits it)
#   1. Read SUBMIT_LIST.md and confirm the 11-row table still matches reality
#      (a row may already have been produced and adopted since this script
#      was written).
#   2. Check for a completed-fit resume collision per row:
#        ls $AP_ROOT/output/searches/nautilus/imaging/<model>/hst/hpc_a100_fp64_ref/
#      and move any existing directory aside so each task starts cold —
#      config-name hpc_a100_fp64_ref is deliberately new, but a prior partial
#      run of THIS script would still collide with itself.
#   3. Confirm PyAutoLabs main / the RAL PyAutoAll checkout actually contains
#      this commit series (delaunay_nn / slam_source_pix(_nn) model_types,
#      SEARCHES_NAUTILUS_N_LIVE) — run HPCPullPyAuto first.

# WALL-BASIS: — one row per cell this submit runs. See scripts/misc/wall/README.md.
#   `rates` cites a step rate measured on THAT cell (wall/rates.py); an MGE rate
#   can never stand in for a pixelized cell — that is what killed RAL job 340576.
#   cell: imaging/delaunay/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/delaunay_matern/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/delaunay_nn/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/knn/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/mge/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/pixelization/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/slam_source_pix/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.
#   cell: imaging/slam_source_pix_nn/hst  device: a100  precision: fp64
#   source: unmeasured  probe-first: yes
#   NOTE: no step rate has been measured on this cell. Run one short arm
#   before trusting --time; a truncated arm still measures s/step.

#SBATCH -J search_nautilus_inference_refs_v1
#SBATCH --partition=gpu
# Node exclusion (2026-08-28): euclid-ral-gpu-1 carries a MIG-mode A100 with
# no instances that SLURM still hands out as a plain gpu:A100 — see
# _gpu_preflight.sh, which stays armed as the backstop.
#SBATCH --exclude=euclid-ral-gpu-1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --array=0-10
#SBATCH -o output/output.%A_%a.out
#SBATCH -e error/error.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=james.w.nightingale@durham.ac.uk

export AP_ROOT=/mnt/ral/jnightin/autolens_profiling
source $AP_ROOT/activate.sh

export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=True
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/matplotlib

export SEARCHES_NAUTILUS_SEED=0
export SEARCHES_CONFIG_NAME=hpc_a100_fp64_ref

# (leaf_script, n_live_fiducial, positions, threshold_mode) per task index.
# n_live_fiducial x 2 is passed via SEARCHES_NAUTILUS_N_LIVE.
case "$SLURM_ARRAY_TASK_ID" in
  0)  MODEL=pixelization       FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  1)  MODEL=delaunay_nn        FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  2)  MODEL=delaunay_nn        FIDUCIAL=150 POSITIONS=on  THRESHOLD_MODE=0.3    ;;
  3)  MODEL=slam_source_pix    FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  4)  MODEL=slam_source_pix    FIDUCIAL=150 POSITIONS=on  THRESHOLD_MODE=auto   ;;
  5)  MODEL=slam_source_pix_nn FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  6)  MODEL=slam_source_pix_nn FIDUCIAL=150 POSITIONS=on  THRESHOLD_MODE=auto   ;;
  7)  MODEL=knn                FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  8)  MODEL=delaunay_matern    FIDUCIAL=150 POSITIONS=off THRESHOLD_MODE=""     ;;
  9)  MODEL=mge                FIDUCIAL=200 POSITIONS=on  THRESHOLD_MODE=auto   ;;
  10) MODEL=delaunay           FIDUCIAL=150 POSITIONS=on  THRESHOLD_MODE=auto   ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

export SEARCHES_NAUTILUS_N_LIVE=$((FIDUCIAL * 2))
export SEARCHES_POSITIONS=$POSITIONS
if [ "$POSITIONS" = "on" ]; then
  export SEARCHES_POSITIONS_THRESHOLD=$THRESHOLD_MODE
fi

nvidia-smi

# Bounce off the MIG-mode GPU that SLURM still hands out as a plain A100.
# Requires the submit to be dispatched with --requeue. See the script's header.
source "$AP_ROOT/hpc/batch_gpu/_gpu_preflight.sh"

echo "=========================================="
date
echo "Task:       $SLURM_ARRAY_TASK_ID"
echo "Cell:       searches/nautilus/imaging/$MODEL"
echo "Instrument: hst"
echo "Precision:  fp64"
echo "n_live:     $SEARCHES_NAUTILUS_N_LIVE (2x fiducial $FIDUCIAL)"
echo "Positions:  $POSITIONS ${THRESHOLD_MODE:+(threshold=$THRESHOLD_MODE)}"

cd $AP_ROOT
python3 scripts/imaging/searches/nautilus/${MODEL}.py \
    --instrument hst \
    --config-name $SEARCHES_CONFIG_NAME \
    --output-dir $AP_ROOT/results/searches/nautilus/imaging/${MODEL}/hst

echo "Finished."
date
