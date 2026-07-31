#!/bin/bash
#
# PyAutoLens#678 phase B — point-source defaults evidence campaign.
#
# Submits every galaxy- and cluster-tier A100 search cell for the campaign in
# one command: galaxy simple tier, then the pairing-discriminator cells, then
# the near-caustic cells, then the cluster tier. Run from the hpc/batch_gpu
# directory on the RAL HPC login node.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- Pre-generate every dataset the cells need, BEFORE any sbatch. ---
# The datasets are gitignored and auto-simulated on first use; 25 concurrent
# jobs would otherwise race the auto-simulate existence check and corrupt the
# shared dataset dirs. Simulators are seeded, so this is deterministic.
AP_ROOT=${AP_ROOT:-/mnt/ral/jnightin/autolens_profiling}
source "$AP_ROOT/activate.sh"
export NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib
for inst in simple simple_missing simple_extra near_caustic; do
    if [ ! -f "$AP_ROOT/dataset/point_source/$inst/point_dataset_positions_only.json" ]; then
        echo "Pre-generating point_source/$inst"
        (cd "$AP_ROOT" && python3 scripts/misc/simulators/point_source.py --instrument "$inst")
    fi
done
if [ ! -f "$AP_ROOT/dataset/cluster/simple/point_datasets.csv" ]; then
    echo "Pre-generating cluster/simple"
    (cd "$AP_ROOT" && python3 scripts/misc/simulators/cluster.py)
fi

# --- Galaxy tier (simple) — nautilus ---
echo "Submitting submit_search_nautilus_point_source_image_plane_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_image_plane_a100_simple_fp64
echo "Submitting submit_search_nautilus_point_source_source_plane_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_source_plane_a100_simple_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_solved_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_image_plane_solved_a100_simple_fp64
echo "Submitting submit_search_nautilus_point_source_source_plane_solved_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_source_plane_solved_a100_simple_fp64
echo "Submitting submit_search_nautilus_point_source_source_plane_tensor_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_source_plane_tensor_a100_simple_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_fp64"
sbatch submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_fp64

# --- Galaxy tier (simple) — multi_start_prodigy ---
echo "Submitting submit_search_multi_start_prodigy_point_source_image_plane_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_point_source_image_plane_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_point_source_source_plane_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_point_source_source_plane_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_point_source_image_plane_solved_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_point_source_image_plane_solved_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_point_source_source_plane_solved_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_point_source_source_plane_solved_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_point_source_source_plane_tensor_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_point_source_source_plane_tensor_a100_simple_fp64

# --- Discriminator cells (nautilus only) — simple_missing / simple_extra ---
echo "Submitting submit_search_nautilus_point_source_image_plane_solved_a100_simple_missing_fp64"
sbatch submit_search_nautilus_point_source_image_plane_solved_a100_simple_missing_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_missing_fp64"
sbatch submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_missing_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_solved_a100_simple_extra_fp64"
sbatch submit_search_nautilus_point_source_image_plane_solved_a100_simple_extra_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_extra_fp64"
sbatch submit_search_nautilus_point_source_image_plane_repeat_solved_a100_simple_extra_fp64

# --- Near-caustic cells (nautilus only) ---
echo "Submitting submit_search_nautilus_point_source_source_plane_tensor_a100_near_caustic_fp64"
sbatch submit_search_nautilus_point_source_source_plane_tensor_a100_near_caustic_fp64
echo "Submitting submit_search_nautilus_point_source_source_plane_solved_a100_near_caustic_fp64"
sbatch submit_search_nautilus_point_source_source_plane_solved_a100_near_caustic_fp64
echo "Submitting submit_search_nautilus_point_source_image_plane_solved_a100_near_caustic_fp64"
sbatch submit_search_nautilus_point_source_image_plane_solved_a100_near_caustic_fp64

# --- Cluster tier (simple) — nautilus ---
echo "Submitting submit_search_nautilus_cluster_source_plane_solved_a100_simple_fp64"
sbatch submit_search_nautilus_cluster_source_plane_solved_a100_simple_fp64
echo "Submitting submit_search_nautilus_cluster_source_plane_tensor_a100_simple_fp64"
sbatch submit_search_nautilus_cluster_source_plane_tensor_a100_simple_fp64
echo "Submitting submit_search_nautilus_cluster_source_plane_a100_simple_fp64"
sbatch submit_search_nautilus_cluster_source_plane_a100_simple_fp64
echo "Submitting submit_search_nautilus_cluster_image_plane_solved_a100_simple_fp64"
sbatch submit_search_nautilus_cluster_image_plane_solved_a100_simple_fp64

# --- Cluster tier (simple) — multi_start_prodigy ---
echo "Submitting submit_search_multi_start_prodigy_cluster_source_plane_solved_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_cluster_source_plane_solved_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_cluster_source_plane_tensor_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_cluster_source_plane_tensor_a100_simple_fp64
echo "Submitting submit_search_multi_start_prodigy_cluster_image_plane_solved_a100_simple_fp64"
sbatch submit_search_multi_start_prodigy_cluster_image_plane_solved_a100_simple_fp64
