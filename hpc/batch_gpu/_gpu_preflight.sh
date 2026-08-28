#!/bin/bash
# GPU preflight — source this from a batch_gpu submit, immediately after its
# `nvidia-smi` call and before any Python.
#
# WHY IT EXISTS (2026-08-26). One A100 on `euclid-ral-gpu-1` (PCI 07:00.0) was
# switched into MIG mode with **no MIG instances created**, while SLURM went on
# advertising it as a plain `gpu:A100` (`scontrol show node` still reports
# `Gres=gpu:A100:4`, and there is no MIG-aware gres type configured). A job
# allocated that GPU gets a device it cannot use: `cuInit(0)` returns
# `CUDA_ERROR_NO_DEVICE` and JAX dies with
#
#     RuntimeError: Unable to initialize backend 'cuda'
#
# about four seconds in. SLURM records the task as COMPLETED, so an array of
# them looks like a clean fast run rather than a total loss — job 341845 burned
# 14 of 15 arms that way, and the tell was only visible in each task's own
# `nvidia-smi` header (`MIG M. Enabled`, GPU-Util `N/A`).
#
# Since SLURM cannot be asked to avoid that GPU, bounce off it: requeue and let
# the scheduler try again elsewhere. Requeues are cheap; a burned arm is not.
#
# The submit must be dispatched with `--requeue` for this to work.
#
# TO RETIRE: when the RAL admins either create MIG instances on that GPU or
# take it out of MIG mode, this becomes a no-op that costs one `nvidia-smi`
# call per task. Check with `nvidia-smi --query-gpu=mig.mode.current` inside a
# job before removing it — a silent removal re-opens the trap.

_GPU_PREFLIGHT_MAX_REQUEUE="${GPU_PREFLIGHT_MAX_REQUEUE:-60}"
_GPU_PREFLIGHT_BACKOFF_S="${GPU_PREFLIGHT_BACKOFF_S:-45}"

_mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')

if [ "$_mig_mode" = "Enabled" ]; then
    echo "PREFLIGHT: allocated GPU has MIG enabled with no usable instances."
    echo "PREFLIGHT: $(nvidia-smi --query-gpu=pci.bus_id,name --format=csv,noheader 2>/dev/null | head -1)"
    echo "PREFLIGHT: restart count = ${SLURM_RESTART_COUNT:-0} / ${_GPU_PREFLIGHT_MAX_REQUEUE}"
    if [ "${SLURM_RESTART_COUNT:-0}" -lt "$_GPU_PREFLIGHT_MAX_REQUEUE" ]; then
        sleep "$_GPU_PREFLIGHT_BACKOFF_S"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    echo "PREFLIGHT: exhausted ${_GPU_PREFLIGHT_MAX_REQUEUE} requeues — the bad GPU is"
    echo "PREFLIGHT: catching most allocations. Report it to the RAL admins rather"
    echo "PREFLIGHT: than raising the cap again."
    exit 1
fi

echo "PREFLIGHT: GPU usable (MIG mode: ${_mig_mode:-not-reported})."
