#!/usr/bin/env bash
# Activate the shared PyAuto venv and point PYTHONPATH at the canonical
# PyAuto* source checkouts on the HPC.
#
# Mirrors the pattern used by z_projects/euclid/activate.sh and
# autolens_assistant/activate.sh: pip handles only third-party deps in
# the venv, while the five PyAuto* libraries are resolved via PYTHONPATH
# directly from /mnt/ral/jnightin/PyAuto/<repo>. ``HPCPullPyAuto`` then
# becomes the only mechanism needed to keep PyAuto* current — no pip
# install of PyAuto* into the venv ever.
#
# Usage (inside a SLURM submit or interactive shell):
#
#     source /mnt/ral/jnightin/autolens_profiling/activate.sh
#     python3 searches/nautilus/imaging/mge.py ...

BASE=/mnt/ral/jnightin/PyAuto

source "$BASE/PyAuto/bin/activate"

export PYTHONPATH=$BASE:\
$BASE/PyAutoConf:\
$BASE/PyAutoFit:\
$BASE/PyAutoArray:\
$BASE/PyAutoGalaxy:\
$BASE/PyAutoLens
