#!/usr/bin/env bash
# Build one throwaway venv per jax version holding only jax + jaxlib (and numpy via deps).
# Never touches ~/venv/PyAuto313 or the shared RAL install (issue PyAutoLabs/PyAutoNerves#213).
#
#   bash make_venvs.sh <base-dir> 0.9.2 0.10.2 0.11.2 0.12.0 0.12.0.1
#
# Each venv lands at <base-dir>/jax-<version>; an existing one is reused.
set -euo pipefail
base="$1"; shift
py="${PYTHON:-python3}"
mkdir -p "$base"
for v in "$@"; do
  dir="$base/jax-$v"
  if [ ! -x "$dir/bin/python" ]; then
    "$py" -m venv "$dir"
    "$dir/bin/python" -m pip install -q --upgrade pip
    "$dir/bin/python" -m pip install -q "jax==$v" "jaxlib==$v" flatbuffers
  fi
  "$dir/bin/python" -I -c "import jax, jaxlib; print('$dir', jax.__version__, jaxlib.__version__)"
done
