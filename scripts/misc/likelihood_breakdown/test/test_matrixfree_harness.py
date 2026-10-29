"""The matrix-free experiment harness (likelihood-lifecycle 2b): AST and smoke level only.

No dataset is loaded. The script must pin threads before numpy, refuse a library without the
matrix-free route, never write a current row (``experiments/`` or ``scans/`` only, scratch
fallback under ``output/``) and write no parity witness.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/interferometer/rectangular/likelihood_runtime_matrixfree.py"


def _source() -> str:
    return SCRIPT.read_text()


def test_threads_pinned_before_numpy_import():
    source = _source()
    assert source.index("_pin_thread_env(1)") < source.index("import numpy as np")


def test_refuses_without_the_matrix_free_route():
    source = _source()
    assert 'hasattr(aa, "InversionInterferometerMatrixFree")' in source
    assert 'interferometer_route="matrix_free"' in source
    assert source.count("REFUSING") >= 2


def test_model_is_the_only_supported_matrix_free_model():
    source = _source()
    assert "al.mesh.RectangularUniform(shape=(side, side))" in source
    assert "al.reg.MaternSparse(**MATERN)" in source
    assert '"use_edge_zeroed_pixels": False' in source
    assert 'interferometer_route="fft"' in source  # the dense reference arm


def test_rows_are_experiments_or_scans_never_current_and_no_parity():
    source = _source()
    assert "write_parity_json" not in source and "write_interferometer_parity" not in source
    tree = ast.parse(source)
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert "numpy_cpu_matrixfree" in strings
    assert any("p2b_matrixfree_local" in str(s) for s in strings)
    assert "experiments/{EXPERIMENT}" in source and "scans/{EXPERIMENT}_{side}x{side}" in source


def test_smoke_mode_exits_after_imports():
    env = {**os.environ, "AUTOLENS_PROFILING_SMOKE": "1"}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--instrument", "sma", "--mesh-side", "35"],
        env=env,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "[smoke]" in out.stdout
