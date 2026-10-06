"""Compatibility entry point; use scripts/interferometer/delaunay/quick_update.py."""

import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(_root))
from _script_routes import run_legacy

run_legacy(__file__, globals())
