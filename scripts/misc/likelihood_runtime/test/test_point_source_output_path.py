"""The point-source runtime cells place their row through ``resolve_output_paths``.

Both scripts once called ``place_output_path`` directly and so ignored ``--output-path`` and
``--output-dir`` (autolens_profiling#454 item 1). The scripts run a profile on import, so they are
read as source here, never imported.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import _profile_cli as pc  # noqa: E402

CELLS = {
    "point_source_image": "image_plane",
    "point_source_source": "source_plane",
}


def _calls(path: Path) -> dict[str, list[ast.Call]]:
    calls: dict[str, list[ast.Call]] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)
    return calls


def test_scripts_resolve_their_row_from_the_cli():
    for cls, model in CELLS.items():
        script = ROOT / "scripts" / cls / model / "likelihood_runtime.py"
        calls = _calls(script)
        assert "place_output_path" not in calls, script
        (call,) = calls["resolve_output_paths"]
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "_cli", script


def test_default_placement_is_unchanged():
    # The row lands where ``place_output_path`` on the legacy name put it before the change.
    for argv in ([], ["--use-mixed-precision"]):
        cli = pc.parse_profile_cli(argv=argv)
        for cls, model in CELLS.items():
            legacy_dir = ROOT / "results" / "runtime" / cls / model
            basename = f"{model}_summary_v2026.10.23.1"
            hints = {"dataset": "simple", "instrument": "simple"}
            doc = pc._setup_hints(cli, basename, hints)
            before = pc.placed_path(legacy_dir / f"{basename}.json", doc)
            legacy, _ = pc.resolve_output_paths(cli, legacy_dir, basename, hints=hints, place=False)
            assert pc.placed_path(legacy, doc) == before
            assert before.name == "runtime.json"
            assert f"/{cls}/{model}/simple/" in before.as_posix()


def test_output_path_and_output_dir_are_honoured(tmp_path):
    for cls, model in CELLS.items():
        legacy_dir = ROOT / "results" / "runtime" / cls / model
        basename = f"{model}_summary_v2026.10.23.1"
        target = tmp_path / cls / "runtime.json"
        cli = pc.parse_profile_cli(argv=["--output-path", str(target)])
        json_path, png_path = pc.resolve_output_paths(cli, legacy_dir, basename, hints={})
        assert json_path == target.resolve() and png_path == target.resolve().with_suffix(".png")
        cli = pc.parse_profile_cli(argv=["--output-dir", str(tmp_path / "d" / cls)])
        json_path, _ = pc.resolve_output_paths(cli, legacy_dir, basename, hints={})
        assert json_path == (tmp_path / "d" / cls / f"{basename}.json").resolve()
