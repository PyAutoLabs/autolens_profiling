"""Static contract for the five staged models in the resume profiler.

The profiler executes at module scope, so importing it would start a pipeline.
Its model-construction functions are checked through their AST instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "ruff.toml").exists())
SCRIPT = ROOT / "scripts" / "misc" / "pipeline_resume" / "slam_resume.py"
STAGES = ("source_lp", "source_pix_1", "source_pix_2", "light_lp", "mass_total")


def _stage_nodes():
    tree = ast.parse(SCRIPT.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return [functions[name] for name in STAGES]


def _is_attr(node, *parts):
    for part in reversed(parts):
        if not isinstance(node, ast.Attribute) or node.attr != part:
            return False
        node = node.value
    return isinstance(node, ast.Name) and node.id == "al"


def test_every_stage_uses_a_root_field_and_no_galaxy_attached_shear():
    for stage in _stage_nodes():
        model_collection = next(
            node.value
            for node in ast.walk(stage)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "model" for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        )
        assert any(keyword.arg == "fields" for keyword in model_collection.keywords), stage.name

        galaxy_models = [
            node
            for node in ast.walk(stage)
            if isinstance(node, ast.Call) and node.args and _is_attr(node.args[0], "Galaxy")
        ]
        assert galaxy_models
        assert all(
            all(keyword.arg != "shear" for keyword in model.keywords) for model in galaxy_models
        ), stage.name


def test_free_and_fixed_field_handoffs_match_the_pipeline_contract():
    stages = {node.name: ast.unparse(node) for node in _stage_nodes()}

    assert "al.MassField" in stages["source_lp"]
    assert "fields_result=source_lp_result.model.fields" in stages["source_pix_1"]
    assert "fields=source_pix_result_1.instance.fields" in stages["source_pix_2"]
    assert "fields=source_result_for_lens.instance.fields" in stages["light_lp"]
    assert "fields_result=source_result_for_lens.model.fields" in stages["mass_total"]
    assert stages["source_pix_1"].count("mass_and_fields_from") == 1
    assert stages["mass_total"].count("mass_and_fields_from") == 1
