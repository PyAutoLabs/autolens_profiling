"""Exercise real CLI declarations and timing-stream helpers without profiling."""

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT))
from _profile_cli import parse_profile_cli

CELLS = ROOT / "scripts/imaging/likelihood_breakdown"


def cell_parser(name):
    """Execute only argparse declarations, never the module-level benchmark."""
    tree = ast.parse((CELLS / name).read_text())
    statements = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_cell_parser" for t in node.targets
        ):
            statements.append(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "_cell_parser"
        ):
            statements.append(node)
    scope = {"argparse": argparse, "_argparse": argparse, "ALL_STREAM_KEYS": ("iid",)}
    exec(compile(ast.Module(body=statements, type_ignores=[]), name, "exec"), scope)
    return scope["_cell_parser"]


@pytest.mark.parametrize(
    "name",
    sorted(
        p.name
        for p in CELLS.glob("fixed_light*.py")
        if "_cli.parse_cell_args(_cell_parser)" in p.read_text()
    ),
)
def test_shared_and_local_flags_survive_composition(name):
    argv = ["--config-name", "test", "--source-pixels", "500", "--mesh", "delaunay"]
    cli = parse_profile_cli(argv=argv)
    result = cli.parse_cell_args(cell_parser(name), argv=argv)
    assert cli.config_name == result.config_name == "test"
    assert cli.source_pixels == result.source_pixels == 500
    assert result.mesh == "delaunay"


@pytest.mark.parametrize("flag", ["--vmap-batch", "--n-repeatz", "--trace-call", "--source-pixel"])
def test_old_trace_rejects_unknown_and_abbreviated_options(flag, capsys):
    argv = ["--mesh", "delaunay", flag, "16"]
    cli = parse_profile_cli(argv=argv)
    with pytest.raises(SystemExit) as error:
        cli.parse_cell_args(cell_parser("fixed_light_trace.py"), argv=argv)
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_batching_branch_can_declare_its_own_flags():
    parser = cell_parser("fixed_light_trace.py")
    parser.add_argument("--vmap-batch", type=int)
    argv = ["--mesh", "delaunay", "--vmap-batch", "16", "--source-pixels", "500"]
    cli = parse_profile_cli(argv=argv)
    assert cli.parse_cell_args(parser, argv=argv).vmap_batch == 16


@pytest.mark.parametrize("flag", ["--threads", "--n-repeats", "--source-pixels"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cpu_rejects_invalid_counts(flag, value):
    argv = [flag, value]
    with pytest.raises(SystemExit) as error:
        parse_profile_cli(argv=argv).parse_cell_args(cell_parser("fixed_light_numba.py"), argv=argv)
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--n-random", "--walk-max-evals"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_draw_population_and_walk_limit_must_be_positive(flag, value):
    argv = ["--mesh", "delaunay", flag, value]
    with pytest.raises(SystemExit) as error:
        parse_profile_cli(argv=argv).parse_cell_args(cell_parser("fixed_light_draws.py"), argv=argv)
    assert error.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1,0", "", "two"])
def test_pass_budget_population_must_be_nonempty_and_positive(value):
    argv = ["--mesh", "delaunay", f"--budgets={value}"]
    with pytest.raises(SystemExit) as error:
        parse_profile_cli(argv=argv).parse_cell_args(cell_parser("fixed_light_draws.py"), argv=argv)
    assert error.value.code == 2


def test_shared_only_legacy_parser_remains_staged():
    assert parse_profile_cli(argv=["--future-cell-option", "value"]).source_pixels is None


def stream_helpers():
    tree = ast.parse((CELLS / "fixed_light_numba.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_prepare_timed_stream", "_validate_timed_stream", "_one_call")
    ]
    cleared = []
    scope = {
        "_call_index": {"next": 99},
        "_clear_memos": lambda: cleared.append(True),
        "_instance_for": lambda route, index: index,
        "INSTANCE_MODE": "iid",
        "_IID_SEED": 263,
        "instances": list(range(8)),
        "time": SimpleNamespace(perf_counter=lambda: 0.0),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "stream", "exec"), scope)
    return scope, cleared


def test_unequal_warmups_produce_identical_timed_draws():
    scope, cleared = stream_helpers()
    rows, sequences = {}, []
    for route, warmup_end in [("b", 53), ("b_direct", 117), ("d_perm", 22)]:
        seen = []
        analysis = SimpleNamespace(
            log_likelihood_function=lambda instance, seen=seen: seen.append(instance) or 1.0
        )
        scope["_call_index"]["next"] = warmup_end
        stream = scope["_prepare_timed_stream"](analysis, route, 12)
        for _ in range(12):
            scope["_one_call"](analysis, route)
        scope["_validate_timed_stream"](stream, scope["_call_index"]["next"], rows)
        rows[route] = {"timed_stream": stream}
        sequences.append(seen)
    assert len(cleared) == 3
    assert sequences[0] == sequences[1] == sequences[2] == [0] + [i % 8 for i in range(1, 13)]


def test_mismatched_or_truncated_stream_cannot_be_published():
    scope, _ = stream_helpers()
    stream = scope["_prepare_timed_stream"](
        SimpleNamespace(log_likelihood_function=lambda **kw: 0), "b", 8
    )
    with pytest.raises(ValueError, match="number of calls"):
        scope["_validate_timed_stream"](stream, 8, {})
    with pytest.raises(ValueError, match="different timed"):
        scope["_validate_timed_stream"](
            stream, 9, {"b": {"timed_stream": dict(stream, iid_seed=999)}}
        )
