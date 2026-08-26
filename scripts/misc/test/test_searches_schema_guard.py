"""Unit tests for the eval-counter comparability guard (issue #177).

``performance.likelihood_evals`` changed MEANING between results schema v1
and v2 — but only for ``MultiStart*`` searches. These tests pin both halves
of that: the guard must fire on a v1/v2 MultiStart pair, and must NOT fire on
a v1/v2 NESTED pair, which is legitimately comparable and exists on ``main``
(``nautilus/imaging/pixelization/hst`` holds a v1 row at 58,464 evals beside
a v2 row at 55,984).

Run::

    cd autolens_profiling
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python -m pytest scripts/misc/test/test_searches_schema_guard.py -q
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_ROOT = _profiling_root()
_misc_dir = str(_ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json  # noqa: E402

from searches._metrics import (  # noqa: E402
    EVAL_BASIS_REJECT_INCLUSIVE,
    EVAL_BASIS_STORED_ONLY,
    EVAL_BASIS_UNKNOWN,
    basis_conflicts,
    eval_counter_basis,
)

# The real numbers from the 2026-08-25 A100 harvest, one cell directory, one
# Prodigy n256 configuration — the pair that motivated the guard.
_V1_MULTISTART_EVALS = 257
_V1_MULTISTART_PER_EVAL_MS = 874.5818861727585
_V2_MULTISTART_EVALS = 247808
_V2_MULTISTART_PER_EVAL_MS = 2.233140355299327


def _summary(
    *,
    sampler: str,
    schema_version: int | None,
    likelihood_evals: int,
    time_per_eval_ms: float,
    config_name: str,
    stored_samples: int | None = None,
) -> dict:
    payload = {
        "sampler": sampler,
        "dataset_class": "imaging",
        "model": "mge",
        "instrument": "hst",
        "config_name": config_name,
        "version": "2026.8.17.1",
        "device": {"backend": "gpu"},
        "results": {"max_log_likelihood": 31787.9, "log_evidence": float("nan")},
        "performance": {
            "total_wall_s": 224.7,
            "viz_wall_s": 0.0,
            "sampler_wall_s": 224.7,
            "likelihood_evals": likelihood_evals,
            "time_per_eval_ms": time_per_eval_ms,
            "stored_samples": stored_samples,
        },
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    return payload


def _v1_multistart(config_name: str = "n256_seed0") -> dict:
    return _summary(
        sampler="multi_start_prodigy_autoconv",
        schema_version=None,  # the key did not exist in v1
        likelihood_evals=_V1_MULTISTART_EVALS,
        time_per_eval_ms=_V1_MULTISTART_PER_EVAL_MS,
        config_name=config_name,
    )


def _v2_multistart(config_name: str = "n256_seed0_pos_t0.3_f1e8") -> dict:
    return _summary(
        sampler="multi_start_prodigy_autoconv",
        schema_version=2,
        likelihood_evals=_V2_MULTISTART_EVALS,
        time_per_eval_ms=_V2_MULTISTART_PER_EVAL_MS,
        config_name=config_name,
        stored_samples=257,
    )


def _v1_nested(config_name: str = "hpc_a100_fp64") -> dict:
    return _summary(
        sampler="nautilus",
        schema_version=None,
        likelihood_evals=58464,
        time_per_eval_ms=46.54,
        config_name=config_name,
    )


def _v2_nested(config_name: str = "hpc_hpc_a100_fp64") -> dict:
    return _summary(
        sampler="nautilus",
        schema_version=2,
        likelihood_evals=55984,
        time_per_eval_ms=58.83,
        config_name=config_name,
        stored_samples=55984,
    )


# ---------------------------------------------------------------------------
# eval_counter_basis
# ---------------------------------------------------------------------------


def test__basis__v1_multistart_is_stored_only():
    assert eval_counter_basis(_v1_multistart()) == EVAL_BASIS_STORED_ONLY


def test__basis__v2_multistart_is_reject_inclusive():
    assert eval_counter_basis(_v2_multistart()) == EVAL_BASIS_REJECT_INCLUSIVE


def test__basis__v1_nested_is_reject_inclusive():
    """total_samples already counted rejected proposals for a nested sampler."""
    assert eval_counter_basis(_v1_nested()) == EVAL_BASIS_REJECT_INCLUSIVE


def test__basis__v2_nested_is_reject_inclusive():
    assert eval_counter_basis(_v2_nested()) == EVAL_BASIS_REJECT_INCLUSIVE


def test__basis__missing_schema_version_is_treated_as_v1():
    """A missing key must read as v1, never as 'unknown' or as v2."""
    payload = _v1_multistart()
    assert "schema_version" not in payload
    assert eval_counter_basis(payload) == EVAL_BASIS_STORED_ONLY

    explicit_v1 = dict(payload, schema_version=1)
    assert eval_counter_basis(explicit_v1) == EVAL_BASIS_STORED_ONLY


def test__basis__non_search_payload_is_unknown():
    """No `sampler` key = not a search run (e.g. the nan-accounting study)."""
    assert eval_counter_basis({"performance": {"likelihood_evals": 10}}) == EVAL_BASIS_UNKNOWN


def test__basis__every_multi_start_variant_is_caught():
    for sampler in (
        "multi_start_prodigy",
        "multi_start_prodigy_autoconv",
        "multi_start_adam",
        "multi_start_nan_accounting",
    ):
        payload = dict(_v1_multistart(), sampler=sampler)
        assert eval_counter_basis(payload) == EVAL_BASIS_STORED_ONLY, sampler


# ---------------------------------------------------------------------------
# basis_conflicts
# ---------------------------------------------------------------------------


def test__conflicts__multistart_v1_and_v2_conflict():
    conflicts = basis_conflicts({"off": _v1_multistart(), "on": _v2_multistart()})
    assert set(conflicts) == {EVAL_BASIS_STORED_ONLY, EVAL_BASIS_REJECT_INCLUSIVE}
    assert conflicts[EVAL_BASIS_STORED_ONLY] == ["off"]
    assert conflicts[EVAL_BASIS_REJECT_INCLUSIVE] == ["on"]


def test__conflicts__nested_v1_and_v2_do_NOT_conflict():
    """The control. A literal `schema_version`-differs guard fails this.

    Both rows count reject-inclusive evaluations, so their per-eval figures
    are directly comparable and must keep rendering.
    """
    assert basis_conflicts({"v1": _v1_nested(), "v2": _v2_nested()}) == {}


def test__conflicts__single_basis_is_empty():
    assert basis_conflicts({"a": _v1_multistart("a"), "b": _v1_multistart("b")}) == {}


# ---------------------------------------------------------------------------
# aggregate.py
# ---------------------------------------------------------------------------


def _write_cell(tmp_path, payloads: dict[str, dict]):
    cell = tmp_path / "sampler" / "imaging" / "mge" / "hst"
    cell.mkdir(parents=True)
    for name, payload in payloads.items():
        (cell / f"{name}.json").write_text(json.dumps(payload))
    return cell


def test__aggregate_cell__flags_mixed_bases_and_names_both_sides(tmp_path):
    from searches import aggregate

    cell = _write_cell(
        tmp_path, {"n256_seed0": _v1_multistart(), "n256_seed0_pos": _v2_multistart()}
    )
    comparison = aggregate._aggregate_cell(cell)

    conflicts = comparison["eval_counter_conflicts"]
    assert conflicts, "a v1 MultiStart row beside a v2 one must conflict"
    assert conflicts[EVAL_BASIS_STORED_ONLY] == ["n256_seed0"]
    assert conflicts[EVAL_BASIS_REJECT_INCLUSIVE] == ["n256_seed0_pos"]
    assert comparison["eval_counter_bases"]["n256_seed0"] == EVAL_BASIS_STORED_ONLY

    table = aggregate._render_table(comparison, "cell")
    assert "REFUSING to compare eval-derived metrics" in table
    # Both filenames must appear so the split is checkable by hand.
    assert "n256_seed0" in table and "n256_seed0_pos" in table
    # The misleading per-eval figures must be gone entirely, not annotated.
    assert "874.58ms" not in table
    assert "2.23ms" not in table


def test__aggregate_cell__nested_v1_v2_still_renders_per_eval(tmp_path):
    """Control at the aggregate level: no conflict, per-eval still rendered."""
    from searches import aggregate

    cell = _write_cell(tmp_path, {"hpc_a100_fp64": _v1_nested(), "hpc_hpc_a100_fp64": _v2_nested()})
    comparison = aggregate._aggregate_cell(cell)

    assert comparison["eval_counter_conflicts"] == {}
    table = aggregate._render_table(comparison, "cell")
    assert "REFUSING" not in table
    assert "46.54ms" in table
    assert "58.83ms" in table


def test__aggregate_main__exits_non_zero_on_a_mixed_cell(tmp_path, monkeypatch):
    """A sweep must not pass over a cell whose per-eval metrics were withheld."""
    from searches import aggregate

    _write_cell(tmp_path, {"n256_seed0": _v1_multistart(), "n256_seed0_pos": _v2_multistart()})
    monkeypatch.setattr(_sys, "argv", ["aggregate.py", "--output-root", str(tmp_path)])
    assert aggregate.main() == 3


def test__aggregate_main__exits_zero_on_a_single_basis_cell(tmp_path, monkeypatch):
    from searches import aggregate

    _write_cell(tmp_path, {"hpc_a100_fp64": _v1_nested(), "hpc_hpc_a100_fp64": _v2_nested()})
    monkeypatch.setattr(_sys, "argv", ["aggregate.py", "--output-root", str(tmp_path)])
    assert aggregate.main() == 0


# ---------------------------------------------------------------------------
# build_readme.py searches table
# ---------------------------------------------------------------------------


def _artifact(tmp_path, payload: dict, name: str):
    from tooling.build_readme import SearchArtifact

    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return SearchArtifact(
        path=path,
        sampler=payload["sampler"],
        cell="imaging/mge/hst",
        config=payload["config_name"],
        version=(2026, 8, 17, 1),
        raw_version="2026.8.17.1",
    )


def test__searches_table__withholds_evals_for_a_v1_multistart_row(tmp_path):
    from tooling.build_readme import _render_searches_table

    table = _render_searches_table([_artifact(tmp_path, _v1_multistart(), "v1")])
    assert "| stored |" in table
    # Neither the storage count posing as evals nor its per-eval figure.
    assert "257" not in table
    assert "874.6 ms" not in table


def test__searches_table__renders_evals_for_a_v2_multistart_row(tmp_path):
    from tooling.build_readme import _render_searches_table

    table = _render_searches_table([_artifact(tmp_path, _v2_multistart(), "v2")])
    assert "| evals |" in table
    assert "247,808" in table
    assert "2.2 ms" in table


def test__searches_table__v1_nested_row_keeps_its_evals(tmp_path):
    """Control: a v1 NESTED row is comparable and must not be withheld."""
    from tooling.build_readme import _render_searches_table

    table = _render_searches_table([_artifact(tmp_path, _v1_nested(), "v1n")])
    assert "| evals |" in table
    assert "58,464" in table


# ---------------------------------------------------------------------------
# aggregate.py cell discovery — the guard is useless if the cell is never
# discovered, and MultiStart cells were not (see _is_search_payload).
# ---------------------------------------------------------------------------


def test__discover_cells__finds_a_cell_named_outside_CONFIG_ORDER(tmp_path):
    """Real sweep arms are named e.g. `hpc_hpc_a100_fp64_n256_seed0.json`."""
    from searches import aggregate

    _write_cell(tmp_path, {"hpc_hpc_a100_fp64_n256_seed0": _v1_multistart()})
    assert aggregate._discover_cells(tmp_path) == [("sampler", "imaging", "mge", "hst")]


def test__discover_cells__ignores_comparison_json_and_non_search_payloads(tmp_path):
    from searches import aggregate

    cell = tmp_path / "sampler" / "imaging" / "mge" / "hst"
    cell.mkdir(parents=True)
    (cell / "comparison.json").write_text(json.dumps({"configs": {}}))
    # No `sampler` key -> not a search run (e.g. the nan-accounting study).
    (cell / "overhead_study.json").write_text(json.dumps({"performance": {}}))
    assert aggregate._discover_cells(tmp_path) == []
