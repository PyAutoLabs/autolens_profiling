"""Unit tests for the phase-5 switches on the fixed-lens-light library cell (#259).

Phase 5 is the epic's verdict: the whole library likelihood call on HST **and
Euclid**, at three source-pixel counts, on four hardware legs. Three switches on
``scripts/imaging/likelihood_breakdown/fixed_light_library.py`` make that grid
runnable, and each one can silently produce a *different* table than the one the
note claims:

- ``--dataset {hst,euclid}`` — if its default ever drifts off ``hst``, every
  phase-1 and phase-2 artifact this cell owns is being overwritten by a
  different dataset under the same name.
- ``--pass-budget auto`` — if it resolved to the budget that *certifies* rather
  than the budget phase 3 proved safe, the note would quote a production cost
  that falls back on two thirds of a real search's evaluations.
- ``--routes`` — if an unknown token were dropped rather than raised, a leg
  would quietly run fewer rows than the table says it did.

The cell cannot be imported (module level runs the whole profile), so the three
helpers and the constants are lifted out of its AST and executed here. That is
the real code, not a copy of it — a change to the cell that these tests do not
see is a change that did not happen.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_verdict.py
"""

from __future__ import annotations

import ast
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
CELL_PATH = ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_library.py"

#: Names lifted from the cell's module body and executed in one namespace: the
#: three helpers plus the constants they close over.
_LIFTED_FUNCTIONS = ("_parse_routes", "_resolve_pass_budget", "_smallest_certifying_budget")
_LIFTED_CONSTANTS = ("CERTIFYING_BUDGET", "PHASE3_SAFE_BUDGET", "ALL_ROUTE_KEYS")


def _cell_namespace() -> dict:
    """Execute the cell's helpers + constants, and nothing else, in a fresh dict."""
    source = CELL_PATH.read_text()
    tree = ast.parse(source)
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _LIFTED_FUNCTIONS:
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in _LIFTED_CONSTANTS for name in targets):
                wanted.append(node)
    namespace: dict = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(CELL_PATH), "exec"), namespace)
    missing = [name for name in (*_LIFTED_FUNCTIONS, *_LIFTED_CONSTANTS) if name not in namespace]
    if missing:
        raise AssertionError(f"the cell no longer defines {missing} at module level")
    return namespace


@pytest.fixture(scope="module")
def cell() -> dict:
    return _cell_namespace()


@pytest.fixture(scope="module")
def cell_source() -> str:
    return CELL_PATH.read_text()


# ---------------------------------------------------------------------------
# --pass-budget auto — the production budget, with the certifying one recorded
# ---------------------------------------------------------------------------


def test_auto_resolves_to_phase_3_safe_budget_per_mesh(cell):
    """11 rectangular / 7 Delaunay — the zero-fallback budgets, not 7 / 2."""
    for mesh, expected in (("rectangular", 11), ("delaunay", 7), ("delaunay_nn", 7)):
        safe = cell["PHASE3_SAFE_BUDGET"][mesh]
        assert safe == expected
        budget, mode = cell["_resolve_pass_budget"]("auto", mesh, safe)
        assert budget == expected
        assert "auto" in mode and "safe budget" in mode


def test_auto_is_never_the_phase_0_fiducial_budget(cell):
    """Phase 3: the fiducial budgets fall back on 27.5 % / 67.5 % of a draw set."""
    for mesh in ("rectangular", "delaunay"):
        auto, _ = cell["_resolve_pass_budget"]("auto", mesh, cell["PHASE3_SAFE_BUDGET"][mesh])
        assert auto > cell["CERTIFYING_BUDGET"][mesh]


@pytest.mark.parametrize("raw", ["auto", "AUTO", " auto "])
def test_auto_is_case_and_whitespace_insensitive(cell, raw):
    budget, _ = cell["_resolve_pass_budget"](raw, "delaunay", 7)
    assert budget == 7


def test_no_pass_budget_keeps_the_phase_0_default(cell):
    """A leg that names no budget runs exactly what phases 0-2 ran."""
    for mesh, expected in (("rectangular", 7), ("delaunay", 2), ("delaunay_nn", 2)):
        budget, mode = cell["_resolve_pass_budget"](None, mesh, 11)
        assert budget == expected
        assert "phase0" in mode


def test_an_explicit_budget_wins_over_both(cell):
    budget, mode = cell["_resolve_pass_budget"](5, "delaunay", 7)
    assert (budget, mode) == (5, "explicit")
    assert cell["_resolve_pass_budget"]("5", "rectangular", 11)[0] == 5


def test_a_nonsense_budget_is_an_error_not_a_default(cell):
    with pytest.raises(ValueError, match="integer or 'auto'"):
        cell["_resolve_pass_budget"]("eleven", "delaunay", 7)


def test_safe_budget_override_is_honoured(cell):
    """``--safe-budget`` is the lever a later phase re-derives the budget with."""
    assert cell["_resolve_pass_budget"]("auto", "delaunay", 9)[0] == 9


# ---------------------------------------------------------------------------
# the certifying budget the sweep measures — recorded beside the production one
# ---------------------------------------------------------------------------


def test_smallest_certifying_budget_is_the_first_true_row(cell):
    table = [
        {"pass_budget": 1, "certified": False},
        {"pass_budget": 2, "certified": False},
        {"pass_budget": 3, "certified": True},
        {"pass_budget": 4, "certified": True},
    ]
    assert cell["_smallest_certifying_budget"](table) == 3


def test_a_sweep_that_never_certifies_records_none(cell):
    table = [{"pass_budget": n, "certified": False} for n in range(1, 13)]
    assert cell["_smallest_certifying_budget"](table) is None


def test_certifying_budget_is_read_from_the_table_not_assumed(cell):
    """Phase 4: the budget wanders with the model; a later row may certify first."""
    table = [
        {"pass_budget": 1, "certified": True},
        {"pass_budget": 2, "certified": False},
    ]
    assert cell["_smallest_certifying_budget"](table) == 1


def test_an_empty_sweep_is_none_not_an_error(cell):
    assert cell["_smallest_certifying_budget"]([]) is None


# ---------------------------------------------------------------------------
# --dataset — HST stays the default, byte for byte
# ---------------------------------------------------------------------------


def test_dataset_defaults_to_hst(cell_source):
    assert (
        '_cell_parser.add_argument("--dataset", choices=("hst", "euclid"), default="hst")'
        in cell_source
    )


def test_the_instrument_is_the_dataset_switch_and_nothing_else(cell_source):
    """The cell's one lever: no leg may profile HST while its log says Euclid."""
    assert "instrument = DATASET" in cell_source
    assert 'instrument = "hst"' not in cell_source


def test_a_disagreeing_shared_instrument_is_refused(cell_source):
    assert "does not select the dataset in this cell" in cell_source


def test_euclid_only_changes_the_preset_not_the_model(cell_source):
    """Mask radius, over-sampling and the MGE are dataset-independent."""
    assert "mask_radius = 3.5" in cell_source
    assert "sub_size_list=[4, 2, 2]" in cell_source
    assert "radial_list=[0.3, 0.6]" in cell_source
    assert "total_gaussians=60" in cell_source
    # Exactly one pixel-scale source, and it is the preset table.
    assert 'pixel_scale = INSTRUMENTS[instrument]["pixel_scale"]' in cell_source


def test_pins_are_not_asserted_off_hst(cell_source):
    """The phase-0 pins were calibrated on HST at the fiducial mesh, and say so."""
    assert (
        "PIN_IS_FIDUCIAL = (n_source_pixels == FIDUCIAL_SOURCE_PIXELS[MESH]) and DATASET"
        ' == "hst"' in cell_source
    )


def test_a_leg_without_pins_records_rather_than_skips(cell_source):
    assert '"reference_recorded": reference_recorded,' in cell_source
    assert '"status": "RECORDED",' in cell_source
    assert '"log_evidence_eager_s0"' in cell_source
    assert '"smallest_certifying_budget": _certifying_budget_measured,' in cell_source


def test_the_output_name_separates_datasets_and_sizes(cell_source):
    assert 'if DATASET != "hst":\n    _cell_name = f"{_cell_name}_{DATASET}"' in cell_source
    assert (
        "if SOURCE_PIXELS_REQUESTED is not None:\n"
        '    _cell_name = f"{_cell_name}_n{int(n_source_pixels)}"' in cell_source
    )


# ---------------------------------------------------------------------------
# --routes — what a leg runs is what the table says it ran
# ---------------------------------------------------------------------------


def test_default_selection_is_every_route(cell):
    assert cell["_parse_routes"](None, True) == ("a", "b", "c", "d", "d0", "e")


def test_the_phase_5_selection_is_four_rows_in_canonical_order(cell):
    assert cell["_parse_routes"]("a,b,c,d", True) == ("a", "b", "c", "d")


def test_route_order_is_canonical_not_the_order_typed(cell):
    assert cell["_parse_routes"]("d,a,c,b", True) == ("a", "b", "c", "d")


def test_whitespace_and_case_are_tolerated(cell):
    assert cell["_parse_routes"](" A , b ,C ", True) == ("a", "b", "c")


def test_an_unknown_route_is_an_error_not_a_dropped_row(cell):
    with pytest.raises(ValueError, match="unknown route"):
        cell["_parse_routes"]("a,b,z", True)


def test_an_empty_route_list_is_an_error(cell):
    with pytest.raises(ValueError, match="names no routes"):
        cell["_parse_routes"](" , ", True)


def test_no_fallback_row_still_owns_the_cond_shapes(cell):
    """``--no-fallback-row`` drops d and e however they were requested."""
    assert cell["_parse_routes"]("a,b,c,d", False) == ("a", "b", "c")
    assert cell["_parse_routes"](None, False) == ("a", "b", "c", "d0")


def test_routes_are_filtered_before_they_are_built(cell_source):
    """An unselected route is not compiled — that is where the leg time goes."""
    assert "ROUTES: list[tuple] = [_ROUTE_SPECS[_token] for _token in ROUTE_SELECTION]" in (
        cell_source
    )


def test_the_equivalence_pins_follow_the_selection(cell_source):
    assert 'if "d" in ROUTE_SELECTION and _route_d_certifies:' in cell_source
    assert 'if "e" in ROUTE_SELECTION:' in cell_source


def test_the_json_records_which_routes_ran(cell_source):
    assert '"routes_selected": list(ROUTE_SELECTION),' in cell_source
    assert '"pass_budget_mode": PASS_BUDGET_MODE,' in cell_source
    assert '"safe_budget": SAFE_BUDGET,' in cell_source
