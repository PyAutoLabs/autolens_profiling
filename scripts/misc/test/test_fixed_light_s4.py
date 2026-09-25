"""Unit tests for phase 4 of the fixed-lens-light numba CPU campaign (#274).

Wave A is a **measurement**, not a rewrite: the library already carries two
curvature-matrix kernels and dispatches between them on a pixel-count cap that
was calibrated on the HST *rectangular* fiducial and never measured on the
Delaunay fixed-light cell. What this phase adds is the seam that lets all three
kernels run in one process under the cell's ABBA harness, a candidate kernel, a
witness and a RAL leg.

What is tested here
-------------------

1. **The kernels agree.** On the same fixture the two library branches agree to
   1e-9 relative and the candidate is **bit-identical** to the two-stage kernel
   it copies — the claim the module docstring argues for and which licenses
   promoting it into PyAutoArray without touching the library's own reference
   test. All three also equal the result the un-injected dispatcher returns,
   which is the check that the seam is comparing kernels rather than problems.

2. **The seam fires, and restores.** On a real ``FitImaging.figure_of_merit``
   over the shared ``tiny_s3_pair`` fixture, with the counter asserted — a patch
   that never fired would report route ``b``'s milliseconds wearing another
   route's label — and a stranger rebinding the attribute mid-block is refused
   rather than clobbered.

3. **The witness and the submit are what they say they are.** Static checks: the
   witness imports no JAX at module scope and declares the flags the submit
   passes it; the submit parses under ``bash -n``, runs only scripts that exist,
   and carries the routes, repeat count and thread pins the phase's verdict rule
   is defined against.

The fixtures are imported from ``test_fixed_light_numba`` rather than rebuilt, so
a failure here is a failure of *this* phase and not of a second copy of the
geometry that has drifted from the first.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_s4.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc = ROOT / "scripts" / "misc"
if str(_misc) not in _sys.path:
    _sys.path.insert(0, str(_misc))

# This directory carries an `__init__.py`, so pytest imports its modules as a
# package and `test_fixed_light_numba` is not importable as a top-level name.
# The fixtures below are shared with that module and must not be copied, so the
# directory itself goes on the path.
_here = _Path(__file__).resolve().parent
if str(_here) not in _sys.path:
    _sys.path.insert(0, str(_here))

from likelihood_breakdown import fixed_light_numpy_kernels as flnk  # noqa: E402

# The fixtures — `tiny` and `tiny_s3_pair` — are the ones the numba cell's own
# tests build, imported so there is exactly one copy of that geometry. They are
# re-bound under their own names rather than imported under them: pytest resolves
# a fixture by the name it finds in this module's namespace, and an imported
# fixture that a test then names as a parameter reads to the linter as a
# redefinition of the import.
from test_fixed_light_numba import _clear_nnls_memo  # noqa: E402
from test_fixed_light_numba import tiny as _tiny_fixture  # noqa: E402
from test_fixed_light_numba import tiny_s3_pair as _tiny_s3_pair_fixture  # noqa: E402

tiny = _tiny_fixture
tiny_s3_pair = _tiny_s3_pair_fixture

WITNESS_PATH = (
    ROOT / "scripts" / "imaging" / "likelihood_breakdown" / "fixed_light_numba_s4_witness.py"
)
SUBMIT_PATH = (
    ROOT
    / "hpc"
    / "batch_cpu"
    / "submit_breakdown_imaging_fixed_light_numba_s4_delaunay_ral_hst_fp64"
)

#: The phase's correctness tolerance between the two LIBRARY kernels, which sum
#: the same products in a different order. The library's own pin on this pair is
#: 1e-6; the observed difference on the production HST geometries is ~4e-13 and
#: on this fixture ~1e-16, so 1e-9 is the scale this campaign asserts at.
S4_RTOL = 1.0e-9


def _kernel_arguments(system) -> dict:
    """The seven arguments ``sparse.py:385`` passes, read off a built system."""
    from autoarray.inversion.mappers.abstract import Mapper

    operator = system.dataset.sparse_operator
    mapper = system.fit.inversion.cls_list_from(cls=Mapper)[0]
    return {
        "psf_precision_operator": operator.psf_precision_operator_sparse,
        "psf_precision_indexes": operator.indexes,
        "psf_precision_lengths": operator.lengths,
        "data_to_pix_unique": mapper.unique_mappings.data_to_pix_unique,
        "data_weights": mapper.unique_mappings.data_weights,
        "pix_lengths": mapper.unique_mappings.pix_lengths,
        "pix_pixels": mapper.params,
    }


# ---------------------------------------------------------------------------
# 1. The three kernels, on the fixture's own operator
# ---------------------------------------------------------------------------


def test_the_three_curvature_kernels_agree_and_touched_is_bit_identical(tiny_s3_pair):
    """``direct`` ~ ``two_stage`` to 1e-9; ``two_stage_touched`` is EXACTLY it.

    Two different claims, deliberately asserted differently.

    The library's two branches sum the same products in a different order, so
    they agree to floating-point reassociation and nothing tighter. The
    candidate does not reorder anything: it removes additions of exact ``+0.0``
    and keeps every surviving addition in the order the library performs it, so
    the correct assertion is ``np.array_equal``. A tolerance here would pass a
    kernel that had quietly started reassociating, and reassociation is exactly
    what would make promoting it into PyAutoArray a behaviour change rather than
    an optimisation.

    All three are also compared against the **un-injected** dispatcher, which at
    this fixture's 64 source pixels takes its two-stage branch — the branch
    production takes at every source size this campaign measures.
    """
    from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util

    _dense, sparse = tiny_s3_pair
    arguments = _kernel_arguments(sparse)

    assert (
        arguments["pix_pixels"] <= inversion_imaging_numba_util.CURVATURE_TWO_STAGE_MAX_PIX_PIXELS
    ), "this fixture must sit on the dispatcher's two-stage branch, as production does"

    library = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(**arguments)

    results = {}
    for name in ("two_stage", "direct", "two_stage_touched"):
        with flnk.curvature_kernel_injected(name, label="unit") as counts:
            results[name] = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(
                **arguments
            )
            assert counts["n_calls"] == 1
            assert counts["kernel"] == name
            assert counts["last_pix_pixels"] == int(arguments["pix_pixels"])

    scale = max(float(np.max(np.abs(library))), 1e-300)

    # The control really is the branch production takes.
    assert np.array_equal(results["two_stage"], library), (
        "forcing two_stage_max_pix_pixels=10**9 must reproduce the un-injected "
        "dispatcher bit for bit on a fixture below the cap; if it does not, the "
        "control row is not the production path"
    )

    direct_rel = float(np.max(np.abs(results["direct"] - library))) / scale
    assert direct_rel <= S4_RTOL, (
        f"direct vs two-stage max relative difference {direct_rel:.3e} > {S4_RTOL:g} — "
        f"the two library branches no longer assemble the same matrix"
    )
    assert direct_rel > 0.0 or results["direct"].size == 0, (
        "direct agreeing BIT-for-bit with two-stage would mean one of them is not "
        "running; they sum the same products in a different order"
    )

    assert np.array_equal(results["two_stage_touched"], library), (
        "two_stage_touched is not bit-identical to the library's two-stage kernel. "
        "Its whole promotion case is that it removes additions of exact +0.0 and "
        "reorders nothing; a difference here — however small — refutes that, and "
        "max|Δ| = "
        f"{float(np.max(np.abs(results['two_stage_touched'] - library))):.3e}"
    )


def test_the_touched_kernel_is_symmetric_and_matches_the_reference_loop(tiny_s3_pair):
    """The halved-diagonal / ``A + A.T`` contract is the library's, unchanged."""
    from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util

    _dense, sparse = tiny_s3_pair
    arguments = _kernel_arguments(sparse)

    touched = flnk.curvature_matrix_via_sparse_operator_two_stage_touched_from(**arguments)
    reference = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_reference_from(
        **arguments
    )

    assert np.array_equal(touched, touched.T), "the kernel's output must be symmetric"

    scale = max(float(np.max(np.abs(reference))), 1e-300)
    rel = float(np.max(np.abs(touched - reference))) / scale
    assert rel <= S4_RTOL, (
        f"touched vs the library's straightforward reference loop {rel:.3e} > {S4_RTOL:g}"
    )


# ---------------------------------------------------------------------------
# 2. The seam, over a real fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", ["two_stage", "direct", "two_stage_touched"])
def test_the_kernel_seam_fires_once_per_fit_and_reaches_the_same_evidence(
    tiny_s3_pair, kernel_name
):
    """One call per ``figure_of_merit``, and route ``b``'s answer at the end of it.

    The whole ``FitImaging`` is run each way — the library's own mapper, the
    solve, both log determinants and the evidence — with one function swapped.
    The counter is the evidence the patch took: the inversion resolves the
    dispatcher by module attribute (``sparse.py:385``), and a library that
    stopped doing so would leave this at zero while still producing a perfectly
    plausible number.
    """
    autolens = pytest.importorskip("autolens")
    al = autolens

    _dense, sparse = tiny_s3_pair

    def one_fit():
        _clear_nnls_memo()
        return al.FitImaging(
            dataset=sparse.dataset,
            tracer=sparse.source_only_tracer,
            adapt_images=sparse.fit.adapt_images,
            settings=sparse.fit.settings,
            xp=np,
        )

    fom_b = float(one_fit().figure_of_merit)

    with flnk.curvature_kernel_injected(kernel_name, label=f"b_{kernel_name}") as counts:
        fom_injected = float(one_fit().figure_of_merit)

    assert counts["n_calls"] == 1, (
        f"the {kernel_name} kernel fired {counts['n_calls']} time(s) during one "
        f"figure_of_merit; this row's milliseconds would be route b's wearing "
        f"route b_{kernel_name}'s label"
    )
    assert counts["dotted_name"] == flnk.KERNEL_DOTTED_NAMES[kernel_name]

    rel = abs(fom_injected - fom_b) / max(abs(fom_b), 1e-300)
    assert rel <= S4_RTOL, (
        f"{kernel_name} figure_of_merit {fom_injected!r} vs b {fom_b!r} "
        f"(rel {rel:.3e} > {S4_RTOL:g})"
    )

    if kernel_name in ("two_stage", "two_stage_touched"):
        assert fom_injected == fom_b, (
            f"{kernel_name} is bit-identical to the library's two-stage kernel on the "
            f"matrix, so the evidence it produces must be bit-identical too; got "
            f"Δ {fom_injected - fom_b:+.3e} nats"
        )


def test_the_kernel_seam_restores_by_identity_and_refuses_a_stranger():
    """Restoring over someone else's rebinding would hide a nesting bug."""
    from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util

    original = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from
    stranger = object()

    with pytest.raises(RuntimeError, match="rebound by something else"):
        with flnk.curvature_kernel_injected("two_stage", label="unit"):
            inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from = stranger

    assert inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from is stranger
    inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from = original

    # A clean block restores the exact object it found, not an equal one.
    with flnk.curvature_kernel_injected("direct", label="unit"):
        patched = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from
        assert patched is not original
        assert patched.__wrapped__ is original
    assert inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from is original

    with pytest.raises(KeyError, match="unknown kernel"):
        with flnk.curvature_kernel_injected("two_stage_but_faster", label="unit"):
            pass
    assert inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from is original


def test_the_seam_drops_the_dispatchers_cap_keyword():
    """A caller that passes ``two_stage_max_pix_pixels`` is asking for a choice.

    The injection has taken that choice over for the duration of the row, so the
    keyword is accepted (the signature must not break a caller) and ignored. The
    inversion never passes it — ``sparse.py:385`` supplies seven keywords — but a
    library test that did would otherwise silently un-inject the row.
    """
    from autoarray.inversion.inversion.imaging_numba import inversion_imaging_numba_util

    rng = np.random.default_rng(17)
    data_pixels, pix_pixels, max_map = 24, 11, 3
    lengths = rng.integers(1, 5, size=data_pixels).astype(np.int64)
    total = int(lengths.sum())
    arguments = {
        "psf_precision_operator": rng.normal(size=total),
        "psf_precision_indexes": rng.integers(0, data_pixels, size=total).astype(np.int64),
        "psf_precision_lengths": lengths,
        "data_to_pix_unique": np.zeros((data_pixels, max_map), dtype=np.int64),
        "data_weights": np.zeros((data_pixels, max_map)),
        "pix_lengths": rng.integers(0, max_map + 1, size=data_pixels).astype(np.int64),
        "pix_pixels": pix_pixels,
    }
    for data in range(data_pixels):
        n = int(arguments["pix_lengths"][data])
        if n:
            arguments["data_to_pix_unique"][data, :n] = rng.choice(
                pix_pixels, size=n, replace=False
            )
            arguments["data_weights"][data, :n] = rng.random(n)

    with flnk.curvature_kernel_injected("direct", label="unit") as counts:
        forced = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(
            two_stage_max_pix_pixels=10**9, **arguments
        )
        assert counts["n_calls"] == 1

    with flnk.curvature_kernel_injected("direct", label="unit"):
        plain = inversion_imaging_numba_util.curvature_matrix_via_sparse_operator_from(**arguments)

    assert np.array_equal(forced, plain), (
        "a `two_stage_max_pix_pixels` the caller supplied must not reach the library "
        "dispatcher and un-inject the row"
    )


# ---------------------------------------------------------------------------
# 3. The witness: static checks
# ---------------------------------------------------------------------------


def test_the_s4_witness_never_imports_jax():
    """Not at module level, not in a function, not behind a flag.

    The mirror of the cell's own ``test_the_cell_never_imports_jax``. This
    witness runs beside the numba CPU arm of the s4 leg and its whole premise is
    the numpy path; a JAX import at module scope would also pull a runtime into
    the process the A/B rows are timed in, on the same node, at the same time.
    (``jax`` still lands in ``sys.modules`` via the first ``FitImaging``, which is
    the library's doing and is recorded in the witness JSON rather than hidden.)
    """
    assert WITNESS_PATH.exists(), f"{WITNESS_PATH} does not exist"
    tree = ast.parse(WITNESS_PATH.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "jax", f"`import {alias.name}`"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "jax", f"`from {module} import ...`"

    # ...and `device_info_dict` is never imported or called: it imports jax
    # unconditionally (`_profile_cli.py:392`).
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }
    assert "device_info_dict" not in imported

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "device_info_dict" not in called


def test_the_s4_witness_declares_the_flags_the_submit_passes_it():
    """A flag the submit passes and the witness does not declare is silently dropped.

    ``parse_known_args`` ignores what it does not recognise, so a witness whose
    parser lost ``--threads`` would run at whatever thread count it defaulted to
    and report the submit's number in its JSON anyway. The four flags below are
    exactly what the s4 submit's witness invocation supplies.
    """
    source = WITNESS_PATH.read_text()

    for flag in ('"--mesh"', '"--dataset"', '"--threads"'):
        assert f"_cell_parser.add_argument({flag}" in source, (
            f"the witness does not declare {flag}; the submit passes it"
        )
    # `--config-name` is the shared `_parse_profile_cli()` flag, and the output
    # path is derived from it — a witness that hardcoded a name would overwrite
    # the previous arm's JSON.
    assert "_parse_profile_cli()" in source
    assert "_cli.config_name" in source
    assert 'f"fixed_light_numba_s4_witness_{_config_name}.json"' in source

    # The smoke exit, the verdict and the non-zero exit on FAIL.
    assert 'os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1"' in source
    assert '"verdict": "PASS" if _all_pass else "FAIL"' in source
    assert "raise SystemExit(" in source, (
        "a FAILED witness must exit non-zero, or SLURM records a COMPLETED job with a "
        "bad witness buried in the log"
    )


def test_the_s4_witness_gates_touched_at_exact_equality_not_a_tolerance():
    """W2's two claims are different claims, and must be asserted differently.

    ``direct`` reassociates and is gated at a tolerance. ``two_stage_touched``
    does not reassociate — its promotion case is that it removes additions of
    exact ``+0.0`` — so it is gated at ``np.array_equal``. A witness that gated
    both at 1e-9 would pass a touched kernel that had started reordering, which
    is exactly the change that would make promoting it a behaviour change.
    """
    source = WITNESS_PATH.read_text()

    assert '"bit_identical_to_two_stage": bool(np.array_equal(' in source
    assert (
        '_w2_touched_identical = _w2_rows["two_stage_touched"]["bit_identical_to_two_stage"]'
        in source
    )
    assert "_w2_direct_rel <= W2_RTOL" in source
    assert "W2_RTOL = 1e-9" in source
    assert "W3_RTOL = 1e-9" in source
    assert "W4_N_REPEATS = 20" in source

    # W4 is RECORDED, never gated: there is no calibrated ratio for an arbitrary
    # host and the row that decides the lever is the whole-call A/B.
    assert '"W4_kernel_site_cost",\n    "RECORDED",' in source
    assert 'entry["status"] in ("PASS", "RECORDED")' in source

    # And the control is checked to be the production path.
    assert "_w2_control_is_production" in source


# ---------------------------------------------------------------------------
# 4. The RAL submit: static checks
# ---------------------------------------------------------------------------


def test_the_s4_submit_parses_and_runs_only_scripts_that_exist():
    """``bash -n`` plus a path check on every cell it invokes.

    A submit that references a moved or renamed script fails on the cluster, an
    hour into a queue, with an exit status the log buries. Both of these are
    free to check here.
    """
    assert SUBMIT_PATH.exists(), f"{SUBMIT_PATH} does not exist"

    completed = subprocess.run(["bash", "-n", str(SUBMIT_PATH)], capture_output=True, text=True)
    assert completed.returncode == 0, f"bash -n failed:\n{completed.stderr}"

    source = SUBMIT_PATH.read_text()
    invoked = re.findall(r"^\s*python3 -u (\S+)", source, flags=re.M)
    assert invoked, "the submit invokes no cell at all"
    for relative in invoked:
        assert (ROOT / relative).exists(), (
            f"the submit runs {relative}, which does not exist in this checkout"
        )

    assert "scripts/imaging/likelihood_breakdown/fixed_light_numba.py" in invoked
    assert "scripts/imaging/likelihood_breakdown/fixed_light_numba_s4_witness.py" in invoked


def test_the_s4_submit_runs_the_three_row_ab_at_the_settings_the_verdict_assumes():
    """The flags are the measurement. A quiet drift in any of them is a new question.

    ``parse_known_args`` ignores what it does not recognise, so a route string
    the cell does not know would produce a silently different table rather than
    an error. These are the flags the phase's verdict rule — a kernel is a lever
    if it beats route ``b`` by >= 5 % on the whole call with the ABBA gate PASS —
    is defined against.
    """
    source = SUBMIT_PATH.read_text()

    assert "--routes b,b_direct,b_touched" in source, (
        "the A/B is the three rows in one process; any other route string is a "
        "different measurement"
    )
    assert "--n-repeats 64" in source, (
        "32 counterbalanced blocks. 8 could not resolve this gate on this cell (job "
        "343355), which is the whole reason this family moved off 16 repeats."
    )
    assert "--formalism sparse_numba" in source
    assert "--instances iid" in source
    assert "--nnls-warm-start on" in source, (
        "memo ON is what production pays; a cold row is a different problem"
    )
    assert "--threads 1" in source

    # Single-thread scope has to be pinned in the environment too: OpenBLAS/MKL
    # and numba read their knobs once, when their libraries load, so the cell's
    # own --threads cannot retrofit them.
    for variable in (
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMBA_NUM_THREADS=1",
    ):
        assert variable in source, f"the submit does not pin {variable.split('=')[0]}"

    # The per-job numba cache under output/ is the submit's own; matplotlib's comes
    # from activate.sh ($PYAUTO_HPC_CACHE), never /tmp -- on RAL /tmp and $HOME sit
    # on the node's small root disk (RAL admin, 2026-09-25).
    assert "export NUMBA_CACHE_DIR=$AP_ROOT/output/" in source
    assert "/tmp/" not in source

    # CPUs-only on the gpu partition. Checked on the #SBATCH directives rather
    # than on the whole file, because the header explains the absence in prose
    # and a substring search would read that explanation as the thing it denies.
    directives = re.findall(r"^#SBATCH\s+(.*)$", source, flags=re.M)
    assert "--partition=gpu" in directives
    assert not any("--gres" in directive for directive in directives), (
        "this is a CPU row on the gpu partition's CPUs; asking for a GPU would take "
        "an A100 out of service for four hours to run numba on a core"
    )
    assert "--cpus-per-task=4" in directives
    assert "--mem=32gb" in directives
    assert "--time=4:00:00" in directives


def test_the_s4_submit_carries_a_wall_basis_row_per_cell_it_runs():
    """The gate `check_submits.py --check` enforces, asserted here on this submit.

    The l3 submit this one is modelled on carries no ``# WALL-BASIS:`` block at
    all, which means its ``--time`` rests on nothing a reader can check. This one
    declares a row per cell, and the rows are read back through the checker's own
    parser rather than by a substring search, so a malformed row fails here and
    not on the cluster.
    """
    import importlib.util

    checker_path = ROOT / "scripts" / "misc" / "wall" / "check_submits.py"
    spec = importlib.util.spec_from_file_location("_s4_check_submits", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    source = SUBMIT_PATH.read_text()
    rows = checker.parse_basis_rows(source)

    declared = {row["cell"] for row in rows}
    assert declared == {
        "imaging/fixed_light_numba/hst",
        "imaging/fixed_light_numba_s4_witness/hst",
    }, f"the submit runs two cells; its WALL-BASIS declares {sorted(declared)}"

    for row in rows:
        assert row["source"] == "measured-wall"
        assert "wall" in row and "ref" in row, (
            "`source: measured-wall` needs both `wall:` and `ref:` naming where it was observed"
        )
        assert float(row["headroom"]) >= checker.HEADROOM_FLOOR["measured-wall"]

    assert checker.check_text(source) == [], (
        f"check_submits.py rejects this submit: {checker.check_text(source)}"
    )
