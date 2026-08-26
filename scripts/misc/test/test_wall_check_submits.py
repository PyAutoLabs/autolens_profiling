"""Unit tests for ``wall.check_submits`` — the per-cell `--time` gate.

The load-bearing test here is `test__phase8b_as_shipped_is_rejected`: it
reconstructs `submit_phase8b_bijector_a100` exactly as it was submitted (an MGE
step rate quoted as the basis for a `knn` / `delaunay_adapt_split` array at
``--time=0:30:00``) and asserts the checker refuses it. That submit lost 35 of
39 arms on RAL job 340576. No JAX dependency.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_wall_check_submits.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_misc_dir = str(ROOT / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)

import pytest  # noqa: E402
from wall.check_submits import (  # noqa: E402
    cells_run,
    check_text,
    main,
    parse_basis_rows,
    parse_slurm_time,
)
from wall.rates import UnmeasuredCellError, step_rate_for, wall_estimate  # noqa: E402

# ---------------------------------------------------------------------------
# fixtures — minimal submits in the shapes the repo actually uses
# ---------------------------------------------------------------------------

# phase8b exactly as it was submitted: an MGE citation covering a mostly
# pixelized array. This is the artifact the whole module exists to reject.
PHASE8B_AS_SHIPPED = """#!/bin/bash -l
#
# ESTIMATED WALL - 16 starts x 3000 steps at the #117-validated pixelized
# throughput is ~5 min including compile per task (matches the diagnostic_
# theta_e submit's n16/3000-step citation); --time below gives it 6x headroom.

# WALL-BASIS:
#   cell: imaging/mge/hst  device: a100  precision: fp64
#   lanes: 16  batch_size: 4  steps: 3000  rate: 0.117  source: rates
#   compile: 150  headroom: 1.5

#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --array=0-38

export JAX_ENABLE_X64=True
CELLS=(delaunay_adapt_split knn mge)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
export SEARCHES_BATCH_SIZE=4
N_STARTS=16
N_STEPS=3000

python3 scripts/imaging/searches/multi_start_prodigy/${CELL}.py \\
    --instrument hst \\
    --config-name $CONFIG_NAME
"""


def _submit(basis: str, time: str = "7:00:00", cell: str = "knn") -> str:
    return f"""#!/bin/bash -l
# WALL-BASIS:
{basis}

#SBATCH --partition=gpu
#SBATCH --time={time}

export JAX_ENABLE_X64=True
python3 scripts/imaging/searches/multi_start_prodigy/{cell}.py --instrument hst
"""


KNN_RATES_ROW = (
    "#   cell: imaging/knn/hst  device: a100  precision: fp64\n"
    "#   lanes: 16  batch_size: 4  steps: 3000  rate: 2.23  source: rates\n"
    "#   compile: 300  headroom: 1.5"
)


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------


def test__phase8b_as_shipped_is_rejected():
    """The submit that killed 35 of 39 arms must not pass the gate."""
    problems = check_text(PHASE8B_AS_SHIPPED, "submit_phase8b_bijector_a100")
    joined = "\n".join(problems)

    # The central rule: the two cells with no row of their own are named.
    assert "imaging/delaunay_adapt_split" in joined
    assert "imaging/knn" in joined
    assert "no WALL-BASIS row" in joined

    # And the mge row it *did* carry is not what saves it.
    assert problems


def test__phase8b_as_fixed_passes():
    """Three honest rows and a --time set by the slowest cell."""
    rows = "\n".join(
        [
            "#   cell: imaging/delaunay_adapt_split/hst  device: a100  precision: fp64",
            "#   lanes: 16  batch_size: 4  steps: 3000  rate: 4.83  source: rates",
            "#   compile: 300  headroom: 1.5",
            "#   cell: imaging/knn/hst  device: a100  precision: fp64",
            "#   lanes: 16  batch_size: 4  steps: 3000  rate: 2.23  source: rates",
            "#   compile: 300  headroom: 1.5",
            "#   cell: imaging/mge/hst  device: a100  precision: fp64",
            "#   lanes: 16  batch_size: 4  steps: 3000  rate: 0.117  source: rates",
            "#   compile: 150  headroom: 1.5",
        ]
    )
    fixed = PHASE8B_AS_SHIPPED.replace(
        "# WALL-BASIS:\n"
        "#   cell: imaging/mge/hst  device: a100  precision: fp64\n"
        "#   lanes: 16  batch_size: 4  steps: 3000  rate: 0.117  source: rates\n"
        "#   compile: 150  headroom: 1.5",
        "# WALL-BASIS:\n" + rows,
    ).replace("--time=0:30:00", "--time=7:00:00")
    assert check_text(fixed, "submit_phase8b_bijector_a100") == []


# ---------------------------------------------------------------------------
# the individual rules
# ---------------------------------------------------------------------------


def test__missing_header_is_required_only_on_searches_submits():
    bare = (
        "#!/bin/bash -l\n#SBATCH --time=1:00:00\npython3 scripts/imaging/searches/nautilus/mge.py\n"
    )
    assert check_text(bare, "submit_search_nautilus_imaging_mge_a100_hst_fp64")
    assert check_text(bare, "submit_runtime_imaging_mge_a100_hst_fp64") == []


def test__cited_rate_must_match_the_table():
    wrong = KNN_RATES_ROW.replace("rate: 2.23", "rate: 0.117")
    problems = "\n".join(check_text(_submit(wrong), "submit_search_x"))
    assert "disagrees with wall/rates.py" in problems


def test__rates_source_needs_a_measured_row_for_that_cell():
    """`source: rates` on an unmeasured cell is refused, not silently substituted."""
    row = (
        "#   cell: imaging/delaunay_matern/hst  device: a100  precision: fp64\n"
        "#   lanes: 16  batch_size: 4  steps: 3000  rate: 2.23  source: rates\n"
        "#   compile: 300  headroom: 1.5"
    )
    problems = "\n".join(check_text(_submit(row, cell="delaunay_matern"), "submit_search_x"))
    assert "no measured row" in problems
    assert "do NOT cite another cell's rate" in problems


def test__time_below_headroom_is_rejected():
    problems = "\n".join(check_text(_submit(KNN_RATES_ROW, time="1:00:00"), "submit_search_x"))
    assert "killed mid-run" in problems


def test__unmeasured_needs_probe_first():
    row = "#   cell: imaging/knn/hst  device: a100  precision: fp64\n#   source: unmeasured"
    assert "probe-first" in "\n".join(check_text(_submit(row), "submit_search_x"))

    ok = row + "  probe-first: yes"
    assert check_text(_submit(ok), "submit_search_x") == []


def test__unmeasured_wall_carries_a_3x_floor():
    row = (
        "#   cell: imaging/knn/hst  device: a100  precision: fp64\n"
        "#   source: unmeasured  probe-first: yes  wall: 6000  headroom: 1.5"
    )
    assert "below the 3.0 floor" in "\n".join(check_text(_submit(row), "submit_search_x"))


def test__declared_cell_must_be_one_the_submit_runs():
    row = (
        "#   cell: imaging/mge/hst  device: a100  precision: fp64\n"
        "#   lanes: 16  batch_size: 4  steps: 3000  rate: 0.117  source: rates\n"
        "#   compile: 150  headroom: 1.5"
    )
    problems = "\n".join(check_text(_submit(row, cell="knn"), "submit_search_x"))
    assert "never runs imaging/mge" in problems


def test__measured_wall_needs_a_ref():
    row = "#   cell: imaging/knn/hst  device: a100  precision: fp64\n#   wall: 800  source: measured-wall"
    assert "needs both" in "\n".join(check_text(_submit(row), "submit_search_x"))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,seconds",
    [
        ("30", 1800),
        ("5:00", 300),
        ("0:30:00", 1800),
        ("7:00:00", 25200),
        ("12:00:00", 43200),
        ("1-00:00:00", 86400),
    ],
)
def test__parse_slurm_time(value, seconds):
    assert parse_slurm_time(value) == seconds


def test__rows_split_on_cell_and_tolerate_prose():
    text = _submit(
        "#   prose that explains the block and carries no key/value pairs\n"
        + KNN_RATES_ROW
        + "\n#   cell: imaging/mge/hst  device: a100  precision: fp64\n"
        "#   lanes: 16  batch_size: 4  steps: 3000  rate: 0.117  source: rates"
    )
    rows = parse_basis_rows(text)
    assert [r["cell"] for r in rows] == ["imaging/knn/hst", "imaging/mge/hst"]
    assert rows[0]["rate"] == "2.23"


def test__cells_run_resolves_array_indirection():
    cells, instruments = cells_run(PHASE8B_AS_SHIPPED)
    assert cells == {
        ("imaging", "delaunay_adapt_split"),
        ("imaging", "knn"),
        ("imaging", "mge"),
    }
    assert instruments == {"hst"}


def test__cells_run_resolves_case_assignments():
    text = """#!/bin/bash -l
case "$SLURM_ARRAY_TASK_ID" in
  0)  MODEL=pixelization  ;;
  1)  MODEL=delaunay_nn   ;;
esac
python3 scripts/imaging/searches/nautilus/${MODEL}.py --instrument hst
"""
    cells, _ = cells_run(text)
    assert cells == {("imaging", "pixelization"), ("imaging", "delaunay_nn")}


def test__cells_run_ignores_commented_invocations():
    """A command merely *mentioned* in the header is not a cell the job runs."""
    text = """#!/bin/bash -l
# Score with:
#   python3 scripts/misc/searches/bijector_ab.py --score
python3 scripts/imaging/searches/nautilus/mge.py --instrument hst
"""
    cells, _ = cells_run(text)
    assert cells == {("imaging", "mge")}


# ---------------------------------------------------------------------------
# rates table
# ---------------------------------------------------------------------------


def test__step_rate_never_falls_back_to_another_cell():
    assert step_rate_for("imaging", "mge", "hst", "a100", "fp64", 16, 4) == 0.117
    with pytest.raises(UnmeasuredCellError):
        step_rate_for("imaging", "delaunay_matern", "hst", "a100", "fp64", 16, 4)
    # ...and not across lane counts or batching either.
    with pytest.raises(UnmeasuredCellError):
        step_rate_for("imaging", "knn", "hst", "a100", "fp64", 256, 4)
    with pytest.raises(UnmeasuredCellError):
        step_rate_for("imaging", "knn", "hst", "a100", "fp64", 16, None)


def test__the_41x_spread_that_caused_the_loss():
    mge = step_rate_for("imaging", "mge", "hst", "a100", "fp64", 16, 4)
    knn = step_rate_for("imaging", "knn", "hst", "a100", "fp64", 16, 4)
    delaunay = step_rate_for("imaging", "delaunay_adapt_split", "hst", "a100", "fp64", 16, 4)
    assert round(knn / mge) == 19
    assert round(delaunay / mge) == 41
    # The 0:30:00 budget against the delaunay arm's real 3000-step cost.
    assert wall_estimate(delaunay, 3000) / 1800 > 8


# ---------------------------------------------------------------------------
# the repo itself
# ---------------------------------------------------------------------------


def test__every_submit_in_this_repo_passes():
    assert main(["--check", "--root", str(ROOT)]) == 0
