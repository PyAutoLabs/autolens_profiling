"""Static checks on the two-arm numba-lever A100 submits (autolens_profiling#267).

These legs are a different shape from every other ``submit_*fixed_light*``
script, which is why they have their own file rather than riding on
``test_fixed_light_cell.py``'s single-arm contracts (see the comment beside that
file's ``_submits()``). Each lever submit runs **two arms back-to-back in one
job** — a control PyAutoArray checkout and a feature one — so that both share a
node, a GPU and a driver, and the only difference between the arms is the
library revision. Each arm is a ``( ... )`` subshell that prepends its own
checkout to ``PYTHONPATH``, so the prepend cannot leak into the next arm.

What that shape breaks, and what is checked here instead:

- the run lines are **indented inside a subshell**, so ``^python3 -u`` does not
  find them; here the anchor allows leading whitespace and the count must be
  exactly two, one per arm;
- lever 1 runs ``likelihood_breakdown/delaunay.py``, not ``fixed_light.py`` — it
  takes no ``--mesh`` (the cell *is* the mesh) and writes under its own
  ``lever1`` config names, because it is a verbatim copy of the
  ``..._adapt_split`` submit whose record it is compared against;
- levers 2 and 3 do wipe a fresh compilation cache per arm, but as
  ``JAX_CACHE_CONTROL`` / ``JAX_CACHE_FEATURE`` rather than the single-arm
  ``JAX_COMPILATION_CACHE_DIR`` string.

The contracts that matter for an A/B are the ones about *provenance*: two
distinct private checkouts, neither of them the shared install (which
``HPCPullPyAuto`` moves under the job's feet), each arm echoing the checkout's
``rev-parse HEAD`` and the imported ``autoarray.__file__``, and two paired
``--config-name`` values so the control row cannot overwrite the feature row.

The submit scripts themselves are frozen: the recorded RAL runs depend on their
text, so a failure here is fixed in this file or in a *new* submit, never by
editing one that has run.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_fixed_light_numba_levers_submits.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
BATCH_GPU = ROOT / "hpc" / "batch_gpu"

#: The meshes ``fixed_light.py``'s ``--mesh`` choices accept.
MESHES = ("rectangular", "delaunay", "delaunay_nn")

#: Every ``fixed_light.py`` leg must write under a config name the harvest picks
#: up; the lever legs append their own suffix to it.
FIXED_LIGHT_PREFIX = "hpc_a100_fp64_fixed_light"

#: The private per-arm PyAutoArray checkouts live under this directory on RAL.
CHECKOUT_PARENT = "/mnt/ral/jnightin/PyAuto_wt/fixed-light-numba-levers/"

#: The shared install. It must never reach an arm's PYTHONPATH: HPCPullPyAuto
#: moves it whenever anyone refreshes it, so an arm resolved through it is not a
#: known revision. It may legitimately appear in the `git -C` echo lines that
#: record the other libraries' revisions.
SHARED_INSTALL = "/mnt/ral/jnightin/PyAuto/PyAutoArray"

#: The prior record lever 1's submit is a verbatim copy of.
ADAPT_SPLIT_RECORD = "submit_breakdown_imaging_delaunay_a100_hst_fp64_adapt_split"

LEVER_ONE = "submit_breakdown_imaging_fixed_light_numba_levers_delaunay_a100_hst_fp64"


def _submits() -> list[_Path]:
    return sorted(BATCH_GPU.glob("submit_breakdown_imaging_fixed_light_numba_levers_*"))


def _arms(text: str) -> list[str]:
    """The ``( ... )`` subshell arms of a submit, each returned as its own text.

    A line that is exactly ``(`` opens an arm and a line that is exactly ``)``
    closes it. Heredoc bodies are stepped over rather than parsed, because the
    lever-3 submit builds its provenance snippet with ``$(cat <<'PYPROV' ...)``
    and that Python code has closing parens in column 0.
    """
    arms: list[str] = []
    current: list[str] | None = None
    heredoc: str | None = None

    for line in text.splitlines():
        stripped = line.strip()

        if heredoc is not None:
            if current is not None:
                current.append(line)
            if stripped == heredoc:
                heredoc = None
            continue

        if current is not None:
            if stripped == ")":
                arms.append("\n".join(current))
                current = None
                continue
            current.append(line)
        elif stripped == "(":
            current = []

        opener = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", line)
        if opener is not None:
            heredoc = opener.group(1)

    return arms


def _arms_by_role(path: _Path) -> dict[str, str]:
    """The two arms keyed ``control`` / ``feature`` by the checkout each prepends."""
    arms = _arms(path.read_text())
    assert len(arms) == 2, (
        f"{path.name}: expected exactly two `( ... )` arms (control, feature), found {len(arms)}"
    )

    by_role: dict[str, str] = {}
    for role in ("CONTROL", "FEATURE"):
        matching = [a for a in arms if f"PYTHONPATH=$PYAUTOARRAY_{role}:$PYTHONPATH" in a]
        assert len(matching) == 1, (
            f"{path.name}: expected exactly one arm exporting "
            f"PYTHONPATH=$PYAUTOARRAY_{role}:$PYTHONPATH, found {len(matching)}"
        )
        by_role[role.lower()] = matching[0]

    assert by_role["control"] is not by_role["feature"], (
        f"{path.name}: both roles resolved to the same arm"
    )
    return by_role


def _run_scripts(path: _Path) -> list[str]:
    """The ``python3 -u <script>`` cells a submit runs, one per arm."""
    return re.findall(r"^\s*python3 -u (\S+)", path.read_text(), re.M)


def _cell_name(path: _Path) -> str:
    scripts = _run_scripts(path)
    names = {_Path(s).name for s in scripts}
    assert len(names) == 1, f"{path.name}: the two arms must run the same cell, got {sorted(names)}"
    return names.pop()


def _config_names(path: _Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for role, arm in _arms_by_role(path).items():
        found = re.findall(r"--config-name\s+(\S+)", arm)
        assert len(found) == 1, (
            f"{path.name}: the {role} arm must pass exactly one --config-name, found {found}"
        )
        names[role] = found[0]
    return names


def test__at_least_the_lever_one_submit_exists():
    names = {p.name for p in _submits()}
    assert LEVER_ONE in names, f"missing submit {LEVER_ONE}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_is_valid_bash(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{path.name}: bash -n failed\n{result.stderr}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_runs_two_arms_of_a_script_that_exists(path):
    # The run lines are indented inside their subshells, so the anchor allows
    # leading whitespace. `parse_known_args` ignores unknown flags, so a submit
    # naming a cell that is not there is the failure mode worth catching early.
    scripts = _run_scripts(path)
    assert len(scripts) == 2, (
        f"{path.name}: an A/B job runs exactly two `python3 -u <cell>` lines "
        f"(control, feature), found {len(scripts)}: {scripts}"
    )
    for script in scripts:
        assert (ROOT / script).is_file(), f"{path.name}: runs missing script {script}"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_has_two_arms_on_their_own_checkouts(path):
    # _arms_by_role carries the assertions: exactly two subshells, one
    # prepending $PYAUTOARRAY_CONTROL and one $PYAUTOARRAY_FEATURE.
    _arms_by_role(path)


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__submit_exports_two_distinct_private_checkouts(path):
    text = path.read_text()
    checkouts = {}
    for role in ("CONTROL", "FEATURE"):
        found = re.findall(rf"^export PYAUTOARRAY_{role}=(\S+)", text, re.M)
        assert len(found) == 1, (
            f"{path.name}: expected exactly one `export PYAUTOARRAY_{role}=`, found {found}"
        )
        checkouts[role] = found[0]
        assert found[0].startswith(CHECKOUT_PARENT), (
            f"{path.name}: PYAUTOARRAY_{role}={found[0]} is not a private checkout under "
            f"{CHECKOUT_PARENT}"
        )
    assert checkouts["CONTROL"] != checkouts["FEATURE"], (
        f"{path.name}: both arms point at {checkouts['CONTROL']} — the A/B measures nothing"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__the_shared_install_is_never_prepended_to_pythonpath(path):
    # HPCPullPyAuto moves the shared install whenever anyone refreshes it, so an
    # arm resolved through it is not a pinned revision. Only PYTHONPATH is
    # policed: the submits legitimately `git -C` it to echo the other libraries'
    # revisions.
    for line in path.read_text().splitlines():
        if "PYTHONPATH=" not in line:
            continue
        assert SHARED_INSTALL not in line, (
            f"{path.name}: the shared install {SHARED_INSTALL} is on PYTHONPATH "
            f"({line.strip()}) — it is not a pinned revision"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__each_arm_prints_its_import_provenance(path):
    # The headers explain the trap: without these echoes a moved shared install,
    # or a PYTHONPATH that did not take, is indistinguishable in the .out from a
    # genuine null result.
    text = path.read_text()
    assert "autoarray.__file__" in text, (
        f"{path.name}: no arm prints autoarray.__file__, so the .out cannot say which "
        f"checkout was imported"
    )
    for role, arm in _arms_by_role(path).items():
        assert "python3 -c" in arm, (
            f"{path.name}: the {role} arm runs no `python3 -c` provenance snippet"
        )
        assert "rev-parse HEAD" in arm, (
            f"{path.name}: the {role} arm does not echo its checkout's rev-parse HEAD"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__the_two_arms_pass_paired_config_names(path):
    # Distinct names so the control run cannot overwrite the feature run's JSON,
    # and paired by the `_control` suffix so a harvest can tell which is which.
    names = _config_names(path)
    assert names["control"] != names["feature"], (
        f"{path.name}: both arms write under --config-name {names['control']} — the second "
        f"run would overwrite the first"
    )
    assert names["control"] == f"{names['feature']}_control", (
        f"{path.name}: the control arm writes {names['control']}, expected "
        f"{names['feature']}_control"
    )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__fixed_light_legs_pass_a_mesh_and_the_harvest_prefix(path):
    if _cell_name(path) != "fixed_light.py":
        pytest.skip(f"{path.name} runs {_cell_name(path)}, which takes no --mesh")

    text = path.read_text()
    meshes = re.findall(r"--mesh\s+(\S+)", text)
    assert meshes, f"{path.name}: fixed_light.py requires --mesh and the submit passes none"
    for mesh in meshes:
        assert mesh in MESHES, f"{path.name}: --mesh {mesh} is not one of {MESHES}"

    for role, name in _config_names(path).items():
        assert name.startswith(FIXED_LIGHT_PREFIX), (
            f"{path.name}: the {role} arm writes --config-name {name}, which does not start "
            f"with {FIXED_LIGHT_PREFIX}, so the fixed_light harvest will not find it"
        )


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__fixed_light_legs_wipe_a_fresh_cache_per_arm(path):
    # Both arms compile the same HLO, so one shared cache would serve the second
    # arm out of the first's entries and report a near-zero compile time for it.
    # A seeded XLA autotune cache also silently changes which kernels a leg uses
    # (results/notes/xla_autotune_triton_gemm.md: F at 4.8 vs 25.6 ms on the same
    # code). Hence two directories, wiped before either arm runs.
    if _cell_name(path) != "fixed_light.py":
        pytest.skip(f"{path.name} runs {_cell_name(path)}; see the delaunay-leg test below")

    text = path.read_text()
    wipes = [
        line
        for line in text.splitlines()
        if line.strip().startswith("rm -rf")
        and "$JAX_CACHE_CONTROL" in line
        and "$JAX_CACHE_FEATURE" in line
    ]
    assert wipes, (
        f"{path.name}: no `rm -rf` line wiping both $JAX_CACHE_CONTROL and $JAX_CACHE_FEATURE "
        f"before the arms run"
    )

    for role, arm in _arms_by_role(path).items():
        expected = f"export JAX_COMPILATION_CACHE_DIR=$JAX_CACHE_{role.upper()}"
        assert expected in arm, f"{path.name}: the {role} arm does not set `{expected}`"


@pytest.mark.parametrize("path", _submits(), ids=lambda p: p.name)
def test__the_delaunay_leg_names_the_record_it_copies(path):
    # Lever 1 runs `delaunay.py`, whose comparable prior row is
    # `delaunay_hpc_a100_fp64.json`. Its submit is a VERBATIM copy of that row's
    # submit, so that the only difference between these arms and that record is
    # the PyAutoArray revision — which is also why no compilation cache is
    # asserted for it: the record it is compared against did not set one either,
    # and adding one would make it a different job. It has already run as RAL
    # job 343346.
    if _cell_name(path) != "delaunay.py":
        pytest.skip(f"{path.name} runs {_cell_name(path)}, not the delaunay cell")

    assert ADAPT_SPLIT_RECORD in path.read_text(), (
        f"{path.name}: runs delaunay.py but does not name {ADAPT_SPLIT_RECORD}, the submit it "
        f"is a verbatim copy of — without that the arms cannot be read against its record"
    )
