"""Phase 6b gradient campaign (autolens_profiling#507): the manifest follows the designations,
every leg runs its likelihood's own fused function through a --gradient hook, and the submits are
current and pass check_submits."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "scripts" / "misc" / "campaigns"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "misc" / "tooling"))
spec = importlib.util.spec_from_file_location("gradient_p6", HERE / "gradient_p6.py")
p6 = importlib.util.module_from_spec(spec)
sys.modules["gradient_p6"] = p6
spec.loader.exec_module(p6)
import catalogue_lifecycle as lc  # noqa: E402

#: The fused-likelihood construction sites that carry the --gradient hook.
HOOKED = {
    "scripts/misc/likelihood_breakdown/galaxy_group_standard.py",
    "scripts/misc/likelihood_breakdown/cluster_standard.py",
    "scripts/misc/likelihood_breakdown/datacube_standard.py",
    "scripts/misc/likelihood_breakdown/multi_dataset_standard.py",
    "scripts/misc/likelihood_breakdown/interferometer_standard.py",
    "scripts/imaging/delaunay/likelihood_runtime.py",
    "scripts/imaging/delaunay_nn/likelihood_runtime.py",
    "scripts/imaging/rectangular/likelihood_runtime.py",
    "scripts/interferometer/mge/likelihood_runtime.py",
    "scripts/weak/isothermal/likelihood_runtime.py",
    "scripts/point_source_image/image_plane/likelihood_runtime_solved.py",
    "scripts/point_source_source/source_plane/likelihood_runtime_solved.py",
}


def test_the_manifest_is_the_ruled_cell_set():
    """Human ruling 2026-11-01 (#505): JAX optimal, the JAX CPU secondary of numba-optimal
    cells, and the held datacube A100 group's provisional JAX optimal."""
    tasks = p6.planned_tasks()
    assert len(tasks) == len({t.cell for t in tasks}) == 74
    roles = Counter()
    for t in tasks:
        entry = lc.designation_of(t.dataset_type, t.variant, t.host_class)
        if entry is None:
            assert lc.hold_of(t.dataset_type, t.variant, t.host_class) is not None
            roles["held"] += 1
        elif t.setup_id == entry["optimal"]:
            roles["jax optimal"] += 1
        else:
            assert entry["optimal"].startswith("numba_") and t.setup_id in entry["secondary"]
            assert t.setup_id.startswith("jax_cpu_") and t.host_class == "hpc_ral_cpu"
            roles["jax secondary of numba optimal"] += 1
    assert roles == {"jax optimal": 46, "jax secondary of numba optimal": 25, "held": 3}
    assert not any("_mixed" in t.setup_id or "_streaming" in t.setup_id for t in tasks)
    # Every designated likelihood has a gradient leg on both reference hosts.
    covered = {(t.dataset_type, t.variant, t.host_class) for t in tasks}
    for row in lc._DESIGNATIONS.values():
        dataset_type, variant = row["likelihood"].split("/")
        host = {"cpu_many_core": "hpc_ral_cpu", "gpu_a100": "hpc_a100"}[row["hardware_class"]]
        assert (dataset_type, variant, host) in covered, row["id"]


def test_every_leg_runs_a_hooked_construction_site():
    hooked = {p for p in HOOKED if "gradient_requested()" in (ROOT / p).read_text()}
    assert hooked == HOOKED
    drivers = {Path(p).stem for p in HOOKED if "likelihood_breakdown" in p}
    for t in p6.planned_tasks():
        script = ROOT / "scripts" / f"{t.script}.py"
        assert script.is_file(), t.script
        text = script.read_text()
        if t.dataset_type == "imaging" and t.variant == "mge":
            assert "harness.measure" in text  # the 6a pilot builds the cell itself
            continue
        runtime = re.search(r'with_name\("([a-z_]+\.py)"\)', text).group(1)
        runtime_path = script.with_name(runtime)
        runtime_text = runtime_path.read_text()
        assert str(runtime_path.relative_to(ROOT)) in HOOKED or any(
            driver in runtime_text for driver in drivers
        ), t.cell


def test_a_gradient_leg_keeps_its_runtime_legs_flags_and_cell():
    for t in p6.planned_tasks():
        assert t.output.endswith("/gradient.json")
        assert Path(t.output).parent == Path(t.runtime_output).parent
        assert f"/{t.setup_id}/{t.host_class}/" in t.output
        assert t.flags == t.runtime_flags
        assert t.platform.startswith("cuda") if t.device == "gpu" else t.platform == "cpu"
        assert t.guess_s >= p6.GUESS_FLOOR_S


def test_the_submits_are_current_and_pass_check_submits():
    import argparse

    args = argparse.Namespace(write=False, check=True)
    assert p6._common.write_or_check(p6.submits(), ROOT, args=args, generator="gradient_p6.py") == 0
    sys.path.insert(0, str(ROOT / "scripts" / "misc"))
    from wall import check_submits

    for rel, text in p6.submits().items():
        assert ("--partition=ral" in text) == ("batch_cpu" in rel)
        assert ("--gres=gpu:1" in text) == ("batch_gpu" in rel)
        assert check_submits.parse_basis_rows(text), rel
        assert check_submits.check_text(text) == [], rel
