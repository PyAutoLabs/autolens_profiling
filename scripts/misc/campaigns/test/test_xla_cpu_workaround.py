"""Every campaign JAX CPU leg runs with the XLA:CPU deadlock workaround (autolens_profiling#473, D12).

jaxlib 0.11.2 XLA:CPU executables can block forever on a repeated call;
``--xla_cpu_multi_thread_eigen=false`` removes it. The shared preamble's ``run_leg`` appends it to
``XLA_FLAGS`` on every ``JAX_PLATFORMS=cpu`` leg (the A100 nodes' CPU legs included) and never on a
GPU leg (``cuda`` / ``cuda,cpu``), keeping the base flags either way. The test renders every
campaign's submits through its generator and runs each submit's own flag-selection block in bash,
leg by leg, in submit order (an A100 job runs its legs in one shell, so a CPU leg's flag must not
leak into a later GPU leg). No cell script is run.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGNS = ROOT / "scripts" / "misc" / "campaigns"
GENERATORS = (
    "imaging_p2a",
    "interferometer_p2b",
    "point_source_p2c",
    "multi_galaxy_group_p2d",
    "cluster_p2e",
    "multi_dataset_datacube_p2f",
    "weak_p2g",
)
BASE = "--xla_disable_hlo_passes=constant_folding"
FLAG = "--xla_cpu_multi_thread_eigen=false"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CAMPAIGNS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _submits() -> dict[str, str]:
    out = {}
    for name in GENERATORS:
        out.update(_load(name).submits())
    return out


def _legs(text: str) -> list[tuple[str, str]]:
    """(platform, XLA_FLAGS) for each leg, from the submit's own preamble and run_leg block."""
    base = re.search(r'^export XLA_FLAGS=".*"$', text, re.M).group(0)
    setup = re.search(r"^XLA_FLAGS_BASE=.*\nXLA_CPU_WORKAROUND_FLAG=.*$", text, re.M).group(0)
    block = re.search(
        r'^    if \[ "\$JAX_PLATFORMS" = cpu \]; then\n.*?^    fi$', text, re.M | re.S
    ).group(0)
    platforms = re.search(r"^PLATFORMS=\(\n.*?^\)$", text, re.M | re.S).group(0)
    script = "\n".join(
        [
            base,
            setup,
            platforms,
            'for I in "${!PLATFORMS[@]}"; do',
            "    export JAX_PLATFORMS=${PLATFORMS[$I]}",
            block,
            '    printf "%s\\t%s\\n" "$JAX_PLATFORMS" "$XLA_FLAGS"',
            "done",
        ]
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    return [tuple(line.split("\t")) for line in result.stdout.splitlines()]


def test_the_shared_flag_is_the_2d_drivers_flag():
    sys.path.insert(0, str(CAMPAIGNS))
    import _common

    sys.path.insert(0, str(ROOT))
    driver = (ROOT / "scripts/misc/likelihood_breakdown/galaxy_group_standard.py").read_text()
    assert _common.XLA_CPU_WORKAROUND_FLAG == FLAG
    assert f'XLA_CPU_WORKAROUND_FLAG = "{FLAG}"' in driver
    # The driver appends only when absent, so a 2d CPU leg carries the flag once.
    assert 'if "xla_cpu_multi_thread_eigen" not in flags:' in driver


def test_cpu_legs_get_the_flag_and_gpu_legs_do_not():
    submits = _submits()
    assert len(submits) >= 50
    n_cpu = n_gpu = 0
    for rel, text in submits.items():
        legs = _legs(text)
        assert legs, rel
        for platform, flags in legs:
            words = flags.split()
            assert BASE in words, (rel, platform, flags)
            if platform == "cpu":
                n_cpu += 1
                assert words.count(FLAG) == 1, (rel, platform, flags)
            else:
                n_gpu += 1
                assert platform in ("cuda", "cuda,cpu"), (rel, platform)
                assert FLAG not in words, (rel, platform, flags)
    assert n_cpu and n_gpu


def test_an_a100_job_switches_the_flag_per_leg():
    """A sequential job running a CPU leg then a GPU leg resets XLA_FLAGS for the GPU leg."""
    text = next(
        t
        for t in _submits().values()
        if "cpu" in [p for p, _ in _legs(t)] and any(p != "cpu" for p, _ in _legs(t))
    )
    seen = [(p, FLAG in f.split()) for p, f in _legs(text)]
    assert all(has == (p == "cpu") for p, has in seen)
