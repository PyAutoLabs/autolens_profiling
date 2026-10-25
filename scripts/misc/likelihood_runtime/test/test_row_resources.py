"""Every runtime / breakdown row records ``resources`` (likelihood-lifecycle phase 2x).

``_profile_cli.resources_dict`` gives ``cpus_per_eval`` and the in-process ``peak_rss_gb``,
the inputs to the catalogue's projected per-node throughput.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import _profile_cli as pc  # noqa: E402

WRITERS = sorted(
    p
    for pattern in ("scripts/*/*/likelihood_runtime*.py", "scripts/*/*/likelihood_breakdown*.py")
    for p in ROOT.glob(pattern)
    if '"device": device_info_dict(),' in p.read_text()
)


def test_numba_is_one_cpu(monkeypatch):
    monkeypatch.setenv("NPROC", "8")
    block = pc.resources_dict("numba")
    assert block["cpus_per_eval"] == 1
    assert block["devices_per_eval"] == 0
    assert block["peak_rss_gb"] > 0
    assert block["peak_rss_basis"] == "in_process_ru_maxrss"


def test_jax_cpu_is_nproc(monkeypatch):
    monkeypatch.setenv("NPROC", "8")
    block = pc.resources_dict("jax")
    assert block["device"] == "cpu"  # the test host has no GPU visible to JAX
    assert block["cpus_per_eval"] == 8


def test_unset_nproc_is_null_and_said(monkeypatch):
    monkeypatch.delenv("NPROC", raising=False)
    block = pc.resources_dict("jax")
    assert block["cpus_per_eval"] is None
    assert "NPROC unset" in block["cpus_per_eval_basis"]


def test_peak_rss_is_gb_of_ru_maxrss():
    import resource

    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert abs(pc.peak_rss_gb() - kib * 1024 / 1e9) < 0.05


def test_every_row_writer_records_resources():
    assert len(WRITERS) >= 20
    for path in WRITERS:
        text = path.read_text()
        engine = "numba" if "numba" in path.name else "jax"
        devices = text.count('"device": device_info_dict(),')
        calls = re.findall(r'"resources": resources_dict\("(\w+)"\),', text)
        assert calls == [engine] * devices, path
