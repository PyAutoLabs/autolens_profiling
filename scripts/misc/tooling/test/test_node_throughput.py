"""Projected per-node throughput in the v3 catalogue (likelihood-lifecycle phase 2x, #462)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLING = Path(__file__).resolve().parents[2] / "tooling"
ROOT = TOOLING.parents[2]
sys.path.insert(0, str(TOOLING))
import catalogue_lifecycle as lc  # noqa: E402

HOSTS = {
    "hosts": {
        "cpu_node": {"cores": 236, "mem_gb": 921, "gpus": 0, "host_class": "hpc_ral_cpu"},
        "gpu_node": {"cores": 124, "mem_gb": 885, "gpus": 4, "host_class": "hpc_a100"},
        "laptop": {"cores": 8, "mem_gb": 32, "gpus": 0, "host_class": "local_laptop"},
    },
    "parallel_efficiency": {"value": 1.0, "kind": "upper bound", "source": "declared"},
    "not_modelled": "Nautilus n_batch",
}


def _res(cpus, rss, **kw):
    return {"cpus_per_eval": cpus, "peak_rss_gb": rss, "peak_rss_basis": "sacct_maxrss", **kw}


def _host(block, name):
    """The record's single projected host, which must be ``name``."""
    assert [h["host"] for h in block["hosts"]] == [name]
    return block["hosts"][0]


def test_cores_bound_numba():
    block = lc.project(_res(1, 0.854), 0.95, "warm_iid_median_s", "cpu", "hpc_ral_cpu", HOSTS)
    cpu = _host(block, "cpu_node")
    assert block["basis"] == "projected" and block["status"] == "projected"
    assert cpu["workers"] == 236 and cpu["bound_by"] == "cores"
    assert cpu["limits"] == {"cores": 236, "memory": 1078}
    assert cpu["calls_per_s"] == pytest.approx(236 / 0.95, rel=1e-3)
    assert cpu["host_class"] == "hpc_ral_cpu"
    assert "per_call_measured_on_host" not in cpu


def test_projects_only_onto_the_measured_host_class():
    for host_class, device, name in (
        ("hpc_ral_cpu", "cpu", "cpu_node"),
        ("hpc_a100", "gpu", "gpu_node"),
        ("local_laptop", "cpu", "laptop"),
    ):
        block = lc.project(_res(1, 1.0), 1.0, "warm_iid_median_s", device, host_class, HOSTS)
        assert block["status"] == "projected"
        assert _host(block, name)["host_class"] == host_class


def test_a100_cpu_leg_uses_the_gpu_node_cores_as_cpu_workers():
    # jax_cpu_sparse@hpc_a100: a CPU leg measured on the A100 node's CPUs.
    block = lc.project(_res(8, 3.2), 1.9, "vmap.per_call", "cpu", "hpc_a100", HOSTS)
    gpu = _host(block, "gpu_node")
    assert gpu["limits"] == {"cores": 15, "memory": 276}
    assert gpu["workers"] == 15 and gpu["bound_by"] == "cores"
    assert gpu["calls_per_s"] == pytest.approx(15 / 1.9, rel=1e-3)


def test_unmatched_host_class_is_not_projected():
    block = lc.project(_res(1, 1.0), 1.0, "vmap.per_call", "gpu", "local_gpu", HOSTS)
    assert block["status"] == "not_projected" and block["hosts"] == []
    assert block["reason"] == "no declared reference host has host_class local_gpu"


def test_gpu_record_on_a_host_without_gpus_is_not_projected():
    block = lc.project(_res(1, 1.0), 1.0, "vmap.per_call", "gpu", "local_laptop", HOSTS)
    assert block["status"] == "not_projected" and block["hosts"] == []
    assert "declares no GPUs" in block["reason"]


def test_memory_bound_numba():
    block = lc.project(_res(1, 22.978), 8.67, "warm_iid_median_s", "cpu", "hpc_ral_cpu", HOSTS)
    cpu = _host(block, "cpu_node")
    assert cpu["workers"] == 40 and cpu["bound_by"] == "memory"
    assert cpu["calls_per_s"] == pytest.approx(40 / 8.67, rel=1e-3)


def test_jax_cpu_uses_its_threads():
    block = lc.project(_res(8, 3.2), 1.9, "vmap.per_call", "cpu", "hpc_ral_cpu", HOSTS)
    assert _host(block, "cpu_node")["workers"] == 29


def test_gpu_is_one_process_per_device():
    block = lc.project(_res(8, 10.7), 0.035, "vmap.per_call", "gpu", "hpc_a100", HOSTS)
    gpu = _host(block, "gpu_node")
    assert gpu["workers"] == 4 and gpu["bound_by"] == "gpus"
    assert gpu["limits"] == {"gpus": 4, "cores": 15, "memory": 82}
    assert gpu["calls_per_s"] == pytest.approx(4 / 0.035, rel=1e-3)


def test_gpu_host_memory_still_caps():
    block = lc.project(_res(8, 300.0), 0.4, "vmap.per_call", "gpu", "hpc_a100", HOSTS)
    assert block["hosts"][0]["workers"] == 2
    assert block["hosts"][0]["bound_by"] == "memory"


def test_efficiency_scales_throughput():
    hosts = {**HOSTS, "parallel_efficiency": {"value": 0.5}}
    block = lc.project(_res(1, 1.0), 1.0, "warm_iid_median_s", "cpu", "hpc_ral_cpu", hosts)
    assert _host(block, "cpu_node")["calls_per_s"] == 118.0


def test_missing_peak_rss_is_null_with_reason():
    res = _res(1, None, peak_rss_reason="sacct recorded no MaxRSS")
    block = lc.project(res, 0.5, "warm_iid_median_s", "cpu", "hpc_ral_cpu", HOSTS)
    assert block["status"] == "not_projected" and block["hosts"] == []
    assert "peak_rss_gb not recorded" in block["reason"]
    assert "sacct recorded no MaxRSS" in block["reason"]


def test_missing_per_call_is_null_with_reason():
    seconds, source = lc.per_call({"vmap": {"status": "failed"}}, "jax")
    assert seconds is None
    block = lc.project(_res(8, 1.0), seconds, source, "cpu", "hpc_ral_cpu", HOSTS)
    assert block["status"] == "not_projected" and "vmap.per_call" in block["reason"]


def test_per_call_sources():
    assert lc.per_call({"warm_iid_median_s": 0.9, "vmap": {"per_call": 9}}, "numba") == (
        0.9,
        "warm_iid_median_s",
    )
    assert lc.per_call({"warm_iid_median_s": 0.9, "vmap": {"per_call": 0.03}}, "jax") == (
        0.03,
        "vmap.per_call",
    )


def test_row_block_wins_then_backfill_then_unknown():
    backfill = {"r.json": {"cpus_per_eval": 1, "peak_rss_gb": 2.0, "sacct_job": "1_0"}}
    own = {"resources": {"cpus_per_eval": 8, "peak_rss_gb": 3.0, "engine": "jax"}}
    assert lc.row_resources(own, "r.json", "", backfill)["source"] == "row"
    assert lc.row_resources(own, "r.json", "", backfill)["peak_rss_gb"] == 3.0
    filled = lc.row_resources({}, "r.json", "", backfill)
    assert filled["source"] == lc.PEAK_RSS_BACKFILL and filled["sacct_job"] == "1_0"
    # A multi-row file's rows are never matched by the file-level backfill.
    assert lc.row_resources({}, "r.json", "/configs/a", backfill)["source"] is None
    unknown = lc.row_resources({}, "other.json", "", backfill)
    assert unknown["peak_rss_gb"] is None and unknown["reason"]


def test_committed_catalogue_carries_2a_resources_and_projections():
    doc = json.loads((ROOT / "dashboard/catalogue.json").read_text())
    backfill = lc.peak_rss_backfill(ROOT)
    hosts = lc.node_hosts(ROOT)
    assert set(hosts["hosts"]) == {"ral_cpu_node", "ral_gpu_node", "laptop"}
    seen = set()
    for record in doc["records"]:
        path = record["evidence"]["path"]
        if path not in backfill:
            continue
        seen.add(path)
        assert record["resources"]["source"] == lc.PEAK_RSS_BACKFILL
        assert record["resources"]["cpus_per_eval"] == backfill[path]["cpus_per_eval"]
        if record["measurement"] == "runtime":
            block = record["projected_node_throughput"]
            assert block["basis"] == "projected"
            assert block["parallel_efficiency"]["source"].startswith("declared, unmeasured")
            assert (block["status"] == "projected") == (backfill[path]["peak_rss_gb"] is not None)
    assert seen == set(backfill)


def test_committed_catalogue_projects_each_record_only_onto_its_host_class():
    doc = json.loads((ROOT / "dashboard/catalogue.json").read_text())
    hosts = lc.node_hosts(ROOT)["hosts"]
    for record in doc["records"]:
        block = record.get("projected_node_throughput")
        if block is None:
            continue
        assert len(block["hosts"]) == (1 if block["status"] == "projected" else 0)
        for entry in block["hosts"]:
            assert entry["host_class"] == record["host_class"]
            assert hosts[entry["host"]]["host_class"] == record["host_class"]
            assert "per_call_measured_on_host" not in entry
