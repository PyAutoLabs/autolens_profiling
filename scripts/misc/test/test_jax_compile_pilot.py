"""Protocol and process-safety tests; no production dataset or timing claims."""

import copy
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ruff.toml").exists())
sys.path.insert(0, str(ROOT / "scripts/misc"))
from jax_compile import pilot, pilot_worker


@pytest.fixture
def plan(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("data.fits", "noise_map.fits", "psf.fits"):
        (data / name).write_bytes(b"fixture bytes, not scientific FITS")
    args = SimpleNamespace(
        dataset_path=data,
        output_dir=tmp_path / "output" / pilot.NAMESPACE / "run",
        cpu=min(os.sched_getaffinity(0)),
        timeout=1,
        repetitions=1,
    )
    return pilot.make_plan(args, tmp_path)


def record(plan, step):
    metadata = {
        "workload": {"dataset_sha256": plan["dataset_sha256"]},
        "cpu_affinity": [plan["cpu"]],
        "jax_enable_x64": True,
        "devices": [{"platform": "cpu"}],
        "comparison_context": {"fixture": "same"},
    }
    return {
        "schema": pilot.SCHEMA,
        "version": 1,
        "accepted": False,
        "objective": pilot.OBJECTIVE,
        "worker_id": step["id"],
        "mode": step["mode"],
        "seed_id": step["seed_id"],
        "plan_sha256": pilot.fingerprint(plan),
        "outcome": "measured",
        "provenance": metadata,
        "context_sha256": pilot.fingerprint(metadata["comparison_context"]),
        "cache_evidence": {
            "state": "hit" if step["mode"] == "warm" else "miss",
            "key": "key",
            "target": "jit_objective",
        },
        "timings": {
            "setup_s": 1.2,
            "trace_lower_s": 0.123456789,
            "compile_s": 0.2,
            "first_s": 0.03,
            "steady_samples_s": [0.0123456789] * 10 if step["mode"] == "runtime" else [],
            "warmup_calls": 3 if step["mode"] == "runtime" else 0,
            "synchronized": True,
        },
    }


def fake_runner(command, **kwargs):
    request_path = Path(command[-1])
    request = json.loads(request_path.read_text())
    pilot.write_new(
        request_path.parent / "observation.json", record(request["plan"], request["step"])
    )
    return {"outcome": "completed", "returncode": 0, "worker_wall_s": 0.1}


def test_plan_does_not_create_outputs_and_binds_inputs(plan):
    assert not Path(plan["output_dir"]).exists()
    assert plan["max_processes"] == 4
    assert plan["max_worker_wall_s"] == 4
    assert [s["mode"] for s in plan["steps"]] == ["cold", "seed", "warm", "runtime"]
    assert plan["accepted"] is False
    assert len(plan["dataset_sha256"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"repetitions": 11},
        {"repetitions": 0},
        {"timeout": float("nan")},
        {"timeout": 1801},
        {"cpu": -1},
    ],
)
def test_plan_rejects_caps_before_writing(tmp_path, plan, change):
    args = SimpleNamespace(
        dataset_path=Path(plan["dataset_path"]),
        output_dir=Path(plan["output_dir"]),
        repetitions=1,
        timeout=1,
        cpu=plan["cpu"],
    )
    vars(args).update(change)
    with pytest.raises(ValueError):
        pilot.make_plan(args, tmp_path)
    assert not args.output_dir.exists()


def test_plan_refuses_escape_and_symlinks(tmp_path, plan):
    args = SimpleNamespace(
        dataset_path=Path(plan["dataset_path"]),
        output_dir=tmp_path / "results/run",
        repetitions=1,
        timeout=1,
        cpu=plan["cpu"],
    )
    with pytest.raises(ValueError):
        pilot.make_plan(args, tmp_path)
    namespace = tmp_path / "output" / pilot.NAMESPACE
    namespace.parent.mkdir()
    namespace.symlink_to(tmp_path / "data", target_is_directory=True)
    args.output_dir = namespace / "run"
    with pytest.raises(ValueError, match="symlink"):
        pilot.make_plan(args, tmp_path)


@pytest.mark.parametrize(
    "messages,target,state",
    [
        (
            ["Persistent compilation cache hit for 'jit_objective' with key 'abc'"],
            "jit_objective",
            "hit",
        ),
        (
            ["PERSISTENT COMPILATION CACHE MISS for 'jit_objective' with key 'abc'"],
            "jit_objective",
            "miss",
        ),
        (
            ["Persistent compilation cache hit for 'jit_helper' with key 'abc'"],
            "jit_objective",
            "unknown",
        ),
        (["future unknown format"], "jit_objective", "unknown"),
        (
            ["Persistent compilation cache hit for 'jit_objective' with key 'abc'"] * 2,
            "jit_objective",
            "unknown",
        ),
        (["Persistent compilation cache hit for 'jit_objective' with key 'abc'"], None, "unknown"),
    ],
)
def test_cache_requires_unambiguous_target_event(messages, target, state):
    result = pilot.cache_evidence(messages, target)
    assert result["state"] == state
    assert result["messages"] == messages


def test_receipt_preserves_raw_samples_and_separate_series(plan):
    result = pilot.execute(plan, runner=fake_runner)
    assert result["outcome"] == "recorded"
    assert result["accepted"] is False
    assert len(result["workers"]) == 4
    assert result["summary"]["runtime/steady_samples_s"]["samples_s"] == [0.0123456789] * 10
    assert result["summary"]["cold/trace_lower_s"]["median_s"] == 0.123456789
    assert "seed/compile_s" in result["summary"]
    assert result["unstarted_workers"] == []
    with pytest.raises(FileExistsError):
        pilot.execute(plan, runner=fake_runner)


def test_timeout_stops_without_retry_and_preserves_receipt(plan):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        return {"outcome": "timeout", "worker_wall_s": 1}

    result = pilot.execute(plan, runner=timeout)
    assert len(calls) == 1
    assert result["outcome"] == "incomplete"
    assert len(result["unstarted_workers"]) == 3
    assert (Path(plan["output_dir"]) / "receipt.json").is_file()


def test_bad_warm_witness_retained_but_excluded(plan):
    def runner(command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        value = record(request["plan"], request["step"])
        if value["mode"] == "warm":
            value["cache_evidence"]["key"] = "different-key"
        pilot.write_new(request_path.parent / "observation.json", value)
        return {"outcome": "completed"}

    result = pilot.execute(plan, runner=runner)
    assert result["outcome"] == "incomplete"
    assert result["workers"][-1]["outcome"] == "invalid"
    assert "warm/compile_s" not in result["summary"]
    assert result["unstarted_workers"] == ["runtime"]
    assert (Path(plan["output_dir"]) / "00-warm/observation.json").is_file()


@pytest.mark.parametrize(
    "field",
    ["context", "backend", "affinity", "samples", "warmup", "synchronized", "plan", "input"],
)
def test_actual_mismatch_is_rejected(plan, field):
    step = plan["steps"][-1]
    value = record(plan, step)
    if field == "context":
        value["provenance"]["comparison_context"] = {"different": True}
    elif field == "backend":
        value["provenance"]["devices"] = [{"platform": "cuda"}]
    elif field == "affinity":
        value["provenance"]["cpu_affinity"] = []
    elif field == "samples":
        value["timings"]["steady_samples_s"].pop()
    elif field == "warmup":
        value["timings"]["warmup_calls"] = 2
    elif field == "synchronized":
        value["timings"]["synchronized"] = False
    elif field == "plan":
        value["plan_sha256"] = "wrong"
    elif field == "input":
        value["provenance"]["workload"]["dataset_sha256"] = {}
    with pytest.raises(ValueError):
        pilot.validate_record(value, step, plan, [])


def test_summary_refuses_incompatible_contexts(plan):
    one = record(plan, plan["steps"][0])
    two = copy.deepcopy(one)
    two["context_sha256"] = "different"
    with pytest.raises(ValueError):
        pilot.summary([one, two])


def test_real_child_is_fresh_and_timeout_is_reaped(tmp_path):
    pidfile = tmp_path / "pid"
    result = pilot.run_process(
        [
            sys.executable,
            "-c",
            "import os,pathlib,time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)",
        ],
        env=os.environ.copy(),
        timeout=0.5,
        stdout=tmp_path / "out",
        stderr=tmp_path / "err",
        cwd=tmp_path,
    )
    assert result["outcome"] == "timeout"
    pid = int(pidfile.read_text())
    assert pid != os.getpid()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_preserves_partial_receipt(plan):
    def interrupted(*a, **k):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        pilot.execute(plan, runner=interrupted)
    receipt = json.loads((Path(plan["output_dir"]) / "receipt.json").read_text())
    assert receipt["outcome"] == "incomplete"
    assert receipt["accepted"] is False


def test_measurement_boundaries_and_individual_samples():
    events = []

    def sync(value):
        events.append("sync")
        return value

    def compiled(params):
        events.append("call")
        return 5

    class Lowered:
        def compiler_ir(self):
            return SimpleNamespace(
                operation=SimpleNamespace(
                    attributes={"sym_name": SimpleNamespace(value="jit_objective")}
                )
            )

        def compile(self):
            events.append("compile")
            logging.getLogger("jax._src.compiler").debug(
                "Persistent compilation cache hit for 'jit_objective' with key 'abc'"
            )
            return compiled

    def lower(params):
        events.append("lower")
        return Lowered()

    tick = iter(range(100))
    jax = SimpleNamespace(jit=lambda f: SimpleNamespace(lower=lower), block_until_ready=sync)
    timings, evidence, output, load = pilot_worker.measure(
        lambda x: x, 3, jax=jax, runtime=True, clock=lambda: next(tick)
    )
    assert timings["trace_lower_s"] == timings["compile_s"] == timings["first_s"] == 1
    assert timings["steady_samples_s"] == [1] * 10
    assert events.count("call") == 14
    assert events.index("lower") < events.index("compile") < events.index("call")
    assert all(events[i + 1] == "sync" for i, event in enumerate(events) if event == "call")
    assert output == 5 and evidence["state"] == "hit"
    assert len(load) == 11
