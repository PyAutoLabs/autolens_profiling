"""
Drive the hang matrix: jax version x XLA flag x reproducer x repeat, one child per cell.

Each cell runs :mod:`hang_case` under the interpreter of a venv holding only that jax version
(``python -I``, so nothing but jax is importable). The parent watches the child's progress
lines; when no new line arrives for ``--stall`` seconds it samples the child's CPU time for
``--cpu-window`` seconds and kills it. A stall at ~0 % CPU is the XLA:CPU deadlock; a stall
with CPU work is recorded as ``slow`` (raise ``--stall`` and rerun), never as a hang.

    python run_matrix.py \\
        --venv 0.11.2=/path/venv-0.11.2 --venv 0.12.0.1=/path/venv-0.12.0.1 \\
        --flags none,eigen,sched --cases fft,lapack,export:group_mge_p3 \\
        --modules-dir output/jax_version_policy/modules --repeats 1 --calls 300 \\
        --out output/jax_version_policy/rows/laptop_pilot.jsonl --environment local

One JSON row per cell is appended to ``--out`` as each finishes, so a killed run keeps its
rows. Issue PyAutoLabs/PyAutoNerves#213.
"""

import argparse
import datetime
import json
import os
import selectors
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAGS = {
    "none": "",
    "eigen": "--xla_cpu_multi_thread_eigen=false",
    "sched": "--xla_cpu_enable_concurrency_optimized_scheduler=false",
}


def _cpu_seconds(pid: int) -> float:
    """utime + stime of a process and its threads, from /proc."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    tick = os.sysconf("SC_CLK_TCK")
    return (int(fields[11]) + int(fields[12])) / tick


def run_cell(python, case, module, flag, calls, stall, cpu_window, cpus, first_stall, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "JAX_", "XLA_"))}
    env["JAX_ENABLE_X64"] = "True"
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = FLAGS[flag]
    if case.startswith("library:"):
        # the real library: PyAuto interpreter, jax under test shadowed in front via PYTHONPATH
        env["PYTHONPATH"] = os.pathsep.join(p for p in (pythonpath, os.environ.get("PYTHONPATH")) if p)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [python, str(HERE / "library_case.py"), "--cell", case.split(":", 1)[1], "--calls", str(calls)]
    else:
        cmd = [python, "-I", str(HERE / "hang_case.py"), "--case", case, "--calls", str(calls)]
    if module:
        cmd += ["--module", module]
    if cpus:
        # the first N cores of this process's own affinity (SLURM allocations are not 0..N-1)
        allowed = sorted(os.sched_getaffinity(0))[:cpus]
        cmd = ["taskset", "-c", ",".join(map(str, allowed))] + cmd

    row = {"case": case if not module else f"export:{Path(module).stem}", "flag": flag,
           "jax_source": "shadow" if pythonpath else "venv",
           "xla_flags": FLAGS[flag], "calls_requested": calls}
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    per_call, last = [], time.perf_counter()
    outcome, note = None, None
    while outcome is None:
        # compile can legitimately take minutes; the stall clock for calls is tighter
        limit = first_stall if not per_call else stall
        events = sel.select(timeout=1.0)
        if events:
            line = proc.stdout.readline()
            if not line:
                proc.wait()
                outcome = "pass" if row.get("done") else "error"
                if outcome == "error":
                    note = proc.stderr.read()[-600:]
                continue
            last = time.perf_counter()
            tag, _, rest = line.strip().partition(" ")
            if tag == "HEADER":
                row["header"] = json.loads(rest)
            elif tag == "COMPILED":
                row["compile_s"] = float(rest)
            elif tag == "CALL":
                per_call.append(float(rest.split()[1]))
            elif tag == "DONE":
                row["done"] = True
            elif tag == "UNLOADABLE":
                outcome, note = "unloadable", rest
                proc.wait()
        elif time.perf_counter() - last > limit:
            c0 = _cpu_seconds(proc.pid)
            time.sleep(cpu_window)
            c1 = _cpu_seconds(proc.pid)
            row["stall_cpu_fraction"] = round((c1 - c0) / cpu_window, 4)
            outcome = "hang" if row["stall_cpu_fraction"] < 0.05 else "slow"
            row["stalled_in"] = "compile" if "compile_s" not in row else f"call {len(per_call)}"
            proc.kill()
            proc.wait()
    row.pop("done", None)
    row.update(
        outcome=outcome,
        calls_completed=len(per_call),
        hung_at_call=len(per_call) if outcome == "hang" and "compile_s" in row else None,
        median_call_s=statistics.median(per_call) if per_call else None,
        steady_median_call_s=statistics.median(per_call[2:]) if len(per_call) > 4 else None,
        wall_s=round(time.perf_counter() - started, 2),
        note=note,
    )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--venv", action="append", default=[], help="VERSION=/path/to/venv (pure-jax cases)")
    p.add_argument("--shadow", action="append", default=[],
                   help="VERSION=/path/to/pip-target-dir or VERSION=base (library: cases, run under "
                        "--library-python with that jax put first on PYTHONPATH)")
    p.add_argument("--library-python", default=sys.executable)
    p.add_argument("--flags", default="none,eigen,sched")
    p.add_argument("--cases", default="fft,lapack", help="fft, lapack, export:<module stem>")
    p.add_argument("--modules-dir", default=str(HERE.parents[2] / "output" / "jax_version_policy" / "modules"))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--repeat-offset", type=int, default=0, help="first repeat index (array tasks)")
    p.add_argument("--calls", type=int, default=300)
    p.add_argument("--stall", type=float, default=60.0, help="seconds without a call line")
    p.add_argument("--first-stall", type=float, default=600.0, help="seconds allowed for compile")
    p.add_argument("--cpu-window", type=float, default=5.0)
    p.add_argument("--cpus", type=int, default=0, help="pin to the first N cores (0 = all)")
    p.add_argument("--environment", required=True, help="DNA environment id, e.g. local, ral-cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    venvs = dict(v.split("=", 1) for v in a.venv)
    shadows = dict(v.split("=", 1) for v in a.shadow)
    library = [c for c in a.cases.split(",") if c.startswith("library:")]
    pure = [c for c in a.cases.split(",") if not c.startswith("library:")]
    plan = [(v, str(Path(d) / "bin" / "python"), None, pure) for v, d in venvs.items() if pure]
    plan += [(v, a.library_python, None if d == "base" else str(Path(d).resolve()), library)
             for v, d in shadows.items() if library]
    for rep in range(a.repeats):
        for version, python, pythonpath, cases in plan:
            for case in cases:
                module = None
                if case.startswith("export:"):
                    module = str(Path(a.modules_dir) / f"{case.split(':', 1)[1]}.pkl")
                    case = "export"
                for flag in a.flags.split(","):
                    row = run_cell(python, case, module, flag, a.calls, a.stall, a.cpu_window,
                                   a.cpus, a.first_stall, pythonpath)
                    row.update(
                        jax_requested=version,
                        repeat=rep + a.repeat_offset,
                        environment=a.environment,
                        host=socket.gethostname(),
                        cpus_pinned=a.cpus or None,
                        stall_s=a.stall,
                        recorded=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    )
                    with out.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(f"{version:>9} {flag:>5} {row['case']:<28} rep{row['repeat']} -> {row['outcome']:<10}"
                          f" calls={row['calls_completed']:>3} med={row['median_call_s']}"
                          f" wall={row['wall_s']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
