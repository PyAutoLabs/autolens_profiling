"""Shared SLURM submit machinery for the likelihood-lifecycle campaign generators.

``imaging_p2a.py`` (phase 2a) and ``interferometer_p2b.py`` (phase 2b) each hold their own work
list as tasks; this module renders those tasks into submit scripts the same way for both: the
preamble (private library checkout, import-origin guard, per-leg revision check, dataset
preparation under a lock, ``walls.tsv`` START / WALL lines), the bash arrays the jobs index, the
WALL-BASIS rows ``wall/check_submits.py`` reads, the ``#SBATCH`` block and the tail.

A task here is any object with ``cell``, ``gate_cell``, ``instrument``, ``variant``,
``setup_id``, ``measurement``, ``script``, ``flags``, ``output`` and ``platform``.

Stdlib only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- small helpers


def bash_array(name: str, values: list[str], *, quote: bool = False) -> str:
    """A bash array, one value per line. Values the wall gate resolves (cell scripts,
    instruments) stay bare: ``check_submits.resolve_var`` splits an array on whitespace."""
    body = "\n".join(f"    {json.dumps(v) if quote else v}" for v in values)
    return f"{name}=(\n{body}\n)"


def slurm_time(seconds: float, *, headroom: float, slack_s: int, round_s: int) -> str:
    """``H:MM:SS`` at ``headroom`` x ``seconds`` plus ``slack_s``, rounded up to ``round_s``."""
    need = headroom * seconds + slack_s
    total = math.ceil(need / round_s) * round_s
    return f"{total // 3600}:{total % 3600 // 60:02d}:00"


# --------------------------------------------------------------------------- the preamble

#: The XLA:CPU deadlock workaround every ``JAX_PLATFORMS=cpu`` leg appends to ``XLA_FLAGS``
#: (autolens_profiling#473, decision D12; one flag for every cell, human 2026-10-29). jaxlib 0.11.2
#: XLA:CPU executables can block forever on a repeated call; this flag removes it. GPU legs
#: (``cuda`` / ``cuda,cpu``) keep the base flags. The 2d driver appends the same flag only if it is
#: absent, so a 2d CPU leg carries it once.
XLA_CPU_WORKAROUND_FLAG = "--xla_cpu_multi_thread_eigen=false"

#: The import guard of a release-pinned campaign: every library imports from the private
#: checkout, which sits at the pinned tag.
TAG_GUARD = r"""# The libraries must import from the private checkout at the pinned tag, never the shared mirror.
python3 - <<'EOF_PY' || exit 3
import os, subprocess
import autoarray, autofit, autogalaxy, autolens, autonerves
base, pin = os.path.realpath(os.environ["PYAUTO_PRIVATE_BASE"]), os.environ["@PINVAR@"]
for m in (autonerves, autofit, autoarray, autogalaxy, autolens):
    print(f"{m.__name__:<11} {m.__file__}")
    if not os.path.realpath(m.__file__).startswith(base + os.sep):
        raise SystemExit(f"REFUSING: {m.__name__} imported from {m.__file__}, not {base}")
    repo = os.path.dirname(os.path.dirname(os.path.realpath(m.__file__)))
    tag = subprocess.check_output(["git", "-C", repo, "describe", "--tags"], text=True).strip()
    if tag != pin:
        raise SystemExit(f"REFUSING: {repo} is at {tag}, not tag {pin}")
print(f"import origin + tag {pin}: OK")
EOF_PY
"""

#: The import guard of an unreleased-main experiment: every library imports from the private
#: checkout, whose HEAD is the declared SHA (``@SHAS@`` is a JSON object, repo -> SHA).
SHA_GUARD = r"""# The libraries must import from the private checkout at the declared main SHAs, never the shared mirror.
python3 - <<'EOF_PY' || exit 3
import os, subprocess
import autoarray, autofit, autogalaxy, autolens, autonerves
base = os.path.realpath(os.environ["PYAUTO_PRIVATE_BASE"])
shas = @SHAS@
for m in (autonerves, autofit, autoarray, autogalaxy, autolens):
    print(f"{m.__name__:<11} {m.__file__}")
    if not os.path.realpath(m.__file__).startswith(base + os.sep):
        raise SystemExit(f"REFUSING: {m.__name__} imported from {m.__file__}, not {base}")
    repo = os.path.dirname(os.path.dirname(os.path.realpath(m.__file__)))
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    want = shas[os.path.basename(repo)]
    if head != want:
        raise SystemExit(f"REFUSING: {repo} is at {head}, not the declared main SHA {want}")
print("import origin + declared main SHAs: OK")
EOF_PY
"""

PREAMBLE = r"""export AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"
export PYAUTO_HPC_BASE="${PYAUTO_HPC_BASE:-@HPC_BASE@}"
export PYAUTO_PRIVATE_BASE="${@PRIVATE_VAR@:-@PRIVATE_BASE@}"
export @PINVAR@=@PIN@
echo "AP_ROOT=$AP_ROOT  PYAUTO_PRIVATE_BASE=$PYAUTO_PRIVATE_BASE  pin=$@PINVAR@"
source "$AP_ROOT/activate.sh"
export PYTHONPATH=$PYAUTO_PRIVATE_BASE/PyAutoNerves:$PYAUTO_PRIVATE_BASE/PyAutoFit:$PYAUTO_PRIVATE_BASE/PyAutoArray:$PYAUTO_PRIVATE_BASE/PyAutoGalaxy:$PYAUTO_PRIVATE_BASE/PyAutoLens:$PYTHONPATH

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NPROC=${SLURM_CPUS_PER_TASK:-8}
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"
# XLA:CPU deadlock workaround (autolens_profiling#473, D12): jaxlib 0.11.2 XLA:CPU executables can
# hang on a repeated call; run_leg appends this to XLA_FLAGS on every JAX_PLATFORMS=cpu leg (A100
# nodes' CPU legs included), never on a GPU leg. Rows record it in device.xla_flags.
XLA_FLAGS_BASE="$XLA_FLAGS"
XLA_CPU_WORKAROUND_FLAG="@XLA_CPU_WORKAROUND_FLAG@"
unset PYAUTO_SMALL_DATASETS
export NUMBA_CACHE_DIR="$AP_ROOT/output/numba_cache/@PREFIX@_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$AP_ROOT/output/@LOGS@"

echo "=========================================="
date
hostname
lscpu | grep 'Model name' || true
echo "partition: ${SLURM_JOB_PARTITION:-unset}  nodelist: ${SLURM_JOB_NODELIST:-unset}  cpus: ${SLURM_CPUS_PER_TASK:-unset}"
echo "loadavg:   $(cat /proc/loadavg)"
for r in PyAutoNerves PyAutoFit PyAutoArray PyAutoGalaxy PyAutoLens; do
    printf '%-14s %s %s\n' "$r" "$(git -C $PYAUTO_PRIVATE_BASE/$r rev-parse HEAD 2>&1)" "$(git -C $PYAUTO_PRIVATE_BASE/$r describe --tags 2>&1)"
done
echo "AP_ROOT rev: $(git -C $AP_ROOT rev-parse HEAD 2>&1)"

cd "$AP_ROOT"
@QUIET_DEF@@IMPORT_GUARD@
# After each leg: the row's recorded library revisions must be the private checkout's HEADs.
check_revisions() {
python3 - "$1" <<'EOF_PY'
import json, os, subprocess, sys
path = sys.argv[1]
revs = json.load(open(path))["device"]["provenance"]["library_revisions"]
base = os.environ["PYAUTO_PRIVATE_BASE"]
for repo in ("PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens"):
    head = subprocess.check_output(["git", "-C", f"{base}/{repo}", "rev-parse", "HEAD"], text=True).strip()
    if revs.get(repo) != head:
        raise SystemExit(f"MISMATCH {path}: {repo} row {revs.get(repo)} != private checkout {head}")
print("revisions OK", path)
EOF_PY
}

# Simulate a missing dataset and cache its adapt image once per instrument, under a lock.
prepare_dataset() {
python3 - "$1" <<'EOF_PY'
import sys
sys.path.insert(0, "scripts/misc/campaigns")
from @MODULE@ import prepare_dataset
prepare_dataset(sys.argv[1])
EOF_PY
}

# One leg: fresh JAX compile cache, the leg's platform, wall clock logged, revisions checked.
# A failed leg is recorded and the job carries on (|| keeps activate.sh's ERR trap out of it).
STATUS=0
run_leg() {
    local I=$1
    local LOG="$AP_ROOT/output/@LOGS@/@NAME@_${SLURM_JOB_ID:-local}_${I}.log"
    export JAX_PLATFORMS=${PLATFORMS[$I]}
    if [ "$JAX_PLATFORMS" = cpu ]; then
        export XLA_FLAGS="$XLA_FLAGS_BASE $XLA_CPU_WORKAROUND_FLAG"
    else
        export XLA_FLAGS="$XLA_FLAGS_BASE"
    fi
    export JAX_COMPILATION_CACHE_DIR="$AP_ROOT/output/jax_cache/@PREFIX@_${SLURM_JOB_ID:-local}_${I}"
    rm -rf "$JAX_COMPILATION_CACHE_DIR" && mkdir -p "$JAX_COMPILATION_CACHE_DIR"
    prepare_dataset ${INSTS[$I]} || { echo "LEG $I: dataset preparation FAILED"; STATUS=5; return 0; }
    echo "------------------------------------------"
    echo "LEG $I: ${CELLNAMES[$I]}"
    echo "  cell script: scripts/${CELLS[$I]}.py ${FLAGS[$I]} -> $RESULTS_ROOT/${OUTS[$I]}"
    echo "  JAX_PLATFORMS=$JAX_PLATFORMS  NPROC=$NPROC  log=$LOG"
    echo "  XLA_FLAGS=$XLA_FLAGS"
@QUIET_CALL@    local T0=$(date +%s)
    # A START line first, so a leg killed at --time (which never reaches its WALL line) still shows.
    printf 'START\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$I" "$T0" "${CELLNAMES[$I]}" >> "$AP_ROOT/output/@LOGS@/walls.tsv"
    python3 -u scripts/${CELLS[$I]}.py --instrument ${INSTS[$I]} ${FLAGS[$I]} --output-path "$RESULTS_ROOT/${OUTS[$I]}" > "$LOG" 2>&1 \
        || { echo "LEG $I FAILED (rc $?), see $LOG"; tail -20 "$LOG"; STATUS=4; }
    local WALL=$(( $(date +%s) - T0 ))
    printf 'WALL\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$I" "$WALL" "${CELLNAMES[$I]}" | tee -a "$AP_ROOT/output/@LOGS@/walls.tsv"
    if [ -f "$RESULTS_ROOT/${OUTS[$I]}" ]; then
        check_revisions "$RESULTS_ROOT/${OUTS[$I]}" || STATUS=4
    else
        echo "LEG $I: no row at $RESULTS_ROOT/${OUTS[$I]}"; STATUS=4
    fi
}
"""


#: The quiet-node gate of a rerun whose rows must qualify under the dashboard's load cap
#: (``hpc/release_sweep.conf`` RELEASE_SWEEP_LOADAVG_CAP, judged on the 1-minute load average the
#: row records as ``loadavg_at_import``, which ``_profile_cli`` samples when the cell script
#: imports it, before the leg's own compile and timing work). The job holds its node
#: (``#SBATCH --exclusive``), pins itself to the leg's cores so the leg sees the 8-CPU allocation
#: of the run it repeats, and starts each leg only once the node's 1-minute load has decayed to
#: ``load_max`` (the previous tenant's work, or the dataset preflight, can leave it high).
QUIET_DEF = r"""# QUIET NODE. This job holds its node exclusively (#SBATCH --exclusive), so the leg is the only
# work on it. Every process is pinned to cores 0..$((SLURM_CPUS_PER_TASK-1)) (taskset), the
# allocation size of the run this repeats, and each leg starts only once the 1-minute load
# average (recorded by the row as loadavg_at_import and refused by the dashboard above
# RELEASE_SWEEP_LOADAVG_CAP) is at most @LOAD_MAX@: polled every @POLL@ s for at most @WAIT@ s,
# after which the leg runs anyway and its row shows the load it started at.
echo "exclusive: SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}  node cpus=$(nproc --all)  cpuset=$(nproc)"
taskset -cp "0-$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))" $$ \
    || { echo "REFUSING: cannot pin to ${SLURM_CPUS_PER_TASK:-8} cores (is the node exclusive?)"; exit 6; }
quiet_wait() {
    local waited=0 load1
    while :; do
        load1=$(cut -d' ' -f1 /proc/loadavg)
        if awk -v l="$load1" -v m=@LOAD_MAX@ 'BEGIN { exit !(l <= m) }'; then
            echo "QUIET: 1-min load $load1 <= @LOAD_MAX@ after ${waited} s ($(cat /proc/loadavg))"
            return 0
        fi
        if [ "$waited" -ge @WAIT@ ]; then
            echo "QUIET-WAIT TIMEOUT: 1-min load $load1 > @LOAD_MAX@ after ${waited} s; running anyway"
            return 0
        fi
        sleep @POLL@
        waited=$((waited + @POLL@))
    done
}
"""


@dataclass(frozen=True)
class QuietGate:
    """The quiet-node gate's knobs (:data:`QUIET_DEF`)."""

    load_max: float = 2.0
    wait_s: int = 1200
    poll_s: int = 15

    def text(self) -> str:
        return (
            QUIET_DEF.replace("@LOAD_MAX@", f"{self.load_max:g}")
            .replace("@WAIT@", str(self.wait_s))
            .replace("@POLL@", str(self.poll_s))
        )


@dataclass(frozen=True)
class Campaign:
    """The campaign-specific tokens of the preamble and guards."""

    pin_var: str  # e.g. P2A_PIN
    pin: str
    hpc_base: str
    private_base: str
    logs: str  # output/<logs>/
    prefix: str  # numba / jax cache prefix
    module: str  # the generator module holding prepare_dataset
    allow_var: str  # e.g. P2A_ALLOW_PROVISIONAL_WALL
    private_var: str = "PYAUTO_PRIVATE_BASE"  # env var overriding the private base
    shas: dict = field(default_factory=dict)  # non-empty -> SHA guard instead of tag guard
    quiet: QuietGate | None = None  # set -> the quiet-node gate (exclusive node, load wait)


def preamble(c: Campaign, name: str) -> str:
    """The submit body before the manifest arrays, for campaign ``c`` and submit ``name``."""
    guard = SHA_GUARD.replace("@SHAS@", json.dumps(c.shas)) if c.shas else TAG_GUARD
    return (
        PREAMBLE.replace("@QUIET_DEF@", c.quiet.text() if c.quiet else "")
        .replace("@QUIET_CALL@", "    quiet_wait\n" if c.quiet else "")
        .replace("@IMPORT_GUARD@", guard)
        .replace("@HPC_BASE@", c.hpc_base)
        .replace("@PRIVATE_VAR@", c.private_var)
        .replace("@PRIVATE_BASE@", c.private_base)
        .replace("@PINVAR@", c.pin_var)
        .replace("@PIN@", c.pin)
        .replace("@LOGS@", c.logs)
        .replace("@PREFIX@", c.prefix)
        .replace("@MODULE@", c.module)
        .replace("@NAME@", name)
        .replace("@XLA_CPU_WORKAROUND_FLAG@", XLA_CPU_WORKAROUND_FLAG)
    )


PROVISIONAL_GUARD = r"""
# Batch 1 runs only once batch 0's walls have replaced the provisional WALL-BASIS rows above
# (source: measured-wall, --time from the measurement) and this reads "measured".
WALL_BASIS_STATUS=@WALL_BASIS_STATUS@
if [ "$WALL_BASIS_STATUS" != measured ] && [ "${@ALLOW_VAR@:-0}" != 1 ]; then
    echo "REFUSING: WALL_BASIS_STATUS=$WALL_BASIS_STATUS. Run batch 0, set --time from its walls"
    echo "and mark the WALL-BASIS rows measured-wall first (scripts/misc/campaigns/README.md)."
    exit 2
fi
"""


def provisional_guard(status: str, allow_var: str) -> str:
    return (
        PROVISIONAL_GUARD.strip("\n")
        .replace("@WALL_BASIS_STATUS@", status)
        .replace("@ALLOW_VAR@", allow_var)
    )


# --------------------------------------------------------------------------- WALL-BASIS rows


@dataclass(frozen=True)
class WallBasis:
    """How a campaign's batch-1 rows cite batch 0.

    ``measured_wall(task) -> (seconds, ref)``; ``basis_leg(task)`` the batch-0 leg bounding it;
    ``limitations`` {cell: reason} of known-limitation legs kept in; ``walls`` the batch-0 table
    (cell -> (wall, elapsed, ref)); ``headroom`` the measured rows' declared headroom.
    """

    measured_wall: Callable
    basis_leg: Callable
    limitations: dict
    walls: dict
    headroom: float


def batch_wall_s(tasks, basis: WallBasis, *, sequential: bool) -> int:
    """The measured wall a submit must hold: its slowest leg (array) or their sum (sequential)."""
    walls = [basis.measured_wall(t)[0] for t in tasks]
    return sum(walls) if sequential else max(walls)


def wall_rows(
    tasks,
    device: str,
    guess: int | None,
    *,
    measured: bool,
    sequential: bool,
    basis: WallBasis | None,
) -> list[str]:
    """One WALL-BASIS row per (gate cell, instrument). Batch 0 declares its guess; batch 1 cites
    the batch-0 legs that bound it: the slowest leg of an array task, or the sum of the legs a
    sequential job runs. Prose lines carry no ``key: value`` pairs."""
    rows = []
    for gate_cell, instrument in sorted({(t.gate_cell, t.instrument) for t in tasks}):
        legs = [t for t in tasks if (t.gate_cell, t.instrument) == (gate_cell, instrument)]
        names = ", ".join(f"{t.variant} {t.setup_id} {t.measurement}" for t in legs)
        rows.append(f"#   {len(legs)} leg(s): {names}")
        for t in legs if measured else []:
            if t.cell in basis.limitations:
                ref = basis.walls[t.cell][2]
                rows.append(
                    f"#   KNOWN LIMITATION, kept in - {t.variant} {t.setup_id} {t.measurement} "
                    f"({basis.limitations[t.cell]}; batch 0 task {ref}); it fails fast"
                )
        rows.append(f"#   cell: {gate_cell}/{instrument}  device: {device}  precision: fp64")
        if measured:
            wall = batch_wall_s(legs, basis, sequential=sequential)
            refs = ",".join(sorted({basis.measured_wall(t)[1] for t in legs}))
            on = ",".join(sorted({basis.basis_leg(t).instrument for t in legs}))
            rows.append(
                f"#   source: measured-wall  wall: {wall}  ref: {refs}  measured-on: {on}  "
                f"headroom: {basis.headroom}  median: included"
            )
        elif guess is None:
            rows.append("#   source: unmeasured  probe-first: yes")
        else:
            rows.append(f"#   source: unmeasured  probe-first: yes  wall: {guess}  headroom: 3.0")
    return rows


# --------------------------------------------------------------------------- assembly


def sbatch_block(
    job: str,
    kind: str,
    n: int,
    *,
    mem: int,
    time: str,
    throttle: str | None = None,
    exclusive: bool = False,
) -> list[str]:
    """The ``#SBATCH`` lines. ``kind``: ``ral_cpu`` (array, ``throttle`` e.g. ``%8``),
    ``a100_job`` (one GPU job) or ``a100_array`` (``%2``). ``exclusive`` holds whole nodes."""
    lines = [f"#SBATCH -J {job}"]
    if kind == "ral_cpu":
        lines += ["#SBATCH --partition=ral", f"#SBATCH --array=0-{n - 1}{throttle}"]
    else:
        lines += ["#SBATCH --partition=gpu", "#SBATCH --gres=gpu:1"]
        if kind == "a100_array":
            lines += [f"#SBATCH --array=0-{n - 1}%2"]
    if exclusive:
        lines += ["#SBATCH --exclusive"]
    arrayed = "array" in kind or kind == "ral_cpu"
    lines += [
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --mem={mem}gb",
        f"#SBATCH --time={time}",
        "#SBATCH -o output/output.%A_%a.out" if arrayed else "#SBATCH -o output/output.%A.out",
        "#SBATCH -e error/error.%A_%a.err" if arrayed else "#SBATCH -e error/error.%A.err",
    ]
    return lines


def manifest_arrays(tasks, results_root: str, *, module: str) -> list[str]:
    return [
        f"# The manifest: one entry per leg ({module}.Task).",
        bash_array("CELLNAMES", [t.cell for t in tasks], quote=True),
        bash_array("CELLS", [t.script for t in tasks]),
        bash_array("INSTS", [t.instrument for t in tasks]),
        bash_array("FLAGS", [t.flags for t in tasks], quote=True),
        bash_array("PLATFORMS", [t.platform for t in tasks]),
        bash_array("OUTS", [t.output for t in tasks]),
        f'RESULTS_ROOT="{results_root}"',
    ]


def run_block(kind: str) -> list[str]:
    if kind == "a100_job":
        return ["nvidia-smi || true", 'for I in "${!CELLS[@]}"; do', "    run_leg $I", "done"]
    return [
        'I=${SLURM_ARRAY_TASK_ID:?"submit as an array (sbatch from hpc/<batch>/)"}',
        *(["nvidia-smi || true"] if kind == "a100_array" else []),
        "run_leg $I",
    ]


TAIL = [
    'echo "=========================================="',
    'echo "exit status: $STATUS (0 ok, 4 a leg failed or its revisions/row check failed, 5 dataset)"',
    'echo "loadavg: $(cat /proc/loadavg)"',
    "date",
    "exit $STATUS",
]


def assemble(
    header: list[str],
    sbatch: list[str],
    guard: str | None,
    body: str,
    arrays: list[str],
    run: list[str],
) -> str:
    """The whole submit text from its parts."""
    parts = [
        "\n".join(header),
        "",
        "\n".join(sbatch),
        "",
        "# Submit from hpc/batch_cpu/ or hpc/batch_gpu/ (the -o/-e paths are relative).",
        *([guard] if guard is not None else []),
        body.rstrip("\n"),
        "",
        "\n".join(arrays),
        "",
        "\n".join(run),
        "",
        "\n".join(TAIL),
        "",
    ]
    return "\n".join(parts)


def write_or_check(rendered: dict, root, *, args, generator: str) -> int:
    """``--write`` renders every submit; ``--check`` lists the stale ones (exit 1 if any)."""
    stale = []
    for rel, text in rendered.items():
        path = root / rel
        if args.write:
            path.write_text(text)
            path.chmod(0o755)
            print(f"wrote {rel}")
        elif not path.is_file() or path.read_text() != text:
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel} (run scripts/misc/campaigns/{generator} --write)")
    return 1 if stale else 0
