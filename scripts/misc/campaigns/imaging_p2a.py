"""Likelihood-lifecycle phase 2a: the imaging campaign's task manifest and its SLURM submits.

autolens_profiling#456 re-measures the 106 planned imaging cells (``imaging/{mge, rectangular,
delaunay, delaunay_nn}``; ``dashboard/coverage.json``) at the standard size on the pinned release
2026.10.23.1, from a private RAL checkout. This module is the one copy of that work list: every
cell becomes one :class:`Task` (script, flags, output path), and the submits under ``hpc/`` are
rendered from the tasks, so the manifest the tests check and the bash arrays the jobs run cannot
drift apart::

    python scripts/misc/campaigns/imaging_p2a.py --write   # (re)render the submits
    python scripts/misc/campaigns/imaging_p2a.py --check   # exit 1 if a submit is stale
    python scripts/misc/campaigns/imaging_p2a.py --list    # print the manifest

The submits (README.md beside this file has the batch procedure):

- batch 1, ``hpc/batch_cpu/submit_campaign_p2a_imaging_ral_cpu_fp64`` — one ``--partition=ral``
  array over the 44 ``hpc_ral_cpu`` cells, one cell per task;
- batch 1, ``hpc/batch_gpu/submit_campaign_p2a_imaging_a100_<instrument>_fp64`` — one
  ``--gres=gpu:1`` job per instrument running that instrument's ``hpc_a100`` cells in turn,
  including the ``jax_cpu_sparse_fp64@hpc_a100`` legs, which run on the GPU node's CPU inside
  the GPU job (never a CPU array on the ``gpu`` partition);
- batch 0, ``..._batch0_ral_cpu_fp64`` and ``..._batch0_a100_fp64`` — the timing arms that set
  batch 1's ``--time`` (:data:`BATCH0_INSTRUMENTS`), writing their rows into the gitignored
  ``output/p2a_batch0/`` scratch tree.

Stdlib only; nothing here imports the modelling stack except :func:`prepare_dataset`, which the
submits call to simulate a missing dataset and cache its adapt image under a lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _script_routes import load_routes  # noqa: E402

#: The release every cell is measured on (human ruling 2026-10-24), and where the private RAL
#: checkout of the five libraries at that tag lives (override with ``PYAUTO_PRIVATE_BASE``).
PIN = "2026.10.23.1"
PRIVATE_BASE = f"/mnt/ral/jnightin/profiling/PyAuto_{PIN}"
#: The shared virtualenv (third-party packages only; hpc_ral.md, ``activate.sh``).
HPC_BASE = "/mnt/ral/jnightin/PyAuto"
ISSUE = "autolens_profiling#456"

INSTRUMENTS = ("ao", "euclid", "hst", "jwst")
VARIANT_INSTRUMENTS = {
    "mge": INSTRUMENTS,
    "rectangular": INSTRUMENTS,
    "delaunay": INSTRUMENTS,
    "delaunay_nn": ("hst",),
}
PIXELIZED = ("rectangular", "delaunay", "delaunay_nn")
#: §3.3 provisional designations (``catalogue/provisional_designations.json``), optimal and
#: secondary, per host class.
SETUPS = {
    "parametric": {
        "hpc_ral_cpu": ("jax_cpu_dense_fp64",),
        "hpc_a100": ("jax_gpu_dense_fp64",),
    },
    "pixelized": {
        "hpc_ral_cpu": ("numba_cpu_sparse_fp64", "jax_cpu_sparse_fp64"),
        "hpc_a100": ("jax_gpu_dense_fp64", "jax_gpu_sparse_fp64", "jax_cpu_sparse_fp64"),
    },
}
MEASUREMENTS = ("runtime", "breakdown")
#: ``--config-name`` per (host class, device): the host tier a row's witness judges against.
CONFIG_NAMES = {
    ("hpc_ral_cpu", "cpu"): "hpc_ral_cpu_fp64",
    ("hpc_a100", "gpu"): "hpc_a100_fp64",
    ("hpc_a100", "cpu"): "hpc_a100_cpu_fp64",
}

#: Batch 0 measures the instrument that bounds every other one (masked pixels at the 3.5" mask
#: scale as 1 / pixel_scale^2: euclid ~3.8k < hst ~15k < jwst ~43k < ao ~385k), plus hst for the
#: delaunay_nn cells, which exist only there. A cell's wall on ao bounds the same script on the
#: smaller instruments, so batch 1's ``--time`` rests on a measurement, not a guess.
BATCH0_INSTRUMENTS = {
    "mge": ("ao",),
    "rectangular": ("ao",),
    "delaunay": ("ao",),
    "delaunay_nn": ("hst",),
}
#: Batch 0's declared per-task wall guess (``source: unmeasured``, 3x floor) and its ``--time``.
BATCH0_WALL_GUESS_S = 14000
BATCH0_TIME = "12:00:00"
#: Batch 1's placeholder ``--time`` until batch 0's walls replace the WALL-BASIS rows.
BATCH1_TIME = {"ral_cpu": "12:00:00", "a100": "2-00:00:00"}

#: Pixel scale per instrument (arcsec / pixel; ``scripts/misc/simulators/imaging.INSTRUMENTS``,
#: pinned equal by the campaign test) and the cells' 3.5" circular mask: the masked image size
#: every cell's memory scales with.
PIXEL_SCALES = {"ao": 0.01, "euclid": 0.1, "hst": 0.05, "jwst": 0.03}
MASK_RADIUS = 3.5
#: Columns of the dense mapping matrix: 1280 Delaunay vertices (rectangular 1225) plus up to 60
#: linear MGE Gaussians.
MAPPING_COLUMNS = 1340
MEM_STEPS_GB = (16, 32, 64, 128, 256)


def masked_pixels(instrument: str) -> int:
    """Image pixels inside the 3.5" mask: pi r^2 / pixel_scale^2 (ao ~385k, euclid ~3.8k)."""
    import math

    return int(math.pi * MASK_RADIUS**2 / PIXEL_SCALES[instrument] ** 2)


def mem_gb(instruments, minimum: int = 16) -> int:
    """``--mem`` for a job over ``instruments``: the largest instrument decides.

    Estimated peak = 4 GB (Python, JAX, numba, libraries) + eight fp64 arrays the size of the
    dense mapping matrix (masked pixels x 1340 columns): the mapping and blurred mapping
    matrices of the cell, the same two of the eager NumPy dense parity reference, and JAX /
    numba working copies. The request is the next step of 16 / 32 / 64 / 128 / 256 GB at or above
    twice that peak. An estimate from the dataset sizes, not a measurement: batch 0's
    ``sacct --format=MaxRSS`` replaces it before batch 1.
    """
    peak = max(4.0 + 8 * masked_pixels(i) * MAPPING_COLUMNS * 8 / 1e9 for i in instruments)
    return next(step for step in MEM_STEPS_GB if step >= max(minimum, 2 * peak))


_ROUTES = {row["path"]: row["legacy"] for row in load_routes(ROOT)["routes"]}


@dataclass(frozen=True, order=True)
class Task:
    """One planned cell: ``imaging/<variant>/<instrument> <setup_id>@<host_class> [<measurement>]``."""

    variant: str
    instrument: str
    setup_id: str
    host_class: str
    measurement: str

    @property
    def backend(self) -> str:
        return self.setup_id.split("_")[0]

    @property
    def device(self) -> str:
        return self.setup_id.split("_")[1]

    @property
    def operator(self) -> str:
        return self.setup_id.split("_")[2]

    @property
    def cell(self) -> str:
        """The coverage-matrix cell name (``dashboard/coverage.json`` ``cells[].cell``)."""
        return (
            f"imaging/{self.variant}/{self.instrument} {self.setup_id}@{self.host_class} "
            f"[{self.measurement}]"
        )

    @property
    def script(self) -> str:
        """The cell script, relative to ``scripts/`` and without ``.py``."""
        if self.backend == "numba":
            folder = "rectangular" if self.variant == "rectangular" else "delaunay"
            return f"imaging/{folder}/likelihood_{self.measurement}_numba"
        return f"imaging/{self.variant}/likelihood_{self.measurement}"

    @property
    def gate_cell(self) -> str:
        """The WALL-BASIS cell id: ``imaging/<legacy script stem>`` (``wall/check_submits.py``)."""
        legacy = _ROUTES[f"scripts/{self.script}.py"]
        return f"imaging/{Path(legacy).stem}"

    @property
    def config_name(self) -> str:
        return CONFIG_NAMES[(self.host_class, self.device)]

    @property
    def flags(self) -> str:
        """Every flag but ``--instrument`` and ``--output-path`` (the submits add those)."""
        flags = []
        if self.variant in PIXELIZED:
            flags.append("--standard-size")
        if self.backend == "jax" and self.operator == "sparse":
            flags.append("--sparse")
        if self.backend == "numba" and self.variant == "delaunay_nn":
            flags += ["--mesh-class", "delaunay_nn"]
        flags += ["--config-name", self.config_name]
        return " ".join(flags)

    @property
    def output(self) -> str:
        """The row's mirrored-tree path, relative to the results root."""
        return (
            f"results/imaging/{self.variant}/{self.instrument}/{self.setup_id}/"
            f"{self.host_class}/{self.measurement}.json"
        )

    @property
    def platform(self) -> str:
        """``JAX_PLATFORMS`` for the leg (a numba cell never touches a JAX device)."""
        return "cuda,cpu" if self.device == "gpu" else "cpu"


def planned_tasks() -> list[Task]:
    """Every planned imaging cell of the campaign, sorted (106: 44 RAL CPU + 62 A100)."""
    tasks = []
    for variant, instruments in VARIANT_INSTRUMENTS.items():
        table = SETUPS["pixelized" if variant in PIXELIZED else "parametric"]
        for instrument in instruments:
            for host_class, setups in table.items():
                for setup_id in setups:
                    for measurement in MEASUREMENTS:
                        tasks.append(Task(variant, instrument, setup_id, host_class, measurement))
    return sorted(tasks)


def batch1_ral_cpu(instrument: str | None = None) -> list[Task]:
    return [
        t
        for t in planned_tasks()
        if t.host_class == "hpc_ral_cpu" and instrument in (None, t.instrument)
    ]


def batch1_a100(instrument: str) -> list[Task]:
    return [t for t in planned_tasks() if t.host_class == "hpc_a100" and t.instrument == instrument]


def _batch0(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t.instrument in BATCH0_INSTRUMENTS[t.variant]]


def batch0_ral_cpu() -> list[Task]:
    return _batch0(batch1_ral_cpu())


def batch0_a100() -> list[Task]:
    return _batch0([t for t in planned_tasks() if t.host_class == "hpc_a100"])


# --------------------------------------------------------------------------- dataset preflight


def prepare_dataset(instrument: str, root: Path = ROOT) -> None:
    """Simulate ``dataset/imaging/<instrument>`` if missing and cache its adapt image, under a lock.

    Array tasks of one instrument start together; without the lock two of them would simulate
    the dataset (or write ``lensed_source.fits``) at once and a third could read a half-written
    file. The adapt image is built the way the JAX cells build theirs (3.5" circular mask,
    light-profile bins [4, 2, 2] at [0.3, 0.6]) so the cache does not depend on which cell
    happens to run first.
    """
    import fcntl

    lock_dir = root / "output"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"p2a_dataset_{instrument}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dataset_path = root / "dataset" / "imaging" / instrument
        if (dataset_path / "lensed_source.fits").exists():
            return
        sys.path.insert(0, str(root / "scripts" / "misc"))
        import autolens as al
        from simulators.imaging import INSTRUMENTS as SIMULATED

        from _adapt_image_util import adapt_image_for_dataset
        from _profile_cli import auto_simulate_if_missing

        auto_simulate_if_missing(
            dataset_path, dataset_type="imaging", instrument=instrument, workspace_root=root
        )
        dataset = al.Imaging.from_fits(
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            pixel_scales=SIMULATED[instrument]["pixel_scale"],
        )
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.5
        )
        dataset = dataset.apply_mask(mask=mask)
        dataset = dataset.apply_over_sampling(
            over_sample_size_lp=al.util.over_sample.over_sample_size_via_radial_bins_from(
                grid=dataset.grid,
                sub_size_list=[4, 2, 2],
                radial_list=[0.3, 0.6],
                centre_list=[(0.0, 0.0)],
            ),
            over_sample_size_pixelization=1,
        )
        adapt_image_for_dataset(dataset_path=dataset_path, dataset=dataset)


# --------------------------------------------------------------------------- submit rendering


def _wall_rows(tasks: list[Task], device: str, guess: int | None) -> list[str]:
    rows = []
    for gate_cell, instrument in sorted({(t.gate_cell, t.instrument) for t in tasks}):
        legs = [t for t in tasks if (t.gate_cell, t.instrument) == (gate_cell, instrument)]
        names = ", ".join(f"{t.variant} {t.setup_id} {t.measurement}" for t in legs)
        rows.append(f"#   {len(legs)} leg(s): {names}")
        rows.append(f"#   cell: {gate_cell}/{instrument}  device: {device}  precision: fp64")
        if guess is None:
            rows.append("#   source: unmeasured  probe-first: yes")
        else:
            rows.append(f"#   source: unmeasured  probe-first: yes  wall: {guess}  headroom: 3.0")
    return rows


def _bash_array(name: str, values: list[str], *, quote: bool = False) -> str:
    """A bash array, one value per line. Values the wall gate resolves (cell scripts,
    instruments) stay bare: ``check_submits.resolve_var`` splits an array on whitespace."""
    body = "\n".join(f"    {json.dumps(v) if quote else v}" for v in values)
    return f"{name}=(\n{body}\n)"


_PREAMBLE = r"""export AP_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/../.." && pwd)"
export PYAUTO_HPC_BASE="${PYAUTO_HPC_BASE:-@HPC_BASE@}"
export PYAUTO_PRIVATE_BASE="${PYAUTO_PRIVATE_BASE:-@PRIVATE_BASE@}"
export P2A_PIN=@PIN@
echo "AP_ROOT=$AP_ROOT  PYAUTO_PRIVATE_BASE=$PYAUTO_PRIVATE_BASE  pin=$P2A_PIN"
source "$AP_ROOT/activate.sh"
export PYTHONPATH=$PYAUTO_PRIVATE_BASE/PyAutoNerves:$PYAUTO_PRIVATE_BASE/PyAutoFit:$PYAUTO_PRIVATE_BASE/PyAutoArray:$PYAUTO_PRIVATE_BASE/PyAutoGalaxy:$PYAUTO_PRIVATE_BASE/PyAutoLens:$PYTHONPATH

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NPROC=${SLURM_CPUS_PER_TASK:-8}
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"
unset PYAUTO_SMALL_DATASETS
export NUMBA_CACHE_DIR="$AP_ROOT/output/numba_cache/p2a_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$AP_ROOT/output/p2a_logs"

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
# The libraries must import from the private checkout at the pinned tag, never the shared mirror.
python3 - <<'EOF_PY' || exit 3
import os, subprocess
import autoarray, autofit, autogalaxy, autolens, autonerves
base, pin = os.path.realpath(os.environ["PYAUTO_PRIVATE_BASE"]), os.environ["P2A_PIN"]
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
from imaging_p2a import prepare_dataset
prepare_dataset(sys.argv[1])
EOF_PY
}

# One leg: fresh JAX compile cache, the leg's platform, wall clock logged, revisions checked.
# A failed leg is recorded and the job carries on (|| keeps activate.sh's ERR trap out of it).
STATUS=0
run_leg() {
    local I=$1
    local LOG="$AP_ROOT/output/p2a_logs/@NAME@_${SLURM_JOB_ID:-local}_${I}.log"
    export JAX_PLATFORMS=${PLATFORMS[$I]}
    export JAX_COMPILATION_CACHE_DIR="$AP_ROOT/output/jax_cache/p2a_${SLURM_JOB_ID:-local}_${I}"
    rm -rf "$JAX_COMPILATION_CACHE_DIR" && mkdir -p "$JAX_COMPILATION_CACHE_DIR"
    prepare_dataset ${INSTS[$I]} || { echo "LEG $I: dataset preparation FAILED"; STATUS=5; return 0; }
    echo "------------------------------------------"
    echo "LEG $I: ${CELLNAMES[$I]}"
    echo "  cell script: scripts/${CELLS[$I]}.py ${FLAGS[$I]} -> $RESULTS_ROOT/${OUTS[$I]}"
    echo "  JAX_PLATFORMS=$JAX_PLATFORMS  NPROC=$NPROC  log=$LOG"
    local T0=$(date +%s)
    # A START line first, so a leg killed at --time (which never reaches its WALL line) still shows.
    printf 'START\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$I" "$T0" "${CELLNAMES[$I]}" >> "$AP_ROOT/output/p2a_logs/walls.tsv"
    python3 -u scripts/${CELLS[$I]}.py --instrument ${INSTS[$I]} ${FLAGS[$I]} --output-path "$RESULTS_ROOT/${OUTS[$I]}" > "$LOG" 2>&1 \
        || { echo "LEG $I FAILED (rc $?), see $LOG"; tail -20 "$LOG"; STATUS=4; }
    local WALL=$(( $(date +%s) - T0 ))
    printf 'WALL\t%s\t%s\t%s\t%s\n' "${SLURM_JOB_ID:-local}" "$I" "$WALL" "${CELLNAMES[$I]}" | tee -a "$AP_ROOT/output/p2a_logs/walls.tsv"
    if [ -f "$RESULTS_ROOT/${OUTS[$I]}" ]; then
        check_revisions "$RESULTS_ROOT/${OUTS[$I]}" || STATUS=4
    else
        echo "LEG $I: no row at $RESULTS_ROOT/${OUTS[$I]}"; STATUS=4
    fi
}
"""

_PROVISIONAL_GUARD = r"""
# Batch 1 runs only once batch 0's walls have replaced the provisional WALL-BASIS rows above
# (source: measured-wall, --time from the measurement) and this flips to "measured".
WALL_BASIS_STATUS=provisional
if [ "$WALL_BASIS_STATUS" != measured ] && [ "${P2A_ALLOW_PROVISIONAL_WALL:-0}" != 1 ]; then
    echo "REFUSING: WALL_BASIS_STATUS=$WALL_BASIS_STATUS. Run batch 0, set --time from its walls"
    echo "and mark the WALL-BASIS rows measured-wall first (scripts/misc/campaigns/README.md)."
    exit 2
fi
"""


def render(name: str, tasks: list[Task], *, kind: str, batch: int, title: str) -> str:
    """One submit script. ``kind``: ``ral_cpu`` (array) / ``a100_job`` / ``a100_array``."""
    device = "ral_cpu" if kind == "ral_cpu" else "a100"
    guess = BATCH0_WALL_GUESS_S if batch == 0 else None
    time = BATCH0_TIME if batch == 0 else BATCH1_TIME[device]
    n = len(tasks)
    # Never in place over committed rows: PR B copies complete rows into results/ (README).
    results_root = f"$AP_ROOT/output/p2a_batch{batch}"
    mem = mem_gb({t.instrument for t in tasks}, minimum=16 if kind == "ral_cpu" else 32)
    header = [
        "#!/bin/bash -l",
        "#",
        f"# {title}",
        f"# Likelihood-lifecycle phase 2a imaging campaign ({ISSUE}), batch {batch}: {n} cell(s).",
        "# GENERATED by scripts/misc/campaigns/imaging_p2a.py --write; edit the generator, not this",
        "# file (the campaign test fails on a hand edit). The batch procedure is in",
        "# scripts/misc/campaigns/README.md.",
        "#",
        f"# LIBRARY CODE. A private checkout of the five libraries at tag {PIN} ($PYAUTO_PRIVATE_BASE,",
        f"# default {PRIVATE_BASE}) is prepended to PYTHONPATH over the shared venv",
        f"# ($PYAUTO_HPC_BASE, default {HPC_BASE}, third-party packages only). The job refuses to",
        "# run if a library imports from anywhere else or sits at another tag, and after each leg it",
        "# checks the row's library_revisions equal the checkout's HEADs. The shared mirror is never",
        "# pulled or used. AP_ROOT is the checkout this was submitted from (hpc/<batch>/../..).",
        "#",
        "# CELLS. Every leg runs one cell script at the standard size (--standard-size: Delaunay",
        "# 1250 Hilbert vertices, rectangular 35x35; MGE has no mesh) and writes its row, fit.png and",
        "# parity.json into the mirrored cell named by --output-path under the gitignored",
        f"# output/p2a_batch{batch}/ scratch tree, never in place over committed rows"
        + (" (PR B copies complete rows into results/)." if batch == 1 else "."),
        "#",
        f"# MEMORY. --mem={mem_gb({t.instrument for t in tasks}, minimum=16 if kind == 'ral_cpu' else 32)}gb from the dataset size "
        f"({', '.join(sorted({t.instrument for t in tasks}))}; masked pixels "
        + ", ".join(f"{i} {masked_pixels(i)}" for i in sorted({t.instrument for t in tasks}))
        + "):",
        "# 4 GB + eight fp64 arrays of masked pixels x 1340 mapping columns, doubled, rounded up",
        "# to 16/32/64/128 GB (imaging_p2a.mem_gb). batch 0's sacct MaxRSS replaces the estimate.",
    ]
    if kind == "ral_cpu":
        header += [
            "# One cell per array task on `ral` (CPU work never goes on `gpu`). numba cells pin their",
            "# BLAS threads to 1 (production per-worker cost); JAX CPU cells get NPROC=8.",
        ]
    elif kind == "a100_job":
        header += [
            "# One A100 job runs this instrument's hpc_a100 cells in turn. The jax_cpu_sparse legs",
            "# run on the GPU node's CPU inside this GPU job (JAX_PLATFORMS=cpu, NPROC=8) — the",
            "# hpc_a100 host class's CPU setup, never a CPU array on the gpu partition.",
        ]
    else:
        header += [
            "# One leg per array task, each with --gres=gpu:1, throttled to %2 so the timing arm",
            "# holds at most two A100s. The jax_cpu_sparse legs run on the GPU node's CPU inside",
            "# their GPU task (JAX_PLATFORMS=cpu, NPROC=8).",
        ]
    header += ["#"]
    if batch == 0:
        header += [
            "# BATCH 0 = TIMING ARM. It runs the cells that bound every other instrument's wall (ao,",
            "# the most masked pixels by far, plus hst for delaunay_nn, which exists only there).",
            "# Each leg's wall lands in output/p2a_logs/walls.tsv; batch 1's WALL-BASIS rows cite",
            "# them (source: measured-wall, ref: this job) and its --time is set from them.",
            "#",
            f"# WALL-BASIS: nothing measured at the standard size yet; {BATCH0_WALL_GUESS_S} s per leg is",
            "# a declared guess (3x floor), and the ao legs may run well past it — the timing arm",
            "# exists to find out.",
        ]
    else:
        header += [
            "# WALL-BASIS: PROVISIONAL. No cell has a standard-size wall yet; batch 0 measures them.",
            "# Every row is `unmeasured` until batch 0 lands, and the job refuses to run while",
            "# WALL_BASIS_STATUS=provisional (export P2A_ALLOW_PROVISIONAL_WALL=1 overrides it).",
        ]
    header += _wall_rows(tasks, device, guess)
    sbatch = [f"#SBATCH -J p2a_{'b0' if batch == 0 else 'b1'}_{name.split('imaging_', 1)[1]}"]
    if kind == "ral_cpu":
        throttle = "%8" if batch == 0 else "%16"
        sbatch += [
            "#SBATCH --partition=ral",
            f"#SBATCH --array=0-{n - 1}{throttle}",
        ]
    else:
        sbatch += ["#SBATCH --partition=gpu", "#SBATCH --gres=gpu:1"]
        if kind == "a100_array":
            sbatch += [f"#SBATCH --array=0-{n - 1}%2"]
    sbatch += [
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --mem={mem}gb",
        f"#SBATCH --time={time}",
        "#SBATCH -o output/output.%A_%a.out"
        if "array" in kind or kind == "ral_cpu"
        else "#SBATCH -o output/output.%A.out",
        "#SBATCH -e error/error.%A_%a.err"
        if "array" in kind or kind == "ral_cpu"
        else "#SBATCH -e error/error.%A.err",
    ]
    arrays = [
        "# The manifest: one entry per leg (imaging_p2a.Task).",
        _bash_array("CELLNAMES", [t.cell for t in tasks], quote=True),
        _bash_array("CELLS", [t.script for t in tasks]),
        _bash_array("INSTS", [t.instrument for t in tasks]),
        _bash_array("FLAGS", [t.flags for t in tasks], quote=True),
        _bash_array("PLATFORMS", [t.platform for t in tasks]),
        _bash_array("OUTS", [t.output for t in tasks]),
        f'RESULTS_ROOT="{results_root}"',
    ]
    if kind == "a100_job":
        run = [
            "nvidia-smi || true",
            'for I in "${!CELLS[@]}"; do',
            "    run_leg $I",
            "done",
        ]
    else:
        run = [
            'I=${SLURM_ARRAY_TASK_ID:?"submit as an array (sbatch from hpc/<batch>/)"}',
            *(["nvidia-smi || true"] if kind == "a100_array" else []),
            "run_leg $I",
        ]
    body = (
        _PREAMBLE.replace("@HPC_BASE@", HPC_BASE)
        .replace("@PRIVATE_BASE@", PRIVATE_BASE)
        .replace("@PIN@", PIN)
        .replace("@NAME@", name)
    )
    tail = [
        'echo "=========================================="',
        'echo "exit status: $STATUS (0 ok, 4 a leg failed or its revisions/row check failed, 5 dataset)"',
        'echo "loadavg: $(cat /proc/loadavg)"',
        "date",
        "exit $STATUS",
    ]
    parts = [
        "\n".join(header),
        "",
        "\n".join(sbatch),
        "",
        "# Submit from hpc/batch_cpu/ or hpc/batch_gpu/ (the -o/-e paths are relative).",
        *([_PROVISIONAL_GUARD.strip("\n")] if batch == 1 else []),
        body.rstrip("\n"),
        "",
        "\n".join(arrays),
        "",
        "\n".join(run),
        "",
        "\n".join(tail),
        "",
    ]
    return "\n".join(parts)


def submits() -> dict[str, str]:
    """{repo-relative submit path: rendered text} for batch 0 and batch 1."""
    out = {}
    for instrument in INSTRUMENTS:  # one array per instrument: --mem follows the dataset size
        name = f"submit_campaign_p2a_imaging_ral_cpu_{instrument}_fp64"
        out[f"hpc/batch_cpu/{name}"] = render(
            name,
            batch1_ral_cpu(instrument),
            kind="ral_cpu",
            batch=1,
            title=f"RAL CPU array: the hpc_ral_cpu imaging cells of {instrument} at the standard size.",
        )
    for instrument in INSTRUMENTS:
        name = f"submit_campaign_p2a_imaging_a100_{instrument}_fp64"
        out[f"hpc/batch_gpu/{name}"] = render(
            name,
            batch1_a100(instrument),
            kind="a100_job",
            batch=1,
            title=f"A100 job: the hpc_a100 imaging cells of {instrument} at the standard size.",
        )
    name = "submit_campaign_p2a_imaging_batch0_ral_cpu_fp64"
    out[f"hpc/batch_cpu/{name}"] = render(
        name,
        batch0_ral_cpu(),
        kind="ral_cpu",
        batch=0,
        title="Batch-0 timing arm, RAL CPU: the bounding hpc_ral_cpu cells.",
    )
    name = "submit_campaign_p2a_imaging_batch0_a100_fp64"
    out[f"hpc/batch_gpu/{name}"] = render(
        name,
        batch0_a100(),
        kind="a100_array",
        batch=0,
        title="Batch-0 timing arm, A100: the bounding hpc_a100 cells.",
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="(re)render the submits")
    mode.add_argument("--check", action="store_true", help="exit 1 if a submit is stale")
    mode.add_argument("--list", action="store_true", help="print the manifest")
    args = parser.parse_args(argv)
    if args.list:
        for task in planned_tasks():
            print(f"{task.cell:<72} {task.script}.py {task.flags}")
        return 0
    stale = []
    for rel, text in submits().items():
        path = ROOT / rel
        if args.write:
            path.write_text(text)
            path.chmod(0o755)
            print(f"wrote {rel}")
        elif not path.is_file() or path.read_text() != text:
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel} (run scripts/misc/campaigns/imaging_p2a.py --write)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
