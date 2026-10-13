# Raw HST/MGE CPU pilot

`pilot.py` measures the fixed-parameter `imaging-mge-runtime-v1` likelihood. It
performs no inference, sampling, optimization or posterior estimation. Campaign
intent belongs to [Pulse's pilot contract](https://github.com/PyAutoLabs/PyAutoPulse/blob/71c93fc285a081ffb1f0f506694c043a15ad4e22/tasks/hst_mge_cpu_pilot.md).

This is an opt-in diagnostic producer, not a baseline collector. Every record and
receipt has `accepted: false` and explicit scientific qualification blockers.
The baseline manifest, reader, global compile routes and historical pins remain
unchanged. The historical flat-vector posterior is an incompatible objective.

## Plan, then execute only when authorized

From the project root, select an existing HST FITS directory and one CPU in the
current process's allowed affinity. This command hashes inputs and prints a plan;
it imports no scientific library, creates no files and launches no workers:

```bash
python scripts/misc/jax_compile/pilot.py \
  --dataset-path dataset/imaging/hst \
  --output-dir output/hst_mge_pilot_v1/run-001 --cpu 0
```

Replace CPU 0 with the actual permitted CPU; it is not a portable allocation.
Only append `--run` after authorizing execution. Existing `data.fits`,
`noise_map.fits` and `psf.fits` are required; the driver never simulates data or
submits to SLURM. On RAL, obtain the approved one-CPU allocation on `ral` first;
never use a login node or a GPU partition for this production CPU pilot.
Scheduler memory requests and the actual host allocation are external. The
runner does not claim to enforce the proposed 16 GiB scheduler memory request.

The existing builder supplies 0.05 arcsec pixels, a 3.5 arcsec mask, radial LP
oversampling [4,2,2] at [0.3,0.6], two 20-Gaussian light components, fixed mass and
shear, and prior-median parameters. It supports only CPU/fp64/jit here. Input
shapes and actual bytes are recorded, never assumed from the instrument label.
Inspect and agree any observational dataset/mask before collection.

## Process order and resource boundaries

For each of ten repetitions: cold process, separate seed process, then warm
process. Each cold process owns a fresh empty cache; each seed/warm pair owns a
separate shared cache. A final fresh runtime process performs three untimed
warmups and ten individual timed calls. `--repetitions` accepts 1..10 and
`--timeout` accepts at most 1,800 seconds per process, including imports/setup.
The runner is sequential, with no retries: 31 processes at defaults, at most
15.5 worker wall-hours / allocated CPU-hours. The separate correctness process
reserved by Pulse is not implemented here, leaving its 0.5-hour allowance unused.
This is a conservative upper bound, not an estimated cost or authorization.

Every child has a fresh interpreter, one-CPU affinity, explicit CPU JAX backend,
x64, one numerical thread and `--xla_cpu_multi_thread_eigen=false`. These settings
are scoped to children; inherited XLA flags are replaced and the effective flags
are recorded. Processes with unexpected backend, affinity or floating dtype fail.
Caches live inside the run output tree (on RAL's project filesystem, not /tmp or
home). Timeout and keyboard/SIGTERM interruption kill the worker process group
and reap the child. SIGKILL/power loss cannot guarantee final receipt creation;
the already written plan, requests and logs remain recoverable.

No output is overwritten. The output directory must be new and inside the pilot
namespace, with no symlink traversal. A failed worker, incomplete cache witness or
mismatched context stops the sequence, preserving attempted records and logs.
It does not become a pass, a retry request or a CPU exclusion verdict.

## Timing and cache evidence

Setup (including imports and provenance capture), trace **plus lowering**, AOT
compile and first synchronized execution have separate raw intervals. This does
not claim to split tracing from lowering. The runtime process reports ten
unrounded seconds samples after warmup, synchronizing outputs before and after
each timed call. Its ten samples describe within-process variability only.
Parent-observed worker wall time includes process start and output serialization;
it is not equivalent to a likelihood call. Seed costs are retained separately.

During the target executable's compile interval, a temporary logging handler
captures JAX compiler diagnostics. A hit must identify the exact lowered module
and cache key. Warm evidence requires a matching seed miss with the same key,
module and actual workload/software/input context. Helper hits, duplicated or
unknown messages, file counts and short timings cannot establish a hit. Changed
JAX diagnostic formats fail closed as unknown; the raw text remains available.
The diagnostic handler is part of the measurement method and its overhead is
not subtracted. Cold/seed rows also require a demonstrated target miss.

A summary groups matching context and phase only, retaining raw samples and
reporting median, min/max and inclusive IQR (null with a single sample). There
are no speedup ratios, statistical significance tests or drift verdicts.

## Artifacts and remaining qualification gaps

Each new run directory contains `plan.json`, per-worker requests/logs/observations
and statuses, isolated caches, and `receipt.json`. The receipt binds observation
bytes by SHA256 and lists unstarted workers and invalid outcomes. Raw observation
schema is `profiling-pilot-observation` version 1; it is deliberately different
from `baseline-observation` and cannot fill baseline slots by renaming files.

Provenance includes actual input hashes/model/parameter leaves, source hashes,
project configuration file hashes, package inventory, Git identities and dirty
flags, CPU/host/OS/RAM, affinity, effective environment and devices. A package
inventory is not a reproducible environment lock. Project config hashes and model
serialization do not independently audit every effective library solver default.
Local absolute paths are diagnostic metadata; do not publish raw files without
reviewing their path content.

Recorded load averages include campaign work and other host activity. They are
**not** the background-load metric required for reference qualification. OS peak
RSS is a diagnostic for the worker process lifetime, sampled before JSON output;
it is not an accepted baseline memory axis. A finite likelihood output is only
an execution check. There is no independent likelihood/reconstruction correctness
witness, accepted tolerance or frozen production specification in this producer.
These missing facts remain explicit blockers even when all workers complete.

Before production collection, resolve the exact hardware/stack/environment lock,
dataset/full configuration, independent witness and tolerances, load methodology,
collection window and resource approval in Pulse/project records. This diagnostic
runner neither freezes those decisions nor grants scientific acceptance.
