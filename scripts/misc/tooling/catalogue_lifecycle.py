"""The ``profiling-summary@3`` lifecycle rules for the mirrored results tree (stdlib only).

The day-zero move (autolens_profiling#442) placed every canonical result at
``results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]/
<measurement>.json`` with ``history/``, ``scans/`` and ``experiments/`` beside it. This
module reads that grammar natively and owns the phase-0 labelling rules; it reads no legacy
path and no migration map. See PyAutoPulse REFERENCE.md "``profiling-summary`` v3" and the
lifecycle report §3.3 (provisional designations) and §3.6 (day zero).

The rules, in one place:

- **Key**: (likelihood, instrument, setup_id, host_class, model_variant, measurement).
- **Current** (§3.6 at phase 0): a canonical cell file (not ``history/``, ``scans/`` or
  ``experiments/``) on a reference host class (``hpc_ral_cpu``, ``hpc_a100``) whose
  likelihood × instrument is one of the registry's planned baseline cells, or a declared
  reference candidate. R7: ``local_*`` rows are never current.
- **Size rule** (human, 2026-10-23): the standard source-pixel count is Delaunay 1250 and
  rectangular 35² = 1225. A standard-size candidate wins; otherwise the candidate nearest the
  standard (then one whose size is not recorded) is current with a ``size_fallback``
  limitation, and coverage later reports the standard size missing.
- **Superseded**: every other record on a key with a current record; a ``local_*`` row is
  superseded by the current record of its key on the reference host for its device. Records
  whose key has no current record cannot honestly be superseded and stay out of
  ``records[]``; ``coverage.excluded`` counts them per key.
- **Class** (§3.3 provisional table): provisional Optimal rows are ``class: secondary`` with
  ``provisional_optimal: true`` in ``limitations`` (R3: ``optimal`` needs a Pulse designation
  ``decision_id``; none exists yet); mixed precision is ``experimental``; anything not in the
  table is ``experimental``. Planned cells carry the provisional class itself.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

DATASET_TYPES = (
    "imaging",
    "interferometer",
    "datacube",
    "point_source_image",
    "point_source_source",
    "multi_galaxy",
    "group",
    "cluster",
    "multi_dataset",
    "weak",
)
HOST_CLASSES = ("hpc_ral_cpu", "hpc_a100", "local_laptop", "local_gpu")
REFERENCE_HOSTS = {"cpu": "hpc_ral_cpu", "gpu": "hpc_a100"}
LOCAL_HOSTS = ("local_laptop", "local_gpu")
#: Sub-folders of a cell that are never current.
LOCATIONS = ("history", "scans", "experiments")
#: Measurement file stems that name a lifecycle measurement.
FILE_MEASUREMENTS = ("runtime", "breakdown", "gradient", "parity")
SETUP_RE = re.compile(
    r"^(?P<backend>numba|jax)_(?P<device>cpu|gpu)_(?P<operator>dense|sparse|na)_"
    r"(?P<precision>fp64|mixed)(?:_(?P<transformer>dft|nufft|wtilde|streaming))?$"
)
#: v3 keys need an instrument; a registry cell declared instrument-independent is keyed by the
#: instrument folder its scripts write (point sources and clusters: ``simple``; the
#: multi-dataset cell defaults to ``hst``).
DEFAULT_INSTRUMENTS = {
    "point_source_image": "simple",
    "point_source_source": "simple",
    "cluster": "simple",
    "multi_dataset": "hst",
}
#: The human's standard source-pixel counts (2026-10-23), by likelihood variant.
STANDARD_SOURCE_PIXELS = {"delaunay": 1250, "delaunay_nn": 1250, "rectangular": 35 * 35}
#: Headline metric per measurement, in preference order (the adapter's metric names).
HEADLINES = {
    "runtime": (
        "single_call",
        "full_pipeline.median_s",
        "full_pipeline.per_call_s",
        "full_call.median_s",
        "full_call.per_call_s",
        "single_jit_median",
        "single_jit_block",
        "cube_single_jit_median",
        "cube_single_jit",
        "direct_call",
        "vmap.per_call",
    ),
    "compile": (
        "full_pipeline.compile",
        "compile",
        "operator_setup",
        "full_pipeline.first_call",
        "first_call",
    ),
    "memory_host": ("host_peak_rss",),
    "memory_device": ("device_peak_allocated", "device_peak_reserved"),
}
HEADLINES["breakdown"] = (
    "component_total",
    "cube_component_total",
    "total_s",
    "per_call_s",
) + HEADLINES["runtime"]
HEADLINES["gradient"] = HEADLINES["runtime"]
#: For a compile/memory key reached by two files of one cell, the runtime file wins.
FILE_ORDER = {"runtime": 0, "breakdown": 1, "gradient": 2, "parity": 3}


def mirrored(path: str) -> dict | None:
    """The cell facts of a repo-relative path in the mirrored tree, else None."""
    parts = PurePosixPath(path).parts
    if len(parts) < 7 or parts[0] != "results" or parts[1] not in DATASET_TYPES:
        return None
    dataset_type, variant, instrument, setup_id, host_class = parts[1:6]
    match = SETUP_RE.match(setup_id)
    if match is None or host_class not in HOST_CLASSES:
        return None
    rest = list(parts[6:])
    model_variant = "default"
    if len(rest) > 1 and rest[0] not in LOCATIONS:
        model_variant, rest = rest[0], rest[1:]
    location, run = "cell", None
    if len(rest) > 1 and rest[0] in LOCATIONS:
        location, run = rest[0], "/".join(rest[1:-1]) or None
    elif len(rest) > 1:
        return None  # an unexpected nesting is not a cell
    stem = PurePosixPath(rest[-1]).stem
    setup = match.groupdict()
    setup["transformer"] = setup["transformer"] or "na"
    return {
        "dataset_type": dataset_type,
        "variant": variant,
        "instrument": instrument,
        "setup_id": setup_id,
        "setup": setup,
        "host_class": host_class,
        "model_variant": model_variant,
        "location": location,
        "run": run,
        "file_measurement": stem if stem in FILE_MEASUREMENTS else None,
    }


def measurement_of(axis: str, metric: str, file_measurement: str | None) -> str:
    """The v3 measurement a v2 adapter metric belongs to."""
    if axis in ("runtime", "breakdown"):
        return file_measurement or axis
    if axis == "memory":
        return "memory_host" if metric == "host_peak_rss" else "memory_device"
    return axis


def headline(measurement: str, values: dict) -> str:
    for metric in HEADLINES.get(measurement, ()):
        if metric in values:
            return metric
    return sorted(values)[0]


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def source_pixels(payload: dict) -> int | None:
    """The recorded source-pixel count of a row (its own keys, then its configuration)."""
    for block in (payload, payload.get("configuration")):
        if not isinstance(block, dict):
            continue
        for key in ("source_pixels", "delaunay_vertices", "hilbert_pixels"):
            if _int(block.get(key)):
                return block[key]
        shape = block.get("mesh_shape")
        if isinstance(shape, list) and len(shape) == 2 and all(_int(n) for n in shape):
            return shape[0] * shape[1]
    return None


def regime(variant: str, pixels: int | None) -> str:
    standard = STANDARD_SOURCE_PIXELS.get(variant)
    if standard is None or pixels == standard:
        return "standard"
    return f"source_pixels_{pixels}" if pixels else "source_pixels_unrecorded"


def size_rank(variant: str, pixels: int | None) -> tuple[int, int]:
    """0 = standard size (or no size rule), 1 = nearest recorded size, 2 = size unrecorded."""
    standard = STANDARD_SOURCE_PIXELS.get(variant)
    if standard is None or pixels == standard:
        return (0, 0)
    if pixels:
        return (1, abs(pixels - standard))
    return (2, 0)


def size_note(variant: str, pixels: int | None) -> str:
    standard = STANDARD_SOURCE_PIXELS[variant]
    measured = f"{pixels} source pixels" if pixels else "an unrecorded source-pixel count"
    return (
        f"size_fallback: no row at the standard {standard} source pixels exists for this key; "
        f"this row (measured at {measured}) is current until the standard size is measured"
    )


# ---- §3.3 provisional designations: (optimal setup_ids, secondary setup_ids) per host ----

_PARAMETRIC = {
    "hpc_a100": ({"jax_gpu_dense_fp64"}, set()),
    "hpc_ral_cpu": ({"jax_cpu_dense_fp64"}, set()),
    "laptop": ({"jax_cpu_dense_fp64"}, set()),
}
_PIXELIZED = {
    "hpc_a100": ({"jax_gpu_dense_fp64"}, {"jax_gpu_sparse_fp64", "jax_cpu_sparse_fp64"}),
    "hpc_ral_cpu": ({"numba_cpu_sparse_fp64"}, {"jax_cpu_sparse_fp64"}),
    "laptop": ({"jax_cpu_dense_fp64"}, {"jax_cpu_sparse_fp64"}),
}
_INTERFEROMETER_MGE = {
    "hpc_a100": ({"jax_gpu_dense_fp64_dft", "jax_gpu_dense_fp64_nufft"}, set()),
    "hpc_ral_cpu": ({"jax_cpu_dense_fp64_dft", "jax_cpu_dense_fp64_nufft"}, set()),
    "laptop": ({"jax_cpu_dense_fp64_dft", "jax_cpu_dense_fp64_nufft"}, set()),
}
_INTERFEROMETER_PIXELIZED = {
    "hpc_a100": (
        {"jax_gpu_sparse_fp64_wtilde", "jax_gpu_sparse_fp64_streaming"},
        {"jax_gpu_dense_fp64_dft", "jax_gpu_dense_fp64_nufft"},
    ),
    "hpc_ral_cpu": (
        {"numba_cpu_sparse_fp64_wtilde"},
        {"jax_cpu_sparse_fp64_wtilde", "jax_cpu_dense_fp64_dft", "jax_cpu_dense_fp64_nufft"},
    ),
    "laptop": (
        {"jax_cpu_sparse_fp64_wtilde"},
        {"jax_cpu_dense_fp64_dft", "jax_cpu_dense_fp64_nufft"},
    ),
}
# A point-source likelihood has no linear operator: R1 spells its setups ``..._na_...``.
_POINT_IMAGE = {
    "hpc_a100": ({"jax_gpu_na_fp64"}, set()),
    "hpc_ral_cpu": ({"jax_cpu_na_fp64"}, set()),
    "laptop": ({"jax_cpu_na_fp64"}, set()),
}
_POINT_SOURCE = {
    host: ({"jax_cpu_na_fp64"}, {"jax_gpu_na_fp64"})
    for host in ("hpc_a100", "hpc_ral_cpu", "laptop")
}
_WEAK = {
    host: ({"jax_cpu_dense_fp64"}, {"jax_gpu_dense_fp64"})
    for host in ("hpc_a100", "hpc_ral_cpu", "laptop")
}
#: Regime-specific routes are designated but not planned in the ``standard`` regime: DFT runs
#: only below 10,000 visibilities and streaming only at ``n_vis >= 1e7``.
_REGIME_ONLY = ("_dft", "_streaming")
_PIXELIZED_VARIANTS = ("delaunay", "delaunay_nn", "rectangular")


def _table(dataset_type: str, variant: str) -> dict | None:
    if dataset_type in ("imaging", "multi_galaxy", "group", "multi_dataset"):
        return _PIXELIZED if variant in _PIXELIZED_VARIANTS else _PARAMETRIC
    if dataset_type == "interferometer":
        return _INTERFEROMETER_MGE if variant == "mge" else _INTERFEROMETER_PIXELIZED
    if dataset_type == "datacube":
        return _INTERFEROMETER_PIXELIZED
    if dataset_type == "point_source_image" or (
        dataset_type == "cluster" and variant == "image_plane"
    ):
        return _POINT_IMAGE
    if dataset_type in ("point_source_source", "cluster"):
        return _POINT_SOURCE
    if dataset_type == "weak":
        return _WEAK
    return None


def designation(dataset_type: str, variant: str, host_class: str, setup_id: str) -> str | None:
    """``optimal`` / ``secondary`` per the §3.3 provisional table, else None."""
    table = _table(dataset_type, variant)
    if table is None or setup_id.split("_")[3] == "mixed":
        return None
    host = "laptop" if host_class == "local_laptop" else host_class
    if host == "local_gpu":
        host = "hpc_a100"
    optimal, secondary = table[host]
    if setup_id in optimal:
        return "optimal"
    return "secondary" if setup_id in secondary else None


def record_class(dataset_type, variant, host_class, setup_id) -> tuple[str, list[str]]:
    """(v3 class, limitations) for a measured record (R3)."""
    role = designation(dataset_type, variant, host_class, setup_id)
    if role == "optimal":
        return "secondary", ["provisional_optimal: true"]
    if role == "secondary":
        return "secondary", ["provisional_secondary: true"]
    return "experimental", []


def planned_setups(dataset_type: str, variant: str, host_class: str) -> dict[str, str]:
    """{setup_id: class} the phase-0 plan intends to measure on a reference host."""
    table = _table(dataset_type, variant)
    if table is None:
        return {}
    optimal, secondary = table[host_class]
    plan = {sid: "secondary" for sid in secondary}
    plan.update({sid: "optimal" for sid in optimal})
    return {sid: plan[sid] for sid in sorted(plan) if not sid.endswith(_REGIME_ONLY)}
