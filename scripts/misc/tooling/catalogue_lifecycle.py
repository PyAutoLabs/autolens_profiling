"""The ``profiling-summary@3`` lifecycle rules for the mirrored results tree (stdlib only).

The day-zero move (autolens_profiling#442) placed every canonical result at
``results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]/
<measurement>.json`` with ``history/``, ``scans/`` and ``experiments/`` beside it. This
module reads that grammar natively and owns the phase-0 labelling rules; it reads no legacy
path and no migration map. See PyAutoPulse REFERENCE.md "``profiling-summary`` v3" and the
lifecycle report §3.6 (day zero); classes come from the PyAutoPulse designation records.

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
- **Size variants** (``catalogue/size_variants.json``, autolens_profiling#468): a declared
  count measured beside the standard one (rectangular 784 = 28x28, the SLaM production mesh).
  Its rows live in the cell's ``source_pixels_<n>/`` folder, their key adds ``size_variant``
  so they are current beside the standard row, without a ``size_fallback`` note, and they
  carry a ``size_variant`` limitation. Coverage and the trend dashboard stay standard-only.
- **Superseded**: every other record on a key with a current record; a ``local_*`` row is
  superseded by the current record of its key on the reference host for its device. Records
  whose key has no current record cannot honestly be superseded and stay out of
  ``records[]``; ``coverage.excluded`` counts them per key.
- **Class** (PyAutoPulse ``kind: designation`` records, mirrored in
  ``catalogue/pulse_designations.json``; autolens_profiling#497): the designated optimal setup
  on its key (likelihood, standard regime, hardware class) is ``class: optimal`` with the
  designation's ``decision_id``; a named secondary is ``secondary`` with it; any other setup on
  a designated key is ``experimental``. ``local_*`` hosts and non-standard regimes inherit the
  role as ``secondary`` without a decision_id. A held group (``catalogue/designation_holds.json``)
  keeps the provisional label (``secondary`` + ``provisional_optimal: true``). Mixed precision is
  ``experimental``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

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


#: Released tags and their library SHAs: {tag: {repo: sha}}. A library source checkout reports
#: the last *built* ``__version__`` (2026.8.17.1 at every later tag), so a row run from a tag
#: checkout stamps a stale ``autolens_version``; its ``library_revisions`` name the release.
RELEASE_REVISIONS = "catalogue/release_revisions.json"


#: Supported ``release_revisions.json`` versions (2 adds the declared ``carry_forward`` list).
RELEASE_REVISIONS_VERSIONS = (1, 2)


def _release_doc(root) -> dict:
    path = Path(root) / RELEASE_REVISIONS
    if not path.is_file():
        return {"releases": {}, "carry_forward": []}
    doc = json.loads(path.read_text())
    if (
        doc.get("schema") != "release-revisions"
        or doc.get("version") not in RELEASE_REVISIONS_VERSIONS
        or not isinstance(doc.get("releases"), dict)
    ):
        raise ValueError(f"Unsupported {RELEASE_REVISIONS}")
    return doc


def release_revisions(root) -> dict:
    """``{tag: {repo: sha}}`` from ``catalogue/release_revisions.json`` ({} when absent)."""
    return _release_doc(root)["releases"]


def carry_forwards(root) -> list[dict]:
    """The declared pin carry-forwards of ``catalogue/release_revisions.json`` ([] when none).

    Each entry ``{from, to, domains, basis, ruling}`` says a record measured on tag ``from``
    whose likelihood dataset_type is in ``domains`` publishes ``to`` as its release_pin without
    a rerun: the release diff ``from..to`` (``basis``, per library) touches no per-call
    likelihood in those domains. The measured pin and SHAs stay on the record as provenance.
    """
    doc = _release_doc(root)
    entries = doc.get("carry_forward", [])
    if not isinstance(entries, list):
        raise ValueError(f"{RELEASE_REVISIONS}: carry_forward must be a list")
    releases = doc["releases"]
    for entry in entries:
        ok = (
            isinstance(entry, dict)
            and entry.get("from") in releases
            and entry.get("to") in releases
            and entry.get("from") != entry.get("to")
            and isinstance(entry.get("domains"), list)
            and bool(entry["domains"])
            and set(entry["domains"]) <= set(DATASET_TYPES)
            and isinstance(entry.get("basis"), dict)
            and bool(entry["basis"])
        )
        if not ok:
            raise ValueError(f"{RELEASE_REVISIONS}: invalid carry_forward entry {entry!r}")
    if len({e["from"] for e in entries}) != len(entries):
        raise ValueError(f"{RELEASE_REVISIONS}: one carry_forward per measured tag")
    return entries


def carried_pin(pin, dataset_type, carries) -> dict | None:
    """The carry-forward entry that applies to a record measured on ``pin``, else None."""
    for entry in carries or ():
        if entry["from"] == pin and dataset_type in entry["domains"]:
            return entry
    return None


def release_from_revisions(payload, releases) -> str | None:
    """The tag whose every library SHA the row's ``library_revisions`` records, else None."""
    device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
    provenance = device.get("provenance") if isinstance(device.get("provenance"), dict) else {}
    revisions = provenance.get("library_revisions")
    if not isinstance(revisions, dict):
        return None
    for tag, shas in sorted((releases or {}).items()):
        if shas and all(revisions.get(repo) == sha for repo, sha in shas.items()):
            return tag
    return None


#: Declared size variants measured beside the standard size: ``{schema, version, variants}``
#: where each variant is ``{dataset_type, variant, source_pixels, reason, ruling}``.
SIZE_VARIANTS = "catalogue/size_variants.json"
SIZE_VARIANT_PREFIX = "source_pixels_"


def _size_variants() -> dict[str, dict]:
    """``{folder: entry}`` of the declared size variants (``source_pixels_<n>``)."""
    path = Path(__file__).resolve().parents[3] / SIZE_VARIANTS
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema") != "size-variants" or doc.get("version") != 1:
        raise ValueError(f"Unsupported {SIZE_VARIANTS}")
    out = {}
    for entry in doc["variants"]:
        ok = (
            entry.get("dataset_type") in DATASET_TYPES
            and entry.get("variant") in STANDARD_SOURCE_PIXELS
            and type(entry.get("source_pixels")) is int
            and entry["source_pixels"] > 0
            and entry["source_pixels"] != STANDARD_SOURCE_PIXELS[entry["variant"]]
            and isinstance(entry.get("reason"), str)
            and isinstance(entry.get("ruling"), str)
        )
        if not ok:
            raise ValueError(f"{SIZE_VARIANTS}: invalid size variant {entry!r}")
        out[f"{SIZE_VARIANT_PREFIX}{entry['source_pixels']}"] = entry
    return out


SIZE_VARIANT_FOLDERS = _size_variants()


def size_variant_of(dataset_type: str, variant: str, folder: str) -> dict | None:
    """The declared size variant a cell sub-folder names for this likelihood, else None."""
    entry = SIZE_VARIANT_FOLDERS.get(folder)
    if entry and (entry["dataset_type"], entry["variant"]) == (dataset_type, variant):
        return entry
    return None


def size_variant_note(entry: dict) -> str:
    return (
        f"size_variant: a declared size measured beside the standard "
        f"{STANDARD_SOURCE_PIXELS[entry['variant']]} source pixels ({SIZE_VARIANTS}, "
        f"{entry['ruling']}): {entry['source_pixels']} source pixels, {entry['reason']}"
    )


def mirrored(path: str) -> dict | None:
    """The cell facts of a repo-relative path in the mirrored tree, else None.

    ``size_variant`` is the declared size folder (``source_pixels_<n>``, directly under the host
    class or the model variant) a cell file sits in, else None.
    """
    parts = PurePosixPath(path).parts
    if len(parts) < 7 or parts[0] != "results" or parts[1] not in DATASET_TYPES:
        return None
    dataset_type, variant, instrument, setup_id, host_class = parts[1:6]
    match = SETUP_RE.match(setup_id)
    if match is None or host_class not in HOST_CLASSES:
        return None
    rest = list(parts[6:])
    model_variant = "default"
    if (
        len(rest) > 1
        and rest[0] not in LOCATIONS
        and not size_variant_of(dataset_type, variant, rest[0])
    ):
        model_variant, rest = rest[0], rest[1:]
    size_variant = None
    if len(rest) > 1 and size_variant_of(dataset_type, variant, rest[0]):
        size_variant, rest = rest[0], rest[1:]
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
        "size_variant": size_variant,
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


def requested_source_pixels(payload: dict) -> int | None:
    """The ``source_pixels_requested`` of a row (its own keys, then its configuration)."""
    for block in (payload, payload.get("configuration")):
        if isinstance(block, dict) and _int(block.get("source_pixels_requested")):
            return block["source_pixels_requested"]
    return None


def size_pixels(variant: str, payload: dict) -> int | None:
    """The count the standard-size rule judges a row by.

    A row that asked for its variant's standard count (``source_pixels_requested`` equal to
    :data:`STANDARD_SOURCE_PIXELS`) is at the standard size whatever its construction added:
    a standard-size Delaunay cell builds 1250 Hilbert vertices plus 30 zeroed edge points and
    records 1280 (human ruling 2026-10-24). Every other row is judged
    by its recorded count (:func:`source_pixels`), so the rows recorded before the ruling keep
    their regime: a scan that asked for 1250 rectangular pixels built 35 x 35 = 1225 and is
    standard by its count, one that asked for 1500 built 1521 and is ``source_pixels_1521``.
    """
    requested = requested_source_pixels(payload)
    if requested is not None and requested == STANDARD_SOURCE_PIXELS.get(variant):
        return requested
    return source_pixels(payload)


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


# ---- designations: the Pulse records (catalogue/pulse_designations.json) and the holds ----

#: The machine mirror of PyAutoPulse ``decisions.yaml`` ``kind: designation`` rows, written by
#: ``build_catalogue.py --sync-designations <PyAutoPulse>`` and checked against Pulse by
#: ``build_catalogue.py --check --validate-with <PyAutoPulse>``. It is the source of
#: ``class: optimal`` / ``secondary`` + ``decision_id`` (autolens_profiling#497); nothing here
#: derives a designation from timings.
DESIGNATIONS_PATH = Path(__file__).resolve().parents[3] / "catalogue" / "pulse_designations.json"
#: Groups the human held provisional (no designation) and the planning metadata kept from the
#: retired ``catalogue/provisional_designations.json``.
HOLDS_PATH = Path(__file__).resolve().parents[3] / "catalogue" / "designation_holds.json"
#: Host class -> the designation hardware class it reads. A ``local_*`` host has no designation
#: of its own: it inherits its reference host's role without the decision_id (human ruling
#: 2026-10-31, autolens_profiling#497).
HARDWARE_CLASSES = {
    "hpc_ral_cpu": "cpu_many_core",
    "hpc_a100": "gpu_a100",
    "local_laptop": "cpu_many_core",
    "local_gpu": "gpu_a100",
}
#: The regime a day-zero designation covers; a record in another regime inherits its role.
DESIGNATED_REGIME = "standard"
#: Limitation on a record that inherits a designation's role without its decision_id.
UNDESIGNATED = "undesignated_regime_or_host"
DESIGNATION_FIELDS = (
    "id",
    "likelihood",
    "regime",
    "hardware_class",
    "optimal",
    "secondary",
    "evidence",
)


def load_designations(doc: dict) -> dict:
    """{(likelihood, hardware_class): designation} from a ``pulse-designations`` mirror."""
    if doc.get("schema") != "pulse-designations" or doc.get("version") != 1:
        raise ValueError(f"{DESIGNATIONS_PATH.name}: unsupported schema/version")
    out = {}
    for row in doc["designations"]:
        if row["regime"] != DESIGNATED_REGIME:
            raise ValueError(f"{row['id']}: only {DESIGNATED_REGIME!r} designations are read")
        key = (row["likelihood"], row["hardware_class"])
        if key in out:
            raise ValueError(f"Two designations on {key}: {out[key]['id']}, {row['id']}")
        out[key] = row
    return out


def load_holds(doc: dict) -> dict:
    """{(likelihood, host_class): hold} from ``designation_holds.json``."""
    if doc.get("schema") != "designation-holds" or doc.get("version") != 1:
        raise ValueError(f"{HOLDS_PATH.name}: unsupported schema/version")
    return {(h["likelihood"], h["host_class"]): h for h in doc["holds"]}


_DESIGNATIONS = load_designations(json.loads(DESIGNATIONS_PATH.read_text()))
_HOLDS_DOC = json.loads(HOLDS_PATH.read_text())
_HOLDS = load_holds(_HOLDS_DOC)
for _key in _HOLDS:
    if (_key[0], HARDWARE_CLASSES[_key[1]]) in _DESIGNATIONS:
        raise ValueError(f"{_key} is both designated and held")
#: Regime-specific routes are never planned in the ``standard`` regime: DFT runs only below
#: 10,000 visibilities and streaming only at ``n_vis >= 1e7``.
_REGIME_ONLY = tuple(_HOLDS_DOC["regime_only_suffixes"])
#: Setups planned only on some instruments (ruling R1(a) on #469: the dense pixelized
#: interferometer NUFFT route at sma only): ``[(dataset_types, setups, planned_instruments)]``.
_INSTRUMENT_SCOPED = tuple(
    (
        frozenset(rule["dataset_types"]),
        frozenset(rule["setups"]),
        frozenset(rule["planned_instruments"]),
    )
    for rule in _HOLDS_DOC.get("instrument_scoped_setups", [])
)
#: Hosts with a GPU: a GPU setup is planned only on these.
_GPU_HOSTS = frozenset(_HOLDS_DOC["gpu_hosts"])


def _reference_host(host_class: str) -> str:
    return REFERENCE_HOSTS["gpu" if HARDWARE_CLASSES[host_class] == "gpu_a100" else "cpu"]


def designation_of(dataset_type: str, variant: str, host_class: str) -> dict | None:
    """The Pulse designation a host class reads for a likelihood, else None."""
    if host_class not in HARDWARE_CLASSES:
        return None
    return _DESIGNATIONS.get((f"{dataset_type}/{variant}", HARDWARE_CLASSES[host_class]))


def hold_of(dataset_type: str, variant: str, host_class: str) -> dict | None:
    """The provisional hold a host class reads for a likelihood, else None."""
    if host_class not in HARDWARE_CLASSES:
        return None
    return _HOLDS.get((f"{dataset_type}/{variant}", _reference_host(host_class)))


def _role(entry: dict, setup_id: str) -> str | None:
    optimal = entry["optimal"]
    if setup_id in (optimal if isinstance(optimal, list) else [optimal]):
        return "optimal"
    return "secondary" if setup_id in entry["secondary"] else None


def designation(dataset_type: str, variant: str, host_class: str, setup_id: str) -> str | None:
    """``optimal`` / ``secondary``: the role a designation (or a hold) names, else None."""
    if setup_id.split("_")[3] == "mixed":
        return None
    entry = designation_of(dataset_type, variant, host_class) or hold_of(
        dataset_type, variant, host_class
    )
    return None if entry is None else _role(entry, setup_id)


def record_class(
    dataset_type, variant, host_class, setup_id, regime=DESIGNATED_REGIME
) -> tuple[str, str | None, list[str]]:
    """(v3 class, decision_id, limitations) for a measured record.

    - A setup a Pulse designation names, on its reference host in the standard regime: its role
      (``optimal`` / ``secondary``) with the designation's ``decision_id``.
    - The same setup on a ``local_*`` host or in another regime (``source_pixels_<n>``):
      ``secondary`` with an ``undesignated_regime_or_host`` limitation naming the role it
      inherits, and no ``decision_id`` (optimal needs a decision; ruling 2026-10-31).
    - A setup a designation does not name on its key: ``experimental``.
    - A held group (``designation_holds.json``): the provisional label, ``secondary`` with
      ``provisional_optimal`` / ``provisional_secondary``, no ``decision_id``.
    - Mixed precision and anything undesignated: ``experimental``.
    """
    if setup_id.split("_")[3] == "mixed":
        return "experimental", None, []
    entry = designation_of(dataset_type, variant, host_class)
    if entry is not None:
        role = _role(entry, setup_id)
        if role is None:
            return "experimental", None, []
        where = []
        if host_class in LOCAL_HOSTS:
            where.append(f"host {host_class}")
        if regime != DESIGNATED_REGIME:
            where.append(f"regime {regime}")
        if not where:
            return role, entry["id"], []
        return (
            "secondary",
            None,
            [
                f"{UNDESIGNATED}: {' and '.join(where)} inherits the {role} role of designation "
                f"{entry['id']} ({DESIGNATED_REGIME} regime, {entry['hardware_class']}) without "
                "its decision_id"
            ],
        )
    hold = hold_of(dataset_type, variant, host_class)
    role = None if hold is None else _role(hold, setup_id)
    if role is not None:
        return "secondary", None, [f"provisional_{role}: true"]
    return "experimental", None, []


def _instrument_scoped_out(dataset_type: str, setup_id: str, instrument: str | None) -> bool:
    """Whether an ``instrument_scoped_setups`` rule leaves ``setup_id`` unplanned on ``instrument``."""
    if instrument is None:
        return False
    return any(
        dataset_type in types and setup_id in setups and instrument not in instruments
        for types, setups, instruments in _INSTRUMENT_SCOPED
    )


def planned_setups(
    dataset_type: str, variant: str, host_class: str, instrument: str | None = None
) -> dict[str, str]:
    """{setup_id: class} planned on a reference host: the designation's optimal and secondary
    setups (or a held group's provisional ones, which carry the provisional class).

    Regime-only setups (``_dft``, ``_streaming``) are never planned; an instrument-scoped setup
    (``instrument_scoped_setups``, e.g. the dense pixelized interferometer NUFFT route, sma only)
    is planned only on its instruments. ``instrument`` None skips the instrument scoping. A GPU
    setup is planned only on a ``gpu_hosts`` host.
    """
    entry = designation_of(dataset_type, variant, host_class) or hold_of(
        dataset_type, variant, host_class
    )
    if entry is None:
        return {}
    optimal = entry["optimal"]
    plan = {sid: "secondary" for sid in entry["secondary"]}
    plan.update({sid: "optimal" for sid in (optimal if isinstance(optimal, list) else [optimal])})
    return {
        sid: plan[sid]
        for sid in sorted(plan)
        if not sid.endswith(_REGIME_ONLY)
        and not _instrument_scoped_out(dataset_type, sid, instrument)
        and (host_class in _GPU_HOSTS or sid.split("_")[1] != "gpu")
    }


# ---- likelihood-lifecycle phase 2x: resources per evaluation and projected node throughput ----

#: Declared reference hosts and the declared parallel efficiency (sinfo 2026-10-25).
NODE_HOSTS = "catalogue/reference_hosts.json"
#: ``resources`` for rows measured before the writers recorded it (2a; sacct MaxRSS).
PEAK_RSS_BACKFILL = "catalogue/peak_rss_backfill.json"
#: Where a runtime record's per-call time comes from: numba's warm iid median, JAX's vmap
#: per-call (the batched cost a sampler pays per point).
PER_CALL_SOURCES = {"numba": ("warm_iid_median_s",), "jax": ("vmap", "per_call")}
RESOURCE_KEYS = (
    "cpus_per_eval",
    "cpus_per_eval_basis",
    "devices_per_eval",
    "peak_rss_gb",
    "peak_rss_basis",
    "peak_rss_reason",
    "peak_rss_note",
)


def _sidecar(root, rel: str, schema: str, key: str) -> dict:
    path = Path(root) / rel
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema") != schema or not isinstance(doc.get(key), dict):
        raise ValueError(f"Unsupported {rel}")
    return doc


def node_hosts(root) -> dict:
    """``catalogue/reference_hosts.json`` ({} when absent)."""
    return _sidecar(root, NODE_HOSTS, "reference-hosts", "hosts")


def peak_rss_backfill(root) -> dict:
    """``{row path: entry}`` from ``catalogue/peak_rss_backfill.json`` ({} when absent)."""
    return _sidecar(root, PEAK_RSS_BACKFILL, "peak-rss-backfill", "rows").get("rows", {})


def _positive(value):
    return (
        value
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0
        else None
    )


def row_resources(payload: dict, path: str, base: str, backfill: dict) -> dict:
    """The record's ``resources``: the row's own block, else its backfill entry, else unknown."""
    own = payload.get("resources")
    if isinstance(own, dict):
        block = {k: own.get(k) for k in RESOURCE_KEYS if k in own}
        block["source"] = "row"
        return block
    entry = backfill.get(path) if not base else None
    if isinstance(entry, dict):
        block = {k: entry.get(k) for k in RESOURCE_KEYS if k in entry}
        block["source"] = PEAK_RSS_BACKFILL
        block["sacct_job"] = entry.get("sacct_job")
        return block
    return {
        "cpus_per_eval": None,
        "peak_rss_gb": None,
        "source": None,
        "reason": "The row predates the resources block (#462) and has no backfill entry.",
    }


def per_call(payload: dict, backend: str) -> tuple[float | None, str]:
    """(seconds, source) of one likelihood call as a sampler pays it, else (None, reason)."""
    keys = PER_CALL_SOURCES[backend]
    value = payload
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    source = ".".join(keys)
    if _positive(value) is None:
        return None, f"no finite positive {source} in the row"
    return value, source


#: The 12-core estimate on numba CPU runtime records (human ruling 2026-10-26).
TWELVE_CORE_DIVISOR = 6
TWELVE_CORE_SETUP_PREFIX = "numba_cpu"


def estimate_12_core(setup_id: str, seconds, source: str) -> dict | None:
    """``estimated_12_core`` for a numba CPU runtime record, else None.

    The per-call time (``per_call``: numba warm iid median) divided by 6: the human's estimate
    of one call's time at 12 cores. An estimate, never a measurement; nothing ran at 12 cores.
    """
    if not setup_id.startswith(TWELVE_CORE_SETUP_PREFIX) or _positive(seconds) is None:
        return None
    return {
        "value_s": seconds / TWELVE_CORE_DIVISOR,
        "per_call_s": seconds,
        "per_call_source": source,
        "divisor": TWELVE_CORE_DIVISOR,
        "cores": 12,
        "basis": "estimate",
        "ruling": "human 2026-10-26: per-call / 6 at 12 cores",
    }


def _sig(value: float) -> float:
    return float(f"{value:.4g}")


def project(resources: dict, seconds, source: str, device: str, host_class: str, hosts: dict):
    """``projected_node_throughput`` for one runtime record on the host it was measured on.

    The record is projected only onto the declared reference host whose ``host_class`` is the
    record's measured host class (human ruling 2026-10-25: a per-call time is never carried to
    another machine). A record measured on a class with no declared reference host, or a GPU
    record on a host with no GPUs, is ``not_projected`` with a reason. A CPU record measured on a
    GPU host (the ``jax_cpu_sparse@hpc_a100`` legs) is projected as CPU workers on that node's
    cores and memory.

    CPU: workers = min(floor(cores / cpus_per_eval), floor(mem_gb / peak_rss_gb)). GPU: one
    process per device, also capped by the host's cores and memory. Throughput = workers x the
    declared parallel efficiency / per-call seconds. A projection, never a measurement.
    """
    eta = hosts.get("parallel_efficiency", {})
    block = {
        "basis": "projected",
        "per_call_s": seconds,
        "per_call_source": source,
        "cpus_per_eval": resources.get("cpus_per_eval"),
        "devices_per_eval": resources.get("devices_per_eval"),
        "peak_rss_gb": resources.get("peak_rss_gb"),
        "peak_rss_basis": resources.get("peak_rss_basis"),
        "parallel_efficiency": eta,
        "not_modelled": hosts.get("not_modelled"),
        "hosts": [],
    }
    cpus, rss = _positive(resources.get("cpus_per_eval")), _positive(resources.get("peak_rss_gb"))
    matches = sorted(
        name
        for name, host in hosts.get("hosts", {}).items()
        if host.get("host_class") == host_class
    )
    if len(matches) > 1:
        raise ValueError(f"{NODE_HOSTS}: more than one reference host for {host_class}: {matches}")
    name = matches[0] if matches else None
    host = hosts["hosts"][name] if name else {}
    missing = [
        why
        for why, ok in (
            (f"per-call time: {source}", seconds is not None),
            ("cpus_per_eval not recorded", cpus is not None),
            (
                "peak_rss_gb not recorded"
                + (
                    f" ({resources['peak_rss_reason']})" if resources.get("peak_rss_reason") else ""
                ),
                rss is not None,
            ),
            (f"no declared reference host has host_class {host_class}", name is not None),
            (
                f"a gpu record and reference host {name} declares no GPUs",
                device != "gpu" or name is None or bool(host.get("gpus")),
            ),
        )
        if not ok
    ]
    if missing:
        block.update(status="not_projected", reason="; ".join(missing))
        return block
    block["status"] = "projected"
    limits = {}
    if device == "gpu":
        limits["gpus"] = host["gpus"]
    limits["cores"] = int(host["cores"] // cpus)
    limits["memory"] = int(host["mem_gb"] // rss)
    bound_by = min(limits, key=lambda k: limits[k])
    workers = limits[bound_by]
    block["hosts"].append(
        {
            "host": name,
            "host_class": host_class,
            "workers": workers,
            "bound_by": bound_by,
            "limits": limits,
            "calls_per_s": _sig(workers * eta.get("value", 1.0) / seconds),
        }
    )
    return block
