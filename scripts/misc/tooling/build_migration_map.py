"""Generate the day-zero migration map for ``results/`` (stdlib only).

Run from the repo root::

    python scripts/misc/tooling/build_migration_map.py          # (re)write the map
    python scripts/misc/tooling/build_migration_map.py --check  # exit 1 if the map is stale

The map, ``catalogue/migration_day_zero.json``, names an old -> new path and the
sha256 of the file content for every tracked file under ``results/`` and for every
``wiki/campaigns/*.md`` page. It is the reviewable plan for the move-only PR of the
likelihood-lifecycle epic (phase 0B, autolens_profiling#442); **nothing moves here**.

Target layout (epic report section 3.7)::

    results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/<measurement>.<ext>
        .../history/<release_pin>_<YYYY-MM-DD>/<measurement>.<ext>   # superseded records
    results/<dataset_type>/<variant>/<instrument>/comparison.<ext>    # cross-setup aggregate
    results/lens/<component>/...      results/misc/<group>/...
    wiki/campaigns/<dataset_type>/<variant>/<ledger>.md    wiki/campaigns/{lens,misc}/...
    archive/YYYY/MM/<slug>/...

``setup_id`` is ``<backend>_<device>_<operator>_<precision>[_<transformer>]`` (ruling R1);
the site/tier moves to ``host_class``. Every axis is derived from the path (the legacy
config grammar, ``catalogue/script_routes.json`` legacy stems) or from the result JSON
(``device``, ``instrument``, ``configuration.inversion_path``, ``transformer``,
``use_mixed_precision``). **Nothing is guessed:** a file whose axes cannot be derived, or
whose path and content disagree, is listed under ``unresolved`` with the reason and,
where one exists, a ``proposed`` path for the human to accept or overrule. Two old paths
mapping to one new path is an error and the map is not written.

Kinds: ``mirrored`` (current slot of a mirrored key), ``history`` (an older record of
the same key, label-only cut per ruling R2), ``lens_component``, ``misc``,
``wiki_campaign``, ``archive`` (the listed dead-experiment classes) and ``unchanged``.

Pure stdlib -- no PyAuto* imports. Dates come only from file names and file content (no
git history), so the map is identical on a shallow CI clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

SCHEMA = "autolens-profiling-day-zero-migration"
VERSION = 1
MAP_PATH = "catalogue/migration_day_zero.json"
ROUTES_PATH = "catalogue/script_routes.json"
ISSUE = "PyAutoLabs/autolens_profiling#442"

#: Release-dated cut for the "pre-2026-09 series" dead-experiment class.
PRE_SERIES_CUTOFF = "2026-09-01"

INSTRUMENTS = ("alma_high", "alma", "ao", "euclid", "hst", "jvla", "jwst", "sdp81", "sma", "simple")

#: Legacy config-grammar host/tier token -> (host_class, device). Longest first.
TIERS = {
    "hpc_ral_cpu_epyc7702": ("hpc_ral_cpu", "cpu"),
    "hpc_ral_gpunode_cpu": (None, "cpu"),  # CPU on a GPU node: not in the host_class vocab
    "hpc_ral_a100": ("hpc_a100", "gpu"),
    "hpc_ral_cpu": ("hpc_ral_cpu", "cpu"),
    "hpc_a100": ("hpc_a100", "gpu"),
    "local_rtx2060": ("local_gpu", "gpu"),
    "local_cpu": ("local_laptop", "cpu"),
    "local_gpu": ("local_gpu", "gpu"),
    "laptop_cpu": ("local_laptop", "cpu"),
}
PRECISIONS = {"fp64": "fp64", "mp": "mixed", "fp32": None}  # fp32 is not a setup precision

#: Canonical per-likelihood script measurements (from script_routes) -> backend.
#: Any other route measurement is an experiment script with no section-3.7 slot.
CELL_MEASUREMENTS = {
    "likelihood_runtime": "jax",
    "likelihood_breakdown": "jax",
    "likelihood_runtime_numba": "numba",
    "likelihood_breakdown_numba": "numba",
    "likelihood_runtime_jax": "jax",
    "likelihood_runtime_solved": "jax",
}
POINT_LIKE = ("point_source_image", "point_source_source", "cluster")
INTERFEROMETRIC = ("interferometer", "datacube")
TRANSFORMERS = {"TransformerNUFFT": "nufft", "TransformerDFT": "dft"}

#: results/ top-level folders that become results/misc/<group>/ (group mirrors scripts/misc/).
MISC_GROUPS = {
    "hazards": "hazards",
    "quick_update": "quick_update",
    "logs": "logs",
    "sweep": "likelihood_breakdown",
    "delaunay_nn": "delaunay_nn",
    "nnls_warm_start": "nnls_warm_start",
    "simulators": "simulators",
    "pipeline_resume": "pipeline_resume",
}
#: results/ top-level experiment folders: (dataset_type, variant) when one is known.
EXPERIMENT_TOPS = {
    "parallel_scaling": ("imaging", "rectangular"),
    "streaming_scaling": ("interferometer", "delaunay"),
    "misc": (None, None),
}
FROZEN_PACKS = ("notes/clipper_campaign/", "notes/point_source_cpu_2026_09_17_reported/")
FROZEN_BASELINE = "baselines/PreOptimizationTimes/"
#: Archive months that no file name or content records, with their source.
SLUG_MONTH = {
    "PreOptimizationTimes": ("2026/07", "results/notes/design_lock_in.md title (2026-07-08)"),
}
#: Campaign families whose archived files stay together under one slug.
CAMPAIGN_FAMILIES = ("fixed_light",)

MULTI = "spans several variants"
MULTI_DT = "spans several dataset types"

#: results/notes/*.md ledgers and wiki/campaigns/*.md pages -> wiki/campaigns/<dt>/<variant>.
#: ("misc", None) and ("lens", <component>) follow section 3.7; MULTI* are unresolved.
LEDGER_ROUTES = {
    # results/notes ledgers
    "a100_pixelized_baseline_2026_09": ("imaging", MULTI),
    "certified_solver_phase_c1_lane_rate_2026_09": ("imaging", MULTI),
    "certified_solver_policy_phase_b_2026_09": ("imaging", MULTI),
    "cluster_pointsolver_campaign": ("cluster", "image_plane"),
    "critical_curves_dispatch": ("lens", "critical_curves"),
    "delaunay_adapt_split_regularization": ("imaging", MULTI),
    "delaunay_nn_breakdown": ("imaging", "delaunay_nn"),
    "delaunay_nn_cap_audit": ("imaging", "delaunay_nn"),
    "delaunay_nn_constant_split_assembly": ("imaging", "delaunay_nn"),
    "delaunay_nn_launch_latency": ("imaging", "delaunay_nn"),
    "delaunay_walk_early_exit": ("imaging", "delaunay"),
    "design_lock_in": ("misc", None),
    "fixed_lens_light_hardware_2026_09": ("imaging", MULTI),
    "fixed_lens_light_levers_2026_09": ("imaging", MULTI),
    "fixed_lens_light_library_path_2026_09": ("imaging", MULTI),
    "fixed_lens_light_low_likelihood_draws_2026_09": ("imaging", MULTI),
    "fixed_lens_light_numba_2026_09": ("imaging", MULTI),
    "fixed_lens_light_numba_memo_2026_09": ("imaging", MULTI),
    "fixed_lens_light_numba_memo_policy_2026_09": ("imaging", MULTI),
    "fixed_lens_light_numba_scaling_2026_09": ("imaging", MULTI),
    "fixed_lens_light_s4_2026_09": ("imaging", "delaunay"),
    "fixed_lens_light_source_only_2026_09": ("imaging", MULTI),
    "fixed_lens_light_source_pixel_scaling_2026_09": ("imaging", MULTI),
    "fixed_lens_light_verdict_2026_09": ("imaging", MULTI),
    "hst_gpu_residue_phase1_2026_09": ("imaging", MULTI),
    "hst_gpu_residue_phase2_vmap_2026_09": ("imaging", MULTI),
    "hst_gpu_residue_phase3_psf_2026_09": ("imaging", MULTI),
    "hst_gpu_residue_phase4_logdet_2026_09": ("imaging", MULTI),
    "interferometer_direct_conv_symmetric_2026_10": ("interferometer", MULTI),
    "interferometer_likelihood_decision_matrix_2026_09": ("interferometer", MULTI),
    "interferometer_mesh_a100_breakdown_2026_09": ("interferometer", MULTI),
    "interferometer_mesh_cpu_breakdown_2026_09": ("interferometer", MULTI),
    "interferometer_mge_breakdown_2026_09": ("interferometer", "mge"),
    "interferometer_route_gate_aris_2026_10": ("interferometer", MULTI),
    "interferometer_w_tilde_fft_size_levers_2026_10": ("interferometer", MULTI),
    "linear_solver_accuracy_2026_09": ("lens", "solver"),
    "matrix_free_pixelized_2026_09": ("imaging", MULTI),
    "multiband_pyloop_productized": ("misc", None),
    "multistart_prodigy_compile_census": ("misc", None),
    "nnls_solver_ledger": ("lens", "solver"),
    "numba_curvature_matrix_f_split": ("imaging", MULTI),
    "numba_interferometer_verdict": ("interferometer", MULTI),
    "numpy_deflections_cpu": ("lens", "deflections"),
    "point_source_cpu_campaign": ("point_source_image", "image_plane"),
    "point_source_defaults_campaign": (MULTI_DT, None),
    "point_source_gpu_breakdown_2026_09": (MULTI_DT, None),
    "point_source_shared_likelihood_breakdown": (MULTI_DT, None),
    "point_source_source_plane_campaign": ("point_source_source", "source_plane"),
    "preopt_breakdown_baseline": ("misc", None),
    "production_representative_cells": ("imaging", MULTI),
    "profiling_campaign_status_2026_09": ("imaging", MULTI),
    "sparse_vs_dense_inversion_path": ("imaging", MULTI),
    "timing_noise_audit_2026_10": ("misc", None),
    "vram_validation_2026_07_08": ("misc", None),
    "xla_autotune_triton_gemm": ("misc", None),
    # wiki/campaigns pages
    "a100_pixelized_baseline": ("imaging", MULTI),
    "certified_positive_solver": ("imaging", MULTI),
    "cluster_pointsolver": ("cluster", "image_plane"),
    "compile_time": ("misc", None),
    "critical_curves": ("lens", "critical_curves"),
    "delaunay_nn_a100_series": ("imaging", "delaunay_nn"),
    "fixed_lens_light": ("imaging", MULTI),
    "fixed_light_numba_cpu": ("imaging", MULTI),
    "gaussian_deflections_precompute": ("lens", "deflections"),
    "hst_gpu_residue": ("imaging", MULTI),
    "image_source_mappings": (MULTI_DT, None),
    "imaging_over_sampling": ("imaging", MULTI),
    "interferometer_likelihood": ("interferometer", MULTI),
    "interferometer_streaming": ("interferometer", "delaunay"),
    "linear_solver_accuracy": ("lens", "solver"),
    "mass_field_profiling": ("misc", None),
    "matrix_free_pixelized": ("imaging", MULTI),
    "measurement_tools": ("misc", None),
    "numba_interferometer_revisit": ("interferometer", MULTI),
    "point_source_gpu_breakdown": (MULTI_DT, None),
    "point_source_image_plane_cpu": ("point_source_image", "image_plane"),
    "point_source_source_plane": ("point_source_source", "source_plane"),
    "post_certified_breakdown": ("imaging", MULTI),
    "preoptimization_baseline": ("misc", None),
    "reference_notes": ("misc", None),
    "setup_baseline": ("misc", None),
}

VERSION_RE = re.compile(r"_v(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d+)(\+\d+)?")
RPIN_RE = re.compile(r"_r(\d{4})_(\d{1,2})_(\d{1,2})_(\d+)$")
DATED_RE = re.compile(r"(?:^|_)(20\d\d)_(\d\d)(?:_(\d\d))?(?=_|$)")
PROBE_RE = re.compile(r"(?:^|_)probes?(?:_|$)")
LEVER_RE = re.compile(r"(?:^|_)levers?\d*(?:_|$)")
ISO_DATE_RE = re.compile(r"^(20\d\d)-(\d\d)-(\d\d)")
SCAN_RE = re.compile(r"^(?:n\d+|r\d+(?:\.\d+)?)$")
SUFFIXES = (".unusable.json", ".declaration.json", ".json", ".png", ".stdout", ".out", ".md")


def _root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


def tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "results", "wiki/campaigns"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode()
    paths = [p for p in out.split("\0") if p]
    return sorted(
        p
        for p in paths
        if p.startswith("results/")
        or (p.startswith("wiki/campaigns/") and p.count("/") == 2 and p.endswith(".md"))
    )


def source_commit(root: Path) -> str | None:
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return run.stdout.strip() or None


def sha256(root: Path, path: str) -> str:
    return hashlib.sha256((root / path).read_bytes()).hexdigest()


def split_ext(name: str) -> tuple[str, str]:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    stem, dot, ext = name.rpartition(".")
    return (stem, ext) if dot else (name, "")


def load_json(root: Path, path: str):
    try:
        return json.loads((root / path).read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def companion_json(root: Path, path: str, files: set[str]) -> str | None:
    """The JSON a figure/log was written beside (same stem), when tracked."""
    parent, name = path.rsplit("/", 1)
    stem, ext = split_ext(name)
    if ext in ("json", "unusable.json"):
        return path
    for suffix in ("json", "unusable.json"):
        candidate = f"{parent}/{stem}.{suffix}"
        if candidate in files:
            return candidate
    return None


# --------------------------------------------------------------------------- facts


def version_tuple(text: str | None) -> tuple[int, ...] | None:
    if not isinstance(text, str):
        return None
    match = re.match(r"^v?(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d+)", text)
    return tuple(int(g) for g in match.groups()) if match else None


def pin_string(vt: tuple[int, ...] | None) -> str | None:
    return ".".join(str(v) for v in vt) if vt else None


def iso(y, m, d) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d or 1):02d}"


def facts(root: Path, path: str, files: set[str]) -> dict:
    """Release pin and date recorded by the file name or the companion JSON."""
    name = path.rsplit("/", 1)[1]
    stem, _ = split_ext(name)
    out = {"pin": None, "date": None, "date_basis": None, "doc": None}
    vmatch = None
    jpath = companion_json(root, path, files)
    doc = load_json(root, jpath) if jpath else None
    out["doc"] = doc if isinstance(doc, dict) else None
    if out["doc"]:
        d = out["doc"]
        out["pin"] = version_tuple(d.get("autolens_version") or d.get("release_version"))
        device = d.get("device") if isinstance(d.get("device"), dict) else {}
        prov = device.get("provenance") if isinstance(device.get("provenance"), dict) else {}
        for basis, value in (
            ("device.provenance.captured_at", prov.get("captured_at")),
            ("captured_at", d.get("captured_at")),
            ("timestamp", d.get("timestamp")),
            ("measured_at", d.get("measured_at")),
            ("generated_at", d.get("generated_at")),
        ):
            match = ISO_DATE_RE.match(value) if isinstance(value, str) else None
            if match:
                out["date"], out["date_basis"] = iso(*match.groups()), basis
                break
    vmatch = VERSION_RE.search(stem)
    rmatch = RPIN_RE.search(stem)
    name_pin = (
        tuple(int(g) for g in vmatch.groups()[:4])
        if vmatch
        else tuple(int(g) for g in rmatch.groups())
        if rmatch
        else None
    )
    out["pin"] = out["pin"] or name_pin
    if out["date"] is None:
        dmatch = DATED_RE.search(VERSION_RE.sub("", stem))
        if dmatch:
            out["date"], out["date_basis"] = iso(*dmatch.groups()), "file name date"
    # The series date also accepts the file-name release (results/README.md: the version
    # string is the release that produced the numbers). A JSON ``autolens_version`` is never
    # a date: source checkouts stamp a stale release.
    out["series_date"], out["series_basis"] = out["date"], out["date_basis"]
    if out["series_date"] is None and vmatch:
        out["series_date"] = iso(*vmatch.groups()[:3])
        out["series_basis"] = "file name release"
    return out


def json_host(doc: dict | None) -> tuple[str | None, str | None, str | None]:
    """(host_class, device, basis) from a recorded host name; (None, ...) if unknown."""
    if not doc:
        return None, None, None
    device = doc.get("device") if isinstance(doc.get("device"), dict) else {}
    prov = device.get("provenance") if isinstance(device.get("provenance"), dict) else {}
    host = prov.get("host") or device.get("hostname") or doc.get("hostname")
    backend = device.get("backend")
    dev = "gpu" if backend in ("gpu", "cuda") else "cpu" if backend == "cpu" else None
    if not isinstance(host, str) or not host:
        return None, dev, None
    h = host.lower()
    if "ral-gpu" in h:
        return ("hpc_a100" if dev == "gpu" else "gpunode_cpu"), dev, f"host {host}"
    if "ral-compute" in h:
        return "hpc_ral_cpu", dev, f"host {host}"
    if h.startswith("desktop-") or "laptop" in h:
        return ("local_gpu" if dev == "gpu" else "local_laptop"), dev, f"host {host}"
    return "unknown_host", dev, f"host {host}"


# --------------------------------------------------------------------------- routes


def cell_index(root: Path) -> dict:
    """{(dataset, 'likelihood_runtime'|'likelihood_breakdown'): {legacy stem: route}}."""
    routes = json.loads((root / ROUTES_PATH).read_text())["routes"]
    index: dict = defaultdict(dict)
    for row in routes:
        parts = PurePosixPath(row["legacy"]).parts
        stem = PurePosixPath(row["legacy"]).stem
        if len(parts) == 4 and parts[2] in ("likelihood_runtime", "likelihood_breakdown"):
            index[(parts[1], parts[2])][stem] = row
        elif len(parts) == 5 and parts[3] == "datacube":
            index[("datacube", parts[2])][stem] = row
    return index


def match_cell(index: dict, dataset: str, category: str, stem: str):
    """Longest legacy stem that prefixes ``stem`` at a token boundary."""
    stems = index.get((dataset, category), {})
    best = None
    for legacy in stems:
        if (stem == legacy or stem.startswith(legacy + "_")) and (
            best is None or len(legacy) > len(best)
        ):
            best = legacy
    return (best, stems[best], stem[len(best) + 1 :]) if best else (None, None, stem)


# --------------------------------------------------------------------------- classify


class Unresolved(Exception):
    def __init__(self, group: str, reason: str, proposed: str | None = None, **extra):
        super().__init__(reason)
        self.group, self.reason, self.proposed, self.extra = group, reason, proposed, extra


def archive_class(rel: str) -> tuple[str, str] | None:
    """(slug-class, why) for the listed dead-experiment classes, else None."""
    if rel.startswith(FROZEN_PACKS):
        return "frozen_pack", "frozen evidence pack"
    if rel.startswith(FROZEN_BASELINE):
        return "frozen_baseline", "baselines/PreOptimizationTimes"
    body = rel.rsplit(".", 1)[0]
    tokens = "/".join(split_ext(part)[0] for part in body.split("/"))
    for part in tokens.split("/"):
        if PROBE_RE.search(part):
            return "probes", "_probe experiment"
        if LEVER_RE.search(part):
            return "levers", "_lever experiment"
        if DATED_RE.search(VERSION_RE.sub("", part)):
            return "dated", "dated-suffix snapshot"
    return None


def slug_for(rel: str, cls: str) -> tuple[str, str]:
    """(slug, inner path) of an archived file."""
    if cls == "frozen_pack":
        pack = next(p for p in FROZEN_PACKS if rel.startswith(p))
        return pack.split("/")[1], rel[len(pack) :]
    if cls == "frozen_baseline":
        return "PreOptimizationTimes", rel[len(FROZEN_BASELINE) :]
    name = rel.rsplit("/", 1)[1]
    family = next((f for f in CAMPAIGN_FAMILIES if name.startswith(f)), None)
    return (family or cls), rel


def parse_config(rest: str):
    """Legacy config grammar: ``<tier>_<precision>[_<extra>...]`` -> pieces or None."""
    for tier in sorted(TIERS, key=len, reverse=True):
        if rest == tier or rest.startswith(tier + "_"):
            tail = rest[len(tier) + 1 :].split("_") if rest != tier else []
            if tail and tail[0] in PRECISIONS:
                return tier, tail[0], tail[1:]
            return None
    return None


VERSIONED_RE = re.compile(
    r"^(?:(?P<purpose>breakdown|likelihood_summary|summary)_)?"
    r"(?:(?P<inst>" + "|".join(INSTRUMENTS) + r")_)?"
    r"v(?P<ver>\d{4}\.\d{1,2}\.\d{1,2}\.\d+(?:\+\d+)?)(?P<extra>(?:_[a-z0-9.]+)*)$"
)


def operator_of(doc: dict | None, explicit: str | None, dataset: str) -> tuple[str, str]:
    config = (doc or {}).get("configuration")
    path = config.get("inversion_path") if isinstance(config, dict) else None
    from_doc = None
    if isinstance(path, str):
        low = path.lower()
        from_doc = (
            "sparse"
            if "sparse" in low
            else "dense"
            if ("dense" in low or "mapping" in low)
            else None
        )
    if from_doc and explicit and from_doc != explicit:
        raise Unresolved(
            "path/content disagree",
            f"operator: file name says {explicit}, configuration.inversion_path={path!r}",
        )
    if from_doc:
        return from_doc, "configuration.inversion_path"
    if explicit:
        return explicit, "file name"
    if dataset in POINT_LIKE:
        return "na", "point-source likelihood (no linear inversion)"
    return "dense", "legacy grammar: no _sparse suffix"


def precision_of(doc: dict | None, token: str | None, backend: str) -> tuple[str, str]:
    mixed = None
    if doc:
        mixed = doc.get("use_mixed_precision", doc.get("mixed_precision"))
        if not isinstance(mixed, bool) and isinstance(doc.get("precision"), dict):
            mixed = doc["precision"].get("use_mixed_precision")
    from_doc = ("mixed" if mixed else "fp64") if isinstance(mixed, bool) else None
    if token is not None:
        value = PRECISIONS[token]
        if value is None:
            raise Unresolved("precision outside setup vocabulary", f"precision token {token!r}")
        if from_doc and from_doc != value:
            raise Unresolved(
                "path/content disagree", f"precision: file name {token}, JSON mixed={mixed}"
            )
        return value, "file name"
    if from_doc:
        return from_doc, "use_mixed_precision"
    if backend == "numba":
        return "fp64", "numba path is fp64-only"
    raise Unresolved("setup axis not recorded", "precision not in file name or JSON")


def transformer_of(doc: dict | None, dataset: str) -> tuple[str | None, str | None]:
    if dataset not in INTERFEROMETRIC:
        return None, None
    name = (doc or {}).get("transformer")
    if not isinstance(name, str):
        config = (doc or {}).get("configuration")
        name = config.get("transformer") if isinstance(config, dict) else None
    if name in TRANSFORMERS:
        return TRANSFORMERS[name], "transformer"
    raise Unresolved(
        "setup axis not recorded", f"interferometric transformer not recorded ({name!r})"
    )


def instrument_of(doc, from_name, from_dir) -> tuple[str, str]:
    candidates = {}
    if from_dir:
        candidates["directory"] = from_dir
    if from_name:
        candidates["file name"] = from_name
    recorded = (doc or {}).get("instrument")
    if isinstance(recorded, str) and recorded:
        candidates["JSON instrument"] = recorded
    values = set(candidates.values())
    if len(values) > 1:
        raise Unresolved("path/content disagree", f"instrument: {candidates}")
    if not values:
        raise Unresolved("setup axis not recorded", "instrument not in path or JSON")
    value = values.pop()
    if value not in INSTRUMENTS:
        raise Unresolved("setup axis not recorded", f"instrument {value!r} not in vocabulary")
    return value, next(iter(candidates))


def classify_cell(root, rel, files, index, fx) -> dict:
    """runtime/** and breakdown/** -> a mirrored key, or Unresolved."""
    parts = rel.split("/")
    category = {"runtime": "likelihood_runtime", "breakdown": "likelihood_breakdown"}[parts[0]]
    measurement = parts[0]
    dataset = parts[1]
    stem, ext = split_ext(parts[-1])
    dirs = parts[2:-1]
    doc = fx["doc"]
    if stem == "comparison" and len(dirs) == 2:
        variant = {"pixelization": "rectangular"}.get(dirs[0], dirs[0])
        legacy, route, _ = match_cell(index, dataset, category, dirs[0])
        if route is None or dirs[1] not in INSTRUMENTS:
            raise Unresolved("no canonical cell", f"comparison under {'/'.join(dirs)}")
        variant = route["model"]
        return {
            "aggregate": True,
            "dataset_type": dataset,
            "variant": variant,
            "instrument": dirs[1],
            "new": f"results/{dataset}/{variant}/{dirs[1]}/comparison.{ext}",
            "basis": {"cell": f"directory {dirs[0]}"},
        }
    legacy, route, rest = match_cell(index, dataset, category, stem)
    dir_inst = next((d for d in dirs if d in INSTRUMENTS), None)
    other_dirs = [d for d in dirs if d not in INSTRUMENTS]
    if route is None:
        raise Unresolved(
            "no canonical cell", f"stem {stem!r} matches no {dataset}/{category} legacy route"
        )
    if CELL_MEASUREMENTS.get(route["measurement"]) is None:
        raise Unresolved(
            "experiment script output",
            f"route {route['path']} is not a per-likelihood runtime/breakdown script",
            dataset_type=dataset,
            variant=route["model"],
            archive_slug=route["measurement"],
        )
    if any(d != legacy for d in other_dirs):
        raise Unresolved(
            "size-scan row" if any("sweep" in d for d in other_dirs) else "scan or side folder",
            f"folder {'/'.join(other_dirs)} is not the {legacy} cell or an instrument level",
            dataset_type=dataset,
            variant=route["model"],
        )
    backend = CELL_MEASUREMENTS[route["measurement"]]
    variant = route["model"]
    basis = {"cell": f"script_routes {route['legacy']}"}
    name_inst = None
    vmatch = VERSIONED_RE.match(rest)
    extras: list[str]
    if vmatch:
        tier = token = None
        name_inst = vmatch.group("inst")
        extras = [e for e in vmatch.group("extra").split("_") if e]
    else:
        cfg = parse_config(rest)
        if cfg is None:
            raise Unresolved(
                "experiment arm",
                f"suffix {rest!r} is not the legacy config grammar",
                dataset_type=dataset,
                variant=variant,
            )
        tier, token, extras = cfg
        if RPIN_RE.search("_" + "_".join(extras)):
            extras = []  # _r<YYYY_MM_DD_N> is a release-pin marker, not an arm
    operator_token = None
    for extra in list(extras):
        if extra in ("sparse", "dense"):
            operator_token = extra
            extras.remove(extra)
    if extras:
        scan = all(SCAN_RE.match(e) for e in extras)
        raise Unresolved(
            "size-scan row" if scan else "experiment arm",
            f"extra suffix {'_'.join(extras)!r} on {legacy}",
            dataset_type=dataset,
            variant=variant,
        )
    host_doc, dev_doc, host_basis = json_host(doc)
    if tier is not None:
        host, device = TIERS[tier]
        if host is None:
            raise Unresolved("host outside host_class vocabulary", f"tier {tier}")
        if host_doc and host_doc != host:
            raise Unresolved("path/content disagree", f"host: tier {tier}, JSON {host_basis}")
        if dev_doc and dev_doc != device:
            raise Unresolved("path/content disagree", f"device: tier {tier}, JSON {dev_doc}")
        basis["host_class"] = f"tier {tier}"
    else:
        if host_doc is None:
            raise Unresolved(
                "host not recorded",
                "versioned summary with no host in file name or JSON",
                dataset_type=dataset,
                variant=variant,
            )
        if host_doc in ("gpunode_cpu", "unknown_host"):
            raise Unresolved("host outside host_class vocabulary", host_basis)
        host, device, basis["host_class"] = host_doc, dev_doc, host_basis
    if device is None:
        raise Unresolved("setup axis not recorded", "device not recorded")
    operator, basis["operator"] = operator_of(doc, operator_token, dataset)
    precision, basis["precision"] = precision_of(doc, token, backend)
    transformer, tbasis = transformer_of(doc, dataset)
    if tbasis:
        basis["transformer"] = tbasis
    instrument, basis["instrument"] = instrument_of(doc, name_inst, dir_inst)
    setup_id = f"{backend}_{device}_{operator}_{precision}" + (
        f"_{transformer}" if transformer else ""
    )
    return {
        "aggregate": False,
        "dataset_type": dataset,
        "variant": variant,
        "instrument": instrument,
        "setup_id": setup_id,
        "host_class": host,
        "measurement": measurement,
        "ext": ext,
        "stem": stem,
        "basis": basis,
    }


def ledger_target(name: str, out_name: str | None = None) -> str:
    """Route a ledger/page; ``out_name`` renames a ledger that shares a page's name."""
    route = LEDGER_ROUTES.get(name)
    out = out_name or name
    if route is None:
        raise Unresolved("ledger not routed", f"no LEDGER_ROUTES row for {name}")
    dataset, variant = route
    if dataset == MULTI_DT:
        raise Unresolved(
            "ledger spans several dataset types",
            "campaign covers more than one dataset type",
            proposed=f"wiki/campaigns/misc/{out}.md",
        )
    if variant == MULTI:
        raise Unresolved(
            "ledger spans several variants",
            f"campaign covers several {dataset} variants",
            proposed=f"wiki/campaigns/{dataset}/{out}.md",
        )
    if dataset == "misc":
        return f"wiki/campaigns/misc/{out}.md"
    return f"wiki/campaigns/{dataset}/{variant}/{out}.md"


def classify(root, path, files, index, fx) -> dict:
    """One file -> {kind, new, ...}; raises Unresolved."""
    if path.startswith("wiki/campaigns/"):
        name = path.rsplit("/", 1)[1][:-3]
        if name == "_template":
            return {"kind": "unchanged", "new": path}
        return {"kind": "wiki_campaign", "new": ledger_target(name)}
    rel = path[len("results/") :]
    top = rel.split("/", 1)[0]
    if rel in ("README.md", "provenance_grandfathered.txt"):
        return {"kind": "unchanged", "new": path}
    if top == "lens":
        return {"kind": "lens_component", "new": path}
    if top == "notes" and rel.count("/") == 1 and rel.endswith(".md"):
        name = rel[6:-3]
        # A ledger named like a campaign page keeps both files: the ledger gains _ledger.
        out = f"{name}_ledger" if f"wiki/campaigns/{name}.md" in files else name
        return {"kind": "wiki_campaign", "new": ledger_target(name, out)}
    cls = None if top == "logs" else archive_class(rel)  # log names carry their run date
    if cls:
        return {"kind": "archive", "archive_class": cls[0], "why": cls[1], "rel": rel}
    if top in ("runtime", "breakdown"):
        name = rel.rsplit("/", 1)[1]
        if name.startswith(CAMPAIGN_FAMILIES):
            raise Unresolved(
                "fixed-light campaign output",
                "model_variant light_fixed has no section-3.7 path level; campaign complete",
                dataset_type="imaging",
                variant=None,
                archive_slug="fixed_light",
            )
        if split_ext(name)[1] in ("out", "stdout"):
            folder = rel.split("/")[-2]
            raise Unresolved(
                "log capture outside logs/",
                "job stdout committed beside results, not under results/logs/<campaign>/",
                proposed=f"results/misc/logs/{rel.split('/')[1]}_{folder}/{name}",
            )
        if re.search(r"_job\d+$", split_ext(name)[0]):
            raise Unresolved(
                "campaign job sidecar",
                "per-job sidecar of a closed campaign, not a cell record",
                archive_slug="job_sidecars",
            )
        try:
            key = classify_cell(root, rel, files, index, fx)
        except Unresolved as exc:
            if (
                exc.group
                in (
                    "experiment script output",
                    "experiment arm",
                    "no canonical cell",
                    "scan or side folder",
                )
                and (fx["series_date"] or "9999") < PRE_SERIES_CUTOFF
            ):
                return {
                    "kind": "archive",
                    "archive_class": "pre_2026_09",
                    "why": f"pre-2026-09 series ({exc.group})",
                    "rel": rel,
                }
            raise
        return {"kind": "mirrored", **key}
    if top in MISC_GROUPS:
        if VERSION_RE.search(rel) and (fx["series_date"] or "9999") < PRE_SERIES_CUTOFF:
            return {
                "kind": "archive",
                "archive_class": "pre_2026_09",
                "why": "pre-2026-09 versioned series",
                "rel": rel,
            }
        inner = rel.split("/", 1)[1]
        return {"kind": "misc", "new": f"results/misc/{MISC_GROUPS[top]}/{inner}"}
    if top in EXPERIMENT_TOPS:
        if (fx["series_date"] or "9999") < PRE_SERIES_CUTOFF:
            return {
                "kind": "archive",
                "archive_class": "pre_2026_09",
                "why": "pre-2026-09 series (experiment folder)",
                "rel": rel,
            }
        dataset, variant = EXPERIMENT_TOPS[top]
        raise Unresolved(
            "experiment script output",
            f"results/{top}/ holds experiment outputs with no section-3.7 measurement slot",
            dataset_type=dataset,
            variant=variant,
        )
    raise Unresolved("unknown results folder", f"results/{top}/ has no rule")


# --------------------------------------------------------------------------- build


def proposal(unres: Unresolved, path: str, slug_months: dict, own: str | None) -> str | None:
    if unres.proposed:
        return unres.proposed
    slug = unres.extra.get("archive_slug")
    if slug is None and unres.group in ("experiment script output", "experiment arm"):
        slug = "experiments"
    if slug is None:
        return None
    month = (own or "")[:7].replace("-", "/") or slug_months.get(slug, "YYYY/MM")
    return f"archive/{month}/{slug}/{path[len('results/') :]}"


def build(root: Path) -> dict:
    root = Path(root)
    paths = tracked(root)
    files = set(paths)
    index = cell_index(root)
    entries, unresolved = {}, {}
    facts_by = {p: facts(root, p, files) for p in paths}
    classified = {}
    for path in paths:
        try:
            classified[path] = classify(root, path, files, index, facts_by[path])
        except Unresolved as exc:
            unresolved[path] = exc

    # Archive slugs: month = latest recorded date among the slug's files.
    slug_dates: dict = defaultdict(list)
    for path, c in classified.items():
        if c["kind"] == "archive":
            slug, _ = slug_for(c["rel"], c["archive_class"])
            c["slug"] = slug
            if facts_by[path]["series_date"]:
                slug_dates[slug].append(facts_by[path]["series_date"])
    for path, exc in unresolved.items():
        slug = exc.extra.get("archive_slug")
        if exc.group in ("experiment script output", "experiment arm"):
            slug = slug or "experiments"
        if slug and facts_by[path]["series_date"]:
            slug_dates[slug].append(facts_by[path]["series_date"])
    for path, c in classified.items():
        if c["kind"] == "archive" and c["archive_class"] == "frozen_pack":
            named = DATED_RE.search(c["slug"])
            if named:
                slug_dates[c["slug"]].append(iso(*named.groups()))
    slug_months = {s: max(d)[:7].replace("-", "/") for s, d in slug_dates.items()}
    slug_basis = {s: "latest file date in slug" for s in slug_months}
    for slug, (month, why) in SLUG_MONTH.items():
        if slug not in slug_months:
            slug_months[slug], slug_basis[slug] = month, why
    for path, c in list(classified.items()):
        if c["kind"] != "archive":
            continue
        slug, inner = slug_for(c["rel"], c["archive_class"])
        own = facts_by[path]["series_date"]
        if c["archive_class"] in ("frozen_pack", "frozen_baseline") or not own:
            month = slug_months.get(slug)
        else:
            month = own[:7].replace("-", "/")
        if month is None:
            unresolved[path] = Unresolved(
                "archive month unknown",
                f"{c['why']}: no date in the file name, its JSON or its slug {slug}",
                proposed=f"archive/YYYY/MM/{slug}/{inner}",
            )
            del classified[path]
            continue
        c["new"] = f"archive/{month}/{slug}/{inner}"

    # Mirrored keys: newest record -> current slot, older records -> history/.
    records: dict = defaultdict(lambda: defaultdict(list))
    for path, c in classified.items():
        if c["kind"] == "mirrored" and not c["aggregate"]:
            key = (
                c["dataset_type"],
                c["variant"],
                c["instrument"],
                c["setup_id"],
                c["host_class"],
                c["measurement"],
            )
            record = path.rsplit("/", 1)[0] + "/" + c["stem"]
            records[key][record].append(path)
    for key, recs in records.items():
        order = []
        for record, members in recs.items():
            fx = next(
                (facts_by[m] for m in members if facts_by[m]["doc"] is not None),
                facts_by[members[0]],
            )
            order.append((fx["pin"] or (), fx["series_date"] or "", record, members, fx))
        order.sort(key=lambda o: (o[0], o[1]), reverse=True)
        base = "results/" + "/".join(key[:5])
        measurement = key[5]
        if len(order) > 1 and (order[0][0], order[0][1]) == (order[1][0], order[1][1]):
            tied = [o for o in order if (o[0], o[1]) == (order[0][0], order[0][1])]
            for o in tied:
                for m in o[3]:
                    unresolved[m] = Unresolved(
                        "ambiguous current record",
                        f"{len(tied)} records share key {'/'.join(key)} with the same "
                        f"release pin and date: {sorted(t[2] for t in tied)}",
                    )
                    del classified[m]
            order = [o for o in order if o not in tied]
            current = None
        else:
            current = order[0][2] if order else None
        for pin, date, record, members, fx in order:
            for m in members:
                ext = classified[m]["ext"]
                c = classified[m]
                c["release_pin"] = pin_string(pin) if pin else None
                c["date"] = date or None
                c["date_basis"] = fx["series_basis"]
                if record == current:
                    c["new"] = f"{base}/{measurement}.{ext}"
                else:
                    c["kind"] = "history"
                    c["new"] = (
                        f"{base}/history/{pin_string(pin) or 'unpinned'}_{date or 'undated'}/"
                        f"{measurement}.{ext}"
                    )

    # Collisions are errors.
    targets: dict = defaultdict(list)
    for path, c in classified.items():
        targets[c["new"]].append(path)
    collisions = {new: olds for new, olds in targets.items() if len(olds) > 1}

    out_entries = []
    for path in sorted(classified):
        c = classified[path]
        entry = {"old": path, "new": c["new"], "sha256": sha256(root, path), "kind": c["kind"]}
        if c["kind"] in ("mirrored", "history"):
            entry["key"] = {
                k: c[k]
                for k in (
                    "dataset_type",
                    "variant",
                    "instrument",
                    "setup_id",
                    "host_class",
                    "measurement",
                    "release_pin",
                    "date",
                    "date_basis",
                )
                if c.get(k) is not None
            }
            entry["basis"] = c["basis"]
        if c["kind"] == "archive":
            entry["archive_class"] = c["archive_class"]
            entry["why"] = c["why"]
        out_entries.append(entry)
        entries[path] = entry
    out_unresolved = []
    for path in sorted(unresolved):
        exc = unresolved[path]
        row = {
            "old": path,
            "sha256": sha256(root, path),
            "group": exc.group,
            "reason": exc.reason,
        }
        hint = {k: v for k, v in exc.extra.items() if k != "archive_slug" and v is not None}
        if hint:
            row["derived"] = hint
        proposed = proposal(exc, path, slug_months, facts_by[path]["series_date"])
        if proposed:
            row["proposed"] = proposed
        out_unresolved.append(row)

    # Mirror witness: results/<dt>/<variant>/ vs scripts/<dt>/<variant>/.
    result_pairs = sorted(
        {
            (e["new"].split("/")[1], e["new"].split("/")[2])
            for e in out_entries
            if e["kind"] in ("mirrored", "history")
        }
    )
    script_pairs = sorted(
        (d.parent.name, d.name)
        for d in (root / "scripts").glob("*/*")
        if d.is_dir()
        and d.parent.name not in ("misc", "lens")
        and d.name != "__pycache__"
        and any(f.suffix == ".py" for f in d.iterdir())
    )
    misc_groups = sorted(
        {e["new"].split("/")[2] for e in out_entries if e["new"].startswith("results/misc/")}
    )
    script_misc = sorted(d.name for d in (root / "scripts/misc").iterdir() if d.is_dir())
    counts = defaultdict(int)
    for e in out_entries:
        counts[e["kind"]] += 1
    groups = defaultdict(int)
    for row in out_unresolved:
        groups[row["group"]] += 1
    covered = len(out_entries) + len(out_unresolved)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "issue": ISSUE,
        "generator": "scripts/misc/tooling/build_migration_map.py",
        "source_commit": source_commit(root),
        "note": (
            "Day-zero migration plan: old -> new path + sha256 for every tracked file under "
            "results/ and every wiki/campaigns/*.md page. Nothing has moved. Unresolved rows "
            "need a human ruling before the move-only PR; 'proposed' is a suggestion only."
        ),
        "layout": {
            "mirrored": "results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/<measurement>.<ext>",
            "history": "results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>/history/<release_pin>_<YYYY-MM-DD>/<measurement>.<ext>",
            "aggregate": "results/<dataset_type>/<variant>/<instrument>/comparison.<ext>",
            "setup_id": "<backend>_<device>_<operator>_<precision>[_<transformer>]",
            "archive": "archive/YYYY/MM/<slug>/<path under results/ or inside the pack>; YYYY/MM is the file's own date (name or JSON), else the latest date in its slug",
            "ledgers": "wiki/campaigns/<dataset_type>/<variant>/<name>.md | wiki/campaigns/{lens/<component>,misc}/<name>.md",
        },
        "counts": {
            "tracked_files": len(paths),
            "covered": covered,
            "by_kind": dict(sorted(counts.items())),
            "unresolved": len(out_unresolved),
            "unresolved_by_group": dict(sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "collisions": [
            {"new": new, "old": sorted(olds)} for new, olds in sorted(collisions.items())
        ],
        "mirror": {
            "result_variants": [f"{a}/{b}" for a, b in result_pairs],
            "result_variants_without_scripts": [
                f"{a}/{b}" for a, b in result_pairs if (a, b) not in script_pairs
            ],
            "script_variants_without_results": [
                f"{a}/{b}" for a, b in script_pairs if (a, b) not in result_pairs
            ],
            "misc_groups_without_scripts": [g for g in misc_groups if g not in script_misc],
        },
        "setup_ids": sorted(
            {e["key"]["setup_id"] for e in out_entries if "setup_id" in e.get("key", {})}
        ),
        "archive_slugs": {
            s: {"month": slug_months[s], "basis": slug_basis[s]} for s in sorted(slug_months)
        },
        "entries": out_entries,
        "unresolved": out_unresolved,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def comparable(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "source_commit"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the map is stale")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or _root()
    doc = build(root)
    if doc["collisions"]:
        for c in doc["collisions"]:
            print(f"COLLISION {c['new']} <- {c['old']}", file=sys.stderr)
        print("refusing to write a map with collisions", file=sys.stderr)
        return 1
    target = root / MAP_PATH
    if args.check:
        try:
            committed = json.loads(target.read_text())
        except (OSError, ValueError):
            print(f"{MAP_PATH} missing or unreadable", file=sys.stderr)
            return 1
        if comparable(committed) != comparable(doc):
            print(f"{MAP_PATH} is stale: re-run build_migration_map.py", file=sys.stderr)
            return 1
        print(f"{MAP_PATH} is current ({doc['counts']['covered']} files)")
        return 0
    target.write_text(render(doc))
    c = doc["counts"]
    print(f"wrote {MAP_PATH}: {c['covered']}/{c['tracked_files']} files, {c['by_kind']}")
    print(f"unresolved {c['unresolved']}: {c['unresolved_by_group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
