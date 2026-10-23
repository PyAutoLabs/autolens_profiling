"""Generate the day-zero migration map for ``results/`` (stdlib only).

Run from the repo root::

    python scripts/misc/tooling/build_migration_map.py          # (re)write the map
    python scripts/misc/tooling/build_migration_map.py --check  # exit 1 if the map is stale

The map, ``catalogue/migration_day_zero.json``, names an old -> new path and the
sha256 of the file content for every tracked file under ``results/`` and for every
``wiki/campaigns/*.md`` page. It is the reviewable plan for the move-only PR of the
likelihood-lifecycle epic (phase 0B, autolens_profiling#442); **nothing moves here**.

Target layout (epic report section 3.7, plus the human rulings of 2026-10-23)::

    <cell> = results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]
    <cell>/<measurement>.<ext>                                  # the current record of the key
    <cell>/history/<release_pin>_<YYYY-MM-DD>/<measurement>.<ext>  # older records, same key
    <cell>/scans/<name>/<measurement>.<ext>        # size scans (_nN, _rX, n_sweep/); never current
    <cell>/experiments/<name>/<measurement>.<ext>  # arms + experiment scripts; never current
    results/<dataset_type>/<variant>/<instrument>/comparison.<ext>  # cross-setup aggregate
    results/lens/<component>/...      results/misc/<group>/...
    wiki/campaigns/<dataset_type>/<variant>/<ledger>.md   (or /<dataset_type>/, lens/, misc/)
    archive/YYYY/MM/<slug>/...        (archive/undated/<slug>/... when no date is derivable)

``<model_variant>`` is present only for non-default model variants (``light_fixed``,
``rtu`` rectangular mesh, ``mge_mass``). ``setup_id`` is
``<backend>_<device>_<operator>_<precision>[_<transformer>]`` (ruling R1); the site/tier
moves to ``host_class``. Rules applied (human rulings, 2026-10-23):

* host: a CPU measurement is ``hpc_ral_cpu`` whatever RAL node ran it (``hpc_ral_cpu``
  tier on a ``euclid-ral-gpu-*`` host, ``hpc_ral_gpunode_cpu``); a versioned ``_v``
  summary with no host anywhere is archived, month from its ``_v`` release.
* transformer: ``setup_id`` names the likelihood route, not the data-transformer class.
  A sparse interferometric row is the W~ route -> ``_wtilde`` (report section 3.3), even
  though its JSON records ``TransformerNUFFT``; dense rows keep the recorded class
  (``_nufft`` / ``_dft``).

Every axis is derived from the path (the legacy config grammar, the
``catalogue/script_routes.json`` legacy stems, the ``scripts/<dt>/<variant>/`` script
names) or from the result JSON. **Nothing is guessed:** a canonical row whose axes cannot
be derived, or whose path and content disagree, is listed under ``unresolved`` with the
reason; a scan/experiment file with no derivable cell goes to ``results/misc/<group>/``.
Two old paths mapping to one new path is an error and the map is not written.

Pure stdlib -- no PyAuto* imports. Dates come only from file names and file content (no
git history), so the map is identical on a shallow CI clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

SCHEMA = "autolens-profiling-day-zero-migration"
VERSION = 2
MAP_PATH = "catalogue/migration_day_zero.json"
ROUTES_PATH = "catalogue/script_routes.json"
ISSUE = "PyAutoLabs/autolens_profiling#442"

#: Release-dated cut for the "pre-2026-09 series" dead-experiment class.
PRE_SERIES_CUTOFF = "2026-09-01"

INSTRUMENTS = ("alma_high", "alma", "ao", "euclid", "hst", "jvla", "jwst", "sdp81", "sma", "simple")

#: Legacy config-grammar host/tier token -> (host_class, device).
TIERS = {
    "hpc_ral_cpu_epyc7702": ("hpc_ral_cpu", "cpu"),
    "hpc_ral_gpunode_cpu": ("hpc_ral_cpu", "cpu"),  # ruling: CPU on any RAL node
    "hpc_ral_a100": ("hpc_a100", "gpu"),
    "hpc_ral_cpu": ("hpc_ral_cpu", "cpu"),
    "hpc_a100": ("hpc_a100", "gpu"),
    "local_rtx2060": ("local_gpu", "gpu"),
    "local_cpu": ("local_laptop", "cpu"),
    "local_gpu": ("local_gpu", "gpu"),
    "laptop_cpu": ("local_laptop", "cpu"),
}
TIER_TOKENS = sorted((t.split("_") for t in TIERS), key=len, reverse=True)
PRECISIONS = {"fp64": "fp64", "mp": "mixed", "fp32": None}  # fp32 is not a setup precision

#: Canonical per-likelihood script measurements (from script_routes) -> (backend,
#: model_variant). Any other route measurement is an experiment script.
CELL_MEASUREMENTS = {
    "likelihood_runtime": ("jax", None),
    "likelihood_breakdown": ("jax", None),
    "likelihood_runtime_numba": ("numba", None),
    "likelihood_breakdown_numba": ("numba", None),
    "likelihood_runtime_jax": ("jax", None),
    "likelihood_runtime_solved": ("jax", None),
    "likelihood_runtime_numba_mge_mass": ("numba", "mge_mass"),
}
#: File-name tokens that name a model variant (``--rect-mesh rtu`` appends ``_rtu``).
MODEL_VARIANT_TOKENS = {"rtu": "rtu"}
MESHES = (("delaunay", "nn"), ("delaunay",), ("rectangular",))
#: Fixed-light script tags measuring several backends in one file (no single setup).
MIXED_BACKEND_TAGS = {("cpu", "kernels")}
PURPOSE_TOKENS = {"breakdown", "summary", "likelihood"}
POINT_LIKE = ("point_source_image", "point_source_source", "cluster")
INTERFEROMETRIC = ("interferometer", "datacube")
TRANSFORMERS = {"TransformerNUFFT": "nufft", "TransformerDFT": "dft"}
CATEGORY = {"runtime": "likelihood_runtime", "breakdown": "likelihood_breakdown"}

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
    "parallel_scaling": "parallel_scaling",
    "streaming_scaling": "streaming_scaling",
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
#: ("misc", None) and ("lens", <component>) follow section 3.7; (<dt>, MULTI) goes to
#: wiki/campaigns/<dt>/ and (MULTI_DT, None) stays at wiki/campaigns/ (human ruling).
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
ANY_ISO_RE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
SCAN_RE = re.compile(r"^(?:n\d+|r\d+(?:\.\d+)?)$")
SIDECAR_RE = re.compile(r"_jobs?\d*$")
SUFFIXES = (
    ".unusable.json",
    ".declaration.json",
    ".locked.json",
    ".json",
    ".png",
    ".stdout",
    ".out",
    ".md",
)


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
    if ext.endswith("json"):
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
    """Release pin and dates recorded by the file name or the companion JSON."""
    jpath = companion_json(root, path, files)
    return facts_from_doc(path, load_json(root, jpath) if jpath else None)


def facts_from_doc(path: str, doc) -> dict:
    """``facts`` for a file whose companion JSON content is ``doc`` (may be in memory)."""
    name = path.rsplit("/", 1)[1]
    stem, _ = split_ext(name)
    out = {"pin": None, "date": None, "date_basis": None, "doc": None}
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
    out["versioned"] = bool(vmatch)
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
        # Ruling: a CPU measurement is hpc_ral_cpu whatever RAL node ran it.
        return ("hpc_a100" if dev == "gpu" else "hpc_ral_cpu"), dev, f"host {host}"
    if "ral-compute" in h:
        return "hpc_ral_cpu", dev, f"host {host}"
    if h.startswith("desktop-") or "laptop" in h:
        return ("local_gpu" if dev == "gpu" else "local_laptop"), dev, f"host {host}"
    return "unknown_host", dev, f"host {host}"


# --------------------------------------------------------------------------- routes


def cell_index(root: Path) -> dict:
    """{(dataset, legacy category dir): {legacy stem: route}} from script_routes."""
    routes = json.loads((root / ROUTES_PATH).read_text())["routes"]
    index: dict = defaultdict(dict)
    for row in routes:
        parts = PurePosixPath(row["legacy"]).parts
        stem = PurePosixPath(row["legacy"]).stem
        if len(parts) == 4:
            index[(parts[1], parts[2])][stem] = row
        elif len(parts) == 5 and parts[3] == "datacube":
            index[("datacube", parts[2])][stem] = row
    return index


def script_index(root: Path) -> dict:
    """{dataset: {script stem: variant}} over scripts/<dataset>/<variant>/*.py."""
    index: dict = defaultdict(dict)
    for script in sorted((root / "scripts").glob("*/*/*.py")):
        dataset, variant = script.parent.parent.name, script.parent.name
        if dataset not in ("misc", "lens") and not script.stem.startswith("_"):
            index[dataset][script.stem] = variant
    return index


def longest_prefix(stems, stem: str) -> str | None:
    best = None
    for legacy in stems:
        if (stem == legacy or stem.startswith(legacy + "_")) and (
            best is None or len(legacy) > len(best)
        ):
            best = legacy
    return best


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


def find_config(tokens: list[str]):
    """Locate ``<tier>[_<precision>]`` anywhere in the tokens -> (pre, tier, prec, post)."""
    for i in range(len(tokens)):
        for tier in TIER_TOKENS:
            if tokens[i : i + len(tier)] == tier:
                j = i + len(tier)
                prec = tokens[j] if j < len(tokens) and tokens[j] in PRECISIONS else None
                return tokens[:i], "_".join(tier), prec, tokens[j + (1 if prec else 0) :]
    return None


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


def transformer_of(doc: dict | None, dataset: str, operator: str):
    """Likelihood route, not transformer class: sparse interferometric rows are W~."""
    if dataset not in INTERFEROMETRIC:
        return None, None
    if operator == "sparse":
        return "wtilde", "sparse interferometric operator = W~ route (report 3.3)"
    name = (doc or {}).get("transformer")
    if not isinstance(name, str):
        config = (doc or {}).get("configuration")
        name = config.get("transformer") if isinstance(config, dict) else None
    if name in TRANSFORMERS:
        return TRANSFORMERS[name], "transformer"
    raise Unresolved(
        "setup axis not recorded", f"dense interferometric transformer not recorded ({name!r})"
    )


def nested_datasets(value, depth: int = 0) -> set[str]:
    """Instrument names recorded under nested ``instrument``/``dataset`` keys, including
    dataset paths ``dataset/<type>/<instrument>/...``."""
    found: set[str] = set()
    if depth > 4:
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("instrument", "dataset") and isinstance(item, str):
                if item in INSTRUMENTS:
                    found.add(item)
                parts = item.split("/")
                if len(parts) > 3 and parts[0] == "dataset" and parts[2] in INSTRUMENTS:
                    found.add(parts[2])
            else:
                found |= nested_datasets(item, depth + 1)
    elif isinstance(value, list):
        for item in value[:50]:
            found |= nested_datasets(item, depth + 1)
    return found


def instrument_of(doc, from_name, from_dir) -> tuple[str, str]:
    candidates = {}
    if from_dir:
        candidates["directory"] = from_dir
    if from_name:
        candidates["file name"] = from_name
    for key in ("instrument", "dataset"):
        recorded = (doc or {}).get(key)
        if isinstance(recorded, str) and recorded in INSTRUMENTS:
            candidates[f"JSON {key}"] = recorded
    if not candidates:
        nested = nested_datasets(doc)
        if len(nested) == 1:
            candidates["JSON nested dataset"] = nested.pop()
    values = set(candidates.values())
    if len(values) > 1:
        raise Unresolved("path/content disagree", f"instrument: {candidates}")
    if not values:
        raise Unresolved("setup axis not recorded", "instrument not in path or JSON")
    return values.pop(), next(iter(candidates))


def misc_fallback(rel: str, why: str) -> dict:
    top, inner = rel.split("/", 1)
    group = CATEGORY.get(top, top)
    return {"kind": "misc", "new": f"results/misc/{group}/{inner}", "why": why}


def strip_tokens(stem: str) -> list[str]:
    stem = RPIN_RE.sub("", VERSION_RE.sub("", stem))
    return [t for t in stem.split("_") if t]


def cell_identity(rel, stem, dataset, category, routes, scripts):
    """(variant, backend, model_variant, arm prefix, legacy, rest tokens) or a fallback dict."""
    if stem.startswith("fixed_light_"):
        tokens = strip_tokens(stem)[2:]
        for i in range(len(tokens)):
            mesh = next((m for m in MESHES if tuple(tokens[i : i + len(m)]) == m), None)
            if mesh:
                tag = tokens[:i]
                rest = [
                    t for t in tokens[i + len(mesh) :] if t not in ("fixed", "light") + tuple(tag)
                ]
                if tuple(tag) in MIXED_BACKEND_TAGS:
                    return misc_fallback(rel, "fixed-light kernel file measures several backends")
                backend = "numba" if "numba" in tag else "jax"
                return "_".join(mesh), backend, "light_fixed", tag, "fixed_light", rest
        return misc_fallback(rel, "fixed-light file names no mesh")
    stems = routes.get((dataset, category), {})
    legacy = longest_prefix(stems, stem)
    if legacy:
        route = stems[legacy]
        rest = strip_tokens(stem[len(legacy) + 1 :])
        cell = CELL_MEASUREMENTS.get(route["measurement"])
        if cell:
            return route["model"], cell[0], cell[1], [], legacy, rest
        backend = "numba" if "numba" in route["measurement"] else "jax"
        variant = route["model"]
        if variant == "pixelized":  # legacy name retired: the mesh token names the variant
            for i in range(len(rest)):
                mesh = next((m for m in MESHES if tuple(rest[i : i + len(m)]) == m), None)
                if mesh:
                    variant, rest = "_".join(mesh), rest[:i] + rest[i + len(mesh) :]
                    break
            else:
                return misc_fallback(rel, f"{route['path']} output names no mesh")
        return variant, backend, None, [route["measurement"]], legacy, rest
    script = longest_prefix(scripts.get(dataset, {}), stem)
    if script:
        rest = strip_tokens(stem[len(script) + 1 :])
        backend = "numba" if "numba" in script else "jax"
        return scripts[dataset][script], backend, None, [script], script, rest
    return misc_fallback(rel, f"stem {stem!r} matches no route or scripts/{dataset}/ script")


def classify_cell(rel, fx, routes, scripts) -> dict:
    """runtime/** and breakdown/** -> mirrored / scan / experiment / misc, or Unresolved."""
    parts = rel.split("/")
    top, dataset = parts[0], parts[1]
    category, measurement = CATEGORY[top], top
    stem, ext = split_ext(parts[-1])
    dirs = parts[2:-1]
    doc = fx["doc"]
    if stem == "comparison":
        legacy = longest_prefix(routes.get((dataset, category), {}), dirs[0]) if dirs else None
        if len(dirs) != 2 or legacy is None or dirs[1] not in INSTRUMENTS:
            return misc_fallback(rel, "cross-setup comparison with no instrument level")
        variant = routes[(dataset, category)][legacy]["model"]
        return {
            "kind": "mirrored",
            "aggregate": True,
            "dataset_type": dataset,
            "variant": variant,
            "instrument": dirs[1],
            "new": f"results/{dataset}/{variant}/{dirs[1]}/comparison.{ext}",
            "basis": {"cell": f"directory {dirs[0]}"},
        }
    ident = cell_identity(rel, stem, dataset, category, routes, scripts)
    if isinstance(ident, dict):
        return ident
    variant, backend, model_variant, arm, legacy, tokens = ident
    all_legacy = set(routes.get((dataset, category), {})) | {"pixelization"}
    dir_inst = next((d for d in dirs if d in INSTRUMENTS), None)
    scan_dirs = [d for d in dirs if "sweep" in d]
    side_dirs = [
        d for d in dirs if d not in INSTRUMENTS and d not in all_legacy and d not in scan_dirs
    ]
    basis = {"cell": f"{legacy} ({'route' if legacy in all_legacy else 'script'})"}
    found = find_config(tokens)
    if found:
        pre, tier, prec_token, post = found
    else:
        pre, tier, prec_token, post = tokens, None, None, []
    extras = pre + post
    if fx["versioned"]:
        extras = [t for t in extras if t not in PURPOSE_TOKENS]
    operator_token = next((t for t in extras if t in ("sparse", "dense")), None)
    extras = [t for t in extras if t not in ("sparse", "dense")]
    joined = "_".join(extras)
    name_inst = None
    for inst in INSTRUMENTS:  # longest names first ("alma_high" before "alma")
        if re.search(rf"(?:^|_){inst}(?:_|$)", joined):
            name_inst = inst
            joined = re.sub(rf"(?:^|_){inst}(?=_|$)", "", joined, count=1).strip("_")
            break
    extras = [t for t in joined.split("_") if t]
    for token, mv in MODEL_VARIANT_TOKENS.items():
        if token in extras:
            extras.remove(token)
            model_variant = mv
    arm_words = {w for a in arm for w in a.split("_")}
    extras = side_dirs + [
        t for t in extras if t not in arm_words
    ]  # e.g. matrix_free_..._matrix_free
    if arm or (extras and not all(SCAN_RE.match(t) for t in extras)):
        slot = ("experiments", "_".join(arm + extras))
    elif extras or scan_dirs:
        slot = ("scans", "_".join(scan_dirs + extras))
    else:
        slot = None
    try:
        host_doc, dev_doc, host_basis = json_host(doc)
        if tier is not None:
            host, device = TIERS[tier]
            if host_doc and host_doc != host:
                raise Unresolved("path/content disagree", f"host: tier {tier}, JSON {host_basis}")
            if dev_doc and dev_doc != device:
                raise Unresolved("path/content disagree", f"device: tier {tier}, JSON {dev_doc}")
            basis["host_class"] = f"tier {tier}"
        elif host_doc is None and fx["versioned"]:
            return {
                "kind": "archive",
                "archive_class": "hostless_versioned",
                "why": "versioned summary with no host in file name or JSON (ruling)",
                "rel": rel,
            }
        elif host_doc is None:
            raise Unresolved("setup axis not recorded", "no host in file name or JSON")
        elif host_doc == "unknown_host":
            raise Unresolved("host outside host_class vocabulary", host_basis)
        else:
            host, device, basis["host_class"] = host_doc, dev_doc, host_basis
        if device is None:
            raise Unresolved("setup axis not recorded", "device not recorded")
        operator, basis["operator"] = operator_of(doc, operator_token, dataset)
        precision, basis["precision"] = precision_of(doc, prec_token, backend)
        transformer, tbasis = transformer_of(doc, dataset, operator)
        if tbasis:
            basis["transformer"] = tbasis
        instrument, basis["instrument"] = instrument_of(doc, name_inst, dir_inst)
    except Unresolved as exc:
        if slot is None:
            raise
        return misc_fallback(rel, f"{slot[0][:-1]} with no derivable cell: {exc.reason}")
    setup_id = f"{backend}_{device}_{operator}_{precision}" + (
        f"_{transformer}" if transformer else ""
    )
    cell = f"results/{dataset}/{variant}/{instrument}/{setup_id}/{host}" + (
        f"/{model_variant}" if model_variant else ""
    )
    if slot:
        name = slot[1] + (f"_v{pin_string(fx['pin'])}" if fx["versioned"] else "")
        return {
            "kind": slot[0][:-1] if slot[0] == "scans" else "experiment",
            "new": f"{cell}/{slot[0]}/{name}/{measurement}.{ext}",
            "cell": cell,
            "basis": basis,
        }
    return {
        "kind": "mirrored",
        "aggregate": False,
        "dataset_type": dataset,
        "variant": variant,
        "instrument": instrument,
        "setup_id": setup_id,
        "host_class": host,
        "model_variant": model_variant,
        "measurement": measurement,
        "ext": ext,
        "stem": stem,
        "cell": cell,
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
        return f"wiki/campaigns/{out}.md"
    if variant == MULTI:
        return f"wiki/campaigns/{dataset}/{out}.md"
    if dataset == "misc":
        return f"wiki/campaigns/misc/{out}.md"
    return f"wiki/campaigns/{dataset}/{variant}/{out}.md"


def classify(path, files, routes, scripts, fx) -> dict:
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
    pre_series = (fx["series_date"] or "9999") < PRE_SERIES_CUTOFF
    if top in CATEGORY:
        name = rel.rsplit("/", 1)[1]
        stem, ext = split_ext(name)
        if ext in ("out", "stdout"):
            folder = rel.split("/")[-2]
            campaign = folder if folder == rel.split("/")[1] else f"{rel.split('/')[1]}_{folder}"
            return {
                "kind": "misc",
                "new": f"results/misc/logs/{campaign}/{name}",
                "why": "log capture outside results/logs/",
            }
        if SIDECAR_RE.search(stem):
            return {"kind": "sidecar", "rel": rel}
        result = classify_cell(rel, fx, routes, scripts)
        if result["kind"] in ("scan", "experiment", "misc") and pre_series:
            return {
                "kind": "archive",
                "archive_class": "pre_2026_09",
                "why": f"pre-2026-09 series ({result['kind']})",
                "rel": rel,
            }
        return result
    if top in MISC_GROUPS:
        if VERSION_RE.search(rel) and pre_series:
            return {
                "kind": "archive",
                "archive_class": "pre_2026_09",
                "why": "pre-2026-09 versioned series",
                "rel": rel,
            }
        inner = rel.split("/", 1)[1]
        return {"kind": "misc", "new": f"results/misc/{MISC_GROUPS[top]}/{inner}"}
    raise Unresolved("unknown results folder", f"results/{top}/ has no rule")


# --------------------------------------------------------------------------- writers


def destination(root: Path, path: str, doc: dict | None = None) -> tuple[str, str]:
    """(new path, kind) for a result a writer is about to create at legacy ``path``.

    ``path`` is the repo-relative path the writer would have used before the move
    (``results/runtime/...`` / ``results/breakdown/...``); ``doc`` is the result JSON (or
    the setup facts known so far: ``instrument``, ``use_mixed_precision``, ``device``,
    ``transformer``, ...). The same classification as the day-zero map places it:
    a canonical row goes to its cell's current slot, a size scan or experiment arm to
    ``<cell>/scans|experiments/<name>/``. A row whose cell cannot be derived, or that the
    map would archive, goes to ``results/misc/<group>/`` -- never to an archive or a
    guessed cell.
    """
    root = Path(root)
    routes, scripts = cell_index(root), script_index(root)
    fx = facts_from_doc(path, doc)
    rel = path[len("results/") :]
    try:
        c = classify(path, set(), routes, scripts, fx)
    except Unresolved as exc:
        c = misc_fallback(rel, f"no derivable cell: {exc.reason}")
    if c["kind"] in ("archive", "sidecar"):
        c = misc_fallback(rel, f"would be archived: {c.get('why', c['kind'])}")
    if c["kind"] == "mirrored" and not c["aggregate"]:
        return f"{c['cell']}/{c['measurement']}.{c['ext']}", "mirrored"
    return c["new"], c["kind"]


def applied(root: Path, doc: dict) -> bool:
    """True once ``apply_migration_map.py --apply`` has moved the tree."""
    moves = [e for e in doc.get("entries", []) if e["new"] != e["old"]]
    # A pointer README written at an emptied folder's old README path is a new file.
    return bool(moves) and all(
        (root / e["new"]).exists()
        and (not (root / e["old"]).exists() or sha256(root, e["old"]) != e["sha256"])
        for e in moves
    )


# --------------------------------------------------------------------------- build


def ledger_dates(root: Path, files: set[str]) -> list[tuple[str, str, str]]:
    """(ledger path, date, text) for every results/notes ledger with a derivable date."""
    out = []
    for path in sorted(files):
        if path.startswith("results/notes/") and path.count("/") == 2 and path.endswith(".md"):
            text = (root / path).read_text()
            named = DATED_RE.search(path.rsplit("/", 1)[1][:-3])
            heading = ANY_ISO_RE.search(text.split("\n", 1)[0])
            if named:
                out.append((path, iso(*named.groups()), text))
            elif heading:
                out.append((path, iso(*heading.groups()), text))
    return out


def place_sidecar(root, path, classified) -> str | None:
    """A job sidecar goes beside the results it names (their common cell directory)."""
    doc = load_json(root, path) or {}
    text = json.dumps(doc)
    names = set((doc.get("artifact_sha256") or {}).keys()) if isinstance(doc, dict) else set()
    names |= {p.rsplit("/", 1)[1] for p in re.findall(r"results/[A-Za-z0-9_./+-]+", text)}
    by_name = defaultdict(list)
    for old, c in classified.items():
        by_name[old.rsplit("/", 1)[1]].append(c)
    targets = []
    for name in sorted(names):
        for c in by_name.get(name, []):
            if c.get("kind") in ("mirrored", "history", "scan", "experiment"):
                targets.append(c["new"].rsplit("/", 1)[0])
    if not targets:
        return None
    common = os.path.commonpath(targets)
    for marker in ("/experiments/", "/scans/"):
        if marker in common + "/":
            common = (common + "/").split(marker)[0] + marker.rstrip("/")
    if common.count("/") < 5:
        return None
    return f"{common}/{path.rsplit('/', 1)[1]}"


def build(root: Path) -> dict:
    root = Path(root)
    paths = tracked(root)
    files = set(paths)
    routes, scripts = cell_index(root), script_index(root)
    unresolved = {}
    facts_by = {p: facts(root, p, files) for p in paths}
    classified = {}
    for path in paths:
        try:
            classified[path] = classify(path, files, routes, scripts, facts_by[path])
        except Unresolved as exc:
            unresolved[path] = exc

    # Archive: month = the file's own date, else the latest date in its slug, else the
    # date of a ledger that names the file, else archive/undated/<slug>/.
    slug_dates: dict = defaultdict(list)
    for path, c in classified.items():
        if c["kind"] == "archive":
            c["slug"], c["inner"] = slug_for(c["rel"], c["archive_class"])
            if c["archive_class"] == "hostless_versioned":
                c["slug"] = "hostless_summaries"
            if facts_by[path]["series_date"]:
                slug_dates[c["slug"]].append(facts_by[path]["series_date"])
            named = DATED_RE.search(c["slug"]) if c["archive_class"] == "frozen_pack" else None
            if named:
                slug_dates[c["slug"]].append(iso(*named.groups()))
    slug_months = {s: max(d)[:7].replace("-", "/") for s, d in slug_dates.items()}
    slug_basis = {s: "latest file date in slug" for s in slug_months}
    for slug, (month, why) in SLUG_MONTH.items():
        if slug not in slug_months:
            slug_months[slug], slug_basis[slug] = month, why
    ledgers = ledger_dates(root, files)
    packs = {c["slug"] for c in classified.values() if c.get("archive_class") == "frozen_pack"}
    for slug in sorted(packs):
        if slug not in slug_months:
            cite = next(((lp, d) for lp, d, text in ledgers if slug in text), None)
            if cite:
                slug_months[slug] = cite[1][:7].replace("-", "/")
                slug_basis[slug] = f"date of citing ledger {cite[0]}"
    for path, c in classified.items():
        if c["kind"] != "archive":
            continue
        own = facts_by[path]["series_date"]
        pack = c["archive_class"] in ("frozen_pack", "frozen_baseline")
        month = None if pack or not own else own[:7].replace("-", "/")
        if month is None and not pack:
            stem = split_ext(path.rsplit("/", 1)[1])[0]
            cite = next(((lp, d) for lp, d, text in ledgers if stem in text), None)
            if cite:
                month = cite[1][:7].replace("-", "/")
                c["why"] += f"; month from citing ledger {cite[0]}"
        if month is None:
            month = slug_months.get(c["slug"])
        if month is None:
            month = "undated"
        c["new"] = f"archive/{month}/{c['slug']}/{c['inner']}"

    # Mirrored keys: newest record -> current slot, older records -> history/.
    records: dict = defaultdict(lambda: defaultdict(list))
    for path, c in classified.items():
        if c["kind"] == "mirrored" and not c["aggregate"]:
            key = (c["cell"], c["measurement"])
            record = path.rsplit("/", 1)[0] + "/" + c["stem"]
            records[key][record].append(path)
    for (cell, measurement), recs in records.items():
        order = []
        for record, members in recs.items():
            fx = next(
                (facts_by[m] for m in members if facts_by[m]["doc"] is not None),
                facts_by[members[0]],
            )
            order.append((fx["pin"] or (), fx["series_date"] or "", record, members, fx))
        order.sort(key=lambda o: (o[0], o[1]), reverse=True)
        current = order[0][2]
        if len(order) > 1 and (order[0][0], order[0][1]) == (order[1][0], order[1][1]):
            tied = [o for o in order if (o[0], o[1]) == (order[0][0], order[0][1])]
            for o in tied:
                for m in o[3]:
                    unresolved[m] = Unresolved(
                        "ambiguous current record",
                        f"{len(tied)} records share {cell}/{measurement} with the same "
                        f"release pin and date: {sorted(t[2] for t in tied)}",
                    )
                    del classified[m]
            order = [o for o in order if o not in tied]
            current = None
        for pin, date, record, members, fx in order:
            for m in members:
                c = classified[m]
                c["release_pin"] = pin_string(pin) if pin else None
                c["date"] = date or None
                c["date_basis"] = fx["series_basis"]
                if record == current:
                    c["new"] = f"{cell}/{measurement}.{c['ext']}"
                else:
                    c["kind"] = "history"
                    c["new"] = (
                        f"{cell}/history/{pin_string(pin) or 'unpinned'}_{date or 'undated'}/"
                        f"{measurement}.{c['ext']}"
                    )

    # Job sidecars go beside the results they describe.
    for path, c in list(classified.items()):
        if c["kind"] == "sidecar":
            target = place_sidecar(root, path, classified)
            if target is None:
                classified[path] = misc_fallback(
                    path[len("results/") :],
                    "job sidecar: the results it names span several cells, or none is tracked",
                )
            else:
                classified[path] = {"kind": "experiment", "new": target, "why": "job sidecar"}

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
                    "model_variant",
                    "measurement",
                    "release_pin",
                    "date",
                    "date_basis",
                )
                if c.get(k) is not None
            }
        if c.get("basis"):
            entry["basis"] = c["basis"]
        if c["kind"] == "archive":
            entry["archive_class"] = c["archive_class"]
        if c.get("why"):
            entry["why"] = c["why"]
        out_entries.append(entry)
    out_unresolved = []
    for path in sorted(unresolved):
        exc = unresolved[path]
        row = {"old": path, "sha256": sha256(root, path), "group": exc.group}
        row["reason"] = exc.reason
        if exc.proposed:
            row["proposed"] = exc.proposed
        out_unresolved.append(row)

    # Mirror witness: results/<dt>/<variant>/ vs scripts/<dt>/<variant>/.
    in_tree = ("mirrored", "history", "scan", "experiment")
    result_pairs = sorted(
        {
            (e["new"].split("/")[1], e["new"].split("/")[2])
            for e in out_entries
            if e["kind"] in in_tree and not e["new"].startswith("results/misc/")
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
    cell = "results/<dataset_type>/<variant>/<instrument>/<setup_id>/<host_class>[/<model_variant>]"
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "issue": ISSUE,
        "generator": "scripts/misc/tooling/build_migration_map.py",
        "source_commit": source_commit(root),
        "note": (
            "Day-zero migration plan: old -> new path + sha256 for every tracked file under "
            "results/ and every wiki/campaigns/*.md page. Nothing has moved. Unresolved rows "
            "need a human ruling before the move-only PR."
        ),
        "layout": {
            "cell": cell,
            "mirrored": "<cell>/<measurement>.<ext> -- the only slot that may become current",
            "history": "<cell>/history/<release_pin>_<YYYY-MM-DD>/<measurement>.<ext> -- superseded",
            "scan": "<cell>/scans/<name>/<measurement>.<ext> -- size scans; never current",
            "experiment": (
                "<cell>/experiments/<name>/<measurement>.<ext> (and job sidecars in "
                "<cell>/experiments/) -- arms and experiment scripts; never current"
            ),
            "aggregate": "results/<dataset_type>/<variant>/<instrument>/comparison.<ext>",
            "setup_id": (
                "<backend>_<device>_<operator>_<precision>[_<transformer>]; the transformer "
                "names the likelihood route: sparse interferometric = wtilde"
            ),
            "model_variant": "path segment only for non-default variants (light_fixed, rtu, mge_mass)",
            "misc": "results/misc/<group>/... -- files with no derivable cell; never current",
            "archive": (
                "archive/YYYY/MM/<slug>/<path under results/ or inside the pack>; YYYY/MM is "
                "the file's own date (name or JSON), else the latest date in its slug, else a "
                "citing ledger's date, else archive/undated/<slug>/"
            ),
            "ledgers": (
                "wiki/campaigns/<dataset_type>/<variant>/<name>.md | "
                "wiki/campaigns/<dataset_type>/<name>.md (several variants) | "
                "wiki/campaigns/<name>.md (several dataset types) | "
                "wiki/campaigns/{lens/<component>,misc}/<name>.md"
            ),
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
            {e["new"].split("/")[4] for e in out_entries if e["kind"] in in_tree}
            - {"comparison.json", "comparison.png"}
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
    target = root / MAP_PATH
    if args.check and target.exists():
        committed = json.loads(target.read_text())
        if applied(root, committed):
            print(
                f"{MAP_PATH} is applied (day-zero map, frozen): "
                f"{len([e for e in committed['entries'] if e['new'] != e['old']])} moves in the tree"
            )
            return 0
    doc = build(root)
    if doc["collisions"]:
        for c in doc["collisions"]:
            print(f"COLLISION {c['new']} <- {c['old']}", file=sys.stderr)
        print("refusing to write a map with collisions", file=sys.stderr)
        return 1
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
