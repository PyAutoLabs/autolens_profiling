"""The linear-solver corpus: captured positive-only systems ``(Q, q)`` plus their reference solutions.

Every study in ``scripts/lens/solver/`` runs on the same frozen set of linear systems —
``Q = curvature_reg_matrix``, ``q = data_vector``, exactly as the JAX likelihood hands them to
``autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from`` (post
regularisation, pre Jacobi scaling). Capturing a system once and re-solving it offline is what
makes a solver comparison reproducible across releases: the system does not move when the
library's likelihood code does, so a changed metric is always a changed *solver*.

Layout (``results/lens/solver/corpus/``)
----------------------------------------

``manifest.json``
    ``{"schema": 1, "groups": [{"name", "npz", "source", "systems": [...]}, ...]}``. Each system
    entry carries ``name``, ``group``, ``source`` (``script``, ``args``, ``git_sha``), ``model``,
    ``n``, ``n_source_columns``, ``source_column_index_list``,
    ``no_regularization_index_list``, ``cond_Q``, ``max_abs_q``, ``library_versions`` and any
    group-specific extras. A field that is genuinely unknown is ``null`` — never a guess.
``<group>.npz``
    Compressed; ``Q_<name>``, ``q_<name>`` and ``x_ref_<name>`` per system. ``x_ref`` is the
    NumPy ``fnnls_cholesky`` solution (the production NumPy path's solver, started from the sign
    of the dense solve exactly as ``reconstruction_positive_only_from`` does without the memo),
    computed once when the group is added and stored, so a study never re-derives its truth.

API: :func:`load_corpus`, :func:`iter_systems`, :func:`add_group`, and the first ingest
:func:`import_571_fixture` (``python scripts/lens/solver/_corpus.py`` runs it).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
CORPUS_DIR = REPO_ROOT / "results" / "lens" / "solver" / "corpus"
MANIFEST_NAME = "manifest.json"
SCHEMA = 1


@dataclass
class System:
    """One captured positive-only system and its reference solution."""

    name: str
    group: str
    Q: np.ndarray
    q: np.ndarray
    x_ref: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.q.shape[0])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_manifest(corpus_dir: Path = CORPUS_DIR) -> dict:
    path = Path(corpus_dir) / MANIFEST_NAME
    if not path.is_file():
        return {"schema": SCHEMA, "groups": []}
    return json.loads(path.read_text())


def group_names(corpus_dir: Path = CORPUS_DIR) -> list[str]:
    return [g["name"] for g in read_manifest(corpus_dir)["groups"]]


def iter_systems(groups=None, corpus_dir: Path = CORPUS_DIR) -> Iterator[System]:
    """Yield every :class:`System` of the named groups (all groups when ``None``), in manifest order."""
    corpus_dir = Path(corpus_dir)
    manifest = read_manifest(corpus_dir)
    known = {g["name"] for g in manifest["groups"]}
    if groups is not None:
        unknown = sorted(set(groups) - known)
        if unknown:
            raise KeyError(f"unknown corpus group(s) {unknown}; known: {sorted(known)}")
    for group in manifest["groups"]:
        if groups is not None and group["name"] not in groups:
            continue
        with np.load(corpus_dir / group["npz"]) as arrays:
            for meta in group["systems"]:
                name = meta["name"]
                yield System(
                    name=name,
                    group=group["name"],
                    Q=np.asarray(arrays[f"Q_{name}"], dtype=np.float64),
                    q=np.asarray(arrays[f"q_{name}"], dtype=np.float64),
                    x_ref=np.asarray(arrays[f"x_ref_{name}"], dtype=np.float64),
                    meta=dict(meta),
                )


def load_corpus(groups=None, corpus_dir: Path = CORPUS_DIR) -> list[System]:
    """Every :class:`System` of the named groups (all groups when ``None``)."""
    return list(iter_systems(groups, corpus_dir=corpus_dir))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def reference_solution(Q: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The corpus truth: ``autoarray.util.fnnls.fnnls_cholesky`` from the dense-sign start.

    Identical to the NumPy branch of ``reconstruction_positive_only_from`` with the warm-start
    memo off (``P_initial = solve(Q, q) > 0``).
    """
    from autoarray.util.fnnls import fnnls_cholesky

    Q = np.asarray(Q, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.asarray(fnnls_cholesky(Q, q, P_initial=np.linalg.solve(Q, q) > 0), dtype=np.float64)


def _system_entry(group: str, spec: dict, source: dict) -> tuple[dict, dict]:
    Q = np.asarray(spec["Q"], dtype=np.float64)
    q = np.asarray(spec["q"], dtype=np.float64)
    n = int(q.shape[0])
    if Q.shape != (n, n):
        raise ValueError(f"system {spec['name']!r}: Q shape {Q.shape} does not match q ({n},)")
    x_ref = spec.get("x_ref")
    x_ref = reference_solution(Q, q) if x_ref is None else np.asarray(x_ref, dtype=np.float64)
    source_cols = spec.get("source_column_index_list")
    entry = {
        "name": spec["name"],
        "group": group,
        "source": dict(source),
        "model": spec.get("model"),
        "n": n,
        "n_source_columns": None if source_cols is None else len(source_cols),
        "source_column_index_list": None if source_cols is None else [int(i) for i in source_cols],
        "no_regularization_index_list": (
            None
            if spec.get("no_regularization_index_list") is None
            else [int(i) for i in spec["no_regularization_index_list"]]
        ),
        "cond_Q": float(np.linalg.cond(Q)),
        "max_abs_q": float(np.max(np.abs(q))),
        "library_versions": spec.get("library_versions"),
    }
    reserved = {"name", "Q", "q", "x_ref", *entry}
    entry.update({k: v for k, v in spec.items() if k not in reserved})
    arrays = {f"Q_{spec['name']}": Q, f"q_{spec['name']}": q, f"x_ref_{spec['name']}": x_ref}
    return entry, arrays


def add_group(name: str, systems: list[dict], source: dict, corpus_dir: Path = CORPUS_DIR) -> dict:
    """Add (or replace) corpus group ``name`` and return its manifest entry.

    ``systems`` is a list of dicts with at least ``name``, ``Q`` and ``q``; optional keys are
    ``model``, ``source_column_index_list``, ``no_regularization_index_list``,
    ``library_versions``, ``x_ref`` (else computed with :func:`reference_solution`) and any
    extra JSON-serialisable metadata, which is kept verbatim on the system's manifest entry.
    ``source`` is ``{"script", "args", "git_sha"}`` (plus extras) and is copied onto every
    system. System names must be unique within the group and match ``[A-Za-z0-9_]+``.
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in systems]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate system names in group {name!r}")
    for n in names:
        if not n.replace("_", "").isalnum():
            raise ValueError(f"system name {n!r} must match [A-Za-z0-9_]+")

    entries, arrays = [], {}
    for spec in systems:
        entry, arr = _system_entry(name, spec, source)
        entries.append(entry)
        arrays.update(arr)
    npz_name = f"{name}.npz"
    np.savez_compressed(corpus_dir / npz_name, **arrays)

    group = {"name": name, "npz": npz_name, "source": dict(source), "systems": entries}
    manifest = read_manifest(corpus_dir)
    manifest["schema"] = SCHEMA
    manifest["groups"] = [g for g in manifest["groups"] if g["name"] != name] + [group]
    (corpus_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return group


# ---------------------------------------------------------------------------
# Group 1: the PyAutoArray#571 SLaM MGE fixture
# ---------------------------------------------------------------------------

FIXTURE_571_GROUP = "slam_fixture_571"
FIXTURE_571_NPZ = (
    REPO_ROOT
    / "results"
    / "hazards"
    / "component"
    / "mge"
    / "nnls_capture_slam_hst_v2026.8.17.1.npz"
)

# Column layout of the capture's model, VERIFIED (not inferred) on 2026-09-30 by rebuilding the
# capture's dataset + model (``mge_nnls_capture._dataset`` / ``_slam_source_lp_model``) and
# fitting vector 26 on the NumPy path: ``inversion.param_range_list_from(LinearObj)`` =
# ``[0, 40]`` (lens galaxy, z=0.5, 2x20 MGE basis) and ``[40, 60]`` (source galaxy, z=1.0,
# 20-Gaussian MGE); ``inversion.no_regularization_index_list`` = 0..59 (every column is an
# unregularised linear light profile).
_FIXTURE_571_SOURCE_COLUMNS = list(range(40, 60))
_FIXTURE_571_NO_REG = list(range(60))
_FIXTURE_571_LAYOUT_NOTE = (
    "Column layout verified 2026-09-30 by rebuilding the capture's model at vector 26 (NumPy "
    "path): param_range_list_from(LinearObj) = [0,40] lens z=0.5, [40,60] source z=1.0; "
    "no_regularization_index_list = 0..59."
)


def _git_last_commit(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%h", "--", str(path)],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def import_571_fixture(corpus_dir: Path = CORPUS_DIR, npz_path: Path = FIXTURE_571_NPZ) -> dict:
    """Ingest the PyAutoArray#571 capture verbatim as group ``slam_fixture_571`` (8 systems).

    ``Q`` / ``q`` are copied bit-for-bit from the capture's ``.npz``; per-system metadata comes
    from its ``meta`` JSON string and its sibling summary JSON (``model``). ``source.git_sha``
    is ``null``: the capture records its *library* SHAs but not the autolens_profiling revision
    it ran at — the commit that added the fixture file is recorded separately as
    ``fixture_commit``.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        meta = json.loads(str(data["meta"]))
        raw = {k: np.asarray(data[k]) for k in data.files if k != "meta"}
    sibling = npz_path.with_suffix(".json")
    summary = json.loads(sibling.read_text()) if sibling.is_file() else {}

    source = {
        "script": "scripts/imaging/hazards/mge_nnls_capture.py",
        "args": [],
        "git_sha": None,
        "fixture": str(npz_path.relative_to(REPO_ROOT)),
        "fixture_commit": _git_last_commit(npz_path),
        "capture_date": meta.get("date"),
        "issue": meta.get("issue"),
        "dataset": summary.get("dataset"),
        "device": meta.get("device"),
        "x64": meta.get("x64"),
        "note": meta.get("note"),
    }
    systems = []
    for entry in meta["systems"]:
        k = entry["key"]
        systems.append(
            {
                "name": f"k{k}",
                "Q": raw[f"Q_{k}"],
                "q": raw[f"q_{k}"],
                "model": summary.get("model"),
                "source_column_index_list": _FIXTURE_571_SOURCE_COLUMNS,
                "no_regularization_index_list": _FIXTURE_571_NO_REG,
                "library_versions": meta.get("versions"),
                "column_layout_note": _FIXTURE_571_LAYOUT_NOTE,
                "fixture_key": k,
                "vector_index": entry.get("vector_index"),
                "category": entry.get("category"),
                "capture_converged_50": entry.get("converged_50"),
                "capture_pdip_iter_50": entry.get("pdip_iter_50"),
                "capture_converged_200": entry.get("converged_200"),
                "capture_pdip_iter_200": entry.get("pdip_iter_200"),
                "capture_objective_fnnls": entry.get("objective_fnnls"),
            }
        )
    return add_group(FIXTURE_571_GROUP, systems, source, corpus_dir=corpus_dir)


if __name__ == "__main__":
    group = import_571_fixture()
    for s in load_corpus([group["name"]]):
        obj = float(0.5 * s.x_ref @ s.Q @ s.x_ref - s.q @ s.x_ref)
        cap = s.meta.get("capture_objective_fnnls")
        print(
            f"{s.group}/{s.name}: n={s.n} cond={s.meta['cond_Q']:.3e} "
            f"objective={obj:.10g} capture={cap:.10g} n_pos={int(np.sum(s.x_ref > 0))}"
        )
    print(f"wrote {CORPUS_DIR / MANIFEST_NAME}")
