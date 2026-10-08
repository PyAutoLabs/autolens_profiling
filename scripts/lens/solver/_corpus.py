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
    Compressed; ``Q_<name>``, ``q_<name>`` and ``x_ref_<name>`` per system. A group whose manifest
    entry carries ``"encoding": "sym_tri_xor"`` (the n ~ 1500 Mapper groups, whose dense ``Q``
    would put one group past GitHub's 100 MB file limit) stores ``Q`` losslessly as
    ``Qtri_<name>`` — the upper triangle, diagonal included, in ``numpy.triu_indices`` order — and
    ``Qxor_<name>``, the bitwise XOR (``uint64``) of the transposed lower triangle with it, which is
    zero wherever ``Q`` is bitwise symmetric and so compresses to almost nothing;
    :func:`iter_systems` rebuilds ``Q`` bit for bit (checked when the group is written). ``x_ref`` is the
    NumPy ``fnnls_cholesky`` solution (the production NumPy path's solver, started from the sign
    of the dense solve exactly as ``reconstruction_positive_only_from`` does without the memo),
    computed once when the group is added and stored, so a study never re-derives its truth.

Storage (``"storage"`` on each group)
-------------------------------------

``"git"`` groups (the small ones) are committed beside the manifest. ``"external"`` groups — the
n ~ 1500 Mapper groups ``delaunay_hst``, ``rectangular_hst``, ``slam_mixed_hst``, 53–59 MB each —
are **not** in git (autolens_profiling#399, human decision 2026-10-08): their ``.npz`` is
gitignored by name and kept as copies at :data:`EXTERNAL_COPIES`. Every group's manifest entry
records ``sha256`` and ``bytes`` of its ``.npz``, its ``encoding`` and ``regenerate`` (the exact
capture command plus the library tag / profiling revision it ran at). Reading an external group
whose file is absent raises :class:`ExternalCorpusMissing` naming the file, its sha256, where the
copies live and the regenerate command; a present external file is checked against its sha256
before use. A regenerated file reproduces the systems, not necessarily the stored bytes (the
``.npz`` zip members carry write timestamps), so a fresh capture records its own sha256.

API: :func:`load_corpus`, :func:`iter_systems`, :func:`add_group`, and the first ingest
:func:`import_571_fixture` (``python scripts/lens/solver/_corpus.py`` runs it).
"""

from __future__ import annotations

import hashlib
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

STORAGES = ("git", "external")
# Where the external groups' .npz copies live (also written into each external group's manifest
# entry as ``copies``). A copy is placed into CORPUS_DIR of the checkout that needs it.
EXTERNAL_COPIES = (
    "RAL: /mnt/ral/jnightin/autolens_profiling_corpus/<npz>",
    "laptop: the canonical autolens_profiling checkout's results/lens/solver/corpus/<npz> (gitignored)",
)


class ExternalCorpusMissing(FileNotFoundError):
    """An ``"external"`` corpus group's ``.npz`` is not present in the corpus directory."""


class CorpusHashMismatch(ValueError):
    """An external corpus group's ``.npz`` does not match the sha256 in the manifest."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        encoding = group.get("encoding", "dense")
        with np.load(_group_npz_path(group, corpus_dir)) as arrays:
            for meta in group["systems"]:
                name = meta["name"]
                yield System(
                    name=name,
                    group=group["name"],
                    Q=_decode_Q(arrays, name, encoding, int(meta["n"])),
                    q=np.asarray(arrays[f"q_{name}"], dtype=np.float64),
                    x_ref=np.asarray(arrays[f"x_ref_{name}"], dtype=np.float64),
                    meta=dict(meta),
                )


def _regenerate_text(group: dict) -> str:
    regen = group.get("regenerate")
    if isinstance(regen, dict):
        extras = ", ".join(f"{k} {v}" for k, v in regen.items() if k != "command")
        return f"{regen.get('command')}" + (f"  ({extras})" if extras else "")
    return str(regen) if regen else "unknown"


def _group_npz_path(group: dict, corpus_dir: Path) -> Path:
    """The group's ``.npz`` path; for an external group, checked present and sha256-verified."""
    path = Path(corpus_dir) / group["npz"]
    if group.get("storage", "git") != "external":
        return path
    if not path.is_file():
        copies = group.get("copies") or [c.replace("<npz>", group["npz"]) for c in EXTERNAL_COPIES]
        raise ExternalCorpusMissing(
            f"corpus group {group['name']!r} is stored outside git and its file is missing: "
            f"{path}\n  sha256 {group.get('sha256')}  ({group.get('bytes')} bytes)\n"
            f"  copy it into {Path(corpus_dir)}/ from one of:\n    "
            + "\n    ".join(copies)
            + f"\n  or regenerate it: {_regenerate_text(group)}"
        )
    expected = group.get("sha256")
    if expected:
        actual = sha256_of(path)
        if actual != expected:
            raise CorpusHashMismatch(
                f"corpus group {group['name']!r}: {path} has sha256 {actual}, the manifest "
                f"records {expected}; fetch the recorded copy (see EXTERNAL_COPIES) or, after a "
                f"deliberate re-capture, let add_group rewrite the manifest entry."
            )
    return path


def load_corpus(groups=None, corpus_dir: Path = CORPUS_DIR) -> list[System]:
    """Every :class:`System` of the named groups (all groups when ``None``)."""
    return list(iter_systems(groups, corpus_dir=corpus_dir))


# ---------------------------------------------------------------------------
# Q encodings
# ---------------------------------------------------------------------------

ENCODINGS = ("dense", "sym_tri_xor")


def _encode_Q(Q: np.ndarray, name: str, encoding: str) -> dict:
    if encoding == "dense":
        return {f"Q_{name}": Q}
    if encoding != "sym_tri_xor":
        raise ValueError(f"unknown Q encoding {encoding!r}; known: {ENCODINGS}")
    iu = np.triu_indices(Q.shape[0])
    upper = np.ascontiguousarray(Q[iu])
    lower_t = np.ascontiguousarray(Q.T[iu])
    arrays = {
        f"Qtri_{name}": upper,
        f"Qxor_{name}": upper.view(np.uint64) ^ lower_t.view(np.uint64),
    }
    if not np.array_equal(
        _decode_Q(arrays, name, encoding, Q.shape[0]).view(np.uint64), Q.view(np.uint64)
    ):
        raise RuntimeError(f"system {name!r}: sym_tri_xor round trip is not bit-exact")
    return arrays


def _decode_Q(arrays, name: str, encoding: str, n: int) -> np.ndarray:
    if encoding == "dense":
        return np.asarray(arrays[f"Q_{name}"], dtype=np.float64)
    if encoding != "sym_tri_xor":
        raise ValueError(f"unknown Q encoding {encoding!r}; known: {ENCODINGS}")
    upper = np.ascontiguousarray(arrays[f"Qtri_{name}"], dtype=np.float64)
    xor = np.ascontiguousarray(arrays[f"Qxor_{name}"], dtype=np.uint64)
    iu = np.triu_indices(n)
    Q = np.empty((n, n), dtype=np.float64)
    Q[iu[1], iu[0]] = (upper.view(np.uint64) ^ xor).view(np.float64)
    Q[iu] = upper
    return Q


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


def _system_entry(
    group: str, spec: dict, source: dict, encoding: str = "dense"
) -> tuple[dict, dict]:
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
    arrays = {
        **_encode_Q(Q, spec["name"], encoding),
        f"q_{spec['name']}": q,
        f"x_ref_{spec['name']}": x_ref,
    }
    return entry, arrays


def add_group(
    name: str,
    systems: list[dict],
    source: dict,
    corpus_dir: Path = CORPUS_DIR,
    encoding: str = "dense",
    storage: str = "git",
    regenerate: dict | None = None,
) -> dict:
    """Add (or replace) corpus group ``name`` and return its manifest entry.

    ``systems`` is a list of dicts with at least ``name``, ``Q`` and ``q``; optional keys are
    ``model``, ``source_column_index_list``, ``no_regularization_index_list``,
    ``library_versions``, ``x_ref`` (else computed with :func:`reference_solution`) and any
    extra JSON-serialisable metadata, which is kept verbatim on the system's manifest entry.
    ``source`` is ``{"script", "args", "git_sha"}`` (plus extras) and is copied onto every
    system. System names must be unique within the group and match ``[A-Za-z0-9_]+``.
    ``encoding`` is how ``Q`` is stored (``"dense"``, or the lossless ``"sym_tri_xor"`` for large
    systems — see the module docstring). ``storage`` is ``"git"`` (the ``.npz`` is committed) or
    ``"external"`` (gitignored, kept at :data:`EXTERNAL_COPIES`); ``regenerate`` is
    ``{"command", ...}`` (default: ``python <source.script> <source.args>``). The group entry
    records ``encoding``, ``storage``, ``sha256``, ``bytes`` and ``regenerate``.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown Q encoding {encoding!r}; known: {ENCODINGS}")
    if storage not in STORAGES:
        raise ValueError(f"unknown corpus storage {storage!r}; known: {STORAGES}")
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
        entry, arr = _system_entry(name, spec, source, encoding)
        entries.append(entry)
        arrays.update(arr)
    npz_name = f"{name}.npz"
    npz_path = corpus_dir / npz_name
    np.savez_compressed(npz_path, **arrays)

    if regenerate is None:
        regenerate = {
            "command": " ".join(
                ["python", str(source.get("script") or "?"), *map(str, source.get("args") or [])]
            )
        }
    group = {
        "name": name,
        "npz": npz_name,
        "encoding": encoding,
        "storage": storage,
        "sha256": sha256_of(npz_path),
        "bytes": npz_path.stat().st_size,
        "regenerate": dict(regenerate),
    }
    if storage == "external":
        group["copies"] = [c.replace("<npz>", npz_name) for c in EXTERNAL_COPIES]
    group.update({"source": dict(source), "systems": entries})
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
        "script": "scripts/imaging/mge/hazards_nnls_capture.py",
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
    regenerate = {
        "command": "python scripts/lens/solver/_corpus.py",
        "input": source["fixture"],
        "note": "verbatim copy of the PyAutoArray#571 capture fixture",
    }
    return add_group(
        FIXTURE_571_GROUP, systems, source, corpus_dir=corpus_dir, regenerate=regenerate
    )


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
