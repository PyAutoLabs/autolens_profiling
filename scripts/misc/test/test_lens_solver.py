"""``scripts/lens/solver/``: the corpus round-trip and the accuracy metrics on a tiny system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SOLVER_DIR = Path(__file__).resolve().parents[2] / "lens" / "solver"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

import _corpus  # noqa: E402
import _metrics  # noqa: E402


def _tiny_system():
    """A 3x3 SPD system whose NNLS solution has one active bound (x_2 = 0)."""
    Q = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
    q = np.array([2.0, 1.0, -1.0])
    return Q, q


def test_corpus_round_trip(tmp_path):
    Q, q = _tiny_system()
    source = {"script": "tests", "args": ["--x"], "git_sha": None}
    _corpus.add_group(
        "tiny",
        [
            {
                "name": "a",
                "Q": Q,
                "q": q,
                "model": "hand-built",
                "source_column_index_list": [1, 2],
                "no_regularization_index_list": None,
                "category": "unit",
            },
            {"name": "b", "Q": 2.0 * Q, "q": q},
        ],
        source,
        corpus_dir=tmp_path,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [g["name"] for g in manifest["groups"]] == ["tiny"]

    systems = _corpus.load_corpus(corpus_dir=tmp_path)
    assert [s.name for s in systems] == ["a", "b"]
    a = systems[0]
    np.testing.assert_array_equal(a.Q, Q)
    np.testing.assert_array_equal(a.q, q)
    assert a.meta["source"] == source
    assert a.meta["n"] == 3 and a.meta["n_source_columns"] == 2
    assert a.meta["no_regularization_index_list"] is None
    assert a.meta["category"] == "unit"
    assert a.meta["cond_Q"] == pytest.approx(np.linalg.cond(Q))
    assert a.meta["max_abs_q"] == 2.0
    # The stored reference is the fnnls solution: x_2 at the bound, KKT satisfied.
    assert a.x_ref[2] == 0.0
    grad = Q @ a.x_ref - q
    np.testing.assert_allclose(grad[:2], 0.0, atol=1e-12)
    assert grad[2] >= 0.0

    # Re-adding a group replaces it; a second group appends; group filtering works.
    _corpus.add_group("tiny", [{"name": "c", "Q": Q, "q": q}], source, corpus_dir=tmp_path)
    _corpus.add_group("other", [{"name": "d", "Q": Q, "q": q}], source, corpus_dir=tmp_path)
    assert _corpus.group_names(tmp_path) == ["tiny", "other"]
    assert [s.name for s in _corpus.load_corpus(["other"], corpus_dir=tmp_path)] == ["d"]
    with pytest.raises(KeyError):
        _corpus.load_corpus(["missing"], corpus_dir=tmp_path)


def test_corpus_external_storage(tmp_path):
    """An external group records sha256/bytes, fails loudly when absent, and is hash-checked."""
    Q, q = _tiny_system()
    source = {"script": "tests", "args": ["--x"], "git_sha": None}
    entry = _corpus.add_group(
        "ext",
        [{"name": "a", "Q": Q, "q": q}],
        source,
        corpus_dir=tmp_path,
        storage="external",
        regenerate={"command": "python capture.py --source ext", "library_tag": "T"},
    )
    npz = tmp_path / "ext.npz"
    assert entry["storage"] == "external" and entry["encoding"] == "dense"
    assert entry["sha256"] == _corpus.sha256_of(npz) and entry["bytes"] == npz.stat().st_size
    assert entry["copies"] and all("ext.npz" in c for c in entry["copies"])
    assert [s.name for s in _corpus.load_corpus(corpus_dir=tmp_path)] == ["a"]

    data = npz.read_bytes()
    npz.unlink()
    with pytest.raises(_corpus.ExternalCorpusMissing) as missing:
        _corpus.load_corpus(corpus_dir=tmp_path)
    message = str(missing.value)
    assert "ext.npz" in message and entry["sha256"] in message
    assert "python capture.py --source ext" in message and "/mnt/ral/" in message

    npz.write_bytes(data + b"\0")
    with pytest.raises(_corpus.CorpusHashMismatch):
        _corpus.load_corpus(corpus_dir=tmp_path)

    # A git group is never hash-checked on read (git tracks it); defaults are recorded.
    git_entry = _corpus.add_group("in_git", [{"name": "b", "Q": Q, "q": q}], source, tmp_path)
    assert git_entry["storage"] == "git"
    assert git_entry["regenerate"] == {"command": "python tests --x"}
    with pytest.raises(ValueError):
        _corpus.add_group("bad", [{"name": "c", "Q": Q, "q": q}], source, tmp_path, storage="s3")


def test_corpus_rejects_bad_names(tmp_path):
    Q, q = _tiny_system()
    with pytest.raises(ValueError):
        _corpus.add_group("g", [{"name": "a-b", "Q": Q, "q": q}], {}, corpus_dir=tmp_path)
    with pytest.raises(ValueError):
        _corpus.add_group(
            "g", [{"name": "a", "Q": Q, "q": q}, {"name": "a", "Q": Q, "q": q}], {}, tmp_path
        )


def test_metrics_exact_and_perturbed():
    Q, q = _tiny_system()
    x_ref = np.linalg.solve(Q[:2, :2], q[:2])
    x_ref = np.array([x_ref[0], x_ref[1], 0.0])

    class S:
        pass

    system = S()
    system.Q, system.q, system.x_ref = Q, q, x_ref
    system.meta = {"source_column_index_list": [1, 2]}

    exact = _metrics.metrics(x_ref.copy(), system)
    assert exact["finite"] is True
    for key in ("amp_rel_l2", "amp_rel_max", "amp_rel_max_sig", "flux_rel_all"):
        assert exact[key] == 0.0
    assert exact["flux_rel_source"] == 0.0
    assert exact["objective_gap"] == 0.0
    assert exact["kkt_residual_scaled"] < 1e-14

    x = x_ref * np.array([1.1, 1.0, 1.0]) + np.array([0.0, 0.0, 0.01])
    m = _metrics.metrics(x, system)
    assert m["amp_rel_max"] == pytest.approx(0.1)
    assert m["flux_rel_all"] == pytest.approx((0.1 * x_ref[0] + 0.01) / x_ref.sum())
    assert m["flux_rel_source"] == pytest.approx(0.01 / x_ref[1])
    assert m["objective_gap"] > 0.0
    assert m["kkt_residual_scaled"] > 0.0

    no_cols = S()
    no_cols.Q, no_cols.q, no_cols.x_ref, no_cols.meta = Q, q, x_ref, {}
    assert _metrics.metrics(x, no_cols)["flux_rel_source"] is None

    bad = _metrics.metrics(np.array([np.nan, 0.0, 0.0]), system)
    assert bad["finite"] is False and bad["amp_rel_max"] is None


def test_timing_returns_median_seconds():
    calls = []

    def fn(Q, q):
        calls.append(1)

    wall = _metrics.timing(fn, None, None, repeats=3)
    assert wall >= 0.0 and len(calls) == 4
