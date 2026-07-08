"""Unit tests for ``vram.config`` batch-size lookup + resolution.

Covers the sparse-path fallback rule and the probe-JSON-over-table
resolution order added for the PreOptimizationTimes campaign (phase 2,
autolens_profiling#54). No JAX dependency.

Run::

    cd autolens_profiling
    python -m pytest test/test_vram_config.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `vram` importable without installing the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vram.config import (  # noqa: E402
    VMAP_BATCH,
    VMAP_BATCH_SPARSE,
    resolve_vmap_batch,
    vmap_batch_for,
)


def _write_probe(
    tmp_path: Path,
    *,
    model: str = "pixelization",
    sparse: bool = False,
    dataset: str = "imaging",
    instrument: str = "hst",
    recommended=24,
) -> Path:
    suffix = "_sparse" if sparse else ""
    p = tmp_path / f"vmap_probe_{model}{suffix}.json"
    p.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "model": model,
                "instrument": instrument,
                "recommended_batch_size": recommended,
            }
        )
    )
    return p


# ---------------------------------------------------------------------------
# vmap_batch_for — path axis
# ---------------------------------------------------------------------------


def test_dense_lookup_unchanged():
    assert (
        vmap_batch_for("imaging", "pixelization", "hst")
        == VMAP_BATCH[("imaging", "pixelization", "hst")]
    )


def test_sparse_falls_back_to_dense_when_unprobed():
    key = ("imaging", "pixelization", "hst")
    assert key not in VMAP_BATCH_SPARSE  # precondition: sparse row unprobed
    assert vmap_batch_for(*key, path="sparse") == VMAP_BATCH[key]


def test_sparse_row_wins_when_present(monkeypatch):
    key = ("imaging", "pixelization", "hst")
    monkeypatch.setitem(VMAP_BATCH_SPARSE, key, 32)
    assert vmap_batch_for(*key, path="sparse") == 32
    assert vmap_batch_for(*key) == VMAP_BATCH[key]  # dense unaffected


def test_intentional_none_preserved():
    assert vmap_batch_for("datacube", "delaunay", "sma") is None
    assert vmap_batch_for("interferometer", "mge", "alma") is None


# ---------------------------------------------------------------------------
# resolve_vmap_batch — probe JSON over table
# ---------------------------------------------------------------------------


def test_resolve_without_output_dir_uses_table():
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst")
    assert batch == VMAP_BATCH[("imaging", "pixelization", "hst")]
    assert source == "table"


def test_resolve_prefers_matching_probe(tmp_path):
    _write_probe(tmp_path, recommended=24)
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst", output_dir=tmp_path)
    assert batch == 24
    assert source.startswith("probe")


def test_resolve_sparse_probe_selected_by_path(tmp_path):
    _write_probe(tmp_path, recommended=24)  # dense probe
    _write_probe(tmp_path, sparse=True, recommended=40)
    batch, source = resolve_vmap_batch(
        "imaging", "pixelization", "hst", output_dir=tmp_path, path="sparse"
    )
    assert batch == 40
    assert "sparse" in source


def test_resolve_ignores_cell_mismatch(tmp_path):
    _write_probe(tmp_path, instrument="euclid")  # probe for a different instrument
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst", output_dir=tmp_path)
    assert batch == VMAP_BATCH[("imaging", "pixelization", "hst")]
    assert "mismatch" in source


def test_resolve_ignores_invalid_recommendation(tmp_path):
    _write_probe(tmp_path, recommended=None)
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst", output_dir=tmp_path)
    assert batch == VMAP_BATCH[("imaging", "pixelization", "hst")]
    assert "no valid recommendation" in source


def test_resolve_survives_corrupt_json(tmp_path):
    (tmp_path / "vmap_probe_pixelization.json").write_text("{not json")
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst", output_dir=tmp_path)
    assert batch == VMAP_BATCH[("imaging", "pixelization", "hst")]
    assert "unreadable" in source


def test_resolve_missing_probe_uses_table(tmp_path):
    batch, source = resolve_vmap_batch("imaging", "pixelization", "hst", output_dir=tmp_path)
    assert batch == VMAP_BATCH[("imaging", "pixelization", "hst")]
    assert source == "table"
