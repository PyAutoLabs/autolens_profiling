"""Unit tests for the reduced N_src sweep aggregator (``size_sweep_table.py``).

Synthetic manifest + minimal result JSONs in a tmp dir. No JAX, no real result
files: the point is the **classification** — a leg that fits, a leg that died
out of memory, a leg still queued, and a leg whose log exists but says nothing —
and the crossover sentence computed from them.

The module is loaded straight off its path rather than as
``likelihood_breakdown.size_sweep_table``: that package's ``__init__`` imports
``timing``, which imports JAX, and this file has no business paying a JAX import
to parse JSON.

Run::

    cd autolens_profiling
    python -m pytest scripts/misc/test/test_size_sweep_table.py
"""

from __future__ import annotations

import importlib.util
import json
import sys as _sys
from pathlib import Path as _Path

import pytest


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


ROOT = _profiling_root()
_MODULE_PATH = ROOT / "scripts" / "misc" / "likelihood_breakdown" / "size_sweep_table.py"
_spec = importlib.util.spec_from_file_location("size_sweep_table_under_test", _MODULE_PATH)
sst = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = sst
_spec.loader.exec_module(sst)


# ---------------------------------------------------------------------------
# fixtures — the smallest JSONs carrying the keys the aggregator reads
# ---------------------------------------------------------------------------


def _exact_json(*, total_s: float, solve_s: float, logdet_s: float, source_pixels: int) -> dict:
    return {
        "autolens_version": "2026.8.17.1",
        "device": {"hostname": "euclid-ral-gpu-2"},
        "instrument": "hst",
        "configuration": {"source_pixels": source_pixels, "inversion_path": "dense"},
        "total_step_by_step": total_s,
        "steps_reconstruction_sub_rows": {
            "Cholesky solve (unconstrained)": solve_s,
            "Log det Cholesky (F+λH reduced)": logdet_s,
            "Log det Cholesky (H reduced)": logdet_s,
            "NNLS PDIP @vmap 16 (identical lanes)": 0.02,
        },
    }


def _matrix_free_json(
    *, pcg_ms: float, slq_ms: float, source_pixels: int, log_ev_err: float, peak_bytes: int
) -> dict:
    return {
        "autolens_version": "2026.8.17.1",
        "device": {"hostname": "euclid-ral-gpu-2"},
        "instrument": "hst",
        "configuration": {"source_pixels": source_pixels, "inversion_path": "matrix_free"},
        "steps_matrix_free_rows": [
            {"name": sst.MF_SOLVE_ROW, "ms": pcg_ms, "cg_iterations": 3967},
            {"name": sst.MF_SLQ_CURV_ROW, "ms": slq_ms},
            {"name": sst.MF_SLQ_REG_ROW, "ms": slq_ms},
            {"name": "PCG solve @vmap 16 (identical lanes)", "ms": pcg_ms * 0.6},
        ],
        "matrix_free": {
            "cond_curvature_reg": 4.2e10,
            "evidence": {"log_evidence_error": log_ev_err},
            "peak_bytes": {"after_matrix_free": peak_bytes},
        },
    }


@pytest.fixture
def sweep(tmp_path):
    """A tmp sweep: results/, a manifest, and SLURM logs for the dead legs."""
    results = tmp_path / "results" / "breakdown" / "imaging"
    results.mkdir(parents=True)
    logs = tmp_path / "hpc" / "batch_gpu"
    (logs / "output").mkdir(parents=True)
    (logs / "error").mkdir(parents=True)

    # n=1500 fiducial, rectangular: the recon-split dense leg is the exact
    # comparator the crossover is measured against.
    (results / "pixelization_hpc_a100_fp64_recon_split.json").write_text(
        json.dumps(_exact_json(total_s=0.060, solve_s=0.0016, logdet_s=0.0012, source_pixels=1521))
    )
    (results / "pixelization_hpc_a100_fp64_sparse.json").write_text(
        json.dumps(_exact_json(total_s=0.071, solve_s=0.0016, logdet_s=0.0012, source_pixels=1521))
    )
    # n=3000: dense fits, sparse OOMs, matrix-free fits and wins.
    (results / "pixelization_hpc_a100_fp64_n3000.json").write_text(
        json.dumps(_exact_json(total_s=0.090, solve_s=0.0020, logdet_s=0.0015, source_pixels=3025))
    )
    (results / "matrix_free_rectangular_hpc_a100_fp64_matrix_free_n3000.json").write_text(
        json.dumps(
            _matrix_free_json(
                pcg_ms=0.500,
                slq_ms=0.250,
                source_pixels=3025,
                log_ev_err=-112.5,
                peak_bytes=2 * 1024**3,
            )
        )
    )
    # n=5000: matrix-free fits but loses to the exact path at the same N.
    (results / "pixelization_hpc_a100_fp64_n5000.json").write_text(
        json.dumps(_exact_json(total_s=0.120, solve_s=0.0025, logdet_s=0.0018, source_pixels=5041))
    )
    (results / "matrix_free_rectangular_hpc_a100_fp64_matrix_free_n5000.json").write_text(
        json.dumps(
            _matrix_free_json(
                pcg_ms=40.0,
                slq_ms=10.0,
                source_pixels=5041,
                log_ev_err=0.25,
                peak_bytes=3 * 1024**3,
            )
        )
    )

    # The dead legs. OOM at n=3000 sparse; a silent log at n=12000 dense; a
    # queued leg (manifest line, no log at all) at n=12000 matrix-free.
    (logs / "error" / "error.900001.err").write_text(
        "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory allocating "
        "12884901888 bytes.\n"
    )
    (logs / "output" / "output.900001.out").write_text(
        "SWEEP_LEG mesh=rectangular path=sparse n=3000 job=900001\nSWEEP_EXIT code=1\n"
    )
    (logs / "output" / "output.900002.out").write_text(
        "SWEEP_LEG mesh=rectangular path=dense n=12000 job=900002\nnothing to see here\n"
    )

    manifest_path = tmp_path / "results" / "sweep" / "size_sweep_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    lines = [
        {"job": "900000", "mesh": "rectangular", "path": "matrix_free", "n": 3000},
        {"job": "900001", "mesh": "rectangular", "path": "sparse", "n": 3000},
        {"job": "900003", "mesh": "rectangular", "path": "dense", "n": 3000},
        {"job": "900004", "mesh": "rectangular", "path": "matrix_free", "n": 5000},
        {"job": "900005", "mesh": "rectangular", "path": "dense", "n": 5000},
        {"job": "900002", "mesh": "rectangular", "path": "dense", "n": 12000},
        {"job": "900006", "mesh": "rectangular", "path": "matrix_free", "n": 12000},
    ]
    manifest_path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    return {"results": results, "manifest": manifest_path, "logs": logs}


def _legs(sweep):
    legs, _ = sst.collect(sweep["results"], sweep["manifest"], sweep["logs"])
    return legs


# ---------------------------------------------------------------------------
# filename resolution
# ---------------------------------------------------------------------------


def test__basenames_match_resolve_output_paths():
    """`_sparse` is appended AFTER the config name, so the label keeps `_n<N>`."""
    assert sst.json_candidates("rectangular", "sparse", 3000) == [
        "pixelization_hpc_a100_fp64_n3000_sparse"
    ]
    assert sst.json_candidates("delaunay_nn", "dense", 5000) == ["delaunay_nn_hpc_a100_fp64_n5000"]
    # The matrix-free cell names itself `matrix_free_<mesh>`, so the mesh is in
    # the basename as well as the flag.
    assert sst.json_candidates("delaunay", "matrix_free", 12000) == [
        "matrix_free_delaunay_hpc_a100_fp64_matrix_free_n12000"
    ]
    # The fiducial dense row prefers the recon-split leg: the plain one has no
    # exact comparators.
    assert sst.json_candidates("rectangular", "dense", 1500)[0].endswith("_recon_split")


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test__status_of_every_leg(sweep):
    legs = _legs(sweep)
    assert legs[("rectangular", "dense", 3000)].status == "ok"
    assert legs[("rectangular", "matrix_free", 3000)].status == "ok"
    # Submitted, no JSON, RESOURCE_EXHAUSTED in the .err -> the memory datum.
    assert legs[("rectangular", "sparse", 3000)].status == "oom"
    # Submitted, a log that says nothing and no JSON.
    assert legs[("rectangular", "dense", 12000)].status == "missing"
    # Submitted, no log written yet.
    assert legs[("rectangular", "matrix_free", 12000)].status == "pending"
    # Never submitted and never run: the fiducial matrix-free legs.
    assert legs[("delaunay", "matrix_free", 1500)].status == "pending"


def test__classify_logs_orders_oom_before_timeout_and_traceback(tmp_path):
    (tmp_path / "error").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "error" / "error.5.err").write_text(
        "Traceback (most recent call last):\nRESOURCE_EXHAUSTED\nDUE TO TIME LIMIT\n"
    )
    assert sst.classify_logs(tmp_path, "5") == "oom"
    (tmp_path / "error" / "error.6.err").write_text("slurmstepd: CANCELLED DUE TO TIME LIMIT\n")
    assert sst.classify_logs(tmp_path, "6") == "timeout"
    (tmp_path / "error" / "error.7.err").write_text("Traceback (most recent call last):\nboom\n")
    assert sst.classify_logs(tmp_path, "7") == "failed"
    assert sst.classify_logs(tmp_path, "8") == "pending"
    assert sst.classify_logs(tmp_path, None) == "pending"


def test__logs_alone_identify_a_leg_without_a_manifest(sweep):
    """The SWEEP_LEG header is what makes a `.out` readable with no manifest."""
    found = sst.scan_logs(sweep["logs"])
    assert found[("rectangular", "sparse", 3000)] == "900001"
    legs, manifest = sst.collect(
        sweep["results"], sweep["manifest"].parent / "absent.jsonl", sweep["logs"]
    )
    assert manifest == []
    assert legs[("rectangular", "sparse", 3000)].status == "oom"


# ---------------------------------------------------------------------------
# the rendered tables
# ---------------------------------------------------------------------------


def test__fits_table_cells(sweep):
    markdown = sst.render(sweep["results"], sweep["manifest"], sweep["logs"])
    fits = markdown.split("## 2.")[0]
    assert "| rectangular | 3000 | 3025 | ✓ | OOM | ✓ 2.0 GB |" in fits
    assert "| rectangular | 12000 | — | missing | — | pending |" in fits
    # The dense/sparse cells record no peak memory, so their tick is bare.
    assert "| rectangular | 1500 | 1521 | ✓ | ✓ | pending |" in fits


def test__per_call_table_reads_the_named_rows(sweep):
    markdown = sst.render(sweep["results"], sweep["manifest"], sweep["logs"])
    per_call = markdown.split("## 2.")[1].split("## 3.")[0]
    # matrix-free: 0.500 + 0.250 + 0.250 = 1.000 ms, with its cg iterations,
    # condition number and evidence error alongside.
    assert "| rectangular | 3000 | matrix_free | 1.000 |" in per_call
    assert "3967" in per_call
    assert "4.2e+10" in per_call
    assert "-112.500" in per_call
    # exact: the whole-likelihood total, and the components the crossover uses.
    assert "| rectangular | 3000 | dense | 90.000 |" in per_call
    assert "chol solve 2.000 · logdet(F+λH) 1.500 · logdet(H) 1.500" in per_call


def test__crossover_is_computed_not_interpolated(sweep):
    markdown = sst.render(sweep["results"], sweep["manifest"], sweep["logs"])
    crossover = markdown.split("## 3.")[1]
    # 1.000 ms vs 2.000 + 1.500 + 1.500 = 5.000 ms at N=3000.
    assert "N = 3000: matrix-free 1.000 ms beats the dense exact 5.000 ms" in crossover
    # 60.000 ms vs 2.500 + 1.800 + 1.800 = 6.100 ms at N=5000.
    assert "N = 5000: matrix-free 60.000 ms loses to the dense exact 6.100 ms" in crossover
    assert "First measured N where matrix-free wins: 3000." in crossover
    # The memory verdict, and the 0.5-nat bar at each N.
    assert "Sparse stops fitting at N = 3000." in crossover
    assert "-112.500 nats (outside the 0.5-nat bar)" in crossover
    assert "+0.250 nats (within the 0.5-nat bar)" in crossover
    # Nothing is filled in between measured points.
    assert "Dense fits at every measured N up to 5000" in crossover
    assert sst.NOT_MEASURED in crossover


def test__a_mesh_with_no_matrix_free_row_says_so(sweep):
    markdown = sst.render(sweep["results"], sweep["manifest"], sweep["logs"])
    delaunay = markdown.split("**delaunay.**")[1].split("**delaunay_nn.**")[0]
    assert f"Matrix-free per-call vs exact: {sst.NOT_MEASURED}." in delaunay
    assert f"SLQ log-evidence error: {sst.NOT_MEASURED}." in delaunay


def test__main_writes_the_markdown(sweep, tmp_path, capsys):
    out = tmp_path / "sweep.md"
    rc = sst.main(
        [
            "--results-dir",
            str(sweep["results"]),
            "--manifest",
            str(sweep["manifest"]),
            "--logs-dir",
            str(sweep["logs"]),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    written = out.read_text()
    assert written == capsys.readouterr().out
    assert written.startswith("# Reduced N_src size sweep")
