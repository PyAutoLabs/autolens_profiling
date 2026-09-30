"""Shared driver for the linear-solver study cells (``accuracy.py``, ``early_stopping.py``).

The cells are thin: prologue, :func:`select_device_from_argv` (which must run before JAX is
imported), the smoke short-circuit, then one call into this module. Everything shared lives
here — the CLI, the per-(system, candidate) evaluation, the artefact naming and the JSON write —
so the two cells cannot drift apart in method.

Artefacts: ``results/lens/solver/<cell>_summary_<corpus>[_gpu]_v<version>.{json,png}``, where
``<corpus>`` is the single group's name when one group ran, else ``all``. The JSON carries the
``device`` block from ``_profile_cli.device_info_dict()`` (with its mandatory ``provenance``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
RESULTS_DIR = REPO_ROOT / "results" / "lens" / "solver"
#: Floor for log-scale plotting of exact zeros (e.g. the fnnls self-check row).
PLOT_FLOOR = 1.0e-17


def select_device_from_argv(argv=None) -> str:
    """Read ``--device cpu|gpu`` before JAX is imported and pin the JAX platform to it."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    platform = "cuda" if known.device == "gpu" else "cpu"
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORM_NAME"] = platform
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    return known.device


def parse_cli(description: str, default_candidates: list[str], argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument(
        "--groups", nargs="+", default=None, help="Corpus groups to run (default: all)."
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help=f"Candidate names from _solvers.CANDIDATES (default: {' '.join(default_candidates)}).",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5, help="Timed warm calls per row.")
    parser.add_argument(
        "--target-kappa",
        type=float,
        default=1.0e-11,
        help="Relaxed-KKT target (library general.inversion.nnls_target_kappa).",
    )
    args = parser.parse_args(argv)
    if args.candidates is None:
        args.candidates = list(default_candidates)
    return args


def corpus_label(groups, all_groups) -> str:
    groups = list(all_groups) if groups is None else list(groups)
    return groups[0] if len(groups) == 1 else "all"


def output_paths(cell: str, label: str, version: str, device: str, output_dir=None):
    out_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_gpu" if device == "gpu" else ""
    base = f"{cell}_summary_{label}{suffix}_v{version}"
    return out_dir / f"{base}.json", out_dir / f"{base}.png"


def evaluate(candidate, system, *, repeats: int, target_kappa: float) -> dict:
    """One (system, candidate) row: the solve, its metrics, its status and its warm wall."""
    import _metrics
    import numpy as np

    x, stats = candidate.fn(system.Q, system.q, system.meta, target_kappa=target_kappa)
    row = {
        "system": f"{system.group}/{system.name}",
        "group": system.group,
        "candidate": candidate.name,
        **_metrics.metrics(x, system),
        "converged": stats.get("converged"),
        "iterations": stats.get("iterations"),
        "stats": {k: v for k, v in stats.items() if k not in ("converged", "iterations")},
    }
    if candidate.backend == "jax":
        import jax.numpy as jnp

        Q, q = jnp.asarray(system.Q), jnp.asarray(system.q)
    else:
        Q, q = np.asarray(system.Q), np.asarray(system.q)
    row["wall_ms"] = 1.0e3 * _metrics.timing(
        candidate.kernel, Q, q, repeats=repeats, target_kappa=target_kappa
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    """Aggregate a set of rows (one candidate, or one cap) into the table statistics."""
    import _metrics

    return {
        "n_systems": len(rows),
        "n_unconverged": sum(1 for r in rows if r["converged"] is False),
        "n_nonfinite": sum(1 for r in rows if not r["finite"]),
        "worst_amp_rel_max": _metrics.worst_abs(r["amp_rel_max"] for r in rows),
        "median_amp_rel_max": _metrics.median_or_none(r["amp_rel_max"] for r in rows),
        "worst_amp_rel_max_sig": _metrics.worst_abs(r["amp_rel_max_sig"] for r in rows),
        "worst_amp_rel_l2": _metrics.worst_abs(r["amp_rel_l2"] for r in rows),
        "worst_flux_rel_all": _metrics.worst_abs(r["flux_rel_all"] for r in rows),
        "worst_flux_rel_source": _metrics.worst_abs(r["flux_rel_source"] for r in rows),
        "worst_objective_gap": _metrics.worst_abs(r["objective_gap"] for r in rows),
        "worst_kkt_residual_scaled": _metrics.worst_abs(r["kkt_residual_scaled"] for r in rows),
        "median_iterations": _metrics.median_or_none(r["iterations"] for r in rows),
        "max_iterations": max(
            (r["iterations"] for r in rows if r["iterations"] is not None), default=None
        ),
        "median_wall_ms": _metrics.median_or_none(r["wall_ms"] for r in rows),
    }


def summary_header(cell: str, args, systems, label: str) -> dict:
    import _corpus
    import autolens as al
    import jax

    from _profile_cli import device_info_dict

    return {
        "autolens_version": al.__version__,
        "device": device_info_dict(),
        "cell": cell,
        "corpus": {
            "label": label,
            "groups": sorted({s.group for s in systems}),
            "systems": [f"{s.group}/{s.name}" for s in systems],
            "n_systems": len(systems),
            "manifest": str((_corpus.CORPUS_DIR / _corpus.MANIFEST_NAME).relative_to(REPO_ROOT)),
        },
        "configuration": {
            "candidates": list(args.candidates),
            "target_kappa": args.target_kappa,
            "repeats": args.repeats,
            "requested_device": args.device,
            "x64": bool(jax.config.jax_enable_x64),
            "reference": "autoarray.util.fnnls.fnnls_cholesky (dense-sign start), stored in the corpus",
        },
    }


def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=1, default=_json_default, allow_nan=False) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")


def log_floor(v):
    """``v`` for a log axis: ``None`` stays ``None``, zeros/negatives go to :data:`PLOT_FLOOR`."""
    if v is None:
        return None
    return max(abs(float(v)), PLOT_FLOOR)
