"""
Matrix-free experiment: NumPy CPU rectangular interferometer runtime (numpy_cpu_matrixfree)
===========================================================================================

Likelihood-lifecycle 2b (autolens_profiling#469), human 2026-10-26: the per-call runtime of the
matrix-free interferometer inversion (``aa.InversionInterferometerMatrixFree``, PyAutoArray#658)
beside the dense W~ route it replaces, with the log-evidence agreement between the two. An
**experiment**: its rows go under ``experiments/numpy_cpu_matrixfree/`` (35 x 35) or
``scans/numpy_cpu_matrixfree_<N>x<N>/`` (the alma mesh scan) of the NumPy-dispatch CPU W~ setup
``numba_cpu_sparse_fp64_wtilde@hpc_ral_cpu``, never a current row, and no ``parity.json``.

Library
-------

The route merged on PyAutoArray main after the 2026.10.25.1 pin, so the campaign runs it from a
private checkout at exact main SHAs (labelled ``unreleased main``; the row's provenance carries
the SHAs). On a library without it the script refuses.

Model
-----

- Isothermal + ExternalShear with the interferometer standard cell's priors
  (``_production_config.interferometer_production_model``), no lens light;
- source ``al.Pixelization(mesh=al.mesh.RectangularUniform(shape=(N, N)),
  regularization=al.reg.MaternSparse(coefficient=1.0, scale=1.0, nu=1.0))`` — the only mesh and
  regularization the matrix-free route supports; the Matérn values are the library defaults, a
  declared choice (no ruling names them), recorded in the row;
- ``al.Settings(interferometer_route="matrix_free", use_edge_zeroed_pixels=False,
  use_positive_only_solver=True)`` plus the cells' border relocator and NNLS memo off
  (``_production_config.analysis_settings``); the dataset carries the NumPy W~ sparse operator
  (``preload_sparse_dataset``, cached beside the dataset).

Arms
----

- ``matrix_free`` (headline): the seeded iid stream (seed 235, the central 20 % of every prior),
  instance 0 a discarded warm-up, then ``--cold-evals`` cold and the rest warm;
  ``AnalysisInterferometer(use_jax=False).log_likelihood_function``; threads pinned to 1.
- ``dense_fft`` (the reference): the same model and instances through
  ``interferometer_route="fft"`` (``InversionInterferometerSparse``: dense F and H, exact Cholesky
  log determinants), timed over ``--fft-repeats`` warm calls. A failure (e.g. memory at 192 x 192)
  is recorded, never fatal.
- ``chunked_dense`` (sma, or ``--chunked-reference``): the W~-independent visibility-chunked
  NumPy dense reference (``_production_config.interferometer_dense_reference``) at instance 0.

The agreement block records both log evidences at instance 0, their absolute and relative
difference, and the stochastic Lanczos ``ln det(F + H)`` against the exact one, plus the
interior-point diagnostics (``inversion.positive_result``).

Output
------

The JSON named by ``--output-path`` (the campaign submit always passes one, under
``output/p2b_matrixfree/``), else ``output/p2b_matrixfree_local/results/...``; never
``results/`` in place. ``fit.png`` is drawn beside it.
"""

import sys as _sys
from pathlib import Path as _Path


def _profiling_root() -> _Path:
    for _p in _Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc_dir = str(_profiling_root() / "scripts" / "misc")
if _misc_dir not in _sys.path:
    _sys.path.insert(0, _misc_dir)
_sys.path.insert(0, str(_profiling_root()))

# Threads are pinned BEFORE numpy is imported (OpenBLAS / MKL read their thread counts once).
import argparse  # noqa: E402

from _production_config import pin_thread_env as _pin_thread_env  # noqa: E402
from _profile_cli import parse_profile_cli  # noqa: E402

_cli = parse_profile_cli()
_thread_env = _pin_thread_env(1)
instrument = _cli.instrument or "sma"

#: The experiment's folder names (``experiments/`` at 35 x 35, ``scans/`` otherwise).
EXPERIMENT = "numpy_cpu_matrixfree"
MESH_SIDE_DEFAULT = 35
#: MaternSparse at the library defaults (declared; no ruling names the values).
MATERN = {"coefficient": 1.0, "scale": 1.0, "nu": 1.0}
#: Above this many visibilities the chunked dense reference runs only on request.
CHUNKED_REFERENCE_MAX_VIS = 100_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="matrix-free experiment", allow_abbrev=False)
    parser.add_argument("--mesh-side", type=int, default=MESH_SIDE_DEFAULT)
    parser.add_argument(
        "--fft-repeats",
        type=int,
        default=3,
        help="warm calls of the dense fft route timed after one warm-up (0 = reference only)",
    )
    parser.add_argument(
        "--chunked-reference",
        action="store_true",
        help="also compute the chunked NUFFT dense reference above sma (costly)",
    )
    return parser


_args, _unknown = _parser().parse_known_args()

import os as _smoke_os  # noqa: E402

if _smoke_os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    _sys.exit(0)

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import autoarray as aa  # noqa: E402
import autolens as al  # noqa: E402
import numpy as np  # noqa: E402

if not hasattr(aa, "InversionInterferometerMatrixFree"):
    raise SystemExit(
        "REFUSING: this autoarray has no InversionInterferometerMatrixFree (PyAutoArray#658). Run "
        "from the private main checkout the matrix-free submit names, not the 2026.10.25.1 pin."
    )
try:
    al.Settings(interferometer_route="matrix_free")
except Exception as _exc:  # noqa: BLE001
    raise SystemExit(
        f'REFUSING: Settings(interferometer_route="matrix_free") is not accepted here: {_exc}'
    ) from None

from autoarray.inversion.inversion.interferometer import (  # noqa: E402
    inversion_interferometer_matrix_free_util as matrix_free_util,
)
from likelihood_breakdown import interferometer_pixelized as shared  # noqa: E402
from likelihood_breakdown import timing as _timing  # noqa: E402
from likelihood_breakdown.interferometer_pixelized_numpy import (  # noqa: E402
    preload_sparse_dataset,
)

from _production_config import (  # noqa: E402
    analysis_settings,
    iid_instances,
    interferometer_dense_reference,
    interferometer_production_model,
    interferometer_standard_preset,
    timing_summary,
)
from _profile_cli import (  # noqa: E402
    device_info_dict,
    render_fit_png,
    resources_dict,
)


def _error(exc) -> dict:
    text = str(exc)
    oom = isinstance(exc, MemoryError) or "out of memory" in text.lower()
    return {
        "status": "oom" if oom else "error",
        "error": f"{type(exc).__name__}: {text.splitlines()[0][:500] if text else ''}",
        "traceback": traceback.format_exc()[-2000:],
    }


def _host_class(config_name) -> str:
    if config_name and config_name.startswith("hpc_ral_cpu"):
        return "hpc_ral_cpu"
    if config_name and config_name.startswith("hpc_a100"):
        return "hpc_a100"
    return "local_laptop"


def output_path(cli, instrument: str, side: int, root: _Path) -> _Path:
    """``--output-path``, else a scratch path under ``output/`` (never ``results/`` in place)."""
    if cli.output_path is not None:
        return _Path(cli.output_path)
    location = (
        f"experiments/{EXPERIMENT}"
        if side == MESH_SIDE_DEFAULT
        else f"scans/{EXPERIMENT}_{side}x{side}"
    )
    return (
        root
        / "output"
        / "p2b_matrixfree_local"
        / "results"
        / "interferometer"
        / "rectangular"
        / instrument
        / "numba_cpu_sparse_fp64_wtilde"
        / _host_class(cli.config_name)
        / location
        / "runtime.json"
    )


root = _profiling_root()
side = int(_args.mesh_side)
timer = _timing.Timer()

# --------------------------------------------------------------------------- dataset + model
ds = shared.load_dataset(instrument, None, root, timer)
dataset, preload = preload_sparse_dataset(ds, _cli.sparse_batch_size, True, timer)
print(f"  W~ preload: build {preload['operator_build_s']:.2f} s (cache hit {preload['cache_hit']})")

preset = interferometer_standard_preset("rectangular")  # its seed / iid range only
mesh = al.mesh.RectangularUniform(shape=(side, side))
regularization = al.reg.MaternSparse(**MATERN)
model = interferometer_production_model(preset, mesh, regularization=regularization)
n_instances = max(int(_cli.n_instances), 2 + int(_cli.cold_evals))
instances = iid_instances(model, n_instances, preset)
instance0 = instances[0]

route_extra = {"use_edge_zeroed_pixels": False, "use_positive_only_solver": True}
settings_mf, memo_provenance = analysis_settings(
    preset, "off", interferometer_route="matrix_free", **route_extra
)
settings_fft, _ = analysis_settings(preset, "off", interferometer_route="fft", **route_extra)


def analysis_for(settings):
    return al.AnalysisInterferometer(dataset=dataset, settings=settings, use_jax=False)


def fit_for(instance, settings, ds=None):
    return al.FitInterferometer(
        dataset=dataset if ds is None else ds,
        tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
        settings=settings,
        xp=np,
    )


# --------------------------------------------------------------------------- matrix-free arm
print(f"\n--- matrix_free: {n_instances} iid instances, {side}x{side} mesh ---")
analysis_mf = analysis_for(settings_mf)
per_eval_s, log_likelihoods = [], []
for index, instance in enumerate(instances):
    t0 = time.perf_counter()
    value = float(analysis_mf.log_likelihood_function(instance=instance))
    per_eval_s.append(time.perf_counter() - t0)
    log_likelihoods.append(value)
    print(f"  [{index}] {per_eval_s[-1]:.4f} s  log evidence {value!r}")
timing = timing_summary(per_eval_s, n_cold=min(int(_cli.cold_evals), n_instances - 2))


def inversion_diagnostics(fit) -> dict:
    inv = fit.inversion
    out = {"inversion_class": type(inv).__name__}
    try:
        _ = inv.reconstruction
        result = getattr(inv, "positive_result", None)
        if result is not None:
            out["positive_result"] = {
                "interior_point_steps": int(result.interior_point_steps),
                "polish_iterations": int(result.polish_iterations),
                "gap": float(result.gap),
                "residual": float(result.residual),
                "converged": bool(result.converged),
                "free_pixels": int(np.sum(result.support)),
            }
        out["log_det_curvature_reg"] = float(inv.log_det_curvature_reg_matrix_term)
        out["log_det_regularization"] = float(inv.log_det_regularization_matrix_term)
        out["regularization_term"] = float(inv.regularization_term)
        out["figure_of_merit"] = float(fit.figure_of_merit)
    except Exception as exc:  # noqa: BLE001
        out.update(_error(exc))
    return out


fit_mf0 = fit_for(instance0, settings_mf)
mf_diag = inversion_diagnostics(fit_mf0)
try:
    inv = fit_mf0.inversion
    _, probes = matrix_free_util.log_det_curvature_reg_via_slq_from(
        operator=inv.operator,
        regularization_matrix_sparse=inv.regularization_matrix_sparse,
        regularization_factor_sparse=inv.regularization_factor_sparse,
        coefficient=1.0,
        n_probes=inv.slq_n_probes,
        n_steps=inv.slq_n_steps,
        seed=inv.slq_seed,
    )
    probes = np.asarray(probes, dtype=float).ravel()
    mf_diag["slq"] = {
        "n_probes": int(inv.slq_n_probes),
        "n_steps": int(inv.slq_n_steps),
        "seed": int(inv.slq_seed),
        "probe_values": [float(p) for p in probes],
        "probe_std": float(np.std(probes)) if probes.size else None,
        "standard_error": float(np.std(probes, ddof=1) / np.sqrt(probes.size))
        if probes.size > 1
        else None,
    }
except Exception as exc:  # noqa: BLE001
    mf_diag["slq"] = _error(exc)
print(f"  matrix-free diagnostics (instance 0): {json.dumps(mf_diag, default=float)[:600]}")

# --------------------------------------------------------------------------- dense fft arm
print("\n--- dense_fft reference (interferometer_route='fft') ---")
dense = {"route": "fft", "inversion": "InversionInterferometerSparse (dense F, H; Cholesky)"}
try:
    analysis_fft = analysis_for(settings_fft)
    t0 = time.perf_counter()
    dense["log_evidence_instance0"] = float(
        analysis_fft.log_likelihood_function(instance=instance0)
    )
    dense["first_call_s"] = time.perf_counter() - t0
    calls = []
    for instance in instances[1 : 1 + max(0, int(_args.fft_repeats))]:
        t0 = time.perf_counter()
        analysis_fft.log_likelihood_function(instance=instance)
        calls.append(time.perf_counter() - t0)
    dense["warm_call_s"] = calls
    dense["warm_median_s"] = float(np.median(calls)) if calls else None
    dense["diagnostics_instance0"] = inversion_diagnostics(fit_for(instance0, settings_fft))
    dense["status"] = "ok"
    print(
        f"  fft log evidence {dense['log_evidence_instance0']!r}; median {dense['warm_median_s']}"
    )
except Exception as exc:  # noqa: BLE001
    dense.update(_error(exc))
    print(f"  dense fft route failed: {dense['error']}")

# --------------------------------------------------------------------------- chunked dense
chunked = {"status": "skipped"}
if ds.n_vis <= CHUNKED_REFERENCE_MAX_VIS or _args.chunked_reference:
    print("\n--- chunked NUFFT dense reference (W~-independent) ---")
    try:
        ref = interferometer_dense_reference(dataset, None, instance0, settings_fft)
        chunked = {"status": "ok", **ref}
    except Exception as exc:  # noqa: BLE001
        chunked = _error(exc)
else:
    chunked["reason"] = (
        f"N_vis {ds.n_vis} > {CHUNKED_REFERENCE_MAX_VIS}: costly (pass --chunked-reference)"
    )

# --------------------------------------------------------------------------- agreement
mf0 = log_likelihoods[0]
agreement = {"quantity": "log evidence at iid draw 0", "matrix_free": mf0}
for name, block, key in (
    ("dense_fft", dense, "log_evidence_instance0"),
    ("chunked_dense", chunked, "log_evidence"),
):
    ref = block.get(key) if block.get("status") == "ok" else None
    if ref is not None:
        agreement[name] = {
            "log_evidence": ref,
            "abs_delta": abs(mf0 - ref),
            "rel_delta": abs(mf0 - ref) / max(abs(ref), 1e-300),
        }
exact_ld = (dense.get("diagnostics_instance0") or {}).get("log_det_curvature_reg")
slq_ld = mf_diag.get("log_det_curvature_reg")
if exact_ld is not None and slq_ld is not None:
    agreement["log_det_curvature_reg"] = {
        "slq": slq_ld,
        "exact_cholesky": exact_ld,
        "abs_delta": abs(slq_ld - exact_ld),
        "rel_delta": abs(slq_ld - exact_ld) / max(abs(exact_ld), 1e-300),
    }
print(f"\n  agreement: {json.dumps(agreement, default=float)}")

# --------------------------------------------------------------------------- row
provenance = device_info_dict()
revisions = (provenance.get("provenance") or {}).get("library_revisions")
row = {
    "experiment": EXPERIMENT,
    "release_label": "unreleased main",
    "library_shas": revisions,
    "autolens_version": al.__version__,
    "device": provenance,
    "resources": resources_dict("numba"),
    "instrument": instrument,
    "model": "rectangular_matrixfree",
    "configuration": {
        "mesh": "RectangularUniform",
        "mesh_shape": [side, side],
        "source_pixels": side * side,
        "regularization": {"scheme": "MaternSparse", **MATERN, "declared": True},
        "settings": {"interferometer_route": "matrix_free", **route_extra},
        "lens_light": None,
        "mass": "Isothermal + ExternalShear, interferometer standard-cell priors",
        "visibilities": int(ds.n_vis),
        "image_pixels_masked": int(ds.n_image_pixels),
        "mask_radius_arcsec": ds.mask_radius,
        "pixel_scale_arcsec": ds.pixel_scale,
        "n_instances": n_instances,
        "instance_stream": {"kind": "iid", "seed": preset.seed, "unit_range": [0.4, 0.6]},
        "thread_env": _thread_env,
        "memo_provenance": memo_provenance,
        "preload": preload,
        "inversion_path": "matrix_free",
    },
    "full_pipeline_single_jit": timing["warm_iid_median_s"],
    **timing,
    "log_likelihood_sequence": log_likelihoods,
    "matrix_free_diagnostics": mf_diag,
    "dense_fft": dense,
    "chunked_dense_reference": chunked,
    "agreement": agreement,
    "never_current": "experiment / scan row (likelihood-lifecycle 2b); no parity.json",
}
path = output_path(_cli, instrument, side, root)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(row, indent=2, default=float))
print(f"\n  row -> {path}")
print(
    f"  matrix-free warm median {timing['warm_iid_median_s']:.4f} s; "
    f"dense fft median {dense.get('warm_median_s')}"
)

render_fit_png(fit_mf0, path.parent, dataset_type="interferometer", cli=_cli)
