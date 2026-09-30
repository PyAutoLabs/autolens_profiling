"""
Linear-solver euclid latent: ``total_source_flux`` from each candidate's reconstruction
=======================================================================================

POST-HOC, NOT PRE-REGISTERED (added 2026-09-30 after the phase-1 deciding run; an input to the
phase-2 fix, not a verdict). The euclid pipeline's
``tests/test_compute_latent_variable.py::test_latent_euclid_variables_traces_under_jax_jit``
compares the jitted ``total_source_flux`` latent with the eager NumPy one at rel 1e-3; on the
released library the jitted value is +5.76 % off, because the raw forward PDIP stop leaves
amplitude on source columns the fnnls reference has at zero. This cell answers directly *which
candidate would turn that test green*: it re-solves the corpus system ``euclid_vis_lp_k0`` with
every candidate and pushes each reconstruction through the pipeline's own latent code.

Method (the pipeline's code, not a transcription of it): build the eager NumPy analysis, model
and parameter vector exactly as the test does (``_analysis_from``, ``_vis_lp_model``,
``_ordered_median_vector``), then for each candidate swap
``inversion_util.reconstruction_positive_only_from`` for a stub that returns the candidate's
``x`` (the corpus system *is* this fit's positive-only system — captured by ``capture.py
--source euclid_vis_lp``), build the fit the way ``LatentEuclid.variables`` does
(``latent_instance_from`` + ``analysis.fit_from``) and read ``total_source_flux`` from
``LatentEuclid._source_flux_latents_on_uniform_grid`` — the method ``variables`` uses for that
key. The stub checks the shape and counts its calls; the linear-profile tracer then turns the
injected amplitudes into the source image summed on the uniform over-sample-4 grid.

Validation (gates publication): injecting the stored fnnls ``x_ref`` must reproduce the corpus
``latent_reference.eager_numpy`` and injecting ``pdip_raw``'s ``x`` the jitted latent of the
library it runs against (``PDIP_RAW_JIT_EXPECTED``), both to rel 1e-6. If either fails the cell
writes nothing and exits 1. Phase 2 (PyAutoArray#595) re-based the second leg: the corpus
``latent_reference.jit`` (3.511093374, +5.76 %) is the capture-time *released* library's value
and stays in the manifest as a record; library main now returns the polished iterate.

Run from the repo root (needs the euclid pipeline checkout beside this repo, and its dataset)::

    python scripts/lens/solver/euclid_latent.py

Output
------
``results/lens/solver/euclid_latent_by_candidate_v<version>.json``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _profiling_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "ruff.toml").exists():
            return p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO_ROOT = _profiling_root()
for _p in (REPO_ROOT / "scripts" / "misc", REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _driver  # noqa: E402

_DEVICE = _driver.select_device_from_argv()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from autoarray.inversion.inversion import inversion_util  # noqa: E402

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

import _corpus  # noqa: E402
import _solvers  # noqa: E402

EUCLID_ROOT = REPO_ROOT.parent / "euclid_strong_lens_modeling_pipeline"
GROUP = "euclid_vis_lp"
SYSTEM = "euclid_vis_lp_k0"
VALIDATION_RTOL = 1.0e-6
TEST_RTOL = 1.0e-3

# The jitted ``total_source_flux`` the ``pdip_raw -> jit`` validation leg must reproduce.
# Re-based in phase 2 for PyAutoArray#595 (merged 2026-09-30, merge 7a89e19a0): the jit path
# (``solve_nnls_primal_raw_forward``) now returns the #573 polished iterate instead of the raw
# forward stop, so injecting ``pdip_raw``'s ``x`` no longer reproduces the capture-time value
# ``latent_reference.jit`` = 3.511093374207152 (+5.76e-2 vs eager, the released library before
# #595), which this constant replaces. Value: the euclid test's own jitted
# ``LatentEuclid.variables`` on library main 7a89e19a0, euclid pipeline 26e4385b, measured
# 2026-09-30 (+7.47e-5 vs eager). The manifest keeps the old value as the capture record.
PDIP_RAW_JIT_EXPECTED = 3.320127603567922
PDIP_RAW_JIT_SOURCE = (
    "jitted LatentEuclid.variables on PyAutoArray main 7a89e19a0 (post-#595), "
    "euclid pipeline 26e4385b, 2026-09-30; replaces the pre-#595 released jit 3.511093374207152"
)


def _euclid_fit_factory():
    """``x -> total_source_flux`` through the euclid pipeline's own eager latent code."""
    os.environ.pop("PYAUTO_SMALL_DATASETS", None)
    for p in (EUCLID_ROOT, EUCLID_ROOT / "tests"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import test_compute_latent_variable as tl
    import util
    from autolens.analysis.latent import latent_instance_from

    dataset = util.load_vis_dataset(tl.SIMULATED_DATASET, sample_name=tl.SIMULATED_SAMPLE)
    analysis = tl._analysis_from(dataset)
    model = tl._vis_lp_model(dataset)
    vector = tl._ordered_median_vector(model)
    magzero = analysis.kwargs.get("magzero", None)
    original = inversion_util.reconstruction_positive_only_from

    def total_source_flux(x) -> tuple[float, int]:
        x = np.asarray(x, dtype=np.float64)
        calls = []

        def stub(data_vector, curvature_reg_matrix, *args, **kwargs):
            if np.shape(data_vector) != x.shape:
                raise ValueError(f"injected x {x.shape} != system {np.shape(data_vector)}")
            calls.append(1)
            return x.copy()

        inversion_util.reconstruction_positive_only_from = stub
        try:
            instance = latent_instance_from(model=model, parameters=vector, xp=np)
            fit = analysis.fit_from(instance=instance)
            values = util.LatentEuclid._source_flux_latents_on_uniform_grid(
                fit=fit, magzero=magzero, keys=["total_source_flux"], xp=np
            )
            flux = float(values["total_source_flux"])
        finally:
            inversion_util.reconstruction_positive_only_from = original
        return flux, len(calls)

    from likelihood_breakdown.provenance import git_revision

    info = {
        "euclid_pipeline_git_sha": git_revision(EUCLID_ROOT.resolve()),
        "dataset": f"euclid_strong_lens_modeling_pipeline/dataset/{tl.SIMULATED_SAMPLE}/"
        f"{tl.SIMULATED_DATASET}",
        "test": "tests/test_compute_latent_variable.py::"
        "test_latent_euclid_variables_traces_under_jax_jit (rel 1e-3)",
    }
    return total_source_flux, info


def main() -> int:
    args = _driver.parse_cli(
        __doc__.splitlines()[1], _solvers.ACCURACY_DEFAULT + _solvers.POSTHOC_CANDIDATES
    )
    (system,) = [s for s in _corpus.load_corpus([GROUP]) if s.name == SYSTEM]
    group = next(g for g in _corpus.read_manifest()["groups"] if g["name"] == GROUP)
    ref = group["source"]["latent_reference"]
    eager, jit_released = float(ref["eager_numpy"]), float(ref["jit"])
    jit = PDIP_RAW_JIT_EXPECTED

    total_source_flux, info = _euclid_fit_factory()

    # --- validation: x_ref -> eager, pdip_raw -> jit (library main, post-#595) -------------
    flux_ref, n_ref = total_source_flux(system.x_ref)
    x_raw, _ = _solvers.CANDIDATES["pdip_raw"].fn(system.Q, system.q, system.meta)
    flux_raw, n_raw = total_source_flux(x_raw)
    validation = {
        "x_ref_to_eager": {
            "value": flux_ref,
            "expected": eager,
            "rel": (flux_ref - eager) / eager,
            "stub_calls": n_ref,
        },
        "pdip_raw_to_jit": {
            "value": flux_raw,
            "expected": jit,
            "rel": (flux_raw - jit) / jit,
            "stub_calls": n_raw,
            "expected_source": PDIP_RAW_JIT_SOURCE,
        },
        "rtol": VALIDATION_RTOL,
    }
    ok = all(
        abs(v["rel"]) <= VALIDATION_RTOL and v["stub_calls"] >= 1
        for k, v in validation.items()
        if k != "rtol"
    )
    validation["passed"] = ok
    for k, v in validation.items():
        if isinstance(v, dict):
            print(f"  validation {k}: {v['value']:.9g} vs {v['expected']:.9g} rel {v['rel']:.2e}")
    if not ok:
        print("  VALIDATION FAILED -- no latent numbers written.")
        return 1

    rows = []
    for name in args.candidates:
        cand = _solvers.CANDIDATES[name]
        x, stats = cand.fn(system.Q, system.q, system.meta, target_kappa=args.target_kappa)
        finite = bool(np.all(np.isfinite(x)))
        flux, _ = total_source_flux(x) if finite else (None, 0)
        rel = None if flux is None else (flux - eager) / eager
        rows.append(
            {
                "candidate": name,
                "posthoc": name in _solvers.POSTHOC_CANDIDATES,
                "total_source_flux": flux,
                "rel_vs_eager": rel,
                "passes_test_rtol": None if rel is None else abs(rel) <= TEST_RTOL,
                "converged": stats.get("converged"),
                "iterations": stats.get("iterations"),
            }
        )
        print(
            f"  {name:<28s} flux={flux!s:<20s} rel={'None' if rel is None else f'{rel:+.3e}'} "
            f"conv={stats.get('converged')} it={stats.get('iterations')}"
        )

    import autolens as al

    from _profile_cli import device_info_dict

    summary = {
        "autolens_version": al.__version__,
        "device": device_info_dict(),
        "cell": "euclid_latent",
        "posthoc": "POST-HOC, NOT PRE-REGISTERED: an input to the phase-2 fix, not a verdict.",
        "system": f"{GROUP}/{SYSTEM}",
        "latent_key": "total_source_flux",
        "latent_reference": {
            "eager_numpy": eager,
            "jit_released": jit_released,
            "jit_library_main": jit,
        },
        "method": "reconstruction injected through inversion_util.reconstruction_positive_only_from "
        "into the eager NumPy fit; total_source_flux from "
        "LatentEuclid._source_flux_latents_on_uniform_grid",
        "euclid": info,
        "test_rtol": TEST_RTOL,
        "validation": validation,
        "rows": rows,
    }
    out_dir = Path(args.output_dir) if args.output_dir is not None else _driver.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _driver.write_json(out_dir / f"euclid_latent_by_candidate_v{al.__version__}.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
