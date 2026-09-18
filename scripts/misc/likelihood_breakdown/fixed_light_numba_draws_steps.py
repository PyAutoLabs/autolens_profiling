"""Small, testable helpers for the phase-5 production NNLS replay.

``autoarray`` is imported only inside the two context managers.  Importing this
module therefore remains a NumPy-only operation for manifest and result tests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

WALK_PARAMETERS = ("einstein_radius", "ell_comps_0", "centre_x", "slope")
WALK_TARGETS = (-10.0, -100.0, -1000.0, -10000.0)
N_RANDOM = 24
RANDOM_SEED = 0
EVIDENCE_REL_TOL = 1.0e-9
APPROVED_MANIFEST_SHA256 = "7d10341d96302c187600a5f3201f76847594bb6808b76d651c483dd4a42e854f"


def _expected_names() -> list[str]:
    suffixes = ("m10", "m100", "m1000", "m1e4")
    return (
        ["fiducial"]
        + [f"walk_{parameter}_{suffix}" for parameter in WALK_PARAMETERS for suffix in suffixes]
        + [f"random_{index:02d}" for index in range(N_RANDOM)]
    )


def load_manifest(path: str | Path) -> tuple[list[dict], dict]:
    """Load and strictly validate the frozen 41-draw phase-3 manifest.

    Rows are copied whole: later phases need the original measured fields as
    provenance and must not silently reduce the manifest to offsets alone.
    """
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    rows = payload.get("draws")
    configuration = payload.get("configuration", {})

    if not isinstance(rows, list) or len(rows) != 41:
        raise ValueError("manifest must contain exactly 41 draws")
    if configuration.get("seed") != RANDOM_SEED:
        raise ValueError("manifest random seed must be 0")
    if configuration.get("n_random") != N_RANDOM:
        raise ValueError("manifest must declare 24 random draws")

    names = [row.get("name") for row in rows]
    if names != _expected_names() or len(set(names)) != len(names):
        raise ValueError("manifest draw names or order do not match the frozen study")

    fiducials = [row for row in rows if row.get("kind") == "fiducial"]
    walks = [row for row in rows if row.get("kind") == "walk"]
    randoms = [row for row in rows if row.get("kind") == "random"]
    if len(fiducials) != 1 or len(walks) != 16 or len(randoms) != 24:
        raise ValueError("manifest families must be 1 fiducial, 16 walks, and 24 random draws")

    walk_membership = {(row.get("parameter"), float(row.get("target_d_log_l"))) for row in walks}
    expected_walks = {
        (parameter, target) for parameter in WALK_PARAMETERS for target in WALK_TARGETS
    }
    if walk_membership != expected_walks:
        raise ValueError("manifest walk parameter/target membership is incomplete")
    if [row.get("index") for row in randoms] != list(range(N_RANDOM)):
        raise ValueError("manifest random indices must be exactly 0 through 23")

    random_parameters = {
        "einstein_radius",
        "ell_comps_0",
        "ell_comps_1",
        "centre_0",
        "centre_1",
    }
    for row in rows:
        offsets = row.get("offsets")
        if not isinstance(offsets, dict) or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in offsets.values()
        ):
            raise ValueError("manifest offsets must be finite numeric values")
        if row["kind"] == "fiducial":
            consistent = (
                row.get("parameter") is None
                and row.get("index") is None
                and not offsets
                and row.get("mass_cls") == "Isothermal"
            )
        elif row["kind"] == "walk":
            parameter = row.get("parameter")
            expected_mass = "PowerLaw" if parameter == "slope" else "Isothermal"
            consistent = (
                row.get("index") is None
                and set(offsets) == {parameter}
                and row.get("mass_cls") == expected_mass
            )
        else:
            consistent = (
                row.get("parameter") is None
                and set(offsets) == random_parameters
                and row.get("mass_cls") == "Isothermal"
            )
        if not consistent:
            raise ValueError(f"manifest row {row.get('name')!r} is inconsistent with its family")

    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != APPROVED_MANIFEST_SHA256:
        raise ValueError("manifest SHA256 does not match the approved frozen draw set")

    provenance = {
        "source_name": manifest_path.name,
        "sha256": manifest_sha256,
        "seed": RANDOM_SEED,
        "n_draws": len(rows),
        "family_counts": {"fiducial": 1, "walk": 16, "random": 24},
        "autolens_version": payload.get("autolens_version"),
        "configuration": dict(configuration),
    }
    return [dict(row) for row in rows], provenance


@contextlib.contextmanager
def memo_scope(enabled: bool) -> Iterator[None]:
    """Give one traversal an empty production memo and restore prior state."""
    from autoarray.inversion.inversion import nnls_memo

    memo = nnls_memo._nnls_passive_set_memo
    snapshot = dict(memo)
    old_environment = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    try:
        memo.clear()
        os.environ["AUTOARRAY_NNLS_WARM_START"] = "1" if enabled else "0"
        yield
    finally:
        memo.clear()
        memo.update(snapshot)
        if old_environment is None:
            os.environ.pop("AUTOARRAY_NNLS_WARM_START", None)
        else:
            os.environ["AUTOARRAY_NNLS_WARM_START"] = old_environment


def _json_stats(stats: dict) -> dict:
    result = {}
    for key in (
        "seed_source",
        "warm_start_fallback",
        "outer_iterations",
        "inner_iterations",
        "warm_start_errors",
        "n_passive",
    ):
        if key in stats:
            value = stats[key]
            result[key] = value.item() if isinstance(value, np.generic) else value
    if "passive_set" in stats:
        result["passive_set"] = np.asarray(stats["passive_set"], dtype=int).tolist()
    return result


@contextlib.contextmanager
def observe_solver() -> Iterator[list[dict]]:
    """Observe production NNLS calls without replacing either algorithm.

    The outer wrapper reads the shared stats dictionary only after production
    has applied its memo guard.  The kernel wrapper additionally retains every
    attempt, including a failed memo seed followed by the production cold retry.
    """
    from autoarray.inversion.inversion import inversion_util
    from autoarray.util import fnnls

    original_outer = inversion_util.reconstruction_positive_only_from
    original_kernel = fnnls.fnnls_cholesky
    calls: list[dict] = []
    current: list[dict] = []

    def observed_kernel(*args, **kwargs):
        stats = kwargs.get("stats")
        initial = kwargs.get("P_initial")
        attempt = {
            "initial_seed_source": "dense" if np.asarray(initial).dtype == bool else "memo",
            "_stats": stats,
        }
        current[-1]["kernel_attempts"].append(attempt)
        try:
            result = original_kernel(*args, **kwargs)
        except Exception as error:
            attempt.update(_json_stats(stats or {}))
            attempt["error"] = f"{type(error).__name__}: {error}"
            raise
        attempt.update(_json_stats(stats or {}))
        attempt["error"] = None
        return result

    def observed_outer(*args, **kwargs):
        call = {"kernel_attempts": []}
        current.append(call)
        try:
            result = original_outer(*args, **kwargs)
            attempts = call["kernel_attempts"]
            successful = next(
                (item for item in reversed(attempts) if item.get("error") is None), {}
            )
            successful.update(_json_stats(successful.pop("_stats", {}) or {}))
            for attempt in attempts:
                attempt.pop("_stats", None)
            call.update({key: value for key, value in successful.items() if key != "error"})
            call["warm_start_fallback"] = bool(successful.get("warm_start_fallback", False))
            call["warm_start_errors"] = successful.get("warm_start_errors")
            call["final_passive_set"] = successful.get("passive_set", [])
            call["error"] = None
            return result
        except Exception as error:
            for attempt in call["kernel_attempts"]:
                attempt.pop("_stats", None)
            call["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            calls.append(call)
            current.pop()

    inversion_util.reconstruction_positive_only_from = observed_outer
    fnnls.fnnls_cholesky = observed_kernel
    try:
        yield calls
    finally:
        inversion_util.reconstruction_positive_only_from = original_outer
        fnnls.fnnls_cholesky = original_kernel


def compare_solutions(warm: dict, cold: dict) -> dict:
    """Apply the declared evidence and exact-active-set correctness gates."""
    warm_evidence = float(warm["evidence"])
    cold_evidence = float(cold["evidence"])
    warm_reconstruction = np.asarray(warm["reconstruction"], dtype=float)
    cold_reconstruction = np.asarray(cold["reconstruction"], dtype=float)

    same_shape = warm_reconstruction.shape == cold_reconstruction.shape
    finite = bool(
        math.isfinite(warm_evidence)
        and math.isfinite(cold_evidence)
        and np.all(np.isfinite(warm_reconstruction))
        and np.all(np.isfinite(cold_reconstruction))
    )
    evidence_abs = abs(warm_evidence - cold_evidence)
    evidence_scale = max(abs(warm_evidence), abs(cold_evidence))
    evidence_rel = evidence_abs / evidence_scale if evidence_scale else evidence_abs

    if same_shape and warm_reconstruction.size:
        difference = np.abs(warm_reconstruction - cold_reconstruction)
        reconstruction_max_abs = float(np.max(difference))
        reconstruction_scale = max(
            float(np.max(np.abs(warm_reconstruction))),
            float(np.max(np.abs(cold_reconstruction))),
        )
        reconstruction_max_rel = (
            reconstruction_max_abs / reconstruction_scale
            if reconstruction_scale
            else reconstruction_max_abs
        )
    elif same_shape:
        reconstruction_max_abs = 0.0
        reconstruction_max_rel = 0.0
    else:
        reconstruction_max_abs = math.inf
        reconstruction_max_rel = math.inf

    warm_passive = [int(value) for value in warm["passive_set"]]
    cold_passive = [int(value) for value in cold["passive_set"]]
    active_sets_valid = len(warm_passive) == len(set(warm_passive)) and len(cold_passive) == len(
        set(cold_passive)
    )
    active_set_equal = active_sets_valid and sorted(warm_passive) == sorted(cold_passive)
    evidence_passed = finite and evidence_rel <= EVIDENCE_REL_TOL
    return {
        "finite": finite,
        "evidence_absolute_difference": evidence_abs if math.isfinite(evidence_abs) else None,
        "evidence_relative_difference": evidence_rel if math.isfinite(evidence_rel) else None,
        "evidence_relative_tolerance": EVIDENCE_REL_TOL,
        "evidence_passed": evidence_passed,
        "active_sets_valid": active_sets_valid,
        "active_set_equal": active_set_equal,
        "reconstruction_same_shape": same_shape,
        "reconstruction_max_absolute_difference": (
            reconstruction_max_abs if math.isfinite(reconstruction_max_abs) else None
        ),
        "reconstruction_max_relative_difference": (
            reconstruction_max_rel if math.isfinite(reconstruction_max_rel) else None
        ),
        "passed": bool(finite and evidence_passed and active_set_equal and same_shape),
    }


def replay_sequence(evaluate: Callable[[int], float], indices, enabled: bool) -> list[dict]:
    """Evaluate one ordered traversal with timing scoped to each evaluation."""
    rows = []
    with memo_scope(enabled=enabled):
        for index in indices:
            start = time.perf_counter_ns()
            evidence = evaluate(index)
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
            rows.append({"index": index, "evidence": float(evidence), "elapsed_ms": elapsed_ms})
    return rows
