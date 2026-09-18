"""Paired production-fit witnesses for the permuted NNLS experiment (#276).

Each lane owns its memo history. The identity lane replays the library exactly;
its checks must pass before the candidate's numerical comparisons are useful.
"""

import copy

import numpy as np
from autoarray.inversion.inversion import nnls_memo

from likelihood_breakdown import fixed_light_numpy_solvers as solvers


def evaluate_fit(make_fit, kernel, memo):
    """Evaluate a fresh fit, observing every kernel attempt and preserving state."""
    saved = dict(nnls_memo._nnls_passive_set_memo)
    nnls_memo._nnls_passive_set_memo.clear()
    nnls_memo._nnls_passive_set_memo.update(copy.deepcopy(memo))
    attempts = []

    def observed(matrix, rhs, **kwargs):
        attempt = {"error": None}
        attempts.append(attempt)
        try:
            x = kernel(matrix, rhs, **kwargs)
        except Exception as error:
            attempt["error"] = type(error).__name__
            raise
        attempt.update(
            matrix=np.asarray(matrix),
            rhs=np.asarray(rhs),
            x=np.asarray(x),
            stats=kwargs.get("stats"),
            factor=kwargs.get("factor"),
        )
        return x

    try:
        with solvers.fnnls_kernel_injected(observed, label="s4b_witness") as counts:
            fit = make_fit()
            evidence = float(fit.figure_of_merit)
        if not attempts or attempts[-1]["error"] is not None:
            raise AssertionError("Witness did not observe a successful NNLS kernel")
        last = attempts[-1]
        stats = last["stats"] or {}
        factor = last["factor"]
        result = {
            "evidence": evidence,
            "x": last["x"].copy(),
            "passive_set": np.asarray(stats["passive_set"]).copy(),
            "iterations": solvers.iteration_row(stats),
            "factor": solvers.log_det_factor_check(last["matrix"], factor),
            "kkt": solvers.kkt_residuals(
                last["matrix"], last["rhs"], last["x"], stats["passive_set"]
            ),
            "kernel_calls": counts["calls"],
            "kernel_errors": [a["error"] for a in attempts if a["error"] is not None],
            "matrix": last["matrix"],
            "rhs": last["rhs"],
            "inversion_class": type(fit.inversion).__name__,
        }
        return result, copy.deepcopy(nnls_memo._nnls_passive_set_memo)
    finally:
        nnls_memo._nnls_passive_set_memo.clear()
        nnls_memo._nnls_passive_set_memo.update(saved)


def compare(reference, candidate, *, identity=False):
    """Fail closed: exact passive-set equality; evidence/factor tolerances pinned."""
    active = solvers.active_set_jaccard(reference["passive_set"], candidate["passive_set"])
    evidence = solvers.evidence_delta(candidate["evidence"], reference["evidence"])
    fallback = all(
        reference["iterations"][k] == candidate["iterations"][k]
        for k in ("seed_source", "warm_start_fallback")
    )
    fallback = fallback and reference["kernel_errors"] == candidate["kernel_errors"]
    fallback = fallback and reference["kernel_calls"] == candidate["kernel_calls"]
    same_problem = np.array_equal(reference["matrix"], candidate["matrix"]) and np.array_equal(
        reference["rhs"], candidate["rhs"]
    )
    identity_pass = (
        np.array_equal(reference["x"], candidate["x"])
        and evidence["bit_identical"]
        and reference["iterations"] == candidate["iterations"]
    )
    gates = {
        "same_problem": same_problem,
        "W1_identity": identity_pass if identity else True,
        "W2_active_set": active["identical"],
        "W3_evidence": evidence["within_rtol"],
        "W4_fallback": bool(fallback),
        "W6_factor": reference["factor"]["within_atol"] and candidate["factor"]["within_atol"],
    }
    return {
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "active_set": active,
        "evidence": evidence,
        "max_abs_delta_reconstruction": float(np.max(np.abs(reference["x"] - candidate["x"]))),
        "kkt": {"library": reference["kkt"], "candidate": candidate["kkt"]},
        "iterations": {"library": reference["iterations"], "candidate": candidate["iterations"]},
        "kernel_errors": {
            "library": reference["kernel_errors"],
            "candidate": candidate["kernel_errors"],
        },
        "kernel_calls": {
            "library": reference["kernel_calls"],
            "candidate": candidate["kernel_calls"],
        },
        "factor": {"library": reference["factor"], "candidate": candidate["factor"]},
    }
