"""Gradient harness (phase 6a, autolens_profiling#505) on toy likelihoods."""

from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from gradient import harness  # noqa: E402

gw = harness.gw  # tooling/gradient_witness.py, on the path the harness sets


def gaussian(params):
    """A smooth log-likelihood over a pytree (a dict of arrays and a scalar)."""
    x = params["x"]
    return -0.5 * jnp.sum((x - params["mu"]) ** 2) + jnp.sum(jnp.sin(x)) * params["s"]


PARAMS = {"x": jnp.linspace(-1.0, 2.0, 6), "mu": jnp.float64(0.3), "s": jnp.float64(1.7)}


@jax.custom_vjp
def wrong(x):
    return jnp.sum(x**2)


def _wrong_fwd(x):
    return wrong(x), x


def _wrong_bwd(x, g):
    return (g * 3.0 * x,)  # the true gradient is 2x


wrong.defvjp(_wrong_fwd, _wrong_bwd)


def _doc(measured):
    return harness.witness(
        measured,
        dataset_type="imaging",
        variant="mge",
        instrument="hst",
        setup_id="jax_cpu_dense_fp64",
        host="hpc_ral_cpu",
        autolens_version="2026.10.25.1",
        device={"backend": "cpu"},
    )


def test_a_correct_gradient_passes_the_fd_rule_and_writes_a_valid_witness(tmp_path):
    measured = harness.measure(gaussian, PARAMS, n_repeats=3, vmap_batch=4)
    assert measured["fd"]["status"] == "pass"
    assert measured["fd"]["max_rel_error"] <= gw.FD_TOLERANCE
    assert len(measured["fd"]["directions"]) == gw.FD_DIRECTIONS
    assert measured["gradient"] == {
        "finite": True,
        "l2_norm": measured["gradient"]["l2_norm"],
        "nonzero": True,
    }
    assert measured["n_parameters"] == 8 and measured["precision"] == "float64"
    assert measured["log_likelihood"] == pytest.approx(float(gaussian(PARAMS)))
    t = measured["timings"]
    assert all(t[k] and t[k] > 0 for k in gw.TIMINGS)
    assert measured["grad_over_forward"] == pytest.approx(
        t["value_and_grad.single_call_s"] / t["forward.single_call_s"]
    )
    doc = _doc(measured)
    assert gw.validate(doc) == []
    cell = harness.cell_dir(
        tmp_path,
        dataset_type="imaging",
        variant="mge",
        instrument="hst",
        setup_id="jax_cpu_dense_fp64",
        host="hpc_ral_cpu",
    )
    path = harness.write_witness(cell, doc)
    assert path == tmp_path / "results/imaging/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/gradient.json"
    assert gw.validate(json.loads(path.read_text())) == []


def test_a_wrong_gradient_fails_the_fd_rule_and_is_a_hazard():
    measured = harness.measure(
        lambda p: wrong(p["x"]), {"x": PARAMS["x"]}, n_repeats=2, vmap_batch=None
    )
    assert measured["fd"]["status"] == "fail"
    # The backward pass returns 3x where the truth is 2x: |2 - 3| / 3 in every direction.
    assert measured["fd"]["max_rel_error"] == pytest.approx(1 / 3, rel=1e-6)
    assert measured["hazard"]["flag"] is True
    assert "finite-difference check failed" in measured["hazard"]["reasons"]
    assert gw.validate(_doc(measured)) == []


def test_an_expensive_gradient_is_a_hazard_by_the_ratio():
    assert gw.hazard(4.5, "pass")["flag"] is True
    assert gw.hazard(3.9, "pass") == {
        "flag": False,
        "reasons": [],
        "threshold_grad_over_forward": gw.HAZARD_GRAD_OVER_FORWARD,
    }


def test_the_validator_refuses_a_witness_off_the_rules():
    doc = _doc(harness.measure(gaussian, PARAMS, n_repeats=2, vmap_batch=None))
    bad = json.loads(json.dumps(doc))
    bad["fd"]["status"] = "fail"
    assert any("pre-registered rule" in e for e in gw.validate(bad))
    bad = json.loads(json.dumps(doc))
    bad["hazard"]["flag"] = True
    assert any("hazard" in e for e in gw.validate(bad))
    bad = json.loads(json.dumps(doc))
    bad["timings"]["value_and_grad.single_call_s"] = 0
    assert any("positive" in e for e in gw.validate(bad))
    bad = json.loads(json.dumps(doc))
    bad["fd"]["directions"][0]["rel_error"] = 0.5
    assert any("rel_error disagrees" in e for e in gw.validate(bad))


def test_richardson_cancels_the_leading_error():
    # f(x) = x**3 at x = 1: central difference error is h**2, Richardson removes it exactly.
    def d(h):
        return ((1 + h) ** 3 - (1 - h) ** 3) / (2 * h)

    assert gw.richardson(d(1e-2), d(5e-3)) == pytest.approx(3.0, rel=1e-12)
