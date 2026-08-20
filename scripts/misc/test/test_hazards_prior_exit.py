"""The prior-exit hazard detector (autolens_profiling#128)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _misc_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ruff.toml").exists():
            return parent / "scripts" / "misc"
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


sys.path.insert(0, str(_misc_dir()))

from hazards.checks import CHECKS  # noqa: E402
from hazards.checks._base import ScanContext  # noqa: E402
from hazards.checks.prior_exit import PriorExitCheck  # noqa: E402


@pytest.fixture
def context():
    return ScanContext(
        repo_root=Path("."),
        workspace_root=_misc_dir().parents[1].parent,
        output_root=Path("."),
        backends=("numpy", "jax"),
        sample_count=500,
        seed=0,
    )


def test__prior_exit_is_registered():
    assert any(isinstance(check, PriorExitCheck) for check in CHECKS)


def test__non_finite_exactly_where_the_value_left_the_box(context):
    # The decisive property: the -inf tracks the box EDGE, so this is a support
    # boundary rather than a numerical instability that happens to be nearby.
    finding = PriorExitCheck().run(context)[0]

    assert finding.reproducer["non_finite_iff_outside"] is True
    assert finding.reproducer["outside_box"] > 0
    assert finding.reproducer["non_finite_log_prior"] == finding.reproducer["outside_box"]


def test__numpy_path_enforces_the_bound_since_pyautofit_1489(context):
    # PyAutoFit#1489 (2026-08-18) made the NumPy path evaluate the bound —
    # 0.0 inside, -inf outside, same `xp.where` as JAX. Before that the NumPy
    # path short-circuited to a scalar 0.0 and the hazard was JAX-only; this
    # test's predecessor pinned that asymmetry and correctly broke when it
    # flipped. If THIS test breaks, the NumPy path has regressed to the
    # unevaluated-bound behaviour and backend_reachability must flip back.
    finding = PriorExitCheck().run(context)[0]

    assert finding.reproducer["numpy_evaluates_bound"] is True
    assert finding.reproducer["numpy_non_finite_outside_box"] == finding.reproducer["outside_box"]
    assert finding.backend_reachability["numpy"]["log_prior_outside_box"].startswith("non_finite")
    assert finding.backend_reachability["jax"]["log_prior_outside_box"] == "non_finite"


def test__records_the_clipper_as_what_blocks_it(context):
    # The mitigation is opt-in, so the hazard must still report as existing
    # while naming what blocks it.
    finding = PriorExitCheck().run(context)[0]

    assert finding.code_exists is True
    assert finding.affects_science is True
    assert any("ClipperPriorBox" in anchor.symbol for anchor in finding.blocked_by)


def test__a_numpy_only_scan_adjudicates_since_the_paths_are_symmetric():
    # Pre-PyAutoFit#1489 a NumPy-only scan could not adjudicate this hazard
    # (the NumPy path never evaluated the bound). Both paths now share the
    # same `xp.where`, so a NumPy-only scan measures the same support
    # boundary — it adjudicates, and the hazard persists (the -inf outside
    # the box IS the hazard). The jax entry must still be absent: reporting
    # reachability for a backend the scan never ran stays forbidden.
    numpy_only = ScanContext(
        repo_root=Path("."),
        workspace_root=_misc_dir().parents[1].parent,
        output_root=Path("."),
        backends=("numpy",),
        sample_count=200,
        seed=0,
    )
    finding = PriorExitCheck().run(numpy_only)[0]

    assert finding.reproducer["adjudicated"] is True
    assert finding.reproducer["hazard_persists"] is True
    assert finding.reproducer["non_finite_iff_outside"] is True
    assert "jax" not in finding.backend_reachability
