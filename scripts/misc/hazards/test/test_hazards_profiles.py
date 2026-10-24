"""Public profile-registry discovery tests with fake namespaces."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _profiling_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "ruff.toml").is_file():
            return path
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


_misc = _profiling_root() / "scripts" / "misc"
if str(_misc) not in sys.path:
    sys.path.insert(0, str(_misc))

from hazards._profiles import coverage_payload, discover_profile_registry  # noqa: E402


class Sersic:
    pass


class GaussianSph:
    pass


class Isothermal:
    pass


def test_registry_discovers_families_geometry_and_deduplicates_aliases():
    fake = SimpleNamespace(
        lp=SimpleNamespace(Sersic=Sersic, GaussianSph=GaussianSph),
        lp_linear=SimpleNamespace(Sersic=Sersic),
        lmp=SimpleNamespace(),
        mp=SimpleNamespace(Isothermal=Isothermal),
    )
    specs = discover_profile_registry(fake)
    assert len(specs) == 3
    assert {spec.family for spec in specs} == {"light", "mass"}
    assert next(spec for spec in specs if spec.class_name == "GaussianSph").geometry == "spherical"
    payload = coverage_payload(specs)
    assert payload["counts"]["total"] == 3
    assert "not multiplied by backend" in payload["matrix_policy"]
