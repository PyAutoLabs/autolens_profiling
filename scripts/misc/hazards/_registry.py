"""Small registry describing the phase-one vertical-slice subjects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectSpec:
    subject: str
    name: str
    description: str

    @property
    def key(self) -> str:
        return f"{self.subject}:{self.name}"


SUBJECTS = {
    spec.key: spec
    for spec in (
        SubjectSpec("component", "ell_comps", "Ellipticity conversion and profile construction"),
        SubjectSpec("component", "spherical_geometry", "Radial geometry at profile centre"),
        SubjectSpec("component", "power_law", "Elliptical power-law deflections"),
        SubjectSpec("matrix", "curvature_matrix", "Synthetic inversion curvature matrices"),
        SubjectSpec(
            "likelihood",
            "imaging_pixelization",
            "Complete 7x7 imaging likelihood with a rectangular source inversion",
        ),
        SubjectSpec(
            "likelihood",
            "prior_support",
            "Prior support of the MAP objective, where a bounded prior can make a "
            "finite likelihood non-finite",
        ),
        SubjectSpec(
            "likelihood",
            "positions_penalty",
            "PositionsLH threshold hinge, zero-gradient interior plateau, and "
            "argmax-switch kinks (Phase 4 Stage 1, issue #159)",
        ),
    )
}


def resolve_subjects(scope: str) -> tuple[SubjectSpec, ...]:
    if scope == "all":
        return tuple(SUBJECTS.values())
    return tuple(spec for spec in SUBJECTS.values() if spec.subject == scope)
