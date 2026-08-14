"""Human-readable markdown and PNG rendering for hazard findings."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._record import Finding


def _finish_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_finding_plot(finding: Finding, path: Path) -> None:
    """Render an at-a-glance plot from the finding's reproducer payload."""

    data = finding.reproducer
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if finding.hazard_class == "saturation":
        x = np.asarray(data["parameter_values"], dtype=float)
        for backend, values in data["axis_ratio"].items():
            ax.plot(x, values, marker="o", label=backend)
        plateau = data["detected_plateau"]
        ax.axhline(plateau["value"], color="black", linestyle=":", alpha=0.6)
        ax.set_xlabel(data["parameter"])
        ax.set_ylabel("axis ratio q")
        ax.legend()
    elif finding.hazard_class == "nonfinite_gradient":
        radius = np.asarray(data["radius"], dtype=float)
        gradient = np.asarray(
            [np.nan if value is None else value for value in data["gradient_norm"]],
            dtype=float,
        )
        ax.plot(radius, gradient, marker="o")
        finite_gradient = gradient[np.isfinite(gradient)]
        marker_height = float(np.max(finite_gradient)) if finite_gradient.size else 1.0
        ax.scatter(
            [0.0],
            [marker_height],
            marker="x",
            s=80,
            color="red",
            label="gradient non-finite",
        )
        ax.set_xscale("symlog", linthresh=1.0e-13)
        ax.set_xlabel("radius")
        ax.set_ylabel("gradient norm")
        if data.get("parameter") == "ell_comps_radius":
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend()
    elif finding.hazard_class == "backend_divergence":
        ax.plot(data["factor"], data["relative_error"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("factor = (1-q)/(1+q)")
        ax.set_ylabel("relative deflection error")
    elif finding.hazard_class == "conditioning_floor" and "matrix_scale" in data:
        scale = np.asarray(data["matrix_scale"], dtype=float)
        for backend, curves in data["backends"].items():
            ax.plot(
                scale,
                curves["floor_over_matrix_scale"],
                marker="o",
                label=backend,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("synthetic matrix scale")
        ax.set_ylabel("floor / matrix scale")
        ax.legend()
    elif finding.hazard_class == "conditioning_floor" and "noise_scale" in data:
        noise_scale = np.asarray(data["noise_scale"], dtype=float)
        ax.plot(
            noise_scale,
            data["curvature_floor_fraction"],
            marker="o",
            label="absolute curvature floor",
        )
        if "scale_aware_curvature_floor_fraction" in data:
            ax.plot(
                noise_scale,
                data["scale_aware_curvature_floor_fraction"],
                marker="o",
                label="scale-aware curvature floor",
            )
        ax.plot(
            noise_scale,
            data["regularization_jitter_fraction"],
            marker="o",
            label="regularization jitter",
        )
        ax.set_yscale("log")
        ax.set_xlabel("noise-map scale")
        ax.set_ylabel("absolute floor / matrix scale")
        ax.legend()
    elif finding.hazard_class == "active_set":
        parameter = np.asarray(data["parameter"], dtype=float)
        ax.plot(parameter, np.zeros_like(parameter), color="0.7")
        for index, location in enumerate(data["transition_locations"]):
            ax.axvline(
                location,
                color="tab:red",
                label="support transition" if index == 0 else None,
            )
        ax.set_xlabel("Einstein radius")
        ax.set_yticks(())
        ax.legend()
    elif finding.hazard_class == "solver_divergence":
        for backend, curve in data["curves"].items():
            ax.plot(
                curve["parameter"],
                curve["figure_of_merit"],
                marker="o",
                label=f"{backend} figure of merit",
            )
            ax.plot(
                curve["parameter"],
                curve["reconstruction"],
                marker=".",
                linestyle="--",
                label=f"{backend} reconstruction",
            )
        for policy, value in data.get("same_system_reconstruction_error_max", {}).items():
            ax.axhline(
                value,
                linestyle=":",
                label=f"same system: {policy}",
            )
        ax.set_yscale("log")
        ax.set_xlabel(data["parameter"])
        ax.set_ylabel("relative error vs NumPy")
        ax.legend()
    elif finding.hazard_class == "structural_degeneracy":
        spans = data["axis_ratio_to_figure_of_merit_span"]
        axis_ratio = np.asarray(sorted(float(value) for value in spans), dtype=float)
        ax.plot(axis_ratio, [spans[value] for value in axis_ratio], marker="o")
        ax.set_xlabel("axis ratio")
        ax.set_ylabel("figure-of-merit span over orientation")
    else:
        ax.text(0.5, 0.5, finding.summary, ha="center", va="center", wrap=True)
        ax.set_axis_off()
    ax.set_title(finding.title)
    ax.grid(True, linestyle=":", alpha=0.35)
    _finish_figure(fig, path)


def write_finding_plots(findings: list[Finding], output_root: Path) -> list[Path]:
    written = []
    for finding in findings:
        path = (output_root / finding.record_relative_path).with_suffix(".png")
        render_finding_plot(finding, path)
        written.append(path)
    return written


def _format_percent(value: float) -> str:
    percent = 100.0 * value
    return f"{percent:.2f}%" if percent < 1.0 else f"{percent:.1f}%"


def write_ell_comps_seed_summary(finding: Finding, path: Path) -> None:
    """Generate the worked ell_comps result from the detector record."""

    prior_rows = []
    for measurement in finding.measurements:
        if measurement.basis != "prior_mass" or measurement.value is None:
            continue
        prior = measurement.details["prior"]
        if prior["type"] == "independent_truncated_gaussian":
            label = f"TruncatedGaussian(0, {prior['sigma']}) per component"
        else:
            label = "Uniform(-1, 1) per component"
        low, high = measurement.details["confidence_interval"]
        prior_rows.append(
            f"| {label} | {_format_percent(measurement.value)} | "
            f"[{_format_percent(low)}, {_format_percent(high)}] | "
            f"{measurement.details['sample_count']:,} |"
        )

    axis_ratio = finding.reproducer["detected_plateau"]["value"]
    anchor_lines = []
    for anchor in finding.anchors:
        identity = anchor.symbol or anchor.config_key or "expression"
        anchor_lines.append(
            f"- `PyAutoGalaxy/{anchor.path}:{anchor.start_line}` at commit "
            f"`{anchor.commit[:12]}`\n  (`{identity}`; token fingerprint "
            f"`{anchor.token_fingerprint[:16]}…`)."
        )
    if not anchor_lines:
        anchor_lines.append("- No source anchor located; the reproducer remains authoritative.")
    text = f"""# ell_comps clamp — generated seed finding

Semantic finding ID: `{finding.finding_id}`

`autogalaxy.convert` clamps the ellipticity magnitude at `0.999`, so the
axis ratio saturates at `q = {axis_ratio:.15g}`. The value stays finite; the
hazard is a flat, zero-gradient plateau rather than a NaN rejection.

## Reachability

- NumPy construction reaches the narrow `0.999 <= |ell_comps| < 1` annulus.
- `validate_ell_comps` blocks the scientifically invalid `|ell_comps| >= 1`
  region for concrete NumPy/Python scalars.
- JAX scalar arrays and tracers are not concrete to that guard, so construction
  and tracing both reach the beyond-unit plateau.

That region split matters: `code_exists`, `reachable_via`, `blocked_by`, and
`affects_science` are separate fields in the machine-readable record.

## Prior mass beyond the unit circle

Deterministic Monte Carlo seed `{finding.measurements[0].details["seed"]}`;
Wilson 95% intervals are reported rather than a bare percentage.

| Independent prior | Estimate | 95% interval | Samples |
|---|---:|---:|---:|
{chr(10).join(prior_rows)}

These reproduce the established rounded results: 0.22%, 5.1%, and 21.4%.

## Code anchors

{chr(10).join(anchor_lines)}

The semantic finding ID is stable. Anchors help locate the implementation, but
persistence is decided by re-running the reproducer (`scan.py --check`), not by
treating a source hash as the finding's identity.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
