"""Explicit legacy formats -> rows and metrics, never scientific imports.

Every metric retains its exact JSON pointer. Unknown/custom experiments remain
in the inventory instead of guessing a number's meaning from its magnitude.
"""

from __future__ import annotations

import math


def pointer(base, key):
    return base + "/" + str(key).replace("~", "~0").replace("/", "~1")


def number(value):
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def rows(doc, adapter):
    """Yield (payload, JSON pointer, row-specific scientific settings)."""
    if adapter == "compile":
        if isinstance(doc, list):
            for i, row in enumerate(doc):
                if isinstance(row, dict):
                    yield row, pointer("", i), {}
        return
    if not isinstance(doc, dict):
        return
    if adapter == "streaming":
        # Rows inherit only the actual enclosing run metadata, never another arm.
        groups = [("", doc)] + [
            (pointer("/arms", k), v) for k, v in doc.get("arms", {}).items() if isinstance(v, dict)
        ]
        for base, group in groups:
            for i, row in enumerate(group.get("rows", [])):
                if not isinstance(row, dict):
                    continue
                merged = {**doc, **group, **row}
                context = {
                    k: row[k]
                    for k in (
                        "mode",
                        "n_vis",
                        "chunk",
                        "nufft_chunk_size",
                        "pixels_in_mask",
                        "cap_gb",
                        "timeout_s",
                    )
                    if k in row
                }
                context["arm"] = base or "default"
                yield merged, pointer(pointer(base, "rows"), i), context
        return
    if adapter != "profile":
        return
    if isinstance(doc.get("configs"), dict):
        for key, payload in doc["configs"].items():
            if isinstance(payload, dict):
                yield (
                    {**payload, "_config_label": key},
                    pointer("/configs", key),
                    {"sweep_config": key},
                )
    else:
        yield doc, "", {}


# Full likelihood measurements and component totals deliberately have separate axes.
DIRECT = {
    "full_pipeline_per_call": ("runtime", "single_call"),
    "full_pipeline_single_jit": ("runtime", "single_jit_block"),
    "full_pipeline_single_jit_median": ("runtime", "single_jit_median"),
    "full_pipeline_cube_single_jit": ("runtime", "cube_single_jit"),
    "direct_log_likelihood_function_per_call": ("runtime", "direct_call"),
    "total_step_by_step": ("breakdown", "component_total"),
    "total_step_by_step_cube": ("breakdown", "cube_component_total"),
    "operator_build_s": ("compile", "operator_setup"),
    "full_pipeline_lower_s": ("compile", "full_pipeline.lower"),
    "full_pipeline_compile_s": ("compile", "full_pipeline.compile"),
    "full_pipeline_first_call_s": ("compile", "full_pipeline.first_call"),
    "regularization_matrix_prefix_s": ("breakdown", "regularization_matrix_prefix"),
    "interpolator_prefix_s": ("breakdown", "interpolator_prefix"),
    "regularization_matrix_prefix_vmap_per_call_s": (
        "breakdown",
        "regularization_matrix_prefix_vmap_per_call",
    ),
    "interpolator_prefix_vmap_per_call_s": ("breakdown", "interpolator_prefix_vmap_per_call"),
    "setup_prefix_per_call_s": ("breakdown", "setup_prefix"),
}
COMPILE = {
    "trace_s": "trace",
    "compile_s": "compile",
    "first_s": "first_call",
    "steady_s": "steady_call",
}


def metrics(row, adapter, base):
    """Yield axis, metric, unit, value, exact pointer, statistic, repetitions."""

    def emit(axis, metric, unit, value, path, statistic=None, repetitions=None):
        # Keep bad values in the candidate stream so the builder records refusals.
        if isinstance(value, (int, float)) or value is None:
            return [(axis, metric, unit, value, path, statistic, repetitions)]
        return []

    if adapter == "streaming":
        if row.get("outcome") != "OK":
            return
        for key, axis, metric, unit in [
            ("wall_s", "compile", "dataset_setup_wall", "s"),
            ("peak_rss_mb", "memory", "host_peak_rss", "MiB"),
        ]:
            if key in row:
                yield from emit(
                    axis, metric, unit, row[key], pointer(base, key), "single observation", 1
                )
        return
    if adapter == "compile":
        for key, metric in COMPILE.items():
            if key in row:
                # A transformed/batched compile probe's steady cost is not a
                # standalone full likelihood. The transform stays in its setup.
                axis = "breakdown" if key == "steady_s" else "compile"
                yield from emit(axis, metric, "s", row[key], pointer(base, key))
        return
    for key, (axis, metric) in DIRECT.items():
        if key in row:
            yield from emit(
                axis,
                metric,
                "s",
                row[key],
                pointer(base, key),
                "median" if "median" in key else None,
            )
    for group, axis in [
        ("full_call", "runtime"),
        ("full_pipeline", "runtime"),
        ("vmap", "runtime"),
    ]:
        block = row.get(group)
        if not isinstance(block, dict) or str(block.get("status", "ok")).lower() not in (
            "ok",
            "success",
        ):
            continue
        fields = (
            ("per_call_s", "mean_s", "median_s", "min_s", "max_s")
            if group != "vmap"
            else ("per_call", "batch_time")
        )
        for key in fields:
            if key in block:
                yield from emit(
                    axis,
                    f"{group}.{key}",
                    "s",
                    block[key],
                    pointer(pointer(base, group), key),
                    key.split("_")[0] if key in ("mean_s", "median_s", "min_s", "max_s") else None,
                    block.get("n"),
                )
    for group in ("steps", "steps_median", "dense_steps", "steps_vmap_per_call"):
        block = row.get(group)
        if isinstance(block, dict):
            for name, value in block.items():
                if isinstance(value, (int, float)) or value is None:
                    yield (
                        "breakdown",
                        f"{group}.{name}",
                        "s",
                        value,
                        pointer(pointer(base, group), name),
                        "median" if group == "steps_median" else None,
                        None,
                    )
    phases = row.get("jit_phases")
    if isinstance(phases, dict):
        for name, block in phases.items():
            if not isinstance(block, dict):
                continue
            for key in ("lower_s", "compile_s", "first_call_s", "steady_per_call_s"):
                if key in block:
                    axis = "breakdown" if key == "steady_per_call_s" else "compile"
                    yield from emit(
                        axis,
                        f"{name}.{key}",
                        "s",
                        block[key],
                        pointer(pointer(pointer(base, "jit_phases"), name), key),
                    )
    if "peak_rss_mb" in row:
        # These Linux/WSL producers divide ru_maxrss (KiB) by 1024.
        yield from emit(
            "memory",
            "host_peak_rss",
            "MiB",
            row["peak_rss_mb"],
            pointer(base, "peak_rss_mb"),
            "peak",
        )
    # Explicit component producer formats (not a likelihood headline).
    for key in ("total_s", "per_call_s"):
        if key in row:
            yield from emit("breakdown", key, "s", row[key], pointer(base, key))
