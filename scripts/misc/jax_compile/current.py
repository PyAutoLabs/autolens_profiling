"""Explicit, isolated current-runtime compile workload; no setup on import."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

OBJECTIVE = "imaging-mge-runtime-v1"
OUTPUT_NAMESPACE = "compile_probe_runtime_v1"


def validate_request(args, root):
    """Reject unsupported requests before imports, dataset setup or output writes."""
    if getattr(args, "objective", None) != OBJECTIVE:
        raise ValueError(f"Select --objective {OBJECTIVE}; the legacy posterior is unavailable")
    if (args.dataset_class, args.model_type, args.instrument) != ("imaging", "mge", "hst"):
        raise ValueError("Only imaging/mge/hst is supported by the current runtime builder")
    if args.mixed_precision:
        raise ValueError("Only fp64 is supported")
    if args.transforms.strip() != "jit":
        raise ValueError("Only --transforms jit is supported; gradients and batches are deferred")
    if args.n_batch != 16 or args.batch_size != 4:
        raise ValueError("Batch options are unused by jit; leave them at their defaults")
    if not getattr(args, "output", None):
        raise ValueError("An explicit --output is required for the new objective")
    output = Path(args.output).resolve()
    namespace = Path(root) / "output" / OUTPUT_NAMESPACE
    # Do not allow a symlinked namespace to redirect results into historical trees.
    if namespace.resolve() != namespace.absolute():
        raise ValueError("The output namespace must not be symlinked")
    if not output.is_relative_to(namespace) or output.suffix != ".json":
        raise ValueError(f"--output must be a new .json file under {namespace.relative_to(root)}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    return output


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_state():
    state = {"python": platform.python_version()}
    for name in (
        "jax",
        "jaxlib",
        "numpy",
        "autonerves",
        "autofit",
        "autoarray",
        "autogalaxy",
        "autolens",
    ):
        module = importlib.import_module(name)
        entry = {"version": str(getattr(module, "__version__", "unknown"))}
        directory = str(Path(module.__file__).parent)
        try:
            entry["git_revision"] = subprocess.check_output(
                ["git", "-C", directory, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            entry["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            entry["git_revision"] = None
        state[name] = entry
    return state


@dataclass
class Workload:
    objective: object
    params: object
    model: object
    dataset: object
    instance: object


def from_dataset(dataset):
    """Build the same model, input and analysis used by the runtime cell.

    The caller supplies an already prepared dataset (also permits tiny correctness
    fixtures). Production requests use ``build_workload`` to enforce HST/fp64.
    """
    import jax
    import jax.numpy as jnp
    from autofit.jax import register_model
    from cell_builders.imaging_mge import build_analysis, build_model

    model = build_model()
    instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    register_model(model)
    params = jax.tree_util.tree_map(jnp.asarray, instance)
    analysis = build_analysis(dataset)

    def objective(params_tree):
        return analysis.log_likelihood_function(instance=params_tree)

    return Workload(objective, params, model, dataset, instance)


def build_workload(args, root):
    validate_request(args, root)
    import autolens as al
    import jax
    from cell_builders.imaging_mge import prepare_dataset
    from simulators.imaging import INSTRUMENTS

    if not jax.config.jax_enable_x64:
        raise ValueError("fp64 requires JAX_ENABLE_X64=1; precision is never changed implicitly")
    directory = Path(args.dataset_path) if args.dataset_path else Path(root) / "dataset/imaging/hst"
    for name in ("data.fits", "noise_map.fits", "psf.fits"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing {directory / name}; this probe never auto-simulates")
    dataset = al.Imaging.from_fits(
        data_path=directory / "data.fits",
        noise_map_path=directory / "noise_map.fits",
        psf_path=directory / "psf.fits",
        pixel_scales=INSTRUMENTS["hst"]["pixel_scale"],
    )
    return from_dataset(prepare_dataset(dataset))


def provenance(args, root, workload):
    import jax
    import numpy as np
    from cell_builders import imaging_mge

    directory = Path(args.dataset_path) if args.dataset_path else Path(root) / "dataset/imaging/hst"
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(workload.params)]
    identity = {
        "objective": OBJECTIVE,
        "objective_semantics": "AnalysisImaging.log_likelihood_function; no external prior or scaling",
        "input_representation": "registered-model-instance-pytree",
        "parameter_initialization": "physical prior medians",
        "model_free_parameters": workload.model.total_free_parameters,
        "model_configuration": workload.model.dict(),
        "input_leaves": [
            {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            }
            for x in leaves
        ],
        "dataset_sha256": {
            name: digest(directory / name) for name in ("data.fits", "noise_map.fits", "psf.fits")
        },
        "image_shape": list(workload.dataset.shape_native),
        "masked_pixels": int(workload.dataset.data.shape[0]),
        "pixel_scales": list(workload.dataset.pixel_scales),
        "mask_radius": imaging_mge.MASK_RADIUS,
        "over_sampling": {
            "sub_size_list": [4, 2, 2],
            "radial_list": [0.3, 0.6],
            "centre_list": [[0.0, 0.0]],
        },
        "linear_gaussians": {"lens": 20, "source": 20},
        "sparse_operator": False,
        "precision": "fp64",
        "source_sha256": {
            "builder": digest(imaging_mge.__file__),
            "workload": digest(__file__),
            "probe": digest(Path(__file__).with_name("probe.py")),
        },
        "software": software_state(),
    }
    return {
        "workload": identity,
        "workload_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "devices": [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def write_result(output, records, metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(
            {
                "schema_version": 1,
                "record_kind": OUTPUT_NAMESPACE,
                "comparable_to_legacy_pins": False,
                "provenance": metadata,
                "records": records,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
