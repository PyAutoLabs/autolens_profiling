"""OFFLINE recovery of a MultiStart* results JSON from a crashed arm's own
``search_internal.dill``.

The failure this exists for: an arm completes every step on the GPU and then
crashes inside PyAutoFit's post-fit ``save_results`` — ``Result.instance`` ->
``SamplesSummary`` raises ``SamplesException`` when the best point's
``ell_comps`` lie outside the unit disk (PyAutoFit#1487, still open). The
search's answer is on disk in ``search_internal.dill``; only the driver's
write step is missing. Six Phase 8B arms of RAL jobs 341874/341875 were
recovered this way on 2026-08-27 (issue #182), and this is the tool that did
it, generalised from ``recover_phase8b.py``.

Nothing here runs a search or touches a GPU. Per arm it reads the harvested
PyAutoFit output directory (``model.json`` / ``search.json`` /
``settings.json`` / ``search_internal.dill``) plus the SLURM log, rebuilds
``Samples`` via ``MultiStartProdigy.samples_via_internal_from``, then feeds
the SAME driver helpers the leaf script would have used
(``searches._metrics.collect_metrics``, ``searches._per_lane.per_lane_block``,
``searches._runner._build_summary``). Every recovered file carries a
``recovered_offline`` marker block naming what is first-hand and what is not.

WHAT IS AND IS NOT GENERAL
--------------------------
Paths, the arm list, the device block and the output layout are CLI inputs.
Two things are NOT derivable from the harvested artifacts and must be supplied:

- **The ``SEARCHES_*`` environment the submit script exported.** It is what
  ``_samplers.multi_start_settings()`` reads to rebuild the recorded
  ``sampler_config`` block. Guessing it would silently write a WRONG
  ``sampler_config`` onto otherwise-sound data — so it is required
  (``--search-env KEY=VALUE``, repeatable), not defaulted.
- **The device block.** A crashed run never reached ``device_info_dict()``.
  Carrying it over from a successful sibling of the SAME campaign and submit
  script is legitimate and is what ``--device-json`` is for; inventing one is
  not. Without it the device fields are recorded as unavailable.

``--preset phase8b`` supplies both for the 2026-08-27 Phase 8B bijector A/B,
along with that campaign's arm-label and filename conventions, so the six
recovered rows in ``results/searches/multi_start_prodigy/imaging/*/hst/
phase8b/`` are reproducible from this repo.

Usage (from the ``autolens_profiling/`` root)::

    python3 scripts/misc/searches/recover_multistart_from_internal.py \
        --preset phase8b \
        --dill-root <harvest>/output_dills \
        --log-dir  <harvest>/hpc/batch_gpu/output \
        --out-root results/searches/multi_start_prodigy/imaging \
        --arm n16_s3000_seed1_bij_logit/knn_auto_logit_seed1:341874_31

Importable: :func:`recover` takes a :class:`RecoverySpec` and one
``(arm_rel, slurm_task)`` pair and returns the recovered payload, so a future
campaign can drive it from its own script without this module's CLI.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import types
from pathlib import Path


def _profiling_root() -> Path:
    for _p in Path(__file__).resolve().parents:
        if (_p / "ruff.toml").exists():
            return _p
    raise RuntimeError("autolens_profiling root (ruff.toml) not found")


REPO = _profiling_root()

# CPU JAX: this module rebuilds Samples and re-runs the driver's arithmetic.
# It never evaluates a likelihood, so a GPU would buy nothing and a GPU-only
# import failure would turn a recovery into a second outage.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

for _p in (str(REPO), str(REPO / "scripts" / "misc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("AUTOLENS_PROFILING_SMOKE") == "1":
    print(f"[smoke] {__file__}: imports + module setup OK; exiting.")
    raise SystemExit(0)

import argparse  # noqa: E402
import tempfile  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import dill  # noqa: E402
import numpy as np  # noqa: E402
from autonerves import conf  # noqa: E402

# `samples_via_internal_from` touches `self.timer.time`, which reads
# `<output_path>/.time`. Pin PyAutoFit's output_path at a throwaway directory
# so a recovery can never write into the repo's own `output/`.
_AUTOFIT_SCRATCH = Path(tempfile.mkdtemp(prefix="recover_multistart_"))
conf.instance.output_path = str(_AUTOFIT_SCRATCH)

import autofit as af  # noqa: E402
from searches import _runner, _samplers  # noqa: E402
from searches._metrics import collect_metrics  # noqa: E402
from searches._per_lane import _ell_comps_pairs, _magnitudes, per_lane_block  # noqa: E402

# ---------------------------------------------------------------------------
# Presets — the per-campaign facts the harvested artifacts do NOT carry.
# ---------------------------------------------------------------------------
#
# A preset is not a convenience: `sampler_config` is rebuilt by
# `_samplers.multi_start_settings()` reading the SEARCHES_* environment, and
# the device block cannot be reconstructed from a crashed run at all. Both
# have to come from the campaign that ran the arm — its submit script and its
# successful sibling arms — or not at all.

# 2026-08-27 Phase 8B bijector A/B (RAL jobs 341874 / 341875, submit
# `hpc/batch_gpu/submit_phase8b_bijector_a100`). The SEARCHES_* values below
# are that submit script's exports, verbatim.
PRESET_PHASE8B = {
    "search_env": {
        "SEARCHES_CLIPPER": "prior_box",
        "SEARCHES_SCALER": "none",
        "SEARCHES_LANE_HISTORY": "1",
        "SEARCHES_DISABLE_VIZ": "1",
        "SEARCHES_TRACE_PARAMS": (
            "galaxies.source.pixelization.regularization.inner_coefficient,"
            "galaxies.source.pixelization.regularization.outer_coefficient"
        ),
    },
    # Carried over from the SUCCESSFUL arms of the same campaign+submit, all of
    # which agree exactly. Only the instantaneous nvidia-smi memory reading is
    # genuinely unrecoverable, and it says so in place of a number.
    "device": {
        "backend": "gpu",
        "device": "cuda:0",
        "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0",
        "omp_num_threads": None,
        "cpu_count": 124,
        "nvidia_smi": "NVIDIA A100 80GB PCIe, <memory.used unrecorded>, 81920 MiB",
    },
    "out_subdir": "phase8b",
    "wall_bracket_note": (
        'total_wall_s = t("Fit Running: Updating results") - '
        't("Starting non-linear search with JAX"). Calibrated on the successful sibling '
        'knn_auto_log_reg_seed3: the same bracket closed at "Search complete, returning '
        'result" reproduces its recorded total_wall_s to 0.003 s. A crashed arm never logs '
        "that line, so this value UNDERSTATES the true search.fit() wall by the post-fit "
        "result-update tail (~75 s in the sibling)."
    ),
}

PRESETS = {"phase8b": PRESET_PHASE8B}


@dataclass
class RecoverySpec:
    """Everything one recovery run needs that is not inside the arm's own files."""

    dill_root: Path
    log_dir: Path
    out_root: Path
    search_env: dict[str, str]
    device: dict | None = None
    out_subdir: str | None = None
    instrument_default: str = "hst"
    wall_bracket_note: str = (
        'total_wall_s = t("Fit Running: Updating results") - '
        't("Starting non-linear search with JAX"). A crashed arm never logs "Search '
        'complete, returning result", so this UNDERSTATES the true search.fit() wall by '
        "the post-fit result-update tail."
    )
    arms: list[tuple[str, str]] = field(default_factory=list)


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _stamp(line: str):
    m = _TS.match(line)
    if not m:
        return None
    return _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def slurm_facts(task: str, spec: RecoverySpec) -> dict:
    """Wall-clock bracket + the final logged step line, from the SLURM log.

    Calibrated against the successful sibling arm knn_auto_log_reg_seed3
    (output.341874_28.out): t("Search complete, returning result") -
    t("Starting non-linear search with JAX") = 6560.657 s vs the driver's
    recorded total_wall_s = 6560.65927 s -- i.e. this bracket IS `search.fit()`.
    A crashed arm never logs "Search complete", so the bracket closes at
    "Fit Running: Updating results", which understates the true `fit()` wall
    by the post-fit result-update tail (~75 s in that sibling).
    """
    log_path = spec.log_dir / f"output.{task}.out"
    text = log_path.read_text(errors="replace").splitlines()
    t0 = t_complete = t_update = None
    last_step = None
    for line in text:
        if t0 is None and "Starting non-linear search with JAX" in line:
            t0 = _stamp(line)
        if "MultiStartGradient sampling complete" in line:
            t_complete = _stamp(line)
        if "Fit Running: Updating results" in line:
            t_update = _stamp(line)
        if "3000/3000" in line:
            last_step = line.strip()
    return {
        "fit_start": t0,
        "sampling_complete": t_complete,
        "updating_results": t_update,
        "total_wall_s": (t_update - t0).total_seconds() if (t0 and t_update) else None,
        "sampling_wall_s": (t_complete - t0).total_seconds() if (t0 and t_complete) else None,
        "final_step_line": last_step,
        # Recorded as the SLURM job's own relative log path, never this
        # machine's harvest-mirror location — an absolute path into a
        # transient mirror is a dead reference the moment the mirror goes.
        "log": f"hpc/batch_gpu/output/output.{task}.out",
        "log_path_local": log_path,
    }


def recover(arm_rel: str, task: str, spec: RecoverySpec) -> dict:
    arm_dir = spec.dill_root / arm_rel
    (ident,) = [p for p in arm_dir.iterdir() if p.is_dir()]
    files = ident / "files"

    search_json = json.loads((files / "search.json").read_text())
    settings_json = json.loads((files / "settings.json").read_text())
    config_name = search_json["arguments"]["name"]
    prefix = search_json["arguments"]["path_prefix"][
        "path"
    ]  # searches/<sampler>/<dc>/<cell>/<inst>
    _, sampler, dataset_class, model_type, instrument = prefix.split("/")
    log_det_method = settings_json["arguments"]["log_det_method"]
    seed = search_json["arguments"]["seed"]
    bijector_kind = (
        arm_rel.rsplit("/", 1)[1].removeprefix(f"{model_type}_").removeprefix(f"{log_det_method}_")
    )
    bijector = bijector_kind.rsplit("_seed", 1)[0]
    if config_name.startswith("knn_auto_"):
        bijector = config_name[len("knn_auto_") :].rsplit("_seed", 1)[0]

    model = af.from_json(str(files / "model.json"))
    search = af.from_json(str(files / "search.json"))
    # `samples_via_internal_from` touches `self.timer.time`, which reads
    # `<output_path>/.time`; `conf.instance.output_path` is pinned to this
    # scratchpad above so nothing is written into the repo. The value only
    # lands in samples_info, which the results JSON does not use.
    with open(files / "search_internal" / "search_internal.dill", "rb") as f:
        si = dill.load(f)

    samples = search.samples_via_internal_from(model=model, search_internal=si)

    facts = slurm_facts(task, spec)

    metrics = collect_metrics(
        result=types.SimpleNamespace(samples=samples),
        total_wall_s=facts["total_wall_s"],
        viz_wall_s=0.0,
        is_multi_start=True,
        n_starts=int(search.n_starts),
        multi_start_total_steps=int(si["total_steps"]),
    )

    diagnostics = per_lane_block(captured=si, model=model, n_starts=int(search.n_starts))

    # The best point itself. `Samples.max_log_likelihood()` silently falls back
    # to the highest-likelihood VALID stored sample when the true best cannot be
    # rebuilt -- which is exactly the #1487 condition these arms crashed on. So
    # try the true best point FIRST and record honestly which one we got.
    best_instance = None
    best_fit_fallback = None
    try:
        best_instance = model.instance_from_vector(vector=[float(v) for v in si["best_params"]])
        best_fit = _runner.format_best_fit(best_instance)
        best_fit_is_best_point = True
    except Exception as exc:  # noqa: BLE001 -- mirrors _runner.py's handler
        best_fit = f"(unavailable: {exc!r})"
        best_fit_is_best_point = False
        try:
            best_fit_fallback = _runner.format_best_fit(samples.max_log_likelihood())
        except Exception:  # noqa: BLE001
            best_fit_fallback = None

    recovery_block = None
    if best_instance is not None:
        try:
            recovery_block = _runner._recovery_for_cell(dataset_class, instrument, best_instance)
        except Exception:  # noqa: BLE001
            recovery_block = None

    # SEARCHES_* env exactly as submit_phase8b_bijector_a100 exported it, so
    # `multi_start_settings()` rebuilds the recorded sampler_config block.
    # Derived from the arm's own search.json where it is recorded there; the
    # rest comes from the campaign's submit script via `spec.search_env`,
    # which is REQUIRED rather than guessed (see the module docstring).
    env = {
        "SEARCHES_N_STARTS": str(search.n_starts),
        "SEARCHES_N_STEPS": str(search.n_steps),
        "SEARCHES_BATCH_SIZE": str(search.batch_size),
        "SEARCHES_SEED": str(seed),
        "SEARCHES_BIJECTOR": bijector,
        **spec.search_env,
    }
    os.environ.update(env)
    # `_trace_param_indices` re-probes a fresh cell model; the indices that
    # ACTUALLY ran are recorded in search.json, so use those (ground truth).
    trace_idx = list(search_json["arguments"]["trace_param_indices"]["values"])
    _samplers._trace_param_indices = lambda *_a, **_k: trace_idx

    cli = _runner.dataclasses.replace(_make_cli(config_name, instrument), config_name=config_name)

    # The crashed runs never called device_info_dict()/resolve_log_det_method()
    # at write time; pin both to what the run actually used. With no
    # `--device-json` / preset device the fields are recorded as unavailable
    # rather than filled in with this CPU machine's own hardware, which would
    # be a plain falsehood about where the numbers came from.
    device_block = (
        dict(spec.device)
        if spec.device
        else {
            "backend": None,
            "device": None,
            "xla_flags": None,
            "omp_num_threads": None,
            "cpu_count": None,
            "nvidia_smi": "<unrecorded: crashed before device_info_dict(); no sibling supplied>",
        }
    )
    orig_device, orig_ldm = _runner.device_info_dict, _runner.resolve_log_det_method
    _runner.device_info_dict = lambda: dict(device_block)
    _runner.resolve_log_det_method = lambda **_k: log_det_method
    try:
        summary = _runner._build_summary(
            sampler=sampler,
            dataset_class=dataset_class,
            model_type=model_type,
            instrument=instrument,
            config_name=config_name,
            cli=cli,
            use_jax=True,
            model=model,
            n_live=None,
            metrics=metrics,
            viz_n_calls=0,
            best_fit=best_fit,
            recovery=recovery_block,
            viz_disabled=True,
            truth_anchor=None,
            posterior_stats=None,
            diagnostics=diagnostics,
            target_override=None,
        )
    finally:
        _runner.device_info_dict, _runner.resolve_log_det_method = orig_device, orig_ldm

    summary["performance"]["wall_source"] = "slurm_log"
    summary["model_summary"]["best_fit_is_best_point"] = best_fit_is_best_point
    if not best_fit_is_best_point:
        summary["model_summary"]["best_fit_fallback"] = best_fit_fallback
        summary["model_summary"]["best_fit_note"] = (
            "The maximum-log-posterior point CANNOT be turned into a model instance -- its "
            "ell_comps lie outside the unit disk, which is the very PyAutoFit#1487 condition "
            "that crashed this arm's save_results. `best_fit` therefore carries the exception, "
            "and `best_fit_fallback` is `Samples.max_log_likelihood()`'s silent fallback: the "
            "highest-likelihood VALID stored sample, i.e. some lane's FINAL point, NOT the best "
            "point. `results.max_log_likelihood` is unaffected -- it reads the best point's own "
            "log likelihood off the sample list and needs no instance."
        )

    # ---- the offline marker, first key in the file.
    names = list(model.model_component_and_parameter_names)
    pairs = _ell_comps_pairs(names)
    best_mag = _magnitudes(np.asarray(si["best_params"]), pairs)

    marker = {
        "recovered_offline": True,
        "recovered_offline_note": (
            "NOT driver output. This results JSON was reconstructed OFFLINE on a CPU from the "
            "arm's harvested PyAutoFit output directory (model.json / search.json / settings.json "
            "/ search_internal.dill) plus its SLURM log, after the A100 run completed all 3000 "
            "prodigy steps but crashed in PyAutoFit's post-fit save_results "
            "(Result.instance -> SamplesSummary raises SamplesException when the best point's "
            "ell_comps lie outside the unit disk; PyAutoFit#1487). Every `results`, "
            "`diagnostics` and `performance.likelihood_evals`/`gradient_evals`/`stored_samples` "
            "number is derived from the run's own search_internal via the same "
            "samples_via_internal_from / collect_metrics / per_lane_block / _build_summary path "
            "the driver uses, so it is the driver's arithmetic on the driver's data. See "
            "recovered_offline_provenance for the fields that are NOT first-hand."
        ),
        "recovered_offline_provenance": {
            "recovered_on": _dt.datetime.now().isoformat(timespec="seconds"),
            "recovery_script": "scripts/misc/searches/recover_multistart_from_internal.py",
            "slurm_task": task,
            # Paths are recorded RELATIVE to the run's own output tree and the
            # SLURM job, never as this machine's harvest-mirror location: a
            # mirror path is a dead absolute reference the moment the mirror is
            # cleaned up, and the repo forbids machine-specific absolute paths
            # in tracked files.
            "slurm_log": facts["log"],
            "search_internal_dill": (
                f"{arm_rel}/{ident.name}/files/search_internal/search_internal.dill"
            ),
            "model_json": f"{arm_rel}/{ident.name}/files/model.json",
            "search_json": f"{arm_rel}/{ident.name}/files/search.json",
            "settings_json": f"{arm_rel}/{ident.name}/files/settings.json",
            "source_tree": (
                "paths above are relative to the RAL run's own PyAutoFit output root "
                "(<output_path>/searches/...); the harvest mirror they were read from is "
                "transient and deliberately not recorded"
            ),
            "identifier": ident.name,
            "unique_tag": search_json["arguments"]["unique_tag"],
            "wall_source": "slurm_log",
            "wall_bracket": spec.wall_bracket_note,
            "sampling_wall_s": facts["sampling_wall_s"],
            "final_step_log_line": facts["final_step_line"],
            "unavailable_fields": [
                "results.truth_log_likelihood / results.delta_max_ll_vs_truth / results.bar_source "
                "(truth anchor omitted: it needs a real truth-tracer likelihood evaluation, which "
                "is a GPU/search-class computation this offline recovery is not permitted to run)",
                "device.nvidia_smi memory.used (never sampled; the GPU name and total memory come "
                "from the job's own PREFLIGHT nvidia-smi banner)",
                "device.backend / device.device / device.xla_flags / device.omp_num_threads / "
                "device.cpu_count (carried over from the successful arms of the SAME campaign and "
                "submit script, all of which agree exactly)",
                "performance.viz_wall_s / viz_n_calls (SEARCHES_DISABLE_VIZ=1, so 0.0 / 0 is what "
                "the driver would have written)",
                "recovery block (truth-recovery report; omitted whenever the best-fit instance "
                "cannot be built -- which is the very crash being recovered from)",
            ],
        },
        "recovered_offline_verification": {
            "log_best_log_posterior": _log_best_log_post(facts["final_step_line"]),
            "search_internal_best_log_posterior": -0.5 * float(si["best_fom"]),
            "best_fom": float(si["best_fom"]),
            "best_point_ell_comps_magnitude": best_mag,
        },
    }
    out = {**marker, **summary}

    json_name = f"{config_name.split('_', 1)[0]}_{config_name}.json"
    out_dir = spec.out_root / model_type / instrument
    if spec.out_subdir:
        out_dir = out_dir / spec.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / json_name).write_text(json.dumps(out, indent=2))

    return {
        "path": out_dir / json_name,
        "config_name": config_name,
        "cell": model_type,
        "log_det_method": log_det_method,
        "bijector": bijector,
        "seed": seed,
        "summary": out,
        "si": si,
        "facts": facts,
        "best_mag": best_mag,
    }


def _log_best_log_post(line: str | None):
    if not line:
        return None
    m = re.search(r"best log_post ([-\d.eE+]+)", line)
    return float(m.group(1)) if m else None


def _make_cli(config_name: str, instrument: str):
    from _profile_cli import ProfileCLI

    return ProfileCLI(
        config_name=config_name,
        output_dir=None,
        use_mixed_precision=False,
        instrument=instrument,
        vmap_probe=False,
        use_sparse_operator=False,
        rect_mesh="bilinear",
    )


def build_spec(args) -> RecoverySpec:
    preset = PRESETS[args.preset] if args.preset else {}
    search_env = dict(preset.get("search_env", {}))
    for item in args.search_env or []:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--search-env expects KEY=VALUE, got {item!r}")
        search_env[key] = value
    if not search_env:
        raise SystemExit(
            "no SEARCHES_* environment supplied. sampler_config is rebuilt by reading it, "
            "so recovering without it writes a WRONG sampler_config onto sound data. Pass "
            "--preset <name> or one or more --search-env KEY=VALUE."
        )
    device = preset.get("device")
    if args.device_json:
        device = json.loads(Path(args.device_json).read_text())

    arms = []
    for item in args.arm:
        arm_rel, _, task = item.rpartition(":")
        if not arm_rel:
            raise SystemExit(f"--arm expects <dill-subdir>:<slurm-task>, got {item!r}")
        arms.append((arm_rel, task))

    spec = RecoverySpec(
        dill_root=Path(args.dill_root),
        log_dir=Path(args.log_dir),
        out_root=Path(args.out_root),
        search_env=search_env,
        device=device,
        out_subdir=args.out_subdir or preset.get("out_subdir"),
        instrument_default=args.instrument,
        arms=arms,
    )
    if preset.get("wall_bracket_note"):
        spec.wall_bracket_note = preset["wall_bracket_note"]
    return spec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="DILL_SUBDIR:SLURM_TASK",
        help="one crashed arm, repeatable, e.g. "
        "n16_s3000_seed1_bij_logit/knn_auto_logit_seed1:341874_31",
    )
    p.add_argument("--dill-root", required=True, help="root holding each arm's output dir")
    p.add_argument("--log-dir", required=True, help="dir holding output.<task>.out SLURM logs")
    p.add_argument("--out-root", required=True, help="results/searches/<sampler>/<dataset_class>")
    p.add_argument("--out-subdir", default=None, help="campaign subdir under <model>/<instrument>")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument(
        "--search-env",
        action="append",
        metavar="KEY=VALUE",
        help="a SEARCHES_* export from the campaign's submit script (repeatable); "
        "overrides the preset's value for the same key",
    )
    p.add_argument("--device-json", default=None, help="device block from a successful sibling")
    p.add_argument("--instrument", default="hst")
    p.add_argument("--table-json", default=None, help="write a per-arm summary table here")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    spec = build_spec(args)

    rows = []
    for arm_rel, task in spec.arms:
        print(f"--- {arm_rel}  (slurm {task})", flush=True)
        rows.append(recover(arm_rel, task, spec))
        print(f"    wrote {rows[-1]['path']}", flush=True)

    table = []
    for r in rows:
        s_, si = r["summary"], r["si"]
        d = s_["diagnostics"]
        foms = sorted(
            [ln["lane_best_fom"] for ln in d["per_lane"] if ln["lane_best_fom"] is not None]
        )
        table.append(
            {
                "cell": r["cell"],
                "log_det_method": r["log_det_method"],
                "bijector": r["bijector"],
                "seed": r["seed"],
                "config_name": r["config_name"],
                "json": str(r["path"]),
                "best_fom": float(si["best_fom"]),
                "best_log_posterior": -0.5 * float(si["best_fom"]),
                "log_best_log_posterior": _log_best_log_post(r["facts"]["final_step_line"]),
                "max_log_likelihood": s_["results"]["max_log_likelihood"],
                "total_steps": d["counters"]["total_steps"],
                "stop_reason": d["counters"]["stop_reason"],
                "alive_final": d["alive_history"][-1],
                "alive_min": min(d["alive_history"]),
                "n_constrained_lane_steps": d["counters"]["n_constrained_lane_steps"],
                "n_clipped_lane_steps": d["counters"]["n_clipped_lane_steps"],
                "n_value_nan_lane_steps": d["counters"]["n_value_nan_lane_steps"],
                "n_grad_nan_lane_steps": d["counters"]["n_grad_nan_lane_steps"],
                "n_resurrections": d["counters"]["n_resurrections"],
                "lane_best_foms_sorted": foms,
                "lane_best_log_posteriors_sorted": sorted(
                    [
                        ln["lane_best_log_posterior"]
                        for ln in d["per_lane"]
                        if ln["lane_best_log_posterior"] is not None
                    ],
                    reverse=True,
                ),
                "best_point_ell_comps_magnitude": r["best_mag"],
                "total_wall_s": s_["performance"]["total_wall_s"],
                "sampling_wall_s": r["facts"]["sampling_wall_s"],
                "target_id": s_["target"]["target_id"],
                "best_fit": s_["model_summary"]["best_fit"],
            }
        )
    if args.table_json:
        Path(args.table_json).write_text(json.dumps(table, indent=2))
        print("\nwrote " + args.table_json)


if __name__ == "__main__":
    main()
