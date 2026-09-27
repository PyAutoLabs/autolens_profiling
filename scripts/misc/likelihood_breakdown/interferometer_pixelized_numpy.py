"""NumPy / numba harness for the interferometer pixelized breakdown cells (library dispatch).

``scripts/interferometer/likelihood_breakdown/delaunay_numba.py`` (Hilbert-1500 Delaunay)
and ``.../pixelization_numba.py`` (rectangular 39x39) are thin CLI wrappers over this
module (autolens_profiling#326, interferometer likelihood campaign 3/3, phase 1). It is the
CPU sibling of the JAX harness ``interferometer_pixelized.py`` and builds its inputs with
that module's helpers (``load_dataset``, ``build_model``, ``adapt_images_builder``, the
mesh builders), so every arm below sees the same dataset, mask, mesh, adapt image and
model as the JAX-CPU / A100 rows.

What it measures
----------------

The **library's own dispatch**, not a prototype: ``FitInterferometer(..., xp=np)`` on a
dataset with a sparse operator reaches ``inversion_interferometer_from``
(PyAutoArray ``inversion/inversion/factory.py``), whose gate
``_use_interferometer_numba`` routes a single-mapper NumPy inversion to
``InversionInterferometerSparseNumba`` (``direct_conv`` curvature kernel) when the mean
non-zeros per source column is at or below
``Settings.interferometer_numba_nnz_per_source_max`` (60.0 packaged), else to
``InversionInterferometerSparse`` (the FFT route, here on NumPy). Two arms run in one
process on identical inputs, selected only through that setting:

- ``numba``     -- the gate forced to admit (``--numba-gate``, default 1e9), so the arm is
  numba whatever the geometry; ``would_route_numba_at_default_gate`` records what the
  packaged 60.0 would have done;
- ``numpy_fft`` -- the gate set to 0 (the kill switch): ``InversionInterferometerSparse``
  with ``xp=np``.

The **JAX-CPU FFT arm** is the JAX harness itself (``delaunay.py`` / ``pixelization.py``
with ``JAX_PLATFORMS=cpu``); this JSON carries a pointer to its row under
``arms.jax_cpu_fft`` (same instrument / config / mask radius) and the log-evidence gap at
the shared prior-median instance. Each arm asserts the inversion class it asked for
before anything is timed, and the headline ``configuration.inversion_path`` is the class
name the numba arm actually built.

The steps (one fresh ``FitInterferometer`` per instance, cached properties touched in the
library's own dependency order, every access timed):

1. ``fit.inversion`` + the mapper arrays -- ray-trace, mesh, interpolation and the routing
   gate (which itself reads the mapper's ``pix_sizes``);
2. the mapping matrix ``L``; 3. the regularization matrix ``H``;
4. numba arm only: ``kernel_index_arrays``, the CSR / CSC / extent marshalling of the
   triplets the kernel takes (cached, so step 5 is the kernel alone);
5. ``F + H`` (``curvature_reg_matrix``): F by ``direct_conv``, or by the sparse triplets +
   NumPy ``rfft2`` column blocks;
6. the reconstruction (``D = Lᵀ d~`` then fnnls; memo off);
7. ``fast_chi_squared`` -- ``curvature_matrix`` and ``data_vector`` are plain properties on
   the sparse classes, so on NumPy this row **re-evaluates F and D** (counted in
   ``f_evaluations_per_figure_of_merit``; JAX's CSE merges the duplicate, NumPy cannot);
8. ``sᵀHs``; 9. log det(F + H) off the fnnls Cholesky factor; 10. log det H;
11. noise normalisation + evidence assembly.

The sum of those rows is compared with the directly-timed library
``FitInterferometer.figure_of_merit`` on a fresh fit of the same instance
(``step_sum_over_full_call``, expected in [0.9, 1.1]), and the assembled evidence must
reproduce it. Standalone sub-rows (never summed): F alone, D alone and, for the FFT arm,
the sparse triplets alone.

Protocol (the #235 imaging numba discipline)
--------------------------------------------

- threads pinned to 1 (BLAS family + ``NUMBA_NUM_THREADS``) by the wrapper before numpy
  is imported; recorded under ``configuration.thread_env``;
- the NNLS cross-evaluation memo off (``Settings`` flag + ``AUTOARRAY_NNLS_WARM_START=0``),
  recorded under ``configuration.memo_provenance``;
- an iid instance stream (``--n-instances``, unit values uniform in the central 20 % of
  every prior, seed 235); instance 0 is a discarded warm-up (numba compile), the arm order
  alternates per instance (ABBA) so host drift cancels between the arms;
- a 1500x1500 dgemm machine-speed control at the head and the tail;
- the log evidence at the prior-median instance is pinned (``pinned_expected`` /
  ``pinned_drift``, rtol 1e-6) for the fiducial mesh at the preset mask radius.

The W~ preload (``nufft_precision_operator``) is one-off per dataset and geometry, so it
is cached beside the dataset keyed by the real-space shape and mask radius
(``nufft_precision_operator_nufft_<ny>x<nx>_r<radius>.npy``, written atomically so
concurrent jobs cannot tear it) and is not a per-evaluation step.

Output
------

``results/breakdown/interferometer/<cell>_numba_breakdown_<instrument>_v<version>.{json,png}``
untagged, ``[<instrument>/]<cell>_numba_<config>.{json,png}`` with ``--config-name``
(``interferometer/`` for alma, ``interferometer/<instrument>/`` otherwise), plus
``_r<radius>`` for a non-default mask radius. One JSON holds both NumPy arms: the numba arm
is the headline (``steps`` / ``total_step_by_step``), the FFT arm is under
``arms.numpy_fft``.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import autoarray as aa
import autofit as af
import autolens as al
import numpy as np
from autoarray.fit import fit_util
from autoarray.inversion.inversion.interferometer.sparse import InversionInterferometerSparse
from autoarray.inversion.inversion.interferometer_numba import (
    inversion_interferometer_numba_util,
)
from autoarray.inversion.inversion.interferometer_numba.sparse import (
    InversionInterferometerSparseNumba,
    _numba_parallel,
)
from autoarray.inversion.mappers.abstract import Mapper

from . import interferometer_pixelized as shared
from . import timing
from .provenance import source_revisions

#: arm -> the inversion class the factory must build for it.
ARM_CLASSES = {
    "numba": "InversionInterferometerSparseNumba",
    "numpy_fft": "InversionInterferometerSparse",
}

#: A gate no geometry reaches: the numba arm is numba whatever its nnz per column.
NUMBA_GATE_FORCED = 1.0e9

#: The campaign witness bar (numba vs FFT log evidence) and the local verification bar.
AGREEMENT_BAR_NATS = 0.5
LOCAL_AGREEMENT_NATS = 1e-6

#: The step sum must cover the directly-timed library call to within this band.
STEP_SUM_BAND = (0.9, 1.1)

IID_SEED = 235
IID_UNIT_RANGE = (0.4, 0.6)

L_BUILD = "Inversion build: ray-trace + mesh + mapper (+ routing gate)"
L_MAPPING = "Mapping matrix L"
L_REG_MATRIX = "Regularization matrix H"
L_MARSHAL = "F marshalling: kernel_index_arrays (CSR / CSC / extent)"
L_CURVATURE = {
    "numba": "Curvature F + H: F by numba direct_conv",
    "numpy_fft": "Curvature F + H: F by sparse triplets + NumPy rfft2 blocks",
}
L_SOLVE = "Reconstruction: D = Lᵀ d~ + fnnls"
L_CHI2 = "Fast chi-squared (sᵀFs - 2sᵀD + dᵀN⁻¹d; F, D re-evaluated)"
L_REG_TERM = "Regularization term sᵀHs"
L_LOGDET_CREG = "Log-det (F + H) (fnnls Cholesky reuse)"
L_LOGDET_REG = "Log-det H"
L_EVIDENCE = "Noise normalisation + evidence assembly"


def add_cell_args(parser) -> None:
    """The cell-local flags both NumPy / numba cells declare (on their ``_cell_parser``)."""
    shared.add_mask_radius_arg(parser)
    parser.add_argument(
        "--arms",
        default="numba,numpy_fft",
        help="Comma-separated NumPy arms to run: numba, numpy_fft (default both). The "
        "numba arm is the headline; without it the FFT arm is.",
    )
    parser.add_argument(
        "--numba-gate",
        type=float,
        default=NUMBA_GATE_FORCED,
        help="Settings.interferometer_numba_nnz_per_source_max for the numba arm "
        "(default 1e9 = forced to admit; 60.0 is the packaged gate).",
    )
    parser.add_argument(
        "--n-sub-repeats",
        type=int,
        default=None,
        help="Repeats of each standalone sub-row (default 5; 2 above 200k visibilities).",
    )
    parser.add_argument(
        "--preload-cache",
        choices=("on", "off"),
        default="on",
        help="Cache the W~ preload beside the dataset, keyed by shape and mask radius.",
    )


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dgemm_control_s(n: int = 1500, repeats: int = 3) -> float:
    """A fixed BLAS dgemm: the machine-speed control timed at the head and tail of a run."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    a @ b
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    return (time.perf_counter() - start) / repeats


def iid_instances(model, n: int, seed: int = IID_SEED) -> list:
    """``n`` iid instances, unit values uniform in the central 20 % of every prior."""
    rng = np.random.default_rng(seed)
    low, high = IID_UNIT_RANGE
    return [
        model.instance_from_unit_vector(
            unit_vector=list(rng.uniform(low, high, size=model.prior_count))
        )
        for _ in range(n)
    ]


def memo_off_settings(gate: float):
    """``al.Settings`` for one arm: the numba gate set, the NNLS memo off at both layers."""
    env_before = os.environ.get("AUTOARRAY_NNLS_WARM_START")
    os.environ["AUTOARRAY_NNLS_WARM_START"] = "0"
    settings = al.Settings(
        nnls_warm_start_memo=False, interferometer_numba_nnz_per_source_max=float(gate)
    )
    provenance = {
        "requested": "off",
        "settings_nnls_warm_start_memo": bool(settings.nnls_warm_start_memo),
        "env_AUTOARRAY_NNLS_WARM_START": os.environ.get("AUTOARRAY_NNLS_WARM_START"),
        "env_before": env_before,
    }
    return settings, provenance


def preload_sparse_dataset(ds, batch_size: int, cache: bool, timer):
    """``(dataset_sparse, provenance)``: the NumPy sparse operator, W~ from the cache if any.

    The preload is built by the same ``method="nufft"`` builder the JAX harness uses
    (``use_jax`` is ignored under it), so the two harnesses hold the same operator.
    """
    ny, nx = ds.real_space_shape
    cache_path = (
        ds.dataset_path / f"nufft_precision_operator_nufft_{ny}x{nx}_r{float(ds.mask_radius)}.npy"
    )
    kwargs = dict(
        method="nufft",
        batch_size=batch_size,
        nufft_chunk_size=ds.transformer_chunk,
        use_jax=False,
    )
    hit = bool(cache and cache_path.exists())
    with timer.section("apply_sparse_operator"):
        if hit:
            dataset_sparse = ds.dataset.apply_sparse_operator(
                nufft_precision_operator=np.load(cache_path), **kwargs
            )
        else:
            dataset_sparse = ds.dataset.apply_sparse_operator(**kwargs)
    build_s = timer.records[-1][1]
    if cache and not hit:
        # Atomic: concurrent RAL jobs on one dataset must never read a torn file.
        tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, np.asarray(dataset_sparse.sparse_operator.nufft_precision_operator))
        os.replace(tmp, cache_path)
    return dataset_sparse, {
        "cache": "on" if cache else "off",
        "cache_hit": hit,
        "cache_path": str(cache_path),
        "operator_build_s": float(build_s),
        "note": "One-off per dataset + geometry (W~ preload + dirty image); not a "
        "per-evaluation step. A cache hit times only the dirty image and operator wrap.",
    }


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n": int(arr.size),
    }


def run(
    spec: shared.CellSpec,
    cli,
    cell_args,
    script_file: str,
    profiling_root: Path,
    thread_env: dict,
) -> None:
    """Run one NumPy / numba pixelized interferometer breakdown cell end to end."""
    import autogalaxy as ag

    from _profile_cli import check_pinned, device_info_dict, record_pinned_check

    revisions = source_revisions(profiling_root)
    print("--- Import provenance ---")
    print(f"  cell __file__ = {script_file}")
    for module in (aa, ag, al, af):
        print(f"  {module.__name__}.__file__ = {module.__file__}")
    for repo, rev in revisions.items():
        print(f"  {repo:<18} {rev}")
    import numba

    print(
        f"  numba {numba.__version__} NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')} "
        f"(numba.config {numba.config.NUMBA_NUM_THREADS}); parallel kernel {_numba_parallel()}"
    )

    arms = [a.strip() for a in cell_args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_CLASSES]
    if unknown or not arms:
        raise SystemExit(f"--arms {cell_args.arms!r}: choose from {tuple(ARM_CLASSES)}")
    headline = "numba" if "numba" in arms else arms[0]

    load_start = list(os.getloadavg())
    instrument = cli.instrument or "sma"
    timer = timing.Timer()

    # ------------------------------------------------------------------
    # Dataset + sparse operator (NumPy) + mesh + model: the JAX harness's inputs
    # ------------------------------------------------------------------
    ds = shared.load_dataset(instrument, cell_args.mask_radius, profiling_root, timer)
    dataset_sparse, preload = preload_sparse_dataset(
        ds, cli.sparse_batch_size, cell_args.preload_cache == "on", timer
    )
    operator = dataset_sparse.sparse_operator
    print(
        f"  sparse operator: extent {operator.y_shape}x{operator.x_shape} (M={operator.M}), "
        f"batch_size={operator.batch_size}, build {preload['operator_build_s']:.2f} s "
        f"(cache {'hit' if preload['cache_hit'] else 'miss'}), peak RSS {_peak_rss_mb():.0f} MB"
    )

    print("\n--- Adapt image + mesh ---")
    from _adapt_image_util import adapt_image_for_dataset

    with timer.section("adapt_image_build"):
        adapt_image = adapt_image_for_dataset(dataset_path=ds.dataset_path, dataset=ds.dataset)
    n_requested = cli.source_pixels
    mesh = spec.build_mesh(cli, ds.dataset, adapt_image, n_requested)
    n_src = int(mesh.n_source_pixels)
    pin_is_fiducial = (
        n_requested is None
        and cli.regularization in (None, "adapt_split")
        and not ds.mask_suffix
        and cli.rect_mesh == "bilinear"
    )
    print(
        f"  source pixels: {n_src} (requested {n_requested}); regularization {mesh.regularization}"
    )

    model = shared.build_model(mesh)
    median_instance = model.instance_from_vector(vector=model.physical_values_from_prior_medians)
    n_instances = max(int(cli.n_instances), 2 + int(cli.cold_evals))
    instances = iid_instances(model, n_instances)
    adapt_images_for = shared.adapt_images_builder(adapt_image, mesh)

    default_gate = float(al.Settings().interferometer_numba_nnz_per_source_max)
    gates = {"numba": float(cell_args.numba_gate), "numpy_fft": 0.0}
    settings = {}
    memo_provenance = None
    for arm in arms:
        settings[arm], memo_provenance = memo_off_settings(gates[arm])

    def fit_from(instance, arm):
        return al.FitInterferometer(
            dataset=dataset_sparse,
            tracer=al.Tracer(galaxies=list(instance.galaxies), fields=[instance.fields]),
            adapt_images=adapt_images_for(instance.galaxies.source),
            settings=settings[arm],
            xp=np,
        )

    # ------------------------------------------------------------------
    # Route check: each arm must build the inversion it asked for
    # ------------------------------------------------------------------
    print("\n--- Route check (prior-median instance) ---")
    route: dict[str, dict] = {}
    for arm in arms:
        fit = fit_from(median_instance, arm)
        inv = fit.inversion
        mapper = inv.cls_list_from(cls=Mapper)[0]
        built = type(inv).__name__
        nnz = float(inversion_interferometer_numba_util.nnz_per_source_column_from(mapper=mapper))
        ids = inv.solve_ids_to_keep
        route[arm] = {
            "inversion_class": built,
            "mapper_class": type(mapper).__name__,
            "nnz_per_source_column": nnz,
            "nnz_total": int(np.asarray(mapper.pix_sizes_for_sub_slim_index).sum()),
            "gate": gates[arm],
            "solve_subset_edge_zeroed": ids is not None and len(ids) != n_src,
            "pixels_solved": int(len(ids)) if ids is not None else n_src,
        }
        print(f"  {arm:<10} gate {gates[arm]:g} -> {built} (nnz/col {nnz:.2f})")
        if built != ARM_CLASSES[arm]:
            raise SystemExit(
                f"arm {arm!r} built {built}, expected {ARM_CLASSES[arm]} (gate {gates[arm]}, "
                f"nnz/col {nnz:.2f}): the factory did not route as requested"
            )
        del fit, inv
    nnz_per_col = route[headline]["nnz_per_source_column"]

    # ------------------------------------------------------------------
    # Call accounting: how often one figure_of_merit evaluates F and D
    # ------------------------------------------------------------------
    def counted_figure_of_merit(arm) -> tuple[float, dict]:
        counts = {"curvature_matrix_diag": 0, "data_vector": 0}
        patched = []
        for cls, name in (
            (InversionInterferometerSparse, "curvature_matrix_diag"),
            (InversionInterferometerSparseNumba, "curvature_matrix_diag"),
            (InversionInterferometerSparse, "data_vector"),
        ):
            original = cls.__dict__[name]

            def getter(self, _original=original, _name=name):
                counts[_name] += 1
                return _original.fget(self)

            patched.append((cls, name, original))
            setattr(cls, name, property(getter))
        try:
            fom = float(fit_from(median_instance, arm).figure_of_merit)
        finally:
            for cls, name, original in patched:
                setattr(cls, name, original)
        return fom, counts

    print("\n--- Call accounting + prior-median evidence ---")
    median_fom: dict[str, float] = {}
    call_counts: dict[str, dict] = {}
    for arm in arms:
        median_fom[arm], call_counts[arm] = counted_figure_of_merit(arm)
        print(f"  {arm:<10} figure_of_merit {median_fom[arm]!r}; evaluations {call_counts[arm]}")

    # ------------------------------------------------------------------
    # The decomposition chain and the direct call
    # ------------------------------------------------------------------
    def chain(arm, instance) -> tuple[dict[str, float], float]:
        fit = fit_from(instance, arm)
        times: dict[str, float] = {}

        def tick(label, fn):
            t0 = time.perf_counter()
            out = fn()
            times[label] = time.perf_counter() - t0
            return out

        def build():
            inv = fit.inversion
            mapper = inv.cls_list_from(cls=Mapper)[0]
            # The interpolation is lazy; touch its outputs so it lands in this row.
            _ = (
                mapper.pix_indexes_for_sub_slim_index,
                mapper.pix_weights_for_sub_slim_index,
                mapper.pix_sizes_for_sub_slim_index,
            )
            return inv, mapper

        inv, mapper = tick(L_BUILD, build)
        tick(L_MAPPING, lambda: mapper.mapping_matrix)
        tick(L_REG_MATRIX, lambda: inv.regularization_matrix)
        if arm == "numba":
            tick(L_MARSHAL, lambda: inv.kernel_index_arrays)
        tick(L_CURVATURE[arm], lambda: inv.curvature_reg_matrix)
        tick(L_SOLVE, lambda: inv.reconstruction)
        chi2 = tick(L_CHI2, lambda: inv.fast_chi_squared)
        reg = tick(L_REG_TERM, lambda: inv.regularization_term)
        ld_creg = tick(L_LOGDET_CREG, lambda: inv.log_det_curvature_reg_matrix_term)
        ld_reg = tick(L_LOGDET_REG, lambda: inv.log_det_regularization_matrix_term)
        fom = tick(
            L_EVIDENCE,
            lambda: fit_util.log_evidence_from(
                chi_squared=chi2,
                regularization_term=reg,
                log_curvature_regularization_term=ld_creg,
                log_regularization_term=ld_reg,
                noise_normalization=fit.noise_normalization,
            ),
        )
        return times, float(fom)

    def direct(arm, instance) -> tuple[float, float]:
        t0 = time.perf_counter()
        fom = float(fit_from(instance, arm).figure_of_merit)
        return time.perf_counter() - t0, fom

    print("\n--- Machine-speed control (head) ---")
    control_head = dgemm_control_s()
    print(f"  1500x1500 dgemm: {control_head:.4f} s")

    records = {
        arm: {"chain": [], "chain_fom": [], "direct_s": [], "direct_fom": []} for arm in arms
    }
    print(f"\n--- iid stream: {n_instances} instances (0 = warm-up), arms {arms} ---")
    for i, instance in enumerate(instances):
        order = arms if i % 2 == 0 else list(reversed(arms))
        line = []
        for arm in order:
            steps_i, fom_chain = chain(arm, instance)
            t_direct, fom_direct = direct(arm, instance)
            rec = records[arm]
            rec["chain"].append(steps_i)
            rec["chain_fom"].append(fom_chain)
            rec["direct_s"].append(t_direct)
            rec["direct_fom"].append(fom_direct)
            line.append(f"{arm} {sum(steps_i.values()) * 1e3:.1f}/{t_direct * 1e3:.1f} ms")
        gc.collect()
        print(f"  [{i:>2}] " + "  ".join(line) + f"  (RSS {_peak_rss_mb():.0f} MB)")

    print("\n--- Machine-speed control (tail) ---")
    control_tail = dgemm_control_s()
    print(f"  1500x1500 dgemm: {control_tail:.4f} s (drift {control_tail / control_head:.3f}x)")

    # ------------------------------------------------------------------
    # Standalone sub-rows (never summed)
    # ------------------------------------------------------------------
    n_sub = (
        int(cell_args.n_sub_repeats)
        if cell_args.n_sub_repeats is not None
        else (2 if ds.n_vis > shared.LARGE_VIS else 5)
    )

    def repeat_s(fn) -> float:
        fn()
        t0 = time.perf_counter()
        for _ in range(n_sub):
            fn()
        return (time.perf_counter() - t0) / n_sub

    sub_rows: dict[str, dict[str, float]] = {}
    for arm in arms:
        inv = fit_from(median_instance, arm).inversion
        mapper = inv.cls_list_from(cls=Mapper)[0]
        rows_arm = {
            "F alone (curvature_matrix_diag)": repeat_s(lambda inv=inv: inv.curvature_matrix_diag),
            "D alone (data_vector = Lᵀ d~)": repeat_s(lambda inv=inv: inv.data_vector),
        }
        if arm == "numpy_fft":
            rows_arm["Sparse triplets alone (extent grid)"] = repeat_s(
                lambda inv=inv, m=mapper: inv._sparse_triplets_curvature_from(mapper=m)
            )
        sub_rows[arm] = rows_arm
        del inv, mapper
        gc.collect()
    print(f"\n  sub-rows: {json.dumps(sub_rows, default=float)}")

    # ------------------------------------------------------------------
    # Per-arm summaries
    # ------------------------------------------------------------------
    def arm_summary(arm) -> dict:
        rec = records[arm]
        warm = rec["chain"][1:]
        labels = list(rec["chain"][0].keys())
        steps_mean = {k: float(np.mean([s[k] for s in warm])) for k in labels}
        steps_median = {k: float(np.median([s[k] for s in warm])) for k in labels}
        total = float(sum(steps_mean.values()))
        direct_warm = rec["direct_s"][1:]
        full = _stats(direct_warm)
        chain_vs_direct = [abs(a - b) for a, b in zip(rec["chain_fom"], rec["direct_fom"])]
        n_cold = min(int(cli.cold_evals), max(len(direct_warm) - 1, 0))
        return {
            "inversion_path": route[arm]["inversion_class"],
            **route[arm],
            "steps": steps_mean,
            "steps_median": steps_median,
            "total_step_by_step": total,
            "full_call": {
                **full,
                "warmup_incl_numba_compile_s": float(rec["direct_s"][0]),
                "cold_eval_s": [float(t) for t in direct_warm[:n_cold]],
                "note": "FitInterferometer(..., xp=np).figure_of_merit on a fresh fit per "
                "instance; instance 0 (numba compile) excluded.",
            },
            "step_sum_over_full_call": total / full["mean_s"],
            "step_sum_in_band": bool(
                STEP_SUM_BAND[0] <= total / full["mean_s"] <= STEP_SUM_BAND[1]
            ),
            "figure_of_merit_prior_median": median_fom[arm],
            "figure_of_merit_per_instance": [float(v) for v in rec["direct_fom"]],
            "chain_vs_direct_max_abs_nats": float(max(chain_vs_direct)),
            "evaluations_per_figure_of_merit": call_counts[arm],
            "sub_rows": sub_rows[arm],
        }

    arm_records = {arm: arm_summary(arm) for arm in arms}
    head = arm_records[headline]

    agreement: dict[str, Any] = {
        "bar_nats": AGREEMENT_BAR_NATS,
        "local_bar_nats": LOCAL_AGREEMENT_NATS,
    }
    if "numba" in arms and "numpy_fft" in arms:
        per_instance = [
            abs(a - b)
            for a, b in zip(
                records["numba"]["direct_fom"], records["numpy_fft"]["direct_fom"], strict=True
            )
        ]
        agreement.update(
            {
                "numba_vs_numpy_fft_max_abs_nats": float(max(per_instance)),
                "numba_vs_numpy_fft_prior_median_abs_nats": abs(
                    median_fom["numba"] - median_fom["numpy_fft"]
                ),
                "holds_bar": bool(max(per_instance) <= AGREEMENT_BAR_NATS),
                "holds_local_bar": bool(max(per_instance) <= LOCAL_AGREEMENT_NATS),
            }
        )
        agreement["numba_over_numpy_fft_full_call"] = (
            arm_records["numba"]["full_call"]["mean_s"]
            / arm_records["numpy_fft"]["full_call"]["mean_s"]
        )

    al_version = al.__version__
    dict_path, chart_path = shared.result_paths(
        cli, profiling_root, instrument, f"{spec.cell}_numba", al_version, ds.mask_suffix
    )
    jax_path = shared.result_paths(
        cli, profiling_root, instrument, spec.cell, al_version, ds.mask_suffix
    )[0]
    jax_arm: dict[str, Any] = {
        "path": str(jax_path.relative_to(profiling_root))
        if jax_path.is_relative_to(profiling_root)
        else str(jax_path),
        "exists": jax_path.exists(),
        "xla_flags_env": os.environ.get("XLA_FLAGS"),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "note": "The JAX-CPU FFT arm is the JAX harness (interferometer_pixelized.py) run on the "
        "same instrument / config / mask radius with JAX_PLATFORMS=cpu; its prior-median "
        "figure_of_merit (PDIP positive-only solve) is compared with this run's (fnnls).",
    }
    if jax_path.exists():
        try:
            jd = json.loads(jax_path.read_text())
            jax_fom = jd.get("figure_of_merit_reference")
            jax_arm.update(
                {
                    "device_backend": (jd.get("device") or {}).get("backend"),
                    "inversion_class": (jd.get("configuration") or {}).get("inversion_class"),
                    "total_step_by_step": jd.get("total_step_by_step"),
                    "full_pipeline_single_jit": jd.get("full_pipeline_single_jit"),
                    "figure_of_merit_prior_median": jax_fom,
                    "same_geometry": (jd.get("configuration") or {}).get("source_pixels") == n_src
                    and (jd.get("configuration") or {}).get("image_pixels_masked")
                    == ds.n_image_pixels,
                }
            )
            if jax_fom is not None:
                agreement["numpy_vs_jax_prior_median_abs_nats"] = abs(
                    median_fom[headline] - float(jax_fom)
                )
        except (OSError, json.JSONDecodeError) as exc:
            jax_arm["error"] = str(exc)

    summary = {
        "autolens_version": al_version,
        "device": device_info_dict(),
        "instrument": instrument,
        "model": f"{spec.cell}_numba",
        "transformer": "TransformerNUFFT",
        "configuration": {
            "pixel_scale_arcsec": ds.pixel_scale,
            "mask_radius_arcsec": ds.mask_radius,
            "mask_radius_preset_arcsec": ds.mask_radius_default,
            "real_space_shape": list(ds.real_space_shape),
            "image_pixels_masked": ds.n_image_pixels,
            "visibilities": ds.n_vis,
            "source_pixels": n_src,
            "source_pixels_requested": n_requested,
            **mesh.configuration,
            "lens_light": None,
            "inversion_path": head["inversion_path"],
            "inversion_class": head["inversion_class"],
            "mapper_class": head["mapper_class"],
            "headline_arm": headline,
            "arms": arms,
            "xp": "numpy",
            "numba_gate": gates["numba"],
            "numba_gate_library_default": default_gate,
            "nnz_per_source_column": nnz_per_col,
            "would_route_numba_at_default_gate": bool(nnz_per_col <= default_gate),
            "numba_parallel_kernel": bool(_numba_parallel()),
            "numba_version": numba.__version__,
            "solve_subset_edge_zeroed": head["solve_subset_edge_zeroed"],
            "pixels_solved": head["pixels_solved"],
            "positive_only_solver": "fnnls",
            "sparse_operator_method": "nufft",
            "sparse_batch_size": int(operator.batch_size),
            "curvature_blocks_fft": int(-(-n_src // int(operator.batch_size))),
            "operator_extent_shape": [int(operator.y_shape), int(operator.x_shape)],
            "operator_M": int(operator.M),
            "transformer_chunk_size": ds.transformer_chunk,
            "n_instances": n_instances,
            "n_warm": n_instances - 1,
            "n_sub_repeats": n_sub,
            "instance_stream": {
                "kind": "iid",
                "seed": IID_SEED,
                "unit_low": IID_UNIT_RANGE[0],
                "unit_high": IID_UNIT_RANGE[1],
                "arm_order": "ABBA (alternates per instance)",
            },
            "memo_provenance": memo_provenance,
            "thread_env": thread_env,
            "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
            "host_load_avg_start": load_start,
            "host_load_avg_end": list(os.getloadavg()),
        },
        "regularization": mesh.regularization,
        "operator_build_s": preload["operator_build_s"],
        "preload": preload,
        "steps": head["steps"],
        "steps_median": head["steps_median"],
        "total_step_by_step": head["total_step_by_step"],
        "full_call": head["full_call"],
        "step_sum_over_full_call": head["step_sum_over_full_call"],
        "step_sum_band": list(STEP_SUM_BAND),
        "figure_of_merit_reference": median_fom[headline],
        "log_evidence": median_fom[headline],
        "evaluations_per_figure_of_merit": head["evaluations_per_figure_of_merit"],
        "evaluations_note": "Counted on one prior-median figure_of_merit by wrapping the "
        "sparse classes' curvature_matrix_diag / data_vector properties: both are plain "
        "properties on NumPy, so reconstruction and fast_chi_squared each evaluate them.",
        "arms": {**arm_records, "jax_cpu_fft": jax_arm},
        "agreement": agreement,
        "control_dgemm_head_s": control_head,
        "control_dgemm_tail_s": control_tail,
        "source_revisions": revisions,
        "peak_rss_mb": float(_peak_rss_mb()),
    }
    dict_path.write_text(json.dumps(summary, indent=2, default=float))
    _plot(summary, chart_path, spec, arm_records)

    pin = spec.pinned.get(instrument) if pin_is_fiducial and headline == "numba" else None
    drift = []
    if pin is None:
        print(
            f"  Pinned check SKIPPED (no pin for this configuration); fom {median_fom[headline]!r}"
        )
    else:
        for arm in arms:
            r = check_pinned(median_fom[arm], pin, label=f"{arm}_prior_median", rtol=1e-6)
            if r is not None:
                drift.append(r)
    record_pinned_check(dict_path, pin, drift)

    w = max(len(k) for a in arms for k in arm_records[a]["steps"])
    print("\n" + "=" * 86)
    print(
        f"{spec.title} — {instrument.upper()} — NumPy library dispatch (nnz/col {nnz_per_col:.1f})"
    )
    for arm in arms:
        a = arm_records[arm]
        print(f"  [{arm}] {a['inversion_class']}")
        for i, (k, v) in enumerate(a["steps"].items(), 1):
            print(f"    {i:>2}. {k:<{w}} {v * 1e3:11.3f} ms")
        print(f"        {'TOTAL (step-by-step)':<{w}} {a['total_step_by_step'] * 1e3:11.3f} ms")
        print(
            f"        {'FitInterferometer.figure_of_merit':<{w}} {a['full_call']['mean_s'] * 1e3:11.3f} ms"
        )
        print(
            f"        step sum / full call {a['step_sum_over_full_call']:.3f}; F evaluations {a['evaluations_per_figure_of_merit']}"
        )
    print(f"  agreement: {json.dumps(agreement, default=float)}")
    print(f"  Peak RSS {summary['peak_rss_mb']:.0f} MB -> {dict_path}")
    print("=" * 86)

    # Fail loudly (after the JSON is on disk) on a broken reproduction.
    for arm in arms:
        gap = arm_records[arm]["chain_vs_direct_max_abs_nats"]
        if not np.isfinite(gap) or gap > 1e-3:
            raise SystemExit(f"{arm}: step chain evidence off the library call by {gap} nats")
    if agreement.get("holds_bar") is False:
        raise SystemExit(
            f"numba vs NumPy FFT log evidence off by more than {AGREEMENT_BAR_NATS} nats"
        )


def _plot(summary: dict, chart_path: Path, spec, arm_records: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"numba": "#4C72B0", "numpy_fft": "#DD8452"}
    labels, times, cols = [], [], []
    for arm, rec in arm_records.items():
        for k, v in rec["steps"].items():
            labels.append(f"{arm}: {k}")
            times.append(v)
            cols.append(colors.get(arm, "#55A868"))
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(labels) + 2))
    bars = ax.barh(range(len(labels)), times, color=cols, edgecolor="white", height=0.6)
    tmax = max(times) if times else 1.0
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + tmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{t * 1e3:.3f} ms",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time per call (s)")
    c = summary["configuration"]
    fig.suptitle(
        f"{spec.title} — {summary['instrument'].upper()} — NumPy library dispatch",
        fontsize=11,
        fontweight="bold",
    )
    totals = " | ".join(
        f"{arm} {rec['total_step_by_step'] * 1e3:.1f} ms (full {rec['full_call']['mean_s'] * 1e3:.1f})"
        for arm, rec in arm_records.items()
    )
    ax.set_title(
        f"AutoLens v{summary['autolens_version']} | {c['visibilities']} vis | "
        f'{c["image_pixels_masked"]} px (r={c["mask_radius_arcsec"]}") | {c["source_pixels"]} src px | '
        f"nnz/col {c['nnz_per_source_column']:.1f} | {totals}",
        fontsize=8,
    )
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)
