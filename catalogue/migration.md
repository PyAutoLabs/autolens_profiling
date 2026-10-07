# Source migration inventory and wrapper retirement

The routing contract is [script_routes.json](script_routes.json), version 1.
It inventories 80 moved scientific leaves/helpers. Scripts under `scripts/lens/`
are dataset-independent components; shared machinery stays in `scripts/misc/`.
Legacy cell IDs, numerical configurations and result destinations are unchanged.
`pixelization` remains a legacy name for the rectangular source route; that alias
is not evidence that different scientific configurations can be compared.

`datacube` is now a dataset family at the source root. Multi-dataset shared-preload
experiments retain their Delaunay model. Point-source likelihoods retain their
image-plane/source-plane distinction. Imaging `pixelized/` holds experimental
measurements with several CLI-selectable meshes, not a new mesh family. Sersic
latent measurements and MGE-mass runtime variants are explicitly distinguished.

The 80 compatibility wrappers were retired in the approved layout-completion
phase after auditing project scripts, HPC submissions, CI, tests, Brain profiling
and assistant lookup. These consumers use canonical paths; the remaining test
loader was migrated. `_script_routes.load_routes` validates alias syntax and
canonical file existence without requiring legacy files. `run_legacy` was removed.
Old shell commands and Python imports must use the canonical paths below;
`canonical_path`, `runtime_path` and `legacy_stem` retain historical lookup and
output naming. Shared framework packages under `scripts/misc/` remain importable.

Likelihood hazard documentation now lives beside the MGE and rectangular
imaging cells. Parallel and streaming guides also live beside their canonical
models. Historical note/campaign text and captured source strings remain intact;
the generated solver corpus table still reports the original capturing script.
New executions identify the canonical source in provenance. Archived source paths,
measurements, pins and evidence qualifications are not rewritten.

The generic `scripts/misc/jax_compile/probe.py` still reports its retired builder
as unavailable; this migration does not invent an executable replacement or run
compile measurements. Existing runtime `--vmap-probe` dispatch remains supported.

Before migration, every moved source was inventoried with its literal callers.
Validation compared the entire numerical AST (only source-path plumbing normalized),
CLI/default declarations and 1,388 tracked result files; no numerical runs were used.
Callers include cross-script loaders, runtime/latent sweeps, manual workflow dispatch,
HPC submissions, static CLI contracts, smoke tests and documentation. The table
counts pre-migration literal callers; dynamic callers were separately audited.

| Legacy path | Canonical path | Literal callers |
|---|---|---:|
| `scripts/cluster/likelihood_breakdown/image_plane.py` | `scripts/cluster/image_plane/likelihood_breakdown.py` | 2 |
| `scripts/cluster/likelihood_breakdown/source_plane.py` | `scripts/cluster/source_plane/likelihood_breakdown.py` | 0 |
| `scripts/imaging/hazards/mge_nnls_capture.py` | `scripts/imaging/mge/hazards_nnls_capture.py` | 4 |
| `scripts/imaging/hazards/pixelization.py` | `scripts/imaging/rectangular/hazards.py` | 3 |
| `scripts/imaging/latent/effective_einstein_radius.py` | `scripts/imaging/sersic/latent_effective_einstein_radius.py` | 0 |
| `scripts/imaging/latent/magnification.py` | `scripts/imaging/sersic/latent_magnification.py` | 0 |
| `scripts/imaging/latent/total_lens_flux_mujy.py` | `scripts/imaging/sersic/latent_total_lens_flux_mujy.py` | 0 |
| `scripts/imaging/latent/total_lensed_source_flux_mujy.py` | `scripts/imaging/sersic/latent_total_lensed_source_flux_mujy.py` | 0 |
| `scripts/imaging/latent/total_source_flux_mujy.py` | `scripts/imaging/sersic/latent_total_source_flux_mujy.py` | 0 |
| `scripts/imaging/likelihood_breakdown/delaunay.py` | `scripts/imaging/delaunay/likelihood_breakdown.py` | 13 |
| `scripts/imaging/likelihood_breakdown/delaunay_nn.py` | `scripts/imaging/delaunay_nn/likelihood_breakdown.py` | 14 |
| `scripts/imaging/likelihood_breakdown/delaunay_numba.py` | `scripts/imaging/delaunay/likelihood_breakdown_numba.py` | 0 |
| `scripts/imaging/likelihood_breakdown/fixed_light.py` | `scripts/imaging/pixelized/fixed_light.py` | 10 |
| `scripts/imaging/likelihood_breakdown/fixed_light_cpu_kernels.py` | `scripts/imaging/pixelized/fixed_light_cpu_kernels.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_draws.py` | `scripts/imaging/pixelized/fixed_light_draws.py` | 2 |
| `scripts/imaging/likelihood_breakdown/fixed_light_library.py` | `scripts/imaging/pixelized/fixed_light_library.py` | 8 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba.py` | `scripts/imaging/pixelized/fixed_light_numba.py` | 7 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_draws.py` | `scripts/imaging/pixelized/fixed_light_numba_draws.py` | 3 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_l2_witness.py` | `scripts/imaging/pixelized/fixed_light_numba_levers_l2_witness.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_l3_witness.py` | `scripts/imaging/pixelized/fixed_light_numba_levers_l3_witness.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_levers_witness.py` | `scripts/imaging/pixelized/fixed_light_numba_levers_witness.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_memo_policy.py` | `scripts/imaging/pixelized/fixed_light_numba_memo_policy.py` | 2 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_s4_witness.py` | `scripts/imaging/pixelized/fixed_light_numba_s4_witness.py` | 2 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_s4b_witness.py` | `scripts/imaging/pixelized/fixed_light_numba_s4b_witness.py` | 2 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_scaling.py` | `scripts/imaging/pixelized/fixed_light_numba_scaling.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_numba_solvers.py` | `scripts/imaging/pixelized/fixed_light_numba_solvers.py` | 1 |
| `scripts/imaging/likelihood_breakdown/fixed_light_trace.py` | `scripts/imaging/pixelized/fixed_light_trace.py` | 13 |
| `scripts/imaging/likelihood_breakdown/matrix_free.py` | `scripts/imaging/pixelized/matrix_free.py` | 5 |
| `scripts/imaging/likelihood_breakdown/mge.py` | `scripts/imaging/mge/likelihood_breakdown.py` | 1 |
| `scripts/imaging/likelihood_breakdown/nautilus_batch_capture.py` | `scripts/imaging/pixelized/nautilus_batch_capture.py` | 2 |
| `scripts/imaging/likelihood_breakdown/pixelization.py` | `scripts/imaging/rectangular/likelihood_breakdown.py` | 6 |
| `scripts/imaging/likelihood_breakdown/pixelization_numba.py` | `scripts/imaging/rectangular/likelihood_breakdown_numba.py` | 2 |
| `scripts/imaging/likelihood_runtime/delaunay.py` | `scripts/imaging/delaunay/likelihood_runtime.py` | 8 |
| `scripts/imaging/likelihood_runtime/delaunay_nn.py` | `scripts/imaging/delaunay_nn/likelihood_runtime.py` | 9 |
| `scripts/imaging/likelihood_runtime/delaunay_numba.py` | `scripts/imaging/delaunay/likelihood_runtime_numba.py` | 0 |
| `scripts/imaging/likelihood_runtime/mge.py` | `scripts/imaging/mge/likelihood_runtime.py` | 5 |
| `scripts/imaging/likelihood_runtime/mge_mass_jax.py` | `scripts/imaging/mge_mass/likelihood_runtime_jax.py` | 0 |
| `scripts/imaging/likelihood_runtime/pixelization.py` | `scripts/imaging/rectangular/likelihood_runtime.py` | 4 |
| `scripts/imaging/likelihood_runtime/pixelization_numba.py` | `scripts/imaging/rectangular/likelihood_runtime_numba.py` | 0 |
| `scripts/imaging/likelihood_runtime/pixelization_numba_mge_mass.py` | `scripts/imaging/rectangular/likelihood_runtime_numba_mge_mass.py` | 1 |
| `scripts/imaging/parallel_scaling/pixelization_numba.py` | `scripts/imaging/rectangular/parallel_scaling_numba.py` | 4 |
| `scripts/imaging/quick_update/delaunay.py` | `scripts/imaging/delaunay/quick_update.py` | 0 |
| `scripts/imaging/quick_update/mge.py` | `scripts/imaging/mge/quick_update.py` | 0 |
| `scripts/interferometer/likelihood_breakdown/datacube/delaunay.py` | `scripts/datacube/delaunay/likelihood_breakdown.py` | 2 |
| `scripts/interferometer/likelihood_breakdown/datacube/inversion_setup_decompose.py` | `scripts/datacube/delaunay/inversion_setup_decompose.py` | 2 |
| `scripts/interferometer/likelihood_breakdown/delaunay.py` | `scripts/interferometer/delaunay/likelihood_breakdown.py` | 16 |
| `scripts/interferometer/likelihood_breakdown/delaunay_numba.py` | `scripts/interferometer/delaunay/likelihood_breakdown_numba.py` | 10 |
| `scripts/interferometer/likelihood_breakdown/mge.py` | `scripts/interferometer/mge/likelihood_breakdown.py` | 12 |
| `scripts/interferometer/likelihood_breakdown/pixelization.py` | `scripts/interferometer/rectangular/likelihood_breakdown.py` | 14 |
| `scripts/interferometer/likelihood_breakdown/pixelization_numba.py` | `scripts/interferometer/rectangular/likelihood_breakdown_numba.py` | 9 |
| `scripts/interferometer/likelihood_breakdown/preload_numba.py` | `scripts/interferometer/pixelized/preload_numba.py` | 0 |
| `scripts/interferometer/likelihood_runtime/datacube/delaunay.py` | `scripts/datacube/delaunay/likelihood_runtime.py` | 1 |
| `scripts/interferometer/likelihood_runtime/datacube/shared_preloads.py` | `scripts/datacube/delaunay/likelihood_runtime_shared_preloads.py` | 0 |
| `scripts/interferometer/likelihood_runtime/delaunay.py` | `scripts/interferometer/delaunay/likelihood_runtime.py` | 1 |
| `scripts/interferometer/likelihood_runtime/mge.py` | `scripts/interferometer/mge/likelihood_runtime.py` | 1 |
| `scripts/interferometer/likelihood_runtime/pixelization.py` | `scripts/interferometer/rectangular/likelihood_runtime.py` | 1 |
| `scripts/interferometer/quick_update/delaunay.py` | `scripts/interferometer/delaunay/quick_update.py` | 0 |
| `scripts/interferometer/quick_update/mge.py` | `scripts/interferometer/mge/quick_update.py` | 0 |
| `scripts/interferometer/streaming_scaling/_streaming.py` | `scripts/interferometer/delaunay/_streaming.py` | 0 |
| `scripts/interferometer/streaming_scaling/accumulate.py` | `scripts/interferometer/delaunay/streaming_accumulate.py` | 2 |
| `scripts/interferometer/streaming_scaling/in_memory.py` | `scripts/interferometer/delaunay/streaming_in_memory.py` | 1 |
| `scripts/interferometer/streaming_scaling/parity.py` | `scripts/interferometer/delaunay/streaming_parity.py` | 1 |
| `scripts/interferometer/streaming_scaling/plot_scaling.py` | `scripts/interferometer/delaunay/streaming_plot_scaling.py` | 2 |
| `scripts/multi_dataset/likelihood_runtime/shared_preloads.py` | `scripts/multi_dataset/delaunay/likelihood_runtime_shared_preloads.py` | 0 |
| `scripts/point_source_image/likelihood_breakdown/_point_solver_stage_map.py` | `scripts/point_source_image/image_plane/_point_solver_stage_map.py` | 0 |
| `scripts/point_source_image/likelihood_breakdown/gpu_bottleneck_map.py` | `scripts/point_source_image/image_plane/gpu_bottleneck_map.py` | 3 |
| `scripts/point_source_image/likelihood_breakdown/image_plane.py` | `scripts/point_source_image/image_plane/likelihood_breakdown.py` | 4 |
| `scripts/point_source_image/likelihood_breakdown/solver_config_sweep.py` | `scripts/point_source_image/image_plane/solver_config_sweep.py` | 6 |
| `scripts/point_source_image/likelihood_breakdown/static_lattice_ab.py` | `scripts/point_source_image/image_plane/static_lattice_ab.py` | 2 |
| `scripts/point_source_image/likelihood_breakdown/vertex_dedup_ab.py` | `scripts/point_source_image/image_plane/vertex_dedup_ab.py` | 3 |
| `scripts/point_source_image/likelihood_runtime/image_plane.py` | `scripts/point_source_image/image_plane/likelihood_runtime.py` | 0 |
| `scripts/point_source_image/likelihood_runtime/image_plane_solved.py` | `scripts/point_source_image/image_plane/likelihood_runtime_solved.py` | 0 |
| `scripts/point_source_image/quick_update/point_source.py` | `scripts/point_source_image/image_plane/quick_update.py` | 0 |
| `scripts/point_source_source/likelihood_breakdown/backward_pass_ab.py` | `scripts/point_source_source/source_plane/backward_pass_ab.py` | 2 |
| `scripts/point_source_source/likelihood_breakdown/gradient_mode_crossover.py` | `scripts/point_source_source/source_plane/gradient_mode_crossover.py` | 2 |
| `scripts/point_source_source/likelihood_breakdown/gradient_mode_library_ab.py` | `scripts/point_source_source/source_plane/gradient_mode_library_ab.py` | 2 |
| `scripts/point_source_source/likelihood_breakdown/pytree_input_ab.py` | `scripts/point_source_source/source_plane/pytree_input_ab.py` | 2 |
| `scripts/point_source_source/likelihood_breakdown/source_plane.py` | `scripts/point_source_source/source_plane/likelihood_breakdown.py` | 2 |
| `scripts/point_source_source/likelihood_runtime/source_plane.py` | `scripts/point_source_source/source_plane/likelihood_runtime.py` | 0 |
| `scripts/point_source_source/likelihood_runtime/source_plane_solved.py` | `scripts/point_source_source/source_plane/likelihood_runtime_solved.py` | 2 |
