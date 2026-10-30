<!-- generated: build_setup_wiki.py; do not edit -->
# mge · hst

[Model index](index.md)

Exact setup ID: `multi_galaxy/mge/hst/73047ce4c40bcf534a49`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| dataset_type | "multi_galaxy" | recorded |  |
| fast | false | recorded |  |
| galaxy_group_cell | {"dataset": "autolens_workspace scripts/multi_galaxy/simulator.py (grid 200x200 at 0.05\", galaxies, exposure 900 s, sky 0.1, mask_extra_galaxies 3 x 0.3\") and modeling.py (3.0\" mask, noise scaling, radial-bin over-sampling on both lenses); autolens_visualization true_models/multi_galaxy.py; PSF from instruments/imaging.py hst (21x21, sigma 0.05\")", "dataset_type": "multi_galaxy", "extra_galaxies": 0, "instrument": "hst", "lens_light": "workspace MGE: 20 linear Gaussians per main lens", "main_lenses": 2, "mask_radius_arcsec": 3.0, "noise_scaled_region": {"centre": [2.2, 1.6], "radius": 0.9}, "ruling": "likelihood-lifecycle 2d draft (#473), decisions D1-D9 on the campaign page", "sequence": {"kind": "iid", "seed": 235, "unit_high": 0.6, "unit_low": 0.4}, "source": "MGE, 20 linear Gaussians (one basis)"} | recorded |  |
| image_pixels_masked | 11304 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| instance | "iid draw 0 of the seeded sequence" | recorded |  |
| instance_unit_vector | [0.499784489007424, 0.41402443684610074, 0.41327450881714656, 0.4200167345680767, 0.5789514047288555, 0.4702516857248635, 0.567334291975647, 0.47173759315010344, 0.5239626288064884, 0.5543072472409505, 0.5399346262699198, 0.5099305691842051, 0.5974626164364525, 0.4381789532715836, 0.5552315204832754, 0.5058238713539893, 0.45735024341537395, 0.5870995335054062, 0.509803788830344, 0.41707028300618887] | recorded |  |
| inversion_class | "InversionImagingMapping" | recorded |  |
| mask_radius_arcsec | 3.0 | recorded |  |
| memo_provenance | {"env_AUTOARRAY_NNLS_WARM_START": "0", "env_before": null, "note": "Both gates are set for memo=off: the Settings flag and the AUTOARRAY_NNLS_WARM_START env kill-switch (autoarray/inversion/inversion/nnls_memo.py:63). settings_nnls_warm_start_memo is read back off the constructed Settings, so it records what took effect. The library default is true and both production projects leave it unset.", "requested": "off", "settings_nnls_warm_start_memo": false} | recorded |  |
| n_repeats | 10 | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sampled_pixels | 47904 | recorded |  |
| oversampled_pixels | 47904 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | null | name | Not recorded in this legacy evidence; no current default substituted. |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | null | count | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels_requested | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| sparse_batch_size | null | recorded | Not recorded in this legacy evidence; no current default substituted. |
| standard_size | true | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| use_mixed_precision | false | recorded |  |
| xla_cpu_workaround | "--xla_cpu_multi_thread_eigen=false appended to XLA_FLAGS: jaxlib 0.11.2 XLA:CPU deadlocks on these executables after a few calls with the default (autolens_profiling#473, D12)" | recorded |  |

## Evidence

[Original setup artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>12 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/1fba9ced33a119581619` | prefix_6.steady_per_call_s | {"prefix_6.steady_per_call_s": 0.035397995891980824} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_6/steady_per_call_s` |
| `measurement/2ba92563cfea7a9e41db` | prefix_1.steady_per_call_s | {"prefix_1.steady_per_call_s": 0.0010865670046769083} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_1/steady_per_call_s` |
| `measurement/41652748b7bf5c49cdda` | steps.Blurred image (non-linear light) | {"steps.Blurred image (non-linear light)": 0.00021816639928147192} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Blurred image (non-linear light)` |
| `measurement/4a0a46aca3afb21a228e` | prefix_5.steady_per_call_s | {"prefix_5.steady_per_call_s": 0.025912336201872676} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_5/steady_per_call_s` |
| `measurement/4fec1599a8ed4bdc72a0` | component_total | {"component_total": 0.02948080979986116} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/total_step_by_step` |
| `measurement/7e718ed33200e39803ff` | steps.Ray-trace grids (tracer) | {"steps.Ray-trace grids (tracer)": 0.0010865670046769083} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Ray-trace grids (tracer)` |
| `measurement/96f2e1404180deab498e` | prefix_4.steady_per_call_s | {"prefix_4.steady_per_call_s": 0.02709971070289612} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_4/steady_per_call_s` |
| `measurement/9af8a5a5044daf3bc733` | prefix_2.steady_per_call_s | {"prefix_2.steady_per_call_s": 0.0013047334039583802} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_2/steady_per_call_s` |
| `measurement/aa2dc7f3bc677d4023a0` | prefix_3.steady_per_call_s | {"prefix_3.steady_per_call_s": 0.04022509260103106} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_3/steady_per_call_s` |
| `measurement/c856d32cd8b281b2aff9` | prefix_7.steady_per_call_s | {"prefix_7.steady_per_call_s": 0.029480809799861164} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_7/steady_per_call_s` |
| `measurement/c9c934c415ed7dc7276e` | steps.Reconstruction (NNLS) | {"steps.Reconstruction (NNLS)": 0.009485659690108148} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Reconstruction (NNLS)` |
| `measurement/f6ad2b18f2e15bfa7715` | steps.Data vector D (mapping + PSF) | {"steps.Data vector D (mapping + PSF)": 0.03892035919707268} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/steps/Data vector D (mapping + PSF)` |

</details>

### compile

<details><summary>21 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/050c40e4b5783e31d637` | prefix_4.compile_s | {"prefix_4.compile_s": 5.726406809990294} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_4/compile_s` |
| `measurement/07c7d33fb4b60b81830e` | prefix_5.lower_s | {"prefix_5.lower_s": 3.6989997889613733} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_5/lower_s` |
| `measurement/0ff16ea1f74d1d77efc6` | prefix_1.compile_s | {"prefix_1.compile_s": 0.2745511810062453} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_1/compile_s` |
| `measurement/1c9b36c12b002b99e179` | prefix_4.lower_s | {"prefix_4.lower_s": 3.417123139020987} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_4/lower_s` |
| `measurement/2ac05302c2678fe38aa2` | prefix_1.lower_s | {"prefix_1.lower_s": 0.7541519469814375} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_1/lower_s` |
| `measurement/36fb9a0de7fcaef70b10` | prefix_2.compile_s | {"prefix_2.compile_s": 0.3297420940361917} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_2/compile_s` |
| `measurement/4e3aa7a1a1dc6b1630f6` | prefix_7.lower_s | {"prefix_7.lower_s": 3.4273113219533116} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_7/lower_s` |
| `measurement/5b71fe3040551bf674ef` | prefix_3.compile_s | {"prefix_3.compile_s": 6.368397713988088} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_3/compile_s` |
| `measurement/66d1c893e394c6a17c29` | prefix_3.first_call_s | {"prefix_3.first_call_s": 0.04035995504818857} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_3/first_call_s` |
| `measurement/830ae3707ebb8298cbd8` | prefix_5.compile_s | {"prefix_5.compile_s": 6.3233321600127965} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_5/compile_s` |
| `measurement/93bd67c76b7733f84726` | prefix_7.compile_s | {"prefix_7.compile_s": 5.763271734002046} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_7/compile_s` |
| `measurement/99c7ee699e351b452461` | prefix_6.compile_s | {"prefix_6.compile_s": 6.622877922956832} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_6/compile_s` |
| `measurement/9aeee0ef6412a10d9989` | prefix_6.lower_s | {"prefix_6.lower_s": 3.7224164810031652} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_6/lower_s` |
| `measurement/a614438fd928a24b9364` | prefix_2.lower_s | {"prefix_2.lower_s": 1.8266223940299824} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_2/lower_s` |
| `measurement/ab2e651a2b799672147b` | prefix_4.first_call_s | {"prefix_4.first_call_s": 0.03061276499647647} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_4/first_call_s` |
| `measurement/b41fa16bbae372bde2eb` | prefix_1.first_call_s | {"prefix_1.first_call_s": 0.008669978007674217} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_1/first_call_s` |
| `measurement/d3bd9718e5292f3bec22` | prefix_3.lower_s | {"prefix_3.lower_s": 3.5422026129672304} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_3/lower_s` |
| `measurement/d74ab8d5937d4ffe739e` | prefix_5.first_call_s | {"prefix_5.first_call_s": 0.028805687092244625} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_5/first_call_s` |
| `measurement/e428760a50be5699e28d` | prefix_6.first_call_s | {"prefix_6.first_call_s": 0.03528086899314076} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_6/first_call_s` |
| `measurement/e6d90936b7de10f9c7f3` | prefix_7.first_call_s | {"prefix_7.first_call_s": 0.03554476494900882} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_7/first_call_s` |
| `measurement/eab5dd8c4b131dfe1fe4` | prefix_2.first_call_s | {"prefix_2.first_call_s": 0.005450427066534758} | s | cpu / float64 | archive support; unreviewed | [artifact](../../../../results/multi_galaxy/mge/hst/jax_cpu_dense_fp64/hpc_ral_cpu/breakdown.json) `/jit_phases/prefix_2/first_call_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "cpu",
    "device": "cpu",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "cpu",
      "cache_fresh": true,
      "cpu_count": 236,
      "device": "cpu:0",
      "hostname": "euclid-ral-compute-10-5",
      "omp_num_threads": "1",
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_cpu_multi_thread_eigen=false --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoArray.library_revisions": "0f12fac7789c81d0fdaa324508ced5b4c434d42a",
      "PyAutoArray.library_versions": "2026.8.17.1",
      "PyAutoFit.library_revisions": "8cb533a605f556d85012d5eb7ed8d4f3c2a4dcf6",
      "PyAutoFit.library_versions": "2026.8.17.1",
      "PyAutoGalaxy.library_revisions": "7472211470b9d2b3a76bb12dcacbac51ab4de06f",
      "PyAutoGalaxy.library_versions": "2026.8.17.1",
      "PyAutoLens": "2026.8.17.1",
      "PyAutoLens.library_revisions": "c6927eeb33724c2de2f2e2c773ba3ebe651943a8",
      "PyAutoLens.library_versions": "2026.8.17.1",
      "PyAutoNerves.library_revisions": "ec8b5c1e5f17ac03ac5246b5fbc322624a6cd3f6",
      "PyAutoNerves.library_versions": "2026.8.17.1",
      "autolens_profiling.library_revisions": "830314e9bcafafd469f0985f9d1c9629cca830ff",
      "finufft.dependency_versions": "2.5.1",
      "jax.dependency_versions": "0.11.2",
      "jaxlib.dependency_versions": "0.11.2",
      "nufftax.dependency_versions": "0.6.1",
      "numba.dependency_versions": "0.65.1",
      "numpy.dependency_versions": "2.2.6",
      "scipy.dependency_versions": "1.17.1"
    },
    "unknowns": {}
  },
  "method": {
    "cache_state": null,
    "repetitions": null,
    "statistic": null,
    "synchronization": null,
    "unknowns": {
      "cache_state": "Not recorded in this legacy evidence; no current default substituted.",
      "repetitions": "Not recorded in this legacy evidence; no current default substituted.",
      "statistic": "Not recorded in this legacy evidence; no current default substituted.",
      "synchronization": "Not recorded in this legacy evidence; no current default substituted.",
      "warmup": "Not recorded in this legacy evidence; no current default substituted."
    },
    "warmup": null
  },
  "provenance": {
    "has_provenance": true,
    "host": "euclid-ral-compute-10-5",
    "measured_at": "2026-10-30T11:45:39Z",
    "qualified": false,
    "unknowns": {}
  },
  "validation": {
    "reason": "Legacy evidence transcribed without scientific baseline acceptance.",
    "status": "unreviewed"
  }
}
```

</details>

## Hazards

No explicitly bound applicable evidence; this does not establish absence of hazards or validate a setting.

## Recommendations

No explicitly bound applicable evidence; this does not establish absence of hazards or validate a setting.

## Scripts

Navigation links identify the model family; they do not reconstruct historical settings or promise an executable replay.

No canonical script explicitly registered for this model.

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
