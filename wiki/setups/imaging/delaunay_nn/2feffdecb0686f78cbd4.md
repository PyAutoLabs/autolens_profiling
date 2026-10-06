<!-- generated: build_setup_wiki.py; do not edit -->
# delaunay_nn · hst

[Model index](index.md)

Exact setup ID: `imaging/delaunay_nn/hst/88f6f36ed3f294f90deb`.

Imported support is unreviewed; this page grants no baseline acceptance. Timing axes remain separate; no total fit-time prediction is made.

## Configuration

Identity limitations: Isolated to source row: legacy metadata is insufficient for cross-file joins.

| Setting | Recorded value | Unit | Unknown reason |
|---|---|---|---|
| delaunay_vertices | 1500 | recorded |  |
| edge_zeroed_pixels | 0 | recorded |  |
| image_pixels_masked | 15361 | count |  |
| image_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| mask_radius_arcsec | 3.5 | recorded |  |
| memo | "library_default (inert on the JAX path)" | recorded |  |
| n_vis | null | count | Not recorded in this legacy evidence; no current default substituted. |
| over_sample_size_lp_rule | {"centre": [0.0, 0.0], "note": "Outer sub-size 1 retired repo-wide on 2026-09-08 (autolens_profiling#235): it leaves the outermost annulus un-over-sampled and causes gradient issues.", "radial_list": [0.3, 0.6], "sub_size_list": [4, 2, 2]} | recorded |  |
| over_sampled_pixels | 62752 | recorded |  |
| oversampled_pixels | 62752 | recorded |  |
| oversampling | null | dimensionless | Not recorded in this legacy evidence; no current default substituted. |
| pixel_scale_arcsec | 0.05 | recorded |  |
| preloads | null | name | Not recorded in this legacy evidence; no current default substituted. |
| psf_shape | null | pixel | Not recorded in this legacy evidence; no current default substituted. |
| regularization | {"inner_coefficient": 0.1, "outer_coefficient": 10.0, "scheme": "adapt_split", "signal_scale": 0.1} | name |  |
| solver | null | name | Not recorded in this legacy evidence; no current default substituted. |
| source_pixels | 1500 | count |  |
| thread_env | {"n_threads": null, "note": "Not pinned (legacy variant): recorded as found.", "overridden": {}, "preexisting": {}, "set_to": null, "vars": ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]} | recorded |  |
| total_params | 1560 | recorded |  |
| transformer | null | name | Not recorded in this legacy evidence; no current default substituted. |
| vmap_batch | 16 | recorded |  |

## Evidence

[Original setup artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json); JSON pointer: `(document root)`.

### Selected references

No selected reference for this exact setup; archive evidence is available below.

### breakdown

<details><summary>48 recorded breakdown measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/023a3ae98aa6ac4e8454` | steps.Lens light images (pre-PSF) | {"steps.Lens light images (pre-PSF)": 0.00011486930307000876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Lens light images (pre-PSF)` |
| `measurement/07b8a61033a2ca45eab4` | steps_vmap_per_call.Inversion setup (steps 5-8 combined) | {"steps_vmap_per_call.Inversion setup (steps 5-8 combined)": 0.015316297605750151} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps_vmap_per_call/Inversion setup (steps 5-8 combined)` |
| `measurement/14ea37ae9d60eae51af1` | steps.Profile-subtracted image | {"steps.Profile-subtracted image": 0.00016895199660211803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Profile-subtracted image` |
| `measurement/17facd8c1da57fdd1919` | profile_subtract_jit.steady_per_call_s | {"profile_subtract_jit.steady_per_call_s": 0.00016895199660211803} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/profile_subtract_jit/steady_per_call_s` |
| `measurement/1896d909551d8d7be7d2` | lens_image_jit.steady_per_call_s | {"lens_image_jit.steady_per_call_s": 0.00011486930307000876} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/lens_image_jit/steady_per_call_s` |
| `measurement/18ee50aff57fc7244a1b` | blurred_image_jit.steady_per_call_s | {"blurred_image_jit.steady_per_call_s": 0.0012356430059298873} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/blurred_image_jit/steady_per_call_s` |
| `measurement/1bff51e305890c8b6899` | steps.Blurred image (PSF convolution) | {"steps.Blurred image (PSF convolution)": 0.0012356430059298873} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Blurred image (PSF convolution)` |
| `measurement/1c7d2a9d96bcc62648c2` | inversion_setup_vmap16.steady_per_call_s | {"inversion_setup_vmap16.steady_per_call_s": 0.015316297605750151} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_vmap16/steady_per_call_s` |
| `measurement/1c925939e664ecc542ec` | ray_trace_data_jit.steady_per_call_s | {"ray_trace_data_jit.steady_per_call_s": 0.00018026300240308045} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_data_jit/steady_per_call_s` |
| `measurement/1fe7b0dd760b4d04a477` | cholesky_curvature_reg_jit.steady_per_call_s | {"cholesky_curvature_reg_jit.steady_per_call_s": 0.0012349370867013932} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_curvature_reg_jit/steady_per_call_s` |
| `measurement/1fea79745f7a5b502934` | setup_prefix_6s.steady_per_call_s | {"setup_prefix_6s.steady_per_call_s": 0.014646413899026812} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s/steady_per_call_s` |
| `measurement/22eb9d2a2d1769034c69` | log_det_regularization_jit.steady_per_call_s | {"log_det_regularization_jit.steady_per_call_s": 0.001184337306767702} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_regularization_jit/steady_per_call_s` |
| `measurement/23ca713487e4f0cdcb2e` | log_evidence_jit.steady_per_call_s | {"log_evidence_jit.steady_per_call_s": 0.002261362294666469} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_evidence_jit/steady_per_call_s` |
| `measurement/29890ee42344d72417fe` | reconstruction_jit.steady_per_call_s | {"reconstruction_jit.steady_per_call_s": 0.03756908189971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/reconstruction_jit/steady_per_call_s` |
| `measurement/2a7b86bf57cdedf7ea53` | steps.Data vector (D) | {"steps.Data vector (D)": 0.0003321432042866945} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Data vector (D)` |
| `measurement/2f1d629d9c1ef304ccec` | setup_prefix_6s_vmap16.steady_per_call_s | {"setup_prefix_6s_vmap16.steady_per_call_s": 0.006547379856056068} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s_vmap16/steady_per_call_s` |
| `measurement/35bf2748a70cef8c3253` | steps.Regularized reconstruction | {"steps.Regularized reconstruction": 0.03756908189971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Regularized reconstruction` |
| `measurement/3725026b56631619a736` | setup_prefix_6.steady_per_call_s | {"setup_prefix_6.steady_per_call_s": 0.014795578108169139} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6/steady_per_call_s` |
| `measurement/3ccaf0e0bf934120be02` | setup_prefix_7_vmap16.steady_per_call_s | {"setup_prefix_7_vmap16.steady_per_call_s": 0.006661702950077597} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7_vmap16/steady_per_call_s` |
| `measurement/47e26167cf2daefc4bd3` | nnls_pdip_vmap16.steady_per_call_s | {"nnls_pdip_vmap16.steady_per_call_s": 0.021910605269658844} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_vmap16/steady_per_call_s` |
| `measurement/4c7d8114427ab997ae21` | setup_prefix_11_vmap16.steady_per_call_s | {"setup_prefix_11_vmap16.steady_per_call_s": 0.007381853187689557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_11_vmap16/steady_per_call_s` |
| `measurement/4e2522b2edfd652b4890` | steps_vmap_per_call.Regularization matrix (H) | {"steps_vmap_per_call.Regularization matrix (H)": 0.000856721331365406} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps_vmap_per_call/Regularization matrix (H)` |
| `measurement/52e0e46a5cf9c65f2476` | inversion_setup_jit.steady_per_call_s | {"inversion_setup_jit.steady_per_call_s": 0.032726566400378944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_jit/steady_per_call_s` |
| `measurement/544b049c93bed987525c` | steps.Ray-trace mesh grid | {"steps.Ray-trace mesh grid": 0.0001708820927888155} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Ray-trace mesh grid` |
| `measurement/6df27fac090abe1551af` | setup_prefix_5.steady_per_call_s | {"setup_prefix_5.steady_per_call_s": 0.001089443895034492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5/steady_per_call_s` |
| `measurement/7120bffd4a774d5e0e9a` | setup_prefix_7.steady_per_call_s | {"setup_prefix_7.steady_per_call_s": 0.014517401694320142} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7/steady_per_call_s` |
| `measurement/7121d81a0967fc1c5724` | cholesky_solve_jit.steady_per_call_s | {"cholesky_solve_jit.steady_per_call_s": 0.001521529396995902} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_solve_jit/steady_per_call_s` |
| `measurement/726f168dac6c34483266` | setup_prefix_5_vmap16.steady_per_call_s | {"setup_prefix_5_vmap16.steady_per_call_s": 6.457576819229871e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5_vmap16/steady_per_call_s` |
| `measurement/7645b060df5cae590ba9` | interpolator_prefix_vmap_per_call | {"interpolator_prefix_vmap_per_call": 0.006525131856324151} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/interpolator_prefix_vmap_per_call_s` |
| `measurement/7a15bc065bb9840d6488` | setup_prefix_8_vmap16.steady_per_call_s | {"setup_prefix_8_vmap16.steady_per_call_s": 0.01498365512525197} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8_vmap16/steady_per_call_s` |
| `measurement/7fe0791f8d10d975d4d6` | regularization_matrix_prefix | {"regularization_matrix_prefix": 0.01565156220458448} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/regularization_matrix_prefix_s` |
| `measurement/81497714e5edf716415c` | ray_trace_mesh_jit.steady_per_call_s | {"ray_trace_mesh_jit.steady_per_call_s": 0.0001708820927888155} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_mesh_jit/steady_per_call_s` |
| `measurement/87d7b4061b493ba5a850` | interpolator_prefix | {"interpolator_prefix": 0.014795578108169139} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/interpolator_prefix_s` |
| `measurement/a0a426788dd4aaf70d02` | steps.Regularization matrix (H) | {"steps.Regularization matrix (H)": 0.0008559840964153399} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Regularization matrix (H)` |
| `measurement/b1464f984d41822e9de2` | setup_prefix_8.steady_per_call_s | {"setup_prefix_8.steady_per_call_s": 0.022734869993291794} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8/steady_per_call_s` |
| `measurement/b4c71686bde7035e5739` | nnls_pdip_jit.steady_per_call_s | {"nnls_pdip_jit.steady_per_call_s": 0.037304298300296065} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_jit/steady_per_call_s` |
| `measurement/b92d805077e521f23e9c` | log_det_curvature_reg_jit.steady_per_call_s | {"log_det_curvature_reg_jit.steady_per_call_s": 0.0011636593844741583} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_curvature_reg_jit/steady_per_call_s` |
| `measurement/bb8184d460aecf637e45` | data_vector_jit.steady_per_call_s | {"data_vector_jit.steady_per_call_s": 0.0003321432042866945} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/data_vector_jit/steady_per_call_s` |
| `measurement/c3e4e5df073a64831852` | steps.Mapped recon + log evidence | {"steps.Mapped recon + log evidence": 0.002261362294666469} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Mapped recon + log evidence` |
| `measurement/c8b7c7d070891ef35999` | component_total | {"component_total": 0.0804358759894967} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/total_step_by_step` |
| `measurement/ca52705b1761f07d19be` | regularization_matrix_prefix_vmap_per_call | {"regularization_matrix_prefix_vmap_per_call": 0.007381853187689557} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/regularization_matrix_prefix_vmap_per_call_s` |
| `measurement/cd60cbe0f3e01fa3c780` | nnls_pdip_one_iteration_jit.steady_per_call_s | {"nnls_pdip_one_iteration_jit.steady_per_call_s": 0.0031838789116591217} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_one_iteration_jit/steady_per_call_s` |
| `measurement/cee9785467f2e1a3c14e` | steps.Ray-trace data grid | {"steps.Ray-trace data grid": 0.00018026300240308045} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Ray-trace data grid` |
| `measurement/d06491d5a94b6fdce57e` | curvature_matrix_jit.steady_per_call_s | {"curvature_matrix_jit.steady_per_call_s": 0.004820128693245352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/curvature_matrix_jit/steady_per_call_s` |
| `measurement/d6bbcf7bcb0a5450210d` | setup_prefix_6_vmap16.steady_per_call_s | {"setup_prefix_6_vmap16.steady_per_call_s": 0.006525131856324151} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6_vmap16/steady_per_call_s` |
| `measurement/ea9aa4359add6a0650d7` | steps.Inversion setup (steps 5-8 combined) | {"steps.Inversion setup (steps 5-8 combined)": 0.032726566400378944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Inversion setup (steps 5-8 combined)` |
| `measurement/f36d49c9eafaf12a2827` | steps.Curvature matrix (F) | {"steps.Curvature matrix (F)": 0.004820128693245352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/steps/Curvature matrix (F)` |
| `measurement/f9504d91d18d40b848c5` | regularization_matrix_jit.steady_per_call_s | {"regularization_matrix_jit.steady_per_call_s": 0.01565156220458448} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/regularization_matrix_jit/steady_per_call_s` |

</details>

### compile

<details><summary>74 recorded compile measurements</summary>

| Record | Metric | Value | Unit | Device / precision | Selection | Evidence pointer |
|---|---|---|---|---|---|---|
| `measurement/03968092c74381c6f301` | ray_trace_data_jit.compile_s | {"ray_trace_data_jit.compile_s": 0.3768580639734864} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_data_jit/compile_s` |
| `measurement/08ca3c3ac874a7ffc74e` | ray_trace_mesh_jit.lower_s | {"ray_trace_mesh_jit.lower_s": 0.07737896195612848} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_mesh_jit/lower_s` |
| `measurement/0a76f4e7ab88860fa345` | nnls_pdip_one_iteration_jit.compile_s | {"nnls_pdip_one_iteration_jit.compile_s": 0.4571617569308728} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_one_iteration_jit/compile_s` |
| `measurement/0f621131c2025f8548ad` | inversion_setup_jit.lower_s | {"inversion_setup_jit.lower_s": 3.3883437330368906} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_jit/lower_s` |
| `measurement/10011b62ce220d1185ef` | setup_prefix_6s_vmap16.first_call_s | {"setup_prefix_6s_vmap16.first_call_s": 4.744781622895971} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s_vmap16/first_call_s` |
| `measurement/15a0e7e55d35ab5b88af` | data_vector_jit.compile_s | {"data_vector_jit.compile_s": 0.05832659010775387} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/data_vector_jit/compile_s` |
| `measurement/16b1ff2ff9fd916fcfb4` | setup_prefix_8.compile_s | {"setup_prefix_8.compile_s": 3.5459788248408586} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8/compile_s` |
| `measurement/1893a286ece426ccf622` | log_det_regularization_jit.lower_s | {"log_det_regularization_jit.lower_s": 0.0003663189709186554} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_regularization_jit/lower_s` |
| `measurement/2a19849940db64a02934` | setup_prefix_6_vmap16.first_call_s | {"setup_prefix_6_vmap16.first_call_s": 4.712920404970646} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6_vmap16/first_call_s` |
| `measurement/2a48e0738c5763c04d7e` | setup_prefix_7_vmap16.first_call_s | {"setup_prefix_7_vmap16.first_call_s": 4.39461158006452} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7_vmap16/first_call_s` |
| `measurement/2be53cd65251912225da` | setup_prefix_8.lower_s | {"setup_prefix_8.lower_s": 0.43772352603264153} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8/lower_s` |
| `measurement/36764295997a6ffc8aea` | reconstruction_jit.compile_s | {"reconstruction_jit.compile_s": 0.4489419630263001} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/reconstruction_jit/compile_s` |
| `measurement/385689ba407cb07dfbc4` | inversion_setup_jit.compile_s | {"inversion_setup_jit.compile_s": 13.598880673991516} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_jit/compile_s` |
| `measurement/393d3835dacd1d34c930` | setup_prefix_7.first_call_s | {"setup_prefix_7.first_call_s": 0.021573967998847365} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7/first_call_s` |
| `measurement/3aa5437a81307904cb94` | setup_prefix_8_vmap16.first_call_s | {"setup_prefix_8_vmap16.first_call_s": 5.070741324918345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8_vmap16/first_call_s` |
| `measurement/406c6a7e39328770b00c` | profile_subtract_jit.first_call_s | {"profile_subtract_jit.first_call_s": 0.000688586151227355} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/profile_subtract_jit/first_call_s` |
| `measurement/4125ec36f6be9be14b68` | setup_prefix_6s.first_call_s | {"setup_prefix_6s.first_call_s": 0.021598057122901082} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s/first_call_s` |
| `measurement/487b8f142ed07a58a9c5` | curvature_matrix_jit.compile_s | {"curvature_matrix_jit.compile_s": 0.0571294748224318} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/curvature_matrix_jit/compile_s` |
| `measurement/4921fc7b021bc8053839` | log_det_curvature_reg_jit.lower_s | {"log_det_curvature_reg_jit.lower_s": 0.019636929035186768} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_curvature_reg_jit/lower_s` |
| `measurement/4af0059492881b5a569a` | setup_prefix_8.first_call_s | {"setup_prefix_8.first_call_s": 0.031200014054775238} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_8/first_call_s` |
| `measurement/4b0273f6e0040e8afc0a` | setup_prefix_6s.compile_s | {"setup_prefix_6s.compile_s": 3.1433044190052897} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s/compile_s` |
| `measurement/4c5e93b5888e1e89706e` | blurred_image_jit.first_call_s | {"blurred_image_jit.first_call_s": 0.0052109998650848866} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/blurred_image_jit/first_call_s` |
| `measurement/58719a936d6ef4c0d622` | cholesky_curvature_reg_jit.compile_s | {"cholesky_curvature_reg_jit.compile_s": 0.10118447593413293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_curvature_reg_jit/compile_s` |
| `measurement/5d71eb84b0e1e84d66aa` | cholesky_curvature_reg_jit.lower_s | {"cholesky_curvature_reg_jit.lower_s": 0.013646982843056321} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_curvature_reg_jit/lower_s` |
| `measurement/6583656e712d68f661be` | setup_prefix_5.compile_s | {"setup_prefix_5.compile_s": 1.0006054339464754} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5/compile_s` |
| `measurement/680713a267feb35de58e` | nnls_pdip_jit.first_call_s | {"nnls_pdip_jit.first_call_s": 0.03873504977673292} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_jit/first_call_s` |
| `measurement/6be2f0df974696e1b960` | regularization_matrix_jit.compile_s | {"regularization_matrix_jit.compile_s": 4.078863521106541} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/regularization_matrix_jit/compile_s` |
| `measurement/6e4062262e5bedce2622` | nnls_pdip_jit.lower_s | {"nnls_pdip_jit.lower_s": 0.024074014043435454} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_jit/lower_s` |
| `measurement/6fc5f75490b772c3ecbb` | lens_image_jit.first_call_s | {"lens_image_jit.first_call_s": 0.0006970670074224472} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/lens_image_jit/first_call_s` |
| `measurement/70d7760b9f3d09a492a8` | log_det_regularization_jit.compile_s | {"log_det_regularization_jit.compile_s": 1.1620111763477325e-05} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_regularization_jit/compile_s` |
| `measurement/7386b9f349474086cb74` | log_evidence_jit.compile_s | {"log_evidence_jit.compile_s": 0.20455138897523284} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_evidence_jit/compile_s` |
| `measurement/77bd19ed2eb17fd5ab39` | inversion_setup_vmap16.first_call_s | {"inversion_setup_vmap16.first_call_s": 21.003029592102394} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_vmap16/first_call_s` |
| `measurement/7b2fb8c28ca1315924d6` | lens_image_jit.compile_s | {"lens_image_jit.compile_s": 0.04071560921147466} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/lens_image_jit/compile_s` |
| `measurement/7de33412df1d29d58dea` | blurred_image_jit.compile_s | {"blurred_image_jit.compile_s": 0.19482357613742352} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/blurred_image_jit/compile_s` |
| `measurement/82df8c20652842e7f2b7` | log_det_regularization_jit.first_call_s | {"log_det_regularization_jit.first_call_s": 0.0012569930404424667} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_regularization_jit/first_call_s` |
| `measurement/899effb2b213e680afe1` | ray_trace_mesh_jit.compile_s | {"ray_trace_mesh_jit.compile_s": 0.47717934707179666} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_mesh_jit/compile_s` |
| `measurement/8d09c84ecb42c4333958` | cholesky_curvature_reg_jit.first_call_s | {"cholesky_curvature_reg_jit.first_call_s": 0.001802078215405345} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_curvature_reg_jit/first_call_s` |
| `measurement/91fc6b714350775072c7` | setup_prefix_7.lower_s | {"setup_prefix_7.lower_s": 0.42736536590382457} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7/lower_s` |
| `measurement/9afdf2d1102b6e3571f8` | setup_prefix_5_vmap16.first_call_s | {"setup_prefix_5_vmap16.first_call_s": 1.2809117638971657} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5_vmap16/first_call_s` |
| `measurement/9e908c55eac72ffe3f72` | lens_image_jit.lower_s | {"lens_image_jit.lower_s": 0.0643011461943388} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/lens_image_jit/lower_s` |
| `measurement/9f4e1eb016d865c842a5` | regularization_matrix_jit.lower_s | {"regularization_matrix_jit.lower_s": 0.49838454299606383} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/regularization_matrix_jit/lower_s` |
| `measurement/a485a68fedad506a4895` | log_evidence_jit.lower_s | {"log_evidence_jit.lower_s": 0.01735679991543293} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_evidence_jit/lower_s` |
| `measurement/a6a67d1b9649a30e73bf` | ray_trace_data_jit.first_call_s | {"ray_trace_data_jit.first_call_s": 0.0019415691494941711} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_data_jit/first_call_s` |
| `measurement/a73b7dc883b231b7dee6` | log_det_curvature_reg_jit.compile_s | {"log_det_curvature_reg_jit.compile_s": 0.09553425805643201} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_curvature_reg_jit/compile_s` |
| `measurement/a8a9b36b8d855dd2e21d` | nnls_pdip_one_iteration_jit.first_call_s | {"nnls_pdip_one_iteration_jit.first_call_s": 0.0046596829779446125} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_one_iteration_jit/first_call_s` |
| `measurement/aba7dbe8fdf3fe120866` | curvature_matrix_jit.lower_s | {"curvature_matrix_jit.lower_s": 0.01119722705334425} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/curvature_matrix_jit/lower_s` |
| `measurement/abdd6d402ca7a3d6358e` | ray_trace_mesh_jit.first_call_s | {"ray_trace_mesh_jit.first_call_s": 0.0013992320746183395} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_mesh_jit/first_call_s` |
| `measurement/ac79196fb74b94d65d2b` | regularization_matrix_jit.first_call_s | {"regularization_matrix_jit.first_call_s": 0.029486222891137004} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/regularization_matrix_jit/first_call_s` |
| `measurement/ae6dda0f6010435dae03` | log_det_curvature_reg_jit.first_call_s | {"log_det_curvature_reg_jit.first_call_s": 0.0018888700287789106} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_det_curvature_reg_jit/first_call_s` |
| `measurement/b20b12a7330dc9bb2599` | inversion_setup_jit.first_call_s | {"inversion_setup_jit.first_call_s": 0.1963862651027739} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/inversion_setup_jit/first_call_s` |
| `measurement/b602c9b69cc3005ea4bc` | setup_prefix_5.first_call_s | {"setup_prefix_5.first_call_s": 0.005017681047320366} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5/first_call_s` |
| `measurement/b6850551bbfa415b2236` | nnls_pdip_vmap16.first_call_s | {"nnls_pdip_vmap16.first_call_s": 0.8916076521854848} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_vmap16/first_call_s` |
| `measurement/b97c62cbe85d95b219f4` | setup_prefix_6s.lower_s | {"setup_prefix_6s.lower_s": 0.578270920086652} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6s/lower_s` |
| `measurement/bc1a008956bc77e3373c` | setup_prefix_6.compile_s | {"setup_prefix_6.compile_s": 3.4552815288770944} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6/compile_s` |
| `measurement/be18fcdd49eba47fbe23` | setup_prefix_11_vmap16.first_call_s | {"setup_prefix_11_vmap16.first_call_s": 5.161567189032212} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_11_vmap16/first_call_s` |
| `measurement/c23fe280a0b1a080b3d5` | profile_subtract_jit.compile_s | {"profile_subtract_jit.compile_s": 0.02732075611129403} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/profile_subtract_jit/compile_s` |
| `measurement/c7c5678708ab92f9db2c` | setup_prefix_5.lower_s | {"setup_prefix_5.lower_s": 0.25830468419007957} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_5/lower_s` |
| `measurement/cb541dc97534a1cd6767` | data_vector_jit.lower_s | {"data_vector_jit.lower_s": 0.006619761930778623} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/data_vector_jit/lower_s` |
| `measurement/d05cb531beae1936b61f` | reconstruction_jit.lower_s | {"reconstruction_jit.lower_s": 0.04209105111658573} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/reconstruction_jit/lower_s` |
| `measurement/d294a871d8f0e429f031` | nnls_pdip_one_iteration_jit.lower_s | {"nnls_pdip_one_iteration_jit.lower_s": 0.023846602998673916} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_one_iteration_jit/lower_s` |
| `measurement/d38ef80b08bece5e12f3` | setup_prefix_6.first_call_s | {"setup_prefix_6.first_call_s": 0.027405745116993785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6/first_call_s` |
| `measurement/d747c2b1f1f9f5985baa` | cholesky_solve_jit.first_call_s | {"cholesky_solve_jit.first_call_s": 0.002082269173115492} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_solve_jit/first_call_s` |
| `measurement/dece9a31a0838655b607` | reconstruction_jit.first_call_s | {"reconstruction_jit.first_call_s": 0.03967002499848604} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/reconstruction_jit/first_call_s` |
| `measurement/e0c0837691bd4a2fe71d` | profile_subtract_jit.lower_s | {"profile_subtract_jit.lower_s": 0.006692930823192} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/profile_subtract_jit/lower_s` |
| `measurement/e16a786d518ef6d98ec2` | curvature_matrix_jit.first_call_s | {"curvature_matrix_jit.first_call_s": 0.005375119857490063} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/curvature_matrix_jit/first_call_s` |
| `measurement/e344b54884a0cf3aa5e6` | cholesky_solve_jit.compile_s | {"cholesky_solve_jit.compile_s": 0.07113801687955856} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_solve_jit/compile_s` |
| `measurement/e52088c70870df052aa7` | setup_prefix_6.lower_s | {"setup_prefix_6.lower_s": 0.41193408286198974} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_6/lower_s` |
| `measurement/e66e774b7df93bef4f32` | log_evidence_jit.first_call_s | {"log_evidence_jit.first_call_s": 0.0033651599660515785} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/log_evidence_jit/first_call_s` |
| `measurement/e6835deed3b624bda2ce` | ray_trace_data_jit.lower_s | {"ray_trace_data_jit.lower_s": 0.12536550988443196} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/ray_trace_data_jit/lower_s` |
| `measurement/e75484d2c40cbd5e5ce5` | data_vector_jit.first_call_s | {"data_vector_jit.first_call_s": 0.0009655940812081099} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/data_vector_jit/first_call_s` |
| `measurement/e79ea9bcafcb1dd1bb96` | cholesky_solve_jit.lower_s | {"cholesky_solve_jit.lower_s": 0.011301114922389388} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/cholesky_solve_jit/lower_s` |
| `measurement/faa5944a9f86da26d6b0` | nnls_pdip_jit.compile_s | {"nnls_pdip_jit.compile_s": 0.4304507579654455} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/nnls_pdip_jit/compile_s` |
| `measurement/fe392aad2fdc40992bee` | blurred_image_jit.lower_s | {"blurred_image_jit.lower_s": 0.05929609388113022} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/blurred_image_jit/lower_s` |
| `measurement/ffc03d9953f988af3814` | setup_prefix_7.compile_s | {"setup_prefix_7.compile_s": 3.405427950900048} | s | a100 / float64 | archive support; unreviewed | [artifact](../../../../results/breakdown/imaging/delaunay_nn_hpc_a100_fp64_recon_split.json) `/jit_phases/setup_prefix_7/compile_s` |

</details>

<details><summary>Recorded hardware, software, method and limitations</summary>

```json
{
  "identity": {
    "backend": "gpu",
    "device": "a100",
    "hardware_details": {
      "autotune_cache_entries_at_start": 0,
      "backend": "gpu",
      "cache_fresh": true,
      "cpu_count": 124,
      "device": "cuda:0",
      "hostname": "euclid-ral-gpu-2",
      "nvidia_smi": "NVIDIA A100 80GB PCIe, 41495 MiB, 81920 MiB",
      "omp_num_threads": null,
      "xla_flags": "--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
    },
    "library_version": "2026.8.17.1",
    "precision": "float64",
    "software": {
      "PyAutoLens": "2026.8.17.1"
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
    "has_provenance": false,
    "host": "euclid-ral-gpu-2",
    "measured_at": null,
    "qualified": false,
    "unknowns": {
      "measured_at": "Not recorded in this legacy evidence; no current default substituted."
    }
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

- [likelihood_breakdown](../../../../scripts/imaging/delaunay_nn/likelihood_breakdown.py)
- [likelihood_runtime](../../../../scripts/imaging/delaunay_nn/likelihood_runtime.py)

## Campaigns

No campaign explicitly bound; campaign applicability is unknown.
