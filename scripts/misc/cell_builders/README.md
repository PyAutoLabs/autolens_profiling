# cell_builders

Model and dataset setup that more than one script needs, kept in one place so
the scripts build exactly the same thing.

- `imaging_mge.py` prepares the masked imaging dataset and builds the MGE model
  and analysis. `scripts/imaging/mge/likelihood_runtime.py` and the
  [`jax_compile/`](../jax_compile/README.md) probes both use it.
- `galaxy_group.py` simulates the multi_galaxy (hst) and group (euclid) datasets, masks them and
  builds the likelihood-lifecycle 2d cells (workspace MGE model, standard-size pixelization)
  for every setup of `scripts/{multi_galaxy,group}/` through
  `likelihood_breakdown/galaxy_group_standard.py` (autolens_profiling#473).

Nothing here runs on its own or writes results (`galaxy_group.py` writes only its datasets). The tests are in
[`jax_compile/test/`](../jax_compile/test/test_current_compile_builder.py).
