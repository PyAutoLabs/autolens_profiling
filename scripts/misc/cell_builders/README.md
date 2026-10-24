# cell_builders

Model and dataset setup that more than one script needs, kept in one place so
the scripts build exactly the same thing.

- `imaging_mge.py` prepares the masked imaging dataset and builds the MGE model
  and analysis. `scripts/imaging/mge/likelihood_runtime.py` and the
  [`jax_compile/`](../jax_compile/README.md) probes both use it.

Nothing here runs on its own or writes results. The tests are in
[`jax_compile/test/`](../jax_compile/test/test_current_compile_builder.py).
