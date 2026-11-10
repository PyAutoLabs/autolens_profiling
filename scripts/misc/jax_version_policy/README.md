# JAX version policy: the XLA:CPU hang matrix

Research for [PyAutoNerves#213](https://github.com/PyAutoLabs/PyAutoNerves/issues/213). Compiled
JAX likelihoods on the CPU backend sometimes stop for good: a later call of a compiled
executable never returns and the process sits at 0 % CPU. These scripts test which jax versions
and which `XLA_FLAGS` settings avoid this, by calling known reproducers many times.

The results are recorded in [PyAutoDNA](https://github.com/PyAutoLabs/PyAutoDNA) as research
evidence (`dna import-matrix`), where the policy decision and the dashboard panel live. Timing
here is a by-product; the question is hang or no hang.

## The reproducers

| Case | What it is | Needs PyAuto? |
|---|---|---|
| `fft` | Scatter-fed FFT convolutions: the Eigen-pool / ducc0 deadlock (PyAutoFit#1530, upstream [jax#41265](https://github.com/jax-ml/jax/issues/41265), fixed by [openxla/xla#50091](https://github.com/openxla/xla/pull/50091)) | no |
| `lapack` | Vmapped Cholesky + `cho_solve`: the path behind the 0.10.x / 0.11.0 exclusion (PyAutoHeart#274) | no |
| `export:<name>` | A real 2d likelihood saved with `jax.export` by `export_reproducers.py` (group MGE prefix 3, group MGE full, multi_galaxy MGE full, multi_galaxy rectangular full) | only to export |
| `library:<dataset>/<variant>` | The same likelihoods through the library, for cells `jax.export` cannot save (Delaunay's host callback) | yes |

## Running it

```bash
# 1. one throwaway venv per jax version (never ~/venv/PyAuto313 or the shared RAL install)
bash make_venvs.sh output/jax_version_policy/venvs 0.9.2 0.10.2 0.11.2 0.12.0 0.12.0.1

# 2. export the likelihoods (PyAuto environment; needs flatbuffers on PYTHONPATH)
python export_reproducers.py

# 3. the matrix: one child process per cell
python run_matrix.py --environment local \
    --venv 0.11.2=output/jax_version_policy/venvs/jax-0.11.2 \
    --venv 0.12.0.1=output/jax_version_policy/venvs/jax-0.12.0.1 \
    --flags none,eigen,sched --cases fft,lapack,export:group_mge_p3 --calls 100 \
    --out output/jax_version_policy/rows/<run>.jsonl

# library cases: the jax under test is a `pip install --no-deps --target` dir put first on PYTHONPATH
python run_matrix.py --environment local --shadow 0.11.2=base \
    --shadow 0.12.0.1=output/jax_version_policy/shadow/jax-0.12.0.1 \
    --cases library:group/delaunay --calls 30 --out ...

# 4. record it in DNA
python -m dna --root ../PyAutoDNA import-matrix output/jax_version_policy/rows/<run>.jsonl
```

`run_matrix.py` watches each child's per-call output. If no call finishes within `--stall`
seconds it samples the child's CPU time: near 0 % is recorded as `hang`, anything else as
`slow` (raise `--stall` and rerun). Rows append as cells finish, so a stopped run keeps its rows.

Laptop runs are pilots only (under 30 minutes). Full repeated matrices go to a RAL CPU array on
the `ral` partition with a private checkout.
