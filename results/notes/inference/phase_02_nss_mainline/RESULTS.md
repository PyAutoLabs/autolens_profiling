# Phase 2 — Global MGE: Nautilus vs mainline BlackJAX NSS: results

Running record for Phase 2 (`../PROGRAMME.md` §4). Entry work landed
2026-08-19/20 (issue #149): the framework wiring and the CP-2 GPU MGE smoke.

## Framework wiring (2026-08-19)

`af.NSS` (mainline `blackjax.nss`, PyAutoFit PR#1492) is wired into the
searches framework: `nss` in `SAMPLER_BUILDERS`, config recorded in the
fork-era JSON shape, leaf scripts for imaging `mge` / `delaunay` /
`pixelization`. Defaults replicate fork history (inner steps 5, num_delete
50, dlogz −3, seed 42) so mainline rows diff cleanly against the recorded
v2026.5.21.1 fork rows; the scan arms drive `SEARCHES_NSS_{N_LIVE,
NUM_MCMC_STEPS,NUM_DELETE,CHUNK_SIZE,TERMINATION,SEED}`. Deliberately not
in `sweep.py` CELLS — pixelized NSS rows cost 19–30 ks and must not creep
into release sweeps.

## CP-2 GPU MGE smoke — laptop RTX 2060 (2026-08-19/20)

**Finding 1 (hardware): the unchunked inner step does not fit a 6 GB
card.** At n_live=200 / num_delete=50 the first jitted step allocates
937 MiB against the ~3.9 GB pool the desktop leaves free →
`RESOURCE_EXHAUSTED`. `chunk_size=10` (the PR#1492 GPU-memory lever,
bit-identical at fixed seed) runs it. Laptop-tier NSS rows therefore
always record a `chunk_size`; A100 rows keep `null`.

**Finding 2 (throughput caveat, expected): GeForce fp64 is 1:32.** The
RTX 2060's fp64:fp32 ratio makes wall times on this tier incomparable to
A100 fp64 (1:2) even beyond the usual cross-tier rule. The laptop row
answers "does mainline `af.NSS` run the production MGE cell end-to-end on
GPU" — the Phase 2 performance scan belongs on the A100.

**Run in flight as of 2026-08-20 ~02:10 UTC** (started 2026-08-20 02:05
UTC): `scripts/imaging/searches/nss/mge.py --config-name local_gpu_fp64`
with `SEARCHES_NSS_CHUNK_SIZE=10` via `~/venv/PyAutoGPU`
(`JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`). Results JSON will land under
`results/searches/nss/imaging/mge/hst/` and this note + the PROGRAMME
CP-2 entry get finalized from it (verdict criteria: completes to dlogz −3;
max logL / logZ consistent with the fork rows' 31786.3–31786.5 /
31697.7–31700.4 band — same seed 42, same knobs, mainline code).

## Next (per PROGRAMME §4 Phase 2)

- The scan: n_live {200, 500, 1000} × num_delete {0.1m, 0.25m, 0.5m} ×
  inner steps {5, 2d, 3d} × dlogz {−3, −10} on laptop-GPU + RAL-CPU, then
  one A100 confirmation + one pixelized probe. H2.1 (logZ bias from
  under-mixing) is the sharpest pre-registered test.
- Gate A judged per model family, not on MGE alone.
