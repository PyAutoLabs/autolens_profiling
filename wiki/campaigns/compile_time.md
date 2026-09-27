# JAX compile time

**Status:** complete
**Question:** Is JAX compile time a problem for the endorsed multi-start gradient searches across MGE and pixelized meshes?
**Pre-registered rule:** not recorded
**Verdict:** `af.MultiStartProdigy` compile time is a non-problem on every endorsed model type; multi-band batching hoisted to a Python loop in `MultiStartGradient` (PyAutoFit#1430)
**Headline:** worst case ~3.5 min cold on a 1-core laptop
**Library PRs:** PyAutoFit#1430 (release not verified)
**Profiling PRs:** not recorded (issue #93)
**Ledger:** [multistart_prodigy_compile_census.md](../../results/notes/multistart_prodigy_compile_census.md), [multiband_pyloop_productized.md](../../results/notes/multiband_pyloop_productized.md)
**Mind contract:** not recorded
**Next:** A100 tier outstanding (per census note)

Backfill pending (phase 2 of epic profiling-research-wiki).

- [multistart_prodigy_compile_census.md](../../results/notes/multistart_prodigy_compile_census.md)
- [multiband_pyloop_productized.md](../../results/notes/multiband_pyloop_productized.md)

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.
