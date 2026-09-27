# HST GPU non-solver residue

**Status:** complete
**Question:** Around the certified solve, what else costs time in the HST A100 Delaunay likelihood, and is there an fp64 lever?
**Pre-registered rule:** per phase; phase 3 used a pre-registered 1e-9 gate
**Verdict:** no fp64 lever (phases 1–4); phase-2 batching verdict inconclusive; 5.44 ms qhull `pure_callback` idle routed elsewhere
**Headline:** budget-7 single-call baseline 31.64 ms on the A100; jobs 343376, 344635, 350573, 350651
**Library PRs:** none
**Profiling PRs:** #270, #294, #296, #306 (issues #273, #295, #303)
**Ledger:** [hst_gpu_residue_phase1_2026_09.md](../../results/notes/hst_gpu_residue_phase1_2026_09.md), [hst_gpu_residue_phase2_vmap_2026_09.md](../../results/notes/hst_gpu_residue_phase2_vmap_2026_09.md), [hst_gpu_residue_phase3_psf_2026_09.md](../../results/notes/hst_gpu_residue_phase3_psf_2026_09.md), [hst_gpu_residue_phase4_logdet_2026_09.md](../../results/notes/hst_gpu_residue_phase4_logdet_2026_09.md)
**Mind contract:** epic `hst-gpu-non-solver-residue`; `complete/2026/09/hst-gpu-non-solver-residue.md`
**Next:** none

Backfill pending (phase 2 of epic profiling-research-wiki).

- [hst_gpu_residue_phase1_2026_09.md](../../results/notes/hst_gpu_residue_phase1_2026_09.md)
- [hst_gpu_residue_phase2_vmap_2026_09.md](../../results/notes/hst_gpu_residue_phase2_vmap_2026_09.md)
- [hst_gpu_residue_phase3_psf_2026_09.md](../../results/notes/hst_gpu_residue_phase3_psf_2026_09.md)
- [hst_gpu_residue_phase4_logdet_2026_09.md](../../results/notes/hst_gpu_residue_phase4_logdet_2026_09.md)

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.
