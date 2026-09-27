# Imaging pixelizations at production over-sampling

**Status:** draft
**Question:** What do the imaging pixelization likelihoods cost at production pixelization over-sampling (sub-size 4 above source S/N 3, 2 below), on HST and on Euclid `EUCLID_VIS_PIX` data, and why does the HST `delaunay_numba` production row read about 2× slower than production?
**Pre-registered rule:** not yet written as a go/no-go rule; the draft's acceptance line (`Witness:`) is an A100 row in `results/runtime/imaging/delaunay/` recording `over_sample_size_pixelization_rule` = {source_snr_cut 3.0, sub_size_above 4, sub_size_below 2} on Euclid data.
**Verdict:** not started
**Headline:** none yet
**Library PRs:** none
**Profiling PRs:** none
**Ledger:** none yet; related [production_representative_cells.md](../../results/notes/production_representative_cells.md)
**Mind contract:** `draft/research/autolens_profiling/imaging_production_over_sampling.md` (filed 2026-09-27); precursor `complete/2026/09/profiling-production-representative.md` (#235 / #236).
**Next:** issue when scheduled

## Why this campaign

A Euclid session on 2026-09-27 compared a 1.5 s cold likelihood evaluation from a real Euclid fit
with this repo's imaging headlines and found them not comparable. The JAX Delaunay runtime cell
hard-codes `over_sample_size_pixelization=1`, and the A100 headline row (65 ms per eval,
AdaptSplit) is HST data with `[4, 2, 2]` light-profile bins and no pixelization over-sampling
rule recorded. Euclid production (`EUCLID_VIS_PIX`, vis_pix stage, job 342301) runs the 4/2 S/N
rule with `[4, 4, 2]` bins; the HST subhalo preset uses the same 4/2 rule. The AdaptSplit vs
ConstantSplit gap on the A100 (~5 ms, 65 vs 60 ms) does not explain the difference.

The precursor, #235 / #236 (completed 2026-09-08), moved only the four CPU numba cells onto the
production presets and left "GPU production-representativeness for the JAX cells" unfiled; this
draft is that follow-up. Its prompt also records that the fixed-lens-light numba result
(932 → 459 ms, job 343311) was taken on the old cell settings, and that the one production-preset
numba HST Delaunay row reads 1.46 s cold / 1.97 s warm against production job 342311's 0.52–0.81 s.

## Phases

Planned, from the draft's `## What`; none has been issued.

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 Production presets in the JAX cells | not started | Can the JAX imaging pixelization cells (runtime + breakdown; Delaunay family, rectangular where relevant) take `--instrument euclid\|hst` presets with the 4/2 rule? | the legacy flat-1 setting stays selectable so historic rows reproduce | | | |
| 2 A100 headline re-measure | not started | What is the A100 JAX headline at production over-sampling, on HST and on Euclid `EUCLID_VIS_PIX` data? | the draft's `Witness:` (A100 Delaunay runtime row recording the 4/2 S/N rule on Euclid data) | | | |
| 3 Fixed-light numba production row | not started | How long does HST take with the lens light fixed, at production settings (Euclid if cheap)? | not recorded | | | |
| 4 `delaunay_numba` HST witness gap | not started | Is the 1.46 s vs 0.52–0.81 s gap the cell, the host or the reference? | not recorded | | | |
| 5 Label headlines | not started | Do headline numbers state over-sampling, preset and dataset, with flat-1 rows labelled? | not recorded | | | |

## What shipped and where it is

Nothing yet. Library changes to the over-sampling code are out of scope for the draft.

## Open / parked / drafts

- `draft/research/autolens_profiling/imaging_production_over_sampling.md` — this campaign, unstarted.
- Out of scope in the draft: timing of the fitting library's summary-time evaluation (the 1.5 s figure being one cold call in the main process).

## Caveats

- **Witness label mismatch.** The draft calls the HST `delaunay_numba` production row a witness **FAIL**; the ledger's table labels the same row (1.463 s cold) "above range", and the precursor record attributes the Delaunay misses to dataset realism (simulated mask vs real VIS cut-out) and host, not configuration. Phase 4 is meant to settle it.
- **Laptop rows.** The precursor's production-representative numbers are a laptop i9-10885H, memo off; they are not RAL figures and do not transfer to the A100 JAX cells.
- **Cross-page link.** The 932 → 459 ms fixed-light figure belongs to [Fixed-light numba CPU](fixed_light_numba_cpu.md) and was measured on the pre-#235 cell settings.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the draft's motivation (the Euclid 1.5 s comparison), its five planned phases,
and its acceptance line (the `Witness:` A100 row), with #235 / #236 cited as the CPU-cell precursor.
Resolved: the "not yet written" rule is now the draft's witness, stated as such. Open: the witness
FAIL vs "above range" label for the HST numba row; the campaign is unissued.
