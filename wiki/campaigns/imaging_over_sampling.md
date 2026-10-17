# Imaging pixelizations at production over-sampling

**Status:** open
**Question:** What do the imaging pixelization likelihoods cost at production pixelization over-sampling (sub-size 4 above source S/N 3, 2 below), on HST and on Euclid `EUCLID_VIS_PIX` data, and why does the HST `delaunay_numba` production row read about 2× slower than production?
**Pre-registered rule:** not yet written as a go/no-go rule; the draft's acceptance line (`Witness:`) is an A100 row in `results/runtime/imaging/delaunay/` recording `over_sample_size_pixelization_rule` = {source_snr_cut 3.0, sub_size_above 4, sub_size_below 2} on Euclid data.
**Verdict:** phase 1 shipped for review (presets, no measurement); phases 2–5 not started
**Headline:** none yet
**Library PRs:** none
**Profiling PRs:** issue #430 (phase 1)
**Ledger:** none yet; related [production_representative_cells.md](../../results/notes/production_representative_cells.md)
**Mind contract:** `draft/research/autolens_profiling/imaging_production_over_sampling.md` (filed 2026-09-27); precursor `complete/2026/09/profiling-production-representative.md` (#235 / #236).
**Next:** phase 2, the A100 re-measure with `--over-sampling production` on HST and Euclid (needs a separate compute authorisation)

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
| 1 Production presets in the JAX cells | 2026-10-17 | Can the JAX imaging pixelization cells (runtime + breakdown; Delaunay family, rectangular where relevant) take `--instrument euclid\|hst` presets with the 4/2 rule? | the legacy flat-1 setting stays selectable so historic rows reproduce | Delaunay + DelaunayNN runtime and breakdown cells take `--over-sampling production`, resolved by `--instrument`; flat-1 stays the default. Laptop CPU functional smoke only, no rows recorded. Rectangular not done | none | #430 |
| 2 A100 headline re-measure | not started | What is the A100 JAX headline at production over-sampling, on HST and on Euclid `EUCLID_VIS_PIX` data? | the draft's `Witness:` (A100 Delaunay runtime row recording the 4/2 S/N rule on Euclid data) | | | |
| 3 Fixed-light numba production row | not started | How long does HST take with the lens light fixed, at production settings (Euclid if cheap)? | not recorded | | | |
| 4 `delaunay_numba` HST witness gap | not started | Is the 1.46 s vs 0.52–0.81 s gap the cell, the host or the reference? | not recorded | | | |
| 5 Label headlines | not started | Do headline numbers state over-sampling, preset and dataset, with flat-1 rows labelled? | not recorded | | | |

## What shipped and where it is

Phase 1 (#430, under review): a shared `--over-sampling {flat1,production}` flag (`_profile_cli.py`,
default `flat1`) read by the four JAX Delaunay-family imaging cells
(`scripts/imaging/delaunay{,_nn}/likelihood_{runtime,breakdown}.py`). `production` applies
`_production_config.apply_production_over_sampling_jax` for `--instrument euclid` or `hst`. The
breakdown cells now read `--instrument`; HST is still the default. Library changes to the
over-sampling code are out of scope for the draft.

## Open / parked / drafts

- `draft/research/autolens_profiling/imaging_production_over_sampling.md` — this campaign, unstarted.
- Out of scope in the draft: timing of the fitting library's summary-time evaluation (the 1.5 s figure being one cold call in the main process).

## Caveats

- **Witness label mismatch.** The draft calls the HST `delaunay_numba` production row a witness **FAIL**; the ledger's table labels the same row (1.463 s cold) "above range", and the precursor record attributes the Delaunay misses to dataset realism (simulated mask vs real VIS cut-out) and host, not configuration. Phase 4 is meant to settle it.
- **Laptop rows.** The precursor's production-representative numbers are a laptop i9-10885H, memo off; they are not RAL figures and do not transfer to the A100 JAX cells.
- **The numba cells' production threshold is not production's** (found in phase 1, not fixed).
  `apply_production_over_sampling` thresholds the cell's adapt image after the preset's S/N cap.
  That image is the cached truth lensed source, which is flux and not S/N (`_adapt_image_util`). For
  HST the cap (3.0) equals the cut (3.0), so no pixel exceeds it and the whole mask runs at sub-size 2.
  Both production scripts threshold the uncapped source S/N map. The HST `delaunay_numba` production
  row therefore has less pixelization over-sampling than production, so this cannot explain its 2×
  gap. Phase 4 has to account for it.
- **Results-folder keying for phase 2.** The dashboard keys a runtime row by folder and instrument,
  not by script name, so a production row written beside the flat-1 rows would join their trend.
  Production runs default to an `os_production/` subfolder for this reason. Phase 2 should keep that
  folder, or pass its own `--output-dir`.
- **Cross-page link.** The 932 → 459 ms fixed-light figure belongs to [Fixed-light numba CPU](fixed_light_numba_cpu.md) and was measured on the pre-#235 cell settings.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the draft's motivation (the Euclid 1.5 s comparison), its five planned phases,
and its acceptance line (the `Witness:` A100 row), with #235 / #236 cited as the CPU-cell precursor.
Resolved: the "not yet written" rule is now the draft's witness, stated as such. Open: the witness
FAIL vs "above range" label for the HST numba row; the campaign is unissued.

### 2026-10-17 — phase 1: production over-sampling presets in the JAX cells (#430)

Added `--over-sampling {flat1,production}`, default `flat1`, to the Delaunay and DelaunayNN JAX
runtime and breakdown cells. `production` takes its values from `_production_config`, re-read
against the production scripts on 2026-10-17:

| `--instrument` | preset | light-profile bins | pixelization rule | source |
|---|---|---|---|---|
| `euclid` | `EUCLID_VIS_PIX` | `[4, 4, 2]` at `[0.1, 0.3]` | 4 where source S/N > 3.0, else 2 | `euclid_strong_lens_modeling_pipeline@9b1bc4c` `scripts/initial_lens_model.py:614-626`, `util.py:1505-1512` |
| `hst` | `HST_SOURCE_PIX_2` | `[4, 2, 2]` at `[0.1, 0.3]` | 4 where source S/N > 3.0, else 2 | `subhalo_validation@ab4bf32` `scripts/imaging.py:175-184`, `:213-255` |

Both production scripts threshold the uncapped source S/N adapt map. The profiling datasets cache
the truth lensed source as flux, so the cells use `lensed_source / noise_map`. This puts 2022 of
15361 masked HST pixels and 1057 of 3841 Euclid pixels at sub-size 4. Only the over-sampling
changes. The cell's mesh, MGE and regularization are not production-matched. Production rows
record `configuration.over_sample_size_pixelization_rule` (the witness field), skip the flat-1
log-evidence pins and default to an `os_production/` results subfolder. Flat-1 rows now record
`{flat_sub_size: 1}`.

Verified with a laptop CPU functional smoke and no timings: all eight production runs (4 cells ×
2 instruments) and a sparse HST runtime completed. A throwaway catalogue build with those JSONs
placed under `results/` passed the Pulse v2 contract check. The dashboard keyed every one as its
own series. `flat1` output is bit-identical to unmodified `main`. Open: the HST flat-1
`adapt_split` eager pin already drifts on `main` (got 29155.000174, pinned 29140.296, 5.0e-4
relative; the breakdown cell asserts and exits 1, the runtime cell records `pinned_drift`). The
value equals the pre-#235 pin, so it needs its own re-pin or investigation. Also open: the numba
threshold caveat above, and the rectangular JAX cells.
