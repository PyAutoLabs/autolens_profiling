# Image ↔ source plane mappings

**Status:** complete
**Question:** Restore image ↔ source plane mapping visuals (`subplot_mappings`), ShapeSolver validation and the guide.
**Pre-registered rule:** not recorded (feature work; no timing rule — each phase carried test and verification gates, see the table)
**Verdict:** 3 phases shipped 2026-09-02/03 (feature work, not a timing campaign). ShapeSolver validated for the first time: per-image magnifications within 0.2 % of the analytic SIS, quad positions within 0.034″ of `PointSolver`; its JAX path now raises `NotImplementedError` rather than returning a wrong answer.
**Headline:** not a timing campaign
**Library PRs:** PyAutoArray#517, PyAutoArray#518, PyAutoLens#720 (all released 2026.9.4.1).
**Profiling PRs:** none
**Ledger:** none in `results/notes/`
**Mind contract:** epic `image-source-mappings`; `complete/archive/epics/image_source_mappings_epic.md`; records `complete/2026/09/image-source-mappings-p{1,2,3}.md`
**Next:** none (library follow-ups in `draft/bug/autoarray/mapping_overlay_follow_ups_forward_regions_throu.md`)

## Why this campaign

The mapping-colouring feature (source pixels and their matched image-plane counterparts drawn in one
colour) died in PyAutoArray `0cb75ebd` (2025-07-21), and by 2026-09 `aplt.subplot_mappings` computed
peak source pixels, discarded them and drew a plain 2×2 subplot; `tutorial_2_mappers.py` self-flagged
"VISUALS SLIGHTLY BUGGY". On 2026-09-02 the human asked for a clean one-look figure showing how the
brightest source regions map between planes, for pixelized **and** parametric sources, with per-clump
colours, plus the brightest image-plane coordinate of each image for spectroscopic fibre pointing (4MOST)
and a step-by-step guide (verbatim request in the epic ledger).

The human's decisions (2026-09-02): three phases, one issue and PR each, library-first; ShapeSolver is
the source → image engine for non-pixelized sources, so phase 2 doubles as the validation it never had;
WCS stays guide-level. It ran alongside [Numpy deflections CPU](numpy_deflections_cpu.md) by human
decision. This page is indexed for completeness: it is not a profiling campaign and has no ledger here.

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| 1 PyAutoArray regions + clumps | 2026-09-02 | Plotting-agnostic `Mapping` / `ImageRegion` objects, source clump finding, a generic `regions=` overlay, a restored `subplot_mappings` | not recorded (tests + verification in the prompt; must merge and release before phase 2) | new `autoarray/inversion/mappings/` package; `Inversion.source_clumps_from`, `mappings_from`; 2×2 colour-matched `subplot_mappings`; optional `Shape.contains` / `boundary` deferred to phase 2a; `test_autoarray` 1382 passed | none | PyAutoArray#517 (issue PyAutoArray#515) |
| 2 + 2a ShapeSolver engine | 2026-09-02 | ShapeSolver as the parametric source → image engine, with the validation suite it never had | not recorded as a go/no-go; verification: magnifications against analytic SIS, oracle agreement, positions against `PointSolver` | `Shape.contains` / `boundary` plus three defects fixed (reflected-triangle containment, transposed `for_limits_and_scale` tiling that made `PointSolver` miss images on non-square grids, `plot_regions` labels); `al.mappings`, `subplot_fit_imaging_mappings`; SIS magnifications within 0.2 %, positions within 0.034″ | none | PyAutoArray#518, PyAutoLens#720 (issue PyAutoLens#719) |
| 3 guide + tutorials | 2026-09-03 | `guides/mappings.py`, the `tutorial_2_mappers` rewrite, dead index sections fixed | not recorded (docs; gated on both releases, waived by the human 2026-09-03) | new guide (point, parametric and pixelized mappings, WCS for fibre pointing); HowToLens / HowToGalaxy tutorials draw `regions=`; the planned `total_mappings_pixels` config sweep was a no-op | none | autolens_workspace#526, HowToLens#76, HowToGalaxy#72, autogalaxy_workspace#232 (issue autolens_workspace#525) |

## What shipped and where it is

| PR | What | Merge | Release |
|---|---|---|---|
| PyAutoArray#517 | `inversion/mappings/` package, `source_clumps_from`, `regions=` overlay, rewritten `subplot_mappings` | `501c373f` | 2026.9.4.1 |
| PyAutoArray#518 | `Shape.contains` / `Shape.boundary`; reflected-triangle containment, transposed tiling and `plot_regions` fixes | `c9f67e78` | 2026.9.4.1 |
| PyAutoLens#720 | ShapeSolver validation suite and audit fixes, `image_regions_from`, `mapping_from`, per-image magnification, `al.mappings`, fit-level `subplot_mappings` | `091fbdff` | 2026.9.4.1 |

## Open / parked / drafts

- `draft/bug/autoarray/mapping_overlay_follow_ups_forward_regions_throu.md` — `autolens.plot` / `autogalaxy.plot` wrappers do not forward `regions=` / `indexes=`; degenerate `RectangularBilinearAdaptDensity` edge cells map to nothing silently; per-polygon labels; `plot_mapper` guards.
- ShapeSolver under JAX: the strict xfail `test_shape_solver.py::test_jax_and_numpy_kept_triangles_agree` is the trigger to re-enable it once the containment cap is lifted; no draft names it.

## Caveats

- **Not a timing campaign**: no profiling PR, no ledger, no RAL job. Numbers above are test verifications.
- **Release discrepancy settled by the release sheet.** The epic's retirement line (2026-09-14) says the blocking releases landed as v2026.9.11.1 / v2026.9.14.1; tag containment puts all three merges in 2026.9.4.1.
- **ShapeSolver under JAX was silently wrong**: `MAX_CONTAINING_SIZE = 15` capped each step, giving magnification 0.13 against a true 6.86; JAX `area` also cannot survive `jit`. [Point-source image-plane CPU](point_source_image_plane_cpu.md) IP-4c later raised the cap to 20 (PyAutoArray#584, UNRELEASED); whether that un-xfails the ShapeSolver test is not verified.
- Engine A's `magnifications_from` returns relative flux fractions, not engine B's absolute scale.
- Pre-existing silent `except` guards remain in `subplot_mappings` and `_plot_source_plane`.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records the three feature phases from the Mind records, with releases verified (2026.9.4.1 for
#517, #518, #720; the epic's own v2026.9.11.1 / v2026.9.14.1 claim is superseded). Phase 1 has no profiling
note: PyAutoArray issue #515 → PR #517 (merged `501c373f`, 2026-09-02) added the `inversion/mappings/`
package, source clump finding, the `regions=` overlay and a restored 2×2 `subplot_mappings`. Open: the
ShapeSolver JAX path stays disabled pending the containment cap.
