# Mass-field profiling builders

**Status:** complete
**Question:** Move the live profiling builders, and the staged pipeline-resume stages, from galaxy-attached shear to top-level `MassField` models.
**Pre-registered rule:** not recorded as a go/no-go (maintenance sweep). The live-builder prompt required its named witness scripts and `profile_registry_coverage.json` to be preserved exactly and committed result rows never rewritten; the resume prompt's witness: each of the five resumed stage models holds the expected free or fixed top-level field, and the resume smoke completes without changing committed result artifacts.
**Verdict:** 47 live builders and all 5 staged resume calls migrated; 5 inline historical witnesses deliberately left unchanged (maintenance sweep, not a timing campaign). CI's timing-only real-likelihood threshold was relaxed 3.0 % → 3.1 % with human approval.
**Headline:** not a timing campaign
**Library PRs:** none (consumer sweep). It depends on PyAutoGalaxy#621, PyAutoLens#742 (`MassField`, released 2026.9.19.1) and, for the resume stages, PyAutoGalaxy#625, PyAutoLens#745 (`mass_and_fields_from`, released 2026.9.26.1, consumed from source `main` before release).
**Profiling PRs:** #288 (issue #287), #290 (issue #289).
**Ledger:** none in `results/notes/`
**Mind contract:** epic `mass-field` (phase 11 of `complete/archive/epics/mass_field_epic.md`); `complete/2026/09/mass-field-profiling-live.md`; `complete/2026/09/mass-field-pipeline-resume.md`
**Next:** none

## Why this campaign

The `mass-field` epic (filed 2026-09-17) made external shear, mass sheets and external potentials a
standalone `MassField` with its own `fields=` model slot and tracer argument, by human ruling ("MassField
is its own thing"; the user-facing API is `fields=` everywhere). Once the libraries and workspaces moved,
the human asked on 2026-09-19 to find where the update had not reached and finish it. An AST audit that day
found 57 galaxy-attached shear calls under `scripts/`: 5 inline historical witnesses, 5 staged calls in
`scripts/misc/pipeline_resume/slam_resume.py`, and 47 other live builders.

This repo's share is epic phase 11, done as two PRs: the 47 independent builders first, then the chained
resume stages once the chaining helper had merged. The repo was claimed by `hst-gpu-residue-p2` at the
time; the human waived the conflict for a separate worktree on disjoint paths (recorded in the record).

## Phases

| Phase | Dates | Question | Pre-registered rule | Result | Jobs | PRs |
|---|---|---|---|---|---|---|
| Live builders | 2026-09-19 | Move 47 live non-chaining builders to a top-level `MassField` via flat `fields=` and manually built tracers | preserve the named witness scripts and the registry coverage artifact exactly; never rewrite committed result rows; AST audit, Ruff, README, smoke, one representative runtime script | 47 migrated; 5 inline witnesses and 5 staged calls left; 665 passed, 5 skipped; imaging and interferometer runtime witnesses run; CI real-likelihood timing threshold 3.0 % → 3.1 % (approved) | none | #288 (issue #287, merge `5584028a`) |
| Pipeline resume | 2026-09-19 | Carry one top-level field through the five `slam_resume.py` stages | each stage model holds the expected free or fixed field; resume smoke completes without changing committed artifacts | 5 stages migrated (free field in source LP, `mass_and_fields_from` in source PIX 1 and mass total, fixed instance fields in source PIX 2 and light LP); zero staged Galaxy shear sites; 667 passed, 5 skipped | none | #290 (issue #289, merge `a8bee1d1`) |

## What shipped and where it is

No library PR shipped from this sweep; both PRs are in this repo. The library APIs it consumes are listed
under **Library PRs** above.

## Open / parked / drafts

- none for this repo.
- `draft/maintenance/autolens_profiling/mass_field_flat_adoption_science_repos.md` — the original four-repo survey, marked historical in its 2026-09-19 update ("do not issue this broad prompt as a single task").

## Caveats

- **Not a timing campaign**: no ledger, no RAL job, no headline.
- **One CI gate was loosened.** The timing-only real-likelihood threshold moved from 3.0 % to 3.1 % so #288 could pass; the record calls it the approved minimal relaxation.
- **Five inline historical witnesses still use galaxy-attached shear** by design, so the galaxy-attached form still appears in `scripts/`. The prompt's preserve list names `fixed_light_numba_s4_witness.py`, `*_levers_l{2,3}_witness.py`, `fixed_light_draws.py` and `fixed_light_trace.py`; the record does not map the five sites to those files.
- The live-builder record says the staged pipeline "remains tracked in" `draft/maintenance/autolens_profiling/mass_field_pipeline_resume.md`; that draft shipped the same day as #290 and no longer exists as a draft.
- The resume stages ran against current PyAutoGalaxy/PyAutoLens source `main`, before PyAutoGalaxy#625 / PyAutoLens#745 were released.

## Journal

### 2026-09-27 — stub created

Header filled from the ledger and the verified facts sheet; body pending backfill.

### 2026-09-27 — backfilled (phase 2)

Page now records both halves of mass-field epic phase 11: the 47 live builders (#287 → #288, merge
`5584028a`) and the five staged resume stages (#289 → #290, merge `a8bee1d1`), which the stub omitted.
Library dependencies are listed with verified releases. Nothing remains open for this repo.
