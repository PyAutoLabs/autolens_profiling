# Where the rulings of record live

The **rulings of record** for the inference programme — what is decided, and therefore
what a run is allowed to assume — live in
[PyAutoCortex](https://github.com/PyAutoLabs/PyAutoCortex), in `projects.yaml`, row
`inference_programme`. That row is the machine-readable body map for this repository: the
remote, the RAL root, the local mirror, the sync CLI and its verbs, the ledger, and the
witness the programme is judged by.

`results/notes/inference/DECISIONS.md` is **scientific commentary**, not the register. It
explains why a decision was reached, what the evidence was, and what it cost — and it
cites ruling ids so a reader can go from the argument to the binding statement. Where the
two disagree, the Cortex ruling is the one that counts. Backfilling the existing
`DECISIONS.md` entries into Cortex rulings was **phase 4 of the cortex-birth epic**, landed
2026-09-01: the map from ledger entry to ruling id is the appended DECISIONS entry
"2026-09-01 — Rulings of record (PyAutoCortex backfill)". Entries written before
2026-08-31 remain commentary with no ruling id.

The laptop mirror at `/mnt/c/Users/Jammy/Science/inference_programme` (where
`hpc/sync pull` puts run outputs) holds **data, not science**: pulled run directories,
staged result rows and SLURM logs. Nothing is decided there, and nothing there is a
source of record.

## Rulings of record filed 2026-09-01

- **R-20260831-01** — the 2026-08-31 programme rewind: mesh/pixelization evidence
  quarantined, restart at Phase 1 under batch-and-review (phase
  `inference_programme/rewind_2026_08_31`, drop).
- **R-20260831-02** — `smc_probe_342018` dropped (2026-08-31-am, rejected wholesale by the
  rewind).
- **R-20260831-03** — `knn_rerun_342016_7` dropped (same).
- **R-20260831-04** — `refs_5_6_342016_56` dropped (same).
- **R-20260831-05** — `slogdet_ab_342017` dropped (same).
- **R-20260901-01** — `mge_pos_ref_reuse` accepted: `340210_9` adopted as the
  `mge_pos_fp64` InferenceRefs_v1 reference.
- **R-20260901-02** — `mge_fp64_retro_baseline` accepted: the retro-adopted `mge_fp64`
  baseline stands under the redo standard.
- **R-20260901-03** — `delaunay_fp64_retro_baseline` dropped (demagnified-source solution);
  carries the binding PositionsLH directive for every mesh/pixelization redo run
  (follow-up autolens_profiling#203).
- **R-20260901-04** — `failed_submissions_342008_10` dropped: carried, then superseded by
  the rewind.

`inference_refs_v1_redo` (the `342091` wave) has **no ruling yet** — it is awaiting the
human's next review slot.
