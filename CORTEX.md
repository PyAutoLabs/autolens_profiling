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
`DECISIONS.md` entries into Cortex rulings is **phase 4 of the cortex-birth epic**; until
that lands, entries written before 2026-09-01 have no ruling id to cite yet.

The laptop mirror at `/mnt/c/Users/Jammy/Science/inference_programme` (where
`hpc/sync pull` puts run outputs) holds **data, not science**: pulled run directories,
staged result rows and SLURM logs. Nothing is decided there, and nothing there is a
source of record.
