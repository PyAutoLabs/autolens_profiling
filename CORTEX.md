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

## The rulings themselves

There is no hand-maintained list here — it went stale, and a copy of a ledger is
not a ledger. Read the rulings where they live:

- **`PyAutoCortex/rulings/<YYYY>/<MM>/`** — every ruling file, append-only.
- **<https://pyautolabs.github.io/PyAutoCortex/>** — the Cortex board: what is
  awaiting a ruling, what is running, what has been ruled, by project.
- **`PyAutoCortex/phases/inference_programme/`** — this project's phases; each
  one's `Ruling:` header names its chain head.

Two ledgers of commentary sit in this repo. `results/notes/inference/PROGRAMME.md`
is the **retired** `jax-inference-profiling` programme (frozen 2026-09-04,
R-20260904-01); `results/notes/gradient_slam/LEDGER.md` is its live successor,
the `gradient-slam-baseline` epic.
