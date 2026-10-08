# Profiling research wiki

This wiki is the navigation layer over six weeks (and counting) of likelihood profiling in this
repo. It answers three questions for every campaign: **why was it run, what did it find, and where
did the result go** (which library PRs, which release). It does not replace the ledgers: the full
record of every measurement stays in the campaign's note under
[`results/notes/`](../results/notes/), and the wiki links to it.

- [`index.md`](index.md) — one row per campaign: question, status, headline, verdict, PRs,
  ledger, Mind epic/contract, next step.
- [`campaigns/`](campaigns/) — one page per campaign, built from
  [`campaigns/_template.md`](campaigns/_template.md).
- [`research/`](research/) — focused research notes that a campaign phase produced for a later
  decision (question, method, evidence, supported explanation, decision table), linked from their
  campaign page: [Jacobi PDIP batched vs unbatched on the A100](research/jacobi_a100_batched_divergence.md)
  (linear-solver phase 4a).

## Header contract

Every campaign page opens with the same ten labelled lines, in this order, then a
`## Journal` section of dated entries (newest last):

| Label | What goes there |
|---|---|
| `**Status:**` | one of `open` / `complete` / `parked` / `no-go` / `draft` |
| `**Question:**` | the question the campaign set out to answer, in one sentence |
| `**Pre-registered rule:**` | the decision rule written down before the deciding run (or "not recorded") |
| `**Verdict:**` | what the rule said, and any human decision that overrode or re-based it |
| `**Headline:**` | the one quotable number, with host and RAL job id |
| `**Library PRs:**` | PRs in PyAutoArray / PyAutoGalaxy / PyAutoLens / PyAutoFit, with release state |
| `**Profiling PRs:**` | PRs in this repo |
| `**Ledger:**` | relative link(s) to the `results/notes/` ledger(s) |
| `**Mind contract:**` | the PyAutoMind epic and prompt path (as code text: Mind is a separate repo) |
| `**Next:**` | the next action, or "none" |

Status vocabulary: `open` (work continues), `complete` (verdict reached, nothing pending),
`parked` (deliberately paused with candidates left), `no-go` (the rule said no), `draft`
(filed in the Mind, not started).

Release states are written only when verified against a library git tag (first tag containing
the merge commit). When unsure, write "not recorded" or "not verified" — never guess.

## Rules

- **Every campaign PR updates its campaign page and its index row** in the same PR: a journal
  entry, the header lines that changed, and the index row's Status / Headline / Next / Last
  updated cells.
- A new `results/notes/*.md` ledger must be linked from at least one wiki page.
- A new campaign page needs a row in `index.md` and all ten header labels.
- Links inside the wiki are repo-relative; no absolute machine paths.

`scripts/misc/tooling/check_wiki.py --check` enforces campaign structure (every ledger
linked, every campaign indexed with the full header), relative links, and generated setup freshness and runs in
`lint.yml` on every PR.

## Exact setup navigation

[Setup index](setups/index.md) provides dataset → model → instrument → exact
configuration navigation generated from the exported catalogue. Each exact page
preserves recorded settings and unknowns, separates measurement axes, distinguishes
selected references from archive support, and shows hardware/software/method metadata.
Advice and hazards appear only for explicitly bound setup IDs. No bound finding means
unknown coverage, never proof that a setup is safe. All imported support is unreviewed.

Run `python scripts/misc/tooling/build_setup_wiki.py` after exporting the catalogue;
`--check` rejects stale/missing/obsolete generated pages and invalid shard checksums or
identities. `build_dashboard.py` regenerates the same pages alongside its catalogue;
its `--check` and the existing `check_wiki.py --check` CI gate enforce freshness.
Generated pages carry a marker; cleanup removes only marked generated filenames.
Campaign journals remain authoritative and are never generated or rewritten.

[`catalogue/wiki_bindings.json`](../catalogue/wiki_bindings.json) explicitly maps
campaign pages to setup IDs; there is no family-name or prose inference. Add bindings
only with supporting evidence. Unbound setups say campaign applicability is unknown.
Historical JSON evidence links open the artifact; the separately printed RFC 6901
pointer identifies the precise measurement (it is not a Markdown/GitHub anchor).
