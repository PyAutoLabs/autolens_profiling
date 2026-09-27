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

`scripts/misc/tooling/check_wiki.py --check` enforces the three structural rules (every ledger
linked, every page indexed with the full header, every relative link resolves) and runs in
`lint.yml` on every PR.
