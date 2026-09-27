# results/logs

Committed SLURM job logs (`.out`), one folder per campaign, named
`<campaign>_<YYYY_MM_DD>_ral_job_<jobid>[_<what>].out` so the RAL job id is in the filename.

A log is **provenance, not a result**: the numbers a campaign quotes live in the result JSON
under `results/breakdown/` or `results/runtime/` and in the campaign's ledger under
`results/notes/`. A ledger cites a log by relative link (`../logs/<campaign>/<file>.out`) when
the log carries something the JSON does not (the node, the load average line, a failed arm's
stderr). Logs that carry nothing beyond the JSON stay on RAL and are cited by job id only.

| Folder | Campaign | Ledger |
|---|---|---|
| `point_source_image/` | point-source image-plane CPU (IP-1 … IP-4c) | [`point_source_cpu_campaign.md`](../notes/point_source_cpu_campaign.md) |
| `point_source_source/` | point-source source-plane (2a … 2e) | [`point_source_source_plane_campaign.md`](../notes/point_source_source_plane_campaign.md) |

The policy — what goes here, what goes beside a result JSON, and what `results/notes/` may
hold — is the "Artefact policy" section of [`../README.md`](../README.md);
`scripts/misc/tooling/check_results_layout.py --check` enforces it in `lint.yml`.
