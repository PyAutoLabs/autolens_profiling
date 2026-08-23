# Phase 0 — reconnaissance remainder & unblocking: results

Running record for the Phase 0 work items (`../PROGRAMME.md` §4, Phase 0).
Items (a), (d), (e) closed elsewhere (see the phase-state table); this note
records item (b) and will absorb (c) when the RAL harvest runs.

## 0(b) — blackjax ≥1.6.2 validation + mainline NSS smoke (2026-08-18)

**Verdict: environment half of CP-2 validated on cloud CPU — blackjax 1.6.2
installs cleanly next to released autofit, mainline `blackjax.nss` passes a
2D analytic-evidence smoke, and `af.BlackJAXNUTS` runs on 1.6.2 unmodified.**
Issue #142. Human-directed scope (2026-08-18): install/API validation only —
no GPU, no profiling meaning in any timing below. Remaining for laptop/RAL
sessions: apply the same upgrade to the local venvs and the RAL stack and
re-run `scripts/misc/searches/nss_smoke.py` there, plus the GPU MGE smoke
half of CP-2.

### Environment

Clean venv, Python 3.11 (cloud; PyAuto first-class support is 3.12/3.13 — a
warning, not a failure): `pip install "blackjax>=1.6.2" "jax[cpu]" autofit`
resolved **blackjax 1.6.2, jax 0.10.2, autofit 2026.7.29.1** with no
conflicts (`pip check` clean). Two dependency facts worth recording:

- **PyAutoFit does not cap blackjax.** `pyproject.toml` on main pins
  `blackjax>=1.2.0` — a floor, no ceiling — and only in the `optional`
  extra. The Gate-A "floor ≥1.6.2" change (`../PROGRAMME.md` §7) is a
  one-line bump with no cap to fight.
- **blackjax is not a hard autofit dependency** (the released wheel's
  `Requires` omits it), so environment upgrades must install/upgrade blackjax
  explicitly — `pip install -U autofit` alone will never deliver 1.6.

### `blackjax.nss` 2D toy smoke — PASS

Gaussian likelihood N([1, −1], 0.5²I) × uniform prior on [−5, 5]² →
analytic logZ = −log 100 = −4.605. Mainline API exercised end-to-end
(the surface Phase 2's profiling-local runner wires):

- `blackjax.nss(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete)`
  — native-space `logprior_fn`, no unit-cube transform.
- `algo.init(particles, rng_key=...)`; jitted `algo.step`.
- dlogz termination read from `state.integrator.logZ_live − logZ` (−3 used).
- `ns.utils.finalise(live, dead)` → `log_weights(key, dead, shape=100)`
  simulated-volume ensemble (logZ per draw = logsumexp over the particle
  axis) → **logZ = −4.609 ± 0.069 vs analytic −4.605**.
- `ns.utils.sample` posterior draws (positions under `.position`):
  mean (0.993, −1.002), σ (0.511, 0.500) vs truth (1, −1), σ 0.5.

500 live points, num_delete 10, num_inner_steps 5 (= max(5, 2·d) at d=2);
356 steps, ~3 s CPU (toy scale, no profiling meaning).

**H2.1 provenance check:** the installed 1.6.2 docstring states the
inner-steps rule and the bias direction verbatim — "use
`num_inner_steps >= max(5, 2 * dim)` ... bare `dim` ... can bias the
evidence *upward* for `dim > 10`" — upgrading the citation in
`../PROGRAMME.md` §2.2 from release-notes reading to verified-in-source.

### `af.BlackJAXNUTS` against blackjax 1.6.2 — PASS

Released autofit 2026.7.29.1 wrapper runs **unmodified**: the API surface it
uses (`blackjax.window_adaptation(blackjax.nuts, ...)`, `blackjax.nuts(...)`
kernel `.step`, `blackjax.diagnostics.effective_sample_size`) is all intact
in 1.6.2. Toy 2D posterior (`use_jax=True` Analysis): median (0.88, −0.89)
vs truth (1, −1) within the 200-warmup/300-sample budget, 0 divergences,
acceptance 0.92, ESS_min 244. No PyAutoFit source change needed for 1.6
compatibility — the §7 Phase-6 items (multi-chain, mass-matrix/start-point
injection) remain features, not fixes.

### Reproduction

`python scripts/misc/searches/nss_smoke.py` (exits non-zero on FAIL; guards
on blackjax < 1.6). Run it in each environment after its blackjax upgrade.

## 0(b) — local venv + RAL stack upgrades (2026-08-19)

**Verdict: the remaining environment half of 0(b) executed — local venv
upgraded 1.5 → 1.6.2 with the full smoke PASS (now including `af.NSS`
end-to-end); RAL stack upgraded off the obsolete 2026-01 fork build.**

### PyAutoFit PR#1492 merged

The re-mainlined `af.NSS` merged 2026-08-18 (PyAutoFit#1491 / PR#1492), so
this session extended `scripts/misc/searches/nss_smoke.py` with a third
check: `af.NSS` end-to-end through `search.fit` on the same 2D analytic-
evidence toy. Absence of `af.NSS` is a FAIL, not a skip — both profiling
environments resolve PyAutoFit from source checkouts that carry the merge,
and Phase 2 drives this exact surface.

### Local venv (`~/venv/PyAuto`, WSL laptop)

- `pip install -U "blackjax>=1.6.2"`: 1.5 → **1.6.2**, jax 0.10.2 untouched,
  no new dependency conflicts (`pip check` deltas are pre-existing,
  unrelated packages).
- Smoke (all three checks) **PASS**: `blackjax.nss` logZ −4.609 ± 0.069 vs
  analytic −4.605; `af.BlackJAXNUTS` 0 divergences, acc 0.92; `af.NSS`
  logZ −4.650 ± 0.085, median (0.982, −0.996) vs truth (1, −1).
- `af.NSS` unit suite against the local checkout + blackjax 1.6.2:
  **17/17 pass**.

### RAL stack (`/mnt/ral/jnightin/PyAuto/PyAuto` venv)

- Pre-upgrade state: blackjax **0.1.0b1.dev86+g795058671** — the obsolete
  2026-01-era fork build the programme flagged (§2.2), incompatible with the
  mainline `nss` API. jax 0.10.2 + cuda12 plugin already in place; no SLURM
  jobs running at upgrade time.
- Pre-upgrade `pip freeze` snapshot saved beside the venv's existing ones:
  `freeze_pre_blackjax162_20260819.txt`.
- Library mains re-synced (`HPCPullPyAuto` path): PyAutoFit on RAL now
  carries the `af.NSS` re-mainline.
- Post-upgrade smoke (login node, `JAX_PLATFORMS=cpu` — no GPU there) —
  **all three checks PASS**: `blackjax.nss` logZ −4.609 ± 0.069;
  `af.BlackJAXNUTS` 0 divergences, acc 0.92; `af.NSS` logZ −4.650 ± 0.085.
  Seeded numbers identical to the local run — deterministic across
  environments at fixed seed.

**Phase 0's gate condition — items (a) and (b) both landed — is now
satisfied in full. Phases 1–2 are unblocked (see `DECISIONS.md`).**

## 0(c) — stranded-artifact harvest, local slice (2026-08-19)

Three fork-NSS A100 result pairs cited by `PROGRAMME.md` §1.2 were sitting
**untracked** in the local working tree (pulled from RAL 2026-08-04, never
committed) — the delaunay and pixelization rows existed nowhere in git:

- `results/searches/nss/imaging/mge/hst/hpc_hpc_a100_fp64.{json,png}` — the
  657 s / 383,289-eval / logZ 31700.4 half of the canonical "657–679 s"
  row-pair.
- `results/searches/nss/imaging/delaunay/hst/hpc_hpc_a100_fp64.{json,png}` —
  the 29,770 s / 206,448-eval / logZ 30567.8 row (the 11×-slower evidence).
- `results/searches/nss/imaging/pixelization/hst/hpc_hpc_a100_fp64.{json,png}`
  — the 19,190 s / 266,043-eval / logZ 29078.9 row (the 7×-slower evidence).

All three are committed with this note and now render in the searches README
dashboard. The RAL-side harvest — SMC warm arms (job 331058), NaN-counter
split arms (335003-5), any Nautilus pixgrad baseline logs — remains open;
0(c) stays partial until a RAL session walks the NFS output trees.

## 0(c) — RAL NFS harvest (2026-08-19) — COMPLETE

The RAL-side artifacts named by the plan are pulled off NFS into
`ral_harvest/` beside this note. 0(c) is closed.

### SMC warm arms — job 331058 (`smc_331058/`)

All four arms of the parked warm-started gradient-SMC wave (wsdev#113)
**completed** on RAL CPU (x64, blackjax 1.5-era prototype, node
euclid-ral-compute-10-1). Cell: `imaging/mge/hst`. The headline: **the
geometric evidence bridge is validated across kernels** — every warm arm's
logZ brackets the Nautilus truth bar (31690.5) within ±0.8 nats, with the
whitening Jacobian (log|J| = −108.03) correctly booked:

| Arm | max logL | logZ | Wall (CPU) | Evals | Converged |
|---|---:|---:|---:|---:|---|
| HMC warm | 31785.49 | 31690.33 | 1585 s | 28,928 | yes (λ=1) |
| MALA warm | 31785.79 | 31690.66 | 843 s | 7,040 | yes (λ=1) |
| MALA tuned warm | 31782.10 | 31689.73 | 712 s | 7,040 | yes (λ=1) |
| MALA tuned **cold** | **−179,466** | −179,475 | 693 s | 3,648 | "yes" (λ=1) |

The cold arm is the standing lesson made concrete: it tempered to λ=1 and
reported "Converged: yes" while sitting ~211,000 nats below the solution —
exactly the "silent λ=1 garbage" failure the risk register says never to
judge by. Truth bar for comparison: Nautilus max logL 31786.78 at 63,800
evals — the warm SMC arms reach within ~1–1.3 nats of it at 4–9× fewer
evals (CPU tier; not comparable to A100 rows). Phase 7 resumes from these
artifacts, not from scratch.

The parked wave's two pre-registered kernel questions are answered by the
per-λ acceptance traces:

- **Does `--tune` hold acceptance near ~0.57 as λ→1? NO — it collapses
  it.** Fixed-auto-step MALA decays 0.838 → 0.151 (the known pattern);
  MALA *with* tuning sits at 0.04–0.12 from step 2 onward and finishes
  with the worst max logL of the warm arms (31782.10). The tuner is
  mis-adapted for this tempering path — a Phase 7 fix-or-drop item.
- **Does HMC beat MALA? On robustness yes, on efficiency no.** HMC is the
  only kernel that holds acceptance as λ→1 (0.94–0.98 throughout), but it
  spends 4× the evals per particle-step (24 vs 5) and 2× the wall for the
  same answer — plain warm MALA matched its max logL and logZ at a quarter
  of the evals. On this unimodal cell the fixed-step decay is cosmetic;
  HMC's robustness is the thing to buy only where acceptance collapse
  actually costs accuracy (Phase 7's multimodal/transition tests).

### NaN-counter split arms — jobs 335003-5 (`nan_check_335003_5/`)

The verification runs behind the "NaN-counters CONFIRMED" record
(PyAutoFit PR#1473 / autolens_profiling PR#127; RESUME.md in the harvest
dir is the checkpoint note): per-mesh `cpu_cpu_nan_check.{json,png}` for
delaunay / delaunay_matern / knn plus the delaunay_matern probe pair and
the three job logs. Counters read zero value-NaN / zero gradient-NaN
lane-steps on the CPU tier at 10-step scale — the instrumentation works;
the MGE 62% value-NaN finding (#128) remains a GPU-scale phenomenon.

### Nautilus pixgrad baselines (`pix_nautilus_*.txt`)

Two one-line wsdev `searches_minimal` baselines: delaunay wall 8,331 s
max logL 19,982.3 (r_E 0.962); knn wall 19,102 s max logL 5,704.2
(r_E 1.011). **Different target than the profiling cells** (wsdev dataset/
masking — not comparable to the §1.2 truth bars); harvested so the numbers
stop living only on NFS, labelled for what they are.
