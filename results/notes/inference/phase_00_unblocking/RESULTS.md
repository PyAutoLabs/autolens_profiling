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
