# Clipper validation campaign — results and phase-3 recommendation

Phase 2 of the prior-support work (#128, #131). Phase 1 shipped `ClipperPriorBox`
(PyAutoFit#1477); this measures whether it earns the default in phase 3.

**Recommendation: do NOT flip the default on the evidence asked for. The fix does
exactly what it claims and it does not improve the answer.** A narrower case for
flipping it on hygiene grounds is set out at the end. The momentum-reset variant
should be dropped.

Run: `multi_start_prodigy`, `imaging/mge`, `hst`, 16 starts x 3000 steps,
**laptop RTX 2060, JAX float64**, two seeds per arm, `check_for_convergence=False`.
Raw rows in `multi_start_prodigy_imaging_mge_hst.json`.

A100s were checked first and were unavailable: all 8 allocated across
`euclid-ral-gpu-1/2` by another user's array, longest job 2d11h against a 27-day
limit.

## The result

Reference bar: Nautilus `max_log_likelihood = 31786.782462488976`
(the searches framework's `nautilus/imaging/mge/hst/hpc_a100_fp64.json` row, A100 fp64,
archived in PyAutoGut ref `refs/heads/archive/condemned/autolens-profiling/inference-programme` @ `c8b605801068ec3de04314b47da8f7272a038ba1`).
Nautilus samples in unit-cube coordinates, so it is structurally immune to this
failure mode. A negative `gap` means the MAP optimizer exceeded it, which is
expected — it maximises the posterior rather than sampling it.

| arm | seed | max_log_likelihood | gap to bar | value-NaN | clips | alive fraction | pinned | wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0 | 31787.929 | **-1.146** | 31655 | 0 | 0.341 | — | 1126 |
| `none` | 1 | -139485.799 | **171272.581** | 27870 | 0 | 0.419 | — | 2203 |
| `prior_box` | 0 | 31787.929 | **-1.146** | 2 | 15142 | 1.000 | 6/16 | 1255 |
| `prior_box` | 1 | -120880.568 | **152667.350** | 0 | 15744 | 1.000 | 4/16 | 1251 |
| `prior_box_reset` | 0 | 31787.929 | **-1.146** | 2523 | 18611 | 0.947 | 7/16 | 1744 |
| `prior_box_reset` | 1 | -137783.609 | **169570.391** | 0 | 8798 | 1.000 | 4/16 | 1123 |

Every `ClipperPriorBox` arm clipped tens of thousands of lane-steps, so no arm is
the "reported zero clips" broken case, and each arm ran under a unique search
name with its output directory cleared and `total_steps == n_steps` asserted.

## The load-bearing question

> Does clipped `MultiStartProdigy` get closer to the Nautilus bar than unclipped?

**No.**

- **Where the search converges (seed 0), the answer is identical.** All three arms
  land on `31787.929` — the same value to 15 significant figures. The winning lane
  never left the prior box, so clipping never touched it, and every lane clipping
  rescued stayed worse than the incumbent best.
- **Where the search fails (seed 1), clipping does not rescue it.** -139485.8 ->
  -120880.6 is a genuine 18,605-nat move toward the bar, but the run is still
  **152,667 nats away**. Recovering 11% of a catastrophic gap is not a rescue; both
  arms failed.

Pre-registered falsification #1 — "lane deaths fall but best-fit logL does not move
toward the reference, so clipping keeps lanes alive without making them useful" —
**fires on seed 0 exactly, and substantively on seed 1.**

## What the fix does do

It works, and it is not free of value:

- **Prior-exit deaths are eliminated**: 31655 -> 2 and 27870 -> 0 value-NaN
  lane-steps; alive fraction 0.34 / 0.42 -> **1.00**.
- **It costs nothing, and sometimes less than nothing**: +11% wall on seed 0
  (1126 -> 1255 s) but **-43% on seed 1** (2203 -> 1251 s). The unclipped bad seed
  spent nearly twice the wall time, because dead lanes keep stepping and keep
  paying a full likelihood-and-gradient evaluation whose output is discarded.
  Pre-registered falsification #3 (wall time rises materially) does **not** fire.
- **It makes the counters mean something.** With prior deaths gone the constrained
  count becomes visible (`constrained 6/16` in the seed-1 log) — the `ell_comps`
  trapping #128 predicted would surface once the larger failure was removed.

## The alive-versus-step curve

Graded on the curve, not the scalar, since the counters are survival integrals.
Living lanes at steps 1 / 10 / 50 / 100 / 300 / 1000 / 3000:

| arm | seed | curve |
|---|---:|---|
| `none` | 0 | 16, 16, **6**, 4, 6, 6, 5 |
| `none` | 1 | 16, 16, **5**, 6, 8, 7, 6 |
| `prior_box` | 0 | 16, 16, 16, 16, 16, 16, 16 |
| `prior_box` | 1 | 16, 16, 16, 16, 16, 16, 16 |
| `prior_box_reset` | 0 | 16, 16, 16, 16, 16, 15, 14 |

**The die-off is an early transient, not a steady bleed.** The population collapses
between steps 10 and 50 — while Prodigy is still taking its largest steps — and
then never recovers. This corrects the natural reading of the 60%-style scalar as
an ongoing hazard: it is one short window, integrated over the whole run.

It also shows lanes *returning*. `none` seed 0 reads 4 alive at step 100 and 6 at
step 300, so "dead" is a per-step non-finite state, not an absorbing one — a lane
that steps outside the box can step back in. The #128 framing of a lane that
"stays dead for every remaining step" is right about the accounting and wrong
about the mechanism.

## Arm 3 (momentum reset) — drop it

`prior_box_reset` is worse than plain clipping on both seeds.

On seed 0 it gives the same answer while degrading everything else: deaths
2 -> 2523, clips 15142 -> 18611, pinned 6 -> **7**, wall +39%.

On seed 1 it makes the answer itself **worse**: -120880.6 -> **-137783.6**, giving
back ~16,900 of the 18,600 nats plain clipping had recovered and landing almost
exactly on the unclipped -139485.8. So the one arm where clipping had bought
anything measurable is the arm the momentum reset undoes.

It was motivated by the prototype's 5/16 and the CPU run's 11/16 pinned lanes, on
the theory that a clipped lane keeps the velocity that pushed it out and is
re-projected onto the same bound forever. The measurement does not support the
remedy: zeroing momentum shortens each step, so lanes loiter near a bound and get
re-clipped rather than escaping, and some lose enough state to die outright.

Pre-registered falsification #2 — "most surviving lanes end pinned" — **does not
fire**: 6/16 and 4/16 (37%, 25%) on `prior_box`, well below the CPU run's 11/16.
The wall is not absorbing the population, so the premise for this arm was weaker
than the earlier float32 CPU numbers suggested.

## Do the clips calm down once the search finds the basin? (not measured)

15,142 clips is 31.5% of the 48,000 lane-steps, which invites the question of
whether clipping is an early transient that settles, or a steady state. **The
artefacts cannot answer it**: `n_clipped_lane_steps` is recorded only as a
lifetime total, and the per-step log reports `alive` and `constrained` but not
clips. `alive_history` was added for exactly this class of question and the clip
equivalent should have been added with it — that is a gap in this campaign, not a
property of the fix.

The indirect evidence says the clips **persist** rather than settling:

| arm | seed | clips | clips/step | pinned at end |
|---|---:|---:|---:|---:|
| `prior_box` | 0 | 15142 | 5.05 | 6/16 |
| `prior_box` | 1 | 15744 | 5.25 | 4/16 |
| `prior_box_reset` | 0 | 18611 | 6.20 | 7/16 |
| `prior_box_reset` | 1 | 8798 | 2.93 | 4/16 |

- ~5 of 16 lanes are clipped on an average step and ~4-7 lanes end pinned to a
  bound. Those matching is the tell: most of the count is a few lanes sitting *on*
  a boundary being re-projected every step, not a burst of early crossings.
- Seed 0 and seed 1 clip at nearly the same rate (31.5% vs 32.8%) despite one
  converging past the bar and the other failing by 152,667 nats. An early-transient
  effect would not survive that difference.
- Prodigy's step scale grows through the run (`d` ends at 2.26e-02 / 5.49e-02 /
  8.29e-01 against 1.00e-06 at step 1), so late steps are larger and make *more*
  boundary contact, not less. Clipping continued long after `gained 0.0000`.

`prior_box_reset` seed 1 is the one row that does not fit (2.93 clips/step on 4
pinned lanes), so the picture is not uniform and this stays an inference.

**Next step:** add `clipped_history` beside `alive_history` and re-run one arm at
a short budget. Cheap, and it converts this section from argument to measurement.

## Do the rescued lanes contribute anything? No — and why they end up at the wall

**They contribute nothing, measured directly.** On seed 0 `best_fom` is identical
to every digit across all three arms — `-63575.67359062914` — at final alive
counts of 5/16, 16/16 and 14/16. Clipping kept eleven extra lanes alive and not
one of them ever became the best. Clipping stops the dead-sampling loop (the lane
reverses off the wall and keeps evaluating) without those lanes ever approaching
the peak.

**They are not initialised at the edges.** `_broad_starts` draws in the UNIT cube
at `[0.15, 0.85]`, the middle 70% of every prior:

| parameter | prior | start range | box width | nearest wall |
|---|---|---|---:|---:|
| `mass.einstein_radius` | U(0, 8) | [+1.20, +6.80] | 8.00 | 1.20 |
| `shear.gamma_1/2` | U(-0.3, 0.3) | [-0.21, +0.21] | 0.60 | 0.09 |
| `bulge.centre_0/1` | U(-0.1, 0.1) | [-0.07, +0.07] | 0.20 | **0.03** |

No lane starts at an edge. They walk there.

**The mechanism is a step-scale / box-width mismatch.** Box widths across this
15-parameter model span 8.0 down to 0.2 — a **40x range** — while the search steps
in PHYSICAL space with a single global step scale. Prodigy's `d` grows from
`1.00e-06` at step 1 to `2.26e-02 / 5.49e-02 / 8.29e-01` by the end. A step scale
that is sensible for `einstein_radius` is a wall-crossing for `bulge.centre`.

That predicts escapes concentrate in the NARROW-box parameters, and #128's autopsy
found exactly that: 10 of 11 exits were the shear, at ±0.30-0.35 just outside its
±0.3 box.

It also explains the null result. Lanes pin in **nuisance** parameters (shear,
bulge centre), not in the parameters that set fit quality. Rescuing a lane that
has railed its shear does not make it competitive — it only stops it being NaN.

**Implication: this is evidence for revisiting unit-cube stepping**, which phase 1
rejected (reasons in `complete/2026/08/prior-support-clipper.md`). A step in unit
space is automatically commensurate with each box's width, so the 40x disparity
stops mattering. That addresses the cause; clipping addresses the symptom.
Per-parameter step scaling is the cheaper variant of the same idea.

**Limit:** which coordinates were pinned is not recorded — per-lane params were
not saved, the same recording gap as the clip history. Only the totals exist
(19 pinned coordinates across 6 lanes, of 16x15 = 240, i.e. 7.9%). The shear
attribution above is #128's, not this campaign's.

## The bigger finding: seed dependence dwarfs the clipper

Identical settings, two seeds, and the outcome swings **171,000 nats**: seed 0
beats the Nautilus bar, seed 1 never finds the basin. That is a property of the
search, not of the clipper, and it is larger than every effect this campaign set
out to measure.

It was invisible before this campaign. `_broad_starts` hardcoded
`np.random.default_rng(0)`, so every run drew seed 0's population — the lucky one.
Seeding `random`/`numpy` reached only the initializer. A single-seed study on this
cell would have reported the unclipped search as healthy with total confidence.

**This deserves its own task**, and it matters more than phase 3.

## Recommendation for phase 3

**Not on the evidence the campaign asked for.** The claim "clipping recovers the
lost lanes without changing the answer" is now measured: it recovers the lanes
completely, and the second half is true in a way that undercuts the first — the
answer does not change because the recovered lanes never mattered.

A narrower case for flipping the default does stand, and should be argued honestly
as hygiene rather than accuracy:

- it removes a silent failure mode where two thirds of a search population is
  discarded without any error surfacing;
- it costs nothing measurable and saves substantial time on the bad seed;
- it unmasks the constrained-lane accounting that is currently hidden behind it.

If phase 3 proceeds on those grounds, the re-baseline is cheap: on this cell the
converged answer is bit-identical, so no existing benchmark number moves.

**Not yet generalised.** One cell, one sampler, two seeds. The pixelized cells —
the ones `resurrect=True` exists for, and the strongest test of whether clipping
and resurrection interact badly — have not been run; they need more VRAM than the
laptop card has. `point_source` and the unbounded-prior negative control are also
outstanding. Phase 3 should not be written on `imaging/mge` alone.

## Reproducing

```bash
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
JAX_ENABLE_X64=True SEARCHES_DISABLE_VIZ=1 \
python clipper_campaign.py  # removed with the inference programme; recover from the archive ref \
    --arms none,prior_box,prior_box_reset --seeds 0,1 \
    --n-starts 16 --n-steps 3000 --sampler multi_start_prodigy
```

Requires PyAutoFit at or after the `seed` / `alive_history` /
`reset_momentum_on_clip` commit; a released wheel has none of it, and
`autofit.__file__` must resolve to the checkout.

Two traps worth keeping, both paid for here:

- **`--out` used to be a no-op** unless the directory sat under the repo's
  `output/`, so the per-arm clearing deleted an unrelated directory, a stale
  `.completed` short-circuited `fit()`, and the arm returned a **cached result in
  2.9 s with every counter `None`**. The `total_steps == n_steps` assertion is what
  caught it. The output root now goes through `conf`.
- **`search.summary` is `Key = Value`, not colon-separated.** A colon parser
  returns an empty dict and every counter reads `None` — indistinguishable from a
  clipper that never fired.

## Budget is not a detail

The cloud CPU session ran this comparison at 16x150 and found the arms identical,
concluding clipping was cosmetic. Both of its arms were ~47,316 nats from the bar
— neither had found the basin, so the comparison could not have detected a
difference either way.

At 105 steps on GPU/fp64 the clipped arm was **114 nats** closer to the bar than
the unclipped one; by 3000 steps that advantage is gone and the two agree exactly.
So clipping does buy convergence *rate* at short budgets, and nothing at long
ones. Any statement about this fix has to name the step budget it was measured at.

---

# Phase 3 (cause side): per-parameter step scaling — FALSIFIED

`ScalerPriorWidth` (PyAutoFit#1483) gives every parameter its own step scale,
derived from its prior and applied as a linear change of variables `phi = theta/s`
with the objective still evaluated at the physical `theta = s * phi`. It was built
to treat the **cause** diagnosed in the section above: steps that are sensible for
a wide prior are wall-crossings for a narrow one.

**It does not work on this cell.** Three of the four pre-registered falsification
conditions fired. This section records that plainly rather than re-scoping to save
the idea, per the pre-registration.

## The side-by-side

`multi_start_prodigy`, `imaging/mge`, `hst`, 16 starts x 3000 steps, seeds 0 and
1, JAX float64, `check_for_convergence=False`. The clipper is **on** in the scaled
arm, so the clip rate is a clean diagnostic rather than a confound. Truth bar:
Nautilus `31786.782462488976`; a negative gap means the MAP optimizer exceeded it.

| arm | seed | max_log_likelihood | gap to bar | value-NaN | clips | clip rate | alive frac | pinned lanes | pinned coords |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0 | 31787.929 | **-1.146** | 31655 | 0 | — | 0.341 | — | — |
| `none` | 1 | -139485.799 | 171272.581 | 27870 | 0 | — | 0.419 | — | — |
| `prior_box` | 0 | 31787.929 | **-1.146** | 2 | 15142 | 31.55% | 1.000 | 6/16 | 19 |
| `prior_box` | 1 | -120880.568 | 152667.350 | 0 | 15744 | 32.80% | 1.000 | 4/16 | 6 |
| `prior_box_reset` | 0 | 31787.929 | **-1.146** | 2523 | 18611 | 38.77% | 0.947 | 7/16 | 30 |
| `prior_box_reset` | 1 | -137783.609 | 169570.391 | 0 | 8798 | 18.33% | 1.000 | 4/16 | 18 |
| **`prior_box_scaled`** | **0** | **31785.186** | **+1.597** | 3215 | 15097 | **31.45%** | 0.933 | 6/16 | 6 |
| **`prior_box_scaled`** | **1** | **-145975.777** | **177762.560** | 5884 | 18487 | **38.51%** | 0.877 | 3/16 | 8 |

## Against the pre-registered readouts

**1. "Clip rate collapses" — FAILED, and this was the primary readout.**
Seed 0: 31.55% -> **31.45%**. That is not a collapse, it is a 0.3% change — the
two runs clip at the same rate to three significant figures. Seed 0 clips
15,142 times unscaled and 15,097 times scaled. Seed 1 goes the **wrong way**:
32.80% -> 38.51%.

The pre-registration is explicit about what this means: *"Clip rate does not fall
-> the diagnosis in this prompt is wrong, and the wall contact is not a step-scale
effect. Say so; do not re-scope to save the idea."*

**2. "Seed 0 must not regress past 31787.929" — FAILED.** It lands at
**31785.186**, which no longer beats the Nautilus bar (gap `+1.597` where every
other arm reaches `-1.146`). Small in absolute terms, but it is the one arm in the
whole campaign that fails to exceed the reference, and the pre-registration reads
that as *"the preconditioner is fighting Prodigy's own `d` estimate."*

**3. "Seed 1 moves materially toward the bar" — FAILED, it moved away.**
-145975.777 against plain clipping's -120880.568: **25,095 nats worse**, and worse
even than the unclipped baseline's -139485.799. This was the test the whole line of
work was looking for and the answer is negative.

**4. "Pinned lanes fall" — mixed, and the one place scaling did something.**
Lanes pinned: 6/16 -> 6/16 (seed 0), 4/16 -> 3/16 (seed 1). But pinned
*coordinates* fall sharply on seed 0: **19 -> 6**. The same number of lanes are
pinned, in a third as many coordinates each. Scaling did change *where* lanes meet
walls; it did not change *how often*.

**5. "Wall time does not rise materially" — INCONCLUSIVE, do not cite it.**
Recorded 1841 s (seed 0) and 5351 s (seed 1) against a 1255/1251 s baseline. **The
seed-1 number is invalid**: the run was deliberately `SIGSTOP`ed for ~58 minutes
mid-arm to free the laptop, and the driver's wall clock counts that. Corrected for
the two pause gaps (3501 s of excess between steps 2140-2150 and 2160-2170 in the
run log) it is ~1850 s. That leaves a consistent ~+47% on both seeds — but the
baseline rows were measured in a **different session on a laptop GPU**, and
within-session spread in that same baseline was already +39% (`prior_box` 1255 s
vs `prior_box_reset` 1744 s on seed 0). A cross-session laptop wall-time
comparison at this precision is not evidence. A diagonal multiply should still be
free; if the cost is real it needs a same-session A/B to establish.

## Why it fails: normalisation conserves the thing being measured

The most striking number above is that seed 0 clips 15,142 times unscaled and
15,097 times scaled — a 0.3% difference across 48,000 lane-steps. That is too
close to be luck, and it points at the mechanism.

The scale vector is **normalised to a geometric mean of exactly 1**, a choice made
so that only the *ratios* between coordinates change and the A/B is not confounded
with an effective learning-rate change. But the aggregate propensity to reach a
wall is set by the *global* step magnitude relative to the box widths — and that
is precisely the quantity the normalisation holds fixed.

So scaling **redistributes** wall contact rather than reducing it. Before, narrow
coordinates reached their walls quickly and wide ones essentially never did. After,
the step-to-width ratio is equalised, so every coordinate reaches its wall at the
same intermediate rate. The total is conserved. The `pinned_coords` 19 -> 6 on seed
0 is that redistribution showing up directly.

Read that way the experiment, as designed, **could not** have produced a clip-rate
collapse. The design choice that made the comparison clean is the same one that
pinned the primary readout in place.

The measured scale vector (15 parameters, geometric mean 1.0):

```
0.597 0.597 0.895 0.895 0.298 0.298 0.895 0.895 23.871 1.790 1.790 0.895 0.895 0.895 0.895
```

**Correction to this note's own earlier claim:** the spread is **80x**, not the
40x quoted in "Why the rescued lanes contribute nothing". 40x counted the
`UniformPrior` widths alone (`einstein_radius` 8.0 against `bulge.centre` 0.2);
including the Gaussian and TruncatedGaussian sigmas doubles it. `einstein_radius`
sits at 23.9 against a 0.298 floor.

## Scaling also undoes clipping's one clear win

Plain clipping drove value-NaN lane-steps to **2** (seed 0) and **0** (seed 1) —
the cleanest effect in the whole campaign. The scaled arm puts them back:
**3215** and **5884**, with the alive fraction falling from 1.000 to 0.933/0.877.

Those deaths happen with the clipper **on**, so the lanes are inside their prior
boxes: it is the *likelihood* going non-finite, not the prior. Scaling moves lanes
onto trajectories that enter the genuinely NaN regions of this likelihood — the
`ell_comps`/shear degeneracies #128 identified. So the change is not merely
ineffective, it costs a property that was already won.

## Verdict

**Do not ship `ScalerPriorWidth` as a default, and do not pursue prior-width
scaling further on the strength of this cell.** It ships default-off so this row
stays reproducible, exactly as `reset_momentum_on_clip` did, and for the same
reason: the measurement is the artefact worth keeping, not the feature.

The diagnosis behind it was *half* right. The 80x step-scale/box-width mismatch is
real and measured. What is now falsified is the inference that **equalising** it
reduces wall contact. On this evidence wall contact is a step-**size** effect, not
a step-**shape** effect: to clip less you must take smaller steps relative to the
boxes, and redistributing a fixed global step among coordinates cannot do it.

That is a hypothesis this campaign has **not** tested and it must not be presented
as a rescue of this one. It would need its own pre-registration, and it directly
trades against convergence rate, which is the reason the normalisation was there in
the first place.

## What this settles for the clipper default

This arm was meant to feed the phase-3 decision by showing that a collapsed clip
rate makes defaulting the clipper on nearly free. **It does not show that**, so
that argument is unavailable.

The case for the default therefore rests where it always more properly did — on
**constrained-optimizer semantics** rather than on accuracy or cost. A search
advertising a posterior with hard prior support is solving a *constrained*
problem; a state outside that support is **infeasible**, not merely a poor
candidate. Projection onto the box does not alter the constrained optimum, makes
that invariant explicit, prevents silent invalid trajectories, stops prior exits
masking later pathologies, and is what makes the lane diagnostics interpretable at
all. This campaign's own numbers support that framing and not an accuracy one: at
3000 steps clipping changes the answer by nothing at all.

The remaining blocker is **empirical breadth, not mathematics**. Everything
measured here is `MultiStartProdigy` on one lens cell, while `MultiStartAdam`,
`MultiStartLion` and `MultiStartADABelief` are unmeasured — and they are *more*
exposed to prior exits, not less: Adam's update is `lr * m_hat / (sqrt(v_hat) +
eps)` with `m_hat/sqrt(v_hat) ~ 1`, so it steps by roughly `lr` in **physical**
units in every coordinate, and Lion, being sign-based, by exactly `lr`.

Recommendation: make hard-support enforcement the **intended** default for the
gradient/MLE searches, validate it across the other gradient rules before
flipping, and do not sell it as a long-budget accuracy improvement.

Keep the scaler and the clipper as separate features regardless. Even had the clip
rate collapsed to 0.1%, projection would stay as the last-line invariant.

## Reproducing this arm

```bash
source ~/Code/PyAutoLabs-wt/per-parameter-step-scaling/activate.sh
python -c "import autofit; print(autofit.__file__)"   # MUST be the task checkout

PYAUTO_SKIP_API_GATE=1 SEARCHES_DISABLE_VIZ=1 \
JAX_PLATFORM_NAME=cuda JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 JAX_ENABLE_X64=True \
~/venv/PyAutoGPU/bin/python clipper_campaign.py  # removed with the inference programme; recover from the archive ref \
  --sampler multi_start_prodigy --arms prior_box_scaled --seeds 0,1 \
  --n-starts 16 --n-steps 3000 --out /tmp/gpu_out
```

Two traps this arm added to the pile:

- **`autofit.__file__` is not optional.** Run without sourcing `activate.sh` and
  `autofit` resolves to the *installed* stack, where `ScalerPriorWidth` does not
  exist; the arm dies on `AttributeError` rather than silently mismeasuring, but
  only because the symbol is new. A knob that already existed would have run the
  wrong code.
- **The driver's "zero clips = broken arm" check inverts on a scaled arm**, where
  zero clips is the *hoped-for result*. Left uncorrected the campaign would have
  reported its own success as a broken arm — the exact way a null result is
  manufactured. The scaled arm is validated from the recorded **scale vector**
  (non-trivial spread) instead of the clip count.
